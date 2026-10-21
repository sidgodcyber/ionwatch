"""Parse a Quectel L86 (MediaTek) NMEA 0183 log into the tidy C/N0 table and the l86_obs table.

Capture formats (detected per line):
  plain         $GPGSV,...*hh
  iso_prefix    2026-10-20T13:30:00.123Z,$GPGSV,...*hh   (comma or space separated)
  epoch_prefix  1792500000.123 $GPGSV,...*hh

GSV carries no time: each GSV group gets the UTC time of the most recent RMC / GGA / ZDA
(date from RMC / ZDA). The host prefix is only a fallback, and the median host - NMEA
offset is reported. Multi-sentence GSV groups are assembled per talker (and NMEA 4.11
signal id). Not yet validated on a real L86 log.

Talkers (config l86.talkers): GP GPS, GL GLONASS, GA Galileo, GB/BD BeiDou, GQ and PQ QZSS,
GI NavIC. The L86-M33 (MediaTek MT3333) does not track NavIC, so in IonWatch NavIC comes
only from the phone; GI is parsed so other NMEA receivers can be used unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ionwatch.proc import common

ISO_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z?[,\s]+(\$.*)$")
EPOCH_PREFIX = re.compile(r"^(\d{9,11}(?:\.\d+)?)[,\s]+(\$.*)$")
NMEA = re.compile(r"^\$([A-Z0-9]{2,})(,.*)?\*([0-9A-Fa-f]{2})$")
TABLE_COLS = ["time_utc", "talker", "constellation", "prn_raw", "svid", "signal", "elev_deg", "az_deg",
              "cn0_dbhz", "used_in_fix", "time_source"]
INDEX_EXCLUDE = {"SBAS"}  # tracked and tagged, but excluded from indices


@dataclass
class L86Log:
    table: pd.DataFrame                  # spec 4.2 l86_obs (all satellites, incl. SBAS)
    obs: pd.DataFrame                    # common.OBS_COLS, SBAS excluded
    quality: dict = field(default_factory=dict)


def checksum_ok(sentence: str) -> bool:
    m = NMEA.match(sentence)
    if not m:
        return False
    body = sentence[1:sentence.rindex("*")]
    x = 0
    for ch in body:
        x ^= ord(ch)
    return x == int(m.group(3), 16)


def normalise_prn(talker: str, prn: int, talkers: dict) -> tuple[str | None, int, str | None]:
    """(constellation, svid, note) from talker + raw PRN (MediaTek/Quectel conventions)."""
    const = None if talker == "GN" else talkers.get(talker)   # GN: decide by PRN range
    if const == "GPS" or talker == "GN":
        if 1 <= prn <= 32:
            return "GPS", prn, None
        if 33 <= prn <= 64:
            return "SBAS", prn, None
        if 120 <= prn <= 158:
            return "SBAS", prn, None
        if 193 <= prn <= 202:
            return "QZSS", prn, None
        if 65 <= prn <= 96 and talker == "GN":
            return "GLONASS", prn - 64, None
        if 301 <= prn <= 336 and talker == "GN":
            return "Galileo", prn - 300, None
        if 401 <= prn <= 463 and talker == "GN":
            return "BeiDou", prn - 400, None
        return const or "GPS", prn, f"{talker} PRN {prn} outside known ranges"
    if const == "GLONASS":
        if 65 <= prn <= 96:
            return const, prn - 64, None
        return const, prn, (None if 1 <= prn <= 24 else f"GLONASS PRN {prn} kept raw")
    if const == "Galileo":
        if 1 <= prn <= 36:
            return const, prn, None
        if 301 <= prn <= 336:
            return const, prn - 300, None
        return const, prn, f"Galileo PRN {prn} not in 1-36 / 301-336: kept raw"
    if const == "BeiDou":
        if 201 <= prn <= 263:
            return const, prn - 200, None
        return const, prn, None
    if const == "QZSS":
        return const, prn + 192 if 1 <= prn <= 10 else prn, None
    return const, prn, (None if const else f"unknown talker {talker}")


def _f(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return np.nan


def _hms(s: str) -> timedelta | None:
    if len(s) < 6:
        return None
    try:
        return timedelta(hours=int(s[0:2]), minutes=int(s[2:4]), seconds=float(s[4:]))
    except ValueError:
        return None


class NmeaParser:
    """Incremental NMEA parser: feed() one line at a time, get the satellite rows of every
    GSV group it completes. Used by the batch reader and by live mode.

    used_in_fix comes from the GSA sentences seen earlier in the same update cycle (MTK
    order is RMC, VTG, GGA, GSA, GSV, GLL).
    """

    def __init__(self, cfg: dict):
        self.talkers = cfg["l86"]["talkers"]
        self.counts = {"lines": 0, "nmea": 0, "bad_checksum": 0, "pmtk": 0, "noise": 0, "other_sentences": 0,
                       "gsv_groups": 0, "gsv_incomplete": 0, "gsv_no_time": 0, "rmc_no_fix": 0}
        self.formats = {"plain": 0, "iso_prefix": 0, "epoch_prefix": 0}
        self.notes: dict[str, int] = {}
        self.offsets: list[float] = []
        self.cur_date = None          # datetime.date (UTC)
        self.cur_time = None          # datetime (UTC) of the latest RMC/GGA/ZDA
        self.fix_valid = None         # RMC status A/V of the latest RMC
        self.acks: list[tuple[str, str]] = []
        self._last_tod = None
        self._groups: dict[tuple, dict] = {}
        self._used: dict[tuple, set] = {}

    def _flush(self, key, complete=True) -> list[dict]:
        g = self._groups.pop(key, None)
        if not g:
            return []
        self.counts["gsv_groups"] += 1
        if not complete:
            self.counts["gsv_incomplete"] += 1
        for r in g["sats"]:
            r["used_in_fix"] = r["svid"] in self._used.get((g["time"], r["constellation"]), ())
        return g["sats"]

    def finish(self) -> list[dict]:
        """Flush GSV groups still open (end of file / stream)."""
        out = []
        for key in list(self._groups):
            out += self._flush(key, complete=False)
        return out

    def feed(self, raw_line: str, host: datetime | None = None) -> list[dict]:
        """Parse one line; ``host`` is the receive time when the line has no host prefix."""
        self.counts["lines"] += 1
        line = raw_line.strip()
        if not line:
            return []
        if line.startswith("$"):
            fmt, sentence = "plain", line
        elif (m := ISO_PREFIX.match(line)):
            fmt, sentence = "iso_prefix", m.group(2)
            host = pd.Timestamp(m.group(1).replace(" ", "T"), tz="UTC").to_pydatetime()
        elif (m := EPOCH_PREFIX.match(line)):
            fmt, sentence = "epoch_prefix", m.group(2)
            host = datetime.fromtimestamp(float(m.group(1)), tz=timezone.utc)
        else:
            self.counts["noise"] += 1
            return []
        if not checksum_ok(sentence):
            self.counts["bad_checksum" if (NMEA.match(sentence) or "*" in sentence) else "noise"] += 1
            return []
        self.formats[fmt] += 1
        self.counts["nmea"] += 1
        fields = sentence[1:sentence.rindex("*")].split(",")
        addr = fields[0]
        if addr.startswith("PMTK"):
            self.counts["pmtk"] += 1
            if addr == "PMTK001" and len(fields) > 2:      # ack: command, flag (3 = success)
                self.acks.append((fields[1], fields[2]))
            return []
        talker, stype = addr[:2], addr[2:]
        if stype in ("RMC", "GGA", "ZDA"):
            self._time_sentence(stype, fields, host)
            return []
        if stype == "GSA":
            sys_id = fields[18] if len(fields) > 18 and fields[18] else None
            t = self.cur_time or host
            tk = {"1": "GP", "2": "GL", "3": "GA", "4": "GB", "5": "GQ", "6": "GI"}.get(sys_id, talker)
            for p in fields[3:15]:
                if p.strip().isdigit():
                    c, sv, _ = normalise_prn(tk, int(p), self.talkers)
                    self._used.setdefault((t, c), set()).add(sv)
            return []
        if stype != "GSV":
            self.counts["other_sentences"] += 1
            return []
        return self._gsv(talker, fields, host)

    def _time_sentence(self, stype, fields, host):
        tod = _hms(fields[1]) if len(fields) > 1 else None
        if stype == "RMC":
            self.fix_valid = len(fields) > 2 and fields[2] == "A"
            if not self.fix_valid:
                self.counts["rmc_no_fix"] += 1
        if stype == "RMC" and len(fields) > 9 and len(fields[9]) == 6:
            d = fields[9]
            self.cur_date = datetime(2000 + int(d[4:6]), int(d[2:4]), int(d[0:2]), tzinfo=timezone.utc).date()
        elif stype == "ZDA" and len(fields) > 4 and fields[4]:
            self.cur_date = datetime(int(fields[4]), int(fields[3]), int(fields[2]), tzinfo=timezone.utc).date()
        elif self.cur_date is None and host is not None:
            self.cur_date = host.date()
        elif (stype == "GGA" and self.cur_date is not None and tod is not None and self._last_tod is not None
              and tod < self._last_tod - timedelta(hours=12)):
            self.cur_date = self.cur_date + timedelta(days=1)   # midnight rollover seen only by GGA
        if tod is not None and self.cur_date is not None:
            t = datetime.combine(self.cur_date, datetime.min.time(), tzinfo=timezone.utc) + tod
            if t != self.cur_time:      # new update cycle: keep only this and the previous cycle's GSA
                self._used = {k: v for k, v in self._used.items() if k[0] in (self.cur_time, t)}
            self.cur_time = t
            self._last_tod = tod
            if host is not None:
                self.offsets.append((host - t).total_seconds())

    def _gsv(self, talker, fields, host) -> list[dict]:
        out = []
        try:
            total, num = int(fields[1]), int(fields[2])
        except (ValueError, IndexError):
            self.counts["other_sentences"] += 1
            return out
        body = fields[4:]
        sig_id = body.pop() if len(body) % 4 == 1 else ""
        key = (talker, sig_id)
        if num == 1:
            if key in self._groups:
                out += self._flush(key, complete=False)
            t, src = (self.cur_time, "nmea") if self.cur_time is not None else (host, "host")
            if t is None:
                self.counts["gsv_no_time"] += 1
                return out
            self._groups[key] = {"time": t, "src": src, "sats": []}
        g = self._groups.get(key)
        if g is None:            # continuation without its first sentence
            self.counts["gsv_incomplete"] += 1
            return out
        for i in range(0, len(body) - 3, 4):
            prn_s = body[i].strip()
            if not prn_s.isdigit():
                continue
            prn = int(prn_s)
            const, svid, note = normalise_prn(talker, prn, self.talkers)
            if note:
                self.notes[note] = self.notes.get(note, 0) + 1
            cn0 = _f(body[i + 3]) if body[i + 3] else np.nan
            g["sats"].append({"time_utc": g["time"], "talker": talker, "constellation": const, "prn_raw": prn,
                              "svid": svid, "signal": "L5" if const == "NavIC" else "L1",
                              "elev_deg": _f(body[i + 1]) if body[i + 1] else np.nan,
                              "az_deg": _f(body[i + 2]) if body[i + 2] else np.nan,
                              "cn0_dbhz": cn0 if cn0 > 0 else np.nan, "time_source": g["src"]})
        if num == total:
            out += self._flush(key)
        return out


def rows_to_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=TABLE_COLS)
    if len(table):
        table["time_utc"] = pd.to_datetime(table["time_utc"], utc=True)
        table["used_in_fix"] = table["used_in_fix"].astype(bool)
        table = table.drop_duplicates(["time_utc", "constellation", "svid", "signal"]).reset_index(drop=True)
    return table[TABLE_COLS]


def parse_lines(lines, cfg: dict, name: str = "") -> L86Log:
    p = NmeaParser(cfg)
    rows = []
    for line in lines:
        rows += p.feed(line)
    rows += p.finish()
    table = rows_to_table(rows)
    q = quality(table, p.counts, p.formats, p.notes, p.offsets, cfg, name)
    return L86Log(table=table, obs=to_obs(table), quality=q)


def to_obs(table: pd.DataFrame) -> pd.DataFrame:
    t = table[~table["constellation"].isin(INDEX_EXCLUDE) & table["constellation"].notna()]
    obs = pd.DataFrame({"time_utc": t["time_utc"], "receiver": "l86", "source": "nmea",
                        "constellation": t["constellation"], "svid": t["svid"].astype(int), "signal": t["signal"],
                        "band": t["signal"], "elev_deg": t["elev_deg"], "az_deg": t["az_deg"],
                        "cn0_dbhz": t["cn0_dbhz"], "time_quality": t["time_source"]})
    return obs[common.OBS_COLS].sort_values(["constellation", "svid", "time_utc"]).reset_index(drop=True)


def effective_rate(out_rate: float | None, frac_change: float) -> float | None:
    """Effective C/N0 update rate. Only above ~1.5 Hz output can repeats mean stale values
    (10 Hz output, C/N0 updated once per fix); at 1 Hz an unchanged integer C/N0 is a real
    sample, so the effective rate is the output rate."""
    if not out_rate or not np.isfinite(frac_change) or out_rate <= 1.5:
        return round(float(out_rate), 2) if out_rate else out_rate
    return round(min(out_rate, max(1.0, frac_change * out_rate)), 2)


def _stamp(t, tz):
    return {"utc": f"{t:%Y-%m-%dT%H:%M:%S.%f}"[:-3] + "Z", "ist": t.tz_convert(tz).strftime("%Y-%m-%dT%H:%M:%S%z")}


def quality(table, counts, formats, notes, offsets, cfg, name="") -> dict:
    tz = cfg["display_tz"]
    warn = []
    q = {"file": name, "format": max(formats, key=formats.get) if counts["nmea"] else None,
         "format_counts": formats, **counts, "id_notes": notes, "warnings": warn}
    if not len(table):
        q.update(epochs=0, output_rate_hz=None, effective_rate_hz=None)
        warn.append({"code": "no_gsv", "message": "No usable GSV data in the L86 log."})
        return q
    ep = np.unique(table["time_utc"].astype("int64").to_numpy()) / 1e9
    d = np.diff(ep)
    out_rate = 1 / np.median(d) if len(d) else None
    eff = []
    for _, g in table[table["cn0_dbhz"].notna()].groupby(["constellation", "svid"]):
        v = g.sort_values("time_utc")["cn0_dbhz"].to_numpy()
        if len(v) > 10:
            eff.append(np.mean(v[1:] != v[:-1]))
    frac_change = float(np.median(eff)) if eff else np.nan
    eff_rate = effective_rate(out_rate, frac_change)
    gaps = np.flatnonzero(d > cfg["quality"]["gap_report_s"])
    t0, t1 = table["time_utc"].min(), table["time_utc"].max()
    q.update({
        "start": _stamp(t0, tz), "end": _stamp(t1, tz), "duration_min": round((t1 - t0).total_seconds() / 60, 1),
        "epochs": int(len(ep)), "output_rate_hz": round(float(out_rate), 2) if out_rate else None,
        "epoch_spacing_ms_hist": {str(k): int(v) for k, v in
                                  pd.Series(np.round(d * 1000, -1)).value_counts().sort_index().head(15).items()},
        "cn0_change_fraction": round(frac_change, 3) if np.isfinite(frac_change) else None,
        "effective_rate_hz": eff_rate,
        "satellites_per_constellation": table.groupby("constellation")["svid"].nunique().to_dict(),
        "prn_ranges_per_talker": {tk: {"min": int(g["prn_raw"].min()), "max": int(g["prn_raw"].max()),
                                       "n_ids": int(g["prn_raw"].nunique())} for tk, g in table.groupby("talker")},
        "time_source": table["time_source"].value_counts().to_dict(),
        "host_minus_nmea_offset_s": {"median": round(float(np.median(offsets)), 3), "n": len(offsets)}
        if offsets else None,
        "gaps_over_s": {"count": int(len(gaps)),
                        "list": [{"start": _stamp(pd.Timestamp(ep[i], unit="s", tz="UTC"), tz),
                                  "length_s": round(float(d[i]), 1)} for i in gaps[:20]]},
    })
    if out_rate and out_rate > 1.5 and eff_rate is not None and eff_rate < 0.5 * out_rate:
        warn.append({"code": "l86_low_effective_rate",
                     "message": f"L86 repeats C/N0 between fixes; S4 from L86 is low-rate "
                                f"(effective {eff_rate} Hz at {out_rate:.0f} Hz output)."})
    if counts["bad_checksum"]:
        warn.append({"code": "l86_bad_checksum", "message": f"{counts['bad_checksum']} NMEA lines failed the checksum."})
    if len(gaps):
        warn.append({"code": "l86_gaps", "message": f"{len(gaps)} L86 gaps longer than "
                                                    f"{cfg['quality']['gap_report_s']} s."})
    if notes:
        warn.append({"code": "l86_id_notes", "message": "Unrecognised satellite IDs: " +
                     "; ".join(f"{k} ({v}x)" for k, v in notes.items())})
    if counts["gsv_no_time"]:
        warn.append({"code": "l86_gsv_no_time",
                     "message": f"{counts['gsv_no_time']} GSV groups before any RMC/GGA/ZDA time were dropped."})
    return q


def read_l86(path: str | Path, cfg: dict) -> L86Log:
    with open(path, encoding="ascii", errors="replace") as f:
        return parse_lines(f, cfg, Path(path).name)


def obs_table(log: L86Log) -> pd.DataFrame:
    return log.table


def format_quality(q: dict) -> str:
    L = [f"L86 {q['file']}  (not yet validated on real data)",
         f"Format: {q['format']} {q['format_counts']}; lines {q['lines']}, NMEA {q['nmea']}, "
         f"bad checksum {q['bad_checksum']}, PMTK {q['pmtk']}, noise {q['noise']}"]
    if q.get("epochs"):
        L += [f"Span: {q['start']['ist']} -> {q['end']['ist']} IST ({q['duration_min']} min), "
              f"{q['epochs']} epochs, output {q['output_rate_hz']} Hz, effective C/N0 rate {q['effective_rate_hz']} Hz "
              f"(C/N0 changes in {q['cn0_change_fraction']} of consecutive samples)",
              f"Epoch spacing (ms): {q['epoch_spacing_ms_hist']}",
              f"GSV groups {q['gsv_groups']} (incomplete {q['gsv_incomplete']}, no time {q['gsv_no_time']}); "
              f"time source {q['time_source']}; host-NMEA offset {q['host_minus_nmea_offset_s']}",
              f"Satellites per constellation: {q['satellites_per_constellation']}",
              f"PRN ranges per talker: {q['prn_ranges_per_talker']}",
              f"Gaps > 5 s: {q['gaps_over_s']['count']}"]
    L.append("WARNINGS:" if q["warnings"] else "WARNINGS: none")
    L += [f"  - [{w['code']}] {w['message']}" for w in q["warnings"]]
    return "\n".join(L)
