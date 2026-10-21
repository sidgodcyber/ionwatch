"""Live (serial L86) and replay (recorded GnssLogger) modes.

Both push observations into a MinuteEngine: a rolling buffer that runs the batch code (fades,
S4, dualfreq, interference, classify, schema.records_from_tables) and publishes every minute
twice: a provisional record ~5 s after the minute ends (half-window baseline) and a final
record once the result can no longer change, which equals the batch result. Records are
appended to minutes.jsonl and latest.json is replaced atomically, so batch, live and replay
outputs share one format.

Time: the clock follows the data (NMEA / GnssLogger time) and is extrapolated with the
wall clock (x clock_scale) between data, so minutes keep closing through an unplug and a Pi
without network time cannot shift the records.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ionwatch import schema
from ionwatch.proc import classify, common, dualfreq, fades, interference, s4

log = logging.getLogger("ionwatch.live")
KEYS = common.series_keys()
MINUTE = pd.Timedelta(minutes=1)
COUNTER_WARNINGS = {"bad_checksum": "bad_checksum", "noise": "garbage_lines", "gsv_incomplete": "gsv_incomplete",
                    "gsv_no_time": "no_nmea_time", "rmc_no_fix": "no_fix"}


def setup_logging(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("ionwatch")
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        if getattr(h, "_ionwatch", False):
            root.removeHandler(h)
            h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(out_dir / name, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        h._ionwatch = True
        root.addHandler(h)


class StreamClock:
    """Data time, extrapolated by wall time x scale once data is ``hold_s`` wall seconds overdue
    (scale 0 = data only). The hold keeps a busy loop (processing a minute) from looking like a
    receiver silence when the clock runs fast; at 1x it only delays closing by hold_s."""

    def __init__(self, scale: float = 1.0, hold_s: float = 2.0):
        self.scale, self.hold_s = scale, hold_s
        self.t_data: pd.Timestamp | None = None
        self._wall = None
        self._last = None

    def on_data(self, t: pd.Timestamp) -> None:
        if self.t_data is None or t > self.t_data:
            self.t_data = t
            self._wall = time.monotonic()

    def now(self) -> pd.Timestamp:
        if self.t_data is None:
            t = pd.Timestamp.now(tz="UTC")
        elif self.scale:
            late = max(0.0, time.monotonic() - self._wall - self.hold_s)
            t = self.t_data + pd.Timedelta(seconds=late * self.scale)
        else:
            t = self.t_data
        self._last = t if self._last is None or t > self._last else self._last
        return self._last


class RecordWriter:
    def __init__(self, out_dir: Path):
        self.jsonl = Path(out_dir) / "minutes.jsonl"
        self.latest = Path(out_dir) / "latest.json"
        self.count = 0
        self.finals = 0
        self._latest = None

    def write(self, rec: dict) -> bool:
        try:
            schema.validate(rec)
        except Exception as e:  # a bug, not a data problem: keep running, never publish it
            log.error("record for %s failed schema validation, not written: %s", rec.get("minute_utc"), e)
            return False
        schema.write_jsonl([rec], self.jsonl, append=True)
        key = (rec["minute_utc"], rec["rev"])
        if self._latest is None or key >= self._latest:      # an older minute's final never replaces it
            schema.write_latest(rec, self.latest)
            self._latest = key
        self.count += 1
        self.finals += rec["status"] == "final"
        log.info("%s %s: %d sats tracked, %d satellites, warnings %s", rec["minute_utc"], rec["status"],
                 rec["sats_tracked"], len(rec["satellites"]), rec["warnings"] or "-")
        return True


class RawNmeaLog:
    """Raw NMEA with a host-time prefix, one file per (data-clock) UTC day: nmea_YYYY-MM-DD.nmea.
    The prefix makes the file re-runnable in batch (`ionwatch run --l86`) and exposes any host
    clock offset in the L86 quality report."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._day, self._f = None, None

    def write(self, line: str, host: datetime, clock: pd.Timestamp) -> None:
        if not line:
            return
        day = f"{clock:%Y-%m-%d}"
        if day != self._day:
            self.close()
            self._day = day
            self._f = open(self.folder / f"nmea_{day}.nmea", "a", encoding="ascii", errors="replace")
        self._f.write(f"{host:%Y-%m-%dT%H:%M:%S}.{host.microsecond // 1000:03d}Z,{line}\n")

    def flush(self):
        if self._f:
            self._f.flush()

    def close(self):
        if self._f:
            self._f.close()
            self._f = None


class FrameBuffer:
    """Time-ordered DataFrame chunks with cached time ranges. Small chunks (one per data second
    in replay) are merged into blocks so windowing and trimming stay cheap."""

    def __init__(self, block: int = 120):
        self.block = block
        self.blocks: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []
        self._tail: list[pd.DataFrame] = []

    def append(self, df: pd.DataFrame) -> None:
        self._tail.append(df)
        if len(self._tail) >= self.block:
            self._compact()

    def _compact(self) -> None:
        if self._tail:
            df = pd.concat(self._tail, ignore_index=True) if len(self._tail) > 1 else self._tail[0]
            self.blocks.append((df["time_utc"].min(), df["time_utc"].max(), df))
            self._tail = []

    def empty(self) -> bool:
        return not self.blocks and not self._tail

    def last_time(self) -> pd.Timestamp:
        self._compact()
        return max(b for _, b, _ in self.blocks)

    def window(self, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
        self._compact()
        parts = [d for a, b, d in self.blocks if b >= lo and a <= hi]
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        return df[(df["time_utc"] >= lo) & (df["time_utc"] <= hi)].reset_index(drop=True)

    def trim(self, keep_from: pd.Timestamp) -> None:
        self.blocks = [x for x in self.blocks if x[1] >= keep_from]


class MinuteEngine:
    """Rolling buffer of tidy C/N0 observations -> two records per minute.

    provisional (rev 1): ``provisional_delay_s`` after the minute ends, from the series seen in
        that minute and the data so far (half-window baseline).
    final (rev 2): once nothing that batch would use can still change: the centred 15-min
        baseline and AGC median (7.5 min after the minute), the edge-of-gap guard (a silence must
        have lasted silence_min_s), and every fade event touching the minute has ended (+7.5 min).
        Computed with the batch functions over all series, so it equals the batch result as long
        as those events started within ``history_min`` of the minute. After ``final_max_wait_min``
        the final is published anyway with the warning ``final_capped``.
    """

    def __init__(self, receiver: str, node_id: str, cfg: dict):
        self.receiver, self.node_id, self.cfg = receiver, node_id, cfg
        self.live = cfg["live"]
        f, it = cfg["fades"], cfg["interference"]
        self.prov_delay = pd.Timedelta(seconds=self.live["provisional_delay_s"])
        # How long after a minute (or an event) ends until no input of the final can change.
        self.settle = pd.Timedelta(seconds=max(f["baseline_window_min"] * 30, it["agc_baseline_window_min"] * 30,
                                               f["edge_guard_s"] + f["silence_min_s"]) + 5)
        self.max_wait = pd.Timedelta(minutes=self.live["final_max_wait_min"])
        self.frames = FrameBuffer()
        self.agc_frames = FrameBuffer()
        self._active_parts: list[np.ndarray] = []
        self._active: np.ndarray = np.array([], dtype=np.int64)
        self.next_minute: pd.Timestamp | None = None            # next minute to publish provisionally
        self.pending: dict[pd.Timestamp, dict] = {}               # minute -> {ready_at, warnings}
        self._flags: dict[pd.Timestamp, list[str]] = defaultdict(list)
        self._counts: dict[pd.Timestamp, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.l86_rate_hz: float | None = None                    # set by live mode after measuring

    # ---------------------------------------------------------------- input
    def start_at(self, t: pd.Timestamp, from_wall_clock: bool = False) -> None:
        if self.next_minute is None:
            self.next_minute = t.floor("min")
            self._wall_start = from_wall_clock

    def add(self, obs: pd.DataFrame, agc: pd.DataFrame | None = None) -> None:
        if len(obs):
            first = obs["time_utc"].min().floor("min")
            if self.next_minute is None:
                self.next_minute = first
            elif getattr(self, "_wall_start", False) and abs(first - self.next_minute) >= pd.Timedelta(minutes=10):
                # Minutes were started from the host clock before any data (no network time on a
                # Pi): follow the receiver's time from now on. Gaps *between* data are kept and
                # closed as no_data minutes.
                log.warning("host clock %s and data time %s disagree; following data time", self.next_minute, first)
                self.flag(first, "clock_jump")
                self.next_minute = first
            self._wall_start = False
            self.frames.append(obs)
            self._active_parts.append(common.active_seconds(obs["time_utc"]))
        if agc is not None and len(agc):
            self.agc_frames.append(agc)

    def flag(self, t: pd.Timestamp, code: str) -> None:
        m = t.floor("min")
        if code not in self._flags[m]:
            self._flags[m].append(code)

    def count(self, t: pd.Timestamp, code: str, n: int = 1) -> None:
        self._counts[t.floor("min")][code] += n

    def _warnings(self, m: pd.Timestamp) -> list[str]:
        w = list(self._flags.pop(m, []))
        w += [f"{k}:{v}" if k != "no_fix" else k for k, v in self._counts.pop(m, {}).items()]
        return w

    @property
    def active(self) -> np.ndarray:
        """All receiver-active seconds seen (global, so silences that began before the buffer count)."""
        if self._active_parts:
            self._active = np.unique(np.concatenate([self._active, *self._active_parts]))
            self._active_parts = []
        return self._active

    # ---------------------------------------------------------------- output
    def close_ready(self, clock: pd.Timestamp) -> list[dict]:
        """Provisional records for minutes that just ended, final records that became exact."""
        out = []
        if self.next_minute is None:
            return out
        while self.next_minute + MINUTE + self.prov_delay <= clock:
            m = self.next_minute
            warnings = self._warnings(m)
            rec, _ = self._safe_compute(m, clock, warnings, final=False, now=clock)
            out.append(rec)
            self.pending[m] = {"ready_at": m + MINUTE + self.settle, "warnings": warnings}
            self.next_minute += MINUTE
        out += self._finals(clock, now=clock)
        self._trim()
        return out

    def _finals(self, clock: pd.Timestamp, now: pd.Timestamp | None, force: bool = False) -> list[dict]:
        out = []
        for m in sorted(self.pending):
            p = self.pending[m]
            if not force and p["ready_at"] > clock:
                continue
            rec, ev_end = self._safe_compute(m, clock, p["warnings"], final=True, now=now)
            capped = clock - (m + MINUTE) >= self.max_wait
            if not force and ev_end is not None and ev_end + self.settle > clock and not capped:
                p["ready_at"] = ev_end + self.settle          # an event touching m is still open
                continue
            if capped and not force and ev_end is not None and ev_end + self.settle > clock:
                rec["warnings"].append("final_capped")
            out.append(rec)
            del self.pending[m]
        return out

    def flush(self) -> list[dict]:
        """End of a replay / file: publish every remaining minute, provisional then final.
        The end of the data is not a receiver silence (now=None), as in batch."""
        if self.frames.empty() or self.next_minute is None:
            return []
        last = self.frames.last_time()
        end = last.floor("min") + MINUTE
        out = []
        while self.next_minute < end:
            m = self.next_minute
            warnings = self._warnings(m)
            rec, _ = self._safe_compute(m, last, warnings, final=False, now=None)
            out.append(rec)
            self.pending[m] = {"ready_at": m, "warnings": warnings}
            self.next_minute += MINUTE
        out += self._finals(last, now=None, force=True)
        return out

    def _trim(self) -> None:
        oldest = min([*self.pending, self.next_minute]) if self.pending else self.next_minute
        keep_from = oldest - pd.Timedelta(minutes=self.live["history_min"])
        self.frames.trim(keep_from)
        self.agc_frames.trim(keep_from)
        a = self.active
        self._active = a[a >= keep_from.timestamp() - 86400]     # one day back is plenty for silences

    def _rates(self, buf: pd.DataFrame) -> tuple[float, float]:
        """(sample rate, effective C/N0 rate) from the buffer; the phone is fixed at 1 Hz."""
        if self.receiver == "phone":
            return 1.0, 1.0
        ep = np.unique(buf["time_utc"].astype("int64").to_numpy()) / 1e9
        d = np.diff(ep)
        rate = float(round(1 / np.median(d))) if len(d) > 5 and np.median(d) > 0 else float(self.live["l86_rate_hz"])
        rate = max(rate, 1.0)
        from ionwatch.io.nmea_l86 import effective_rate
        frac = [np.mean(v[1:] != v[:-1]) for v in
                (g["cn0_dbhz"].dropna().to_numpy() for _, g in buf.groupby(KEYS)) if len(v) > 10]
        return rate, effective_rate(rate, float(np.median(frac)) if frac else np.nan)

    def _record(self, m, fmin, s4t, cls, intf, warnings, final, rate=None) -> dict:
        return schema.records_from_tables([m], self.receiver, self.node_id, fmin, s4t, cls, intf, self.cfg,
                                          {m: list(warnings)}, status="final" if final else "provisional",
                                          rev=2 if final else 1, l86_rate_hz=rate)[0]

    def _empty_record(self, m: pd.Timestamp, warnings: list[str], final: bool) -> dict:
        return self._record(m, pd.DataFrame(columns=fades.MINUTE_COLS), pd.DataFrame(columns=s4.COLS),
                            pd.DataFrame(columns=classify.COLS), pd.DataFrame(columns=interference.COLS),
                            warnings, final)

    def _safe_compute(self, m, clock, warnings, final, now):
        try:
            return self._compute(m, clock, warnings, final, now)
        except Exception:
            log.exception("processing minute %s (%s) failed", m, "final" if final else "provisional")
            return self._empty_record(m, list(warnings) + ["processing_error"], final), None

    def _compute(self, m, clock, warnings, final, now) -> tuple[dict, pd.Timestamp | None]:
        """One record for minute m, plus the latest end of the fade events touching m."""
        cfg = self.cfg
        buf = self.frames.window(m - pd.Timedelta(minutes=self.live["history_min"]), clock)
        in_m = buf[(buf["time_utc"] >= m) & (buf["time_utc"] < m + MINUTE)] if len(buf) else buf
        if not len(in_m):
            return self._empty_record(m, warnings, final), None
        active = self.active
        if not final:                                   # provisional: only the series seen in m
            buf = buf.merge(in_m[KEYS].drop_duplicates(), on=KEYS)
        rate, eff = self._rates(buf)
        obs_b = fades.add_baseline(buf, active, cfg, rate)
        if not final:
            obs_b = obs_b[obs_b["time_utc"] >= m - pd.Timedelta(minutes=3)]
        fmin = fades.per_minute(obs_b, active, cfg, rate, now=now)
        ev = fades.events(fmin)
        ev_m = ev[(ev["start_utc"] <= m) & (ev["end_utc"] > m)]
        lo = min([m, *ev_m["start_utc"]])
        hi = max([m + MINUTE, *ev_m["end_utc"]])
        t = obs_b["time_utc"]
        span = obs_b[(t >= lo - MINUTE) & (t < hi + MINUTE)]
        method = "indicative_1Hz" if self.receiver == "phone" else f"nmea_{eff}Hz"
        s4t = s4.compute(span, cfg, eff, method)
        if self.receiver == "phone":
            dual, _ = dualfreq.compute(span, fmin, cfg)
        else:
            dual = pd.DataFrame(columns=dualfreq.COLS)
        agc = None
        if self.receiver == "phone" and not self.agc_frames.empty():
            agc = self.agc_frames.window(lo - pd.Timedelta(minutes=8), clock)
            agc = agc if len(agc) else None
        intf, _ = interference.compute(fmin, agc, active, cfg, self.receiver)
        cls = classify.classify(ev_m, intf, dual, s4t, cfg)
        rate_m = self.l86_rate_hz if self.l86_rate_hz is not None else common.epoch_rate(in_m["time_utc"])
        rec = self._record(m, fmin, s4t, cls, intf, warnings, final, rate=rate_m)
        return rec, (ev_m["end_utc"].max() if len(ev_m) else None)


# -------------------------------------------------------------------- live L86 over serial

def pmtk(body: str) -> bytes:
    x = 0
    for ch in body:
        x ^= ord(ch)
    return f"${body}*{x:02X}\r\n".encode("ascii")


# PMTK314 field order: GLL, RMC, VTG, GGA, GSA, GSV, GRS, GST, (10 reserved), ZDA, MCHN.
PMTK314_RMC_GGA_GSA_GSV = "PMTK314,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0"
HIGH_BAUD = 115200


class L86RateSetup:
    """Configure the L86 output rate on every (re)open; the module forgets its settings when it
    loses power (no backup battery).

    rate 1: leave the module at its defaults (9600 baud, 1 Hz).
    rate 5/10: at the start baud send PMTK251 (115200), reopen at 115200, then PMTK314 (only
    RMC, GGA, GSA, GSV) and PMTK220 (fix interval). If the module already runs at 115200 the
    9600-baud PMTK251 is ignored and the 115200 reopen still works. The GSV rate is then
    measured over ``measure_s`` seconds of data; below 80 % of the request the module is put
    back to 1 Hz at the start baud and the run continues there (warning l86_rate_fallback).
    """

    def __init__(self, rate: int, baud: int, measure_s: float = 30.0):
        self.requested, self.baud, self.measure_s = rate, baud, measure_s
        self.fallen_back = False
        self.measured: float | None = None
        self._epochs: set = set()
        self._t0 = None

    @property
    def high(self) -> bool:
        return self.requested > 1 and not self.fallen_back

    def open(self, serial_mod, port: str, timeout: float):
        ser = serial_mod.serial_for_url(port, baudrate=self.baud, timeout=timeout)
        if not self.high:
            if self.fallen_back:                         # module may have kept 5/10 Hz settings
                ser.write(pmtk("PMTK220,1000"))
            return ser
        ser.write(pmtk(f"PMTK251,{HIGH_BAUD}"))
        ser.flush()
        time.sleep(0.1)
        ser.close()
        ser = serial_mod.serial_for_url(port, baudrate=HIGH_BAUD, timeout=timeout)
        ser.write(pmtk(PMTK314_RMC_GGA_GSA_GSV))
        ser.write(pmtk(f"PMTK220,{int(1000 / self.requested)}"))
        ser.flush()
        self._epochs, self._t0, self.measured = set(), None, None     # re-measure after every open
        log.info("L86 configured: %d baud, RMC/GGA/GSA/GSV only, %d Hz requested", HIGH_BAUD, self.requested)
        return ser

    def observe(self, rows: list[dict]) -> str | None:
        """Feed parsed rows; returns 'ok' or 'fallback' once the measurement window is complete."""
        if not self.high or self.measured is not None or not rows:
            return None
        for r in rows:
            t = pd.Timestamp(r["time_utc"])
            if self._t0 is None:
                self._t0 = t.ceil("s")                   # start on a whole second
            if t >= self._t0:
                self._epochs.add(t.value)
        last = max(pd.Timestamp(r["time_utc"]) for r in rows)
        if self._t0 is not None and (last - self._t0).total_seconds() >= self.measure_s:
            span = (last - self._t0).total_seconds()
            self.measured = round(len(self._epochs) / span, 2)
            if self.measured < 0.8 * self.requested:
                return "fallback"
            log.info("L86 GSV rate measured %.2f Hz over %.0f s (requested %d Hz): ok",
                     self.measured, span, self.requested)
            return "ok"
        return None

    def fallback(self, ser, serial_mod, port: str, timeout: float):
        log.warning("L86 GSV rate %.2f Hz < 80 %% of the requested %d Hz: back to 1 Hz at %d baud",
                    self.measured, self.requested, self.baud)
        ser.write(pmtk("PMTK220,1000"))
        ser.write(pmtk(f"PMTK251,{self.baud}"))
        ser.flush()
        time.sleep(0.1)
        ser.close()
        self.fallen_back = True
        return serial_mod.serial_for_url(port, baudrate=self.baud, timeout=timeout)


def write_run_info(out_dir: Path, **info) -> None:
    import json
    info["started_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    (Path(out_dir) / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


def run_live(port: str, baud: int, out_dir, cfg: dict, node_id: str, clock_scale: float = 1.0,
             max_minutes: int | None = None, duration_s: float | None = None, l86_rate: int = 1) -> int:
    """Read NMEA from ``port`` (a device such as /dev/serial0 or COM3, or any pyserial URL such
    as socket://127.0.0.1:5000) until stopped. Survives unplug/replug, bad checksums, partial
    lines and no-fix periods. ``max_minutes`` stops after that many minutes have their final
    record. Returns the number of records written (provisional + final)."""
    import serial

    from ionwatch.io import nmea_l86

    out_dir = Path(out_dir)
    setup_logging(out_dir, "live.log")
    write_run_info(out_dir, mode="live", node_id=node_id, receiver="l86", port=port, baud=baud,
                   l86_rate_requested=l86_rate)
    lv = cfg["live"]
    parser = nmea_l86.NmeaParser(cfg)
    engine = MinuteEngine("l86", node_id, cfg)
    writer = RecordWriter(out_dir)
    raw = RawNmeaLog(out_dir / "raw")
    clock = StreamClock(clock_scale, lv["clock_hold_s"])
    rate = L86RateSetup(l86_rate, baud, lv["rate_measure_s"])
    prev = dict(parser.counts)
    n_acks = 0
    ser, pending, connected_once, down_since, last_bytes = None, b"", False, None, time.monotonic()
    t0 = time.monotonic()
    log.info("live: port %s @ %d baud, L86 rate %d Hz requested, node %s, out %s", port, baud, l86_rate,
             node_id, out_dir)

    def done():
        return max_minutes and writer.finals >= max_minutes

    try:
        while True:
            if duration_s is not None and time.monotonic() - t0 > duration_s:
                log.info("duration reached, stopping")
                break
            if engine.next_minute is None and time.monotonic() - t0 > 120 / max(clock_scale, 1e-9):
                engine.start_at(clock.now(), from_wall_clock=True)   # no data: still one record per minute
            if ser is None:
                try:
                    ser = rate.open(serial, port, lv["read_timeout_s"])
                    log.info("serial %s open%s", port, " (reconnected)" if connected_once else "")
                    if connected_once:
                        engine.flag(clock.now(), "serial_reconnected")
                    connected_once, down_since, pending, last_bytes = True, None, b"", time.monotonic()
                except (serial.SerialException, OSError, ValueError) as e:
                    if down_since is None:
                        log.warning("serial %s unavailable: %s; retrying every %.1f s", port, e, lv["reconnect_s"])
                        down_since = time.monotonic()
                    engine.flag(clock.now(), "serial_disconnected")
                    _emit(engine.close_ready(clock.now()), writer)
                    if done():
                        break
                    time.sleep(lv["reconnect_s"])
                    continue
            if time.monotonic() - last_bytes > lv["silence_reopen_s"]:
                # A GPIO UART does not error when the L86 is unplugged; it just goes quiet.
                log.warning("no bytes from %s for %.0f s (L86 unplugged or not powered?); reopening",
                            port, time.monotonic() - last_bytes)
                engine.flag(clock.now(), "no_serial_data")
                ser.close()
                ser, last_bytes = None, time.monotonic()
                continue
            try:
                chunk = ser.read(4096)
            except (serial.SerialException, OSError) as e:
                log.warning("serial %s lost: %s", port, e)
                engine.flag(clock.now(), "serial_disconnected")
                try:
                    ser.close()
                except Exception:
                    pass
                ser, down_since = None, time.monotonic()
                if pending:
                    engine.count(clock.now(), "partial_line")
                pending = b""
                continue
            rows = []
            if chunk:
                last_bytes = time.monotonic()
                pending += chunk
                *lines, pending = pending.split(b"\n")
                if len(pending) > 1024:                 # no newline for 1 kB: garbage
                    engine.count(clock.now(), "garbage_lines")
                    pending = b""
                host = datetime.now(timezone.utc)
                for bl in lines:
                    line = bl.decode("ascii", errors="replace").strip()
                    rows += parser.feed(line, host=host)
                    day = parser.cur_time or clock.t_data or host    # file per NMEA (not host) day
                    raw.write(line, host, pd.Timestamp(day))
                raw.flush()
            if len(parser.acks) > n_acks:
                for cmd, flag in parser.acks[n_acks:]:
                    log.info("L86 ack PMTK%s: %s", cmd, {"3": "ok", "2": "failed", "1": "unsupported",
                                                         "0": "invalid"}.get(flag, flag))
                n_acks = len(parser.acks)
            if rows:
                obs = nmea_l86.to_obs(nmea_l86.rows_to_table(rows))
                if len(obs):
                    clock.on_data(obs["time_utc"].max())
                    engine.add(obs)
                verdict = rate.observe(rows)
                if verdict:
                    engine.flag(clock.now(), f"l86_rate_measured:{rate.measured}Hz/{rate.requested}Hz")
                if verdict == "fallback":
                    engine.flag(clock.now(), "l86_rate_fallback")
                    ser = rate.fallback(ser, serial, port, lv["read_timeout_s"])
                    pending = b""
            now = clock.now()
            for k, code in COUNTER_WARNINGS.items():
                if parser.counts[k] > prev[k]:
                    engine.count(now, code, parser.counts[k] - prev[k])
            prev = dict(parser.counts)
            _emit(engine.close_ready(now), writer)
            if done():
                break
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        if ser is not None:
            ser.close()
        raw.close()
        log.info("live stopped: %d records (%d final), parser counts %s", writer.count, writer.finals,
                 parser.counts)
    return writer.count


def _emit(records, writer: RecordWriter) -> None:
    for r in records:
        writer.write(r)


# -------------------------------------------------------------------- replay a GnssLogger file

def run_replay(phone_paths: list[str], speed: float, out_dir, cfg: dict, node_id: str,
               max_minutes: int | None = None) -> int:
    """Push a recorded GnssLogger session through the live path at ``speed`` x real time
    (0 = as fast as possible). Produces minutes.jsonl and latest.json like live mode."""
    from ionwatch.io import gnsslogger

    out_dir = Path(out_dir)
    setup_logging(out_dir, "replay.log")
    (out_dir / "minutes.jsonl").unlink(missing_ok=True)
    write_run_info(out_dir, mode="replay", node_id=node_id, receiver="phone",
                   source=[Path(p).name for p in phone_paths], speed=speed)
    plog = gnsslogger.read_phone(phone_paths, cfg)
    gnsslogger.attach_elevation(plog, cfg)
    obs, stats = gnsslogger.cn0_obs(plog, cfg)
    obs = obs.sort_values("time_utc", kind="stable").reset_index(drop=True)
    agc = plog.agc.sort_values("time_utc").reset_index(drop=True) if len(plog.agc) else None
    log.info("replay %s at speed %s: %d C/N0 rows, %s", [Path(p).name for p in phone_paths], speed, len(obs), stats)
    engine = MinuteEngine("phone", node_id, cfg)
    writer = RecordWriter(out_dir)
    clock = StreamClock(speed, cfg["live"]["clock_hold_s"])
    secs = obs["time_utc"].dt.floor("s")
    edges = np.flatnonzero(np.r_[True, secs.to_numpy()[1:] != secs.to_numpy()[:-1], True])
    agc_ns = agc["time_utc"].astype("int64").to_numpy() if agc is not None else None
    t0, wall0 = secs.iloc[0], time.monotonic()
    engine.start_at(t0)

    def done():
        return max_minutes and writer.finals >= max_minutes

    for a, b in zip(edges[:-1], edges[1:]):
        t = secs.iloc[a]
        if speed > 0:
            target = wall0 + (t - t0).total_seconds() / speed
            while (rem := target - time.monotonic()) > 0:
                time.sleep(min(rem, 0.25))
                _emit(engine.close_ready(clock.now()), writer)
                if done():
                    return writer.count
        chunk_agc = None
        if agc is not None:
            lo, hi = np.searchsorted(agc_ns, [t.value, t.value + 10**9])
            chunk_agc = agc.iloc[lo:hi] if hi > lo else None
        engine.add(obs.iloc[a:b], chunk_agc)
        clock.on_data(t)
        _emit(engine.close_ready(clock.now()), writer)
        if done():
            return writer.count
    _emit(engine.flush(), writer)
    log.info("replay finished: %d records", writer.count)
    return writer.count
