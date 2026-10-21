"""One-minute record: the unit written by batch (minutes.jsonl), live and replay modes.

    {schema_version, node_id, receiver, minute_utc, minute_ist, status, rev,
     agc_flag, sats_tracked, navic_status, navic_sats, l86_rate_hz, warnings,
     satellites: [{sys, svid, band, elev, az, cn0_mean, s4, s4_noise, s4_flag,
                   fade_count, max_fade_db, lock_loss, class, source}, ...]}

The JSON Schema is minute_record.schema.json next to this file. records_from_tables() is
the only place records are built, so batch and live outputs cannot drift apart.

status/rev: live and replay publish each minute twice, "provisional" (rev 1, ~5 s after the
minute) and "final" (rev 2, once the result can no longer change; equal to batch). Batch
records are written once, "final" rev 1. Readers keep the highest rev per (receiver, minute).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ionwatch.proc import common

SCHEMA_VERSION = "1.1"
SCHEMA_PATH = Path(__file__).with_name("minute_record.schema.json")
CLASS_PRIORITY = ["iono_likely", "iono_possible", "local", "local_or_unknown", "low_elev", "edge_of_gap"]


@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate(record: dict) -> None:
    """Raise jsonschema.ValidationError if the record does not match the schema."""
    import jsonschema
    jsonschema.validate(record, schema())


def _num(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def s4_flag(reason: str | None, valid: bool) -> str:
    if valid:
        return "valid"
    if not isinstance(reason, str) or not reason:
        return "no_s4"
    head = reason.split(" ")[0]
    return head if head in ("below_noise", "insufficient_samples", "step_artifact", "below_elev_mask") else "no_s4"


def band_of_signal(signal: str) -> str:
    return signal.split("_", 1)[1] if "_" in signal else signal


def _navic_status(fm: pd.DataFrame) -> str:
    nav = fm[fm["constellation"] == "NavIC"]
    if nav.empty:
        return "no_data"
    return "tracked" if (nav["n"] > 0).any() else "visible_not_tracked"


def records_from_tables(minutes, receiver: str, node_id: str, fmin: pd.DataFrame, s4: pd.DataFrame,
                        classified: pd.DataFrame, interference: pd.DataFrame, cfg: dict,
                        extra_warnings: dict | None = None, status: str = "final", rev: int = 1,
                        l86_rate_hz: dict | float | None = None) -> list[dict]:
    """Build one record per minute in ``minutes`` for ``receiver`` from the per-minute tables."""
    tz = cfg["display_tz"]
    thr_cn0 = cfg["quality"]["low_cn0_mean_dbhz"]
    agc_thr = cfg["interference"]["agc_change_db"]
    extra_warnings = extra_warnings or {}
    fm_r = fmin[fmin["receiver"] == receiver]
    s4_r = s4[s4["receiver"] == receiver] if len(s4) else s4
    cl_r = classified[classified["receiver"] == receiver] if len(classified) else classified
    it_r = (interference[interference["receiver"] == receiver].set_index("minute_utc")
            if len(interference) else pd.DataFrame())
    fm_by = dict(tuple(fm_r.groupby("minute_utc"))) if len(fm_r) else {}
    s4_by = dict(tuple(s4_r.groupby("minute_utc"))) if len(s4_r) else {}
    out = []
    for m in minutes:
        m = pd.Timestamp(m)
        fm = fm_by.get(m, fm_r.iloc[0:0])
        s4m = s4_by.get(m, s4_r.iloc[0:0] if len(s4_r) else pd.DataFrame(columns=["s4"]))
        ev = (cl_r[(cl_r["start_utc"] <= m) & (cl_r["end_utc"] > m)] if len(cl_r) else cl_r)
        sats = []
        rows = fm[(fm["n"] > 0) | (fm["lock_losses"] > 0) | (fm["edge_lock_losses"] > 0)] if len(fm) else fm
        for r in rows.itertuples(index=False):
            key = (r.constellation, r.svid, r.signal)
            sr = s4m[(s4m["constellation"] == r.constellation) & (s4m["svid"] == r.svid)
                     & (s4m["signal"] == r.signal)] if len(s4m) else s4m
            if len(sr):   # Raw and Status windows of one signal: keep the one with more samples
                sr = sr.sort_values(["s4_valid", "n"], ascending=False).iloc[0]
            e = ev[(ev["constellation"] == r.constellation) & (ev["svid"] == r.svid)
                   & (ev["signal"] == r.signal)] if len(ev) else ev
            cls = next((c for c in CLASS_PRIORITY if len(e) and (e["class"] == c).any()), "none")
            reason = str(e.loc[e["class"] == cls, "reason"].iloc[0]) if cls != "none" else ""
            bp = e.loc[e["class"] == cls, "both_phones"].iloc[0] if cls != "none" and "both_phones" in e else None
            both_phones = None if bp is None or (isinstance(bp, float) and np.isnan(bp)) else bool(bp)
            sats.append({
                "sys": common.SHORT[key[0]], "svid": int(key[1]), "band": band_of_signal(key[2]),
                "elev": _num(r.elev_deg_mean, 1), "az": _num(r.az_deg_mean, 1), "cn0_mean": _num(r.cn0_mean),
                "s4": _num(sr["s4"], 3) if len(sr) else None,
                "s4_noise": _num(sr["s4_noise"], 3) if len(sr) else None,
                "s4_flag": s4_flag(sr["reason"], bool(sr["s4_valid"])) if len(sr) else "no_s4",
                "fade_count": int((~e["edge_of_gap"].astype(bool)).sum()) if len(e) else 0,
                "max_fade_db": _num(r.fade_depth_db),
                "lock_loss": int(r.lock_losses), "class": cls, "reason": reason, "both_phones": both_phones,
                "source": r.source if r.source in ("raw", "status", "nmea", "mixed") else
                ("nmea" if receiver == "l86" else "mixed")})
        warnings = list(extra_warnings.get(m, []))
        active = int(fm["receiver_active_s"].max()) if len(fm) else 0
        if not len(fm):
            warnings.append("no_data")
        elif active < cfg["live"]["partial_minute_active_s"]:
            warnings.append(f"receiver_partial:{active}s")
        tracked = fm[fm["n"] > 0] if len(fm) else fm
        if len(tracked) and tracked["cn0_mean"].mean() < thr_cn0:
            warnings.append("low_cn0")
        agc_flag = None
        if m in it_r.index:
            irow = it_r.loc[m]
            if irow["local_cause_flag"]:
                warnings.append(f"local_cause:{irow['reason']}")
            if pd.notna(irow["agc_change_max_db"]):
                agc_flag = bool(abs(irow["agc_change_max_db"]) >= agc_thr)
        nav = fm[fm["constellation"] == "NavIC"] if len(fm) else fm
        navic_sats = [{"svid": int(sv), "elev": _num(g["elev_deg_mean"].mean(), 1),
                       "az": _num(common.circmean_deg(g["az_deg_mean"]), 1),
                       "cn0_mean": _num(g["cn0_mean"].mean())} for sv, g in nav.groupby("svid")] if len(nav) else []
        rate = l86_rate_hz.get(m) if isinstance(l86_rate_hz, dict) else l86_rate_hz
        out.append({
            "schema_version": SCHEMA_VERSION, "node_id": node_id, "receiver": receiver,
            "minute_utc": f"{m:%Y-%m-%dT%H:%M}:00Z", "minute_ist": m.tz_convert(tz).isoformat(),
            "status": status, "rev": rev,
            "agc_flag": agc_flag,
            "sats_tracked": int(tracked[["constellation", "svid"]].drop_duplicates().shape[0]) if len(tracked) else 0,
            "navic_status": _navic_status(fm) if len(fm) else "no_data",
            "navic_sats": navic_sats,
            "l86_rate_hz": _num(rate) if receiver == "l86" else None,
            "warnings": warnings,
            "satellites": sorted(sats, key=lambda s: (s["sys"], s["svid"], s["band"]))})
    return out


def write_jsonl(records: list[dict], path: Path, append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_latest(record: dict, path: Path) -> None:
    """Atomically replace latest.json (readers never see a half-written file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_versions(records: list[dict]) -> list[dict]:
    """Highest rev per (receiver, minute_utc), sorted by minute then receiver."""
    best: dict[tuple, dict] = {}
    for r in records:
        k = (r["receiver"], r["minute_utc"])
        if k not in best or r["rev"] > best[k]["rev"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: (r["minute_utc"], r["receiver"]))


def finals(records: list[dict]) -> list[dict]:
    return sorted((r for r in records if r["status"] == "final"), key=lambda r: (r["minute_utc"], r["receiver"]))
