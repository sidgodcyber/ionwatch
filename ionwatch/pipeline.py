"""Run the whole offline pipeline for one session and write data/processed/<session>/.

Phone path: parse -> elevation join -> tidy C/N0 (Raw + Status) -> fades -> NavIC -> S4 ->
code TEC -> dual-frequency -> interference -> classification -> plots + summary.
L86 path (only if given): parse -> fades -> S4 -> cross-check against the phone.
Stages without the data they need (no L5, no NavIC, no AGC, no L86) are skipped with a
note and still write header-only tables, so the output layout never changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ionwatch import schema
from ionwatch.io import gnsslogger
from ionwatch.proc import classify, common, dualfreq, fades, interference, navic, s4, tec_code


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.ndarray, pd.Index)):
        return list(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable, ensure_ascii=False), encoding="utf-8")


def _cat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-receiver tables, skipping empty ones (keeps the columns if all are empty)."""
    full = [p for p in parts if len(p)]
    return pd.concat(full, ignore_index=True) if full else parts[0].iloc[0:0]


def receiver_stages(receiver: str, obs_b: pd.DataFrame, fmin: pd.DataFrame, s4_r: pd.DataFrame,
                    active: np.ndarray, agc: pd.DataFrame | None, cfg: dict,
                    events: pd.DataFrame | None = None) -> dict:
    """Events, dual-frequency check, local-cause check and classes for one receiver.

    Shared by the batch run and live / replay mode (which call it on a rolling buffer).
    ``events`` lets the caller pass events already flagged by twophone.flag_events.
    """
    ev = fades.events(fmin) if events is None else events
    if receiver.startswith("phone"):
        dual, dual_note = dualfreq.compute(obs_b, fmin, cfg)
    else:
        dual, dual_note = pd.DataFrame(columns=dualfreq.COLS), "single-band receiver"
    intf, info = interference.compute(fmin, agc, active, cfg, receiver)
    cls = classify.classify(ev, intf, dual, s4_r, cfg)
    return {"events": ev, "dualfreq": dual, "dualfreq_note": dual_note, "interference": intf,
            "interference_info": info, "classified": cls}


def _trim_window(spans: list[tuple[pd.Timestamp, pd.Timestamp]], trim_start_min: float, trim_end_min: float):
    """[lo, hi] after dropping setup / tear-down minutes from the data span of all receivers."""
    t0, t1 = min(a for a, _ in spans), max(b for _, b in spans)
    lo, hi = t0 + pd.Timedelta(minutes=trim_start_min), t1 - pd.Timedelta(minutes=trim_end_min)
    if hi <= lo:
        raise ValueError(f"trimming {trim_start_min}+{trim_end_min} min leaves no data in {t0} .. {t1}")
    return lo, hi


def _between(df: pd.DataFrame, lo, hi, col: str = "time_utc") -> pd.DataFrame:
    return df[(df[col] >= lo) & (df[col] <= hi)].reset_index(drop=True) if len(df) and col in df else df


def run(phone: list[str] | None, l86: str | None, out: str | Path, cfg: dict, plots: bool = True,
        node_id: str | None = None, session: str | None = None, trim_start_min: float = 0,
        trim_end_min: float = 0, phone2: list[str] | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    session_id = out.name
    tz = cfg["display_tz"]
    notes: dict[str, str] = {}
    tables: dict[str, pd.DataFrame] = {}
    res: dict = {"session_id": session_id, "out": str(out), "receivers": [], "notes": notes, "tables": tables}

    obs_parts, min_parts, ev_parts, s4_parts, actives = [], [], [], [], {}

    # ------------------------------------------------------------ session note
    from ionwatch import session as sessmod
    note_path = sessmod.find_note(session, out, l86, phone)
    res["session_note"] = sessmod.read_note(note_path)
    if note_path is not None and Path(note_path).resolve() != (out / "session.json").resolve():
        (out / "session.json").write_text(Path(note_path).read_text(encoding="utf-8"), encoding="utf-8")

    # ------------------------------------------------------------ read inputs, span from the data, trim
    log = obs = l86_log = log2 = obs2 = None
    spans = []
    if phone:
        log = gnsslogger.read_phone(phone, cfg)
        join = gnsslogger.attach_elevation(log, cfg)
        obs, obs_stats = gnsslogger.cn0_obs(log, cfg)
        spans.append((obs["time_utc"].min(), obs["time_utc"].max()))
    if phone2:
        if not phone:
            raise ValueError("--phone2 needs --phone (the primary phone)")
        log2 = gnsslogger.read_phone(phone2, cfg)
        join2 = gnsslogger.attach_elevation(log2, cfg)
        obs2, obs2_stats = gnsslogger.cn0_obs(log2, cfg, receiver="phone2")
        if len(obs2):
            spans.append((obs2["time_utc"].min(), obs2["time_utc"].max()))
    if l86:
        from ionwatch.io import nmea_l86
        l86_log = nmea_l86.read_l86(l86, cfg)
        if len(l86_log.obs):
            spans.append((l86_log.obs["time_utc"].min(), l86_log.obs["time_utc"].max()))
    if (trim_start_min or trim_end_min) and spans:
        lo, hi = _trim_window(spans, trim_start_min, trim_end_min)
        res["trim"] = {"start_min": trim_start_min, "end_min": trim_end_min,
                       "from_utc": lo.isoformat(), "to_utc": hi.isoformat()}
        if log is not None:
            obs = _between(obs, lo, hi)
            log.raw, log.status, log.fix, log.agc = (_between(log.raw, lo, hi), _between(log.status, lo, hi),
                                                     _between(log.fix, lo, hi), _between(log.agc, lo, hi))
        if l86_log is not None:
            l86_log.obs, l86_log.table = _between(l86_log.obs, lo, hi), _between(l86_log.table, lo, hi)
        if log2 is not None:
            obs2 = _between(obs2, lo, hi)
            log2.raw, log2.status, log2.fix, log2.agc = (_between(log2.raw, lo, hi), _between(log2.status, lo, hi),
                                                         _between(log2.fix, lo, hi), _between(log2.agc, lo, hi))
        notes["trim"] = f"dropped the first {trim_start_min:g} and last {trim_end_min:g} min of data"

    # ------------------------------------------------------------ phone
    if phone:
        q = gnsslogger.phone_quality(log, join, cfg)
        write_json(q, out / "phone_quality.json")
        rate = cfg["phone"]["sample_rate_hz"]
        active = common.active_seconds(obs["time_utc"])
        obs_b = fades.add_baseline(obs, active, cfg, rate)
        fmin = fades.per_minute(obs_b, active, cfg, rate)
        s4_ph = s4.compute(obs_b, cfg, rate, "indicative_1Hz")
        res["receivers"].append("phone")
        res["phone"] = {"quality": q, "obs_stats": obs_stats, "files": log.files, "devices": log.devices}
        obs_parts.append(obs_b)
        min_parts.append(fmin)
        s4_parts.append(s4_ph)
        actives["phone"] = active

    # ------------------------------------------------------------ second phone (optional, may be L1-only)
    if log2 is not None:
        q2 = gnsslogger.phone_quality(log2, join2, cfg)
        write_json(q2, out / "phone2_quality.json")
        rate = cfg["phone"]["sample_rate_hz"]
        active2 = common.active_seconds(obs2["time_utc"])
        obs2_b = fades.add_baseline(obs2, active2, cfg, rate)
        fmin2 = fades.per_minute(obs2_b, active2, cfg, rate)
        s4_p2 = s4.compute(obs2_b, cfg, rate, "indicative_1Hz")
        res["receivers"].append("phone2")
        res["phone2"] = {"quality": q2, "obs_stats": obs2_stats, "files": log2.files, "devices": log2.devices}
        obs_parts.append(obs2_b)
        min_parts.append(fmin2)
        s4_parts.append(s4_p2)
        actives["phone2"] = active2
        notes["phone2"] = "second phone: fades, S4, local-cause and classes; NavIC, TEC and dual-frequency use the primary phone"

    # ------------------------------------------------------------ L86
    if l86:
        write_json(l86_log.quality, out / "l86_quality.json")
        l_rate = l86_log.quality["output_rate_hz"] or 1.0
        l_active = common.active_seconds(l86_log.obs["time_utc"])
        l_obs_b = fades.add_baseline(l86_log.obs, l_active, cfg, l_rate)
        l_min = fades.per_minute(l_obs_b, l_active, cfg, l_rate)
        l_s4 = s4.compute(l_obs_b, cfg, l86_log.quality["effective_rate_hz"] or l_rate,
                          f"nmea_{l86_log.quality['effective_rate_hz']}Hz")
        res["receivers"].append("l86")
        res["l86"] = {"quality": l86_log.quality, "validated_on_real_data": False}
        obs_parts.append(l_obs_b)
        min_parts.append(l_min)
        s4_parts.append(l_s4)
        actives["l86"] = l_active
        tables["l86_obs"] = nmea_l86.obs_table(l86_log)
    else:
        notes["l86"] = "no --l86 log given: L86 and cross-check steps skipped"

    if not obs_parts:
        raise ValueError("nothing to process: give --phone and/or --l86")

    obs_all, fmin_all, s4_all = _cat(obs_parts), _cat(min_parts), _cat(s4_parts)

    # ------------------------------------------------------------ per-receiver events + classes
    evs = {r: fades.events(fmin_all[fmin_all["receiver"] == r]) for r in res["receivers"]}
    if "phone2" in evs:
        from ionwatch.proc import twophone
        fm = {r: fmin_all[fmin_all["receiver"] == r] for r in ("phone", "phone2")}
        evs["phone"], evs["phone2"] = (twophone.flag_events(evs["phone"], evs["phone2"], fm["phone2"]),
                                       twophone.flag_events(evs["phone2"], evs["phone"], fm["phone"]))
        tables["phones_1min"] = twophone.match_minutes(fmin_all, cfg)
    agcs = {"phone": log.agc if log is not None else None, "phone2": log2.agc if log2 is not None else None}
    ev_parts, dual_parts, intf_parts, cls_parts = [], [], [], []
    for rcv in res["receivers"]:
        st = receiver_stages(rcv, obs_all[obs_all["receiver"] == rcv], fmin_all[fmin_all["receiver"] == rcv],
                             s4_all[s4_all["receiver"] == rcv], actives[rcv], agcs.get(rcv), cfg, events=evs[rcv])
        ev_parts.append(st["events"])
        dual_parts.append(st["dualfreq"])
        intf_parts.append(st["interference"])
        cls_parts.append(st["classified"])
        if rcv == "phone":
            notes["dualfreq"] = st["dualfreq_note"]
            res["interference"] = st["interference_info"]
            if not st["interference_info"]["agc_available"]:
                notes["interference"] = "no usable AGC rows: local-cause check uses frac_sats_faded only"
    ev_all, intf_all, cls_all = _cat(ev_parts), _cat(intf_parts), _cat(cls_parts)
    if "phone2" in res["receivers"]:
        res["twophone"] = twophone.summary(cls_all, tables["phones_1min"])
    tables.update(fades_1min=fmin_all, fade_events=ev_all, s4_1min=s4_all,
                  interference_1min=intf_all, events_classified=cls_all)

    # ------------------------------------------------------------ phone-only stages
    if log is not None:
        ph_obs = obs_all[obs_all["receiver"] == "phone"]
        ph_min = fmin_all[fmin_all["receiver"] == "phone"]
        ph_ev = ev_all[ev_all["receiver"] == "phone"]
        nav_min, nav_ev, nav_sum = navic.summarize(ph_obs, ph_min, ph_ev, actives["phone"], cfg)
        notes["navic"] = nav_sum["note"] or "NavIC tracked"
        tec_ep, tec_min, notes["tec_code"] = tec_code.compute(log.raw, cfg)
        tables.update(navic_1min=nav_min, navic_events=nav_ev, tec_code_1min=tec_min, tec_code_epochs=tec_ep,
                      dualfreq_1min=dual_parts[res["receivers"].index("phone")])
        write_json(nav_sum, out / "navic_summary.json")
        res["navic"] = nav_sum

    # ------------------------------------------------------------ one-minute records
    node_id = node_id or cfg["node_id"]
    records = []
    for rcv in res["receivers"]:
        t = obs_all.loc[obs_all["receiver"] == rcv, "time_utc"]
        minutes = pd.date_range(t.min().floor("min"), t.max().floor("min"), freq="1min")
        rates = ({m: common.epoch_rate(g) for m, g in t.groupby(t.dt.floor("min"))} if rcv == "l86" else None)
        records += schema.records_from_tables(minutes, rcv, node_id, fmin_all, s4_all, cls_all, intf_all, cfg,
                                              l86_rate_hz=rates)
    for r in records:
        schema.validate(r)
    records.sort(key=lambda r: (r["minute_utc"], r["receiver"]))
    schema.write_jsonl(records, out / "minutes.jsonl")
    res["minutes"] = len(records)

    # ------------------------------------------------------------ cross-check
    if log is not None and l86_log is not None:
        from ionwatch.proc import crosscheck
        xc_min, xc_ev = crosscheck.compute(obs_all, ev_all, cfg)
        tables.update(crosscheck_1min=xc_min, confirmed_events=xc_ev)
        res["crosscheck"] = {"matched_satellites": sorted(xc_min["satellite"].unique()) if len(xc_min) else [],
                             "time_offset": crosscheck.time_offset(obs_all, cfg)}
    elif l86_log is not None:
        notes["crosscheck"] = "no phone log: cross-check skipped"

    # ------------------------------------------------------------ write
    for name, df in tables.items():
        common.write_table(df, out, name, session_id, tz)
    # Per-sample C/N0 with baselines: Parquet only (large), for the dashboard and re-plotting.
    common.write_table(obs_all, out / "tables", "cn0_obs", session_id, tz, csv=False)
    res["obs"] = obs_all

    if plots:
        from ionwatch.report import plots as plotmod
        res["plots"] = plotmod.make_all(res, cfg, out / "plots")
    from ionwatch.report import summary
    summary.write(res, cfg, out)
    return res
