"""Session summary -> summary.json + summary.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ionwatch.proc import classify, common, tec_code

L86_NOTE = "L86 outputs are not yet validated on real data."
PHONE_S4_NOTE = ("Phone S4 is indicative (1 Hz, quantised C/N0): s4_raw from Raw rows and "
                 "s4_indicative from Status bursts are reported separately.")


def _r(x, nd=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def _ist(t: pd.Timestamp, tz: str) -> str:
    return f"{t.tz_convert(tz):%Y-%m-%d %H:%M} IST"


def _s4_block(s4: pd.DataFrame, tz: str) -> dict:
    out = {}
    for (rcv, lab), g in s4.groupby(["receiver", "s4_label"]):
        ok = g[g["s4_valid"].astype(bool)]
        prefix = "" if rcv in ("phone", "l86") else f"{rcv}: "
        d = {"label": prefix + {"s4_raw": "S4 from Raw rows (indicative, 1 Hz)",
                                "s4_indicative": "S4 from Status bursts (indicative, 1 Hz)"}.get(lab, f"{rcv} S4"),
             "windows": int(len(g)), "windows_valid": int(len(ok)),
             "not_valid_reasons": g.loc[~g["s4_valid"].astype(bool), "reason"].str.split(" ").str[0]
                                   .value_counts().to_dict(),
             "minutes_per_level": ok["level"].value_counts().reindex(
                 ["quiet", "weak", "moderate", "strong"], fill_value=0).to_dict()}
        if len(ok):
            m = ok.loc[ok["s4"].idxmax()]
            d["max"] = {"s4": _r(m["s4"]), "satellite": common.sat_label(m["constellation"], m["svid"]),
                        "signal": m["signal"], "minute_utc": f"{m['minute_utc']:%Y-%m-%dT%H:%MZ}",
                        "minute_ist": _ist(m["minute_utc"], tz), "elev_deg": _r(m["elev_deg_mean"], 1)}
        out[lab if rcv in ("phone", "l86") else f"{rcv}_{lab}"] = d
    return out


def build(res: dict, cfg: dict) -> dict:
    tz = cfg["display_tz"]
    t = res["tables"]
    obs = res["obs"]
    t0, t1 = obs["time_utc"].min(), obs["time_utc"].max()
    warnings: list[str] = []
    s: dict = {"session_id": res["session_id"],
               "session": {"start_utc": f"{t0:%Y-%m-%dT%H:%M:%SZ}", "end_utc": f"{t1:%Y-%m-%dT%H:%M:%SZ}",
                           "start_ist": _ist(t0, tz), "end_ist": _ist(t1, tz),
                           "duration_min": _r((t1 - t0).total_seconds() / 60, 1),
                           "date_utc": f"{t0:%Y-%m-%d}", "doy": int(t0.dayofyear)},
               "receivers": res["receivers"], "notes": res["notes"]}
    sn = res.get("session_note") or {"antenna": "unknown", "phone": "", "phone2": "", "notes": ""}
    s["session_note"] = {k: sn.get(k, "") for k in ("antenna", "phone", "phone2", "notes")}
    if "phone2" in res:
        q2 = res["phone2"]["quality"]
        raw2 = q2.get("raw", {})
        s["phone2"] = {"files": [Path(f).name for f in res["phone2"]["files"]], "devices": res["phone2"]["devices"],
                       "raw_minutes": raw2.get("duration_min"), "mean_cn0_dbhz": raw2.get("mean_cn0_dbhz"),
                       "signals": sorted(raw2.get("rows_per_signal", {})),
                       "navic": q2.get("navic", {}).get("tracked", False), "agc": bool(q2.get("agc")),
                       "warnings": [w["message"] for w in q2.get("warnings", [])]}
        s["twophone"] = res.get("twophone", {})
    if "trim" in res:
        s["session"]["trim"] = res["trim"]
    if sn.get("antenna") == "onboard":
        from ionwatch.session import ONBOARD_WARNING
        warnings.append(ONBOARD_WARNING)

    if "phone" in res:
        q, st = res["phone"]["quality"], res["phone"]["obs_stats"]
        raw = q.get("raw", {})
        act = q.get("status", {}).get("receiver_activity", {})
        s["phone"] = {
            "files": [Path(f).name for f in res["phone"]["files"]], "devices": res["phone"]["devices"],
            "raw_span_ist": f"{raw['start']['ist'][11:19]} → {raw['end']['ist'][11:19]}" if raw else None,
            "raw_minutes": raw.get("duration_min"), "raw_epochs": raw.get("epochs"),
            "raw_epoch_fraction": raw.get("epoch_fraction_1s"), "raw_meas_per_epoch": raw.get("meas_per_epoch"),
            "mean_cn0_dbhz": raw.get("mean_cn0_dbhz"), "adr_valid_rows": raw.get("adr_valid_rows"),
            "receiver_active_fraction": act.get("epoch_fraction_1s"),
            "receiver_silences": act.get("silences", []),
            "status_blocks_dropped_untimed": st.get("status_blocks_dropped_untimed", 0),
            "status_rows_no_frequency": st.get("status_rows_no_frequency", 0),
            "cn0_rows": {"raw": st.get("rows_raw"), "status": st.get("rows_status"),
                         "status_tracked": st.get("rows_status_tracked")},
            "elevation_join_match_rate": q["elevation_join"].get("match_rate"),
            "position_std_m": _r(q["position"].get("horiz_std_m"), 1)}
        warnings += [w["message"] for w in q["warnings"]]
        warnings.append(PHONE_S4_NOTE)
        s["session_lines"] = {"lines": q["session"].get("lines", []),
                              "active_min_in_night_window": q["session"].get("active_min_in_night_window")}
    if "l86" in res:
        s["l86"] = {"quality": {k: v for k, v in res["l86"]["quality"].items() if k != "warnings"},
                    "validated_on_real_data": False, "note": L86_NOTE}
        warnings += [w["message"] for w in res["l86"]["quality"].get("warnings", [])]
        warnings.append(L86_NOTE)

    s["s4"] = _s4_block(t["s4_1min"], tz)

    all_ev = t["fade_events"]
    edge = all_ev[all_ev["edge_of_gap"].astype(bool)]
    ev = all_ev[~all_ev["edge_of_gap"].astype(bool)]
    fm = t["fades_1min"]
    s["fades"] = {"events": int(len(ev)),
                  "events_by": {f"{r}/{src or '-'}/{c}": int(n) for (r, src, c), n in
                                ev.groupby(["receiver", "source", "constellation"]).size().items()} if len(ev) else {},
                  "lock_losses": int(fm["lock_losses"].sum()),
                  "receiver_off_gaps": int(fm["receiver_off_gaps"].sum()),
                  "edge_of_gap": {"events_ignored": int(len(edge)),
                                  "fade_minutes_ignored": int((fm["edge_fade_depth_db"]
                                                               >= cfg["fades"]["fade_threshold_db"]).sum()),
                                  "lock_losses_ignored": int(fm["edge_lock_losses"].sum()),
                                  "guard_s": cfg["fades"]["edge_guard_s"]},
                  "deepest": None}
    if len(ev) and ev["peak_depth_db"].notna().any():
        d = ev.loc[ev["peak_depth_db"].idxmax()]
        s["fades"]["deepest"] = {"event_id": d["event_id"], "satellite": d["satellite"], "signal": d["signal"],
                                 "depth_db": _r(d["peak_depth_db"], 1), "start_ist": _ist(d["start_utc"], tz)}

    if "events_classified" in t:
        cl = t["events_classified"]
        s["classification"] = {
            "note": classify.HEURISTIC_NOTE,
            "counts": cl["class"].value_counts().to_dict() if len(cl) else {},
            "iono_likely": [{"event_id": r.event_id, "satellite": r.satellite, "signal": r.signal,
                             "source": r.source, "start_ist": _ist(r.start_utc, tz),
                             "peak_depth_db": _r(r.peak_depth_db, 1), "reason": r.reason}
                            for r in cl[cl["class"] == "iono_likely"].itertuples()]}
        warnings.append(classify.HEURISTIC_NOTE)
    if "navic" in res:
        s["navic"] = res["navic"]
    if "tec_code_1min" in t:
        tm = t["tec_code_1min"]
        if len(tm):
            tm = tm.assign(dstec=tec_code.relative_stec(tm))
        s["tec"] = {"note": res["notes"].get("tec_code", ""), "calibration": tec_code.CALIBRATION_NOTE,
                    "per_satellite": {sat: {"minutes": int(len(g)), "arcs": int(g["arc_id"].nunique()),
                                            "dstec_min": _r(g["dstec"].min(), 1),
                                            "dstec_max": _r(g["dstec"].max(), 1),
                                            "stec_abs_min": _r(g["stec_mean"].min(), 1),
                                            "stec_abs_max": _r(g["stec_mean"].max(), 1)}
                                      for sat, g in tm.groupby("satellite")}}
    if "dualfreq_1min" in t:
        d = t["dualfreq_1min"]
        s["dualfreq"] = {"note": res["notes"].get("dualfreq", ""), "minutes": int(len(d)),
                         "satellites": sorted(d["satellite"].unique()) if len(d) else [],
                         "both_faded_minutes": int(d["both_faded"].sum()) if len(d) else 0,
                         "median_corr_detrended": _r(d["corr_detrended"].median()) if len(d) else None}
    if "interference_1min" in t:
        it = t["interference_1min"]
        info = res.get("interference", {})
        s["interference"] = {"minutes": int(len(it)), "local_cause_minutes": int(it["local_cause_flag"].sum()),
                             "by_reason": it.loc[it["reason"] != "", "reason"].value_counts().to_dict(),
                             "agc_available": info.get("agc_available"),
                             "agc_placeholder": info.get("agc_placeholder", {})}
    if "confirmed_events" in t:
        xc = res.get("crosscheck", {})
        s["crosscheck"] = {"confirmed_events": int(len(t["confirmed_events"])), "note": L86_NOTE,
                           "matched_satellites": xc.get("matched_satellites", []),
                           "time_offset": xc.get("time_offset", {})}
    s["plots"] = res.get("plots", [])
    s["warnings"] = list(dict.fromkeys(warnings))
    s["bottom_line"] = bottom_line(s, cfg)
    return s


def bottom_line(s: dict, cfg: dict) -> list[str]:
    """Three lines: usable for, not usable for, events still iono_likely after all filters."""
    floor = cfg["classify"]["iono_likely_min_cn0"]
    p = s.get("phone", {})
    q_sess = s.get("session_lines", {})
    usable, not_usable = ["pipeline testing"], []
    cn0 = p.get("mean_cn0_dbhz")
    act = p.get("receiver_active_fraction")
    night_min = q_sess.get("active_min_in_night_window")
    if s["fades"]["events"] or s["fades"]["lock_losses"]:
        usable.append("C/N0 fade and lock-loss statistics")
    tec = s.get("tec", {}).get("per_satellite", {})
    if tec:
        usable.append(f"code-only ΔSTEC trends ({len(tec)} satellites, Raw span only)")
    if s.get("dualfreq", {}).get("satellites"):
        usable.append(f"L1/L5 fade comparison ({len(s['dualfreq']['satellites'])} satellites)")
    nav = s.get("navic", {}).get("status")
    if nav == "tracked":
        usable.append("NavIC L5 tracking health")
    if cn0 is not None and cn0 < floor:
        not_usable.append(f"scintillation detection (mean C/N0 {cn0:.1f} dB-Hz < {floor} dB-Hz floor)")
    if night_min == 0:
        not_usable.append("night-time scintillation (no receiver-active data in the night window)")
    if nav == "visible_not_tracked":
        not_usable.append("NavIC health (visible, not tracked)")
    elif nav == "no_data":
        not_usable.append("NavIC health (no L5/NavIC data)")
    if "tec" in s and not tec:
        not_usable.append("TEC (no L1+L5 pseudoranges)")
    not_usable.append("absolute TEC (code biases)")
    if p.get("adr_valid_rows") == 0:
        not_usable.append("carrier-phase TEC / ROTI (no ADR)")
    if act is not None and act < cfg["quality"]["min_epoch_fraction"]:
        not_usable.append(f"continuous monitoring (receiver active {act:.0%} of the log)")
    if "l86" in s:
        not_usable.append("L86 results as validated output (not yet validated on real data)")
    counts = s.get("classification", {}).get("counts", {})
    total = sum(v for k, v in counts.items() if k != "edge_of_gap")
    return [f"Usable for: {'; '.join(usable)}.",
            f"Not usable for: {'; '.join(not_usable)}.",
            f"Events still iono_likely after all filters: {counts.get('iono_likely', 0)} of {total} "
            f"({counts.get('edge_of_gap', 0)} edge-of-gap events ignored)."]


def _md(s: dict) -> str:
    se, sn = s["session"], s["session_note"]
    L = [f"# IonWatch session {s['session_id']}", "",
         f"**Date (from data):** {se['date_utc']} UTC (DOY {se['doy']}) · **data span:** {se['start_ist']} → "
         f"{se['end_ist'][11:]} ({se['duration_min']} min)  ",
         f"**Antenna:** {sn['antenna']} · **Phone:** {sn['phone'] or 'not noted'}" +
         (f" · **Phone 2:** {sn['phone2'] or 'not noted'}" if sn.get("phone2") or "phone2" in s else "") +
         (f"  \n**Notes:** {sn['notes']}" if sn["notes"] else "") +
         (f"  \n**Trimmed:** first {se['trim']['start_min']:g} min, last {se['trim']['end_min']:g} min of data"
          if "trim" in se else ""),
         "", "## Bottom line"]
    L += [f"{i}. {x}" for i, x in enumerate(s["bottom_line"], 1)]
    L.append("")
    se = s["session"]
    L += ["## Session",
          f"- Receivers: {', '.join(s['receivers'])} · DOY {se['doy']} · UTC {se['start_utc']} → {se['end_utc']}"]
    L += [f"- {x}" for x in s.get("session_lines", {}).get("lines", [])]
    if "session_lines" not in s:
        L.append(f"- Log span: {se['start_ist']} → {se['end_ist']} ({se['duration_min']} min).")
    L.append("")
    if s.get("warnings"):
        L.append("## Warnings")
        L += [f"- {w}" for w in s["warnings"]]
        L.append("")
    if "phone" in s:
        p = s["phone"]
        L += ["## Phone data quality",
              f"- Files: {', '.join(p['files'])}",
              f"- Raw: {p['raw_span_ist']} IST, {p['raw_minutes']} min, {p['raw_epochs']} epochs "
              f"({p['raw_epoch_fraction']:.1%} of 1 s epochs), {p['raw_meas_per_epoch']} measurements/epoch, "
              f"mean C/N0 {p['mean_cn0_dbhz']} dB-Hz, ADR-valid rows {p['adr_valid_rows']}"
              if p["raw_span_ist"] else "- Raw: none",
              f"- Receiver active {p['receiver_active_fraction']:.1%} of the log; silences: " +
              (", ".join(f"{x['start']['ist'][11:16]} IST ({x['length_min']} min)" for x in p["receiver_silences"])
               or "none"),
              f"- C/N0 rows used: Raw {p['cn0_rows']['raw']}, Status {p['cn0_rows']['status']} "
              f"({p['cn0_rows']['status_tracked']} tracked)",
              f"- Status blocks dropped (time could not be placed): **{p['status_blocks_dropped_untimed']}**; "
              f"Status rows without frequency: {p['status_rows_no_frequency']}",
              f"- Elevation join match {p['elevation_join_match_rate']:.1%}; position std {p['position_std_m']} m",
              ""]
    if "l86" in s:
        lq = s["l86"]["quality"]
        L += ["## L86 (not yet validated on real data)",
              f"- Epochs {lq.get('epochs')}, output rate {lq.get('output_rate_hz')} Hz, effective C/N0 rate "
              f"{lq.get('effective_rate_hz')} Hz, bad checksums {lq.get('bad_checksums')}", ""]
    L += ["## S4 (indicative, 1 Hz for the phone)",
          "_Valid = enough samples, no step artefact, and S4 > 2× s4_noise, where "
          "s4_noise = sqrt(1 / (T·10^(C/N0/10))) with T = 1 s (one C/N0 sample)._", "",
          "| series | valid / windows | quiet | weak | moderate | strong | max valid |",
          "|---|---|---|---|---|---|---|"]
    for lab, d in s["s4"].items():
        lv = d["minutes_per_level"]
        mx = d.get("max")
        mxs = f"{mx['s4']} ({mx['satellite']} {mx['signal']}, {mx['minute_ist'][11:]}, {mx['elev_deg']}°)" if mx else "–"
        L.append(f"| {d['label']} | {d['windows_valid']} / {d['windows']} | {lv['quiet']} | {lv['weak']} | "
                 f"{lv['moderate']} | {lv['strong']} | {mxs} |")
    L.append("")
    L += [f"- Not valid, {d['label']}: {d['not_valid_reasons']}" for d in s["s4"].values()]
    L.append("")
    f = s["fades"]
    e = f["edge_of_gap"]
    L += ["## Fades and lock loss",
          f"- {f['events']} fade events; {f['lock_losses']} lock losses; {f['receiver_off_gaps']} gaps "
          "explained by the receiver being off (not counted as lock loss)",
          f"- edge_of_gap (within {e['guard_s']} s of a receiver silence, ignored): {e['events_ignored']} events, "
          f"{e['fade_minutes_ignored']} fade minutes, {e['lock_losses_ignored']} lock losses",
          "- By receiver/source/constellation: " + ", ".join(f"{k} {v}" for k, v in f["events_by"].items())]
    if f["deepest"]:
        d = f["deepest"]
        L.append(f"- Deepest: {d['satellite']} {d['signal']} {d['depth_db']} dB at {d['start_ist']}")
    L.append("")
    if "classification" in s:
        c = s["classification"]
        L += ["## Event classification (heuristic)", f"_{c['note']}_", "",
              "| class | events |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in c["counts"].items()]
        if c["iono_likely"]:
            L += ["", "iono_likely events:"]
            L += [f"- {e['event_id']} {e['satellite']} {e['signal']} ({e['source']}) {e['start_ist']}, "
                  + (f"{e['peak_depth_db']} dB" if e["peak_depth_db"] is not None else "lock loss only")
                  + f": {e['reason']}" for e in c["iono_likely"]]
        L.append("")
    if "navic" in s:
        n = s["navic"]
        L += ["## NavIC", f"- Status: **{n['status']}**" + (f" — {n['note']}" if n.get("note") else "")]
        for d in n.get("satellites", {}).values():
            extra = (f", availability {d.get('availability_pct_session')} % of session "
                     f"({d.get('availability_pct_receiver_active')} % of receiver-active minutes), "
                     f"C/N0 mean {d.get('cn0_mean')} / min {d.get('cn0_min')}, {d.get('fade_events')} fade events"
                     if "minutes_tracked" in d else "")
            L.append(f"- {d['satellite']}: elevation {d['elev_min']}–{d['elev_max']}°{extra}")
        if n.get("status") == "tracked":
            L.append(f"- Impact minutes (all NavIC faded or lost): {n.get('impact_minutes')}")
        L.append("")
    if "tec" in s:
        tc = s["tec"]
        L += ["## Code-only TEC (uncalibrated)", f"_{tc['calibration']}. Trends only; no ROTI._",
              f"- {tc['note']}"]
        L += [f"- {sat}: ΔSTEC {d['dstec_min']} … {d['dstec_max']} TECU over {d['minutes']} min "
              f"({d['arcs']} arc{'s' if d['arcs'] > 1 else ''}; absolute values in tec_code_1min.csv)"
              for sat, d in tc["per_satellite"].items()]
        L.append("")
    if "dualfreq" in s:
        d = s["dualfreq"]
        L += ["## Dual-frequency check", f"- {d['note']}",
              f"- {d['minutes']} satellite-minutes; both bands faded in {d['both_faded_minutes']}; "
              f"median correlation of detrended C/N0 {d['median_corr_detrended']}", ""]
    if "interference" in s:
        i = s["interference"]
        L += ["## Local-cause check",
              f"- {i['local_cause_minutes']} of {i['minutes']} minutes flagged ({i['by_reason']}); "
              f"AGC available: {i['agc_available']}; AGC placeholders excluded: {i['agc_placeholder'] or 'none'}", ""]
    if "twophone" in s:
        tp, p2 = s["twophone"], s["phone2"]
        L += ["## Two phones (same spot)",
              f"- Second phone: {', '.join(p2['files'])} · {', '.join(d for d in p2['devices'] if d)}",
              f"  signals {', '.join(p2['signals']) or 'none'}; mean C/N0 {p2['mean_cn0_dbhz']} dB-Hz; "
              f"NavIC {'tracked' if p2['navic'] else 'none'}; AGC {'yes' if p2['agc'] else 'no'}",
              f"- Satellite/band minutes seen by both: {tp.get('matched_minutes', 0)} "
              f"({tp.get('matched_satellite_bands', 0)} satellite-bands); both faded in {tp.get('both_faded_minutes', 0)}",
              "- A fade on the same satellite and band on both phones within 30 s counts toward iono_possible; "
              "a fade on one phone only (the other tracked it without a fade) counts toward local.",
              "", "| receiver | events | on both phones | one phone only (→ local) | not comparable |",
              "|---|---|---|---|---|"]
        for rcv in ("phone", "phone2"):
            r = tp.get(rcv, {})
            L.append(f"| {rcv} | {r.get('events', 0)} | {r.get('both_phones', 0)} | {r.get('one_phone_only', 0)} | "
                     f"{r.get('not_comparable', 0)} |")
        L.append("")
    if "crosscheck" in s:
        L += ["## L86 vs phone cross-check (not yet validated on real data)",
              f"- Matched satellites (L86 and phone, L1): {len(s['crosscheck']['matched_satellites'])} "
              f"({', '.join(s['crosscheck']['matched_satellites'])})",
              f"- L86 - phone clock offset from C/N0 cross-correlation: "
              f"{s['crosscheck']['time_offset'].get('lag_s')} s over {s['crosscheck']['time_offset'].get('n_sats')} "
              f"satellites (corr {s['crosscheck']['time_offset'].get('corr_at_lag')})",
              f"- Confirmed events: {s['crosscheck']['confirmed_events']}", ""]
    if s["notes"]:
        L += ["## Stage notes"] + [f"- **{k}**: {v}" for k, v in s["notes"].items()] + [""]
    if s.get("plots"):
        L += ["## Plots"] + [f"- [plots/{p}](plots/{p})" for p in s["plots"]] + [""]
    return "\n".join(L)


def write(res: dict, cfg: dict, out: Path) -> dict:
    from ionwatch.pipeline import write_json
    s = build(res, cfg)
    write_json(s, Path(out) / "summary.json")
    (Path(out) / "summary.md").write_text(_md(s), encoding="utf-8")
    return s
