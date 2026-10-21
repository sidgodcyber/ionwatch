"""NavIC health: the NavIC subset of the fade tables plus a summary.

Three outcomes, none of which fail the run:
  * "no_data"              - no NavIC satellite in Raw or Status ("no L5/NavIC data");
  * "visible_not_tracked"  - NavIC listed in Status (almanac/ephemeris) but C/N0 = 0
                             throughout; reports the satellites' elevations instead;
  * "tracked"              - availability, C/N0 stats, fade events, impact minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ionwatch.proc import common, fades

NAVIC = "NavIC"


def _r(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def summarize(obs_b: pd.DataFrame, minutes: pd.DataFrame, events: pd.DataFrame,
              active: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mask = cfg["elev_mask_deg"]
    thr = cfg["fades"]["fade_threshold_db"]
    nav = obs_b[obs_b["constellation"] == NAVIC]
    nav_min = minutes[minutes["constellation"] == NAVIC].reset_index(drop=True)
    nav_ev = events[events["constellation"] == NAVIC].reset_index(drop=True)
    if nav_min.empty:
        nav_min = pd.DataFrame(columns=fades.MINUTE_COLS)
    if nav_ev.empty:
        nav_ev = pd.DataFrame(columns=fades.EVENT_COLS)

    if nav.empty:
        return nav_min, nav_ev, {"status": "no_data", "note": "no L5/NavIC data in this log",
                                 "satellites": {}}

    active_min = np.unique(active // 60) if len(active) else np.array([])
    t_all = obs_b["time_utc"]
    session_min = int((t_all.max().floor("min") - t_all.min().floor("min")) / pd.Timedelta(minutes=1)) + 1
    tracked = nav["cn0_dbhz"].notna().any()
    sats = {}
    for sv, g in nav.groupby("svid"):
        el = g["elev_deg"]
        mins = g["time_utc"].dt.floor("min")
        d = {"satellite": common.sat_label(NAVIC, sv), "signals": sorted(g["signal"].unique()),
             "elev_min": _r(el.min()), "elev_max": _r(el.max()), "elev_mean": _r(el.mean()),
             "az_mean": _r(common.circmean_deg(g["az_deg"])),
             "minutes_listed": int(mins.nunique()),
             "minutes_above_mask": int(mins[el >= mask].nunique())}
        if tracked:
            v = g[g["cn0_dbhz"].notna() & (el >= mask)]
            tm = v["time_utc"].dt.floor("min").nunique()
            d.update(minutes_tracked=int(tm),
                     availability_pct_session=_r(100 * tm / session_min),
                     availability_pct_receiver_active=_r(100 * tm / len(active_min)) if len(active_min) else None,
                     cn0_mean=_r(v["cn0_dbhz"].mean()), cn0_min=_r(v["cn0_dbhz"].min()),
                     fade_events=int(((nav_ev["svid"] == sv) & ~nav_ev["edge_of_gap"].astype(bool)).sum()),
                     lock_losses=int(nav_min.loc[nav_min["svid"] == sv, "lock_losses"].sum()))
        sats[int(sv)] = d

    if not tracked:
        # Nothing was measured: the tables stay header-only; elevations go in the summary.
        listed = ", ".join(f"{d['satellite']} ({d['elev_mean']}°)" for d in sats.values())
        return nav_min.iloc[0:0], nav_ev.iloc[0:0], {
            "status": "visible_not_tracked",
            "note": f"NavIC visible, not tracked (C/N0 = 0 throughout): {listed}. NavIC health metrics unavailable.",
            "satellites": sats, "session_minutes": session_min}

    # Impact: receiver-active minutes where every NavIC satellite above the mask was faded or lost.
    above = nav[nav["elev_deg"] >= mask].assign(minute_utc=lambda x: x["time_utc"].dt.floor("min"))
    listed = above.groupby("minute_utc")["svid"].unique()
    nm = nav_min.set_index(["minute_utc", "svid"])
    active_set = set(active_min.tolist())
    impact = []
    for minute, svs in listed.items():
        if int(minute.timestamp()) // 60 not in active_set:
            continue
        bad = 0
        for sv in svs:
            rows = nm.loc[(minute, sv)] if (minute, sv) in nm.index else None
            if rows is None:
                bad += 1
                continue
            rows = rows if isinstance(rows, pd.DataFrame) else rows.to_frame().T
            if (rows["n"].sum() == 0 or (rows["fade_depth_db"] >= thr).any() or (rows["lock_losses"] >= 1).any()):
                bad += 1
        if bad == len(svs):
            impact.append(minute)
    return nav_min, nav_ev, {
        "status": "tracked", "note": "",
        "satellites": sats, "session_minutes": session_min,
        "receiver_active_minutes": int(len(active_min)),
        "impact_minutes": len(impact),
        "impact_minute_list_utc": [f"{m:%Y-%m-%dT%H:%MZ}" for m in impact]}
