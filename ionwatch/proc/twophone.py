"""Two phones at the same spot: does a fade show on both (sky) or on one only (local)?

The primary phone is receiver "phone", the second "phone2". Both are processed by the same
fades code; here they are compared per satellite and band (signal names such as GPS_L1 are
the same on both phones).

  match_minutes - per minute and satellite/band tracked by both: C/N0 and fade depth on each;
  flag_events   - for each fade event of one phone: both_phones = True if the other phone has a
                  (non edge-of-gap) fade event on the same satellite and band whose window
                  overlaps within `tol_s` (30 s; events are minute-resolved, so this means
                  overlapping or touching windows); False if the other phone tracked that
                  satellite and band during the event without a fade (a fade on one phone only);
                  None if the other phone did not track it (no comparison possible).
classify.py counts both_phones=True toward iono_possible/iono_likely and False toward local.
"""
from __future__ import annotations

import pandas as pd

KEYS = ["constellation", "svid", "signal"]
COLS = ["minute_utc", "constellation", "svid", "signal", "cn0_mean_phone", "cn0_mean_phone2",
        "fade_depth_phone", "fade_depth_phone2", "both_faded"]


def match_minutes(fmin: pd.DataFrame, cfg: dict, a: str = "phone", b: str = "phone2") -> pd.DataFrame:
    thr = cfg["fades"]["fade_threshold_db"]
    cols = ["minute_utc", *KEYS, "cn0_mean", "fade_depth_db"]
    x = fmin[(fmin["receiver"] == a) & (fmin["n"] > 0)][cols]
    y = fmin[(fmin["receiver"] == b) & (fmin["n"] > 0)][cols]
    j = x.merge(y, on=["minute_utc", *KEYS], suffixes=(f"_{a}", f"_{b}"))
    j = j.rename(columns={f"fade_depth_db_{a}": "fade_depth_phone", f"fade_depth_db_{b}": "fade_depth_phone2",
                          f"cn0_mean_{a}": "cn0_mean_phone", f"cn0_mean_{b}": "cn0_mean_phone2"})
    j["both_faded"] = (j["fade_depth_phone"] >= thr) & (j["fade_depth_phone2"] >= thr)
    return j[COLS].sort_values(["minute_utc", *KEYS]).reset_index(drop=True)


def flag_events(ev: pd.DataFrame, ev_other: pd.DataFrame, fmin_other: pd.DataFrame,
                tol_s: float = 30.0) -> pd.DataFrame:
    """Add both_phones (True/False/None) and other_phone_tracked (bool) to a phone's events."""
    out = ev.copy()
    tol = pd.Timedelta(seconds=tol_s)
    oth = ev_other[~ev_other["edge_of_gap"].astype(bool)] if len(ev_other) else ev_other
    tracked = fmin_other[fmin_other["n"] > 0] if len(fmin_other) else fmin_other
    both, seen = [], []
    for e in ev.to_dict("records"):
        o = oth[(oth["constellation"] == e["constellation"]) & (oth["svid"] == e["svid"])
                & (oth["signal"] == e["signal"])] if len(oth) else oth
        match = bool(len(o) and ((o["start_utc"] - tol <= e["end_utc"]) & (e["start_utc"] - tol <= o["end_utc"])).any())
        t = tracked[(tracked["constellation"] == e["constellation"]) & (tracked["svid"] == e["svid"])
                    & (tracked["signal"] == e["signal"]) & (tracked["minute_utc"] >= e["start_utc"])
                    & (tracked["minute_utc"] < e["end_utc"])] if len(tracked) else tracked
        was_tracked = bool(len(t))
        both.append(True if match else (False if was_tracked else None))
        seen.append(was_tracked)
    out["both_phones"] = pd.array(both, dtype="boolean") if len(both) else pd.array([], dtype="boolean")
    out["other_phone_tracked"] = seen
    return out


def summary(cls: pd.DataFrame, matched: pd.DataFrame) -> dict:
    """Counts for summary.md."""
    out = {"matched_minutes": int(len(matched)),
           "matched_satellite_bands": int(matched[KEYS].drop_duplicates().shape[0]) if len(matched) else 0,
           "both_faded_minutes": int(matched["both_faded"].sum()) if len(matched) else 0}
    for rcv in ("phone", "phone2"):
        c = cls[(cls["receiver"] == rcv) & (cls["class"] != "edge_of_gap")] if len(cls) else cls
        bp = c["both_phones"] if len(c) else pd.Series(dtype=object)
        out[rcv] = {"events": int(len(c)), "both_phones": int((bp == True).sum()),          # noqa: E712
                    "one_phone_only": int((bp == False).sum()),                             # noqa: E712
                    "not_comparable": int(bp.isna().sum())}
    return out
