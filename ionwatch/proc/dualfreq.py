"""Dual-frequency fade check (phone): does a fade hit L1 and L5 of the same satellite together?

For GPS / Galileo / BeiDou satellites tracked on an L1-side signal (GPS_L1, GAL_E1,
BDS_B1C, else BDS_B1I) and on L5 (GPS_L5, GAL_E5a, BDS_B2a): per minute, the fade depth
on each band and the Pearson correlation of detrended C/N0 (C/N0 minus the 15-min
median baseline) over the seconds where both bands are valid. Ionospheric fades usually
hit both bands together, often deeper on L5; a local obstruction can do the same, so
this is one input to classify.py rather than a verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ionwatch.proc import common

COLS = ["minute_utc", "receiver", "source", "constellation", "svid", "satellite", "signal_l1", "signal_l5",
        "elev_deg_mean", "n_l1", "n_l5", "n_overlap", "cn0_mean_l1", "cn0_mean_l5",
        "fade_depth_l1_db", "fade_depth_l5_db", "both_faded", "l5_deeper", "corr_detrended"]
L1_SIDE = {"GPS": ["GPS_L1"], "Galileo": ["GAL_E1"], "BeiDou": ["BDS_B1C", "BDS_B1I"]}
L5_SIDE = {"GPS": "GPS_L5", "Galileo": "GAL_E5a", "BeiDou": "BDS_B2a"}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compute(obs_b: pd.DataFrame, minutes: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, str]:
    mask = cfg["elev_mask_deg"]
    thr = cfg["fades"]["fade_threshold_db"]
    min_ov = cfg["dualfreq"]["min_overlap_s"]
    cons = cfg["dualfreq"]["constellations"]
    v = obs_b[obs_b["constellation"].isin(cons) & obs_b["cn0_dbhz"].notna() & (obs_b["elev_deg"] >= mask)].copy()
    v["sec"] = v["time_utc"].dt.floor("s")
    v["minute_utc"] = v["sec"].dt.floor("min")
    rows = []
    for (const, sv), g in v.groupby(["constellation", "svid"]):
        l5_all = g[g["signal"] == L5_SIDE[const]]
        cands = [s for s in L1_SIDE[const] if (g["signal"] == s).any()]
        if l5_all.empty or not cands:
            continue
        for minute, gm in g.groupby("minute_utc"):
            a5 = gm[gm["signal"] == L5_SIDE[const]]
            # L1-side signal with the most seconds in common with L5 in this minute (a per-minute
            # choice, so batch and the live rolling buffer pick the same signal).
            overlap = {s: len(np.intersect1d(gm.loc[gm["signal"] == s, "sec"], a5["sec"])) for s in cands}
            present = [s for s in cands if (gm["signal"] == s).any()]
            if a5.empty and not present:
                continue
            best = max(cands, key=lambda s: (overlap[s], s in present, -cands.index(s)))
            a1 = gm[gm["signal"] == best]
            b = a1.merge(a5, on="sec", suffixes=("_1", "_5"))
            d = b.dropna(subset=["detrended_db_1", "detrended_db_5"])
            srcs = set(a1["source"]) | set(a5["source"])
            rows.append({"minute_utc": minute, "receiver": g["receiver"].iloc[0],
                         "source": srcs.pop() if len(srcs) == 1 else "mixed",
                         "constellation": const, "svid": int(sv), "satellite": common.sat_label(const, sv),
                         "signal_l1": best, "signal_l5": L5_SIDE[const],
                         "elev_deg_mean": pd.concat([a1["elev_deg"], a5["elev_deg"]]).mean(),
                         "n_l1": len(a1), "n_l5": len(a5), "n_overlap": len(b),
                         "cn0_mean_l1": a1["cn0_dbhz"].mean(), "cn0_mean_l5": a5["cn0_dbhz"].mean(),
                         "corr_detrended": _corr(d["detrended_db_1"].to_numpy(), d["detrended_db_5"].to_numpy())
                         if len(d) >= min_ov else np.nan})
    if not rows:
        return pd.DataFrame(columns=COLS), "no L5/NavIC data: no GPS/Galileo/BeiDou satellite tracked on both L1 and L5"
    out = pd.DataFrame(rows)
    fm = minutes[["minute_utc", "constellation", "svid", "signal", "fade_depth_db"]]
    for side, col in (("signal_l1", "fade_depth_l1_db"), ("signal_l5", "fade_depth_l5_db")):
        out = out.merge(fm.rename(columns={"signal": side, "fade_depth_db": col}),
                        on=["minute_utc", "constellation", "svid", side], how="left")
    out["both_faded"] = (out["fade_depth_l1_db"] >= thr) & (out["fade_depth_l5_db"] >= thr)
    out["l5_deeper"] = out["fade_depth_l5_db"] > out["fade_depth_l1_db"]
    note = (f"{out['satellite'].nunique()} dual-frequency satellites: "
            f"{', '.join(sorted(out['satellite'].unique()))}")
    return out[COLS].sort_values(["minute_utc", "satellite"]).reset_index(drop=True), note
