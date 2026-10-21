"""Code-only (pseudorange) slant TEC from phone L1/L5, GPS and Galileo only.

    STEC = k (P_L5 - P_L1),   k = f1^2 f5^2 / (40.3 (f1^2 - f5^2)) / 1e16  [TECU/m]

Uncalibrated: includes receiver and satellite code biases. For trends only; no ROTI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ionwatch.io import gnsslogger
from ionwatch.proc import common

F1, F5 = 1575.42e6, 1176.45e6
K_TECU_PER_M = F1**2 * F5**2 / (40.3 * (F1**2 - F5**2)) / 1e16
R_EARTH_KM = 6371.0
CALIBRATION_NOTE = "uncalibrated: includes receiver and satellite code biases"
SIGNALS = {"GPS_L1": "L1", "GPS_L5": "L5", "GAL_E1": "L1", "GAL_E5a": "L5"}
COLS = ["minute_utc", "source", "constellation", "svid", "satellite", "arc_id", "elev_deg", "n",
        "stec_mean", "stec_std", "vtec_rel_mean", "calibration_note"]
EPOCH_COLS = ["time_utc", "constellation", "svid", "elev_deg", "p_l1_m", "p_l5_m", "stec_tecu", "vtec_rel_tecu"]


def relative_stec(tec_min: pd.DataFrame) -> pd.Series:
    """ΔSTEC per minute: each arc's 60 s means minus that arc's first 60 s mean (TECU).

    Removes the constant code biases of an arc, so only the trend is left; still uncalibrated.
    """
    first = tec_min.sort_values("minute_utc").groupby("arc_id")["stec_mean"].transform("first")
    return tec_min["stec_mean"] - first.reindex(tec_min.index)


def mapping_cos(elev_deg, shell_km: float):
    """cos(z') at the ionospheric pierce point: sin z' = R/(R+h) cos(elev)."""
    sin_z = R_EARTH_KM / (R_EARTH_KM + shell_km) * np.cos(np.radians(elev_deg))
    return np.sqrt(1 - sin_z**2)


def compute(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (per-epoch STEC, per-minute table, note). Empty tables + note if no L1/L5 pairs."""
    t = cfg["tec"]
    empty = pd.DataFrame(columns=EPOCH_COLS), pd.DataFrame(columns=COLS)
    if raw is None or not len(raw):
        return *empty, "no Raw measurements: no TEC"
    p = gnsslogger.pseudorange_m(raw)
    ok = (raw["signal"].isin(SIGNALS) & gnsslogger.tow_valid(raw)
          & (raw["ReceivedSvTimeUncertaintyNanos"] <= t["max_sv_time_unc_ns"]) & p.between(18e6, 30e6))
    d = raw.loc[ok, ["utcTimeMillis", "time_utc", "constellation", "Svid", "signal", "elev_deg"]].copy()
    d["band"] = d["signal"].map(SIGNALS)
    d["P"] = p[ok]
    if not (d["band"] == "L5").any():
        return *empty, "no L5/NavIC data: no GPS/Galileo L5/E5a pseudoranges"
    piv = d.pivot_table(index=["utcTimeMillis", "constellation", "Svid"], columns="band",
                        values=["P", "elev_deg"], aggfunc="first")
    if ("P", "L1") not in piv or ("P", "L5") not in piv:
        return *empty, "no L5/NavIC data: no satellite with both L1 and L5 pseudoranges"
    e = pd.DataFrame({"p_l1_m": piv[("P", "L1")], "p_l5_m": piv[("P", "L5")],
                      "elev_deg": piv["elev_deg"].mean(axis=1)}).dropna(subset=["p_l1_m", "p_l5_m"])
    if e.empty:
        return *empty, "no L5/NavIC data: L1 and L5 never tracked in the same epoch"
    e = e.reset_index().rename(columns={"Svid": "svid"})
    e["svid"] = e["svid"].astype(int)
    e["time_utc"] = pd.to_datetime(e["utcTimeMillis"].astype("float64"), unit="ms", utc=True)
    e["stec_tecu"] = K_TECU_PER_M * (e["p_l5_m"] - e["p_l1_m"])
    keep = e["stec_tecu"].abs() <= t["max_abs_stec_tecu"]
    if t.get("apply_elev_mask", True):
        keep &= e["elev_deg"] >= cfg["elev_mask_deg"]
    n_rej = int((~keep).sum())
    e = e[keep].copy()
    e["vtec_rel_tecu"] = e["stec_tecu"] * mapping_cos(e["elev_deg"], t["shell_height_km"])
    e = e[EPOCH_COLS]
    if e.empty:
        return e, pd.DataFrame(columns=COLS), f"all {n_rej} L1/L5 epochs rejected (|STEC| or elevation mask)"

    e["minute_utc"] = e["time_utc"].dt.floor(f"{t['avg_window_s']}s")
    m = (e.groupby(["minute_utc", "constellation", "svid"])
          .agg(elev_deg=("elev_deg", "mean"), n=("stec_tecu", "size"), stec_mean=("stec_tecu", "mean"),
               stec_std=("stec_tecu", "std"), vtec_rel_mean=("vtec_rel_tecu", "mean"))
          .reset_index())
    m["source"] = "raw"
    m["satellite"] = [common.sat_label(c, s) for c, s in zip(m["constellation"], m["svid"])]
    m = m.sort_values(["satellite", "minute_utc"])
    new_arc = (m["satellite"] != m["satellite"].shift()) | \
        (m["minute_utc"].diff() > pd.Timedelta(seconds=t["arc_gap_s"]))
    m["arc_id"] = m["satellite"] + "-" + new_arc.groupby(m["satellite"]).cumsum().astype(str)
    m["calibration_note"] = CALIBRATION_NOTE
    note = f"{CALIBRATION_NOTE}; {n_rej} epochs rejected" if n_rej else CALIBRATION_NOTE
    return e.drop(columns="minute_utc"), m[COLS], note
