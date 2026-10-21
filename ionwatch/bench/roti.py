"""ROTI (rate of TEC index) from an IGS station's GPS dual-frequency carrier phase.

    STEC  = (L1·λ1 − L2·λ2) · f1² f2² / (40.3 (f1² − f2²)) / 1e16      [TECU, + unknown arc constant]
    ROT   = ΔSTEC / Δt                                                   [TECU/min, within an arc]
    ROTI  = std(ROT) over 5-min windows                                  [TECU/min]

Arcs break at data gaps (> 1.5 × interval), at loss-of-lock flags (LLI bit 0) on either phase,
and at |ΔSTEC| > slip_tecu between epochs (cycle slip). Satellite elevation from the GPS
broadcast ephemeris; windows below elev_mask_deg are dropped. RINEX epochs are GPS time;
they are shifted by the GPS−UTC leap seconds to UTC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ionwatch.bench import rinex
from ionwatch.proc import common

F1, F2 = 1575.42e6, 1227.60e6
C = 299_792_458.0
K_TECU_PER_M = F1 ** 2 * F2 ** 2 / (40.3 * (F1 ** 2 - F2 ** 2)) / 1e16
L2_TYPES = ("L2W", "L2X", "L2L")


def ecef_to_geodetic(xyz) -> tuple[float, float, float]:
    x, y, z = xyz
    a, e2 = common.WGS84_A, common.WGS84_E2
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(6):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + h)))
    return float(np.degrees(lat)), float(np.degrees(lon)), float(h)


def stec_arcs(obs: pd.DataFrame, interval_s: float, slip_tecu: float) -> pd.DataFrame:
    """Per-epoch relative STEC with an arc id per satellite."""
    l2 = next((t for t in L2_TYPES if t in obs and obs[t].notna().any()), None)
    if l2 is None or "L1C" not in obs:
        return pd.DataFrame(columns=["time_utc", "sat", "stec", "arc"])
    o = obs.dropna(subset=["L1C", l2]).copy()
    o["time_utc"] = o["time_gps"] - pd.Timedelta(seconds=rinex.GPS_UTC_LEAP_S)
    o["stec"] = (o["L1C"] * C / F1 - o[l2] * C / F2) * K_TECU_PER_M
    o = o.sort_values(["sat", "time_utc"])
    dt = o.groupby("sat")["time_utc"].diff().dt.total_seconds()
    ds = o.groupby("sat")["stec"].diff().abs()
    lli = ((o["L1C_lli"].fillna(0).astype(int) & 1) | (o[f"{l2}_lli"].fillna(0).astype(int) & 1)).astype(bool)
    brk = dt.isna() | (dt > 1.5 * interval_s) | lli | (ds > slip_tecu)
    o["arc"] = brk.groupby(o["sat"]).cumsum()
    return o[["time_utc", "sat", "stec", "arc"]].reset_index(drop=True)


def elevations(sats_times: pd.DataFrame, nav: pd.DataFrame, site: tuple[float, float, float]) -> np.ndarray:
    """Elevation (deg) for rows (sat, time_utc) from the nearest-toe broadcast ephemeris."""
    out = np.full(len(sats_times), np.nan)
    if nav.empty:
        return out
    nav = nav[nav["health"] == 0] if "health" in nav else nav
    for sat, g in sats_times.groupby("sat"):
        e = nav[nav["sat"] == sat]
        if e.empty:
            continue
        tg = g["time_utc"] + pd.Timedelta(seconds=rinex.GPS_UTC_LEAP_S)
        sow = rinex.gps_sow(tg)
        toe = e["toe"].to_numpy()
        dsec = np.abs(((sow[:, None] - toe[None, :]) + 302400) % 604800 - 302400)
        best = dsec.argmin(axis=1)
        xyz = np.zeros((len(g), 3))
        for k in np.unique(best):
            sel = best == k
            xyz[sel] = rinex.gps_position(e.iloc[k], sow[sel])
        _, el = common.ecef_to_azel(xyz, *site)
        out[np.flatnonzero(sats_times.index.isin(g.index))] = el
    return out


def compute(obs: pd.DataFrame, header: dict, nav: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """ROTI per satellite and 5-min window: time_utc (window start), sat, roti, n_rot, elev_deg."""
    b = cfg["bench"]
    interval = header.get("interval_s", 30.0)
    st = stec_arcs(obs, interval, b["slip_tecu"])
    if st.empty:
        return pd.DataFrame(columns=["time_utc", "sat", "roti", "n_rot", "elev_deg"])
    st["rot"] = st.groupby(["sat", "arc"])["stec"].diff() / (
        st.groupby(["sat", "arc"])["time_utc"].diff().dt.total_seconds() / 60)
    st = st.dropna(subset=["rot"])
    win = f"{b['roti_window_min']}min"
    st["win"] = st["time_utc"].dt.floor(win)
    g = st.groupby(["sat", "win"])["rot"]
    r = pd.DataFrame({"roti": g.std(ddof=0), "n_rot": g.size()}).reset_index()
    need = int(b["roti_window_min"] * 60 / interval * b["min_rot_fraction"])
    r = r[r["n_rot"] >= need].rename(columns={"win": "time_utc"}).reset_index(drop=True)
    site = ecef_to_geodetic(header["xyz"])
    centers = r.assign(time_utc=r["time_utc"] + pd.Timedelta(minutes=b["roti_window_min"] / 2))
    r["elev_deg"] = elevations(centers[["sat", "time_utc"]], nav, site)
    return r[r["elev_deg"] >= b["elev_mask_deg"]].reset_index(drop=True)


def station_series(r: pd.DataFrame) -> pd.DataFrame:
    """Per 5-min window: max and median ROTI over satellites, satellites used."""
    g = r.groupby("time_utc")["roti"]
    return pd.DataFrame({"roti_max": g.max(), "roti_median": g.median(), "n_sats": g.size()})
