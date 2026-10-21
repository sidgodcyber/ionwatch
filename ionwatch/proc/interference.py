"""Local-cause check: is a fade minute explained by something at the receiver?

Per receiver-active minute:
  * frac_sats_faded - share of tracked satellites (valid C/N0 above the mask) with a fade
    >= fade_threshold_db on any signal. Many satellites fading at once points to the
    receiver (body/building blockage, phone moved), not the ionosphere. Needs >= min_sats.
  * AGC change      - AgcDb minus its 15-min running median, per band (Agc rows). A jump
    means RF interference or a front-end change. AGC values at or below
    agc_placeholder_below_db (e.g. NavIC at -135 dB) are placeholders and are excluded.
local_cause_flag = frac_sats_faded >= frac threshold OR |AGC change| >= agc_change_db.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ionwatch.io.gnsslogger import agc_label
from ionwatch.proc import common

COLS = ["minute_utc", "receiver", "source", "receiver_active_s", "n_sats_tracked", "n_sats_faded",
        "frac_sats_faded", "agc_change_max_db", "agc_signal", "agc_n_signals", "local_cause_flag", "reason"]


def agc_changes(agc: pd.DataFrame, cfg: dict, active: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """Per (minute, signal) AGC change vs running median; plus placeholder counts.

    AGC samples within agc_warmup_s of a receiver (re)start are skipped: the front end is
    still settling (seen as ~8 dB swings on L5 in the first minute of a session).
    """
    c = cfg["interference"]
    if agc is None or not len(agc):
        return pd.DataFrame(columns=["minute_utc", "agc_signal", "agc_change_db"]), {}
    a = agc.assign(agc_signal=agc_label(agc))
    ph = a["AgcDb"] <= c["agc_placeholder_below_db"]
    placeholders = a[ph].groupby("agc_signal").size().to_dict()
    a = a[~ph & a["AgcDb"].notna()]
    if active is not None and len(active) and c.get("agc_warmup_s", 0):
        starts, _ = common.receiver_segments(active, cfg["fades"]["lock_loss_min_gap_s"])
        t = a["time_utc"].astype("int64").to_numpy() / 1e9
        i = np.searchsorted(starts, t, side="right") - 1
        since_start = t - starts[np.clip(i, 0, None)]
        a = a[(i < 0) | (since_start >= c["agc_warmup_s"])]
    out = []
    for sig, g in a.groupby("agc_signal"):
        g = g.sort_values("time_utc")
        s = pd.Series(g["AgcDb"].to_numpy(), index=pd.DatetimeIndex(g["time_utc"]))
        med = s.rolling(f"{int(c['agc_baseline_window_min'] * 60)}s", center=True, min_periods=1).median()
        ch = (s - med).to_frame("agc_change_db")
        ch["minute_utc"] = ch.index.floor("min")
        # Keep the signed change with the largest magnitude in each minute.
        idx = ch["agc_change_db"].abs().groupby(ch["minute_utc"]).idxmax()
        m = ch.loc[idx].reset_index(drop=True)
        m["agc_signal"] = sig
        out.append(m)
    return (pd.concat(out, ignore_index=True) if out else
            pd.DataFrame(columns=["minute_utc", "agc_signal", "agc_change_db"])), placeholders


def compute(minutes: pd.DataFrame, agc: pd.DataFrame | None, active: np.ndarray, cfg: dict,
            receiver: str = "phone") -> tuple[pd.DataFrame, dict]:
    c = cfg["interference"]
    thr = cfg["fades"]["fade_threshold_db"]
    fm = minutes[minutes["receiver"] == receiver]
    trk = fm[fm["n"] > 0].assign(faded=lambda x: x["fade_depth_db"] >= thr)
    per_sat = trk.groupby(["minute_utc", "constellation", "svid"]).agg(
        faded=("faded", "any"), source=("source", lambda s: s.iloc[0] if s.nunique() == 1 else "mixed"))
    g = per_sat.groupby("minute_utc")
    m = pd.DataFrame({"n_sats_tracked": g.size(), "n_sats_faded": g["faded"].sum().astype(int),
                      "source": g["source"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")})
    act_min = pd.to_datetime(np.unique(active // 60) * 60, unit="s", utc=True) if len(active) else []
    m = m.reindex(m.index.union(pd.DatetimeIndex(act_min, name="minute_utc")))
    m.index.name = "minute_utc"
    m["n_sats_tracked"] = m["n_sats_tracked"].fillna(0).astype(int)
    m["n_sats_faded"] = m["n_sats_faded"].fillna(0).astype(int)
    m["source"] = m["source"].fillna("")
    m["frac_sats_faded"] = (m["n_sats_faded"] / m["n_sats_tracked"]).where(m["n_sats_tracked"] > 0)

    ch, placeholders = agc_changes(agc, cfg, active)
    if len(ch):
        idx = ch["agc_change_db"].abs().groupby(ch["minute_utc"]).idxmax()
        best = ch.loc[idx].set_index("minute_utc")
        m = m.join(best[["agc_change_db", "agc_signal"]].rename(columns={"agc_change_db": "agc_change_max_db"}))
        m = m.join(ch.groupby("minute_utc")["agc_signal"].nunique().rename("agc_n_signals"))
    else:
        m["agc_change_max_db"], m["agc_signal"], m["agc_n_signals"] = np.nan, None, 0
    m["agc_n_signals"] = m["agc_n_signals"].fillna(0).astype(int)

    m = m.reset_index()
    secs = (m["minute_utc"].astype("int64") // 10**9).to_numpy()
    m["receiver_active_s"] = (np.searchsorted(active, secs + 59, side="right")
                              - np.searchsorted(active, secs, side="left")) if len(active) else 0
    frac_hit = (m["n_sats_tracked"] >= c["min_sats"]) & (m["frac_sats_faded"] >= c["frac_sats_faded"])
    agc_hit = m["agc_change_max_db"].abs() >= c["agc_change_db"]
    m["local_cause_flag"] = frac_hit | agc_hit
    m["reason"] = np.select([frac_hit & agc_hit, frac_hit, agc_hit], ["frac_sats_faded+agc", "frac_sats_faded", "agc"], "")
    m["receiver"] = receiver
    info = {"agc_placeholder": placeholders,
            "agc_available": bool(len(ch)),
            "flagged_minutes": int(m["local_cause_flag"].sum()),
            "minutes": int(len(m))}
    return m[COLS], info
