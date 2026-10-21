"""Amplitude scintillation index S4 from C/N0, per satellite/signal/source in 60 s windows.

    I    = 10 ** (cn0 / 10)
    I_d  = I / running_mean(I, detrend_window_s, centred)
    S4   = sqrt((<I_d^2> - <I_d>^2) / <I_d>^2)

Windows are non-overlapping and aligned to the UTC minute. Raw and Status samples are
kept apart (``source``) so Status-only bursts never mix into a Raw window. A window
needs ``min_samples`` distinct samples at the effective rate (45 of 60 at 1 Hz);
otherwise S4 is NaN and ``reason`` says why.

Thermal-noise floor (reported as ``s4_noise``):
    s4_noise = sqrt(1 / (T * 10**(CN0/10)))
with CN0 the window's mean C/N0 (dB-Hz) and T the averaging time of one C/N0 sample
(``noise_T_s``, default 1 / effective rate, i.e. 1 s for the phone). This is the leading
term of the usual S4 noise expression (1/SNR in a 1/T bandwidth). S4 counts as valid
(``s4_valid``) only if S4 > ``noise_factor`` x s4_noise; otherwise level = "below_noise".

Step artefact: if a window's C/N0 is flat apart from a jump of >= step_db (see
is_step_artifact), the variance comes from the receiver's C/N0 estimator (e.g. a smoothed
C/N0 that re-locks, seen on an L1-only test phone), not from scintillation: S4 = NaN,
reason "step_artifact".

Phone C/N0 is 1 Hz and quantised, so every phone row carries method="indicative_1Hz".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ionwatch.proc import common

KEYS = common.series_keys()
COLS = ["minute_utc", "receiver", "source", "constellation", "svid", "signal", "elev_deg_mean",
        "az_deg_mean", "n", "eff_rate_hz", "cn0_mean", "s4", "s4_noise", "s4_valid", "level", "reason",
        "method", "s4_label", "indicative_1hz"]


def level(s4: pd.Series, cfg: dict) -> pd.Series:
    c = cfg["s4"]
    lab = np.select([s4 < c["quiet_below"], s4 < c["moderate"], s4 < c["strong"], s4 >= c["strong"]],
                    ["quiet", "weak", "moderate", "strong"], default="")
    return pd.Series(lab, index=s4.index)


def s4_label(receiver: str, source: str) -> str:
    if receiver == "phone":
        return "s4_indicative" if source == "status" else "s4_raw"
    return "s4"


def s4_noise(cn0_db, t_s: float):
    """Thermal-noise S4 for a mean C/N0 (dB-Hz) and C/N0 averaging time t_s (s)."""
    return np.sqrt(1.0 / (t_s * 10 ** (np.asarray(cn0_db, dtype=float) / 10)))


def _jumps(v: np.ndarray, step_db: float, span: int) -> list[tuple[int, int]]:
    """Non-overlapping jumps of >= step_db completed within <= span samples."""
    out, i = [], 0
    while i < len(v) - 1:
        k = next((k for k in range(1, span + 1) if i + k < len(v) and abs(v[i + k] - v[i]) >= step_db), 0)
        if k:
            out.append((i, i + k))
            i += k
        else:
            i += 1
    return out


def is_step_artifact(v: np.ndarray, c: dict) -> bool:
    """1..step_max_jumps jumps >= step_db (each within step_span_samples samples), and flat
    elsewhere: sample-to-sample jitter std(diff)/sqrt(2), jumps excluded, < step_flat_std_db.

    Jitter rather than level std, because a smoothed phone C/N0 drifts slowly after a re-lock
    (e.g. 28.7 -> 23.3 dB over 12 s) without any scintillation-like fluctuation; and up to two
    2-sample jumps, because a re-lock often shows as dip + jump or a rise over two epochs.
    step_span_samples = step_max_jumps = 1 gives the strict single-jump rule.
    """
    if len(v) < 4:
        return False
    jumps = _jumps(v, c["step_db"], c["step_span_samples"])
    if not 1 <= len(jumps) <= c["step_max_jumps"]:
        return False
    d = np.diff(v)
    keep = np.ones(len(d), dtype=bool)
    for a, b in jumps:
        keep[a:b] = False
    return keep.sum() >= 2 and np.std(d[keep]) / np.sqrt(2) < c["step_flat_std_db"]


def compute(obs: pd.DataFrame, cfg: dict, eff_rate_hz: float, method: str) -> pd.DataFrame:
    """Per-minute S4 for every series/source that has valid C/N0 in the minute."""
    c = cfg["s4"]
    mask = cfg["elev_mask_deg"]
    o = obs[obs["cn0_dbhz"].notna()].reset_index(drop=True)
    if not len(o):
        return pd.DataFrame(columns=COLS)
    o = common.add_minute(o)
    o["above"] = o["elev_deg"] >= mask
    o["Id"] = np.nan
    for _, g in o[o["above"]].groupby([*KEYS, "source"], sort=False):
        g = g.sort_values("time_utc")
        inten = pd.Series(10 ** (g["cn0_dbhz"].to_numpy() / 10), index=pd.DatetimeIndex(g["time_utc"]))
        mean = inten.rolling(f"{c['detrend_window_s']}s", center=True, min_periods=1).mean()
        o.loc[g.index, "Id"] = (inten / mean).to_numpy()
    o["Id2"] = o["Id"] ** 2
    o["bin"] = np.floor(o["time_utc"].astype("int64") / 1e9 * eff_rate_hz).where(o["above"])
    o["cn0_above"] = o["cn0_dbhz"].where(o["above"])

    gb = o.groupby([*KEYS, "source", "minute_utc"], sort=True)
    m = gb.agg(elev_deg_mean=("elev_deg", "mean"), n=("bin", "nunique"), cn0_mean=("cn0_above", "mean"),
               m1=("Id", "mean"), m2=("Id2", "mean"))
    m["az_deg_mean"] = gb["az_deg"].agg(common.circmean_deg)
    steps = (o[o["above"]].sort_values("time_utc")
             .groupby([*KEYS, "source", "minute_utc"])["cn0_dbhz"]
             .agg(lambda v: is_step_artifact(v.to_numpy(), c)))
    m["step"] = steps.reindex(m.index).astype("boolean").fillna(False).astype(bool)
    m = m.reset_index()

    min_n = int(np.ceil(c["min_samples"] / c["window_s"] * c["window_s"] * eff_rate_hz))
    var = (m["m2"] - m["m1"] ** 2).clip(lower=0)
    s4 = np.sqrt(var / m["m1"] ** 2)
    enough = m["n"] >= min_n
    ok = enough & ~m["step"]
    m["s4"] = s4.where(ok)
    t_s = c.get("noise_T_s") or 1.0 / eff_rate_hz
    m["s4_noise"] = s4_noise(m["cn0_mean"], t_s)
    m["s4_valid"] = ok & (m["s4"] > c["noise_factor"] * m["s4_noise"])
    m["reason"] = np.select(
        [m["n"] == 0, ~enough, m["step"], ~m["s4_valid"]],
        ["below_elev_mask", "insufficient_samples (" + m["n"].astype(str) + f"<{min_n})", "step_artifact",
         f"below_noise (<= {c['noise_factor']}x s4_noise)"], "ok")
    m["level"] = level(m["s4"], cfg).where(m["s4_valid"], np.where(ok, "below_noise", ""))
    m["eff_rate_hz"] = eff_rate_hz
    m["method"] = method
    m["s4_label"] = [s4_label(r, s) for r, s in zip(m["receiver"], m["source"])]
    m["indicative_1hz"] = m["receiver"] == "phone"
    return m[COLS]
