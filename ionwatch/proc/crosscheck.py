"""L86 vs phone agreement on L1 for satellites tracked by both receivers.

GPS svid, GLONASS slot (phone GLONASS svids 93-106 are FCN+100 with unknown slot and
cannot be matched) and Galileo svid. L86 C/N0 is averaged to 1 Hz and aligned with the
phone's L1 C/N0 on UTC seconds; per minute, Pearson correlation of detrended C/N0
(C/N0 minus the 15-min baseline) and the overlap length. A confirmed event is a fade
event on one receiver overlapping (± event_overlap_s) a fade event on the same satellite
on the other. Not yet validated on real L86 data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ionwatch.proc import common

PHONE_L1 = {"GPS": "GPS_L1", "GLONASS": "GLO_G1", "Galileo": "GAL_E1"}
MIN_COLS = ["minute_utc", "constellation", "svid", "satellite", "n_overlap", "cn0_mean_l86", "cn0_mean_phone",
            "corr_detrended", "phone_source"]
CONF_COLS = ["satellite", "constellation", "svid", "start_utc", "end_utc", "l86_event_id", "phone_event_id",
             "l86_peak_db", "phone_peak_db", "corr"]


def _per_second(o: pd.DataFrame) -> pd.DataFrame:
    o = o.assign(sec=o["time_utc"].dt.floor("s"))
    return (o.groupby(["constellation", "svid", "sec"])
             .agg(cn0=("cn0_dbhz", "mean"), det=("detrended_db", "mean"), source=("source", "first"))
             .reset_index())


def time_offset(obs_b: pd.DataFrame, cfg: dict, max_lag_s: int = 10) -> dict:
    """Clock offset between the L86 and phone C/N0 streams, from cross-correlation.

    For each satellite on both receivers, the 1 Hz C/N0 (minus its mean) is correlated at lags
    -max_lag_s..+max_lag_s; the lag with the highest mean correlation over satellites is the
    offset (positive = the L86 stream is late). 0 s means the two clocks agree to 1 s.
    """
    mask = cfg["elev_mask_deg"]
    v = obs_b[obs_b["cn0_dbhz"].notna() & (obs_b["elev_deg"] >= mask) & obs_b["constellation"].isin(PHONE_L1)]
    l86 = _per_second(v[(v["receiver"] == "l86") & (v["signal"] == "L1")])
    ph = _per_second(v[(v["receiver"] == "phone") & (v["signal"] == v["constellation"].map(PHONE_L1))])
    if l86.empty or ph.empty:
        return {"lag_s": None, "n_sats": 0, "corr_at_lag": None, "corr_at_zero": None}
    lags = range(-max_lag_s, max_lag_s + 1)
    scores = {k: [] for k in lags}
    sats = 0
    for (const, sv), a in l86.groupby(["constellation", "svid"]):
        b = ph[(ph["constellation"] == const) & (ph["svid"] == sv)]
        if len(b) < 60:
            continue
        sa = a.set_index("sec")["cn0"]
        sb = b.set_index("sec")["cn0"]
        sats += 1
        for k in lags:
            x = sa.copy()
            x.index = x.index - pd.Timedelta(seconds=k)          # undo a k-second L86 delay
            j = pd.concat([x, sb], axis=1, join="inner").dropna()
            if len(j) >= 30 and j.iloc[:, 0].std() > 0 and j.iloc[:, 1].std() > 0:
                scores[k].append(float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]))
    mean = {k: float(np.mean(v)) for k, v in scores.items() if v}
    if not mean:
        return {"lag_s": None, "n_sats": sats, "corr_at_lag": None, "corr_at_zero": None}
    best = max(mean, key=mean.get)
    return {"lag_s": int(best), "n_sats": sats, "corr_at_lag": round(mean[best], 3),
            "corr_at_zero": round(mean.get(0, np.nan), 3)}


def compute(obs_b: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = cfg["crosscheck"]
    mask = cfg["elev_mask_deg"]
    v = obs_b[obs_b["cn0_dbhz"].notna() & (obs_b["elev_deg"] >= mask) & obs_b["constellation"].isin(PHONE_L1)]
    l86 = v[(v["receiver"] == "l86") & (v["signal"] == "L1")]
    ph = v[(v["receiver"] == "phone") & (v["signal"] == v["constellation"].map(PHONE_L1))]
    if l86.empty or ph.empty:
        return pd.DataFrame(columns=MIN_COLS), pd.DataFrame(columns=CONF_COLS)
    j = _per_second(l86).merge(_per_second(ph), on=["constellation", "svid", "sec"], suffixes=("_l86", "_ph"))
    j["minute_utc"] = j["sec"].dt.floor(f"{c['window_s']}s")
    rows = []
    for (const, sv, minute), g in j.groupby(["constellation", "svid", "minute_utc"]):
        d = g.dropna(subset=["det_l86", "det_ph"])
        ok = len(d) >= c["min_overlap_s"] and d["det_l86"].std() > 0 and d["det_ph"].std() > 0
        rows.append({"minute_utc": minute, "constellation": const, "svid": int(sv),
                     "satellite": common.sat_label(const, sv), "n_overlap": int(len(g)),
                     "cn0_mean_l86": g["cn0_l86"].mean(), "cn0_mean_phone": g["cn0_ph"].mean(),
                     "corr_detrended": float(np.corrcoef(d["det_l86"], d["det_ph"])[0, 1]) if ok else np.nan,
                     "phone_source": g["source_ph"].iloc[0] if g["source_ph"].nunique() == 1 else "mixed"})
    xc = pd.DataFrame(rows, columns=MIN_COLS)

    tol = pd.Timedelta(seconds=c["event_overlap_s"])
    el = events[(events["receiver"] == "l86") & (events["signal"] == "L1")]
    ep = events[(events["receiver"] == "phone") & (events["signal"] == events["constellation"].map(PHONE_L1))]
    conf = []
    for a in el.itertuples():
        for b in ep[(ep["constellation"] == a.constellation) & (ep["svid"] == a.svid)].itertuples():
            if a.start_utc - tol <= b.end_utc and b.start_utc - tol <= a.end_utc:
                s, e = min(a.start_utc, b.start_utc), max(a.end_utc, b.end_utc)
                m = xc[(xc["constellation"] == a.constellation) & (xc["svid"] == a.svid)
                       & (xc["minute_utc"] >= s) & (xc["minute_utc"] < e)]
                conf.append({"satellite": a.satellite, "constellation": a.constellation, "svid": a.svid,
                             "start_utc": s, "end_utc": e, "l86_event_id": a.event_id, "phone_event_id": b.event_id,
                             "l86_peak_db": a.peak_depth_db, "phone_peak_db": b.peak_depth_db,
                             "corr": m["corr_detrended"].mean()})
    return xc, pd.DataFrame(conf, columns=CONF_COLS)
