"""C/N0 baseline, per-minute fade depth, lock loss and fade events (any receiver).

Works on the tidy C/N0 table (common.OBS_COLS). Only samples with a finite C/N0 at
elevation >= mask count as "valid".

Baseline: centred running median of valid C/N0 over ``baseline_window_min``. It needs at
least max(baseline_min_samples, baseline_min_frac x receiver-active seconds in the window)
samples, so receiver-off time (duty cycling) is treated as missing data rather than as a
lack of coverage.

Gaps in a satellite's valid samples are split by receiver activity:
  * lock loss    - the satellite was missing for >= lock_loss_min_gap_s while the receiver
                   kept running (same continuous receiver segment) and its elevation was
                   still >= mask;
  * receiver_off - the gap is explained by the receiver switching off (or the satellite
                   being reacquired after the receiver restarted). Never a lock loss.

Edge-of-gap guard: within ``edge_guard_s`` before a receiver silence (> silence_min_s) or
after the receiver resumes, a duty-cycling phone shuts down / reacquires and C/N0 is not
trustworthy. Fades and lock losses there are kept apart (``edge_*`` columns) and become
``edge_of_gap`` events, which are reported but never classified as real events.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ionwatch.proc import common

KEYS = common.series_keys()
MINUTE_COLS = ["minute_utc", "receiver", "source", "constellation", "svid", "signal",
               "elev_deg_mean", "az_deg_mean", "n_rows", "n", "cn0_mean", "cn0_min", "baseline_db",
               "fade_depth_db", "lock_losses", "receiver_off_gaps", "edge_of_gap", "edge_fade_depth_db",
               "edge_lock_losses", "receiver_active_s", "event", "edge_event"]
EVENT_COLS = ["event_id", "receiver", "source", "constellation", "svid", "satellite", "signal",
              "start_utc", "end_utc", "n_minutes", "peak_utc", "peak_depth_db", "lock_losses",
              "elev_deg_at_peak", "az_deg_at_peak", "cn0_mean", "edge_of_gap"]


def _valid(obs: pd.DataFrame, mask: float) -> pd.Series:
    return obs["cn0_dbhz"].notna() & (obs["elev_deg"] >= mask)


def _secs(t: pd.Series) -> np.ndarray:
    return t.astype("int64").to_numpy() / 1e9


def edge_zones(active: np.ndarray, cfg: dict, now: pd.Timestamp | None = None) -> np.ndarray:
    """Edge-of-gap zones around receiver silences. ``now`` (live mode) also treats the still-open
    gap after the last active second as a silence once it is longer than silence_min_s."""
    f = cfg["fades"]
    zones = common.silence_edge_zones(active, f["silence_min_s"], f["edge_guard_s"])
    if now is not None and len(active) and now.timestamp() - active[-1] > f["silence_min_s"]:
        zones = np.r_[zones, [[active[-1] - f["edge_guard_s"], active[-1] + 1]]]
    return zones


def add_baseline(obs: pd.DataFrame, active: np.ndarray, cfg: dict, rate_hz: float) -> pd.DataFrame:
    """Add ``baseline_db`` and ``detrended_db`` (= cn0 - baseline) to every row."""
    f = cfg["fades"]
    win_s = f["baseline_window_min"] * 60
    obs = obs.reset_index(drop=True)
    obs["baseline_db"] = np.nan
    valid = _valid(obs, cfg["elev_mask_deg"])
    for _, g in obs[valid].groupby(KEYS, sort=False):
        g = g.sort_values("time_utc")
        s = pd.Series(g["cn0_dbhz"].to_numpy(), index=pd.DatetimeIndex(g["time_utc"]))
        roll = s.rolling(f"{win_s}s", center=True, min_periods=1)
        med, cnt = roll.median().to_numpy(), roll.count().to_numpy()
        t = _secs(g["time_utc"])
        act = common.count_in(active, t - win_s / 2, t + win_s / 2)
        need = np.maximum(f["baseline_min_samples"] * rate_hz, f["baseline_min_frac"] * act * rate_hz)
        obs.loc[g.index, "baseline_db"] = np.where(cnt >= need, med, np.nan)
    obs["detrended_db"] = obs["cn0_dbhz"] - obs["baseline_db"]
    return obs


def _gaps(obs: pd.DataFrame, active: np.ndarray, zones: np.ndarray, cfg: dict, rate_hz: float) -> pd.DataFrame:
    """One row per lock loss / receiver_off gap: series keys, minute_utc, kind.

    kind is lock_loss, lock_loss_edge (the loss starts inside an edge zone) or receiver_off.
    """
    mask = cfg["elev_mask_deg"]
    min_gap = cfg["fades"]["lock_loss_min_gap_s"]
    dt_nom = 1.0 / rate_hz
    seg_start, seg_end = common.receiver_segments(active, min_gap)
    out = []
    valid = _valid(obs, mask)
    for key, g in obs.groupby(KEYS, sort=False):
        gv = g[valid.loc[g.index]].sort_values("time_utc")
        if not len(gv):
            continue
        tv = _secs(gv["time_utc"])
        ge = g[g["elev_deg"].notna()].sort_values("time_utc")
        te, ee = _secs(ge["time_utc"]), ge["elev_deg"].to_numpy()
        # Each valid sample's following gap; the last one runs to the end of its segment.
        si = np.clip(np.searchsorted(seg_start, np.floor(tv), side="right") - 1, 0, max(len(seg_start) - 1, 0))
        s_end = seg_end[si] + 1 if len(seg_end) else tv + dt_nom
        nxt = np.r_[tv[1:], np.inf]
        same_seg = nxt <= s_end
        missing_on = np.where(same_seg, nxt, s_end) - tv - dt_nom
        missing_any = nxt - tv - dt_nom
        for i in np.flatnonzero(missing_any >= min_gap - 1e-6):
            t_gap = tv[i] + dt_nom
            el = np.interp(t_gap + min_gap, te, ee) if len(te) else gv["elev_deg"].iloc[i]
            if missing_on[i] >= min_gap - 1e-6 and el >= mask:
                kind = "lock_loss_edge" if common.in_zones([t_gap], zones)[0] else "lock_loss"
            elif not same_seg[i] and np.isfinite(nxt[i]):
                kind = "receiver_off"
            else:
                continue
            # Booked to the minute of the last valid sample, which always has a per-minute row.
            out.append((*key, pd.Timestamp(tv[i], unit="s", tz="UTC").floor("min"), kind))
    return pd.DataFrame(out, columns=[*KEYS, "minute_utc", "kind"])


def _source_label(s: pd.Series) -> str:
    u = s.dropna().unique()
    return u[0] if len(u) == 1 else ("mixed" if len(u) else "")


def _group_label(g) -> pd.Series:
    """Vectorised _source_label over a SeriesGroupBy: the single value, 'mixed', or ''."""
    a = g.agg(["nunique", "first"])
    return pd.Series(np.select([a["nunique"] == 1, a["nunique"] > 1], [a["first"], "mixed"], ""), index=a.index)


def per_minute(obs_b: pd.DataFrame, active: np.ndarray, cfg: dict, rate_hz: float,
               now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per (minute, series) fade table. ``obs_b`` must come from add_baseline(); ``now`` is the
    live-mode clock (see edge_zones)."""
    if not len(obs_b):
        return pd.DataFrame(columns=MINUTE_COLS)
    mask = cfg["elev_mask_deg"]
    zones = edge_zones(active, cfg, now)
    o = common.add_minute(obs_b.copy())
    o["valid"] = _valid(o, mask)
    o["edge"] = common.in_zones(_secs(o["time_utc"]), zones)
    depth = (o["baseline_db"] - o["cn0_dbhz"]).where(o["valid"])
    o["depth"] = depth.where(~o["edge"])
    o["depth_edge"] = depth.where(o["edge"])
    o["cn0_v"] = o["cn0_dbhz"].where(o["valid"])
    o["src_v"] = o["source"].where(o["valid"])
    gb = o.groupby([*KEYS, "minute_utc"], sort=True)
    m = gb.agg(elev_deg_mean=("elev_deg", "mean"), n_rows=("cn0_dbhz", "size"), n=("valid", "sum"),
               cn0_mean=("cn0_v", "mean"), cn0_min=("cn0_v", "min"), baseline_db=("baseline_db", "mean"),
               fade_depth_db=("depth", "max"), edge_fade_depth_db=("depth_edge", "max"),
               edge_of_gap=("edge", "any"))
    m["az_deg_mean"] = gb["az_deg"].agg(common.circmean_deg)
    src_v, src_all = _group_label(gb["src_v"]), _group_label(gb["source"])
    m["source"] = src_v.where(src_v != "", src_all)
    m = m.reset_index()

    gaps = _gaps(obs_b, active, zones, cfg, rate_hz)
    for kind, col in (("lock_loss", "lock_losses"), ("lock_loss_edge", "edge_lock_losses"),
                      ("receiver_off", "receiver_off_gaps")):
        c = gaps[gaps["kind"] == kind].groupby([*KEYS, "minute_utc"]).size().rename(col)
        m = m.merge(c, left_on=[*KEYS, "minute_utc"], right_index=True, how="left")
        m[col] = m[col].fillna(0).astype(int)
    secs = (m["minute_utc"].astype("int64") // 10**9).to_numpy()
    m["receiver_active_s"] = common.count_in(active, secs, secs + 59)
    thr = cfg["fades"]["fade_threshold_db"]
    m["event"] = (m["fade_depth_db"] >= thr) | (m["lock_losses"] >= 1)
    m["edge_event"] = ~m["event"] & ((m["edge_fade_depth_db"] >= thr) | (m["edge_lock_losses"] >= 1))
    m["n"] = m["n"].astype(int)
    return m[MINUTE_COLS]


def events(minutes: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive event minutes (``event`` or ``edge_event``) of a series into fade events.

    An event that touches an edge-of-gap minute is tagged edge_of_gap as a whole: a fade that
    runs into a receiver silence (e.g. a phone's C/N0 sagging for minutes before it powers
    GNSS down) is the shutdown, not the sky, even where it starts before the guard window.
    Edge events are reported but never classified as real events.
    """
    ev = minutes[minutes["event"] | minutes["edge_event"]].sort_values([*KEYS, "minute_utc"]).copy()
    if not len(ev):
        return pd.DataFrame(columns=EVENT_COLS)
    same = (ev[KEYS].eq(ev[KEYS].shift()).all(axis=1)
            & (ev["minute_utc"] - ev["minute_utc"].shift() == pd.Timedelta(minutes=1)))
    ev["eid"] = (~same).cumsum()
    rows = []
    for _, g in ev.groupby("eid"):
        edge = bool(g["edge_of_gap"].any())
        depth = g[["fade_depth_db", "edge_fade_depth_db"]].max(axis=1) if edge else g["fade_depth_db"]
        pk = g.loc[depth.idxmax()] if depth.notna().any() else g.iloc[0]
        r = g.iloc[0]
        rows.append({
            "receiver": r["receiver"], "source": _source_label(g["source"].replace("", np.nan)) or "",
            "constellation": r["constellation"], "svid": int(r["svid"]),
            "satellite": common.sat_label(r["constellation"], r["svid"]), "signal": r["signal"],
            "start_utc": g["minute_utc"].min(), "end_utc": g["minute_utc"].max() + pd.Timedelta(minutes=1),
            "n_minutes": int(len(g)), "peak_utc": pk["minute_utc"],
            "peak_depth_db": float(depth.max()) if depth.notna().any() else np.nan,
            "lock_losses": int(g["lock_losses"].sum() + (g["edge_lock_losses"].sum() if edge else 0)),
            "elev_deg_at_peak": pk["elev_deg_mean"], "az_deg_at_peak": pk["az_deg_mean"],
            "cn0_mean": float(g["cn0_mean"].mean()), "edge_of_gap": edge})
    if not rows:
        return pd.DataFrame(columns=EVENT_COLS)
    out = pd.DataFrame(rows).sort_values(["start_utc", "satellite", "signal"]).reset_index(drop=True)
    out.insert(0, "event_id", [f"{r.receiver}-{i + 1:04d}" for i, r in enumerate(out.itertuples())])
    return out[EVENT_COLS]
