"""Parse Google GnssLogger .txt files into tidy Raw / Status / Fix / Agc tables.

Column names are read from the file's own ``# <Type>,col,...`` header lines.

Many phones (incl. the vivo I2221 / Qualcomm logs) write most ``Status`` rows with an
empty ``UnixTimeMillis``. Those rows are timed from their position in the file relative
to timestamped Raw / Agc / Fix rows (see ``_time_status_blocks``); every Status row
carries a ``time_quality`` saying how its time was obtained.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ionwatch.proc import common

ROW_TYPES = ("Raw", "Status", "Fix", "Agc")
STRING_COLS = {"CodeType", "Provider", "ExtraKeyValuePairs"}
INT_COLS = {"utcTimeMillis", "TimeNanos", "FullBiasNanos", "ReceivedSvTimeNanos",
            "ChipsetElapsedRealtimeNanos", "UnixTimeMillis", "elapsedRealtimeNanos"}
RAW_KEY = ["utcTimeMillis", "ConstellationType", "Svid", "CarrierFrequencyHz"]
ANCHOR_FIX_PROVIDERS = ("GPS", "FLP")  # NLP fix times can lag; not used for timing

STATE_TOW_DECODED = 8
STATE_TOW_KNOWN = 16384
ADR_STATE_VALID = 1
WEEK_NS = 604_800 * 10**9

# Status time qualities usable for analysis.
GOOD_TIME = ("logged", "estimated", "compressed", "extrapolated")


@dataclass
class PhoneLog:
    raw: pd.DataFrame
    status: pd.DataFrame
    fix: pd.DataFrame
    agc: pd.DataFrame
    files: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)
    status_timing: dict = field(default_factory=dict)


# ---------------------------------------------------------------- low-level reading

def _scan(path: Path):
    cols: dict[str, list[str]] = {}
    rows = {t: [] for t in ROW_TYPES}
    lines = {t: [] for t in ROW_TYPES}
    device = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if body.startswith("Version:"):
                    device = body
                head, _, rest = body.partition(",")
                if head in rows and rest:
                    cols[head] = rest.split(",")
                continue
            head, _, rest = line.partition(",")
            if head in rows:
                rows[head].append(rest.split(","))
                lines[head].append(i)
    return cols, rows, lines, device


def _to_int64(values) -> pd.arrays.IntegerArray:
    out = []
    for v in values:
        if v == "" or v is None:
            out.append(pd.NA)
            continue
        try:
            out.append(int(v))
        except ValueError:
            try:
                out.append(int(float(v)))
            except ValueError:
                out.append(pd.NA)
    return pd.array(out, dtype="Int64")


def _frame(cols: list[str], rows: list[list[str]]) -> pd.DataFrame:
    n = len(cols)
    fixed = []
    for r in rows:
        if len(r) > n:  # trailing free-text column (e.g. ExtraKeyValuePairs) containing commas
            r = r[: n - 1] + [",".join(r[n - 1:])]
        elif len(r) < n:
            r = r + [""] * (n - len(r))
        fixed.append(r)
    df = pd.DataFrame(fixed, columns=cols)
    for c in df.columns:
        if c in STRING_COLS:
            df[c] = df[c].replace("", np.nan)
        elif c in INT_COLS:
            df[c] = _to_int64(df[c])
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _f64(s: pd.Series) -> np.ndarray:
    return s.to_numpy(dtype="float64", na_value=np.nan)


def _ms_to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(_f64(s), unit="ms", utc=True)


# ---------------------------------------------------------------- Status timing

def _time_status_blocks(block_pos, block_logged_ms, anc_pos, anc_ms, anc_is_status, gap_tol_s):
    """Assign a time (ms) to each Status block from file order.

    Status blocks are written at 1 Hz while the GNSS engine runs. A block that follows a
    Raw/Agc/Fix anchor belongs to that anchor's epoch; a block following a logged Status
    block is the next second. Between two anchors, N untimed blocks are consistent with
    1 Hz if the anchors are no more than ``gap_tol_s`` further apart than needed; if they
    are much further apart the engine was off for part of the interval and the blocks'
    position inside it is unknown ("ambiguous").
    """
    n = len(block_pos)
    t = block_logged_ms.astype(float).copy()
    quality = np.where(np.isfinite(t), "logged", "none").astype(object)
    if len(anc_pos) == 0:
        return t, quality
    todo = np.flatnonzero(~np.isfinite(t))
    i_prev = np.searchsorted(anc_pos, block_pos[todo], side="left") - 1
    na = len(anc_pos)
    for j, ip in zip(todo, i_prev):
        has_prev, has_next = ip >= 0, ip + 1 < na
        first = np.searchsorted(block_pos, anc_pos[ip], side="right") if has_prev else 0
        last = np.searchsorted(block_pos, anc_pos[ip + 1], side="left") - 1 if has_next else n - 1
        n_blk, r = last - first + 1, j - first + 1
        off = 1 if has_prev and anc_is_status[ip] else 0
        if has_prev and has_next:
            dt = (anc_ms[ip + 1] - anc_ms[ip]) / 1000.0
            need = n_blk - 1 + off
            if dt + gap_tol_s < need:
                step = dt / max(need, 1)
                t[j], quality[j] = anc_ms[ip] + (r - 1 + off) * step * 1000, "compressed"
            else:
                t[j] = anc_ms[ip] + (r - 1 + off) * 1000.0
                quality[j] = "estimated" if dt - need <= gap_tol_s else "ambiguous"
        elif has_prev:
            t[j], quality[j] = anc_ms[ip] + (r - 1 + off) * 1000.0, "extrapolated"
        else:
            t[j], quality[j] = anc_ms[ip + 1] - (n_blk - r + 1) * 1000.0, "extrapolated"
    return t, quality


def _monotonic(pos, ms, is_status):
    order = np.argsort(pos, kind="stable")
    pos, ms, is_status = pos[order], ms[order], is_status[order]
    keep = ms >= np.maximum.accumulate(ms)
    return pos[keep], ms[keep], is_status[keep], int((~keep).sum())


def _time_status(status: pd.DataFrame, raw: pd.DataFrame, agc: pd.DataFrame, fix: pd.DataFrame,
                 gap_tol_s: float):
    si = status["SignalIndex"].to_numpy()
    new_block = np.r_[True, si[1:] <= si[:-1]]
    block_id = np.cumsum(new_block) - 1
    status["_block"] = block_id
    logged = pd.Series(_f64(status["UnixTimeMillis"])).groupby(block_id).first().to_numpy()
    block_pos = status["_line"].to_numpy()[new_block]

    parts = []
    for df, col in ((raw, "utcTimeMillis"), (agc, "utcTimeMillis")):
        if len(df):
            parts.append((df["_line"].to_numpy(), _f64(df[col])))
    if len(fix):
        f = fix[fix["Provider"].isin(ANCHOR_FIX_PROVIDERS)]
        parts.append((f["_line"].to_numpy(), _f64(f["UnixTimeMillis"])))
    ext_pos = np.concatenate([p for p, _ in parts]) if parts else np.array([], dtype=np.int64)
    ext_ms = np.concatenate([m for _, m in parts]) if parts else np.array([])
    ok = np.isfinite(ext_ms)
    ext_pos, ext_ms = ext_pos[ok], ext_ms[ok]

    has_log = np.isfinite(logged)
    a_pos = np.r_[ext_pos, block_pos[has_log]]
    a_ms = np.r_[ext_ms, logged[has_log]]
    a_st = np.r_[np.zeros(len(ext_pos), bool), np.ones(has_log.sum(), bool)]
    a_pos, a_ms, a_st, dropped = _monotonic(a_pos, a_ms, a_st)
    t_blk, q_blk = _time_status_blocks(block_pos, logged, a_pos, a_ms, a_st, gap_tol_s)

    # Hold-out validation: re-time the logged blocks using only non-Status anchors.
    h_pos, h_ms, h_st, _ = _monotonic(ext_pos, ext_ms, np.zeros(len(ext_pos), bool))
    t_hold, q_hold = _time_status_blocks(block_pos, np.full(len(block_pos), np.nan),
                                         h_pos, h_ms, h_st, gap_tol_s)
    sel = has_log & np.isin(q_hold, ["estimated", "compressed"])
    err = (t_hold[sel] - logged[sel]) / 1000.0
    holdout: dict = {"n_blocks": int(sel.sum())}
    if sel.any():
        holdout.update(median_abs_err_s=round(float(np.median(np.abs(err))), 3),
                       p95_abs_err_s=round(float(np.percentile(np.abs(err), 95)), 3),
                       frac_within_1s=round(float(np.mean(np.abs(err) <= 1.0)), 3))

    status["time_utc"] = pd.to_datetime(t_blk[block_id], unit="ms", utc=True)
    status["time_quality"] = q_blk[block_id]
    counts = pd.Series(q_blk).value_counts().to_dict()
    timing = {"blocks": int(len(block_pos)), "rows": int(len(status)),
              "block_quality": {k: int(v) for k, v in counts.items()},
              "anchors_dropped_non_monotonic": dropped, "holdout_validation": holdout}
    return status, timing


# ---------------------------------------------------------------- public API

def read_file(path: str | Path, cfg: dict) -> PhoneLog:
    path = Path(path)
    cols, rows, lines, device = _scan(path)
    frames = {}
    for t in ROW_TYPES:
        if rows[t] and t not in cols:
            raise ValueError(f"{path}: found {t} rows but no '# {t},...' header line")
        df = _frame(cols.get(t, []), rows[t])
        df["_line"] = np.asarray(lines[t], dtype=np.int64)
        frames[t] = df
    raw, status, fix, agc = (frames[t] for t in ROW_TYPES)

    if len(raw):
        raw["time_utc"] = _ms_to_utc(raw["utcTimeMillis"])
        common.add_signal_columns(raw)
    if len(agc):
        agc["time_utc"] = _ms_to_utc(agc["utcTimeMillis"])
        common.add_signal_columns(agc, code_col=None)
    if len(fix):
        fix["time_utc"] = _ms_to_utc(fix["UnixTimeMillis"])
    timing = {}
    if len(status):
        status, timing = _time_status(status, raw, agc, fix, cfg["phone"]["status_time_gap_tol_s"])
        common.add_signal_columns(status, code_col=None)
        cn0 = status["Cn0DbHz"]
        status["tracked"] = cn0 > 0
        status["cn0_dbhz"] = cn0.where(cn0 > 0)
        status["elev_deg"] = status["ElevationDegrees"]
        status["az_deg"] = status["AzimuthDegrees"]
    for df in (raw, status, fix, agc):
        df["file"] = path.name
    return PhoneLog(raw=raw, status=status, fix=fix, agc=agc, files=[str(path)],
                    devices=[device], status_timing={path.name: timing})


def read_phone(paths, cfg: dict) -> PhoneLog:
    """Read one or more GnssLogger files, concatenate and de-duplicate."""
    logs = [read_file(p, cfg) for p in paths]

    def cat(name, key):
        dfs = [getattr(l, name) for l in logs if len(getattr(l, name))]
        if not dfs:
            return getattr(logs[0], name)
        df = pd.concat(dfs, ignore_index=True)
        if "time_utc" in df:
            df = df.sort_values("time_utc", kind="stable")
            if len(logs) > 1:  # GnssLogger split sessions can overlap
                df = df.drop_duplicates(key, keep="first")
        return df.reset_index(drop=True)

    return PhoneLog(
        raw=cat("raw", RAW_KEY),
        status=cat("status", ["time_utc", "ConstellationType", "Svid", "CarrierFrequencyHz"]),
        fix=cat("fix", ["UnixTimeMillis", "Provider"]),
        agc=cat("agc", ["utcTimeMillis", "ConstellationType", "CarrierFrequencyHz"]),
        files=[f for l in logs for f in l.files],
        devices=[d for l in logs for d in l.devices],
        status_timing={k: v for l in logs for k, v in l.status_timing.items()},
    )


def agc_label(agc: pd.DataFrame) -> pd.Series:
    """Signal name per Agc row, or '<constellation>_<MHz>' if the frequency is unclassified."""
    return agc["signal"].fillna(agc["constellation"].fillna("?") + "_" +
                                (agc["CarrierFrequencyHz"] / 1e6).round(2).astype(str))


def receiver_position(fix: pd.DataFrame, cfg: dict) -> dict:
    """Median receiver position from GPS (else FLP) fixes; config location as fallback."""
    for prov in ANCHOR_FIX_PROVIDERS:
        f = fix[(fix["Provider"] == prov) & fix["LatitudeDegrees"].notna()] if len(fix) else fix
        if len(f):
            lat, lon = f["LatitudeDegrees"].median(), f["LongitudeDegrees"].median()
            alt = f["AltitudeMeters"].median()
            north = (f["LatitudeDegrees"] - lat) * 111_320.0
            east = (f["LongitudeDegrees"] - lon) * 111_320.0 * np.cos(np.radians(lat))
            return {"source": f"Fix/{prov}", "n_fixes": int(len(f)), "lat_deg": float(lat),
                    "lon_deg": float(lon),
                    "height_m": float(alt) if np.isfinite(alt) else cfg["location"]["height_m"],
                    "horiz_std_m": float(np.sqrt(north.var(ddof=0) + east.var(ddof=0)))}
    loc = cfg["location"]
    return {"source": "config", "n_fixes": 0, **loc, "horiz_std_m": None}


def attach_elevation(log: PhoneLog, cfg: dict) -> dict:
    """Give Raw rows elev/az from Status (nearest time on constellation+svid), else geometry.

    Elevation is a property of the satellite, not the band, so the join is on
    (ConstellationType, Svid). Rows the join cannot fill get elevation computed from the
    Raw row's SvPositionEcef and the receiver position. ``elev_geom_deg`` is kept on every
    row so the two can be compared.
    """
    raw, status = log.raw, log.status
    stats = {"rows": int(len(raw))}
    if not len(raw):
        return stats
    pos = receiver_position(log.fix, cfg)
    xyz = raw[["SvPositionEcefXMeters", "SvPositionEcefYMeters", "SvPositionEcefZMeters"]].to_numpy(float)
    az_g, el_g = common.ecef_to_azel(xyz, pos["lat_deg"], pos["lon_deg"], pos["height_m"])
    raw["elev_geom_deg"], raw["az_geom_deg"] = el_g, az_g

    raw["elev_deg"], raw["az_deg"], raw["elev_src"] = np.nan, np.nan, None
    if len(status):
        st = status[status["time_quality"].isin(GOOD_TIME) & status["ElevationDegrees"].notna()]
        st = (st.sort_values("time_utc")
                .drop_duplicates(["time_utc", "ConstellationType", "Svid"])
                [["time_utc", "ConstellationType", "Svid", "ElevationDegrees", "AzimuthDegrees"]])
        left = raw[["time_utc", "ConstellationType", "Svid"]].reset_index().sort_values("time_utc")
        m = pd.merge_asof(left, st, on="time_utc", by=["ConstellationType", "Svid"],
                          direction="nearest",
                          tolerance=pd.Timedelta(seconds=cfg["phone"]["status_join_tolerance_s"]))
        m = m.set_index("index").reindex(raw.index)
        hit = m["ElevationDegrees"].notna()
        raw.loc[hit, "elev_deg"] = m.loc[hit, "ElevationDegrees"]
        raw.loc[hit, "az_deg"] = m.loc[hit, "AzimuthDegrees"]
        raw.loc[hit, "elev_src"] = "status"
    fill = raw["elev_deg"].isna() & np.isfinite(el_g)
    raw.loc[fill, "elev_deg"] = el_g[fill.to_numpy()]
    raw.loc[fill, "az_deg"] = az_g[fill.to_numpy()]
    raw.loc[fill, "elev_src"] = "geometry"

    n_status = int((raw["elev_src"] == "status").sum())
    both = (raw["elev_src"] == "status") & np.isfinite(el_g)
    diff = (raw.loc[both, "elev_deg"] - raw.loc[both, "elev_geom_deg"]).abs()
    stats.update(matched_status=n_status, match_rate=round(n_status / len(raw), 4),
                 filled_geometry=int(fill.sum()), unmatched=int(raw["elev_deg"].isna().sum()),
                 status_vs_geometry_median_abs_diff_deg=round(float(diff.median()), 3) if len(diff) else None,
                 status_vs_geometry_p95_abs_diff_deg=round(float(diff.quantile(0.95)), 3) if len(diff) else None,
                 receiver_position=pos)
    return stats


def pseudorange_m(raw: pd.DataFrame) -> pd.Series:
    """Code pseudorange (m) for GPS-time-of-week based signals (GPS, Galileo, QZSS).

    The integer part TimeNanos - FullBiasNanos is formed in int64 before going to float.

    Spec 6.4 wrote ``t_tx = ReceivedSvTimeNanos + TimeOffsetNanos``; that has the sign of
    TimeOffsetNanos reversed. Android defines the measurement time as
    TimeNanos + TimeOffsetNanos, so the offset belongs on the receive side:
        t_rx = (TimeNanos + TimeOffsetNanos) - (FullBiasNanos + BiasNanos)   mod week
        t_tx = ReceivedSvTimeNanos
        P    = (t_rx - t_tx) * c
    (TimeOffsetNanos is 0 on every row of the logs seen so far.)
    """
    t_int = raw["TimeNanos"].astype("Int64") - raw["FullBiasNanos"].astype("Int64")
    tow_int = _f64(t_int % WEEK_NS)
    t_rx = tow_int - raw["BiasNanos"].fillna(0).to_numpy() + raw["TimeOffsetNanos"].fillna(0).to_numpy()
    t_tx = _f64(raw["ReceivedSvTimeNanos"])
    p = (t_rx - t_tx) * common.C_LIGHT / 1e9
    p = np.where(p < 0, p + WEEK_NS * common.C_LIGHT / 1e9, p)
    return pd.Series(p, index=raw.index, name="pseudorange_m")


def tow_valid(raw: pd.DataFrame) -> pd.Series:
    state = raw["State"].fillna(0).astype(np.int64)
    return ((state & STATE_TOW_DECODED) > 0) | ((state & STATE_TOW_KNOWN) > 0)


def cn0_obs(log: PhoneLog, cfg: dict, receiver: str = "phone") -> tuple[pd.DataFrame, dict]:
    """Tidy per-sample C/N0 table (common.OBS_COLS) from Raw + Status.

    Raw rows are used where they exist (exact time, source='raw'). Status rows fill every
    (second, satellite, signal) Raw does not cover (source='status'), including untracked
    satellites (cn0 NaN) so elevation and "listed but not tracked" are kept. Status blocks
    whose time could not be placed (ambiguous / none) and rows with no carrier frequency
    are dropped and counted.
    """
    parts, stats = [], {}
    raw, st = log.raw, log.status
    if len(raw) and "elev_deg" not in raw:
        attach_elevation(log, cfg)
    if len(raw):
        r = raw[raw["signal"].notna()]
        parts.append(pd.DataFrame({
            "time_utc": r["time_utc"], "receiver": receiver, "source": "raw",
            "constellation": r["constellation"], "svid": r["Svid"].astype(int), "signal": r["signal"],
            "band": r["band"], "elev_deg": r["elev_deg"], "az_deg": r["az_deg"],
            "cn0_dbhz": r["Cn0DbHz"].where(r["Cn0DbHz"] > 0), "time_quality": "logged"}))
        stats["raw_rows_unknown_signal"] = int(raw["signal"].isna().sum())
    if len(st):
        good = st["time_quality"].isin(GOOD_TIME)
        blocks = st.drop_duplicates(["file", "_block"])
        stats["status_blocks_dropped_untimed"] = int((~blocks["time_quality"].isin(GOOD_TIME)).sum())
        stats["status_rows_dropped_untimed"] = int((~good).sum())
        stats["status_rows_no_frequency"] = int((good & st["signal"].isna()).sum())
        s = st[good & st["signal"].notna()]
        parts.append(pd.DataFrame({
            "time_utc": s["time_utc"], "receiver": receiver, "source": "status",
            "constellation": s["constellation"], "svid": s["Svid"].astype(int), "signal": s["signal"],
            "band": s["band"], "elev_deg": s["elev_deg"], "az_deg": s["az_deg"],
            "cn0_dbhz": s["cn0_dbhz"], "time_quality": s["time_quality"]}))
    if not parts:
        return pd.DataFrame(columns=common.OBS_COLS), stats
    obs = pd.concat(parts, ignore_index=True)
    obs["_sec"] = obs["time_utc"].dt.floor("s")
    obs["_rank"] = (obs["source"] == "status").astype(int)
    obs = (obs.sort_values(["_sec", "_rank"], kind="stable")
              .drop_duplicates(["_sec", "constellation", "svid", "signal"], keep="first"))
    stats["rows_raw"] = int((obs["source"] == "raw").sum())
    stats["rows_status"] = int((obs["source"] == "status").sum())
    stats["rows_status_tracked"] = int(((obs["source"] == "status") & obs["cn0_dbhz"].notna()).sum())
    obs = obs.sort_values(["constellation", "svid", "signal", "time_utc"]).reset_index(drop=True)
    return obs[common.OBS_COLS], stats


# ---------------------------------------------------------------- data-quality report

def _stamp(t: pd.Timestamp, tz: str) -> dict:
    return {"utc": t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "ist": t.tz_convert(tz).strftime("%Y-%m-%dT%H:%M:%S%z")}


def _active_seconds(*series) -> np.ndarray:
    secs = [np.floor(s.dropna().astype("int64").to_numpy() / 1e9) for s in series if s is not None and len(s)]
    return np.unique(np.concatenate(secs)) if secs else np.array([])


def _r(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def phone_quality(log: PhoneLog, join: dict, cfg: dict) -> dict:
    """Section 5.3 data-quality report plus duty-cycle / movement / clock checks."""
    with warnings.catch_warnings():  # medians of all-NaN groups (untracked sats) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        return _phone_quality(log, join, cfg)


def _phone_quality(log: PhoneLog, join: dict, cfg: dict) -> dict:
    tz, qc, mask = cfg["display_tz"], cfg["quality"], cfg["elev_mask_deg"]
    raw, st, fix, agc = log.raw, log.status, log.fix, log.agc
    warn: list[dict] = []

    def _w(code: str, message: str):
        warn.append({"code": code, "message": message})

    q: dict = {"files": [Path(f).name for f in log.files], "devices": sorted(set(log.devices)),
               "row_counts": {"Raw": int(len(raw)), "Status": int(len(st)), "Agc": int(len(agc)),
                              "Fix": int(len(fix))}}

    st_ok = st[st["time_quality"].isin(GOOD_TIME)] if len(st) else st
    times = [s["time_utc"] for s in (raw, st_ok, fix) if len(s)]
    all_t = pd.concat(times).dropna()
    t0, t1 = all_t.min(), all_t.max()
    span_s = (t1 - t0).total_seconds()
    q["session"] = {"start": _stamp(t0, tz), "end": _stamp(t1, tz), "duration_min": _r(span_s / 60),
                    "date_utc": f"{t0:%Y-%m-%d}", "doy": int(t0.dayofyear)}

    # ---- Raw
    if len(raw):
        ep = np.unique(_f64(raw["utcTimeMillis"]))
        r_span = (ep[-1] - ep[0]) / 1000
        d = np.diff(ep) / 1000
        frac = len(ep) / (round(r_span) + 1)
        gap_i = np.flatnonzero(d > qc["gap_report_s"])
        hw = raw.groupby("utcTimeMillis")["HardwareClockDiscontinuityCount"].first().sort_index()
        ch = hw[hw.diff().fillna(0) != 0]
        sig = raw.groupby(raw["signal"].fillna("unknown"))
        p = pseudorange_m(raw)
        pr_ok = (tow_valid(raw) & (raw["ReceivedSvTimeUncertaintyNanos"] <= cfg["tec"]["max_sv_time_unc_ns"])
                 & raw["constellation"].isin(["GPS", "Galileo"]))
        in_rng = pr_ok & p.between(18e6, 30e6)
        pr = {}
        for s, g in p[pr_ok].groupby(raw.loc[pr_ok, "signal"]):
            pr[s] = {"n_valid_state": int(len(g)), "n_in_18_30e3km": int(g.between(18e6, 30e6).sum()),
                     "min_km": _r(g.min() / 1e3), "median_km": _r(g.median() / 1e3), "max_km": _r(g.max() / 1e3)}
        dual = {}
        for c in ("GPS", "Galileo", "BeiDou"):
            g = raw[(raw["constellation"] == c) & raw["band"].isin(["L1", "L5"])]
            nb = g.groupby(["utcTimeMillis", "Svid"])["band"].nunique()
            both = nb[nb == 2]
            dual[c] = {"sat_epochs": int(len(both)),
                       "svids": sorted(int(s) for s in both.index.get_level_values("Svid").unique())}
        adr_valid = int(((raw["AccumulatedDeltaRangeState"].fillna(0).astype(int) & ADR_STATE_VALID) > 0).sum())
        q["raw"] = {
            "rows": int(len(raw)), "epochs": int(len(ep)),
            "start": _stamp(pd.Timestamp(ep[0], unit="ms", tz="UTC"), tz),
            "end": _stamp(pd.Timestamp(ep[-1], unit="ms", tz="UTC"), tz),
            "duration_min": _r(r_span / 60), "coverage_of_session": _r(r_span / span_s, 3) if span_s else None,
            "epoch_spacing_s_hist": {str(k): int(v) for k, v in
                                     pd.Series(np.round(d, 1)).value_counts().sort_index().items()},
            "epoch_fraction_1s": _r(frac, 4),
            "gaps_over_s": {"threshold_s": qc["gap_report_s"], "count": int(len(gap_i)),
                            "list": [{"start": _stamp(pd.Timestamp(ep[i], unit="ms", tz="UTC"), tz),
                                      "length_s": _r(d[i], 3)} for i in gap_i[:20]]},
            "hw_clock_discontinuity": {
                "initial": int(hw.iloc[0]),
                "changes": [{"at": _stamp(pd.Timestamp(float(k), unit="ms", tz="UTC"), tz), "to": int(v)}
                            for k, v in ch.items()]},
            "rows_per_signal": {k: int(v) for k, v in sig.size().items()},
            "svids_per_signal": {k: sorted(int(x) for x in v.unique()) for k, v in sig["Svid"]},
            "mean_cn0_per_signal": {k: _r(v) for k, v in sig["Cn0DbHz"].mean().items()},
            "adr_valid_rows": adr_valid,
            "adr_state_values": {str(int(k)): int(v) for k, v in
                                 raw["AccumulatedDeltaRangeState"].value_counts().items()},
            "pseudorange": {"rows_tow_ok_gps_gal": int(pr_ok.sum()), "rows_in_range": int(in_rng.sum()),
                            "per_signal": pr},
            "dual_frequency": dual,
            "navic_rows": int((raw["constellation"] == "NavIC").sum()),
        }
        r0 = pd.Timestamp(ep[0], unit="ms", tz="UTC")
        r1 = pd.Timestamp(ep[-1], unit="ms", tz="UTC")
        per_s = len(raw) / len(ep)
        mean_cn0 = float(raw["Cn0DbHz"].mean())
        q["raw"]["meas_per_epoch"] = _r(per_s, 2)
        q["raw"]["mean_cn0_dbhz"] = _r(mean_cn0, 2)
        q["raw"]["night_window_relation"] = common.night_relation(r0, r1, cfg)
        if mean_cn0 < qc["low_cn0_mean_dbhz"]:
            _w("low_cn0", f"Low C/N0 (mean {mean_cn0:.1f} < {qc['low_cn0_mean_dbhz']} dB-Hz): likely indoors or "
                          "obstructed sky; fades/S4 will mostly reflect local blockage, not the ionosphere.")
        if adr_valid == 0:
            _w("no_carrier_phase", "No carrier phase (ADR valid = 0); TEC is code-only, uncalibrated.")
        if per_s < qc["min_raw_meas_per_s"]:
            _w("low_raw_density", f"Low Raw density (~{per_s:.1f} measurements/s).")
        if frac < qc["min_epoch_fraction"]:
            _w("raw_epochs_missing", f"Raw has {frac:.1%} of 1 s epochs over its span "
                                     f"(< {qc['min_epoch_fraction']:.0%}): suggests duty cycling.")
        if span_s and r_span / span_s < 0.9:
            _w("raw_partial", f"Raw measurements cover only {r_span/60:.0f} min of the {span_s/60:.0f} min "
                              f"log (Raw ends {q['raw']['end']['ist'][11:16]} IST, the log continues with "
                              "Status/Fix rows only); TEC and Raw C/N0 exist only in the Raw span.")
        if len(ch):
            _w("hw_clock_discontinuity",
               f"HardwareClockDiscontinuityCount changed {len(ch)} time(s) during Raw logging.")
        if q["raw"]["navic_rows"] == 0:
            _w("navic_not_in_raw", "NavIC is absent from Raw measurements.")
    else:
        _w("no_raw", "No Raw rows in the log: no TEC possible.")

    # ---- Status
    if len(st):
        active = _active_seconds(raw["time_utc"] if len(raw) else None, st_ok["time_utc"])
        act_frac = len(active) / (round(span_s) + 1) if span_s else 1.0
        sil = np.diff(active)
        long_i = np.flatnonzero(sil > 60)
        long_sil = sil[long_i]
        trk = st[st["tracked"]]
        hi = trk[trk["elev_deg"] > 40]
        sg = st.groupby(st["signal"].fillna("unknown"))
        q["status"] = {
            "rows": int(len(st)),
            "timing": log.status_timing,
            "tracked_fraction": _r(st["tracked"].mean(), 4),
            "receiver_activity": {"active_seconds": int(len(active)), "session_seconds": int(round(span_s) + 1),
                                  "epoch_fraction_1s": _r(act_frac, 4),
                                  "silences_over_60s": int(len(long_sil)),
                                  "silent_minutes_total": _r(long_sil.sum() / 60),
                                  "longest_silence_min": _r(long_sil.max() / 60) if len(long_sil) else 0.0,
                                  "silences": [{"start": _stamp(pd.Timestamp(active[i] + 1, unit="s", tz="UTC"), tz),
                                                "length_min": _r(sil[i] / 60)} for i in long_i]},
            "rows_per_signal": {k: int(v) for k, v in sg.size().items()},
            "tracked_rows_per_signal": {k: int(v) for k, v in sg["tracked"].sum().items()},
            "median_cn0_tracked_per_signal": {k: _r(v) for k, v in trk.groupby("signal")["cn0_dbhz"].median().items()},
            "median_cn0_above_40deg_per_constellation": {k: _r(v) for k, v in
                                                         hi.groupby("constellation")["cn0_dbhz"].median().items()},
        }
        if act_frac < qc["min_epoch_fraction"]:
            a = q["status"]["receiver_activity"]
            _w("duty_cycling", f"Receiver produced epochs in only {act_frac:.1%} of the log's seconds "
                               f"({a['silences_over_60s']} silences > 60 s, longest {a['longest_silence_min']} min): "
                               "the phone was duty-cycling GNSS. Lock loss is only counted while the receiver is active.")
        amb = sum(t.get("block_quality", {}).get("ambiguous", 0) for t in log.status_timing.values())
        nblk = sum(t.get("blocks", 0) for t in log.status_timing.values())
        if nblk and amb / nblk > 0.02:
            _w("status_time_ambiguous", f"{amb} of {nblk} Status blocks have ambiguous times (dropped from analysis).")
        if not len(raw):
            mean_cn0 = trk["cn0_dbhz"].mean()
            if np.isfinite(mean_cn0) and mean_cn0 < qc["low_cn0_mean_dbhz"]:
                _w("low_cn0", f"Low C/N0 (mean {mean_cn0:.1f} < {qc['low_cn0_mean_dbhz']} dB-Hz): likely indoors or "
                              "obstructed sky; fades/S4 will mostly reflect local blockage, not the ionosphere.")
        # Satellites listed in Status but (almost) never tracked.
        zero = st.groupby(["constellation", "Svid"])["tracked"].mean()
        zero = zero[1 - zero > qc["untracked_frac"]]
        by_c = {c: [common.sat_label(c, sv) for sv in g.index.get_level_values("Svid")]
                for c, g in zero.groupby(level="constellation")}
        q["status"]["untracked_sats"] = by_c
        if by_c:
            _w("untracked_sats", f"Satellites in Status with C/N0 = 0 in > {qc['untracked_frac']:.0%} of rows: " +
               "; ".join(f"{c}: {', '.join(v)}" for c, v in by_c.items()) + ".")

    # ---- session lines: Raw span, total log span, receiver-active minutes in the night window
    win = common.night_window_label(cfg)
    active_all = _active_seconds(raw["time_utc"] if len(raw) else None, st_ok["time_utc"] if len(st_ok) else None)
    in_night = common.in_night_window(pd.to_datetime(active_all, unit="s", utc=True), cfg) if len(active_all) else []
    night_min = round(float(np.sum(in_night)) / 60, 1)
    lines = []
    if "raw" in q:
        r = q["raw"]
        lines.append(f"Raw span: {r['start']['ist'][11:19]} → {r['end']['ist'][11:19]} IST "
                     f"({r['duration_min']:.0f} min).")
    lines.append(f"Total log span: {q['session']['start']['ist'][11:19]} → {q['session']['end']['ist'][11:19]} "
                 f"IST ({q['session']['duration_min']:.0f} min).")
    lines.append(f"Receiver-active data inside {win}: {night_min:g} min.")
    q["session"].update(lines=lines, active_min_in_night_window=night_min)
    if night_min == 0:
        rel = common.night_relation(t0, t1, cfg)
        _w("outside_night_window", f"No receiver-active data inside the {win} scintillation window" +
           {"before": " (the log ends before it opens).", "after": " (the log starts after it closed).",
            "overlaps": " (the receiver was off throughout it)."}[rel])

    # ---- NavIC
    nav_st = st[st["constellation"] == "NavIC"] if len(st) else st
    nav = {"svids_in_raw": sorted(int(s) for s in raw.loc[raw["constellation"] == "NavIC", "Svid"].unique())
           if len(raw) else [],
           "svids_in_status": sorted(int(s) for s in nav_st["Svid"].unique()) if len(nav_st) else [],
           "per_svid": {}}
    for sv, g in nav_st.groupby("Svid"):
        nav["per_svid"][int(sv)] = {"status_rows": int(len(g)), "tracked_fraction": _r(g["tracked"].mean(), 4),
                                    "median_cn0": _r(g["cn0_dbhz"].median()),
                                    "elev_min": _r(g["elev_deg"].min()), "elev_max": _r(g["elev_deg"].max()),
                                    "frac_above_mask": _r((g["elev_deg"] >= mask).mean(), 3),
                                    "has_ephemeris_frac": _r(g["HasEphemerisData"].mean(), 3)}
    nav["tracked"] = any(v["tracked_fraction"] for v in nav["per_svid"].values()) or bool(nav["svids_in_raw"])
    q["navic"] = nav
    if nav["svids_in_status"] and not nav["tracked"]:
        sats = ", ".join(f"I{s:02d} ({nav['per_svid'][s]['elev_max']:.0f}°)" for s in nav["svids_in_status"])
        _w("navic_not_tracked", f"NavIC satellites in view but not tracked (C/N0 = 0): {sats}; "
                                "NavIC health metrics unavailable.")
    elif not nav["svids_in_status"] and not nav["svids_in_raw"]:
        _w("navic_absent", "No NavIC satellites seen at all; NavIC health metrics unavailable.")

    # ---- elevation join, position, AGC
    q["elevation_join"] = join
    pos = join.get("receiver_position") or receiver_position(fix, cfg)
    q["position"] = pos
    if join.get("match_rate") is not None and join["match_rate"] < 0.9:
        _w("elev_join_low", f"Elevation join matched {join['match_rate']:.1%} of Raw rows from Status; "
                            f"{join['filled_geometry']} rows filled from SvPositionEcef geometry.")
    if pos.get("horiz_std_m") is not None and pos["horiz_std_m"] > qc["max_position_std_m"]:
        _w("phone_moved", f"Phone position std {pos['horiz_std_m']:.1f} m (> {qc['max_position_std_m']} m) "
                          f"over {pos['n_fixes']} {pos['source']} fixes: the phone may have moved.")
    if len(agc):
        label = agc_label(agc)
        ph = agc["AgcDb"] <= cfg["interference"]["agc_placeholder_below_db"]
        placeholders = sorted(label[ph].unique())
        q["agc"] = {"rows": int(len(agc)),
                    "start": _stamp(agc["time_utc"].min(), tz), "end": _stamp(agc["time_utc"].max(), tz),
                    "per_signal": {k: {"rows": int(len(g)), "mean_db": _r(g["AgcDb"].mean(), 2),
                                       "min_db": _r(g["AgcDb"].min(), 2), "max_db": _r(g["AgcDb"].max(), 2)}
                                   for k, g in agc.groupby(label)},
                    "agc_placeholder": {k: int((ph & (label == k)).sum()) for k in placeholders}}
        if placeholders:
            _w("agc_placeholder", f"AGC reads <= {cfg['interference']['agc_placeholder_below_db']} dB "
                                  f"(placeholder, not a measurement) on {', '.join(placeholders)}; "
                                  "excluded from the interference check.")
    else:
        q["agc"] = None
    if len(st):
        nofreq = int(st["CarrierFrequencyHz"].isna().sum())
        q["status"]["rows_no_frequency"] = nofreq
        if nofreq:
            _w("status_no_frequency", f"{nofreq} Status rows have no CarrierFrequencyHz; "
                                      "kept for elevation, excluded from C/N0 analysis.")
    q["warnings"] = warn
    return q


def format_quality(q: dict) -> str:
    """Human-readable rendering of phone_quality()."""
    L = [f"Files: {', '.join(q['files'])}"]
    L += [f"Device: {d}" for d in q["devices"] if d]
    L.append("Rows: " + ", ".join(f"{k} {v}" for k, v in q["row_counts"].items()))
    s = q["session"]
    L.append(f"Log span: {s['start']['ist']} -> {s['end']['ist']} IST  ({s['duration_min']} min), "
             f"date {s['date_utc']} UTC, DOY {s['doy']}")
    L += [f"  {x}" for x in s.get("lines", [])]
    L.append(f"         {s['start']['utc']} -> {s['end']['utc']}")
    if "raw" in q:
        r = q["raw"]
        L += ["", f"RAW  {r['rows']} rows, {r['epochs']} epochs, {r['start']['ist'][11:19]} -> "
                  f"{r['end']['ist'][11:19]} IST ({r['duration_min']} min, {r['coverage_of_session']:.0%} of session)"]
        L.append(f"     epoch spacing (s): {r['epoch_spacing_s_hist']}   1 s epoch fraction: {r['epoch_fraction_1s']:.2%}")
        L.append(f"     gaps > {r['gaps_over_s']['threshold_s']} s: {r['gaps_over_s']['count']}")
        hw = r["hw_clock_discontinuity"]
        L.append(f"     HW clock discontinuity count: {hw['initial']}, changes: {len(hw['changes'])}")
        L.append(f"     ADR valid rows: {r['adr_valid_rows']}  (states {r['adr_state_values']})")
        L.append("     signal       rows  mean C/N0  svids")
        for k, n in sorted(r["rows_per_signal"].items()):
            L.append(f"     {k:<10} {n:>6}  {r['mean_cn0_per_signal'][k]!s:>9}  {r['svids_per_signal'][k]}")
        pr = r["pseudorange"]
        L.append(f"     pseudorange (GPS/GAL, TOW ok): {pr['rows_tow_ok_gps_gal']} rows, "
                 f"{pr['rows_in_range']} in 18-30e3 km")
        for k, v in pr["per_signal"].items():
            L.append(f"       {k:<8} {v['min_km']!s:>10} .. {v['median_km']!s:>10} .. {v['max_km']!s:>10} km")
        L.append("     dual-frequency sat-epochs: " +
                 ", ".join(f"{c} {v['sat_epochs']} (svids {v['svids']})" for c, v in r["dual_frequency"].items()))
    if "status" in q:
        st = q["status"]
        L += ["", f"STATUS {st['rows']} rows, tracked fraction {st['tracked_fraction']:.1%}"]
        for f, t in st["timing"].items():
            L.append(f"     {f}: {t['blocks']} blocks, time quality {t['block_quality']}")
            L.append(f"       hold-out check of estimated vs logged times: {t['holdout_validation']}")
        a = st["receiver_activity"]
        L.append(f"     receiver active {a['active_seconds']} of {a['session_seconds']} s "
                 f"({a['epoch_fraction_1s']:.1%}); {a['silences_over_60s']} silences > 60 s, "
                 f"{a['silent_minutes_total']} min silent, longest {a['longest_silence_min']} min")
        for x in a["silences"]:
            L.append(f"       silent from {x['start']['ist'][11:19]} IST for {x['length_min']} min")
        L.append("     signal       rows  tracked  median C/N0 (tracked)")
        for k, n in sorted(st["rows_per_signal"].items()):
            L.append(f"     {k:<10} {n:>6}  {st['tracked_rows_per_signal'].get(k, 0):>7}  "
                     f"{st['median_cn0_tracked_per_signal'].get(k)}")
        L.append(f"     median C/N0 above 40 deg: {st['median_cn0_above_40deg_per_constellation']}")
    n = q["navic"]
    L += ["", f"NAVIC svids in Raw: {n['svids_in_raw']}, in Status: {n['svids_in_status']}"]
    L += [f"     I{sv:02d}: {v}" for sv, v in n["per_svid"].items()]
    j = q["elevation_join"]
    if j.get("rows"):
        L += ["", f"ELEVATION JOIN  Status match {j['match_rate']:.1%}, geometry-filled {j['filled_geometry']}, "
                  f"unmatched {j['unmatched']}; |Status - geometry| median "
                  f"{j['status_vs_geometry_median_abs_diff_deg']} deg, p95 {j['status_vs_geometry_p95_abs_diff_deg']} deg"]
    p = q["position"]
    std = f"{p['horiz_std_m']:.1f} m" if p.get("horiz_std_m") is not None else "n/a"
    L.append(f"POSITION ({p['source']}, {p['n_fixes']} fixes): {p['lat_deg']:.6f}, {p['lon_deg']:.6f}, "
             f"{p['height_m']:.0f} m; horizontal std {std}")
    if q.get("agc"):
        a = q["agc"]
        L.append(f"AGC {a['rows']} rows, {a['start']['ist'][11:19]} -> {a['end']['ist'][11:19]} IST: " +
                 ", ".join(f"{k} {v['mean_db']} dB" for k, v in a["per_signal"].items()))
    L += ["", "WARNINGS:" if q["warnings"] else "WARNINGS: none"]
    L += [f"  - [{w['code']}] {w['message']}" for w in q["warnings"]]
    return "\n".join(L)
