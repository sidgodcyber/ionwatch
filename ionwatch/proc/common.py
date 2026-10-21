"""Shared helpers: config, constellation/signal naming, time formatting, geometry."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

# Android GnssStatus constellation types.
CONSTELLATIONS = {1: "GPS", 2: "SBAS", 3: "GLONASS", 4: "QZSS", 5: "BeiDou", 6: "Galileo", 7: "NavIC"}
SHORT = {"GPS": "GPS", "SBAS": "SBAS", "GLONASS": "GLO", "QZSS": "QZS", "BeiDou": "BDS",
         "Galileo": "GAL", "NavIC": "NAVIC"}
PRN_LETTER = {"GPS": "G", "SBAS": "S", "GLONASS": "R", "QZSS": "J", "BeiDou": "C",
              "Galileo": "E", "NavIC": "I"}

# Band centres (Hz) and the per-constellation signal suffix for each band.
_BANDS = [("L1", 1575.42e6), ("B1I", 1561.098e6), ("L5", 1176.45e6)]
_GLO_G1 = (1598.0e6, 1606.0e6)
_SUFFIX = {
    ("GPS", "L1"): "L1", ("GPS", "L5"): "L5",
    ("Galileo", "L1"): "E1", ("Galileo", "L5"): "E5a",
    ("BeiDou", "L1"): "B1C", ("BeiDou", "B1I"): "B1I", ("BeiDou", "L5"): "B2a",
    ("QZSS", "L1"): "L1", ("QZSS", "L5"): "L5",
    ("SBAS", "L1"): "L1", ("SBAS", "L5"): "L5",
    ("NavIC", "L5"): "L5",
    ("GLONASS", "G1"): "G1",
}

C_LIGHT = 299_792_458.0
WGS84_A = 6_378_137.0
WGS84_E2 = 6.69437999014e-3


# ---------------------------------------------------------------- config

def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load the default config, overlaid with `path` if given."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        cfg = _deep_merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    return cfg


# ---------------------------------------------------------------- naming

def band_of(freq_hz: float) -> str | None:
    """Classify a carrier frequency into L1 / B1I / L5 / G1 (±1 MHz)."""
    if not np.isfinite(freq_hz):
        return None
    if _GLO_G1[0] <= freq_hz <= _GLO_G1[1]:
        return "G1"
    for name, centre in _BANDS:
        if abs(freq_hz - centre) <= 1e6:
            return name
    return None


def signal_name(constellation: str, freq_hz: float, code_type: str | None = None) -> str | None:
    """Signal label such as GPS_L1, GAL_E5a, BDS_B1C, NAVIC_L5; None if unknown."""
    band = band_of(freq_hz)
    suffix = _SUFFIX.get((constellation, band))
    if suffix is None:
        return None
    # CodeType only matters where one frequency carries two civil signals.
    if constellation == "GPS" and band == "L1" and code_type in ("L", "S", "X"):
        suffix = "L1C"
    return f"{SHORT[constellation]}_{suffix}"


def add_signal_columns(df: pd.DataFrame, freq_col: str = "CarrierFrequencyHz",
                       code_col: str | None = "CodeType") -> pd.DataFrame:
    """Add `constellation`, `band` and `signal` columns from ConstellationType / frequency / code."""
    df["constellation"] = df["ConstellationType"].map(CONSTELLATIONS)
    code = df[code_col] if code_col and code_col in df else pd.Series(None, index=df.index, dtype=object)
    keys = pd.DataFrame({"c": df["constellation"], "f": df[freq_col], "k": code}).astype(object)
    keys = keys.where(keys.notna(), None)  # NaN != NaN would break the lookup below
    uniq = keys.drop_duplicates()
    lut = {(r.c, r.f, r.k): (band_of(r.f) if r.f is not None else None,
                             signal_name(r.c, r.f, r.k) if isinstance(r.c, str) and r.f is not None else None)
           for r in uniq.itertuples(index=False)}
    mapped = [lut[t] for t in zip(keys["c"], keys["f"], keys["k"])]
    df["band"] = [m[0] for m in mapped]
    df["signal"] = [m[1] for m in mapped]
    return df


def sat_label(constellation: str, svid: int) -> str:
    return f"{PRN_LETTER.get(constellation, '?')}{int(svid):02d}"


# ---------------------------------------------------------------- time

def iso_utc(s: pd.Series) -> pd.Series:
    return s.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_local(s: pd.Series, tz: str) -> pd.Series:
    loc = s.dt.tz_convert(tz)
    off = loc.dt.strftime("%z")
    return loc.dt.strftime("%Y-%m-%dT%H:%M:%S") + off.str[:3] + ":" + off.str[3:]


def fmt_both(t: pd.Timestamp, tz: str) -> str:
    return f"{t:%Y-%m-%d %H:%M:%S} UTC ({t.tz_convert(tz):%Y-%m-%d %H:%M:%S} IST)"


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def in_night_window(times: pd.Series | pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """True where the local (display_tz) time-of-day is inside night_window_ist (may wrap midnight)."""
    loc = pd.DatetimeIndex(times).tz_convert(cfg["display_tz"])
    mins = loc.hour * 60 + loc.minute
    a, b = (_hhmm(x) for x in cfg["night_window_ist"])
    return np.asarray((mins >= a) & (mins < b) if a < b else (mins >= a) | (mins < b))


def night_window_label(cfg: dict) -> str:
    a, b = cfg["night_window_ist"]
    return f"{a}–{b} IST"


def night_relation(t0: pd.Timestamp, t1: pd.Timestamp, cfg: dict) -> str:
    """'overlaps', 'before' (ends before the next window opens) or 'after' (starts after one closed)."""
    if in_night_window(pd.date_range(t0, t1, freq="1min"), cfg).any():
        return "overlaps"
    a, b = (_hhmm(x) for x in cfg["night_window_ist"])
    s, e = (t.tz_convert(cfg["display_tz"]) for t in (t0, t1))
    until_start = (a - (e.hour * 60 + e.minute)) % 1440
    since_end = ((s.hour * 60 + s.minute) - b) % 1440
    return "before" if until_start < since_end else "after"


def night_intervals(t0: pd.Timestamp, t1: pd.Timestamp, cfg: dict) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Night-window intervals (UTC) intersecting [t0, t1], for plot shading."""
    tz = cfg["display_tz"]
    a, b = cfg["night_window_ist"]
    out = []
    for day in pd.date_range((t0.tz_convert(tz) - pd.Timedelta(days=1)).normalize(),
                             t1.tz_convert(tz).normalize(), freq="D"):
        start = day + pd.Timedelta(minutes=_hhmm(a))
        end = day + pd.Timedelta(minutes=_hhmm(b))
        if end <= start:
            end += pd.Timedelta(days=1)
        if end > t0 and start < t1:
            out.append((max(start, t0).tz_convert("UTC"), min(end, t1).tz_convert("UTC")))
    return out


# ---------------------------------------------------------------- tidy C/N0 observations

# Column order of the per-sample C/N0 table shared by both receivers.
OBS_COLS = ["time_utc", "receiver", "source", "constellation", "svid", "signal", "band",
            "elev_deg", "az_deg", "cn0_dbhz", "time_quality"]


def series_keys() -> list[str]:
    return ["receiver", "constellation", "svid", "signal"]


def circmean_deg(a) -> float:
    a = np.radians(np.asarray(a, dtype=float))
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


def add_minute(df: pd.DataFrame, col: str = "time_utc") -> pd.DataFrame:
    df["minute_utc"] = df[col].dt.floor("min")
    return df


def active_seconds(times: pd.Series) -> np.ndarray:
    """Sorted unique integer UTC seconds in which a receiver produced an epoch."""
    t = times.dropna()
    return np.unique(t.astype("int64").to_numpy() // 10**9) if len(t) else np.array([], dtype=np.int64)


def epoch_rate(times: pd.Series) -> float | None:
    """Epochs per active second (e.g. the L86 output rate actually received in a minute)."""
    t = times.dropna()
    if not len(t):
        return None
    ns = t.astype("int64").to_numpy()
    return round(len(np.unique(ns)) / len(np.unique(ns // 10**9)), 2)


def receiver_segments(active: np.ndarray, max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Start/end seconds of continuous receiver activity (gaps < max_gap_s don't split)."""
    if not len(active):
        return np.array([]), np.array([])
    brk = np.flatnonzero(np.diff(active) >= max_gap_s)
    starts = np.r_[active[0], active[brk + 1]]
    ends = np.r_[active[brk], active[-1]]
    return starts, ends


def silence_edge_zones(active: np.ndarray, silence_min_s: float, guard_s: float) -> np.ndarray:
    """(N, 2) array of [start, end] seconds within guard_s before a receiver silence
    (> silence_min_s with no epochs) or after the receiver resumes."""
    if len(active) < 2:
        return np.empty((0, 2))
    d = np.diff(active)
    i = np.flatnonzero(d > silence_min_s)
    before = np.c_[active[i] - guard_s, active[i] + 1]
    after = np.c_[active[i + 1], active[i + 1] + guard_s]
    return np.r_[before, after]


def in_zones(secs: np.ndarray, zones: np.ndarray) -> np.ndarray:
    """True where secs falls inside any [start, end] zone."""
    secs = np.asarray(secs, dtype=float)
    out = np.zeros(len(secs), dtype=bool)
    for a, b in zones:
        out |= (secs >= a) & (secs <= b)
    return out


def count_in(sorted_arr: np.ndarray, lo, hi) -> np.ndarray:
    """Number of elements of sorted_arr in [lo, hi] (vectorised)."""
    return np.searchsorted(sorted_arr, hi, side="right") - np.searchsorted(sorted_arr, lo, side="left")


# ---------------------------------------------------------------- output tables

def write_table(df: pd.DataFrame, out_dir: Path, name: str, session_id: str, tz: str,
                csv: bool = True) -> Path:
    """Write <name>.csv and <name>.parquet with session_id and paired *_utc / *_ist columns.

    CSV carries ISO-8601 strings (UTC with 'Z', IST with offset); Parquet keeps timestamps.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.insert(0, "session_id", session_id)
    for c in [c for c in df.columns if c.endswith("_utc")]:
        ist = c[:-4] + "_ist"
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            if ist not in df:
                df.insert(df.columns.get_loc(c) + 1, ist, df[c].dt.tz_convert(tz))
    if csv:
        text = df.copy()
        for c in text.columns:
            if pd.api.types.is_datetime64_any_dtype(text[c]):
                text[c] = iso_utc(text[c]) if c.endswith("_utc") else iso_local(text[c], tz)
        text.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.4f")
    pq = df.copy()
    for c in pq.columns:  # pyarrow needs homogeneous object columns
        if pq[c].dtype == object:
            pq[c] = pq[c].astype("string")
    pq.to_parquet(out_dir / f"{name}.parquet", index=False)
    return out_dir / f"{name}.csv"


# ---------------------------------------------------------------- geometry

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array([(n + h_m) * np.cos(lat) * np.cos(lon),
                     (n + h_m) * np.cos(lat) * np.sin(lon),
                     (n * (1 - WGS84_E2) + h_m) * np.sin(lat)])


def ecef_to_azel(sat_xyz: np.ndarray, lat_deg: float, lon_deg: float, h_m: float):
    """Azimuth/elevation (deg) of ECEF satellite positions (N x 3) seen from a geodetic site."""
    rx = geodetic_to_ecef(lat_deg, lon_deg, h_m)
    d = np.asarray(sat_xyz, dtype=float) - rx
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    e = -np.sin(lon) * d[:, 0] + np.cos(lon) * d[:, 1]
    n = (-np.sin(lat) * np.cos(lon) * d[:, 0] - np.sin(lat) * np.sin(lon) * d[:, 1]
         + np.cos(lat) * d[:, 2])
    u = np.cos(lat) * np.cos(lon) * d[:, 0] + np.cos(lat) * np.sin(lon) * d[:, 1] + np.sin(lat) * d[:, 2]
    az = np.degrees(np.arctan2(e, n)) % 360
    el = np.degrees(np.arctan2(u, np.hypot(e, n)))
    return az, el
