"""Kp and F10.7 from NOAA SWPC and GFZ Potsdam JSON files (as saved by bench fetch)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def kp_swpc(path: Path) -> pd.Series:
    """SWPC noaa-planetary-k-index.json: a list of {time_tag, Kp, ...} (older format: list of lists)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d and isinstance(d[0], list):                       # [["time_tag","Kp",...], [...], ...]
        hdr, rows = d[0], d[1:]
        d = [dict(zip(hdr, r)) for r in rows]
    s = pd.Series({pd.Timestamp(r["time_tag"]).tz_localize("UTC"): float(r["Kp"]) for r in d})
    return s.sort_index()


def kp_gfz(path: Path) -> pd.Series:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.Series([float(x) for x in d["Kp"]], index=pd.to_datetime(d["datetime"], utc=True)).sort_index()


def f107_swpc(path: Path) -> pd.Series:
    """SWPC f107_cm_flux.json: several reports a day; keep the Noon (local Penticton) value."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [r for r in d if r.get("flux") is not None]
    noon = [r for r in rows if str(r.get("reporting_schedule", "")).lower() == "noon"] or rows
    s = pd.Series({pd.Timestamp(r["time_tag"]).tz_localize("UTC"): float(r["flux"]) for r in noon})
    return s.sort_index()


def f107_gfz(path: Path) -> pd.Series:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.Series([float(x) for x in d["Fobs"]], index=pd.to_datetime(d["datetime"], utc=True)).sort_index()


def for_day(s: pd.Series, date: pd.Timestamp) -> pd.Series:
    day = pd.Timestamp(date).normalize()
    return s[(s.index >= day) & (s.index < day + pd.Timedelta(days=1))]
