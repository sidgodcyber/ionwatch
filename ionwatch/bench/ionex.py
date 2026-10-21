"""IONEX reader: VTEC maps, interpolation at a site, and the satellite DCB aux block."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ionwatch.bench.rinex import open_text


def read(path: str | Path) -> dict:
    """{'times': [Timestamp], 'lats', 'lons', 'tec': array (t, lat, lon) in TECU, 'offset_tecu',
    'dcb_ns': {prn: ns}, 'header': {...}}. Missing grid values (9999) become NaN.

    Some producers add a constant to every grid point (JPL: "a 2 TECU offset"); it is detected
    from the header text and reported as offset_tecu (not removed here)."""
    lines = open_text(path)
    h: dict = {}
    dcb: dict[str, float] = {}
    k = 0
    text = []
    while k < len(lines):
        line = lines[k]
        label = line[60:].strip()
        if label in ("DESCRIPTION", "COMMENT"):
            text.append(line[:60])
        elif label == "LAT1 / LAT2 / DLAT":
            h["lat"] = [float(x) for x in line[:60].split()[:3]]
        elif label == "LON1 / LON2 / DLON":
            h["lon"] = [float(x) for x in line[:60].split()[:3]]
        elif label == "EXPONENT":
            h["exponent"] = int(line[:6])
        elif label == "PGM / RUN BY / DATE":
            h["program"] = line[:60].strip()
        elif label == "PRN / BIAS / RMS":
            f = line[:60].split()
            dcb[f[0] if not f[0].isdigit() else f"G{int(f[0]):02d}"] = float(f[1])
        elif label == "END OF HEADER":
            k += 1
            break
        k += 1
    blob = " ".join(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*TECU\s*offset", blob, re.I)
    offset = float(m.group(1)) if m else 0.0
    lat1, lat2, dlat = h["lat"]
    lon1, lon2, dlon = h["lon"]
    lats = np.arange(lat1, lat2 + dlat / 2, dlat) if dlat > 0 else np.arange(lat1, lat2 + dlat / 2, dlat)
    lons = np.arange(lon1, lon2 + dlon / 2, dlon)
    scale = 10.0 ** h.get("exponent", -1)
    times, maps = [], []
    nper = math.ceil(len(lons) / 16)
    while k < len(lines):
        line = lines[k]
        label = line[60:].strip()
        if label == "START OF TEC MAP":
            grid = np.full((len(lats), len(lons)), np.nan)
            t = None
            k += 1
            while lines[k][60:].strip() != "END OF TEC MAP":
                l2 = lines[k]
                lb = l2[60:].strip()
                if lb == "EPOCH OF CURRENT MAP":
                    y, mo, d, hh, mi, ss = (int(x) for x in l2[:36].split())
                    t = pd.Timestamp(year=y, month=mo, day=d, hour=hh, minute=mi, second=ss, tz="UTC")
                elif lb == "EXPONENT":
                    scale = 10.0 ** int(l2[:6])
                elif lb == "LAT/LON1/LON2/DLON/H":
                    la = float(l2[2:8])
                    row = int(round((la - lat1) / dlat))
                    vals = []
                    for j in range(nper):
                        s = lines[k + 1 + j]
                        vals += [int(s[p:p + 5]) for p in range(0, min(len(s), 80), 5) if s[p:p + 5].strip()]
                    k += nper
                    v = np.array(vals[:len(lons)], dtype=float)
                    v[v == 9999] = np.nan
                    grid[row, :len(v)] = v * scale
                k += 1
            times.append(t)
            maps.append(grid)
        k += 1
    return {"times": times, "lats": lats, "lons": lons, "tec": np.array(maps), "offset_tecu": offset,
            "dcb_ns": dcb, "header": h}


def at(ion: dict, lat: float, lon: float, remove_offset: bool = True) -> pd.Series:
    """VTEC (TECU) at a site for every map, bilinear in latitude/longitude."""
    lats, lons, tec = ion["lats"], ion["lons"], ion["tec"]
    order = np.argsort(lats)
    la, tecs = lats[order], tec[:, order, :]
    i = np.clip(np.searchsorted(la, lat) - 1, 0, len(la) - 2)
    j = np.clip(np.searchsorted(lons, lon) - 1, 0, len(lons) - 2)
    wy = (lat - la[i]) / (la[i + 1] - la[i])
    wx = (lon - lons[j]) / (lons[j + 1] - lons[j])
    v = ((1 - wy) * (1 - wx) * tecs[:, i, j] + (1 - wy) * wx * tecs[:, i, j + 1]
         + wy * (1 - wx) * tecs[:, i + 1, j] + wy * wx * tecs[:, i + 1, j + 1])
    if remove_offset:
        v = v - ion["offset_tecu"]
    return pd.Series(v, index=pd.DatetimeIndex(ion["times"]), name="vtec_tecu")
