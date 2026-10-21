"""Minimal RINEX 3 readers for the benchmark: Hatanaka (CRINEX 3) decompression in pure Python,
observation files (selected observation types), and GPS broadcast ephemerides.

The CRINEX decoder follows Hatanaka (2008): epoch lines and flag strings are text-differenced
against the previous ones (' ' = unchanged, '&' = blank), observations are arc-differenced
integers in units of 0.001 ("n&value" starts an arc of order n). It is checked against the
reference crx2rnx (RNXCMP) in tests/test_bench.py.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

GPS_EPOCH = pd.Timestamp("1980-01-06T00:00:00Z")
GPS_UTC_LEAP_S = 18                      # GPS - UTC since 2017


# ------------------------------------------------------------------ files

def open_text(path: str | Path) -> list[str]:
    """Lines of a .gz / .Z / plain text file."""
    data = Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    elif data[:2] == b"\x1f\x9d":
        import ncompress
        data = ncompress.decompress(data)
    return data.decode("ascii", errors="replace").splitlines()


def _repair(ref: str, diff: str) -> str:
    """Text-difference repair: ' ' keeps the reference character, '&' writes a blank."""
    out = list(ref.ljust(len(diff)))
    for i, c in enumerate(diff):
        if c == " ":
            continue
        out[i] = " " if c == "&" else c
    return "".join(out)


class _Arc:
    __slots__ = ("order", "d")

    def __init__(self, order: int, value: int):
        self.order, self.d = order, [value]

    def step(self, diff: int) -> int:
        m = min(len(self.d), self.order)
        new = [0] * (m + 1)
        new[m] = diff
        for k in range(m - 1, -1, -1):
            new[k] = self.d[k] + new[k + 1]
        self.d = new
        return new[0]


def _field(state: dict, key, text: str):
    """Decode one compressed field; returns the integer value or None (missing)."""
    if text == "":
        state.pop(key, None)
        return None
    if "&" in text:
        n, v = text.split("&", 1)
        arc = _Arc(int(n), int(v))
        state[key] = arc
        return arc.d[0]
    arc = state.get(key)
    if arc is None:                      # difference without an initialised arc: corrupt, skip
        return None
    return arc.step(int(text))


def crx_to_rnx(lines: list[str]) -> list[str]:
    """CRINEX 3 lines -> RINEX 3 observation lines (header copied, data reconstructed)."""
    if not lines[0].startswith("3."):
        raise ValueError("only CRINEX version 3 is supported")
    out, i = [], 2                       # skip the two CRINEX header lines
    ntypes: dict[str, int] = {}
    cur_sys = None
    while True:
        line = lines[i]
        i += 1
        out.append(line)
        if line[60:73] == "SYS / # / OBS":
            if line[0] != " ":
                cur_sys = line[0]
                ntypes[cur_sys] = int(line[3:6])
        if "END OF HEADER" in line[60:]:
            break
    ref_epoch = ""
    clock_state: dict = {}
    data_state: dict = {}
    flags_prev: dict[str, str] = {}
    prev_sats: set = set()
    n = len(lines)
    while i < n:
        line = lines[i]
        i += 1
        epoch = line if line.startswith(">") else _repair(ref_epoch, line)
        ref_epoch = epoch
        flag = epoch[31:32]
        if flag.strip() and int(flag) > 1:                     # special event: records copied as they are
            k = int(epoch[32:35])
            out.append(epoch[:41].rstrip())
            out.extend(lines[i:i + k])
            i += k
            ref_epoch = ""
            continue
        nsat = int(epoch[32:35])
        clock_line = lines[i] if i < n else ""
        i += 1
        clock = _field(clock_state, "clk", clock_line.strip()) if clock_line.strip() else None
        if clock is None:
            clock_state.clear()
        head = epoch[:35]
        out.append(head + (f"{' ' * 6}{clock / 1e12:15.12f}" if clock is not None else ""))
        sats = [epoch[41 + 3 * k:44 + 3 * k] for k in range(nsat)]
        cur_flags = {}
        for sat in sats:
            dl = lines[i] if i < n else ""
            i += 1
            nt = ntypes.get(sat[0], 0)
            parts = dl.split(" ", nt)
            fields = parts[:nt] + [""] * max(0, nt - len(parts))
            fdiff = parts[nt] if len(parts) > nt else ""
            if sat not in prev_sats:                          # new satellite: no reference state
                for k in [k for k in data_state if k[0] == sat]:
                    del data_state[k]
            ref_flags = flags_prev.get(sat, "") if sat in prev_sats else ""
            flags = _repair(ref_flags, fdiff)
            cur_flags[sat] = flags
            vals = []
            for j in range(nt):
                v = _field(data_state, (sat, j), fields[j])
                lli = flags[2 * j] if 2 * j < len(flags) else " "
                ssi = flags[2 * j + 1] if 2 * j + 1 < len(flags) else " "
                vals.append((" " * 14 if v is None else f"{v / 1000:14.3f}") + lli + ssi)
            out.append((sat + "".join(vals)).rstrip())
        flags_prev = cur_flags
        prev_sats = set(sats)
    return out


# ------------------------------------------------------------------ observation files

def _header(lines: list[str]) -> tuple[dict, int]:
    h: dict = {"obs_types": {}}
    cur = None
    for k, line in enumerate(lines):
        label = line[60:].strip()
        if label == "SYS / # / OBS TYPES":
            if line[0] != " ":
                cur = line[0]
                h["obs_types"][cur] = []
            h["obs_types"][cur] += line[7:58].split()
        elif label == "MARKER NAME":
            h["marker"] = line[:60].strip()
        elif label == "APPROX POSITION XYZ":
            h["xyz"] = [float(x) for x in line[:42].split()]
        elif label == "INTERVAL":
            h["interval_s"] = float(line[:10])
        elif label == "TIME OF FIRST OBS":
            h["first_obs"] = line[:43].split()
            h["time_system"] = line[48:51].strip() or "GPS"
        elif label == "RINEX VERSION / TYPE":
            h["version"] = line[:9].strip()
        elif label == "END OF HEADER":
            return h, k + 1
    raise ValueError("no END OF HEADER")


def read_obs(path: str | Path, keep: dict[str, list[str]]) -> tuple[dict, pd.DataFrame]:
    """Read a RINEX 3 observation file (.rnx/.crx, optionally .gz/.Z).

    keep: {system letter: [obs types]} e.g. {"G": ["L1C", "L2W"]}. Returns (header, table with
    columns time_gps, sat, <type>, <type>_lli ...). time_gps is the receiver epoch (GPS time).
    """
    lines = open_text(path)
    if lines[0].startswith(("3.0", "1.0")) and "CRINEX" in lines[0][60:]:
        lines = crx_to_rnx(lines)
    h, i = _header(lines)
    idx = {s: {t: h["obs_types"][s].index(t) for t in types if t in h["obs_types"].get(s, [])}
           for s, types in keep.items()}
    rows = []
    t = None
    n = len(lines)
    while i < n:
        line = lines[i]
        i += 1
        if not line.startswith(">"):
            continue
        flag = int(line[31:32] or 0)
        ns = int(line[32:35])
        if flag > 1:
            i += ns
            continue
        t = pd.Timestamp(year=int(line[2:6]), month=int(line[7:9]), day=int(line[10:12]),
                         hour=int(line[13:15]), minute=int(line[16:18]), tz="UTC") + \
            pd.Timedelta(seconds=float(line[19:29]))
        for _ in range(ns):
            sl = lines[i]
            i += 1
            s = sl[0]
            if s not in idx:
                continue
            rec = {"time_gps": t, "sat": sl[:3].replace(" ", "0")}
            for typ, j in idx[s].items():
                a = 3 + 16 * j
                txt = sl[a:a + 14]
                rec[typ] = float(txt) if txt.strip() else np.nan
                rec[typ + "_lli"] = int(sl[a + 14]) if len(sl) > a + 14 and sl[a + 14].isdigit() else 0
            rows.append(rec)
    return h, pd.DataFrame(rows)


# ------------------------------------------------------------------ GPS broadcast ephemerides

def _vals(s: str) -> list[float]:
    """Fixed-width D19.12 values (4 per line after the first column)."""
    s = s.replace("D", "E").replace("d", "e")
    out = []
    for k in range(0, len(s), 19):
        chunk = s[k:k + 19].strip()
        out.append(float(chunk) if chunk else 0.0)
    return out


def read_gps_nav(path: str | Path) -> pd.DataFrame:
    """GPS ephemerides from a RINEX 3 navigation file (BRDM / BRDC mixed)."""
    lines = open_text(path)
    k = next(i for i, l in enumerate(lines) if "END OF HEADER" in l) + 1
    recs = []
    while k < len(lines):
        line = lines[k]
        if line.startswith("G") and len(line) > 22:
            v = _vals(line[23:].ljust(57)) + sum((_vals(lines[k + j][4:80].ljust(76)) for j in range(1, 8)), [])
            toc = pd.Timestamp(year=int(line[4:8]), month=int(line[9:11]), day=int(line[12:14]),
                               hour=int(line[15:17]), minute=int(line[18:20]), second=int(line[21:23]), tz="UTC")
            names = ["af0", "af1", "af2", "iode", "crs", "dn", "m0", "cuc", "e", "cus", "sqrta", "toe", "cic",
                     "omega0", "cis", "i0", "crc", "omega", "omegadot", "idot", "l2codes", "week", "l2p",
                     "sva", "health", "tgd", "iodc", "ttm", "fit"]
            rec = dict(zip(names, v[:len(names)]))
            rec.update(sat=line[:3].replace(" ", "0"), toc=toc)
            recs.append(rec)
            k += 8
        else:
            k += 1
    return pd.DataFrame(recs)


MU = 3.986005e14
OMEGA_E = 7.2921151467e-5


def gps_position(eph: pd.Series, t_gps_sow: np.ndarray) -> np.ndarray:
    """ECEF (m) of a GPS satellite at GPS seconds-of-week t (IS-GPS-200 user algorithm)."""
    a = eph["sqrta"] ** 2
    n = np.sqrt(MU / a ** 3) + eph["dn"]
    tk = t_gps_sow - eph["toe"]
    tk = np.where(tk > 302400, tk - 604800, np.where(tk < -302400, tk + 604800, tk))
    m = eph["m0"] + n * tk
    e = eph["e"]
    E = m.copy()
    for _ in range(10):
        E = m + e * np.sin(E)
    v = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(E), np.cos(E) - e)
    phi = v + eph["omega"]
    du = eph["cus"] * np.sin(2 * phi) + eph["cuc"] * np.cos(2 * phi)
    dr = eph["crs"] * np.sin(2 * phi) + eph["crc"] * np.cos(2 * phi)
    di = eph["cis"] * np.sin(2 * phi) + eph["cic"] * np.cos(2 * phi)
    u, r = phi + du, a * (1 - e * np.cos(E)) + dr
    i = eph["i0"] + di + eph["idot"] * tk
    x, y = r * np.cos(u), r * np.sin(u)
    om = eph["omega0"] + (eph["omegadot"] - OMEGA_E) * tk - OMEGA_E * eph["toe"]
    return np.c_[x * np.cos(om) - y * np.cos(i) * np.sin(om), x * np.sin(om) + y * np.cos(i) * np.cos(om),
                 y * np.sin(i)]


def gps_sow(t: pd.Series) -> np.ndarray:
    return ((t - GPS_EPOCH).dt.total_seconds().to_numpy()) % 604800
