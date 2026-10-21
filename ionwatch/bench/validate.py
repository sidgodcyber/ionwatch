"""Content checks for every benchmark file, and cross-checks between independent sources."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ionwatch.bench import ionex, rinex, spaceweather

HALDWANI = (29.256, 79.537)


def _result(problems: list[str], **info) -> dict:
    return {"ok": not problems, "problems": problems, "info": info}


def check(kind: str, path: Path, date: pd.Timestamp, name: str = "") -> dict:
    try:
        return {"obs": _obs, "nav": _nav, "ionex": _ionex, "bias": _bias, "kp": _kp, "f107": _f107}[kind](
            Path(path), date, name)
    except Exception as e:                                   # unreadable / truncated / not what we expected
        return _result([f"unreadable: {type(e).__name__}: {e}"[:200]])


def _obs(path, date, name):
    lines = rinex.open_text(path)
    if "CRINEX" in lines[0][60:]:
        lines = rinex.crx_to_rnx(lines)
    h, i = rinex._header(lines)
    st = name.split("_")[-1]
    first = pd.Timestamp(f"{h['first_obs'][0]}-{int(h['first_obs'][1]):02d}-{int(h['first_obs'][2]):02d}", tz="UTC")
    epochs = sum(1 for line in lines[i:] if line.startswith(">"))
    gps = h["obs_types"].get("G", [])
    probs = []
    if not h.get("marker", "").upper().startswith(st):
        probs.append(f"marker {h.get('marker')} is not {st}")
    if first.normalize() != date.normalize():
        probs.append(f"first epoch {first.date()} is not {date.date()}")
    if "L1C" not in gps or not ({"L2W", "L2X", "L2L"} & set(gps)):
        probs.append(f"no GPS L1C + L2 phase in {gps}")
    expected = 86400 / h.get("interval_s", 30)
    if epochs < 0.5 * expected:
        probs.append(f"only {epochs} of {expected:.0f} epochs")
    return _result(probs, marker=h.get("marker"), epochs=epochs, interval_s=h.get("interval_s"),
                   gps_types=gps, xyz=h.get("xyz"), version=h.get("version"))


def _nav(path, date, name):
    nav = rinex.read_gps_nav(path)
    sats = nav["sat"].nunique() if len(nav) else 0
    probs = [] if sats >= 24 else [f"only {sats} GPS satellites"]
    if len(nav) and not ((nav["toc"] >= date - pd.Timedelta(days=1)) & (nav["toc"] <= date + pd.Timedelta(days=2))).all():
        probs.append("ephemeris epochs outside the day")
    return _result(probs, gps_sats=int(sats), records=int(len(nav)))


def _ionex(path, date, name):
    d = ionex.read(path)
    probs = []
    if len(d["times"]) < 12:
        probs.append(f"only {len(d['times'])} maps")
    if d["times"] and (d["times"][0].normalize() != date.normalize()):
        probs.append(f"first map {d['times'][0]} not on {date.date()}")
    v = ionex.at(d, *HALDWANI)
    if v.isna().all() or v.max() > 300 or v.min() < -5:
        probs.append(f"implausible VTEC at Haldwani {v.min():.1f}..{v.max():.1f}")
    return _result(probs, program=d["header"].get("program"), maps=len(d["times"]),
                   grid=[float(d["lats"][1] - d["lats"][0]), float(d["lons"][1] - d["lons"][0])],
                   offset_tecu=d["offset_tecu"], sat_dcb=len(d["dcb_ns"]),
                   vtec_haldwani_range=[round(float(v.min()), 1), round(float(v.max()), 1)])


def _bias(path, date, name):
    text = "\n".join(rinex.open_text(path)[:400])
    n_g = sum(1 for line in rinex.open_text(path) if line.startswith((" DSB  G", " OSB  G")))
    probs = [] if ("%=BIA" in text and n_g > 0) else ["not a SINEX-BIAS file with GPS biases"]
    return _result(probs, gps_bias_records=n_g)


def _kp(path, date, name):
    s = spaceweather.kp_gfz(path) if "gfz" in name else spaceweather.kp_swpc(path)
    day = spaceweather.for_day(s, date)
    probs = []
    if len(day) < 8:
        probs.append(f"{len(day)} of 8 three-hour Kp values for {date.date()}"
                     + (" (SWPC JSON only covers the last ~7 days)" if "swpc" in name else ""))
    if len(day) and not day.between(0, 9).all():
        probs.append("Kp outside 0..9")
    return _result(probs, values=[round(float(x), 2) for x in day], max=float(day.max()) if len(day) else None)


def _f107(path, date, name):
    s = spaceweather.f107_gfz(path) if "gfz" in name else spaceweather.f107_swpc(path)
    day = spaceweather.for_day(s, date)
    probs = [] if len(day) else [f"no F10.7 for {date.date()}" + (" (SWPC JSON covers ~1 month)" if "swpc" in name else "")]
    if len(day) and not day.between(50, 400).all():
        probs.append("F10.7 outside 50..400 sfu")
    return _result(probs, values=[round(float(x), 1) for x in day])


def cross_checks(folder: Path, manifest: dict, date: pd.Timestamp) -> list[dict]:
    """Independent sources against each other; each entry says what was compared and the verdict."""
    folder = Path(folder)
    items = manifest["items"]
    out = []

    def f(name):
        r = items.get(name, {})
        return folder / r["file"] if r.get("ok") else None

    if f("kp_swpc") and f("kp_gfz"):
        a = spaceweather.for_day(spaceweather.kp_swpc(f("kp_swpc")), date)
        b = spaceweather.for_day(spaceweather.kp_gfz(f("kp_gfz")), date)
        j = a.to_frame("swpc").join(b.to_frame("gfz"), how="inner")
        dmax = float((j["swpc"] - j["gfz"]).abs().max()) if len(j) else None
        out.append({"check": "Kp SWPC vs GFZ (3-hourly)", "n": len(j), "max_abs_diff": dmax,
                    "ok": dmax is not None and dmax <= 0.67 + 1e-6,
                    "note": "SWPC = NOAA estimated Kp, GFZ = official (preliminary) Kp: ≤ 2/3 Kp unit expected"})
    else:
        out.append({"check": "Kp SWPC vs GFZ", "ok": None, "note": "one source missing"})
    if f("f107_swpc") and f("f107_gfz"):
        a = spaceweather.for_day(spaceweather.f107_swpc(f("f107_swpc")), date)
        b = spaceweather.for_day(spaceweather.f107_gfz(f("f107_gfz")), date)
        d = abs(float(a.mean()) - float(b.mean())) if len(a) and len(b) else None
        out.append({"check": "F10.7 SWPC (noon) vs GFZ Fobs", "swpc": round(float(a.mean()), 1) if len(a) else None,
                    "gfz": round(float(b.mean()), 1) if len(b) else None, "abs_diff": d,
                    "ok": d is not None and d <= 5, "note": "same Penticton flux; ≤ 5 sfu expected"})
    else:
        out.append({"check": "F10.7 SWPC vs GFZ", "ok": None, "note": "one source missing"})
    if f("ionex") and f("ionex_check"):
        a = ionex.at(ionex.read(f("ionex")), *HALDWANI)
        b = ionex.at(ionex.read(f("ionex_check")), *HALDWANI)
        j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
        diff = j["a"] - j["b"]
        corr = float(j.corr().iloc[0, 1]) if len(j) > 2 else None
        out.append({"check": f"GIM {Path(items['ionex']['file']).name} vs {Path(items['ionex_check']['file']).name} "
                             "at Haldwani", "n": len(j), "mean_diff_tecu": round(float(diff.mean()), 2),
                    "rms_tecu": round(float(np.sqrt((diff ** 2).mean())), 2), "corr": round(corr, 3) if corr else None,
                    "ok": corr is not None and corr > 0.9 and abs(diff.mean()) < 8,
                    "note": "independent GIM producers; corr > 0.9 and |mean| < 8 TECU expected"})
    else:
        out.append({"check": "GIM vs a second producer", "ok": None,
                    "note": "only one GIM producer available for this day (others publish 1-2 days later)"})
    stations = [n for n, r in items.items() if n.startswith("obs_") and r.get("ok")]
    out.append({"check": "IGS stations downloaded and valid", "stations": [s[4:] for s in stations],
                "ok": bool(stations),
                "note": "LCK3/HYDE are only on CDDIS (login); LCK4/IISC also on BKG"})
    return out
