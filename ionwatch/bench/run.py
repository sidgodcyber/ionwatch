"""bench run: IGS ROTI over the session's night window, IONEX VTEC at Haldwani, Kp / F10.7,
our event times against IGS ROTI -> bench_summary.md / .json, CSVs and a plot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ionwatch.bench import ionex, rinex, roti, spaceweather
from ionwatch.proc import common


def session_info(processed: Path, cfg: dict) -> dict:
    """Date, data span and night window of a processed session (all from the data)."""
    s = json.loads((Path(processed) / "summary.json").read_text(encoding="utf-8"))
    t0, t1 = pd.Timestamp(s["session"]["start_utc"]), pd.Timestamp(s["session"]["end_utc"])
    wins = common.night_intervals(t0 - pd.Timedelta(days=1), t1 + pd.Timedelta(days=1), cfg)
    full = []
    for a, _ in wins:          # night_intervals clips to its range: rebuild full windows
        start = a.tz_convert(cfg["display_tz"]).normalize() + pd.Timedelta(
            minutes=common._hhmm(cfg["night_window_ist"][0]))
        if a.tz_convert(cfg["display_tz"]).hour < 12:
            start -= pd.Timedelta(days=1)
        end = start.normalize() + pd.Timedelta(days=1, minutes=common._hhmm(cfg["night_window_ist"][1]))
        full.append((start.tz_convert("UTC"), end.tz_convert("UTC")))
    full = sorted(set(full))
    overlap = [max(pd.Timedelta(0), min(b, t1) - max(a, t0)) for a, b in full]
    if max(overlap) > pd.Timedelta(0):
        night = full[int(np.argmax(overlap))]
    else:                      # session outside any night: the evening of the session's IST date
        day = t0.tz_convert(cfg["display_tz"]).normalize()
        night = next(w for w in full if w[0].tz_convert(cfg["display_tz"]).normalize() == day)
    dates = sorted({d.normalize() for d in pd.date_range(night[0], night[1] - pd.Timedelta(seconds=1), freq="1h")})
    return {"start_utc": t0, "end_utc": t1, "night": night, "dates": dates, "summary": s,
            "session_id": s.get("session_id")}


def our_events(processed: Path) -> pd.DataFrame:
    p = Path(processed) / "events_classified.csv"
    if not p.exists():
        return pd.DataFrame(columns=["start_utc", "end_utc", "satellite", "signal", "class", "receiver"])
    ev = pd.read_csv(p, parse_dates=["start_utc", "end_utc"])
    ev = ev[ev["class"] != "edge_of_gap"].copy()
    for c in ("start_utc", "end_utc"):
        ev[c] = pd.to_datetime(ev[c], utc=True)
    return ev


def orbit_check(processed: Path, nav: pd.DataFrame, cfg: dict) -> dict:
    """Broadcast-orbit elevation at Haldwani vs the elevations the phone reported (independent)."""
    p = Path(processed) / "tables" / "cn0_obs.parquet"
    if not p.exists() or nav.empty:
        return {"ok": None, "note": "no phone C/N0 table or no navigation file"}
    o = pd.read_parquet(p, columns=["time_utc", "receiver", "constellation", "svid", "elev_deg"])
    o = o[(o["receiver"] == "phone") & (o["constellation"] == "GPS") & o["elev_deg"].notna()]
    if o.empty:
        return {"ok": None, "note": "no phone GPS elevations"}
    o = o.assign(minute=o["time_utc"].dt.floor("5min")).groupby(["svid", "minute"])["elev_deg"].mean().reset_index()
    o = o.sample(min(len(o), 400), random_state=0)
    q = pd.DataFrame({"sat": [f"G{int(s):02d}" for s in o["svid"]], "time_utc": o["minute"].to_numpy()})
    loc = cfg["location"]
    el = roti.elevations(q, nav, (loc["lat_deg"], loc["lon_deg"], loc["height_m"]))
    d = np.abs(el - o["elev_deg"].to_numpy())
    d = d[np.isfinite(d)]
    med = float(np.median(d)) if len(d) else None
    return {"ok": med is not None and med < 1.0, "n": int(len(d)), "median_abs_diff_deg": round(med, 3) if med else None,
            "p95_abs_diff_deg": round(float(np.percentile(d, 95)), 3) if len(d) else None,
            "note": "GPS broadcast orbit (BRDM) elevation at Haldwani vs the phone's own Status elevations"}


def run(bench_dir: Path, processed: Path, cfg: dict) -> dict:
    b = cfg["bench"]
    info = session_info(processed, cfg)
    n0, n1 = info["night"]
    date = info["dates"][0]
    folder = Path(bench_dir) / f"{date:%Y}-{date.dayofyear:03d}"
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    items = man["items"]

    def path(name):
        r = items.get(name, {})
        return folder / r["file"] if r.get("ok") else None

    nav = rinex.read_gps_nav(path("nav_BRDM")) if path("nav_BRDM") else pd.DataFrame()
    stations = [s for s in [*b["primary_stations"], *b["backup_stations"]] if path(f"obs_{s}")]
    roti_all, series = [], {}
    for st in stations:
        h, o = rinex.read_obs(path(f"obs_{st}"), {"G": ["L1C", *roti.L2_TYPES]})
        r = roti.compute(o, h, nav, cfg).assign(station=st)
        roti_all.append(r)
        series[st] = roti.station_series(r)
        series[st].attrs["site"] = roti.ecef_to_geodetic(h["xyz"])
    roti_df = pd.concat(roti_all, ignore_index=True) if roti_all else pd.DataFrame()
    if len(roti_df):
        roti_df.to_csv(folder / "roti_5min.csv", index=False)

    def night_stats(st):
        s = series[st]
        s = s[(s.index >= n0) & (s.index < n1)]
        lat, lon, _ = series[st].attrs["site"]
        return {"station": st, "lat": round(lat, 3), "lon": round(lon, 3),
                "distance_km": round(_km(lat, lon, cfg["location"]["lat_deg"], cfg["location"]["lon_deg"])),
                "windows": int(len(s)), "roti_max": _r(s["roti_max"].max()), "roti_p95": _r(s["roti_max"].quantile(.95)),
                "roti_median": _r(s["roti_median"].median()),
                "windows_above": int((s["roti_max"] > b["roti_irregular"]).sum()),
                "first_above_ist": _ist(s.index[s["roti_max"] > b["roti_irregular"]].min(), cfg)}

    stats = [night_stats(st) for st in stations]
    ion_path = path("ionex")
    vtec = ionex.at(ionex.read(ion_path), cfg["location"]["lat_deg"], cfg["location"]["lon_deg"]) if ion_path else None
    if vtec is not None:
        vtec.to_frame().to_csv(folder / "ionex_vtec_haldwani.csv")
    kp = _kp(folder, items, date)
    f107 = _f107(folder, items, date)
    ev = our_events(processed)
    primary = next((st for st in stations if st in b["primary_stations"]), stations[0] if stations else None)
    ev_rows = []
    win = pd.Timedelta(minutes=b["event_match_min"])
    for _, e in ev.iterrows():
        row = {"start_ist": _ist(e["start_utc"], cfg), "satellite": e["satellite"], "signal": e["signal"],
               "class": e["class"], "receiver": e["receiver"]}
        for st in stations:
            s = series[st]["roti_max"]
            near = s[(s.index >= e["start_utc"] - win) & (s.index <= e["end_utc"] + win)]
            row[f"roti_{st}"] = _r(near.max()) if len(near) else None
        ev_rows.append(row)
    orbit = orbit_check(processed, nav, cfg)
    verdict = _verdict(stats, primary, ev_rows, cfg)
    kp_for_slides = _kp(folder, items, date)
    slides = for_slides(stats, primary, kp_for_slides, cfg)
    res = {"session": info["session_id"], "date_utc": f"{date:%Y-%m-%d}", "doy": int(date.dayofyear),
           "night_ist": [_ist(n0, cfg), _ist(n1, cfg)], "session_ist": [_ist(info["start_utc"], cfg), _ist(info["end_utc"], cfg)],
           "stations": stats, "primary_station": primary, "kp": kp, "f107": f107,
           "ionex": None if vtec is None else {
               "file": Path(ion_path).name, "producer": items["ionex"]["validation"]["info"].get("program"),
               "offset_removed_tecu": items["ionex"]["validation"]["info"].get("offset_tecu"),
               "night_range_tecu": [_r(vtec[(vtec.index >= n0) & (vtec.index <= n1)].min()),
                                    _r(vtec[(vtec.index >= n0) & (vtec.index <= n1)].max())],
               "day_max_tecu": _r(vtec.max()), "day_max_ist": _ist(vtec.idxmax(), cfg)},
           "for_slides": slides, "events": ev_rows, "verdict": verdict, "orbit_check": orbit, "cross_checks": man.get("cross_checks", []),
           "sources": {k: {"ok": v.get("ok"), "url": v.get("url"), "file": v.get("file"),
                           "sha256": (v.get("sha256") or "")[:12], "status": v.get("status")}
                       for k, v in items.items()}}
    plot(folder / "bench_roti_events.png", series, ev, vtec, info, kp, cfg)
    (folder / "bench_summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (folder / "bench_summary.md").write_text(markdown(res, cfg), encoding="utf-8")
    res["folder"] = str(folder)
    return res


CITY = {"LCK3": "Lucknow", "LCK4": "Lucknow", "IISC": "Bangalore", "HYDE": "Hyderabad"}


def for_slides(stats, primary, kp, cfg) -> str:
    """One sentence for a slide, built only from the numbers."""
    thr = cfg["bench"]["roti_irregular"]
    parts = []
    p = next((s for s in stats if s["station"] == primary), None)
    if p:
        state = "quiet" if p["windows_above"] == 0 else f"disturbed ({p['windows_above']} windows > {thr})"
        parts.append(f"IGS {p['station']} ({p['distance_km']} km) {state}, max ROTI {p['roti_max']:.2f} TECU/min")
    for s in stats:
        if s["station"] != primary and s["windows_above"] and s["first_above_ist"]:
            hhmm = s["first_above_ist"][11:16]
            when = "post-sunset irregularities" if int(hhmm[:2]) >= 18 else "irregularities"
            parts.append(f"{s['station']} ({CITY.get(s['station'], s['station'])}) saw {when} from {hhmm} IST")
    if kp:
        k = kp["max"]
        parts.append(f"Kp ≤ {int(k) if float(k).is_integer() else round(k, 1)}")
    return "; ".join(parts) + "." if parts else "No IGS benchmark data."


def _verdict(stats, primary, ev_rows, cfg) -> list[str]:
    """Plain statements from the numbers (no interpretation beyond them)."""
    thr = cfg["bench"]["roti_irregular"]
    out = []
    p = next((s for s in stats if s["station"] == primary), None)
    if p is None:
        return ["No IGS station available: no benchmark."]
    iono = [e for e in ev_rows if str(e["class"]).startswith("iono")]
    if p["windows_above"] == 0:
        out.append(f"No irregularities at the nearest IGS station {primary} ({p['distance_km']} km) during the night "
                   f"(max ROTI {p['roti_max']} < {thr} TECU/min): IGS does not support the {len(iono)} iono_* IonWatch "
                   "events; local causes remain the likelier explanation.")
    else:
        hit = [e for e in iono if (e.get(f"roti_{primary}") or 0) > thr]
        out.append(f"{primary} ({p['distance_km']} km) shows irregularities ({p['windows_above']} windows > {thr} TECU/min, "
                   f"first {p['first_above_ist']}); {len(hit)} of {len(iono)} iono_* IonWatch events fall within the "
                   "matching window.")
    for s in stats:
        if s["station"] != primary and s["windows_above"]:
            out.append(f"{s['station']} ({s['distance_km']} km away, {s['lat']}°N) saw irregularities "
                       f"(max ROTI {s['roti_max']}, from {s['first_above_ist']}): a low-latitude reference, "
                       "not evidence at Haldwani.")
    return out


def _r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _ist(t, cfg):
    return None if t is None or pd.isna(t) else f"{pd.Timestamp(t).tz_convert(cfg['display_tz']):%Y-%m-%d %H:%M} IST"


def _km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))


def _kp(folder, items, date):
    for name, reader in (("kp_gfz", spaceweather.kp_gfz), ("kp_swpc", spaceweather.kp_swpc)):
        if items.get(name, {}).get("ok"):
            s = spaceweather.for_day(reader(folder / items[name]["file"]), date)
            return {"source": "GFZ Potsdam" if name == "kp_gfz" else "NOAA SWPC (estimated Kp)",
                    "values": {f"{t:%H}": round(float(v), 2) for t, v in s.items()}, "max": float(s.max()),
                    "night_max": float(s[(s.index >= date + pd.Timedelta(hours=12))].max())}
    return None


def _f107(folder, items, date):
    for name, reader in (("f107_gfz", spaceweather.f107_gfz), ("f107_swpc", spaceweather.f107_swpc)):
        if items.get(name, {}).get("ok"):
            s = spaceweather.for_day(reader(folder / items[name]["file"]), date)
            return {"source": "GFZ (Fobs)" if name == "f107_gfz" else "NOAA SWPC (noon)", "sfu": float(s.mean())}
    return None


def plot(path, series, ev, vtec, info, kp, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from ionwatch.report import plots as P
    P._style()
    n0, n1 = info["night"]
    lo = min(n0, info["start_utc"]) - pd.Timedelta(hours=1)
    hi = max(n1, info["end_utc"]) + pd.Timedelta(hours=1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [3, 2]})
    for ax in (a1, a2):
        P._shade_night(ax, lo, hi, cfg)
        P._time_axis(ax, cfg)
    colors = [P.BLUE, P.ORANGE, P.AQUA, P.MUTED]
    handles = []
    for (st, s), c in zip(series.items(), colors):
        s = s[(s.index >= lo) & (s.index <= hi)]
        a1.plot(s.index + pd.Timedelta(minutes=2.5), s["roti_max"], "-", color=c, lw=1.4)
        handles.append(Line2D([], [], color=c, lw=2, label=f"{st} ({series[st].attrs['site'][0]:.1f}°N) max ROTI"))
    a1.axhline(cfg["bench"]["roti_irregular"], color=P.INK2, ls="--", lw=0.8)
    ymax = max([1.0] + [float(s["roti_max"].max()) * 1.1 for s in series.values() if len(s)])
    a1.set_ylim(0, ymax)
    a1.plot([info["start_utc"], info["end_utc"]], [ymax * 0.87] * 2, "-", color=P.INK2, lw=3, solid_capstyle="butt")
    for _, e in ev.iterrows():
        col, mk = P.CLASS_STYLE.get(e["class"], (P.MUTED, "o"))
        a1.plot([e["start_utc"]], [ymax * 0.95], mk, color=col, ms=7, mec=P.SURFACE)
    handles += [Line2D([], [], color=P.INK2, ls="--", lw=0.8, label=f"{cfg['bench']['roti_irregular']} TECU/min"),
                Line2D([], [], color=P.INK2, lw=3, label="IonWatch session (data span)"),
                Patch(color=P.NIGHT_WASH, label="night window 19:30\u201301:00 IST")]
    present = sorted(set(ev["class"])) if len(ev) else []
    handles += [Line2D([], [], color=P.CLASS_STYLE[c][0], marker=P.CLASS_STYLE[c][1], ls="", ms=7,
                       label=f"our event: {c}") for c in present if c in P.CLASS_STYLE]
    a1.set_ylabel("ROTI (TECU/min)")
    if vtec is not None:
        v = vtec[(vtec.index >= lo - pd.Timedelta(hours=1)) & (vtec.index <= hi + pd.Timedelta(hours=1))]
        a2.plot(v.index, v.values, "o-", color=P.BLUE, ms=3, lw=1.4)
    a2.set_ylabel("GIM VTEC at\nHaldwani (TECU)")
    a2.set_xlim(lo, hi)
    a2.set_xlabel("Time (IST)")
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
    kpt = f"Kp max {kp['max']:.1f} ({kp['source']})" if kp else "Kp unavailable"
    fig.suptitle(f"IGS ROTI vs IonWatch events, {info['dates'][0]:%Y-%m-%d} (DOY {info['dates'][0].dayofyear}) · {kpt}",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown(r: dict, cfg) -> str:
    b = cfg["bench"]
    L = [f"# IGS benchmark for session {r['session']}", "", f"**For slides:** {r['for_slides']}", "",
         f"**Date:** {r['date_utc']} UTC (DOY {r['doy']}) · **night window:** {r['night_ist'][0]} → {r['night_ist'][1][11:]} · "
         f"**session (from data):** {r['session_ist'][0]} → {r['session_ist'][1][11:]}", ""]
    kp, f = r["kp"], r["f107"]
    L.append(f"**Geomagnetic / solar:** Kp max {kp['max']:.2f}, night max {kp['night_max']:.2f} ({kp['source']}); "
             f"3-hourly {kp['values']}" if kp else "**Kp:** not available")
    L.append(f"F10.7 {f['sfu']:.1f} sfu ({f['source']})." if f else "F10.7: not available.")
    L += ["", "## IGS ROTI over the night window", "",
          f"GPS L1/L2 carrier phase, 30 s, elevation ≥ {b['elev_mask_deg']}°, 5-min ROTI; "
          f"{b['roti_irregular']} TECU/min marks irregularities.", "",
          "| station | position | from Haldwani | windows | max ROTI | p95 | median | windows > threshold | first above (IST) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in r["stations"]:
        L.append(f"| {s['station']} | {s['lat']}°N {s['lon']}°E | {s['distance_km']} km | {s['windows']} | {s['roti_max']} | "
                 f"{s['roti_p95']} | {s['roti_median']} | {s['windows_above']} | {s['first_above_ist'] or '–'} |")
    if r["ionex"]:
        i = r["ionex"]
        L += ["", "## GIM VTEC at Haldwani (29.26°N, 79.54°E)",
              f"- {i['file']} ({i['producer']}); constant offset removed: {i['offset_removed_tecu']} TECU",
              f"- night range {i['night_range_tecu'][0]} … {i['night_range_tecu'][1]} TECU; "
              f"day maximum {i['day_max_tecu']} TECU at {i['day_max_ist']}"]
    L += ["", "## Verdict", ""] + [f"- {v}" for v in r["verdict"]]
    L += ["", "## IonWatch events against IGS ROTI", "",
          f"IGS ROTI = maximum over satellites in the 5-min windows within ±{b['event_match_min']} min of the event.", ""]
    sts = [s["station"] for s in r["stations"]]
    if r["events"]:
        ev = pd.DataFrame(r["events"])
        L += ["| class | events | " + " | ".join(f"median ROTI {s}" for s in sts) + " |", "|---|---|" + "---|" * len(sts)]
        for c, g in ev.groupby("class"):
            L.append(f"| {c} | {len(g)} | " + " | ".join(
                str(_r(pd.to_numeric(g[f'roti_{s}'], errors='coerce').median())) for s in sts) + " |")
        iono = ev[ev["class"].astype(str).str.startswith("iono")]
        if len(iono):
            L += ["", "iono_* events:", "",
                  "| start (IST) | satellite | signal | class | " + " | ".join(f"ROTI {s}" for s in sts) + " |",
                  "|---|---|---|---|" + "---|" * len(sts)]
            L += [f"| {e['start_ist'][11:16]} | {e['satellite']} | {e['signal']} | {e['class']} | " +
                  " | ".join(str(e.get(f'roti_{s}')) for s in sts) + " |" for e in iono.to_dict("records")]
    else:
        L.append("No IonWatch events (other than edge_of_gap) in this session.")
    L += ["", "![IGS ROTI vs IonWatch events](bench_roti_events.png)", "", "## Source checks", ""]
    o = r["orbit_check"]
    L.append(f"- Orbit check: broadcast-orbit elevation vs the phone's own elevations at Haldwani: "
             f"median |Δ| {o.get('median_abs_diff_deg')}°, p95 {o.get('p95_abs_diff_deg')}° over {o.get('n')} samples "
             f"→ {'OK' if o.get('ok') else 'CHECK' if o.get('ok') is False else 'n/a'}")
    for c in r["cross_checks"]:
        verdict = "OK" if c.get("ok") else "CHECK" if c.get("ok") is False else "not possible"
        extra = {k: v for k, v in c.items() if k not in ("check", "ok", "note")}
        L.append(f"- {c['check']}: **{verdict}** {extra if extra else ''} — {c.get('note', '')}")
    L += ["", "| item | used | file | sha256 | status |", "|---|---|---|---|---|"]
    for k, v in r["sources"].items():
        status = "ok" if v["ok"] else str(v["status"]).split(" (Caused")[0].split("HTTPSConnectionPool")[0][:70]
        L.append(f"| {k} | {'yes' if v['ok'] else 'no'} | {v['file'] or '–'} | {v['sha256'] or '–'} | {status} |")
    missing = [k for k, v in r["sources"].items() if not v["ok"]]
    if missing:
        L += ["", "**To complete elsewhere:** " + ", ".join(missing) +
              " (CDDIS items need an Earthdata login in ~/.netrc; GIMs from other producers appear 1–2 days later; "
              "re-run `bench fetch` then `bench run`)."]
    near = min(r["stations"], key=lambda s: s["distance_km"]) if r["stations"] else None
    L += ["", f"_{near['station'] if near else 'The IGS station'} is ~{near['distance_km'] if near else '?'} km from "
          "Haldwani: its ROTI is a regional reference, not the same ionospheric pierce points. Our classes are "
          "heuristic triage; the benchmark shows whether the region was disturbed._"]
    return "\n".join(L) + "\n"
