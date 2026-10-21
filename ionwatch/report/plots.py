"""PNG plots for one processed session (x-axis in IST).

Encoding conventions, shared by every figure:
  * colour = band role (L1-side blue, L5 orange, B1I aqua) or source; at most three hues
    per axes (the all-pairs-safe slots of the reference palette); more series -> facets;
  * marker = source: Raw '.', Status 'x', L86 NMEA '+';
  * grey wash = night window (19:30-01:00 IST), warm wash = fade event / local-cause minute.
Every figure with >= 2 series has a legend; the CSV tables are the table view.
"""
from __future__ import annotations

import math
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ionwatch.proc import common, tec_code  # noqa: E402

# Reference palette (light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIOUS = "#ec835a"
NIGHT_WASH = "#e1e0d9"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

MARKER = {"raw": ".", "status": "x", "nmea": "+"}
SOURCE_COLOR = {"raw": BLUE, "status": ORANGE, "nmea": AQUA}
CLASS_STYLE = {"iono_likely": (ORANGE, "D"), "iono_possible": (BLUE, "o"),
               "low_elev": (AQUA, "^"), "local": (MUTED, "s"), "local_or_unknown": (MUTED, "x"),
               "edge_of_gap": (AXIS, "v")}
S4_LABEL = "S4 (indicative, 1 Hz)"


def _signal_color(signal: str) -> str:
    if signal.endswith(("L5", "E5a", "B2a")):
        return ORANGE
    if signal.endswith("B1I"):
        return AQUA
    return BLUE


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"]})


def _time_axis(ax, cfg):
    tz = ZoneInfo(cfg["display_tz"])
    loc = mdates.AutoDateLocator(tz=tz, minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))


def _shade_night(ax, t0, t1, cfg):
    for a, b in common.night_intervals(t0, t1, cfg):
        ax.axvspan(a, b, color=NIGHT_WASH, alpha=0.5, lw=0, zorder=0)


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _grid(n: int, ncols: int = 3, w: float = 4.2, h: float = 2.2, sharey=True):
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 0.6), sharex=True,
                             sharey=sharey, squeeze=False, layout="constrained")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def _legend_sources(present) -> list:
    return [Line2D([], [], color=INK2, marker=MARKER[s], ls="", ms=6, label={"raw": "Raw", "status": "Status burst",
                                                                             "nmea": "L86 NMEA"}[s])
            for s in ("raw", "status", "nmea") if s in present]


def _source_of(obs: pd.DataFrame) -> pd.Series:
    return obs["source"].where(obs["receiver"] == "phone", "nmea")


# ------------------------------------------------------------------ 1. C/N0 timelines

def cn0_timelines(obs, events, cfg, out: Path) -> list[str]:
    files = []
    v = obs[obs["cn0_dbhz"].notna()].copy()
    if v.empty:
        return files
    v["src"] = _source_of(v)
    t0, t1 = v["time_utc"].min(), v["time_utc"].max()
    for const, gc in v.groupby("constellation"):
        sats = sorted(gc["svid"].unique())
        fig, axes = _grid(len(sats))
        signals = set()
        for ax, sv in zip(axes, sats):
            g = gc[gc["svid"] == sv]
            _shade_night(ax, t0, t1, cfg)
            for ev in events[(events["constellation"] == const) & (events["svid"] == sv)].itertuples():
                ax.axvspan(ev.start_utc, ev.end_utc, color=AXIS if ev.edge_of_gap else SERIOUS,
                           alpha=0.35 if ev.edge_of_gap else 0.18, lw=0, zorder=0)
            for (sig, src), gs in g.groupby(["signal", "src"]):
                signals.add(sig)
                ax.plot(gs["time_utc"], gs["cn0_dbhz"], MARKER[src], color=_signal_color(sig),
                        ms=3 if src == "raw" else 2.5, mew=0.8, alpha=0.85)
            for sig, gs in g.groupby("signal"):
                ax.plot(gs["time_utc"], gs["baseline_db"], "-", color=_signal_color(sig), lw=1, alpha=0.6)
            ax.set_title(common.sat_label(const, sv), loc="left", fontsize=9)
            _time_axis(ax, cfg)
        handles = [Line2D([], [], color=c, lw=2, label=", ".join(sorted(s for s in signals if _signal_color(s) == c)))
                   for c in (BLUE, ORANGE, AQUA) if any(_signal_color(s) == c for s in signals)]
        handles += _legend_sources(set(gc["src"])) + [
            Patch(color=SERIOUS, alpha=0.3, label="fade event"), Patch(color=AXIS, alpha=0.5, label="edge_of_gap (ignored)"),
            Patch(color=NIGHT_WASH, label="night window"), Line2D([], [], color=INK2, lw=1, label="15-min baseline")]
        fig.legend(handles=handles, loc="outside upper center", ncol=min(len(handles), 8))
        fig.supylabel("C/N0 (dB-Hz)", color=INK2)
        fig.supxlabel("Time (IST)", color=INK2)
        fig.suptitle(f"{const} C/N0 per satellite", x=0.01, ha="left", fontsize=11)
        files.append(_save(fig, out / f"cn0_timeline_{const}.png"))
    return files


# ------------------------------------------------------------------ 2. S4

def s4_timeline(s4, cfg, out: Path) -> str | None:
    s = s4[s4["s4"].notna()].copy()
    if s.empty:
        return None
    s["src"] = s["source"].where(s["receiver"] == "phone", "nmea")
    consts = sorted(s["constellation"].unique())
    fig, axes = plt.subplots(len(consts), 1, figsize=(10, 1.9 * len(consts) + 0.8), sharex=True, squeeze=False, layout="constrained")
    t0, t1 = s["minute_utc"].min(), s["minute_utc"].max() + pd.Timedelta(minutes=1)
    for ax, const in zip(axes[:, 0], consts):
        _shade_night(ax, t0, t1, cfg)
        g = s[s["constellation"] == const]
        for src, gs in g.groupby("src"):
            mk = MARKER[src] if src != "raw" else "o"
            ok = gs["s4_valid"].astype(bool)
            x = gs["minute_utc"] + pd.Timedelta(seconds=30)
            ax.plot(x[ok], gs.loc[ok, "s4"], mk, color=SOURCE_COLOR[src], ms=4, mew=1, ls="", alpha=0.9)
            ax.plot(x[~ok], gs.loc[~ok, "s4"], mk, color=AXIS, mfc="none", ms=4, mew=0.8, ls="")
        for y, lab in ((cfg["s4"]["moderate"], "moderate"), (cfg["s4"]["strong"], "strong")):
            ax.axhline(y, color=INK2, ls="--", lw=0.8)
        ax.set_ylabel(const, rotation=0, ha="right", va="center")
        ax.set_ylim(0, max(1.0, float(g["s4"].max()) * 1.1))
        _time_axis(ax, cfg)
    handles = [Line2D([], [], color=SOURCE_COLOR[x], marker=m, ls="", ms=6, label=lab)
               for x, m, lab in (("raw", "o", "Raw (s4_raw)"), ("status", "x", "Status burst (s4_indicative)"),
                                 ("nmea", "+", "L86 NMEA"))
               if x in set(s["src"])]
    handles += [Line2D([], [], color=AXIS, marker="o", mfc="none", ls="", ms=6, label="below 2× noise floor"),
                Line2D([], [], color=INK2, ls="--", lw=0.8, label="0.3 / 0.6 thresholds"),
                Patch(color=NIGHT_WASH, label="night window 19:30–01:00 IST")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    fig.suptitle(f"{S4_LABEL}, per satellite and minute", x=0.01, ha="left", fontsize=11)
    fig.supxlabel("Time (IST)", color=INK2)
    return _save(fig, out / "s4_timeline.png")


# ------------------------------------------------------------------ 3. NavIC

def navic_health(obs, events, nav_sum, cfg, out: Path) -> str | None:
    nav = obs[obs["constellation"] == "NavIC"].copy()
    if nav.empty or nav_sum.get("status") == "no_data":
        return None
    sats = sorted(nav["svid"].unique())
    fig, axes = _grid(len(sats), ncols=len(sats), w=4.5, h=2.6, sharey=False)
    t0, t1 = nav["time_utc"].min(), nav["time_utc"].max()
    tracked = nav_sum.get("status") == "tracked"
    for ax, sv in zip(axes, sats):
        g = nav[nav["svid"] == sv].sort_values("time_utc")
        _shade_night(ax, t0, t1, cfg)
        if tracked and g["cn0_dbhz"].notna().any():
            for ev in events[(events["constellation"] == "NavIC") & (events["svid"] == sv)].itertuples():
                ax.axvspan(ev.start_utc, ev.end_utc, color=SERIOUS, alpha=0.18, lw=0, zorder=0)
            src = _source_of(g)
            for s_, gs in g.groupby(src):
                ax.plot(gs["time_utc"], gs["cn0_dbhz"], MARKER[s_], color=ORANGE, ms=3)
            ax.plot(g["time_utc"], g["baseline_db"], "-", color=INK2, lw=1)
            ax.set_ylabel("C/N0 (dB-Hz)")
        else:
            ax.plot(g["time_utc"], g["elev_deg"], ".", color=MUTED, ms=2)
            ax.axhline(cfg["elev_mask_deg"], color=INK2, ls="--", lw=0.8)
            ax.set_ylim(0, 90)
            ax.set_ylabel("elevation (deg)")
            ax.text(0.5, 0.5, "visible, not tracked\n(C/N0 = 0)", transform=ax.transAxes, ha="center",
                    va="center", color=INK2, fontsize=10)
        ax.set_title(common.sat_label("NavIC", sv), loc="left")
        _time_axis(ax, cfg)
    if tracked:
        handles = [Line2D([], [], color=ORANGE, marker=".", ls="", label="NavIC L5 C/N0"),
                   Line2D([], [], color=INK2, lw=1, label="15-min baseline"),
                   Patch(color=SERIOUS, alpha=0.3, label="fade event")]
        title = "NavIC L5 health"
    else:
        handles = [Line2D([], [], color=INK2, ls="--", lw=0.8, label="elevation mask")]
        title = "NavIC: visible in Status, never tracked"
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.supxlabel("Time (IST)", color=INK2)
    return _save(fig, out / "navic_health.png")


# ------------------------------------------------------------------ 4. TEC

def tec(tec_min, cfg, out: Path) -> str | None:
    if tec_min is None or tec_min.empty:
        return None
    tm = tec_min.assign(dstec=tec_code.relative_stec(tec_min))
    sats = sorted(tm["satellite"].unique())
    fig, axes = _grid(len(sats), ncols=4, w=3.2, h=2.0, sharey=True)
    for ax, sat in zip(axes, sats):
        c = BLUE if sat.startswith("G") else ORANGE
        for _, g in tm[tm["satellite"] == sat].sort_values("minute_utc").groupby("arc_id"):
            x = g["minute_utc"] + pd.Timedelta(seconds=30)
            ax.errorbar(x, g["dstec"], yerr=g["stec_std"], fmt="o-", color=c, ms=4, lw=1.2, capsize=2,
                        ecolor=AXIS)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(sat, loc="left")
        _time_axis(ax, cfg)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", label="GPS L1/L5"),
                        Line2D([], [], color=ORANGE, marker="o", label="Galileo E1/E5a"),
                        Line2D([], [], color=AXIS, lw=1, label="±1 std within minute")],
               loc="outside upper right", ncol=3)
    fig.suptitle("Code-only ΔSTEC per arc (60 s mean minus the arc's first minute)",
                 x=0.01, ha="left", fontsize=10)
    fig.supylabel("relative STEC (TECU), uncalibrated", color=INK2)
    fig.supxlabel("Time (IST)", color=INK2)
    return _save(fig, out / "tec_code.png")


# ------------------------------------------------------------------ 5/8. Skyplots

def _sky_axes():
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    ax.set_rticks([30, 60, 90])
    ax.set_yticklabels(["60°", "30°", ""], color=MUTED, fontsize=7)
    ax.set_rlabel_position(112.5)  # elevation labels away from the N azimuth label
    ax.grid(color=GRID)
    return fig, ax


def skyplot_tracks(obs, s4, cfg, out: Path) -> str | None:
    v = obs[obs["cn0_dbhz"].notna() & obs["elev_deg"].notna() & obs["az_deg"].notna()]
    if v.empty:
        return None
    valid = s4[s4["s4_valid"].astype(bool)] if len(s4) else s4
    max_s4 = valid.groupby(["constellation", "svid"])["s4"].max() if len(valid) else pd.Series(dtype=float)
    fig, ax = _sky_axes()
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    vmax = max(0.6, float(np.nanmax(max_s4.to_numpy())) if max_s4.notna().any() else 0.6)
    norm = matplotlib.colors.Normalize(0, vmax)
    for (const, sv), g in v.groupby(["constellation", "svid"]):
        g = g.sort_values("time_utc")
        m = max_s4.get((const, sv), np.nan)
        col = cmap(norm(m)) if np.isfinite(m) else AXIS
        ax.plot(np.radians(g["az_deg"]), 90 - g["elev_deg"], ".", color=col, ms=2)
        ax.text(np.radians(g["az_deg"].iloc[-1]), 90 - g["elev_deg"].iloc[-1], common.sat_label(const, sv),
                fontsize=6.5, color=INK2)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.08)
    cb.set_label(f"max valid {S4_LABEL}  (grey = none above noise floor)", color=INK2)
    ax.set_title("Sky tracks (tracked samples)", loc="left", pad=18)
    return _save(fig, out / "skyplot_tracks.png")


def skyplot_events(classified, cfg, out: Path) -> str | None:
    e = classified.dropna(subset=["elev_deg_at_peak", "az_deg_at_peak"]) if len(classified) else classified
    if e is None or not len(e):
        return None
    fig, ax = _sky_axes()
    handles = []
    for cls, (col, mk) in CLASS_STYLE.items():
        g = e[e["class"] == cls]
        if g.empty:
            continue
        ax.plot(np.radians(g["az_deg_at_peak"]), 90 - g["elev_deg_at_peak"], mk, color=col, ms=7,
                mec=SURFACE, mew=1, ls="", alpha=0.9)
        handles.append(Line2D([], [], color=col, marker=mk, ls="", ms=7, label=f"{cls} ({len(g)})"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title("Fade events at peak, by class (heuristic)", loc="left", pad=18)
    return _save(fig, out / "skyplot_events.png")


# ------------------------------------------------------------------ 6. dual-frequency

def dualfreq_events(obs, classified, dual, cfg, out: Path, top: int = 5) -> list[str]:
    if dual is None or dual.empty or classified is None or classified.empty:
        return []
    dual_sats = set(zip(dual["constellation"], dual["svid"]))
    ev = classified[[(c, s) in dual_sats for c, s in zip(classified["constellation"], classified["svid"])]]
    ev = (ev.sort_values("peak_depth_db", ascending=False).drop_duplicates(["constellation", "svid", "start_utc"])
            .rename(columns={"class": "cls"}))  # 'class' is a keyword: itertuples would rename it
    files, used = [], {}
    for e in ev.head(top).itertuples():
        d = dual[(dual["constellation"] == e.constellation) & (dual["svid"] == e.svid)]
        sigs = [d["signal_l1"].iloc[0], d["signal_l5"].iloc[0]]
        pad = pd.Timedelta(minutes=5)
        g = obs[(obs["constellation"] == e.constellation) & (obs["svid"] == e.svid) & obs["signal"].isin(sigs)
                & obs["time_utc"].between(e.start_utc - pad, e.end_utc + pad)]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        _shade_night(ax, e.start_utc - pad, e.end_utc + pad, cfg)
        ax.axvspan(e.start_utc, e.end_utc, color=SERIOUS, alpha=0.18, lw=0)
        for sig in sigs:
            gs = g[g["signal"] == sig].sort_values("time_utc")
            col = BLUE if sig == sigs[0] else ORANGE
            for src, gg in gs.groupby(_source_of(gs)):
                ax.plot(gg["time_utc"], gg["cn0_dbhz"], MARKER[src], color=col, ms=3.5)
            ax.plot(gs["time_utc"], gs["baseline_db"], "-", color=col, lw=1, alpha=0.7)
        dm = d[(d["minute_utc"] >= e.start_utc) & (d["minute_utc"] < e.end_utc)]
        corr = dm["corr_detrended"].mean()
        dep = (f"depth {sigs[0]} {dm['fade_depth_l1_db'].max():.1f} dB, {sigs[1]} {dm['fade_depth_l5_db'].max():.1f} dB"
               if len(dm) else "")
        sat = common.sat_label(e.constellation, e.svid)
        corr_txt = f"; corr {corr:.2f}" if np.isfinite(corr) else ""
        ax.set_title(f"{sat} event {e.event_id} — {e.cls}; {dep}{corr_txt}", loc="left", fontsize=9)
        ax.set_ylabel("C/N0 (dB-Hz)")
        ax.set_xlabel("Time (IST)")
        _time_axis(ax, cfg)
        handles = [Line2D([], [], color=BLUE, lw=2, label=sigs[0]), Line2D([], [], color=ORANGE, lw=2, label=sigs[1]),
                   *_legend_sources(set(_source_of(g))), Patch(color=SERIOUS, alpha=0.3, label="event")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        used[sat] = used.get(sat, 0) + 1
        name = f"dualfreq_{sat}.png" if used[sat] == 1 else f"dualfreq_{sat}_{used[sat]}.png"
        files.append(_save(fig, out / name))
    return files


# ------------------------------------------------------------------ 7. interference

def interference_plot(intf, cfg, out: Path) -> str | None:
    if intf is None or intf.empty:
        return None
    c = cfg["interference"]
    x = intf["minute_utc"] + pd.Timedelta(seconds=30)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, layout="constrained")
    t0, t1 = intf["minute_utc"].min(), intf["minute_utc"].max() + pd.Timedelta(minutes=1)
    for ax in (a1, a2):
        _shade_night(ax, t0, t1, cfg)
        for m in intf.loc[intf["local_cause_flag"], "minute_utc"]:
            ax.axvspan(m, m + pd.Timedelta(minutes=1), color=SERIOUS, alpha=0.18, lw=0)
        _time_axis(ax, cfg)
    a1.bar(x, intf["frac_sats_faded"], width=pd.Timedelta(seconds=50), color=BLUE)
    a1.axhline(c["frac_sats_faded"], color=INK2, ls="--", lw=0.8)
    a1.set_ylim(0, 1.05)
    a1.set_ylabel("share of sats\nfaded ≥ 6 dB")
    if intf["agc_change_max_db"].notna().any():
        a2.plot(x, intf["agc_change_max_db"], "o", color=ORANGE, ms=3.5)
        for y in (c["agc_change_db"], -c["agc_change_db"]):
            a2.axhline(y, color=INK2, ls="--", lw=0.8)
    else:
        a2.text(0.5, 0.5, "no usable AGC rows", transform=a2.transAxes, ha="center", color=INK2)
    a2.set_ylabel("AGC change vs\n15-min median (dB)")
    a2.set_xlabel("Time (IST)")
    fig.legend(handles=[Patch(color=SERIOUS, alpha=0.3, label="local_cause_flag minute"),
                        Line2D([], [], color=INK2, ls="--", lw=0.8, label="thresholds"),
                        Patch(color=NIGHT_WASH, label="night window")],
               loc="outside upper center", ncol=3)
    fig.suptitle("Local-cause check (phone)", x=0.01, ha="left", fontsize=11)
    return _save(fig, out / "interference.png")


# ------------------------------------------------------------------ crosscheck (L86)

def crosscheck_events(obs, confirmed, cfg, out: Path) -> list[str]:
    files = []
    if confirmed is None or confirmed.empty:
        return files
    for e in confirmed.itertuples():
        pad = pd.Timedelta(minutes=5)
        g = obs[(obs["constellation"] == e.constellation) & (obs["svid"] == e.svid)
                & obs["signal"].isin(["L1", "GPS_L1", "GAL_E1", "GLO_G1"])
                & obs["time_utc"].between(e.start_utc - pad, e.end_utc + pad)]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.axvspan(e.start_utc, e.end_utc, color=SERIOUS, alpha=0.18, lw=0)
        for rcv, col in (("l86", AQUA), ("phone", BLUE)):
            gr = g[g["receiver"] == rcv]
            for src, gg in gr.groupby(_source_of(gr)):
                ax.plot(gg["time_utc"], gg["cn0_dbhz"], MARKER[src], color=col, ms=3)
        ax.legend(handles=[Line2D([], [], color=AQUA, marker="+", ls="", label="L86 (not yet validated)"),
                           Line2D([], [], color=BLUE, marker=".", ls="", label="phone")], loc="upper right")
        ax.set_title(f"{e.satellite}: confirmed fade, corr {e.corr:.2f}", loc="left")
        ax.set_ylabel("C/N0 (dB-Hz)")
        _time_axis(ax, cfg)
        files.append(_save(fig, out / f"crosscheck_{e.satellite}.png"))
    return files


def make_all(res: dict, cfg: dict, out: Path) -> list[str]:
    _style()
    # Figures show the primary phone (and the L86); the second phone is compared in summary.md.
    t = {k: (v[v["receiver"] != "phone2"] if isinstance(v, pd.DataFrame) and "receiver" in v else v)
         for k, v in res["tables"].items()}
    obs = res["obs"][res["obs"]["receiver"] != "phone2"]
    files = []
    files += cn0_timelines(obs, t["fade_events"], cfg, out)
    files.append(s4_timeline(t["s4_1min"], cfg, out))
    if "navic_1min" in t:
        files.append(navic_health(obs, t["fade_events"], res.get("navic", {}), cfg, out))
        files.append(tec(t["tec_code_1min"], cfg, out))
        files.append(skyplot_tracks(obs, t["s4_1min"], cfg, out))
        files += dualfreq_events(obs, t["events_classified"], t["dualfreq_1min"], cfg, out)
        it = t["interference_1min"]
        files.append(interference_plot(it[it["receiver"] == "phone"], cfg, out))
        files.append(skyplot_events(t["events_classified"], cfg, out))
    else:
        files.append(skyplot_tracks(obs, t["s4_1min"], cfg, out))
    if "confirmed_events" in t:
        files += crosscheck_events(obs, t["confirmed_events"], cfg, out)
    return [f for f in files if f]
