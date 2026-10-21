"""End-to-end runs: each real-log fixture must take its expected path and never crash."""
import json

import pandas as pd
import pytest
from conftest import BAD, BAD_TAIL, GOOD

from ionwatch import pipeline

OUTPUTS = ["fades_1min", "fade_events", "s4_1min", "navic_1min", "navic_events", "tec_code_1min",
           "dualfreq_1min", "interference_1min", "events_classified"]


def run(path, tmp_path, cfg, plots=False):
    out = tmp_path / path.stem
    res = pipeline.run([str(path)], None, out, cfg, plots=plots)
    return res, out


def common_checks(out):
    for name in OUTPUTS:
        assert (out / f"{name}.csv").exists() and (out / f"{name}.parquet").exists()
        df = pd.read_csv(out / f"{name}.csv")
        assert "session_id" in df.columns
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (out / "summary.md").exists()
    return s


def test_good_fixture(tmp_path, cfg):
    res, out = run(GOOD, tmp_path, cfg)
    s = common_checks(out)
    nav = json.loads((out / "navic_summary.json").read_text(encoding="utf-8"))
    assert nav["status"] == "tracked"
    tracked = {sv: d for sv, d in nav["satellites"].items() if d["minutes_tracked"]}
    assert set(tracked) == {"2", "9"}
    assert all(abs(d["cn0_mean"] - 27) < 4 for d in tracked.values())
    assert set(res["tables"]["tec_code_1min"]["constellation"]) == {"GPS", "Galileo"}
    assert set(res["tables"]["dualfreq_1min"]["constellation"]) == {"GPS", "Galileo", "BeiDou"}
    assert s["s4"]["s4_raw"]["windows_valid"] > 0


def test_bad_fixture_navic_visible_not_tracked(tmp_path, cfg):
    res, out = run(BAD, tmp_path, cfg, plots=True)
    common_checks(out)
    nav = json.loads((out / "navic_summary.json").read_text(encoding="utf-8"))
    assert nav["status"] == "visible_not_tracked" and "not tracked" in nav["note"]
    assert {d["satellite"] for d in nav["satellites"].values()} == {"I02", "I09", "I10"}
    for name in ("navic_1min", "navic_events"):     # header-only but valid
        df = pd.read_csv(out / f"{name}.csv")
        assert df.empty and "fade_depth_db" in df.columns or "peak_depth_db" in df.columns
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert md.index("## Bottom line") < md.index("## Session") < md.index("## Warnings")
    for text in ("Usable for:", "Not usable for:", "Events still iono_likely after all filters:",
                 "Raw span:", "Total log span:", "Receiver-active data inside 19:30–01:00 IST: 0 min",
                 "No receiver-active data inside", "Low C/N0", "NavIC satellites in view but not tracked",
                 "No carrier phase", "Low Raw density", "heuristic"):
        assert text in md, text
    assert "Short session" not in md
    assert (out / "plots" / "navic_health.png").exists()


def test_bad_tail_status_bursts(tmp_path, cfg):
    res, out = run(BAD_TAIL, tmp_path, cfg)
    s = common_checks(out)
    fm = res["tables"]["fades_1min"]
    assert fm["receiver_off_gaps"].sum() > 0
    s4 = res["tables"]["s4_1min"]
    assert "s4_indicative" in set(s4["s4_label"]) and (s4["method"] == "indicative_1Hz").all()
    assert set(fm["source"]) >= {"status"}
    assert s["phone"]["status_blocks_dropped_untimed"] > 0
    assert "s4_indicative" in s["s4"]
    # Bursts reach into the night window: the active minutes are counted, no outside-window warning.
    assert s["session_lines"]["active_min_in_night_window"] > 0
    assert not any("No receiver-active data inside" in w for w in s["warnings"])
    # Fades sagging into the 19:16 IST silence are edge_of_gap, not classified events.
    cl = res["tables"]["events_classified"]
    assert (cl["class"] == "edge_of_gap").sum() == s["fades"]["edge_of_gap"]["events_ignored"] > 0


def test_l1_only_synthetic_skips_l5_stages(tmp_path, cfg, l1_only_log):
    """L1-only log with no NavIC: navic, tec_code and dualfreq skip cleanly; the rest still runs."""
    res, out = run(l1_only_log, tmp_path, cfg, plots=True)
    s = common_checks(out)
    for stage in ("navic", "tec_code", "dualfreq"):
        assert "no L5/NavIC data" in res["notes"][stage]
    for name in ("navic_1min", "tec_code_1min", "dualfreq_1min"):
        assert res["tables"][name].empty
    assert len(res["tables"]["fades_1min"]) and len(res["tables"]["s4_1min"])
    assert len(res["tables"]["interference_1min"])
    assert s["navic"]["status"] == "no_data"


def test_run_without_l86_notes_skip(tmp_path, cfg):
    res, _ = run(GOOD, tmp_path, cfg)
    assert "l86" in res["notes"] and "crosscheck_1min" not in res["tables"]
