"""GnssLogger parser + inspect report on trimmed copies of the real phone logs."""
import json

import numpy as np
import pytest
from click.testing import CliRunner
from conftest import BAD, BAD_TAIL, GOOD

from ionwatch.cli import cli
from ionwatch.io import gnsslogger


def load(path, cfg):
    log = gnsslogger.read_phone([path], cfg)
    join = gnsslogger.attach_elevation(log, cfg)
    return log, gnsslogger.phone_quality(log, join, cfg)


@pytest.fixture(scope="module")
def bad(cfg):
    return load(BAD, cfg)


@pytest.fixture(scope="module")
def good(cfg):
    return load(GOOD, cfg)


def codes(q):
    return {w["code"] for w in q["warnings"]}


def test_columns_from_header(bad):
    log, _ = bad
    for col in ("utcTimeMillis", "FullBiasNanos", "Cn0DbHz", "CodeType", "SvPositionEcefXMeters"):
        assert col in log.raw.columns
    assert str(log.raw["FullBiasNanos"].dtype) == "Int64"  # no float precision loss


def test_one_second_epochs(bad):
    d = np.diff(np.unique(bad[0].raw["utcTimeMillis"].to_numpy(dtype=float))) / 1000
    assert np.median(d) == pytest.approx(1.0, abs=0.01)


def test_adr_invalid(bad):
    assert bad[1]["raw"]["adr_valid_rows"] == 0
    assert "no_carrier_phase" in codes(bad[1])


def test_bad_navic_seen_but_not_tracked(bad):
    q = bad[1]
    assert q["navic"]["svids_in_raw"] == []
    assert q["navic"]["svids_in_status"] == [2, 9, 10]
    assert q["navic"]["tracked"] is False
    assert "navic_not_tracked" in codes(q)


def test_bad_low_cn0_warning(bad):
    assert bad[1]["raw"]["mean_cn0_dbhz"] < 25
    assert "low_cn0" in codes(bad[1])


def test_good_navic_tracked_on_l5(good):
    log, q = good
    nav = log.raw[log.raw["constellation"] == "NavIC"]
    assert set(nav["Svid"]) == {2, 9}
    assert (nav["CarrierFrequencyHz"] == 1176450000).all()
    assert set(nav["signal"]) == {"NAVIC_L5"}
    assert nav["Cn0DbHz"].median() == pytest.approx(27, abs=3)
    assert "navic_not_tracked" not in codes(q)


def test_good_dual_frequency(good):
    dual = good[1]["raw"]["dual_frequency"]
    assert all(dual[c]["sat_epochs"] > 0 for c in ("GPS", "Galileo", "BeiDou"))


@pytest.mark.parametrize("path", [GOOD, BAD])
def test_pseudorange_sanity(path, cfg):
    raw = gnsslogger.read_phone([path], cfg).raw
    p = gnsslogger.pseudorange_m(raw)
    ok = gnsslogger.tow_valid(raw) & (raw["ReceivedSvTimeUncertaintyNanos"] <= 100)
    gps = p[ok & (raw["constellation"] == "GPS")]
    gal = p[ok & (raw["constellation"] == "Galileo")]
    assert len(gps) and len(gal)
    assert gps.between(20_000e3, 24_000e3).all()
    assert gal.between(23_000e3, 28_500e3).all()


@pytest.mark.parametrize("path", [GOOD, BAD])
def test_elevation_join(path, cfg):
    log = gnsslogger.read_phone([path], cfg)
    stats = gnsslogger.attach_elevation(log, cfg)
    assert stats["match_rate"] > 0.95
    assert stats["status_vs_geometry_median_abs_diff_deg"] < 0.5


def test_status_timing_holdout(bad):
    t = next(iter(bad[0].status_timing.values()))
    assert t["holdout_validation"]["frac_within_1s"] >= 0.95


def test_status_bursts_timed_from_file_order(cfg):
    """After Raw stops, untimed Status bursts get times from the Fix rows around them."""
    log = gnsslogger.read_phone([BAD_TAIL], cfg)
    q = next(iter(log.status_timing.values()))["block_quality"]
    assert q.get("estimated", 0) > 400 and q.get("ambiguous", 0) > 0
    obs, stats = gnsslogger.cn0_obs(log, cfg)
    late = obs[obs["time_utc"] > log.raw["time_utc"].max() + np.timedelta64(30, "m")]
    assert len(late) and (late["source"] == "status").all()
    assert stats["status_blocks_dropped_untimed"] == q["ambiguous"]


def test_inspect_cli(tmp_path):
    res = CliRunner().invoke(cli, ["inspect", "--phone", str(BAD), "--out", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "navic_not_tracked" in res.output and "low_cn0" in res.output
    q = json.loads((tmp_path / "phone_quality.json").read_text(encoding="utf-8"))
    assert {"navic_not_tracked", "low_cn0"} <= codes(q)
