import pytest
from conftest import GOOD

from ionwatch.io import gnsslogger
from ionwatch.proc import tec_code


def raw_of(path, cfg):
    log = gnsslogger.read_phone([path], cfg)
    gnsslogger.attach_elevation(log, cfg)
    return log.raw


def test_tec_constant():
    assert tec_code.K_TECU_PER_M == pytest.approx(7.76, abs=0.01)


def test_good_fixture_gives_gps_and_galileo_tec(cfg):
    ep, m, note = tec_code.compute(raw_of(GOOD, cfg), cfg)
    assert set(m["constellation"]) == {"GPS", "Galileo"}
    assert (m["calibration_note"] == tec_code.CALIBRATION_NOTE).all()
    assert (ep["stec_tecu"].abs() <= cfg["tec"]["max_abs_stec_tecu"]).all()
    assert (ep["elev_deg"] >= cfg["elev_mask_deg"]).all()


def test_l1_only_log_skips_with_note(cfg, l1_only_log):
    ep, m, note = tec_code.compute(raw_of(l1_only_log, cfg), cfg)
    assert ep.empty and m.empty
    assert note.startswith("no L5/NavIC data")
    assert list(m.columns) == tec_code.COLS
