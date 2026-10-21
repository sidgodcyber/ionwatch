import numpy as np
import pandas as pd
import pytest
from conftest import synth_obs

from ionwatch.proc import s4


def s4_of(obs, cfg, rate=1.0):
    return s4.compute(obs, cfg, rate, "test")


def test_constant_cn0_gives_zero_s4(cfg):
    out = s4_of(synth_obs(n_sats=1, minutes=5), cfg)
    assert (out["s4"] < 0.01).all()
    assert (~out["s4_valid"]).all() and (out["level"] == "below_noise").all()   # 0 is below the noise floor


def test_s4_noise_formula(cfg):
    # sqrt(1 / (T * 10^(CN0/10))): 20 dB-Hz, T = 1 s -> 0.1; 40 dB-Hz -> 0.01
    assert s4.s4_noise(20, 1.0) == pytest.approx(0.1)
    assert s4.s4_noise(40, 1.0) == pytest.approx(0.01)
    assert s4.s4_noise(30, 0.1) == pytest.approx(0.1)


def test_s4_valid_needs_twice_noise(cfg):
    """Same fluctuation (S4 ~0.15): valid at 40 dB-Hz (noise 0.01), not at 15 dB-Hz (noise 0.18)."""
    rng = np.random.default_rng(7)
    out = {}
    for cn0 in (40.0, 15.0):
        obs = synth_obs(n_sats=1, minutes=5, cn0=cn0)
        obs["cn0_dbhz"] = cn0 + rng.normal(0, 0.65, len(obs))
        out[cn0] = s4_of(obs, cfg)
    assert out[40.0]["s4_valid"].all() and out[40.0]["reason"].eq("ok").all()
    assert (~out[15.0]["s4_valid"]).all() and out[15.0]["reason"].str.startswith("below_noise").all()
    assert (out[15.0]["s4"] < 2 * out[15.0]["s4_noise"]).all()


def test_single_jump_is_step_artifact(cfg):
    obs = synth_obs(n_sats=1, minutes=3, cn0=20.0)
    t = obs["time_utc"]
    obs.loc[t >= t.iloc[0] + pd.Timedelta(minutes=1, seconds=30), "cn0_dbhz"] = 28.2   # +8.2 dB re-lock
    out = s4_of(obs, cfg).sort_values("minute_utc")
    mid = out.iloc[1]
    assert np.isnan(mid["s4"]) and mid["reason"] == "step_artifact"
    assert out.iloc[0]["reason"] != "step_artifact" and out.iloc[2]["reason"] != "step_artifact"


def test_smoothed_relock_patterns_are_step_artifacts(cfg):
    """Smoothed-C/N0 re-lock shapes (seen on an L1-only test phone): jump then slow decay, a rise over 2 epochs, dip + re-lock."""
    decay = np.r_[np.linspace(20.1, 17.4, 30), [17.6, 18.2, 18.9, 20.0], np.linspace(28.2, 23.3, 26)]
    two_step = np.r_[np.linspace(21.6, 26.2, 37), [30.6], np.full(22, 34.0)]
    dip = np.r_[np.linspace(29.2, 26.8, 16), [18.0, 19.3, 21.8, 24.8], np.linspace(35.1, 27.5, 40)]
    for v in (decay, two_step, dip):
        assert s4.is_step_artifact(v, cfg["s4"])
    rng = np.random.default_rng(0)
    scint = 25 + 4 * rng.normal(size=60)          # real fluctuation: many jumps, not flat
    assert not s4.is_step_artifact(scint, cfg["s4"])
