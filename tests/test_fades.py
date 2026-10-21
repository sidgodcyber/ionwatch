import numpy as np
import pandas as pd
import pytest
from conftest import active_of, synth_obs

from ionwatch.proc import fades


def run(obs, cfg, active=None):
    active = active_of(obs) if active is None else active
    ob = fades.add_baseline(obs, active, cfg, 1.0)
    m = fades.per_minute(ob, active, cfg, 1.0)
    return m, fades.events(m)


def test_10db_drop_for_90s_is_one_event(cfg):
    obs = synth_obs(n_sats=1, minutes=40)
    t = obs["time_utc"]
    drop = (t >= t.iloc[0] + pd.Timedelta(minutes=20)) & (t < t.iloc[0] + pd.Timedelta(minutes=20, seconds=90))
    obs.loc[drop, "cn0_dbhz"] -= 10
    m, ev = run(obs, cfg)
    assert len(ev) == 1
    assert ev["peak_depth_db"].iloc[0] == pytest.approx(10, abs=0.5)
    assert ev["lock_losses"].iloc[0] == 0


def test_5s_gap_at_high_elevation_is_one_lock_loss(cfg):
    obs = synth_obs(n_sats=2, minutes=20)   # satellite 2 keeps the receiver active
    t = obs["time_utc"]
    gap = (obs["svid"] == 1) & (t >= t.iloc[0] + pd.Timedelta(minutes=10)) \
        & (t < t.iloc[0] + pd.Timedelta(minutes=10, seconds=5))
    obs.loc[gap, "cn0_dbhz"] = np.nan
    m, ev = run(obs, cfg)
    assert m.loc[m["svid"] == 1, "lock_losses"].sum() == 1
    assert m.loc[m["svid"] == 1, "receiver_off_gaps"].sum() == 0


def test_receiver_off_gap_is_not_lock_loss(cfg):
    """Gaps between duty-cycling bursts are missing data: receiver_off, never lock loss."""
    a = synth_obs(n_sats=3, minutes=10)
    b = synth_obs(n_sats=3, minutes=10, t0="2026-10-20T14:40:00Z")  # 30 min silence
    # Satellite 1 needs 20 s to be reacquired after the restart: still not a lock loss.
    late = (b["svid"] == 1) & (b["time_utc"] < b["time_utc"].iloc[0] + pd.Timedelta(seconds=20))
    b.loc[late, "cn0_dbhz"] = np.nan
    m, ev = run(pd.concat([a, b], ignore_index=True), cfg)
    assert m["lock_losses"].sum() == 0
    assert m["receiver_off_gaps"].sum() == 3
    assert len(ev) == 0


def _with_silence(minutes_before=10, silence_min=30, minutes_after=10, n_sats=3):
    a = synth_obs(n_sats=n_sats, minutes=minutes_before)
    t_after = (a["time_utc"].max() + pd.Timedelta(minutes=silence_min, seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    b = synth_obs(n_sats=n_sats, minutes=minutes_after, t0=t_after)
    return a, b


def test_fade_just_before_silence_is_edge_of_gap(cfg):
    a, b = _with_silence(minutes_before=20)
    t = a["time_utc"]
    # 10 dB fade on sat 1 in the last 40 s before the receiver goes silent
    a.loc[(a["svid"] == 1) & (t >= t.max() - pd.Timedelta(seconds=40)), "cn0_dbhz"] -= 10
    m, ev = run(pd.concat([a, b], ignore_index=True), cfg)
    assert len(ev) == 1 and ev["edge_of_gap"].iloc[0]
    assert ev["peak_depth_db"].iloc[0] == pytest.approx(10, abs=0.5)
    assert not m["event"].any() and m["edge_event"].any()


def test_fade_running_into_silence_is_edge_of_gap_as_a_whole(cfg):
    """C/N0 sagging for 2.5 min before shutdown (the 20 Oct E13 pattern): the whole event is edge."""
    a, b = _with_silence(minutes_before=20)
    t = a["time_utc"]
    sag = (a["svid"] == 1) & (t >= t.max() - pd.Timedelta(seconds=150))
    a.loc[sag, "cn0_dbhz"] -= np.linspace(0, 10, sag.sum())
    _, ev = run(pd.concat([a, b], ignore_index=True), cfg)
    assert len(ev) == 1 and ev["edge_of_gap"].iloc[0] and ev["n_minutes"].iloc[0] >= 2


def test_lock_loss_just_after_resume_is_edge_of_gap(cfg):
    a, b = _with_silence(minutes_before=20, minutes_after=20)
    tb = b["time_utc"]
    gap = (b["svid"] == 1) & (tb >= tb.min() + pd.Timedelta(seconds=20)) & (tb < tb.min() + pd.Timedelta(seconds=26))
    b.loc[gap, "cn0_dbhz"] = np.nan
    m, ev = run(pd.concat([a, b], ignore_index=True), cfg)
    assert m["lock_losses"].sum() == 0 and m["edge_lock_losses"].sum() == 1
    assert len(ev) == 1 and ev["edge_of_gap"].iloc[0]


def test_fade_far_from_silence_is_real(cfg):
    a, b = _with_silence(minutes_before=30)
    t = a["time_utc"]
    a.loc[(a["svid"] == 1) & (t >= t.min() + pd.Timedelta(minutes=15, seconds=30))
          & (t < t.min() + pd.Timedelta(minutes=16, seconds=30)), "cn0_dbhz"] -= 10
    _, ev = run(pd.concat([a, b], ignore_index=True), cfg)
    assert len(ev) == 1 and not ev["edge_of_gap"].iloc[0]


def test_low_elevation_gap_is_not_lock_loss(cfg):
    obs = synth_obs(n_sats=2, minutes=20)
    obs.loc[obs["svid"] == 1, "elev_deg"] = 10.0   # below the 20 deg mask
    t = obs["time_utc"]
    gap = (obs["svid"] == 1) & (t >= t.iloc[0] + pd.Timedelta(minutes=10)) \
        & (t < t.iloc[0] + pd.Timedelta(minutes=10, seconds=5))
    obs.loc[gap, "cn0_dbhz"] = np.nan
    m, _ = run(obs, cfg)
    assert m["lock_losses"].sum() == 0
