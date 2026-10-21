import pandas as pd
from conftest import active_of, synth_obs

from ionwatch.proc import classify, dualfreq, fades, interference, s4


def test_all_satellite_8db_drop_is_local(cfg):
    """Every satellite drops 8 dB in the same minute: local_cause_flag, events classed 'local'."""
    obs = synth_obs(n_sats=6, minutes=30, t0="2026-10-20T15:00:00Z")   # 20:30 IST, inside the night window
    t = obs["time_utc"]
    hit = (t >= t.iloc[0] + pd.Timedelta(minutes=15)) & (t < t.iloc[0] + pd.Timedelta(minutes=16))
    obs.loc[hit, "cn0_dbhz"] -= 8
    active = active_of(obs)
    ob = fades.add_baseline(obs, active, cfg, 1.0)
    m = fades.per_minute(ob, active, cfg, 1.0)
    ev = fades.events(m)
    intf, info = interference.compute(m, None, active, cfg)
    flagged = intf[intf["local_cause_flag"]]
    assert list(flagged["minute_utc"]) == [t.iloc[0] + pd.Timedelta(minutes=15)]
    assert flagged["frac_sats_faded"].iloc[0] == 1.0 and flagged["reason"].iloc[0] == "frac_sats_faded"
    assert len(ev) == 6
    dual, _ = dualfreq.compute(ob, m, cfg)
    cl = classify.classify(ev, intf, dual, s4.compute(ob, cfg, 1.0, "indicative_1Hz"), cfg)
    assert (cl["class"] == "local").all()


def test_single_satellite_night_fade_is_iono_likely(cfg):
    """One satellite, one minute, 8 dB, inside the night window: not local; the step drives the
    1 Hz S4 above 0.3, so the S4 rule makes it iono_likely."""
    obs = synth_obs(n_sats=6, minutes=30, t0="2026-10-20T15:00:00Z")
    t = obs["time_utc"]
    hit = (obs["svid"] == 1) & (t >= t.iloc[0] + pd.Timedelta(minutes=15)) \
        & (t < t.iloc[0] + pd.Timedelta(minutes=16))
    obs.loc[hit, "cn0_dbhz"] -= 8
    active = active_of(obs)
    ob = fades.add_baseline(obs, active, cfg, 1.0)
    m = fades.per_minute(ob, active, cfg, 1.0)
    intf, _ = interference.compute(m, None, active, cfg)
    assert not intf["local_cause_flag"].any()
    cl = classify.classify(fades.events(m), intf, pd.DataFrame(columns=dualfreq.COLS),
                           s4.compute(ob, cfg, 1.0, "indicative_1Hz"), cfg)
    assert list(cl["class"]) == ["iono_likely"]
    assert cl["in_night_window"].iloc[0] and "S4" in cl["reason"].iloc[0]
    assert not cl["both_bands_faded"].iloc[0]


def test_low_cn0_caps_iono_likely(cfg):
    """Same night-time fade as above at 25 dB-Hz mean C/N0: capped at iono_possible, reason low_cn0."""
    obs = synth_obs(n_sats=6, minutes=30, t0="2026-10-20T15:00:00Z", cn0=25.0)
    t = obs["time_utc"]
    hit = (obs["svid"] == 1) & (t >= t.iloc[0] + pd.Timedelta(minutes=15)) \
        & (t < t.iloc[0] + pd.Timedelta(minutes=16))
    obs.loc[hit, "cn0_dbhz"] -= 8
    active = active_of(obs)
    ob = fades.add_baseline(obs, active, cfg, 1.0)
    m = fades.per_minute(ob, active, cfg, 1.0)
    intf, _ = interference.compute(m, None, active, cfg)
    s = s4.compute(ob, cfg, 1.0, "indicative_1Hz")
    assert (s["s4_valid"] & (s["s4"] >= 0.3)).any()          # would qualify on S4 alone
    cl = classify.classify(fades.events(m), intf, pd.DataFrame(columns=dualfreq.COLS), s, cfg)
    assert list(cl["class"]) == ["iono_possible"] and cl["reason"].iloc[0].startswith("low_cn0")


def test_edge_of_gap_event_is_not_classified(cfg):
    ev = pd.DataFrame([{"event_id": "phone-0001", "receiver": "phone", "source": "status", "constellation": "Galileo",
                        "svid": 13, "satellite": "E13", "signal": "GAL_E1",
                        "start_utc": pd.Timestamp("2026-10-20T15:22:00Z"), "end_utc": pd.Timestamp("2026-10-20T15:25:00Z"),
                        "n_minutes": 3, "peak_utc": pd.Timestamp("2026-10-20T15:23:00Z"), "peak_depth_db": 10.2,
                        "lock_losses": 1, "elev_deg_at_peak": 35.0, "az_deg_at_peak": 100.0, "cn0_mean": 35.0,
                        "edge_of_gap": True}])
    intf = pd.DataFrame(columns=interference.COLS).astype({"local_cause_flag": bool})
    cl = classify.classify(ev, intf, pd.DataFrame(columns=dualfreq.COLS), pd.DataFrame(columns=s4.COLS), cfg)
    assert list(cl["class"]) == ["edge_of_gap"]


def test_daytime_fade_is_local_or_unknown(cfg):
    """Outside 19:30-01:00 IST nothing is iono_*: capped at local_or_unknown."""
    obs = synth_obs(n_sats=6, minutes=30, t0="2026-10-20T06:00:00Z", elev=60.0)   # 11:30 IST
    t = obs["time_utc"]
    hit = (obs["svid"] == 1) & (t >= t.iloc[0] + pd.Timedelta(minutes=15)) \
        & (t < t.iloc[0] + pd.Timedelta(minutes=16))
    obs.loc[hit, "cn0_dbhz"] -= 8
    active = active_of(obs)
    ob = fades.add_baseline(obs, active, cfg, 1.0)
    m = fades.per_minute(ob, active, cfg, 1.0)
    intf, _ = interference.compute(m, None, active, cfg)
    cl = classify.classify(fades.events(m), intf, pd.DataFrame(columns=dualfreq.COLS),
                           s4.compute(ob, cfg, 1.0, "indicative_1Hz"), cfg)
    assert list(cl["class"]) == ["local_or_unknown"] and cl["reason"].iloc[0] == "outside_night_window"


def test_agc_placeholder_excluded(cfg):
    t = pd.date_range("2026-10-20T15:00:00Z", periods=600, freq="1s")
    agc = pd.DataFrame({"time_utc": t, "AgcDb": -135.0, "CarrierFrequencyHz": 1176450000.0,
                        "constellation": "NavIC", "signal": "NAVIC_L5"})
    ch, placeholders = interference.agc_changes(agc, cfg)
    assert ch.empty and placeholders == {"NAVIC_L5": 600}
