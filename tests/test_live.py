"""Live mode against the serial simulator (tools/fake_l86.py) and replay mode.

The simulator runs 40x faster than real time (--speed 40) and live mode uses the matching
--clock-scale. A minute's final record needs ~8.6 simulated minutes of later data (full
15-min baseline), so the simulator runs ~13 minutes for 3 final minutes: ~20 s of wall time. On Windows the simulator serves a
local TCP socket (pyserial socket://); on Linux/macOS the pty test also runs.
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from conftest import BAD_TAIL

from ionwatch import live, pipeline, schema
from ionwatch.io import nmea_l86

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "tools" / "fake_l86.py"
RATE = 40


def start_sim(*args, transport="tcp", speed=RATE):
    p = subprocess.Popen([sys.executable, str(SIM), "--transport", transport, "--speed", str(speed),
                          "--start", "2026-10-22T14:00:00Z", *args],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    line = p.stdout.readline()
    assert line.startswith("PORT "), line
    return p, line.split()[1]


def stop(p):
    p.kill()
    p.wait(timeout=10)


def sat(rec, sys_, svid):
    return next((s for s in rec["satellites"] if s["sys"] == sys_ and s["svid"] == svid), None)


@pytest.fixture
def fast_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["live"]["reconnect_s"] = 0.2
    return c


def test_live_three_minutes_with_injected_fade(tmp_path, fast_cfg):
    p, port = start_sim("--duration-s", "800", "--inject-fade", "12:70:40:10",
                        "--inject-garbage", "30", "--inject-dropout", "150:10")
    try:
        live.run_live(port, 9600, tmp_path, fast_cfg, "test-node", clock_scale=RATE, max_minutes=3, duration_s=120)
    finally:
        stop(p)
    allrecs = schema.read_jsonl(tmp_path / "minutes.jsonl")
    for r in allrecs:
        schema.validate(r)
        assert r["receiver"] == "l86" and r["node_id"] == "test-node" and r["sats_tracked"] == 14
    prov = [r for r in allrecs if r["status"] == "provisional"]
    recs = schema.finals(allrecs)
    assert len(recs) == 3 and all(r["rev"] == 2 for r in recs) and all(r["rev"] == 1 for r in prov)
    assert {r["minute_utc"] for r in recs} <= {r["minute_utc"] for r in prov}      # provisional came first
    assert sat(prov[1], "GPS", 12)["max_fade_db"] >= 8                              # seen provisionally too
    assert [r["minute_utc"] for r in recs] == ["2026-10-22T14:00:00Z", "2026-10-22T14:01:00Z", "2026-10-22T14:02:00Z"]
    # injected fade: GPS 12, sim 70-110 s, -10 dB -> minute 14:01 only
    g12 = sat(recs[1], "GPS", 12)
    assert g12["max_fade_db"] >= 8 and g12["fade_count"] == 1 and g12["class"] != "none"
    assert sat(recs[0], "GPS", 12)["fade_count"] == 0 and sat(recs[2], "GPS", 12)["fade_count"] == 0
    assert all(s["fade_count"] == 0 for s in recs[1]["satellites"] if (s["sys"], s["svid"]) != ("GPS", 12))
    # garbage at 30 s is counted, not fatal; the 10 s dropout is not a lock loss
    assert any(w.startswith("bad_checksum") for w in recs[0]["warnings"])
    assert all(s["lock_loss"] == 0 for s in recs[2]["satellites"])
    # latest.json mirrors the last record; raw NMEA saved and re-runnable in batch
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["minute_utc"] == max(r["minute_utc"] for r in allrecs)            # newest minute, not an old final
    assert all(r["l86_rate_hz"] == 1.0 for r in recs[:2])
    raw = tmp_path / "raw" / "nmea_2026-10-22.nmea"
    lg = nmea_l86.read_l86(raw, fast_cfg)
    assert lg.quality["format"] == "iso_prefix" and lg.quality["epochs"] >= 600
    from click.testing import CliRunner

    from ionwatch.cli import cli
    res = CliRunner().invoke(cli, ["validate", str(tmp_path / "minutes.jsonl")])
    assert res.exit_code == 0 and "PASS" in res.output, res.output


def test_live_survives_unplug_replug(tmp_path, fast_cfg):
    p, port = start_sim("--duration-s", "850", "--inject-disconnect", "100:30")
    try:
        live.run_live(port, 9600, tmp_path, fast_cfg, "test-node", clock_scale=RATE, max_minutes=4, duration_s=120)
    finally:
        stop(p)
    recs = schema.finals(schema.read_jsonl(tmp_path / "minutes.jsonl"))
    assert [r["minute_utc"][11:16] for r in recs] == ["14:00", "14:01", "14:02", "14:03"]
    assert "serial_disconnected" in recs[1]["warnings"] and "serial_reconnected" in recs[1]["warnings"]
    assert recs[3]["sats_tracked"] == 14 and not recs[3]["warnings"]       # fully recovered
    assert "lost" in (tmp_path / "live.log").read_text(encoding="utf-8")


def test_live_without_receiver_keeps_running(tmp_path, fast_cfg):
    """Nothing on the port: no crash, the open failures are logged."""
    n = live.run_live("socket://127.0.0.1:9", 9600, tmp_path, fast_cfg, "test-node", duration_s=1.5)
    assert n == 0
    assert "unavailable" in (tmp_path / "live.log").read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="pty needs Linux/macOS (the Pi)")
def test_live_over_pty(tmp_path, fast_cfg):
    p, port = start_sim("--duration-s", "700", transport="pty")
    try:
        live.run_live(port, 9600, tmp_path, fast_cfg, "test-node", clock_scale=RATE, max_minutes=1, duration_s=90)
    finally:
        stop(p)
    fin = schema.finals(schema.read_jsonl(tmp_path / "minutes.jsonl"))
    assert len(fin) == 1 and fin[0]["sats_tracked"] == 14


def _compare_replay_batch(src, tmp_path, cfg):
    live.run_replay([str(src)], 0, tmp_path / "replay", cfg, "test-node")
    allrep = schema.read_jsonl(tmp_path / "replay" / "minutes.jsonl")
    for r in allrep:
        schema.validate(r)
    pipeline.run([str(src)], None, tmp_path / "batch", cfg, plots=False, node_id="test-node")
    bat = schema.read_jsonl(tmp_path / "batch" / "minutes.jsonl")
    fin = schema.finals(allrep)
    prov = [r for r in allrep if r["status"] == "provisional"]
    assert [r["minute_utc"] for r in fin] == [r["minute_utc"] for r in bat] == [r["minute_utc"] for r in prov]

    def classes(recs):
        return {(r["minute_utc"], s["sys"], s["svid"], s["band"]): s["class"] for r in recs for s in r["satellites"]}

    assert classes(fin) == classes(bat)                                   # exactly the batch classes
    for f, b in zip(fin, bat):
        for k in ("sats_tracked", "navic_status", "navic_sats", "warnings", "agc_flag"):
            assert f[k] == b[k], (f["minute_utc"], k)
    return fin, bat


def test_replay_finals_equal_batch_on_20oct_tail(tmp_path, cfg):
    """End of Raw, the 88-minute silence and Status bursts: final classes == batch classes."""
    fin, _ = _compare_replay_batch(BAD_TAIL, tmp_path, cfg)
    assert sum("no_data" in r["warnings"] for r in fin) > 80
    assert any(s["class"] == "edge_of_gap" for r in fin for s in r["satellites"])


FULL_20OCT = ROOT / "data" / "raw" / "gnss_log_2026_10_20_18_58_37.txt"


@pytest.mark.skipif(not FULL_20OCT.exists(), reason="full 20 Oct log not present (data/raw is not in git)")
def test_replay_finals_equal_batch_on_full_20oct(tmp_path, cfg):
    fin, bat = _compare_replay_batch(FULL_20OCT, tmp_path, cfg)
    assert len(fin) == 200


def test_live_edge_guard_sees_open_silence(cfg):
    """A fade sagging into a receiver silence is edge_of_gap in live mode too, although the
    silence has not ended yet when the minute is closed (open gap up to the clock)."""
    from conftest import synth_obs
    obs = synth_obs(n_sats=3, minutes=20)
    t = obs["time_utc"]
    sag = (obs["svid"] == 1) & (t >= t.max() - pd.Timedelta(seconds=150))
    obs.loc[sag, "cn0_dbhz"] -= pd.Series(range(int(sag.sum())), index=obs.index[sag]) * (10 / sag.sum())
    obs = obs.assign(receiver="l86", source="nmea", signal="L1", band="L1")
    e = live.MinuteEngine("l86", "x", cfg)
    e.add(obs)
    last_minute = f"{t.max():%Y-%m-%dT%H:%M}:00Z"
    recs = e.close_ready(t.max() + pd.Timedelta(minutes=5)) + e.close_ready(t.max() + pd.Timedelta(minutes=40))
    for status in ("provisional", "final"):
        r = next(r for r in recs if r["minute_utc"] == last_minute and r["status"] == status)
        s1 = sat(r, "GPS", 1)
        assert s1["class"] == "edge_of_gap" and s1["fade_count"] == 0, status


def test_replay_speed_paces(tmp_path, cfg):
    """--speed paces the data: the 3-minute fixture at 60x takes ~3 s of wall time, not 0."""
    import time

    from conftest import BAD
    t = time.monotonic()
    live.run_replay([str(BAD)], 60, tmp_path, cfg, "test-node")
    assert time.monotonic() - t >= 2.5
    recs = schema.read_jsonl(tmp_path / "minutes.jsonl")
    assert len(schema.finals(recs)) == len(recs) // 2 >= 3


def test_clock_jump_only_before_data(cfg):
    e = live.MinuteEngine("l86", "x", cfg)
    e.start_at(pd.Timestamp("2026-01-01T00:00:00Z"), from_wall_clock=True)
    obs = pd.DataFrame({"time_utc": [pd.Timestamp("2026-10-22T14:00:05Z")], "receiver": "l86", "source": "nmea",
                        "constellation": "GPS", "svid": 5, "signal": "L1", "band": "L1", "elev_deg": 50.0,
                        "az_deg": 10.0, "cn0_dbhz": 40.0, "time_quality": "nmea"})
    e.add(obs)
    assert e.next_minute == pd.Timestamp("2026-10-22T14:00:00Z")
    e.add(obs.assign(time_utc=pd.Timestamp("2026-10-22T16:00:05Z")))   # a real gap: no jump
    assert e.next_minute == pd.Timestamp("2026-10-22T14:00:00Z")


# ------------------------------------------------------------------ L86 rate setup (--l86-rate)

def _rate_run(tmp_path, cfg, rate, *sim_args, seconds=18):
    p, port = start_sim("--duration-s", "400", *sim_args, speed=10)
    try:
        live.run_live(port, 9600, tmp_path, cfg, "test-node", clock_scale=10, duration_s=seconds, l86_rate=rate)
    finally:
        stop(p)
    return schema.read_jsonl(tmp_path / "minutes.jsonl"), (tmp_path / "live.log").read_text(encoding="utf-8")


def test_l86_rate_10hz_verified(tmp_path, fast_cfg):
    recs, logtxt = _rate_run(tmp_path, fast_cfg, 10)
    assert "L86 configured: 115200 baud" in logtxt and "ack PMTK220: ok" in logtxt
    assert "requested 10 Hz): ok" in logtxt
    assert any(w.startswith("l86_rate_measured:") for r in recs for w in r["warnings"])
    assert recs[-1]["l86_rate_hz"] >= 9


def test_l86_rate_falls_back_to_1hz(tmp_path, fast_cfg):
    """The module tops out at 5 Hz: measured < 80 % of 10 Hz -> back to 1 Hz at 9600 with a warning."""
    recs, logtxt = _rate_run(tmp_path, fast_cfg, 10, "--max-rate", "5")
    assert "back to 1 Hz at 9600 baud" in logtxt
    assert any("l86_rate_fallback" in r["warnings"] for r in recs)
    assert recs[-1]["l86_rate_hz"] == 1.0


def test_l86_rate_config_resent_after_power_loss(tmp_path, fast_cfg):
    """Unplug = power loss: the simulator forgets 115200/5 Hz; live mode reconfigures on reopen."""
    recs, logtxt = _rate_run(tmp_path, fast_cfg, 5, "--inject-disconnect", "100:20", seconds=30)
    assert logtxt.count("L86 configured: 115200 baud") >= 2
    after = [r for r in recs if r["minute_utc"] >= "2026-10-22T14:03:00Z"]
    assert after and after[-1]["l86_rate_hz"] >= 4
