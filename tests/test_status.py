"""D3: `ionwatch status` lines, including the STALE line."""
import os
import time

import pandas as pd
from click.testing import CliRunner
from conftest import synth_obs

from ionwatch import live
from ionwatch.cli import cli
from ionwatch.status import status_line


def _live_folder(path, cfg):
    obs = synth_obs(n_sats=6, minutes=4, t0="2026-10-22T14:00:00Z", cn0=38.0).assign(
        receiver="l86", source="nmea", signal="L1", band="L1")
    eng, w = live.MinuteEngine("l86", "haldwani-01", cfg), live.RecordWriter(path)
    eng.add(obs)
    for r in eng.close_ready(obs["time_utc"].max() + pd.Timedelta(seconds=10)):
        w.write(r)
    return path


def test_status_line_fresh(tmp_path, cfg):
    _live_folder(tmp_path, cfg)
    now = pd.Timestamp("2026-10-22T14:05:30Z")
    line, stale = status_line(tmp_path, now=now)
    assert not stale
    assert line.startswith("19:35:30 IST | minute 19:33 IST provisional")
    assert "sats 6" in line and "C/N0 38.0 dB-Hz" in line and "L86 1.0 Hz" in line and "age " in line


def test_status_line_stale(tmp_path, cfg):
    _live_folder(tmp_path, cfg)
    old = time.time() - 300
    for name in ("latest.json", "minutes.jsonl"):
        os.utime(tmp_path / name, (old, old))
    line, stale = status_line(tmp_path)
    assert stale and "STALE: last record 5.0 min old" in line


def test_status_empty_folder(tmp_path):
    line, stale = status_line(tmp_path)
    assert stale and "no records yet" in line


def test_status_cli(tmp_path, cfg):
    _live_folder(tmp_path, cfg)
    res = CliRunner().invoke(cli, ["status", "--data", str(tmp_path), "--count", "2", "--every", "0"])
    assert res.exit_code == 0 and res.output.count("IST |") == 2
