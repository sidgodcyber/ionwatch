"""D1: batch join of a phone log with the daily NMEA file written by live mode."""
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import GOOD

from ionwatch import live, pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from nmea_from_phone import phone_to_nmea  # noqa: E402

SPEED = 40


@pytest.fixture(scope="module")
def live_raw(tmp_path_factory, cfg):
    """Phone-derived L86 NMEA -> fake_l86 (pty/tcp) -> live mode -> data/live/raw/nmea_<day>.nmea."""
    import copy
    tmp = tmp_path_factory.mktemp("join")
    fixture = tmp / "l86_from_phone.nmea"
    n, t0 = phone_to_nmea(GOOD, fixture, cfg)
    p = subprocess.Popen([sys.executable, str(ROOT / "tools" / "fake_l86.py"), "--transport", "tcp",
                          "--fixture", str(fixture), "--speed", str(SPEED),
                          "--start", f"{t0:%Y-%m-%dT%H:%M:%SZ}", "--duration-s", str(n)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    port = p.stdout.readline().split()[1]
    c = copy.deepcopy(cfg)
    c["live"]["reconnect_s"] = 0.2
    try:
        live.run_live(port, 9600, tmp / "live", c, "test-node", clock_scale=SPEED, duration_s=n / SPEED + 4)
    finally:
        p.kill()
        p.wait(timeout=10)
    raw = tmp / "live" / "raw" / f"nmea_{t0:%Y-%m-%d}.nmea"
    assert raw.exists()
    return tmp, raw


def test_join_live_daily_file_with_phone(live_raw, cfg):
    tmp, raw = live_raw
    first = raw.read_text(encoding="ascii").splitlines()[0]
    assert first[23:25] == "Z," and first[25:].startswith("$")              # host-time prefixed, as live writes
    res = pipeline.run([str(GOOD)], str(raw), tmp / "batch", cfg, plots=False)
    assert res["receivers"] == ["phone", "l86"]
    xc = res["crosscheck"]
    assert len(xc["matched_satellites"]) >= 10
    assert {s[0] for s in xc["matched_satellites"]} == {"G", "R", "E"}      # GPS, GLONASS slots, Galileo
    assert xc["time_offset"]["lag_s"] == 0 and xc["time_offset"]["corr_at_lag"] > 0.9
    md = (tmp / "batch" / "summary.md").read_text(encoding="utf-8")
    assert "clock offset from C/N0 cross-correlation: 0 s" in md
    recs = (tmp / "batch" / "minutes.jsonl").read_text(encoding="utf-8")
    assert '"receiver": "l86"' in recs and '"receiver": "phone"' in recs


def test_join_detects_a_clock_offset(tmp_path, cfg):
    """A 3 s L86 clock error is found by the cross-check (negative test of the zero-offset claim)."""
    f = tmp_path / "shifted.nmea"
    phone_to_nmea(GOOD, f, cfg, shift_s=3)
    res = pipeline.run([str(GOOD)], str(f), tmp_path / "out", cfg, plots=False)
    assert res["crosscheck"]["time_offset"]["lag_s"] == 3
