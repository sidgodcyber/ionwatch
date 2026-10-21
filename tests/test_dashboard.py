"""Dashboard API and page: live, replay and empty folders, no_data minutes."""
import json

import pandas as pd
import pytest
from conftest import BAD, BAD_TAIL, synth_obs
from fastapi.testclient import TestClient

from ionwatch import live, schema
from ionwatch.dashboard.server import create_app


@pytest.fixture(scope="module")
def replay_dir(tmp_path_factory, cfg):
    out = tmp_path_factory.mktemp("replay")
    live.run_replay([str(BAD_TAIL)], 0, out, cfg, "haldwani-01")
    return out


@pytest.fixture(scope="module")
def live_dir(tmp_path_factory, cfg):
    """A live-mode folder written by the live engine/writer from synthetic L86 data."""
    out = tmp_path_factory.mktemp("live")
    live.write_run_info(out, mode="live", node_id="haldwani-01", receiver="l86", port="/dev/serial0")
    obs = synth_obs(n_sats=6, minutes=15, t0="2026-10-22T14:00:00Z").assign(receiver="l86", source="nmea",
                                                                            signal="L1", band="L1")
    t = obs["time_utc"]
    obs.loc[(obs["svid"] == 2) & (t >= t.iloc[0] + pd.Timedelta(minutes=5, seconds=20))
            & (t < t.iloc[0] + pd.Timedelta(minutes=6, seconds=30)), "cn0_dbhz"] -= 10
    eng = live.MinuteEngine("l86", "haldwani-01", cfg)
    w = live.RecordWriter(out)
    eng.add(obs)
    for r in eng.close_ready(t.max() + pd.Timedelta(seconds=10)):
        w.write(r)
    return out


def client(folder):
    return TestClient(create_app(folder))


def test_page_and_assets(tmp_path):
    c = client(tmp_path)
    page = c.get("/")
    assert page.status_code == 200 and "<canvas id=\"sky\"" in page.text
    for text in ("heuristic triage, not detection", "TEC is not shown here", "indicative"):
        assert text in page.text
    assert 'name="viewport"' in page.text
    for asset in ("/static/app.js", "/static/style.css", "/static/vendor/chart.umd.min.js"):
        assert c.get(asset).status_code == 200


def test_empty_folder_is_a_valid_empty_state(tmp_path):
    c = client(tmp_path)
    st = c.get("/api/status").json()
    assert st["latest"] is None and st["receiver_status"] == "waiting for data" and st["records"] == 0
    assert c.get("/api/minutes").json() == {"records": []}
    assert c.get("/api/events").json() == {"events": []}
    assert c.get("/api/warnings").json() == {"warnings": []}
    assert c.get("/api/latest").json() is None


def test_replay_folder(replay_dir):
    c = client(replay_dir)
    st = c.get("/api/status").json()
    assert st["mode"] == "replay" and st["node_id"] == "haldwani-01" and st["receivers"] == ["phone"]
    assert st["latest"]["receiver"] == "phone" and st["final_records"] == st["records"] // 2
    recs = c.get("/api/minutes", params={"hours": 3}).json()["records"]
    assert len(recs) == st["minutes"] > 90
    for r in recs:
        schema.validate(r)
        assert r["status"] == "final"                     # the highest rev per minute is served
    assert sum("no_data" in r["warnings"] for r in recs) > 80
    warn = {w["code"]: w for w in c.get("/api/warnings", params={"hours": 3}).json()["warnings"]}
    assert "no_data" in warn and warn["no_data"]["minutes"] > 50
    ev = c.get("/api/events", params={"hours": 3}).json()["events"]
    assert ev and all(e["class"] != "none" and e["status"] in ("provisional", "final") for e in ev)
    assert all("reason" in e and e["start_ist"].endswith("IST") for e in ev)


def test_since_returns_only_newer_minutes(replay_dir):
    c = client(replay_dir)
    allr = c.get("/api/minutes", params={"hours": 3}).json()["records"]
    since = allr[-5]["minute_utc"]
    newer = c.get("/api/minutes", params={"hours": 3, "since": since}).json()["records"]
    assert [r["minute_utc"] for r in newer] == [r["minute_utc"] for r in allr[-5:]]


def test_live_folder(live_dir):
    c = client(live_dir)
    st = c.get("/api/status").json()
    assert st["mode"] == "live" and st["latest"]["receiver"] == "l86"
    assert st["latest"]["l86_rate_hz"] == 1.0 and st["latest"]["sats_tracked"] == 6
    assert st["receiver_status"] == "ok"
    latest = c.get("/api/latest").json()
    schema.validate(latest)
    recs = c.get("/api/minutes").json()["records"]
    assert [r["minute_utc"][11:16] for r in recs][:3] == ["14:00", "14:01", "14:02"]
    ev = c.get("/api/events").json()["events"]
    assert any(e["svid"] == 2 and e["band"] == "L1" for e in ev)
    assert {e["status"] for e in ev} <= {"provisional", "final"}


def test_new_replay_resets_the_store(tmp_path, replay_dir):
    import shutil
    folder = tmp_path / "r"
    shutil.copytree(replay_dir, folder)
    c = client(folder)
    assert c.get("/api/status").json()["minutes"] > 90
    (folder / "minutes.jsonl").write_text("")            # a new replay truncates the file
    (folder / "latest.json").unlink()
    st = c.get("/api/status").json()
    assert st["minutes"] == 0 and st["latest"] is None


def test_stale_live_is_flagged(live_dir, tmp_path):
    import os
    import shutil
    import time
    folder = tmp_path / "stale"
    shutil.copytree(live_dir, folder)
    old = time.time() - 600
    for name in ("latest.json", "minutes.jsonl"):
        os.utime(folder / name, (old, old))
    st = client(folder).get("/api/status").json()
    assert st["stale"] and st["receiver_status"] == "stale"


def test_latest_record_of_no_data_minute(tmp_path, cfg):
    """A receiver that has gone silent: latest record is a no_data minute."""
    live.run_replay([str(BAD)], 0, tmp_path, cfg, "haldwani-01")
    recs = schema.read_jsonl(tmp_path / "minutes.jsonl")
    empty = dict(recs[-1], minute_utc=pd.Timestamp(recs[-1]["minute_utc"]).strftime("%Y-%m-%dT%H:%M:00Z"),
                 satellites=[], sats_tracked=0, warnings=["no_data"], navic_sats=[], navic_status="no_data")
    (tmp_path / "latest.json").write_text(json.dumps(empty))
    st = client(tmp_path).get("/api/status").json()
    assert st["receiver_status"] in ("no data", "replay finished")
