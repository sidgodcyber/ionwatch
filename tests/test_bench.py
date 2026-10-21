"""E: benchmark downloader / validator / ROTI / IONEX / report, on small real offline fixtures.

Set IONWATCH_NET=1 to also run the live download test against the real archives."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from conftest import BAD_TAIL

from ionwatch import pipeline
from ionwatch.bench import fetch, ionex, rinex, roti, spaceweather, validate
from ionwatch.cli import cli

FX = Path(__file__).parent / "fixtures" / "bench"
DATE = pd.Timestamp("2026-10-20", tz="UTC")


def test_crx_decoder_matches_crx2rnx():
    """Pure-Python CRINEX decoding equals RNXCMP crx2rnx on a real LCK4 prefix (all systems, all types)."""
    mine = rinex.crx_to_rnx(rinex.open_text(FX / "lck4_prefix.crx"))
    h, _ = rinex._header(mine)
    keep = dict(h["obs_types"])
    tmp = FX / "_mine.rnx"
    tmp.write_text("\n".join(mine) + "\n")
    try:
        _, a = rinex.read_obs(tmp, keep)
    finally:
        tmp.unlink()
    _, b = rinex.read_obs(FX / "lck4_prefix_ref.rnx", keep)
    assert len(a) == len(b) > 500
    assert list(a["sat"]) == list(b["sat"]) and list(a["time_gps"]) == list(b["time_gps"])
    for c in a.columns[2:]:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        assert np.all((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) < 1e-6)), c


def test_gps_orbit_radius():
    nav = rinex.read_gps_nav(FX / "brdm_gps_afternoon.rnx")
    assert nav["sat"].nunique() >= 30
    e = nav.iloc[0]
    xyz = rinex.gps_position(e, np.array([e["toe"], e["toe"] + 3600]))
    r = np.linalg.norm(xyz, axis=1) / 1e3
    assert np.all((r > 26000) & (r < 27000))                  # GPS orbit radius ~26 560 km


def test_roti_separates_bubbles_from_quiet(cfg):
    """Real 20 Oct 2026 night: IISC (13 N) shows post-sunset bubbles, LCK4 (Lucknow) is quiet."""
    nav = rinex.read_gps_nav(FX / "brdm_gps_afternoon.rnx")
    out = {}
    for st in ("IISC", "LCK4"):
        h, o = rinex.read_obs(FX / f"{st}_night.rnx", {"G": ["L1C", "L2W"]})
        r = roti.compute(o, h, nav, cfg)
        assert len(r) and r["elev_deg"].min() >= cfg["bench"]["elev_mask_deg"]
        out[st] = roti.station_series(r)["roti_max"]
    assert out["IISC"].max() > 1.0 and (out["IISC"] > 0.5).sum() >= 5
    assert out["LCK4"].max() < 0.3
    lat, lon, _ = roti.ecef_to_geodetic(rinex._header(rinex.open_text(FX / "LCK4_night.rnx"))[0]["xyz"])
    assert abs(lat - 26.91) < 0.01 and abs(lon - 80.96) < 0.01


def test_ionex_offset_dcb_and_producer_agreement():
    jpl, upc = ionex.read(FX / "JPLR2920.26I.gz"), ionex.read(FX / "UPC0OPSRAP_20262920000_01D_02H_GIM.INX.Z")
    assert jpl["offset_tecu"] == 2.0 and len(jpl["dcb_ns"]) >= 30 and upc["offset_tecu"] == 0.0
    a, b = ionex.at(jpl, 29.256, 79.537), ionex.at(upc, 29.256, 79.537)
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    assert len(j) >= 12 and j.corr().iloc[0, 1] > 0.9 and abs((j["a"] - j["b"]).mean()) < 8
    assert 0 < a.min() and a.max() < 100


def test_space_weather_cross_checks(tmp_path):
    for f in ("kp_swpc.json", "kp_gfz.json", "f107_swpc.json", "f107_gfz.json"):
        shutil.copy(FX / f, tmp_path / f)
    man = {"items": {n: {"ok": True, "file": f"{n}.json"} for n in ("kp_swpc", "kp_gfz", "f107_swpc", "f107_gfz")}}
    checks = {c["check"].split(" (")[0].split(" vs")[0]: c for c in validate.cross_checks(tmp_path, man, DATE)}
    kp = next(c for k, c in checks.items() if k.startswith("Kp"))
    assert kp["ok"] and kp["n"] == 8 and 0 < kp["max_abs_diff"] <= 0.67     # NOAA estimated vs GFZ Kp
    f = next(c for k, c in checks.items() if k.startswith("F10.7"))
    assert f["ok"] and abs(f["swpc"] - 121.0) < 0.1 and abs(f["gfz"] - 120.8) < 0.1
    assert validate.check("kp", tmp_path / "kp_gfz.json", DATE, "kp_gfz")["ok"]
    assert validate.check("kp", tmp_path / "kp_swpc.json", DATE, "kp_swpc")["info"]["max"] == 3.0


class FakeResponse:
    def __init__(self, url, status=200, content=b"", html=False):
        self.url, self.status_code, self.content = url, status, content
        self.headers = {"content-type": "text/html" if html else "application/octet-stream"}


class FakeSession:
    """CDDIS answers with the Earthdata login page; mirrors serve bytes or 404."""

    def __init__(self, files):
        self.files, self.calls = files, []

    def get(self, url, timeout=None, allow_redirects=True):
        self.calls.append(url)
        if "cddis.nasa.gov" in url:
            return FakeResponse("https://urs.earthdata.nasa.gov/oauth/authorize?x", 200, b"<html>login</html>", html=True)
        for key, data in self.files.items():
            if key in url:
                return FakeResponse(url, 200, data)
        return FakeResponse(url, 404)


def test_fetch_falls_back_from_cddis_and_uses_a_second_gim_producer(tmp_path, monkeypatch):
    files = {"LCK400IND": b"x", "BRDM00DLR": b"x", "JPLR2930": b"x", "UPC0OPSRAP_2026293": b"x",
             "noaa-planetary-k-index": b"[]", "f107_cm_flux": b"[]"}
    monkeypatch.setattr(validate, "check", lambda kind, path, date, name="": {"ok": True, "problems": [], "info": {}})
    monkeypatch.setattr(validate, "cross_checks", lambda folder, man, date: [])
    sess = FakeSession(files)
    m = fetch.fetch(DATE, tmp_path, sess=sess)
    it = m["items"]
    assert it["obs_LCK4"]["ok"] and "bkg.bund.de" in it["obs_LCK4"]["url"]
    assert it["obs_LCK3"]["tried"][0]["status"].startswith("Earthdata login required")
    assert not it["obs_LCK3"]["ok"] and not it["bias"]["ok"]
    assert "JPLR2930" in it["ionex"]["url"] and "UPC0OPSRAP" in it["ionex_check"]["url"]   # different producers
    assert m["stations_ok"] == ["LCK4"] and m["missing_required"] == []
    assert json.loads((tmp_path / "2026-293" / "manifest.json").read_text())["doy"] == 293
    assert not any("password" in u or "@" in u.split("//")[1].split("/")[0] for u in sess.calls)   # no credentials in URLs


def test_validation_rejects_wrong_files(tmp_path):
    bad = tmp_path / "x.rnx"
    bad.write_text("not rinex\n")
    assert not validate.check("obs", bad, DATE, "obs_LCK4")["ok"]
    assert not validate.check("ionex", bad, DATE, "ionex")["ok"]
    v = validate.check("obs", FX / "IISC_night.rnx", DATE, "obs_LCK4")
    assert not v["ok"] and any("marker" in p for p in v["problems"])        # wrong station is caught
    assert any("epochs" in p for p in v["problems"])                        # 2 h is not a daily file


@pytest.fixture(scope="module")
def offline_bench(tmp_path_factory, cfg):
    """A bench folder built from the fixtures + a processed session of the 20 Oct night."""
    root = tmp_path_factory.mktemp("bench")
    folder = root / "bench" / "2026-293"
    folder.mkdir(parents=True)
    names = {"obs_LCK4": "LCK4_night.rnx", "obs_IISC": "IISC_night.rnx", "nav_BRDM": "brdm_gps_afternoon.rnx",
             "kp_swpc": "kp_swpc.json", "kp_gfz": "kp_gfz.json", "f107_swpc": "f107_swpc.json", "f107_gfz": "f107_gfz.json"}
    items = {}
    for n, f in names.items():
        shutil.copy(FX / f, folder / f)
        items[n] = {"ok": True, "file": f, "url": f"fixture:{f}", "sha256": "fixture", "status": "ok",
                    "validation": {"ok": True, "info": {}}}
    for n in ("obs_LCK3", "obs_HYDE", "ionex", "ionex_check", "bias"):
        items[n] = {"ok": False, "status": "not in the offline fixture"}
    man = {"date_utc": "2026-10-20", "doy": 293, "items": items}
    man["cross_checks"] = validate.cross_checks(folder, man, DATE)
    (folder / "manifest.json").write_text(json.dumps(man))
    processed = root / "processed"
    pipeline.run([str(BAD_TAIL)], None, processed, cfg, plots=False)
    return root, processed


def test_bench_run_offline(offline_bench, cfg):
    root, processed = offline_bench
    res = CliRunner().invoke(cli, ["bench", "run", "--bench", str(root / "bench"), "--processed", str(processed)])
    assert res.exit_code == 0, res.output
    folder = root / "bench" / "2026-293"
    md = (folder / "bench_summary.md").read_text(encoding="utf-8")
    s = json.loads((folder / "bench_summary.json").read_text())
    assert s["night_ist"][0].endswith("19:30 IST") and s["doy"] == 293
    st = {x["station"]: x for x in s["stations"]}
    assert st["LCK4"]["windows_above"] == 0 and st["IISC"]["windows_above"] > 0
    assert 290 < st["LCK4"]["distance_km"] < 300
    assert s["kp"]["source"] == "GFZ Potsdam" and s["kp"]["max"] == pytest.approx(3.333)
    assert "No irregularities at the nearest IGS station LCK4" in md
    assert md.splitlines()[2] == ("**For slides:** IGS LCK4 (295 km) quiet, max ROTI 0.12 TECU/min; IISC (Bangalore) "
                                  "saw post-sunset irregularities from 20:15 IST; Kp ≤ 3.3.")
    assert "Kp SWPC vs GFZ" in md and "**OK**" in md
    assert (folder / "bench_roti_events.png").exists() and (folder / "roti_5min.csv").exists()
    assert s["events"], "the 20 Oct tail has IonWatch events to compare"


def test_bench_dates_from_session(offline_bench, cfg):
    from ionwatch.bench.run import session_info
    _, processed = offline_bench
    info = session_info(processed, cfg)
    assert [f"{d:%Y-%m-%d}" for d in info["dates"]] == ["2026-10-20"]
    assert info["night"][0] == pd.Timestamp("2026-10-20T14:00:00Z") and info["night"][1] == pd.Timestamp("2026-10-20T19:30:00Z")


@pytest.mark.skipif(os.environ.get("IONWATCH_NET") != "1", reason="set IONWATCH_NET=1 for live archive downloads")
def test_live_fetch_20oct(tmp_path):
    m = fetch.fetch(DATE, tmp_path)
    assert "LCK4" in m["stations_ok"] and m["items"]["nav_BRDM"]["ok"] and m["items"]["ionex"]["ok"]
