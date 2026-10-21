"""L86 NMEA parser, L86 pipeline path and cross-check, on small synthetic fixtures (no real L86 log yet)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from conftest import GOOD

from ionwatch import pipeline
from ionwatch.io import gnsslogger, nmea_l86
from ionwatch.proc import crosscheck


def nmea(body: str) -> str:
    x = 0
    for ch in body:
        x ^= ord(ch)
    return f"${body}*{x:02X}"


def rmc(t: datetime) -> str:
    return nmea(f"GPRMC,{t:%H%M%S}.{t.microsecond // 1000:03d},A,2915.336,N,07932.220,E,0.0,0.0,{t:%d%m%y},,,A")


def gga(t: datetime) -> str:
    return nmea(f"GPGGA,{t:%H%M%S}.{t.microsecond // 1000:03d},2915.336,N,07932.220,E,1,08,1.0,500.0,M,-40.0,M,,")


def gsv(talker: str, sats: list[tuple], sig: str | None = None) -> list[str]:
    """sats: (prn, elev, az, snr_or_'') -> one or more GSV sentences (4 sats each)."""
    chunks = [sats[i:i + 4] for i in range(0, len(sats), 4)]
    out = []
    for k, ch in enumerate(chunks, 1):
        body = f"{talker}GSV,{len(chunks)},{k},{len(sats):02d}," + ",".join(
            f"{p:02d},{e},{a:03d},{s}" for p, e, a, s in ch)
        out.append(nmea(body + (f",{sig}" if sig else "")))
    return out


T0 = datetime(2026, 10, 20, 14, 0, 0, tzinfo=timezone.utc)   # 19:30 IST


def cycle(t, gp_sats, extra=()):
    return [rmc(t), gga(t), *gsv("GP", gp_sats), *extra]


@pytest.fixture
def simple(cfg):
    sats = [(5, 60, 45, 40), (12, 45, 120, 38), (19, 30, 200, 35), (25, 70, 300, ""), (29, 25, 10, 33),
            (2, 50, 90, 41)]
    lines = []
    for s in range(5):
        lines += cycle(T0 + timedelta(seconds=s), sats)
    return nmea_l86.parse_lines(lines, cfg, "synthetic")


def test_checksum():
    assert nmea_l86.checksum_ok(nmea("GPGGA,1,2,3"))
    assert not nmea_l86.checksum_ok("$GPGGA,1,2,3*00")


def test_multi_sentence_gsv_assembly(simple):
    t = simple.table
    assert simple.quality["gsv_groups"] == 5 and simple.quality["gsv_incomplete"] == 0
    assert len(t) == 5 * 6
    assert t.groupby("time_utc").size().eq(6).all()


def test_empty_snr_is_nan(simple):
    t = simple.table
    assert t.loc[t["prn_raw"] == 25, "cn0_dbhz"].isna().all()
    assert t.loc[t["prn_raw"] == 5, "cn0_dbhz"].eq(40).all()


def test_timestamps_from_rmc(simple):
    times = sorted(simple.table["time_utc"].unique())
    assert times[0] == pd.Timestamp(T0) and times[-1] == pd.Timestamp(T0 + timedelta(seconds=4))
    assert set(simple.table["time_source"]) == {"nmea"}
    assert simple.quality["format"] == "plain"


def test_bad_checksum_and_noise_dropped(cfg):
    good = cycle(T0, [(5, 60, 45, 40)])
    bad = good[2][:-2] + "00"
    lines = [good[0], good[1], bad, "garbage line", nmea("PMTK001,220,3"), *cycle(T0 + timedelta(seconds=1),
                                                                                  [(5, 60, 45, 41)])]
    lg = nmea_l86.parse_lines(lines, cfg)
    q = lg.quality
    assert q["bad_checksum"] == 1 and q["noise"] == 1 and q["pmtk"] == 1
    assert len(lg.table) == 1 and lg.table["cn0_dbhz"].iloc[0] == 41


def test_host_prefix_formats_and_offset(cfg):
    lines = []
    for s in range(3):
        t = T0 + timedelta(seconds=s)
        host = t + timedelta(milliseconds=250)
        for sent in cycle(t, [(5, 60, 45, 40)]):
            lines.append(f"{host:%Y-%m-%dT%H:%M:%S}.{host.microsecond // 1000:03d}Z,{sent}")
    q = nmea_l86.parse_lines(lines, cfg).quality
    assert q["format"] == "iso_prefix"
    assert q["host_minus_nmea_offset_s"]["median"] == pytest.approx(0.25, abs=0.01)
    epoch = [f"{(T0 + timedelta(seconds=1)).timestamp():.3f} {s}" for s in cycle(T0, [(5, 60, 45, 40)])]
    assert nmea_l86.parse_lines(epoch, cfg).quality["format"] == "epoch_prefix"


def test_gsv_before_any_time_uses_host_or_is_dropped(cfg):
    lg = nmea_l86.parse_lines(gsv("GP", [(5, 60, 45, 40)]), cfg)
    assert lg.quality["gsv_no_time"] == 1 and lg.table.empty


def test_prn_normalisation(cfg):
    lines = [rmc(T0), *gsv("GL", [(65, 40, 10, 30), (72, 50, 20, 31)]),
             *gsv("GA", [(301, 40, 30, 32), (7, 45, 40, 33)]), *gsv("GP", [(40, 40, 50, 34)])]
    t = nmea_l86.parse_lines(lines, cfg).table
    pairs = set(zip(t["constellation"], t["prn_raw"], t["svid"]))
    assert {("GLONASS", 65, 1), ("GLONASS", 72, 8), ("Galileo", 301, 1), ("Galileo", 7, 7),
            ("SBAS", 40, 40)} == pairs


def test_talkers_gi_navic_pq_gq_qzss(cfg):
    """$GI -> NavIC (parsed for other receivers; the L86-M33 / MT3333 does not track NavIC),
    $PQ and $GQ -> QZSS."""
    lines = [rmc(T0), *gsv("GI", [(2, 59, 160, 27), (9, 30, 250, 26)], sig="1"),
             *gsv("PQ", [(1, 40, 120, 25)]), *gsv("GQ", [(195, 45, 130, 26)])]
    lg = nmea_l86.parse_lines(lines, cfg)
    t = lg.table
    nav = t[t["constellation"] == "NavIC"]
    assert set(nav["svid"]) == {2, 9} and set(nav["signal"]) == {"L5"} and set(nav["talker"]) == {"GI"}
    qzs = t[t["constellation"] == "QZSS"]
    assert set(zip(qzs["talker"], qzs["svid"])) == {("PQ", 193), ("GQ", 195)}


def test_gsa_marks_used_in_fix(cfg):
    lines = [rmc(T0), nmea("GPGSA,A,3,05,12,,,,,,,,,,,1.8,1.0,1.5"), *gsv("GP", [(5, 60, 45, 40), (19, 30, 200, 35)])]
    t = nmea_l86.parse_lines(lines, cfg).table
    assert dict(zip(t["svid"], t["used_in_fix"])) == {5: True, 19: False}


def test_sbas_excluded_from_indices(cfg):
    lg = nmea_l86.parse_lines([rmc(T0), *gsv("GP", [(5, 60, 45, 40), (40, 40, 50, 34)])], cfg)
    assert "SBAS" in set(lg.table["constellation"]) and "SBAS" not in set(lg.obs["constellation"])


def test_10hz_output_with_1hz_cn0_sets_effective_rate(cfg):
    lines = []
    for k in range(300):                       # 30 s at 10 Hz; C/N0 changes once per second
        t = T0 + timedelta(milliseconds=100 * k)
        lines += cycle(t, [(5, 60, 45, 40 + (k // 10) % 3), (12, 45, 120, 35 + (k // 10) % 2)])
    q = nmea_l86.parse_lines(lines, cfg).quality
    assert q["output_rate_hz"] == pytest.approx(10, rel=0.01)
    assert q["effective_rate_hz"] == pytest.approx(1.0, abs=0.15)
    assert any(w["code"] == "l86_low_effective_rate" for w in q["warnings"])


# ------------------------------------------------------------------ cross-check + pipeline path

def _phone_like(svid, t, cn0, receiver, signal, source):
    return pd.DataFrame({"time_utc": t, "receiver": receiver, "source": source, "constellation": "GPS",
                         "svid": svid, "signal": signal, "band": "L1", "elev_deg": 60.0, "az_deg": 45.0,
                         "cn0_dbhz": cn0, "time_quality": "logged"})


def test_crosscheck_confirms_common_fade(cfg):
    from ionwatch.proc import fades
    t = pd.date_range("2026-10-20T14:00:00Z", periods=40 * 60, freq="1s")
    rng = np.random.default_rng(3)
    # Starts mid-minute so the per-minute correlation sees the step on both receivers.
    common_signal = np.where((t >= t[0] + pd.Timedelta(minutes=20, seconds=30))
                             & (t < t[0] + pd.Timedelta(minutes=22, seconds=30)), -9, 0)
    l86 = _phone_like(5, t, 40 + common_signal + rng.normal(0, 0.3, len(t)), "l86", "L1", "nmea")
    ph = _phone_like(5, t, 30 + common_signal + rng.normal(0, 0.5, len(t)), "phone", "GPS_L1", "raw")
    act = np.unique(t.astype("int64") // 10**9)
    parts = []
    for o in (l86, ph):
        parts.append(fades.add_baseline(o, act, cfg, 1.0))
    obs_b = pd.concat(parts, ignore_index=True)
    m = pd.concat([fades.per_minute(p, act, cfg, 1.0) for p in parts], ignore_index=True)
    ev = fades.events(m)
    xc, conf = crosscheck.compute(obs_b, ev, cfg)
    assert len(conf) == 1 and conf["satellite"].iloc[0] == "G05"
    assert conf["l86_peak_db"].iloc[0] > 6 and conf["phone_peak_db"].iloc[0] > 6
    assert xc["corr_detrended"].max() > 0.8


def test_pipeline_with_l86_and_phone(tmp_path, cfg):
    """--l86 path end to end: synthetic L86 over the good phone fixture's time span."""
    raw = gnsslogger.read_phone([GOOD], cfg).raw
    t0 = raw["time_utc"].min().ceil("s").to_pydatetime()
    lines = []
    for s in range(170):
        lines += cycle(t0 + timedelta(seconds=s), [(3, 60, 45, 40 + s % 2), (8, 50, 90, 38), (9, 45, 200, 36)])
    l86_file = tmp_path / "l86.nmea"
    l86_file.write_text("\n".join(lines))
    out = tmp_path / "session"
    res = pipeline.run([str(GOOD)], str(l86_file), out, cfg, plots=False)
    assert res["receivers"] == ["phone", "l86"]
    for name in ("l86_obs", "crosscheck_1min", "confirmed_events"):
        assert (out / f"{name}.csv").exists()
    assert len(res["tables"]["crosscheck_1min"]) > 0
    assert "not yet validated on real data" in (out / "summary.md").read_text(encoding="utf-8")
    assert (out / "l86_quality.json").exists()


def test_pipeline_l86_only(tmp_path, cfg):
    lines = []
    for s in range(130):
        lines += cycle(T0 + timedelta(seconds=s), [(5, 60, 45, 40), (12, 45, 120, 38)])
    f = tmp_path / "l86.nmea"
    f.write_text("\n".join(lines))
    res = pipeline.run(None, str(f), tmp_path / "l86only", cfg, plots=True)
    assert res["receivers"] == ["l86"] and "crosscheck" in res["notes"]
    assert len(res["tables"]["s4_1min"]) > 0
