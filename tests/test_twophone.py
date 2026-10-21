"""Optional second phone (--phone2): matching, both_phones flag, classes, L1-only robustness."""
import json

import pandas as pd
import pytest
from click.testing import CliRunner
from conftest import GOOD
from fastapi.testclient import TestClient

from ionwatch import pipeline, schema
from ionwatch.cli import cli
from ionwatch.dashboard.server import create_app

SHIFT_MS = 6 * 3600 * 1000            # 14:09 IST -> 20:09 IST: inside the night window
L1, L5 = "1575420000", "1176450000"


def make_log(path, fades, l1_only=False, device=None):
    """Rewrite the good fixture: shift times into the night, fade (svid, start_s, end_s, dB) on GPS L1,
    optionally make it an L1-only phone without NavIC, AGC or IsFullTracking."""
    cols, out, t0 = {}, [], None
    for line in GOOD.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            head, _, rest = line[1:].strip().partition(",")
            if rest:
                cols[head] = rest.split(",")
            if device and line.startswith("# Version:"):
                line = f"# Version: v3.1.0.4 Platform: 14 Manufacturer: OnePlus Model: {device} GNSS Hardware Model Name: test;"
            out.append(line)
            continue
        kind, _, rest = line.partition(",")
        f = rest.split(",")
        c = cols.get(kind)
        if kind == "Agc" and l1_only:
            continue
        tcol = {"Raw": "utcTimeMillis", "Agc": "utcTimeMillis", "Status": "UnixTimeMillis", "Fix": "UnixTimeMillis"}.get(kind)
        if c and tcol:
            i = c.index(tcol)
            if f[i]:
                if kind == "Raw" and t0 is None:
                    t0 = int(f[i])
                f[i] = str(int(f[i]) + SHIFT_MS)
        if c and "CarrierFrequencyHz" in c:
            freq, const = f[c.index("CarrierFrequencyHz")], f[c.index("ConstellationType")]
            if l1_only and (freq.startswith("117645") or const == "7"):
                continue
            tcol_i = c.index(tcol)
            if kind in ("Raw", "Status") and f[tcol_i] and t0 is not None and const == "1" and freq.startswith("157542"):
                rel = (int(f[tcol_i]) - SHIFT_MS - t0) / 1000
                sv = int(f[c.index("Svid")])
                for fsv, a, b, db in fades:
                    if sv == fsv and a <= rel < b:
                        k = c.index("Cn0DbHz")
                        if f[k] and float(f[k]) > 0:
                            f[k] = f"{max(float(f[k]) - db, 1.0):.2f}"
        if l1_only and kind == "Raw" and "IsFullTracking" in c:
            f[c.index("IsFullTracking")] = ""
        out.append(kind + "," + ",".join(f))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def pair(tmp_path_factory, cfg):
    tmp = tmp_path_factory.mktemp("pair")
    a = make_log(tmp / "gnss_log_iqoo.txt", [(8, 115, 170, 10)])            # minute 14:41: no AGC flag
    b = make_log(tmp / "gnss_log_oneplus.txt", [(8, 115, 170, 10), (3, 115, 170, 10)], l1_only=True, device="CPH2569")
    res = pipeline.run([str(a)], None, tmp / "out", cfg, plots=True, phone2=[str(b)])
    return tmp, res


def ev_of(res, rcv, svid, signal="GPS_L1"):
    c = res["tables"]["events_classified"]
    return c[(c["receiver"] == rcv) & (c["svid"] == svid) & (c["signal"] == signal) & (c["constellation"] == "GPS")
             & (c["class"] != "edge_of_gap")]


def test_fade_on_both_phones_counts_toward_iono(pair):
    _, res = pair
    for rcv in ("phone", "phone2"):
        e = ev_of(res, rcv, 8)
        assert len(e) and e["both_phones"].astype(bool).any(), rcv
        row = e[e["both_phones"].astype(bool)].iloc[0]
        assert row["class"] in ("iono_possible", "iono_likely") and "both phones" in row["reason"]


def test_fade_on_one_phone_only_is_local(pair):
    _, res = pair
    e = ev_of(res, "phone2", 3)
    assert len(e) and (e["both_phones"] == False).all()                     # noqa: E712
    assert (e["class"] == "local").all() and e["reason"].str.contains("one phone only").all()
    p1 = ev_of(res, "phone", 3)
    assert p1.empty or (p1["end_utc"] < e["start_utc"].min() - pd.Timedelta(seconds=30)).all()   # no overlapping fade on phone 1


def test_l1_only_second_phone_does_not_break_anything(pair):
    tmp, res = pair
    out = tmp / "out"
    assert res["receivers"] == ["phone", "phone2"]
    q2 = json.loads((out / "phone2_quality.json").read_text(encoding="utf-8"))
    assert q2["agc"] is None and q2["navic"]["svids_in_raw"] == [] and not q2["navic"]["tracked"]
    assert set(q2["raw"]["rows_per_signal"]) <= {"GPS_L1", "GAL_E1", "GLO_G1", "BDS_B1I", "BDS_B1C", "QZS_L1"}
    m = res["tables"]["phones_1min"]
    assert len(m) and m["both_faded"].any() and set(m["signal"]) <= {"GPS_L1", "GAL_E1", "GLO_G1", "BDS_B1I", "BDS_B1C"}
    recs = schema.read_jsonl(out / "minutes.jsonl")
    assert {r["receiver"] for r in recs} == {"phone", "phone2"}
    for r in recs:
        schema.validate(r)
    g8 = [s for r in recs if r["receiver"] == "phone2" for s in r["satellites"]
          if s["sys"] == "GPS" and s["svid"] == 8 and s["band"] == "L1" and s["class"] != "none"]
    assert g8 and g8[0]["both_phones"] is True
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Two phones" in md and "CPH2569" in md and "one phone only" in md
    assert (out / "plots" / "cn0_timeline_GPS.png").exists()


def test_dashboard_event_table_has_both_phones(pair):
    tmp, _ = pair
    ev = TestClient(create_app(tmp / "out")).get("/api/events", params={"hours": 3}).json()["events"]
    g8 = [e for e in ev if e["sys"] == "GPS" and e["svid"] == 8 and e["band"] == "L1"]
    assert g8 and all(e["both_phones"] is True for e in g8)
    g9 = [e for e in ev if e["sys"] == "GPS" and e["svid"] == 3 and e["receiver"] == "phone2" and e["class"] == "local"]
    assert g9 and g9[0]["both_phones"] is False


def test_cli_phone2_and_session_note(pair, tmp_path):
    tmp, _ = pair
    r = CliRunner().invoke(cli, ["session", "note", "--antenna", "external", "--phone", "iQOO vivo I2221",
                                 "--phone2", "OnePlus CPH2569", "--dir", str(tmp_path)])
    assert r.exit_code == 0
    assert json.loads((tmp_path / "session.json").read_text())["phone2"] == "OnePlus CPH2569"
    r = CliRunner().invoke(cli, ["run", "--phone", str(tmp / "gnss_log_iqoo.txt"), "--phone2",
                                 str(tmp / "gnss_log_oneplus.txt"), "--out", str(tmp_path / "o"), "--no-plots",
                                 "--session", str(tmp_path / "session.json")])
    assert r.exit_code == 0, r.output
    head = (tmp_path / "o" / "summary.md").read_text(encoding="utf-8").split("## Bottom line")[0]
    assert "**Phone 2:** OnePlus CPH2569" in head
