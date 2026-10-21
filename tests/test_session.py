"""D2: session notes (no timing), shown at the top of summary.md; --trim-start/--trim-end."""
import json
import sys
from pathlib import Path

from click.testing import CliRunner
from conftest import BAD, GOOD

from ionwatch import pipeline, schema
from ionwatch.cli import cli

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nmea_from_phone import phone_to_nmea  # noqa: E402


def test_note_has_only_the_note_fields(tmp_path):
    res = CliRunner().invoke(cli, ["session", "note", "--antenna", "external", "--phone", "iQOO vivo I2221",
                                   "--notes", "roof, open sky", "--dir", str(tmp_path)])
    assert res.exit_code == 0, res.output
    d = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert d == {"antenna": "external", "phone": "iQOO vivo I2221", "phone2": "", "notes": "roof, open sky"}


def test_note_rejects_unknown_antenna(tmp_path):
    res = CliRunner().invoke(cli, ["session", "note", "--antenna", "helical", "--dir", str(tmp_path)])
    assert res.exit_code != 0


def test_note_found_in_live_folder_and_printed(tmp_path, cfg):
    live = tmp_path / "live"
    (live / "raw").mkdir(parents=True)
    CliRunner().invoke(cli, ["session", "note", "--antenna", "external", "--phone", "iQOO vivo I2221",
                             "--notes", "trial", "--dir", str(live)])
    nmea = live / "raw" / "nmea_2026-10-18.nmea"
    phone_to_nmea(GOOD, nmea, cfg)
    pipeline.run([str(GOOD)], str(nmea), tmp_path / "out", cfg, plots=False)
    md = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    head = md.split("## Bottom line")[0]
    assert "**Antenna:** external" in head and "iQOO vivo I2221" in head and "trial" in head
    assert "2026-10-18 UTC (DOY 291)" in head                     # date from the data
    assert (tmp_path / "out" / "session.json").exists()


def test_missing_note_shows_unknown(tmp_path, cfg):
    pipeline.run([str(BAD)], None, tmp_path, cfg, plots=False)
    head = (tmp_path / "summary.md").read_text(encoding="utf-8").split("## Bottom line")[0]
    assert "**Antenna:** unknown" in head and "2026-10-20 UTC (DOY 293)" in head


def test_onboard_antenna_warning(tmp_path, cfg):
    note = tmp_path / "s.json"
    note.write_text(json.dumps({"antenna": "onboard", "phone": "", "notes": ""}))
    pipeline.run([str(BAD)], None, tmp_path / "o", cfg, plots=False, session=str(note))
    md = (tmp_path / "o" / "summary.md").read_text(encoding="utf-8")
    assert "on-board patch antenna: C/N0 lower, fades less reliable" in md


def test_trim_drops_setup_minutes(tmp_path, cfg):
    pipeline.run([str(BAD)], None, tmp_path / "full", cfg, plots=False)
    pipeline.run([str(BAD)], None, tmp_path / "trim", cfg, plots=False, trim_start_min=1, trim_end_min=0.5)
    full = schema.read_jsonl(tmp_path / "full" / "minutes.jsonl")
    trim = schema.read_jsonl(tmp_path / "trim" / "minutes.jsonl")
    assert trim[0]["minute_utc"] > full[0]["minute_utc"] and len(trim) < len(full)
    md = (tmp_path / "trim" / "summary.md").read_text(encoding="utf-8")
    assert "**Trimmed:** first 1 min, last 0.5 min of data" in md
    res = CliRunner().invoke(cli, ["run", "--phone", str(BAD), "--out", str(tmp_path / "cli"), "--no-plots",
                                   "--trim-start", "1"])
    assert res.exit_code == 0, res.output
