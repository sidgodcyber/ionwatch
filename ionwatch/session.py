"""Session notes: what was set up, never when.

    python -m ionwatch.cli session note --antenna external|onboard --phone "iQOO vivo I2221" --notes "..."

writes session.json with exactly {antenna, phone, phone2, notes} (add --phone2 "OnePlus CPH2569"). A session's date, start and end
always come from the data timestamps (never the host clock or user input). The batch run
finds session.json next to its inputs and prints it at the top of summary.md.
"""
from __future__ import annotations

import json
from pathlib import Path

FIELDS = ("antenna", "phone", "phone2", "notes")
ANTENNAS = ("external", "onboard")
ONBOARD_WARNING = "on-board patch antenna: C/N0 lower, fades less reliable"


def write_note(folder: str | Path, antenna: str, phone: str = "", notes: str = "", phone2: str = "") -> Path:
    if antenna not in ANTENNAS:
        raise ValueError(f"antenna must be one of {ANTENNAS}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "session.json"
    path.write_text(json.dumps({"antenna": antenna, "phone": phone, "phone2": phone2, "notes": notes}, indent=1,
                               ensure_ascii=False),
                    encoding="utf-8")
    return path


def read_note(path: Path | None) -> dict:
    """The note's fields, or antenna "unknown" when there is no note."""
    if path is None or not Path(path).exists():
        return {"antenna": "unknown", "phone": "", "phone2": "", "notes": "", "source": None}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {k: d.get(k, "") for k in FIELDS}
    out["antenna"] = out["antenna"] or "unknown"
    out["source"] = str(path)
    return out


def find_note(explicit: str | Path | None, out_dir: Path, l86: str | None, phone: list[str] | None) -> Path | None:
    """--session if given; else session.json in the output folder, the live folder of the L86
    file (data/live/raw/nmea_*.nmea -> data/live/session.json), or next to a phone log."""
    if explicit:
        return Path(explicit)
    cands = [Path(out_dir) / "session.json"]
    if l86:
        p = Path(l86).resolve()
        cands += [p.parent / "session.json", p.parent.parent / "session.json"]
    for f in phone or []:
        cands.append(Path(f).resolve().parent / "session.json")
    return next((c for c in cands if c.exists()), None)
