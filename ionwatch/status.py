"""One-line health check of a live (or replay) folder, the terminal backup for the dashboard.

    python -m ionwatch.cli status --data data/live/          # a line every 60 s, Ctrl-C to stop

    17:42:05 IST | minute 17:39 IST provisional | sats 9 | C/N0 38.4 dB-Hz | L86 1.0 Hz | age 12 s | warnings -
    17:43:05 IST | STALE: last record 4.1 min old (live mode stopped or receiver silent?) ...
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

STALE_S = 180
TZ = "Asia/Kolkata"


def _latest(folder: Path) -> tuple[dict | None, float | None]:
    """Newest record and the age (s) of the file it came from."""
    for name in ("latest.json", "minutes.jsonl"):
        p = folder / name
        if not p.exists() or p.stat().st_size == 0:
            continue
        age = time.time() - p.stat().st_mtime
        try:
            if name == "latest.json":
                return json.loads(p.read_text(encoding="utf-8")), age
            lines = [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            return json.loads(lines[-1]), age
        except (ValueError, IndexError):
            continue
    return None, None


def status_line(folder: str | Path, now: pd.Timestamp | None = None) -> tuple[str, bool]:
    """(line, stale). stale is True when the newest record is older than 3 min or missing."""
    folder = Path(folder)
    now = now or pd.Timestamp.now(tz="UTC")
    clock = f"{now.tz_convert(TZ):%H:%M:%S} IST"
    rec, age = _latest(folder)
    if rec is None:
        return f"{clock} | STALE: no records yet in {folder} (is live mode running?)", True
    cn0 = [s["cn0_mean"] for s in rec.get("satellites", []) if s.get("cn0_mean") is not None]
    mean = f"{sum(cn0) / len(cn0):.1f} dB-Hz" if cn0 else "-"
    rate = f"{rec['l86_rate_hz']} Hz" if rec.get("l86_rate_hz") is not None else "-"
    minute = pd.Timestamp(rec["minute_utc"]).tz_convert(TZ)
    age_txt = f"{age:.0f} s" if age < 120 else f"{age / 60:.1f} min"
    body = (f"minute {minute:%H:%M} IST {rec.get('status', '')} | sats {rec.get('sats_tracked', 0)} | "
            f"C/N0 {mean} | L86 {rate} | age {age_txt} | warnings {', '.join(rec.get('warnings') or []) or '-'}")
    if age > STALE_S:
        return f"{clock} | STALE: last record {age / 60:.1f} min old (live mode stopped or receiver silent?) | {body}", True
    return f"{clock} | {body}", False


def watch(folder: str | Path, every_s: float = 60, count: int | None = None, echo=print) -> None:
    n = 0
    while count is None or n < count:
        line, _ = status_line(folder)
        echo(line)
        n += 1
        if count is not None and n >= count:
            break
        time.sleep(every_s)
