"""IonWatch dashboard: FastAPI over a live, replay or batch output folder.

    python -m ionwatch.cli dashboard --data data/live/ --port 8080

Serves latest.json, the recent one-minute records from minutes.jsonl (highest rev per
receiver and minute), a derived event table and warning summary, and a single static page
(Chart.js, vendored so it works on a field Wi-Fi without internet). No database: the
records file is read incrementally from where the last request stopped.
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).with_name("static")
STALE_S = 180
TZ = "Asia/Kolkata"


class RecordStore:
    """Incremental reader of minutes.jsonl keeping the highest rev per (receiver, minute)."""

    def __init__(self, path: Path):
        self.path = path
        self._offset = 0
        self._partial = b""
        self.records: dict[tuple[str, str], dict] = {}
        self.n_lines = 0
        self.n_final = 0
        self._lock = threading.Lock()

    def refresh(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._reset()
                return
            size = self.path.stat().st_size
            if size < self._offset:          # file replaced (a new replay): start over
                self._reset()
            if size == self._offset:
                return
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                data = self._partial + f.read()
                self._offset = f.tell()
            *lines, self._partial = data.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.n_lines += 1
                self.n_final += r.get("status") == "final"
                k = (r["receiver"], r["minute_utc"])
                if k not in self.records or r.get("rev", 1) >= self.records[k].get("rev", 1):
                    self.records[k] = r

    def _reset(self):
        self._offset, self._partial, self.records, self.n_lines, self.n_final = 0, b"", {}, 0, 0

    def recent(self, hours: float, since: str | None = None) -> list[dict]:
        self.refresh()
        if not self.records:
            return []
        newest = max(m for _, m in self.records)
        lo = (pd.Timestamp(newest) - pd.Timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:00Z")
        if since and since > lo:
            lo = since
        return sorted((r for (_, m), r in self.records.items() if m >= lo),
                      key=lambda r: (r["minute_utc"], r["receiver"]))


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mode(folder: Path, info: dict | None) -> str:
    if info and info.get("mode"):
        return info["mode"]
    if (folder / "replay.log").exists():
        return "replay"
    if (folder / "live.log").exists():
        return "live"
    return "batch"


def _ist(minute_utc: str) -> str:
    return pd.Timestamp(minute_utc).tz_convert(TZ).strftime("%Y-%m-%d %H:%M IST")


def receiver_status(latest: dict | None, age_s: float | None, mode: str) -> str:
    if latest is None:
        return "waiting for data"
    if age_s is not None and age_s > STALE_S and mode != "batch":
        return "replay finished" if mode == "replay" else "stale"
    w = latest.get("warnings", [])
    if any(x in w for x in ("serial_disconnected", "no_serial_data")):
        return "disconnected"
    if "no_data" in w:
        return "no data"
    if any(x.startswith("receiver_partial") for x in w):
        return "partial minute"
    if latest.get("sats_tracked", 0) == 0:
        return "no satellites"
    return "ok"


def events_from_records(records: list[dict]) -> list[dict]:
    """Merge consecutive minutes with the same satellite, band and class into one row."""
    rows = []
    open_rows: dict[tuple, dict] = {}
    for r in records:
        m = pd.Timestamp(r["minute_utc"])
        seen = set()
        for s in r["satellites"]:
            if s["class"] == "none":
                continue
            key = (r["receiver"], s["sys"], s["svid"], s["band"], s["class"])
            seen.add(key)
            row = open_rows.get(key)
            if row and pd.Timestamp(row["end_utc"]) == m:
                row["end_utc"] = (m + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:00Z")
                row["minutes"] += 1
                row["max_fade_db"] = max(filter(lambda x: x is not None, [row["max_fade_db"], s["max_fade_db"]]),
                                         default=None)
                if r["status"] == "provisional":
                    row["status"] = "provisional"
                if s.get("both_phones") is not None:
                    row["both_phones"] = bool(row.get("both_phones")) or bool(s["both_phones"])
                row["lock_losses"] += s["lock_loss"]
            else:
                row = {"receiver": r["receiver"], "satellite": f"{s['sys']} {s['svid']}", "sys": s["sys"],
                       "svid": s["svid"], "band": s["band"], "class": s["class"], "reason": s.get("reason", ""),
                       "start_utc": r["minute_utc"],
                       "end_utc": (m + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:00Z"),
                       "start_ist": _ist(r["minute_utc"]), "minutes": 1, "max_fade_db": s["max_fade_db"],
                       "lock_losses": s["lock_loss"], "both_phones": s.get("both_phones"),
                       "status": r["status"]}
                rows.append(row)
                open_rows[key] = row
        for key in [k for k in open_rows if k not in seen]:
            del open_rows[key]
    return sorted(rows, key=lambda x: x["start_utc"], reverse=True)


def warnings_from_records(records: list[dict]) -> list[dict]:
    counts, last = Counter(), {}
    for r in records:
        for w in r["warnings"]:
            code = w.split(":")[0]
            counts[code] += 1
            last[code] = (r["minute_utc"], w)
    return [{"code": c, "minutes": n, "last_utc": last[c][0], "last_ist": _ist(last[c][0]), "example": last[c][1]}
            for c, n in counts.most_common()]


def create_app(data_dir: str | Path) -> FastAPI:
    folder = Path(data_dir)
    store = RecordStore(folder / "minutes.jsonl")
    app = FastAPI(title="IonWatch dashboard")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def latest_record():
        rec = _read_json(folder / "latest.json")
        if rec is None:
            store.refresh()
            if store.records:
                rec = max(store.records.values(), key=lambda r: (r["minute_utc"], r.get("rev", 1)))
        return rec

    def age_s() -> float | None:
        for name in ("latest.json", "minutes.jsonl"):
            p = folder / name
            if p.exists():
                return round(time.time() - p.stat().st_mtime, 1)
        return None

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        info = _read_json(folder / "run_info.json") or {}
        mode = _mode(folder, info)
        store.refresh()
        latest = latest_record()
        age = age_s()
        receivers = sorted({r for r, _ in store.records})
        return {
            "node_id": (latest or {}).get("node_id") or info.get("node_id") or "unknown",
            "mode": mode, "source": info.get("source") or info.get("port"),
            "speed": info.get("speed"), "receivers": receivers,
            "latest": None if latest is None else {
                "minute_utc": latest["minute_utc"], "minute_ist": _ist(latest["minute_utc"]),
                "status": latest.get("status"), "rev": latest.get("rev"), "receiver": latest["receiver"],
                "sats_tracked": latest["sats_tracked"], "l86_rate_hz": latest.get("l86_rate_hz"),
                "navic_status": latest["navic_status"], "warnings": latest["warnings"]},
            "updated_age_s": age, "stale": age is not None and age > STALE_S,
            "receiver_status": receiver_status(latest, age, mode),
            "records": store.n_lines, "final_records": store.n_final, "minutes": len(store.records),
            "server_time_utc": pd.Timestamp.now(tz="UTC").isoformat()}

    @app.get("/api/latest")
    def latest():
        return JSONResponse(latest_record())

    @app.get("/api/minutes")
    def minutes(hours: float = Query(3.0, gt=0, le=48), since: str | None = None):
        return {"records": store.recent(hours, since)}

    @app.get("/api/events")
    def events(hours: float = Query(3.0, gt=0, le=48), limit: int = Query(50, ge=1, le=500)):
        return {"events": events_from_records(store.recent(hours))[:limit]}

    @app.get("/api/warnings")
    def warnings(hours: float = Query(1.0, gt=0, le=48)):
        return {"warnings": warnings_from_records(store.recent(hours))}

    return app


def serve(data_dir: str | Path, host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn
    uvicorn.run(create_app(data_dir), host=host, port=port, log_level="warning")
