"""Regenerate the trimmed GnssLogger fixtures from the full logs in data/raw/.

    python tests/fixtures/make_phone_fixture.py

Each fixture keeps every header line and a slice of data lines, cut on timestamped
Raw/Agc/Fix rows (Status rows are kept in file order so their timing logic still works).
"""
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
HERE = Path(__file__).parent


def _time(line: str) -> int | None:
    kind, _, rest = line.partition(",")
    f = rest.rstrip("\n").split(",")
    if kind in ("Raw", "Agc") and f[0]:
        return int(f[0])
    if kind == "Fix" and len(f) > 7 and f[7] and f[0] in ("GPS", "FLP"):
        return int(f[7])
    return None


def trim(src: str, out: str, start_min: float | None = None, end_min: float = 3.0,
         from_end_of_raw: bool = False) -> None:
    """Keep data lines between start_min and end_min (minutes after the first timed row, or
    after the last Raw row when from_end_of_raw)."""
    lines = (RAW / src).read_text(encoding="utf-8").splitlines(keepends=True)
    times = [_time(l) if not l.startswith("#") else None for l in lines]
    ref = (max(t for l, t in zip(lines, times) if t and l.startswith("Raw")) if from_end_of_raw
           else next(t for t in times if t))
    keep, inside = [], start_min is None
    for line, t in zip(lines, times):
        if line.startswith("#"):
            keep.append(line)
            continue
        if t is not None:
            rel = (t - ref) / 60_000
            if not inside and rel >= start_min:
                inside = True
            if rel > end_min:
                break
        if inside:
            keep.append(line)
    (HERE / out).write_text("".join(keep), encoding="utf-8")
    print(f"wrote {out}: {len(keep)} lines")


if __name__ == "__main__":
    # Good: daytime iQOO, NavIC tracked, dual-frequency GPS/Galileo/BeiDou.
    trim("gnss_log_2026_10_18_14_09_05.txt", "gnss_log_2026_10_18_first3min.txt", end_min=3)
    # Bad/partial: first 3 min of Raw (NavIC not tracked, low C/N0) ...
    trim("gnss_log_2026_10_20_18_58_37.txt", "gnss_log_2026_10_20_first3min.txt", end_min=3)
    # ... and its tail: last minute of Raw, the 88 min silence, and the first Status bursts.
    trim("gnss_log_2026_10_20_18_58_37.txt", "gnss_log_2026_10_20_raw_end_bursts.txt",
         start_min=-1, end_min=97, from_end_of_raw=True)
