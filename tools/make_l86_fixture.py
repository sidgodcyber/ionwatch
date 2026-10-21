"""Write a deterministic synthetic L86 NMEA fixture (1 Hz, MT3333 sentence order).

    python tools/make_l86_fixture.py [--minutes 10] [--out tests/fixtures/l86_sim_10min.nmea]

8 GPS, 4 GLONASS and 3 Galileo satellites on slow elevation arcs; C/N0 = 18 + 26 sin(elev)
plus 0.6 dB noise, rounded to integer dB-Hz like the L86 reports it. Each 1 s cycle is
RMC, GGA, GPGSA, GLGSA, GPGSV..., GLGSV, GAGSV. Times start at 2026-10-22T14:00:00Z
(19:30 IST); fake_l86.py re-stamps them anyway.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

T0 = datetime(2026, 10, 22, 14, 0, 0, tzinfo=timezone.utc)
# (talker, prn, elev0, elev rate deg/min, az0, az rate deg/min)
SKY = [("GP", 2, 62, 0.10, 40, 0.3), ("GP", 5, 45, -0.08, 120, 0.2), ("GP", 12, 70, 0.05, 210, -0.4),
       ("GP", 15, 33, 0.12, 300, 0.1), ("GP", 19, 25, -0.10, 80, 0.2), ("GP", 24, 55, 0.07, 160, -0.2),
       ("GP", 25, 38, 0.09, 260, 0.3), ("GP", 29, 15, 0.15, 20, 0.1),
       ("GL", 65, 50, 0.08, 100, 0.2), ("GL", 72, 35, -0.06, 230, 0.2), ("GL", 80, 60, 0.04, 330, -0.1),
       ("GL", 81, 28, 0.11, 190, 0.2),
       ("GA", 3, 48, 0.06, 140, 0.1), ("GA", 8, 31, 0.10, 280, 0.2), ("GA", 13, 66, -0.03, 10, 0.3)]


def nmea(body: str) -> str:
    x = 0
    for ch in body:
        x ^= ord(ch)
    return f"${body}*{x:02X}"


def cycle(t: datetime, k: int, rng) -> list[str]:
    hms = f"{t:%H%M%S}.000"
    out = [nmea(f"GPRMC,{hms},A,2915.336,N,07932.220,E,0.02,0.00,{t:%d%m%y},,,A"),
           nmea(f"GPGGA,{hms},2915.336,N,07932.220,E,1,11,0.9,500.0,M,-40.0,M,,")]
    sats = {}
    for talker, prn, e0, er, a0, ar in SKY:
        el = e0 + er * k / 60
        az = (a0 + ar * k / 60) % 360
        cn0 = int(round(18 + 26 * np.sin(np.radians(el)) + rng.normal(0, 0.6)))
        sats.setdefault(talker, []).append((prn, int(round(el)), int(round(az)), cn0))
    for talker in ("GP", "GL"):
        used = [f"{p:02d}" for p, el, _, _ in sats[talker] if el >= 20][:12]
        out.append(nmea(f"{talker}GSA,A,3,{','.join(used + [''] * (12 - len(used)))},1.6,0.9,1.3"))
    for talker in ("GP", "GL", "GA"):
        s = sats[talker]
        chunks = [s[i:i + 4] for i in range(0, len(s), 4)]
        for n, ch in enumerate(chunks, 1):
            out.append(nmea(f"{talker}GSV,{len(chunks)},{n},{len(s):02d}," +
                            ",".join(f"{p:02d},{el:02d},{az:03d},{c:02d}" for p, el, az, c in ch)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" /
                                         "l86_sim_10min.nmea"))
    a = ap.parse_args()
    rng = np.random.default_rng(2026)
    lines = []
    for k in range(a.minutes * 60):
        lines += cycle(T0 + timedelta(seconds=k), k, rng)
    Path(a.out).write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))   # bytes: no newline translation
    print(f"wrote {a.out}: {len(lines)} sentences, {a.minutes * 60} cycles")


if __name__ == "__main__":
    main()
