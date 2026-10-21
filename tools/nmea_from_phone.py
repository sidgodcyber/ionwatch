"""Make an L86-style NMEA stream from a GnssLogger log: the same satellites, elevations and
C/N0 as the phone's L1 signals (GPS L1, GLONASS G1, Galileo E1), one 1 Hz cycle per second.

    python tools/nmea_from_phone.py data/raw/gnss_log.txt out.nmea [--cn0-offset-db 8] [--shift-s 0]

Used to test the L86 + phone batch join without hardware: feed the file to tools/fake_l86.py
(which re-stamps the time) or give it straight to `ionwatch run --l86`. --shift-s moves the
NMEA clock against the phone (negative test for the cross-check time-offset estimate).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ionwatch.io import gnsslogger  # noqa: E402
from ionwatch.proc.common import load_config  # noqa: E402

L1 = {"GPS_L1": ("GP", 0), "GLO_G1": ("GL", 64), "GAL_E1": ("GA", 0)}


def nmea(body: str) -> str:
    x = 0
    for ch in body:
        x ^= ord(ch)
    return f"${body}*{x:02X}"


def phone_to_nmea(log_path, out_path, cfg=None, cn0_offset_db: float = 8.0, shift_s: int = 0):
    """Write the NMEA file; returns (number of 1 Hz cycles, UTC time of the first cycle)."""
    cfg = cfg or load_config()
    log = gnsslogger.read_phone([log_path], cfg)
    obs, _ = gnsslogger.cn0_obs(log, cfg)
    o = obs[obs["signal"].isin(L1) & obs["elev_deg"].notna() & (obs["elev_deg"] >= 0)].copy()
    o["sec"] = o["time_utc"].dt.floor("s")
    t0, t1 = o["sec"].min(), o["sec"].max()
    lines = []
    for k, t in enumerate(pd.date_range(t0, t1, freq="1s")):
        ts = t + pd.Timedelta(seconds=shift_s)
        hms = f"{ts:%H%M%S}.000"
        lines += [nmea(f"GPRMC,{hms},A,2915.336,N,07932.220,E,0.02,0.00,{ts:%d%m%y},,,A"),
                  nmea(f"GPGGA,{hms},2915.336,N,07932.220,E,1,10,0.9,500.0,M,-40.0,M,,")]
        sec = o[o["sec"] == t]
        by_talker: dict[str, list] = {"GP": [], "GL": [], "GA": []}
        for r in sec.drop_duplicates(["signal", "svid"]).itertuples():
            talker, base = L1[r.signal]
            cn0 = "" if pd.isna(r.cn0_dbhz) else f"{int(round(min(r.cn0_dbhz + cn0_offset_db, 99))):02d}"
            by_talker[talker].append((base + int(r.svid), int(round(r.elev_deg)), int(round(r.az_deg)) % 360, cn0))
        used = [f"{p:02d}" for p, el, _, c in by_talker["GP"] if c and el >= 20][:12]
        lines.append(nmea(f"GPGSA,A,3,{','.join(used + [''] * (12 - len(used)))},1.6,0.9,1.3"))
        for talker, sats in by_talker.items():
            if not sats:
                continue
            chunks = [sats[i:i + 4] for i in range(0, len(sats), 4)]
            for n, ch in enumerate(chunks, 1):
                lines.append(nmea(f"{talker}GSV,{len(chunks)},{n},{len(sats):02d}," +
                                  ",".join(f"{p:02d},{el:02d},{az:03d},{c}" for p, el, az, c in ch)))
    Path(out_path).write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))
    return int((t1 - t0).total_seconds()) + 1, t0 + pd.Timedelta(seconds=shift_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--cn0-offset-db", type=float, default=8.0, help="L86 with an outdoor antenna sees more")
    ap.add_argument("--shift-s", type=int, default=0)
    a = ap.parse_args()
    n, t0 = phone_to_nmea(a.log, a.out, cn0_offset_db=a.cn0_offset_db, shift_s=a.shift_s)
    print(f"wrote {a.out}: {n} cycles from {t0:%Y-%m-%dT%H:%M:%SZ}")


if __name__ == "__main__":
    main()
