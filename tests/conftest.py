from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ionwatch.proc.common import load_config

FIX = Path(__file__).parent / "fixtures"
GOOD = FIX / "gnss_log_2026_10_18_first3min.txt"            # iQOO daytime: NavIC + dual-frequency
BAD = FIX / "gnss_log_2026_10_20_first3min.txt"             # iQOO partial: NavIC not tracked, low C/N0
BAD_TAIL = FIX / "gnss_log_2026_10_20_raw_end_bursts.txt"   # end of Raw, 88 min silence, Status bursts


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def l1_only_log(tmp_path_factory):
    """Small synthetic L1-only GnssLogger file: the good fixture's first 2 minutes with every
    L5-band and NavIC row (Raw, Status, Agc) removed."""
    cols, out, t0 = {}, [], None
    for line in GOOD.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("#"):
            head, _, rest = line[1:].strip().partition(",")
            if rest:
                cols[head] = rest.strip().split(",")
            out.append(line)
            continue
        kind, _, rest = line.partition(",")
        f = rest.rstrip("\n").split(",")
        c = cols.get(kind)
        if kind in ("Raw", "Agc"):
            t = int(f[0])
            t0 = t if t0 is None else t0
            if t - t0 > 120_000:
                break
        if c and "CarrierFrequencyHz" in c:
            freq, const = f[c.index("CarrierFrequencyHz")], f[c.index("ConstellationType")]
            if freq.startswith("117645") or const == "7":
                continue
        out.append(line)
    path = tmp_path_factory.mktemp("synthetic") / "gnss_log_l1_only.txt"
    path.write_text("".join(out), encoding="utf-8")
    return path


def synth_obs(n_sats=5, minutes=30, cn0=40.0, elev=60.0, signal="GPS_L1", t0="2026-10-20T14:00:00Z",
              constellation="GPS", source="raw"):
    """Synthetic 1 Hz tidy C/N0 table (common.OBS_COLS) for n_sats satellites."""
    t = pd.date_range(t0, periods=minutes * 60, freq="1s", tz="UTC") if "Z" not in t0 else \
        pd.date_range(pd.Timestamp(t0), periods=minutes * 60, freq="1s")
    rows = []
    for sv in range(1, n_sats + 1):
        rows.append(pd.DataFrame({
            "time_utc": t, "receiver": "phone", "source": source, "constellation": constellation, "svid": sv,
            "signal": signal, "band": "L1", "elev_deg": elev, "az_deg": 60.0 * sv, "cn0_dbhz": cn0,
            "time_quality": "logged"}))
    return pd.concat(rows, ignore_index=True)


def active_of(obs):
    return np.unique(obs["time_utc"].astype("int64").to_numpy() // 10**9)
