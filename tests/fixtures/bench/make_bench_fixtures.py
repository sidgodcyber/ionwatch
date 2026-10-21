"""Rebuild the small offline benchmark fixtures from real downloads in data/bench/.

    python tests/fixtures/bench/make_bench_fixtures.py      (after `ionwatch bench fetch --date 2026-10-20`
                                                             and `--date 2026-10-19`)

Sources (all public, fetched 2026-10-21): BKG IGS mirror (LCK4 / IISC obs, BRDM nav), JPL sideshow
(JPLR GIM), UPC (UPC0OPSRAP GIM), NOAA SWPC (Kp, F10.7); GFZ Kp / F10.7 were saved from the GFZ
JSON API the same day. The CRINEX reference output comes from RNXCMP crx2rnx (the `hatanaka` wheel).
"""
import gzip
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ionwatch.bench import rinex  # noqa: E402

HERE = Path(__file__).parent
B293, B292 = ROOT / "data" / "bench" / "2026-293", ROOT / "data" / "bench" / "2026-292"
NIGHT = ("2026 10 20 14 30", "2026 10 20 16 30")          # UTC = 20:00-22:00 IST


def crx_prefix(n_epochs=20):
    lines = gzip.decompress((B293 / "LCK400IND_R_20262930000_01D_30S_MO.crx.gz").read_bytes()).decode().splitlines()
    end = next(i for i, l in enumerate(lines) if "END OF HEADER" in l) + 1
    ep = [i for i in range(end, len(lines)) if lines[i].startswith(">")]
    # epoch lines after the first are differenced (no '>'), so count epochs with the decoder instead
    out = lines[:end]
    k, seen = end, 0
    rnx = rinex.crx_to_rnx(lines)
    n_rnx = sum(1 for l in rnx if l.startswith(">"))
    assert n_rnx > n_epochs and ep
    # cut the CRINEX after n_epochs: walk the compressed stream with the decoder's own structure
    hdr_end = end
    ntypes = {}
    cur = None
    for l in lines[:hdr_end]:
        if l[60:73] == "SYS / # / OBS":
            if l[0] != " ":
                cur = l[0]
                ntypes[cur] = int(l[3:6])
    k = hdr_end
    ref = ""
    while seen < n_epochs:
        l = lines[k]
        epoch = l if l.startswith(">") else rinex._repair(ref, l)
        ref = epoch
        ns = int(epoch[32:35])
        k += 2 + ns                                             # epoch line, clock line, one line per satellite
        seen += 1
    out = lines[:k]
    (HERE / "lck4_prefix.crx").write_text("\n".join(out) + "\n", encoding="ascii")
    import hatanaka
    exe = Path(hatanaka.__file__).parent / "bin" / ("crx2rnx.exe" if sys.platform == "win32" else "crx2rnx")
    ref_rnx = subprocess.run([str(exe), "-"], input="\n".join(out) + "\n", capture_output=True, text=True).stdout
    (HERE / "lck4_prefix_ref.rnx").write_text(ref_rnx, encoding="ascii")


def night_obs(station):
    rnx = rinex.crx_to_rnx(rinex.open_text(B293 / f"{station}_R_20262930000_01D_30S_MO.crx.gz"))
    h, i = rinex._header(rnx)
    gps = h["obs_types"]["G"]
    keep = [t for t in ("C1C", "L1C", "L2W") if t in gps]
    idx = [gps.index(t) for t in keep]
    out = []
    for l in rnx[:i]:
        if l[60:73] == "SYS / # / OBS":
            if l[0] == "G":
                out.append(f"G  {len(keep):3d} " + " ".join(keep).ljust(53) + "SYS / # / OBS TYPES")
            continue
        out.append(l)
    j = i
    while j < len(rnx):
        l = rnx[j]
        if l.startswith(">"):
            ns = int(l[32:35])
            sats = rnx[j + 1:j + 1 + ns]
            j += 1 + ns
            if not (NIGHT[0] <= l[2:18] < NIGHT[1]):
                continue
            g = [s for s in sats if s.startswith("G")]
            out.append(l[:32] + f"{len(g):3d}")
            for s in g:
                out.append((s[:3] + "".join(s[3 + 16 * k:3 + 16 * k + 16].ljust(16) for k in idx)).rstrip())
        else:
            j += 1
    (HERE / f"{station[:4]}_night.rnx").write_text("\n".join(out) + "\n", encoding="ascii")


def nav_subset():
    lines = rinex.open_text(B293 / "BRDM00DLR_S_20262930000_01D_MN.rnx.gz")
    end = next(i for i, l in enumerate(lines) if "END OF HEADER" in l) + 1
    out, k = lines[:end], end
    while k < len(lines):
        l = lines[k]
        if l.startswith("G") and "2026 10 20 1" in l[3:17]:    # GPS ephemerides with toc 10:00-19:59 UTC
            out += lines[k:k + 8]
            k += 8
        else:
            k += 1
    (HERE / "brdm_gps_afternoon.rnx").write_text("\n".join(out) + "\n", encoding="ascii")


if __name__ == "__main__":
    crx_prefix()
    night_obs("LCK400IND")
    night_obs("IISC00IND")
    nav_subset()
    import shutil
    shutil.copy(B292 / "JPLR2920.26I.gz", HERE / "JPLR2920.26I.gz")
    shutil.copy(B292 / "UPC0OPSRAP_20262920000_01D_02H_GIM.INX.Z", HERE / "UPC0OPSRAP_20262920000_01D_02H_GIM.INX.Z")
    for f in sorted(HERE.iterdir()):
        print(f"{f.name:48s} {f.stat().st_size:>9,d}")
