"""Download the benchmark data for one UTC day, validate every file, write manifest.json.

Sources, tried in order per item (the first that downloads AND validates wins):
  * NASA CDDIS (https://cddis.nasa.gov/archive), Earthdata login from ~/.netrc
    (machine urs.earthdata.nasa.gov login <user> password <pass>); never hard-coded;
  * anonymous mirrors checked from the development machine on 2026-10-21:
    BKG IGS data centre (obs, broadcast nav), JPL sideshow (GIM rapid/final), UPC (GIM rapid/final);
  * NOAA SWPC (Kp, F10.7; recent days only) and GFZ Potsdam (Kp, F10.7; any date), both kept so
    they can be cross-checked.
DCB/OSB bias files are on CDDIS only; without a login the satellite DCBs in the JPL IONEX aux
block are recorded instead (ROTI does not need biases: it is carrier-phase only).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger("ionwatch.bench")

CDDIS = "https://cddis.nasa.gov/archive/gnss"
BKG = "https://igs.bkg.bund.de/root_ftp/IGS"
JPL = "https://sideshow.jpl.nasa.gov/pub/iono_daily"
UPC = "https://chapman.upc.es/tomion/rapid"
SWPC_KP = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SWPC_F107 = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
GFZ = "https://kp.gfz.de/app/json/"
STATIONS = {"LCK3": "LCK300IND", "LCK4": "LCK400IND", "IISC": "IISC00IND", "HYDE": "HYDE00IND"}
PRIMARY = ("LCK3", "LCK4")          # Lucknow, ~295 km from Haldwani
BACKUP = ("IISC", "HYDE")


@dataclass
class Item:
    name: str
    kind: str                         # obs | nav | ionex | bias | kp | f107
    urls: list[str]
    filename: str
    required: bool = False
    result: dict = field(default_factory=dict)


def day_parts(date: pd.Timestamp) -> dict:
    doy = date.dayofyear
    week = int((date.tz_localize(None) - pd.Timestamp("1980-01-06")).days // 7)
    return {"yyyy": f"{date:%Y}", "yy": f"{date:%y}", "doy": f"{doy:03d}", "week": f"{week}",
            "ymd": f"{date:%y%m%d}", "date": f"{date:%Y-%m-%d}"}


def plan(date: pd.Timestamp) -> list[Item]:
    p = day_parts(date)
    Y, D, yy = p["yyyy"], p["doy"], p["yy"]
    items = []
    for st, long in STATIONS.items():
        f = f"{long}_R_{Y}{D}0000_01D_30S_MO.crx.gz"
        items.append(Item(f"obs_{st}", "obs", [f"{CDDIS}/data/daily/{Y}/{D}/{yy}d/{f}", f"{BKG}/obs/{Y}/{D}/{f}"], f))
    brdm = f"BRDM00DLR_S_{Y}{D}0000_01D_MN.rnx.gz"
    items.append(Item("nav_BRDM", "nav", [f"{CDDIS}/data/daily/{Y}/brdc/{brdm}", f"{BKG}/BRDC/{Y}/{D}/{brdm}"],
                      brdm, required=True))
    gim = [f"{CDDIS}/products/ionex/{Y}/{D}/IGS0OPSFIN_{Y}{D}0000_01D_02H_GIM.INX.gz",
           f"{CDDIS}/products/ionex/{Y}/{D}/IGS0OPSRAP_{Y}{D}0000_01D_02H_GIM.INX.gz",
           f"{CDDIS}/products/ionex/{Y}/{D}/IGS0OPSRAP_{Y}{D}0000_01D_01H_GIM.INX.gz",
           f"{JPL}/IONEX_final/y{Y}/JPLH{D}0.{yy}I.gz",
           f"{JPL}/IONEX_final/y{Y}/JPLG{D}0.{yy}I.gz",
           f"{JPL}/IONEX_rapid/JPLR{D}0.{yy}I.gz",
           f"{UPC}/{Y}/{D}_{p['ymd']}.15min/UPC0OPSFIN_{Y}{D}0000_01D_02H_GIM.INX.Z",
           f"{UPC}/{Y}/{D}_{p['ymd']}.15min/UPC0OPSRAP_{Y}{D}0000_01D_02H_GIM.INX.Z"]
    items.append(Item("ionex", "ionex", gim, "GIM", required=True))
    items.append(Item("ionex_check", "ionex", gim[::-1][:2] + gim[3:6], "GIM_check"))   # a second producer
    items.append(Item("bias", "bias", [f"{CDDIS}/products/bias/{Y}/CAS0OPSRAP_{Y}{D}0000_01D_01D_DCB.BSX.gz",
                                       f"{CDDIS}/products/bias/{Y}/CAS0MGXRAP_{Y}{D}0000_01D_01D_DCB.BSX.gz",
                                       f"{CDDIS}/products/{p['week']}/COD0OPSFIN_{Y}{D}0000_01D_01D_OSB.BIA.gz"],
                      "BIAS"))
    t0, t1 = f"{p['date']}T00:00:00Z", f"{p['date']}T23:59:59Z"
    items.append(Item("kp_swpc", "kp", [SWPC_KP], "kp_swpc.json"))
    items.append(Item("kp_gfz", "kp", [f"{GFZ}?start={t0}&end={t1}&index=Kp"], "kp_gfz.json"))
    items.append(Item("f107_swpc", "f107", [SWPC_F107], "f107_swpc.json"))
    items.append(Item("f107_gfz", "f107", [f"{GFZ}?start={p['date']}T00:00:00Z&end={p['date']}T23:59:59Z&index=Fobs"],
                      "f107_gfz.json"))
    return items


def producer(url: str) -> str:
    u = url.split("/")[-1].upper()
    return "IGS" if u.startswith("IGS0OPS") else "JPL" if u.startswith("JPL") else "UPC" if "UPC" in u else u[:3]


def session():
    """requests session: ~/.netrc credentials are re-applied on the redirect to the Earthdata login."""
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "ionwatch-bench/1.0"
    return s


def _download(sess, url: str, timeout=(15, 120), tries: int = 2) -> tuple[bytes | None, str]:
    for attempt in range(tries):
        try:
            r = sess.get(url, timeout=timeout, allow_redirects=True)
            break
        except Exception as e:                               # network down, DNS, TLS, host unreachable
            err = f"{type(e).__name__}: {str(e).split('(Caused')[0]}"[:200]
            if attempt == tries - 1:
                return None, err
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    ctype = r.headers.get("content-type", "")
    if "urs.earthdata.nasa.gov" in r.url or ("text/html" in ctype and not url.endswith((".json", "Kp", "Fobs"))):
        return None, "Earthdata login required (add urs.earthdata.nasa.gov to ~/.netrc)"
    return r.content, "ok"


def fetch(date: pd.Timestamp, out: Path, sess=None, items: list[Item] | None = None) -> dict:
    """Download and validate everything for `date` into out/<YYYY-DDD>/; returns the manifest."""
    from ionwatch.bench import validate
    date = pd.Timestamp(date).tz_localize("UTC") if pd.Timestamp(date).tzinfo is None else pd.Timestamp(date)
    date = date.normalize()
    p = day_parts(date)
    folder = Path(out) / f"{p['yyyy']}-{p['doy']}"
    folder.mkdir(parents=True, exist_ok=True)
    sess = sess or session()
    items = items or plan(date)
    manifest = {"date_utc": p["date"], "doy": int(p["doy"]), "folder": str(folder), "items": {}}
    got_ionex_from = None
    for it in items:
        tries = []
        for url in it.urls:
            if it.name == "ionex_check" and got_ionex_from and producer(url) == producer(got_ionex_from):
                continue                                     # the check must come from another producer
            data, msg = _download(sess, url)
            entry = {"url": url, "status": msg}
            if data is not None:
                fname = url.split("/")[-1].split("?")[0] if it.kind in ("obs", "nav", "ionex", "bias") else it.filename
                path = folder / fname
                path.write_bytes(data)
                v = validate.check(it.kind, path, date, name=it.name)
                entry.update(file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), validation=v)
                if v["ok"]:
                    tries.append(entry)
                    it.result = {"ok": True, **entry}
                    if it.name == "ionex":
                        got_ionex_from = url
                    break
                entry["status"] = "downloaded but failed validation: " + "; ".join(v["problems"])
            tries.append(entry)
            log.info("%s: %s -> %s", it.name, url, entry["status"])
        if not it.result:
            it.result = {"ok": False, "status": tries[-1]["status"] if tries else "no source"}
        it.result["tried"] = tries
        manifest["items"][it.name] = it.result
    manifest["cross_checks"] = validate.cross_checks(folder, manifest, date)
    manifest["missing_required"] = [i.name for i in items if i.required and not i.result.get("ok")]
    stations_ok = [n[4:] for n, r in manifest["items"].items() if n.startswith("obs_") and r.get("ok")]
    manifest["stations_ok"] = stations_ok
    if not stations_ok:
        manifest["missing_required"].append("obs (no station)")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return manifest
