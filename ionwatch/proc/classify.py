"""Heuristic labelling of fade events. Rules are applied in this order:

  edge_of_gap      - the event lies within edge_guard_s of a receiver silence (fades.py); it is
                     reported but not treated as a real event;
  local            - the event overlaps a local_cause_flag minute (interference.py), OR (two
                     phones) the other phone tracked the same satellite and band during the event
                     without a fade: a fade on one phone only;
  local_or_unknown - the event is outside the night window (config night_window_ist, IST):
                     never iono_* outside it; reason "outside_night_window";
  iono_likely      - inside the night window AND (both bands of the satellite faded, OR a valid
                     indicative S4 >= iono_s4, OR (two phones) the same satellite and band faded on
                     both phones within 30 s) AND the event's mean C/N0 >= iono_likely_min_cn0;
                     below that C/N0 the class is capped at iono_possible with reason "low_cn0";
  low_elev         - elevation at peak < low_elev_deg;
  iono_possible    - everything else inside the night window (single band only).

low_elev is checked before iono_possible; otherwise iono_possible would absorb every
low-elevation event. S4 counts only when s4_valid (S4 > noise_factor x s4_noise).
This is a heuristic triage, not a scintillation detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ionwatch.proc import common, fades

HEURISTIC_NOTE = ("Event classes are heuristic (receiver edge, local cause, night window, dual-band fade, "
                  "second phone, indicative S4 above its noise floor, C/N0 floor, elevation); they are a "
                  "triage aid, not a confirmed detection.")
COLS = [*fades.EVENT_COLS, "in_night_window", "local_minutes", "local_reason", "both_bands_faded",
        "both_phones", "max_s4", "class", "reason"]


def classify(events: pd.DataFrame, interference: pd.DataFrame, dualfreq: pd.DataFrame,
             s4: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """events may carry `both_phones` (True/False) and `other_phone_tracked` (True/False) from
    twophone.flag_events; without a second phone both are absent / None."""
    if not len(events):
        return pd.DataFrame(columns=COLS)
    c = cfg["classify"]
    flagged = interference[interference["local_cause_flag"]].set_index("minute_utc")["reason"]
    s4v = s4[s4["s4_valid"].astype(bool)] if len(s4) else s4
    rows = []
    for ev in events.to_dict("records"):
        mins = pd.date_range(ev["start_utc"], ev["end_utc"] - pd.Timedelta(minutes=1), freq="1min")
        local = flagged[flagged.index.isin(mins)]
        night = bool(common.in_night_window(mins, cfg).any())
        sat = lambda df: df[(df["constellation"] == ev["constellation"]) & (df["svid"] == ev["svid"])  # noqa: E731
                            & df["minute_utc"].isin(mins)]
        both = bool(sat(dualfreq)["both_faded"].any()) if len(dualfreq) else False
        s4s = sat(s4v)["s4"] if len(s4v) else pd.Series(dtype=float)
        max_s4 = float(s4s.max()) if s4s.notna().any() else np.nan
        s4_hit = np.isfinite(max_s4) and max_s4 >= c["iono_s4"]
        two = ev.get("both_phones")
        two = None if two is None or (isinstance(two, float) and np.isnan(two)) else bool(two)
        one_phone_only = two is False and bool(ev.get("other_phone_tracked"))
        elev, cn0 = ev["elev_deg_at_peak"], ev["cn0_mean"]
        low_cn0 = not (np.isfinite(cn0) and cn0 >= c["iono_likely_min_cn0"])
        evidence = [x for x, ok in (("both bands faded", both), (f"S4 {max_s4:.2f} >= {c['iono_s4']}", s4_hit),
                                    ("same satellite faded on both phones", bool(two))) if ok]
        if ev["edge_of_gap"]:
            cls, why = "edge_of_gap", f"within {cfg['fades']['edge_guard_s']} s of a receiver silence"
        elif len(local):
            cls, why = "local", f"overlaps {len(local)} local-cause minute(s): {', '.join(sorted(set(local)))}"
        elif one_phone_only:
            cls, why = "local", "fade on one phone only; the other phone tracked the same satellite and band without a fade"
        elif not night:
            cls, why = "local_or_unknown", "outside_night_window"
        elif evidence and low_cn0:
            cls = "iono_possible"
            why = (f"low_cn0 (mean {cn0:.1f} < {c['iono_likely_min_cn0']} dB-Hz); otherwise iono_likely ("
                   + ", ".join(evidence) + ")")
        elif evidence:
            cls, why = "iono_likely", "night window; " + " and ".join(evidence)
        elif np.isfinite(elev) and elev < c["low_elev_deg"]:
            cls, why = "low_elev", f"elevation at peak {elev:.0f}° < {c['low_elev_deg']}°"
        else:
            cls, why = "iono_possible", "night window; single band only; no valid S4 above threshold"
        rows.append({**ev, "in_night_window": night, "local_minutes": int(len(local)),
                     "local_reason": ",".join(sorted(set(local))), "both_bands_faded": both, "both_phones": two,
                     "max_s4": max_s4, "class": cls, "reason": why})
    return pd.DataFrame(rows)[COLS]
