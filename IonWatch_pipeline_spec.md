# IonWatch: Offline Processing Pipeline Spec (L86 + Android phone)

Build spec for Claude Code. Goal: process one recorded overnight session from two receivers and produce per-minute ionospheric indices, events and plots. **No dashboard, MQTT or server in this phase.** The output files and schemas below will later be streamed to the dashboard unchanged.

---

## 1. Context

| Receiver | Hardware | What it logs | Rate | Used for |
|---|---|---|---|---|
| A | Quectel L86-M33 (7Semi board) + outdoor 28 dB antenna, on Raspberry Pi UART | NMEA 0183 text (GSV, GGA, RMC, …) | 1–10 Hz (configured 10 Hz; must be verified from data) | S4, C/N0 fades on GPS / GLONASS / Galileo L1 |
| B | iQOO phone (vivo I2221, Qualcomm) running Google GnssLogger v3.1 | GnssLogger `.txt` (`Raw`, `Status`, `Fix`, `Agc` rows) | 1 Hz | NavIC L5 health, dual-frequency code-only TEC, C/N0 fades |

Known facts from the 18 Oct 2026 phone test log (use these as acceptance checks):
- 6,422 `Raw` rows, 300 epochs, 1.000 s spacing.
- NavIC = `ConstellationType 7`, Svids {2, 9}, `CarrierFrequencyHz 1176450000` (L5).
- Dual frequency present: GPS L1+L5, Galileo E1+E5a, BeiDou B1+B2a.
- `AccumulatedDeltaRangeState` = 16 on every row, so **no usable carrier phase**. Do not attempt carrier-phase TEC or ROTI.
- Pseudoranges computed as below fall in GPS ≈ 20,100–23,600 km and Galileo ≈ 23,300–28,000 km.

Location: Haldwani, Uttarakhand (~29.256° N, 79.537° E, ~500 m). Display timezone: `Asia/Kolkata` (IST). Store everything in UTC.

---

## 2. Repository layout

```
ionwatch/
  pyproject.toml            # or requirements.txt
  config/default.yaml
  ionwatch/
    __init__.py
    cli.py                  # entry point
    io/
      nmea_l86.py           # parse L86 NMEA log -> tidy C/N0 table
      gnsslogger.py         # parse GnssLogger txt -> raw + status tables
    proc/
      common.py             # signal naming, time utils, elevation join
      s4.py
      fades.py              # baseline, fade depth, lock loss (both receivers)
      navic.py              # NavIC health summary (wraps fades for const 7)
      tec_code.py           # phone L1/L5 pseudorange TEC
      crosscheck.py         # L86 vs phone agreement
    report/
      plots.py
      summary.py
  tests/
    test_gnsslogger.py
    test_nmea_l86.py
    test_s4.py
    test_tec_code.py
    fixtures/               # small trimmed log snippets
  data/
    raw/                    # input logs (not committed)
    processed/<session>/    # outputs
```

Python ≥ 3.10. Dependencies: `pandas`, `numpy`, `pyarrow`, `pynmea2`, `matplotlib`, `pyyaml`, `click` (or argparse), `pytest`. No network access needed.

---

## 3. CLI

```bash
python -m ionwatch.cli run \
  --l86   data/raw/l86_2026-10-20.nmea \
  --phone data/raw/gnss_log_2026_10_20_19_00_00.txt [--phone more_files...] \
  --out   data/processed/2026-10-20 \
  --config config/default.yaml
```

- Either `--l86` or `--phone` may be omitted; run whatever is given.
- Multiple phone files (GnssLogger may split sessions) are concatenated and de-duplicated on (`utcTimeMillis`, `ConstellationType`, `Svid`, `CarrierFrequencyHz`).
- Also provide `python -m ionwatch.cli inspect --l86 FILE` / `--phone FILE`, which prints a data-quality report only (Section 4.3 / 5.3) without computing indices.

---

## 4. Receiver A: L86 NMEA parser (`io/nmea_l86.py`)

### 4.1 Input
A text log captured from `/dev/serial0`. **The exact capture format is not fixed; detect it:**
- (a) plain NMEA lines `$GPGSV,...*hh`, or
- (b) lines prefixed by a host timestamp, e.g. `2026-10-20T13:30:00.123Z,$GPGSV,...` or `1792500000.123 $GPGSV,...`.

Rules:
- Validate the NMEA checksum; drop and count bad lines.
- Ignore non-`$` noise and PMTK acknowledgements (count them).
- **Timestamps:** GSV sentences carry no time. Assign each GSV block the UTC time of the most recent `RMC`, `GGA` or `ZDA` in the same update cycle (`hhmmss.sss` + date from RMC/ZDA). If a host-timestamp prefix exists, use it as a fallback and report the median offset between host time and NMEA time.
- Multi-sentence GSV groups (`total, msg_num`) belong to one epoch. Assemble per talker and epoch.

### 4.2 Output table `l86_obs`

| column | type | notes |
|---|---|---|
| time_utc | datetime64[ns, UTC] | epoch time |
| talker | str | GP, GL, GA, GB, GN… |
| constellation | str | GPS, GLONASS, Galileo, BeiDou, QZSS, SBAS |
| prn_raw | int | as in sentence |
| svid | int | normalised (see below) |
| signal | str | `L1` (L86 is L1-only) |
| elev_deg | float | from GSV |
| az_deg | float | from GSV |
| cn0_dbhz | float | NaN if empty field (tracked but no SNR) |

PRN normalisation (MediaTek/Quectel convention; **verify against data and print the observed ID ranges per talker**):
- GPS 1–32 → svid as-is.
- GLONASS 65–96 → svid = prn − 64 (slot).
- Galileo: if IDs are 1–36 keep; if 301–336, subtract 300; if 211–246 or other, log and keep raw.
- SBAS 33–64 (or 120–158): keep and tag SBAS; excluded from indices.

### 4.3 Data-quality report (print + `l86_quality.json`)
- Time span (UTC and IST), number of epochs, epoch spacing histogram (should be ~100 ms at 10 Hz).
- **Effective C/N0 update rate per satellite:** the fraction of consecutive samples where cn0 changes. If C/N0 only changes about once per second despite 10 Hz output, set `effective_rate_hz ≈ 1` and warn: *"L86 repeats C/N0 between fixes; S4 from L86 is low-rate."*
- Satellites seen per constellation, bad checksums, gaps longer than 5 s.

---

## 5. Receiver B: GnssLogger parser (`io/gnsslogger.py`)

### 5.1 Input
GnssLogger text file. Lines starting with `#` are comments; the column headers are in comment lines like `# Raw,utcTimeMillis,TimeNanos,...`. **Read the column names from the header lines of the file itself; don't hard-code the order.** Row types needed: `Raw`, `Status`; optional `Fix`, `Agc`.

### 5.2 Tables

`phone_raw`: all `Raw` columns as numeric, plus:
- `time_utc` from `utcTimeMillis`
- `constellation` from `ConstellationType`: 1 GPS, 2 SBAS, 3 GLONASS, 4 QZSS, 5 BeiDou, 6 Galileo, 7 NavIC
- `band` from `CarrierFrequencyHz`. Classify ±1 MHz around: 1575.42 → L1/E1/B1C, 1561.098 → B1I, ~1602 (GLONASS FDMA, 1598–1606) → G1, 1176.45 → L5/E5a/B2a/NavIC-L5. Name signals like `GPS_L1`, `GPS_L5`, `GAL_E1`, `GAL_E5a`, `BDS_B1I`, `BDS_B1C`, `BDS_B2a`, `GLO_G1`, `NAVIC_L5`.
- Use `CodeType` when present to disambiguate.

`phone_status`: `Status` rows with `UnixTimeMillis`, `ConstellationType`, `Svid`, `CarrierFrequencyHz`, `Cn0DbHz`, `AzimuthDegrees`, `ElevationDegrees`, `UsedInFix`.

**Elevation join:** `Raw` has no elevation. Attach `elev_deg` and `az_deg` from `Status` by nearest time (tolerance 1.5 s) on (`ConstellationType`, `Svid`, band). Unmatched rows get NaN, and the match rate is reported.

### 5.3 Data-quality report (`phone_quality.json`)
- Span, epochs, spacing, `HardwareClockDiscontinuityCount` changes.
- Rows per constellation/band, NavIC svids seen, mean C/N0 per signal.
- ADR: count of rows with `AccumulatedDeltaRangeState & 1` (VALID). Expected 0, which means carrier phase is unavailable.
- Elevation-join match rate.

---

## 6. Processing

Defaults in `config/default.yaml`:

```yaml
elev_mask_deg: 20
s4:
  window_s: 60
  detrend_window_s: 60      # running mean used to normalise intensity
  min_coverage: 0.8         # fraction of expected samples in window
  quiet_below: 0.2
  moderate: 0.3
  strong: 0.6
fades:
  baseline_window_min: 15   # running median of C/N0
  fade_threshold_db: 6
  lock_loss_min_gap_s: 3    # missing for >= this while elev > mask
tec:
  avg_window_s: 60
  shell_height_km: 350
  max_abs_stec_tecu: 400    # reject epoch outliers before averaging
crosscheck:
  window_s: 60
  min_overlap_s: 30
display_tz: Asia/Kolkata
```

### 6.1 S4 (`proc/s4.py`)
Run on the L86 table (main) and on phone C/N0 (flagged `indicative_1hz`).

Per satellite and signal, in non-overlapping 60 s windows aligned to the minute (UTC):
1. Keep samples with `elev_deg ≥ mask` and finite cn0.
2. `I = 10 ** (cn0_dbhz / 10)`.
3. Detrend: `I_d = I / rolling_mean(I, detrend_window_s, centered)`.
4. `S4 = sqrt( (mean(I_d²) − mean(I_d)²) / mean(I_d)² )`.
5. Require `n ≥ min_coverage × expected_n` (expected from the effective rate in the quality report), otherwise S4 = NaN.
6. `level` = quiet / weak / moderate / strong using the thresholds; values below 0.2 are labelled `quiet` because of the 1 dB-Hz C/N0 quantisation.

Output `s4_1min.csv`: `minute_utc, minute_ist, source(l86|phone), constellation, svid, signal, elev_deg_mean, az_deg_mean, n, eff_rate_hz, cn0_mean, s4, level, indicative_1hz(bool)`.

### 6.2 Fades and lock loss (`proc/fades.py`)
For every satellite and signal on both receivers:
- `baseline` = rolling median of cn0 over 15 min (centered, min 50 % data).
- Per minute: `fade_depth_db = max(baseline − cn0)`, `cn0_min`, `cn0_mean`.
- `lock_losses` = number of gaps ≥ 3 s within the minute while the satellite's last known elevation ≥ mask. Use the satellite's last-known elevation (linear interpolation across the gap is fine).
- Event when `fade_depth_db ≥ 6` or `lock_losses ≥ 1`. Merge consecutive event minutes into one event with start, end, peak depth and total lock losses.

Outputs: `fades_1min.csv`, `fade_events.csv` (`event_id, source, constellation, svid, signal, start_utc, end_utc, start_ist, peak_depth_db, lock_losses, elev_deg_at_peak`).

### 6.3 NavIC health (`proc/navic.py`)
The subset of 6.2 for `constellation == NavIC` (phone only), plus `navic_summary.json`:
- Satellites seen, tracking availability % per satellite (minutes with ≥ 1 sample ÷ session minutes).
- Mean / min C/N0 and number of fade events per satellite.
- Minutes in which **all** NavIC satellites were faded or lost (a NavIC service-impact indicator).

### 6.4 Code-only TEC (`proc/tec_code.py`, phone only)
Pseudorange per `Raw` row:

```
t_rx_ns  = TimeNanos − (FullBiasNanos + BiasNanos)          # GPS time, ns
t_rx_tow = t_rx_ns mod 604800e9                              # GPS & Galileo
t_tx     = ReceivedSvTimeNanos + TimeOffsetNanos
P (m)    = (t_rx_tow − t_tx) × 299792458 / 1e9               # handle week rollover: if P < 0 add 604800e9×c/1e9
```

- Compute `TimeNanos − FullBiasNanos` in **int64** before converting to float (precision).
- Use only GPS and Galileo. Require `State` with TOW known/decoded (bit 3 `TOW_DECODED` = 8 or bit 14 `TOW_KNOWN` = 16384) and `ReceivedSvTimeUncertaintyNanos ≤ 100`.
- Keep only physically valid P (18,000–30,000 km).
- For each epoch and satellite having both L1/E1 and L5/E5a:

```
k = f1² f5² / (40.3 (f1² − f5²)) / 1e16   # TECU per metre, f1 = 1575.42e6, f5 = 1176.45e6  → k ≈ 7.76
STEC = k × (P_L5 − P_L1)
```

- Reject |STEC| > `max_abs_stec_tecu`, then compute a 60 s mean and standard deviation per satellite.
- `VTEC_rel = STEC × cos(z')`, where `sin z' = R/(R+h) × cos(elev)`, R = 6371 km, h = 350 km.
- **Label every TEC output "uncalibrated: includes receiver and satellite code biases".** It's for trends only; no ROTI.

Output `tec_code_1min.csv`: `minute_utc, minute_ist, constellation, svid, elev_deg, n, stec_mean, stec_std, vtec_rel_mean`.

### 6.5 Cross-check (`proc/crosscheck.py`)
Match satellites tracked by **both** receivers on L1: GPS svid, GLONASS slot, Galileo svid.
- Resample L86 cn0 to 1 Hz (mean) and align with phone L1 cn0 on UTC seconds.
- Per minute: Pearson correlation of detrended cn0 (subtract the 15-min baseline) and the overlap length.
- **Confirmed event:** a fade event (6.2) on one receiver that overlaps in time (±60 s) with a fade event on the same satellite on the other receiver.
- Output `crosscheck_1min.csv` and `confirmed_events.csv` (`satellite, start_utc, end_utc, start_ist, l86_peak_db, phone_peak_db, corr`).

---

## 7. Plots (`report/plots.py`) → `out/plots/*.png`, x-axis in IST
1. `cn0_timeline_<constellation>.png`: C/N0 vs time per satellite, both receivers, fade events shaded.
2. `s4_timeline.png`: per-minute S4 per satellite (L86), dashed lines at 0.3 and 0.6, night window 19:30–01:00 IST shaded.
3. `navic_health.png`: NavIC satellites' C/N0 with baseline and fade events.
4. `tec_code.png`: 60 s mean STEC per GPS / Galileo satellite, subtitled "uncalibrated".
5. `skyplot_tracks.png`: azimuth/elevation tracks coloured by max S4.
6. `crosscheck_<sat>.png` for each confirmed event: both receivers' C/N0 overlaid.

---

## 8. Summary (`report/summary.py`) → `summary.json` + `summary.md`
- Session span (IST), receivers present, data-quality headlines.
- Count of minutes per S4 level, max S4 (satellite + time).
- Fade events per receiver/constellation; confirmed (cross-checked) events.
- NavIC availability and impact minutes.
- TEC trend range per satellite (uncalibrated).
- A plain-language list of warnings (e.g. low effective L86 rate, low elevation-join rate, gaps).

---

## 9. Output schema conventions (for the later dashboard)
- All times are written twice: `*_utc` (ISO 8601, `Z`) and `*_ist`.
- CSV and Parquet both written for every table.
- One row per (minute, source, constellation, svid, signal). These rows become the MQTT JSON payloads later, unchanged.
- Add a `session_id` column (folder name) to every table.

---

## 10. Tests (pytest)
- **GnssLogger parser** on the trimmed 18 Oct fixture: correct row counts, NavIC svids {2, 9} at 1176.45 MHz, ADR-valid count 0, 1 s epoch spacing.
- **Pseudorange sanity:** GPS within 20,000–24,000 km and Galileo within 23,000–28,500 km for the fixture.
- **TEC constant:** `k` ≈ 7.76 ± 0.01 TECU/m.
- **S4 synthetic:** constant C/N0 → S4 < 0.01. Intensity with a known std/mean ratio (e.g. lognormal, 10 Hz, 60 s) → S4 within 5 % of truth. 1 dB quantised noise-only input → S4 < 0.2.
- **NMEA parser:** multi-sentence GSV assembly, empty SNR fields → NaN, checksum rejection, timestamp assignment from RMC.
- **Fades:** synthetic 10 dB drop for 90 s → one event, depth ≈ 10 dB. A 5 s gap at high elevation → 1 lock loss.

---

## 11. Build order
1. `inspect` for both inputs. Run it on the real overnight files first and **report the findings before writing the processing code** (L86 timestamp format, ID ranges, effective rate; phone signal inventory).
2. Parsers + tests.
3. Fades → NavIC → S4 → TEC → cross-check.
4. Plots + summary.
5. One command runs everything on the overnight session and writes `data/processed/<session>/`.

## 12. Non-goals for this phase
Dashboard, MQTT/InfluxDB, live serial reading, IGS RINEX/ROTI, ML forecast, carrier-phase anything.
