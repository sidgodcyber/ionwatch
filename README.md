# IonWatch offline pipeline

Per-minute ionospheric indices, fade events and plots from a recorded session of an Android
phone (Google GnssLogger) and, optionally, a Quectel L86 NMEA log.

## Quick start for the team

### 1. Install (Python ≥ 3.10)

```bash
git clone https://github.com/sidgodcyber/ionwatch.git
cd ionwatch
python3 -m venv .venv
source .venv/bin/activate            # Windows (Git Bash): source .venv/Scripts/activate
                                     # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e .[test]
pytest -q                            # expect "passed" with 2 skipped (Linux-only pty test, opt-in network test)
```

On the Raspberry Pi use `sudo bash deploy/setup_pi.sh` instead (then reboot). It enables the UART,
installs the system packages, creates `.venv` and installs the service (`deploy/PI_CHECKLIST.md`).

### 2. Demo replay + dashboard (no hardware needed)

```bash
bash deploy/demo.sh                  # options: --speed 60 --port 8080
```

It replays the 20 Oct 2026 night log at 60× through the live path and serves the dashboard on the
same folder. Open http://localhost:8080 (or `http://<this machine's IP>:8080` from a phone on the
same Wi-Fi). Ctrl-C stops both. `data/` is not in git: on first run the script unpacks the log from
`demo/gnss_log_2026_10_20_18_58_37.txt.gz` into `data/raw/`. Output goes to `data/replay/demo/`
(wiped at each start).

### 3. Joint session on hardware day (full run sheet: `deploy/JOINT_SESSION.md`)

On the Pi, inside `tmux new -s iw`, with `cd ~/ionwatch && source .venv/bin/activate`:

```bash
# Quick trial (10 min) + GnssLogger logging on the phone
python -m ionwatch.cli session note --antenna external --phone "iQOO vivo I2221" --notes "trial" --dir data/trial/
python -m ionwatch.cli live --l86 /dev/serial0 --l86-rate 1 --out data/trial/ --duration-s 600    # pane 1
python -m ionwatch.cli status --data data/trial/                                                  # pane 2
python -m ionwatch.cli run --phone data/raw/<trial gnss_log>.txt --l86 data/trial/raw/nmea_<UTC date>.nmea \
    --out data/processed/trial --no-plots

# Main run (≥ 60 min, ~6 h if possible); start GnssLogger at the same time
python -m ionwatch.cli session note --antenna external --phone "iQOO vivo I2221" --notes "<site, sky>"
#   optional second phone: add  --phone2 "OnePlus CPH2569"
rm -rf data/live/minutes.jsonl data/live/latest.json
python -m ionwatch.cli live --l86 /dev/serial0 --l86-rate <rate from the trial> --out data/live/   # pane 1
python -m ionwatch.cli dashboard --data data/live/ --port 8080                                     # pane 2
python -m ionwatch.cli status --data data/live/                                                    # pane 3

# After: copy the phone log into data/raw/ (adb pull, or GnssLogger -> share -> Drive/WhatsApp), then
python -m ionwatch.cli run --phone data/raw/<gnss_log>.txt --l86 data/live/raw/nmea_<date>.nmea \
    --out data/processed/<date> --trim-start 5          # second phone: --phone2 data/raw/<its gnss_log>.txt
python -m ionwatch.cli validate data/live/minutes.jsonl
python -m ionwatch.cli bench fetch --out data/bench/ --processed data/processed/<date>    # needs internet
python -m ionwatch.cli bench run --bench data/bench/ --processed data/processed/<date>
```

Pass criteria, transfer fallbacks and troubleshooting are in `deploy/JOINT_SESSION.md`. CDDIS
downloads need an Earthdata login in `~/.netrc` (see *Earthdata login* below; never put credentials
in the code or config).

## All commands

```bash
pip install -e .[test]

# data-quality report only
python -m ionwatch.cli inspect --phone data/raw/gnss_log_2026_10_20_18_58_37.txt

# full run -> data/processed/<session>/  (folder name = session_id)
python -m ionwatch.cli run --phone data/raw/gnss_log_2026_10_20_18_58_37.txt \
    --out data/processed/2026-10-20 [--l86 data/raw/l86.nmea] [--config config/default.yaml]

# live L86 (Pi: /dev/serial0; laptop: COM port or the simulator's socket:// URL); --l86-rate 1|5|10
python -m ionwatch.cli live --l86 /dev/serial0 --baud 9600 --node-id haldwani-01 --out data/live/

# dashboard over a live, replay or batch folder (open http://<pi>:8080 from a laptop/phone)
python -m ionwatch.cli dashboard --data data/live/ --port 8080

# demo: replay the 20 Oct night at 60x + dashboard, one command
bash deploy/demo.sh

# replay a phone log through the live path (demo fallback)
python -m ionwatch.cli replay --phone data/raw/gnss_log_2026_10_20_18_58_37.txt --speed 60 --out data/replay/2026-10-20/

# check a minutes.jsonl (schema, one record per minute, no gaps)
python -m ionwatch.cli validate data/live/minutes.jsonl

# session notes (antenna / phone / free text; times always come from the data), health line every 60 s
python -m ionwatch.cli session note --antenna external --phone "iQOO vivo I2221" --notes "roof" --dir data/live/
python -m ionwatch.cli status --data data/live/

# joint L86 + phone batch run (the L86 file is the daily NMEA file live mode saved); drop setup minutes
python -m ionwatch.cli run --phone data/raw/<gnss_log>.txt --l86 data/live/raw/nmea_<date>.nmea \
    --out data/processed/<date> --trim-start 5

# IGS benchmark for the session's night (needs internet; see "Earthdata login" below)
python -m ionwatch.cli bench fetch --out data/bench/ --processed data/processed/<date>   # or --date YYYY-MM-DD
python -m ionwatch.cli bench run --bench data/bench/ --processed data/processed/<date>

# serial simulator: prints "PORT <pty or socket://...>"; inject a fade / dropout / garbage / unplug
python tools/fake_l86.py --speed 40 --inject-fade 12:70:40:10 --inject-garbage 30
python -m ionwatch.cli live --l86 socket://127.0.0.1:<port> --clock-scale 40 --out data/live-sim/
python tools/nmea_from_phone.py data/raw/<gnss_log>.txt sim.nmea   # L86-style NMEA from a phone log

pytest
```

Pi bring-up: `deploy/PI_CHECKLIST.md`. Joint L86 + phone session (any day): `deploy/JOINT_SESSION.md`.

## IGS benchmark (`bench fetch` / `bench run`)

`bench fetch` downloads, for each UTC day of the session (the date comes from the data):
IGS daily RINEX obs for LCK3 and LCK4 (Lucknow, ~295 km from Haldwani) with IISC and HYDE as backups,
BRDM broadcast navigation, a GIM (IGS final → IGS rapid → JPL → UPC) plus one from a second producer
for a cross-check, DCB/OSB biases, and Kp / F10.7 from NOAA SWPC and GFZ. Every file is validated
(station, date, epochs, observation types, map coverage, value ranges) and the sources are cross-checked
(Kp SWPC vs GFZ, F10.7 SWPC vs GFZ, GIM producer vs producer at Haldwani). `manifest.json` records the URL,
size and SHA-256 of every file. `bench run` computes 5-min GPS L1/L2 ROTI (elevation ≥ 30°) over the
night window, GIM VTEC at Haldwani and Kp for that night, checks the broadcast orbits against the phone's
own satellite elevations, and writes `bench_summary.md` with our event times against IGS ROTI.

Without a login, the anonymous mirrors still provide LCK4 and IISC (BKG), BRDM (BKG), JPL and UPC GIMs.
The IGS combined GIM (~2 days after the day), CODE and ESA GIMs (next day) and the bias files (CAS/GFZ
DCB, CODE OSB) are on NASA CDDIS only. Station files vary by day: for 20 Oct 2026 LCK3 and HYDE are not
in any archive (HTTP 404 on CDDIS too), so the benchmark uses LCK4 and IISC.

### Earthdata login (NASA CDDIS)

1. Create a free account at https://urs.earthdata.nasa.gov/users/new and log in once at
   https://cddis.nasa.gov/archive/ in a browser (accepts the terms).
2. Put the credentials in a netrc file, **never in the code or the config**:
   - Linux / macOS / Pi: `~/.netrc`, then `chmod 600 ~/.netrc`
   - Windows: `%USERPROFILE%\.netrc` (`_netrc` is also accepted; `NETRC=<path>` overrides both)
   - Never inside the project folder: `bench fetch` refuses a netrc there, and `.gitignore` excludes
     `.netrc` / `_netrc`.

   ```
   machine urs.earthdata.nasa.gov login <your user name> password <your password>
   ```
3. Check: `python -m ionwatch.cli bench fetch --date 2026-10-20` prints
   `Earthdata login: found in <path> (CDDIS enabled)` (or why it is not used: no file, no
   `urs.earthdata.nasa.gov` entry, a parse error by line number; never the contents), and `ionex_check`
   then comes from `cddis.nasa.gov`. `requests` re-applies the netrc credentials when CDDIS redirects
   to the Earthdata login; session cookies stay in memory, and nothing is written to disk except the
   downloaded files.

## Live / replay output

`minutes.jsonl` (schema `ionwatch/minute_record.schema.json`, built by `ionwatch/schema.py` for
batch, live and replay alike), `latest.json` (the newest minute, replaced atomically),
`run_info.json` (mode, source), `live.log` / `replay.log`, and in live mode `raw/nmea_YYYY-MM-DD.nmea`
(host-time-prefixed NMEA, re-runnable with `run --l86`).

Live and replay publish every minute twice: `status: "provisional"` (`rev` 1) ~5 s after the
minute, and `status: "final"` (`rev` 2) once the full 15-min baseline, the AGC median, the
edge-of-gap guard and every fade event touching the minute are complete (normally ~8.6 min later,
later inside a long event, capped at 30 min). Final records equal the batch result exactly
(tested on the 20 Oct night). Readers keep the highest `rev`; batch writes finals only (`rev` 1).
`ionwatch validate` checks one final per minute.

## Outputs (CSV + Parquet, every row has `session_id`, times as `*_utc` and `*_ist`)

| file | content |
|---|---|
| `phone_quality.json`, `l86_quality.json` | data-quality reports and warnings |
| `fades_1min`, `fade_events` | per-minute fade depth, lock losses, receiver_off gaps; merged events |
| `s4_1min` | S4 per minute; phone rows are `method=indicative_1Hz`, `s4_label` = `s4_raw` / `s4_indicative` |
| `navic_1min`, `navic_events`, `navic_summary.json` | NavIC subset; status `tracked` / `visible_not_tracked` / `no_data` |
| `tec_code_1min`, `tec_code_epochs` | code-only L1/L5 STEC, **uncalibrated** (receiver + satellite code biases) |
| `dualfreq_1min` | L1 vs L5 fade depth and detrended C/N0 correlation |
| `interference_1min` | share of satellites faded, AGC change, `local_cause_flag` |
| `events_classified` | heuristic class: edge_of_gap / local / iono_likely / low_elev / iono_possible |
| `l86_obs`, `crosscheck_1min`, `confirmed_events` | only with `--l86` (not yet validated on real data) |
| `tables/cn0_obs.parquet` | per-sample C/N0 with baseline (both receivers) |
| `summary.json`, `summary.md`, `plots/*.png` | session summary and figures (IST axis) |

Every C/N0-derived row carries `receiver` (phone / l86) and `source` (raw / status / nmea / mixed).

## Phone data handling worth knowing

- **Status timing.** Many Status rows have an empty `UnixTimeMillis`. They are timed from
  their file position between timestamped Raw/Agc/Fix rows (`time_quality`); blocks that
  cannot be placed (`ambiguous`) are dropped and counted.
- **Duty cycling.** After Raw stops, phones may log Status in bursts. Those bursts are used
  (`source=status`); gaps between them are `receiver_off`, never lock loss.
- **Lock loss** = a satellite missing >= 3 s while the receiver kept running and the
  satellite stayed above the mask.
- **Edge of gap.** Fades and lock losses within 60 s of a receiver silence (> 60 s) are
  ignored; any event touching that zone is tagged `edge_of_gap` as a whole and reported only.
- **S4 validity.** `s4_noise = sqrt(1 / (T·10^(C/N0/10)))`, T = one C/N0 sample (1 s on the
  phone); S4 counts only when > 2× s4_noise. Windows that are flat apart from one or two
  re-lock jumps (>= 6 dB within 2 samples) are `step_artifact` (NaN).
- **iono_likely** additionally needs the event's mean C/N0 >= 30 dB-Hz; below that it is
  capped at `iono_possible` (reason `low_cn0`).
- **TEC plots** show ΔSTEC per arc (minus the arc's first 60 s mean); absolute STEC is in the CSV.
- **AGC** values <= -100 dB (NavIC on the vivo I2221) are placeholders and excluded, and the
  first 60 s after each receiver start are skipped while the front end settles.
- **Pseudorange** uses the Android convention `t_rx = TimeNanos + TimeOffsetNanos - (FullBiasNanos + BiasNanos)`
  (the spec's formula had the TimeOffsetNanos sign reversed; it is 0 in all logs so far).

Test fixtures in `tests/fixtures/` are trimmed copies of three real logs; regenerate them with
`python tests/fixtures/make_phone_fixture.py`.
