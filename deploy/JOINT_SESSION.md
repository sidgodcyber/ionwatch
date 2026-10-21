# IonWatch joint session: L86 on the Pi + iQOO phone (any day)

One page for the team. No clock times are fixed: session date, start and end come from the
data. Aim for ~6 h, ideally inside 19:30–01:00 IST. Run everything on the Pi inside `tmux`
(`tmux new -s iw`; detach Ctrl-b d, re-attach `tmux attach -t iw`) so an SSH or Wi-Fi drop
cannot stop live mode. Commands assume `cd ~/ionwatch && source .venv/bin/activate`.

## Before

- [ ] `deploy/PI_CHECKLIST.md` passed, including the 10-min acceptance test and `pytest -q tests/test_live.py -k pty`.
- [ ] **Antenna (external, 3M SMA cable + IPEX-to-SMA pigtail):** connect the IPEX end to the L86 board's antenna
      socket **before powering on**; put the antenna outdoors with open sky (magnetic base on a metal surface);
      keep the cable away from the Pi's power supply.
      **Fallback, if the connector doesn't fit:** the on-board patch: board flat, patch up, ≥ 30 cm from the Pi.
- [ ] **iQOO (vivo I2221) in GnssLogger:** background activity allowed and battery **Unrestricted**
      (App info → Battery); Developer options: **Force full GNSS measurements** and **Stay awake** both on;
      on a charger; logging of **Raw, Status, Fix and AGC** enabled (Settings tab); phone within 1 m of the antenna,
      screen up, not moved during the run.

## Quick trial (10 min)

```bash
python -m ionwatch.cli session note --antenna external --phone "iQOO vivo I2221" --notes "trial" --dir data/trial/
python -m ionwatch.cli live --l86 /dev/serial0 --l86-rate 1 --out data/trial/ --duration-s 600    # tmux pane 1
python -m ionwatch.cli status --data data/trial/                                                  # tmux pane 2
```

Start GnssLogger logging at the same time. After 10 min stop it, copy the phone log (see *After*:
`adb pull`, or GnssLogger → share → Drive/WhatsApp → `data/raw/` if Android 11+ blocks adb) and join:

```bash
python -m ionwatch.cli run --phone data/raw/<trial gnss_log>.txt --l86 data/trial/raw/nmea_<UTC date>.nmea \
    --out data/processed/trial --no-plots
```

Check before the main run (if one fails, move the antenna or phone and repeat the trial):

| Check | Where | Pass |
|---|---|---|
| L86 satellites and C/N0 | `status` line: `sats`, `C/N0` | ≥ 6 sats and mean C/N0 ≥ 35 dB-Hz (external) or ≥ 30 (on-board) |
| NavIC on the phone | GnssLogger **Status** tab, IRNSS/NavIC rows | C/N0 > 0 |
| Join | `data/processed/trial/summary.md` → *L86 vs phone cross-check* | matched satellites listed, clock offset 0 s |

**Rate:** start at 1 Hz. Try `--l86-rate 5`, then `10`, only if the `status` line shows the measured `L86 … Hz`
holding at the requested rate for 2 min (and no `l86_rate_fallback` warning). Otherwise stay at the last rate that held.

## Main run (at least 60 min; ~6 h if possible)

Within 1 min of each other:

```bash
python -m ionwatch.cli session note --antenna external --phone "iQOO vivo I2221" --notes "<site, sky, anything odd>"
#   optional second phone at the same spot: add  --phone2 "OnePlus CPH2569"
rm -rf data/live/minutes.jsonl data/live/latest.json      # keep data/live/raw/ (old days are separate files)
python -m ionwatch.cli live --l86 /dev/serial0 --l86-rate <rate from the trial> --out data/live/   # tmux pane 1
```

…and start GnssLogger logging. Then:

```bash
python -m ionwatch.cli dashboard --data data/live/ --port 8080      # tmux pane 2; open http://<pi-ip>:8080
python -m ionwatch.cli status --data data/live/                     # tmux pane 3 (backup); pi-ip: hostname -I
```

Check every 15 min on the dashboard (or `status`). **Don't touch the devices.** Write anything unusual into
the notes afterwards (`session note … --notes`, same folder).

## After

1. Stop GnssLogger, then live mode (Ctrl-C in pane 1).
2. Copy the phone log(s) into `data/raw/` (the files are in
   `Android/data/com.google.android.apps.location.gps.gnsslogger/files`):
   - **USB:** `adb pull /sdcard/Android/data/com.google.android.apps.location.gps.gnsslogger/files/ data/raw/`.
     On **Android 11 and later** this folder is often blocked (`adb pull` then reports *Permission denied*
     or copies nothing); use the fallback.
   - **Fallback:** GnssLogger → **Files/Share** (share the log) → **Drive** or **WhatsApp** (send as a
     *document*, not a photo/media, so it isn't altered) → download it on the Pi or laptop → copy into
     `data/raw/` with its original name `gnss_log_YYYY_MM_DD_HH_MM_SS.txt`. Zip it first if the app offers
     only media sharing. Check the copy: `python -m ionwatch.cli inspect --phone data/raw/<log>.txt`.
3. Join, validate, package (`<date>` = the UTC date in the NMEA file name):

```bash
python -m ionwatch.cli run --phone data/raw/<gnss_log>.txt --l86 data/live/raw/nmea_<date>.nmea \
    --out data/processed/<date> --trim-start 5     # second phone: add --phone2 data/raw/<its gnss_log>.txt
python -m ionwatch.cli validate data/live/minutes.jsonl
zip -r ionwatch_<date>.zip data/processed/<date>/summary.md data/processed/<date>/plots
python -m ionwatch.cli bench fetch --out data/bench/ --processed data/processed/<date>    # needs internet
python -m ionwatch.cli bench run --bench data/bench/ --processed data/processed/<date>
```

A night inside 19:30–01:00 IST stays in one daily NMEA file (files roll over at 00:00 UTC = 05:30 IST).

## Pass criteria (read them in `data/processed/<date>/summary.md`)

- [ ] ≥ 60 min of L86 records (`validate`: L86 minutes from first to last, 0 missing).
- [ ] iQOO Raw coverage ≥ 90 % (*Phone data quality*: Raw span ≈ the session and ≥ 90 % of 1 s epochs; no duty-cycling warning).
- [ ] NavIC tracked on the phone (*NavIC*: status **tracked**).
- [ ] ≥ 4 matched satellites in *L86 vs phone cross-check*.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No NMEA (`status`: no records / STALE; `live.log`: `no bytes`) | TX/RX swapped, L86 unpowered, console on the UART | PI_CHECKLIST §3–4; `timeout 10 cat /dev/serial0` |
| Low C/N0 (< 35 external / < 30 on-board) | antenna indoors / under cover, cable near the Pi PSU, metal overhead | antenna outdoors, open sky, reroute cable; compare with the phone |
| Antenna not detected (C/N0 same as the patch, or no fix outdoors) | IPEX not seated, pigtail damaged, wrong socket | power off, reseat the IPEX on the L86 antenna socket; else use the on-board patch and note `--antenna onboard` |
| `adb pull` copies nothing / *Permission denied* (Android 11+) | scoped storage blocks `Android/data` | GnssLogger → share → Drive/WhatsApp (as a document) → copy into `data/raw/` |
| Phone log stops early / gaps (`duty-cycling` warning) | battery management killed or throttled GnssLogger | Battery: Unrestricted; allow background activity; lock the app in recents; keep on charger, screen on (Stay awake) |
| NavIC not tracked (Status C/N0 = 0) | no NavIC almanac yet, sky view, phone not in full-measurement mode | wait 15 min outdoors; check "Force full GNSS measurements"; rotate/move the phone to open sky |
| Pi under-voltage (`vcgencmd get_throttled` ≠ 0x0) | weak PSU / thin cable | official supply, short cable, nothing else on USB |
| Live mode STALE (`status` STALE line, dashboard age warning) | live mode stopped, L86 unplugged, Pi rebooted | `tmux attach -t iw`; restart live (same `--out`, it appends); check wiring; `tail data/live/live.log`. After a restart, up to ~9 minutes before it keep only provisional records, so `validate` lists them; the batch join is the final result |
| Dashboard not reachable | wrong IP, different Wi-Fi, port blocked, dashboard not running | `hostname -I` on the Pi; same Wi-Fi as the laptop; dashboard pane running; use `status` meanwhile |
