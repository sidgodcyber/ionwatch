# IonWatch Pi session checklist (L86 → Raspberry Pi)

For the team bringing up live mode on the Pi for the first time. Work top to bottom; each
step says what "good" looks like and what to do when it isn't. Everything here has been
tested on a laptop with the serial simulator (`tools/fake_l86.py`), never on a Pi.

## 0. Bring

- [ ] Raspberry Pi (4 or 5) with the **official 5 V / 3 A (Pi 4) or 5 A (Pi 5) supply**. Phone chargers cause
      under-voltage (see `vcgencmd` below).
- [ ] microSD with **64-bit Raspberry Pi OS (Bookworm)**. On 32-bit, `pyarrow` has no wheels and `pip install` fails or takes hours.
- [ ] Quectel L86-M33 board (7Semi) + the 28 dB outdoor antenna and its cable/adapter (u.FL ↔ SMA as needed).
      **If the external antenna hasn't arrived:** use the L86's on-board patch antenna, board flat with the
      patch facing the sky (window sill or outdoors). Expect lower C/N0 (roughly 5–10 dB below the
      active antenna) and a slower first fix; that is fine for bring-up, not for S4.
- [ ] 4 female–female jumper wires; spares.
- [ ] A **USB–UART adapter at 3.3 V** (CP2102 / FT232) as a fallback path to the L86, and to sniff the L86 from the laptop.
- [ ] Multimeter (check the board's VCC and that L86 TX idles at ~3.3 V, never 5 V).
- [ ] Network for the Pi (Ethernet or Wi-Fi) for `apt` / `pip`, and SSH or keyboard + HDMI.
- [ ] Laptop with this repo; the phone with GnssLogger as a reference receiver.
- [ ] A spot with open sky for the antenna (window sill is the minimum; the 20 Oct phone log shows what indoors does to C/N0).

## 1. Wiring (power off while wiring)

| L86 board | Raspberry Pi header |
|---|---|
| VCC | **3.3 V** – pin 1 (or 17) |
| GND | GND – pin 6 |
| TX  | **GPIO15 / RXD** – pin 10 |
| RX  | **GPIO14 / TXD** – pin 8 |

TX goes to RX and RX to TX. The Pi's GPIO is 3.3 V only: never feed it 5 V logic. Connect the antenna before powering up.

## 2. Software setup (once)

```bash
git clone <repo> ~/ionwatch && cd ~/ionwatch
sudo bash deploy/setup_pi.sh                 # add --disable-bt for the full PL011 UART
sudo reboot
```

`setup_pi.sh` enables the UART, removes the serial console, creates `.venv`, installs IonWatch,
adds you to `dialout` and installs (does not start) the `ionwatch-live` service.

After the reboot:

```bash
groups | grep dialout                         # must list dialout
ls -l /dev/serial0                            # -> ttyS0 (mini-UART) or ttyAMA0 (PL011, with --disable-bt)
vcgencmd get_throttled                        # must be throttled=0x0
```

## 3. Confirm NMEA

```bash
stty -F /dev/serial0 9600 raw -echo
timeout 10 cat /dev/serial0
```

Good: about one burst per second of `$GPRMC`, `$GPGGA`, `$GPGSA`, `$GPGSV` (and `$GLGSV`, `$GAGSV`) lines.
Before the first fix, RMC has status `V` and empty position fields, and GSV lists satellites with some SNR fields empty. That is normal.

Then capture 2 minutes and run the data-quality report. Spec §11 asks for these findings before
we trust the L86 numbers:

```bash
timeout 120 cat /dev/serial0 > /tmp/l86_2min.nmea
.venv/bin/python -m ionwatch.cli inspect --l86 /tmp/l86_2min.nmea
```

Record: the output rate (expect 1 Hz at 9600 baud), the effective C/N0 rate, PRN ranges per talker
(GLONASS 65–96? Galileo 1–36 or 301–336? any `$PQ`/`$GQ` QZSS?), the bad-checksum count, and gaps.

**Expected first fix:** 30–60 s with the antenna under open sky (L86 cold start is spec'd under 35 s).
A brand-new or long-unpowered module may take a few minutes. No fix after 5 minutes outdoors means an
antenna problem. Indoors it may never fix; you'll see GSV with low or empty SNR.

**Pty test on the Pi.** The live-over-pty test is skipped on Windows; run it here once:

```bash
.venv/bin/python -m pytest -q tests/test_live.py -k pty      # expect: 1 passed
```

### Output rate: try 1 Hz first, then 5, then 10

Live mode sets the rate itself (`--l86-rate`). Above 1 Hz it switches the L86 to 115200 baud,
keeps only RMC/GGA/GSA/GSV, sets the fix interval, then **measures the real GSV rate over 30 s**.
If it is below 80 % of the request it puts the module back to 1 Hz at 9600 and logs
`l86_rate_fallback`. It re-sends the setup after every reopen, because the L86 forgets it without backup power.

```bash
python -m ionwatch.cli live --l86 /dev/serial0 --l86-rate 1  --out data/rate1/  --duration-s 180
python -m ionwatch.cli live --l86 /dev/serial0 --l86-rate 5  --out data/rate5/  --duration-s 180
python -m ionwatch.cli live --l86 /dev/serial0 --l86-rate 10 --out data/rate10/ --duration-s 180
grep -h "L86 \|ack" data/rate*/live.log                       # acks "ok", "measured ... ok" or "back to 1 Hz"
```

Keep the highest rate that passes (`l86_rate_hz` in the records ≈ requested, few `bad_checksum`).
Only move on from 1 Hz once 1 Hz is clean. If the module stays at 115200 after a test, live mode
copes on its own (it always starts with the 9600-baud switch command, then reopens at 115200).

Note: the L86-M33 (MT3333) does **not** track NavIC, so NavIC comes only from the phone. Every
L86 record shows `navic_status: no_data`, which is expected.

## 4. Failures and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `cat /dev/serial0` prints nothing | TX/RX swapped; L86 unpowered; UART not enabled | Swap pins 8/10; check VCC with the meter and the board LED; re-run `setup_pi.sh` and reboot; `dmesg \| grep tty` |
| `/dev/serial0` missing | `enable_uart=1` not applied | Check `/boot/firmware/config.txt`, reboot |
| `Permission denied` on /dev/serial0 | not in `dialout` yet | Log out and in (or reboot); `groups` |
| A login prompt or kernel text mixed into the NMEA | serial console still on | Remove `console=serial0,...` from `cmdline.txt`; `systemctl disable --now serial-getty@ttyS0` |
| Garbage characters, many `bad_checksum` | wrong baud; mini-UART clock drift | Try `stty ... 115200`; use `--disable-bt` (PL011); keep `enable_uart=1` |
| Sentences cut off, `gsv_incomplete` / `bad_checksum` climbing | L86 at 5–10 Hz but still 9600 baud: a full GSV cycle is ~0.6–0.8 kB, so 9600 baud carries ~1 Hz | Use `--l86-rate 5/10`, which switches to 115200 first; or stay at 1 Hz |
| `l86_rate_fallback` in the records | the measured GSV rate was < 80 % of `--l86-rate` | Use the next lower rate; check the ack lines in `live.log` (`PMTK251/314/220: ok`) |
| RMC stays `V`, SNR mostly empty | no sky view; antenna unplugged; cable fault | Antenna outdoors, check the connector; compare with the phone |
| Fix OK but C/N0 < 30 dB-Hz everywhere | antenna placement, metal roof, long cable | Move the antenna; S4 / iono classes are capped below 30 dB-Hz anyway |
| `vcgencmd get_throttled` ≠ `0x0` (e.g. `0x50005`) | under-voltage now (bit 0) / since boot (bit 16) | Official PSU, short thick cable, no USB power hogs |
| `live.log`: `no bytes from /dev/serial0 for 30 s ... reopening` | L86 unplugged or browned out; a GPIO UART doesn't error, it just goes quiet | Check wiring and power; live mode recovers by itself when data returns |
| `live.log`: `host clock ... and data time ... disagree` | Pi has no RTC and no network time yet | Harmless: records follow NMEA time. Fix with network/NTP for sane log timestamps |
| `minutes.jsonl` not growing | service stopped, or crashed and restarting | `systemctl status ionwatch-live`, `journalctl -u ionwatch-live -n 50`, check disk with `df -h` |

## 5. Ten-minute acceptance test

Pass criteria in **bold**. Do it in a terminal (foreground) first, then repeat with the service.

```bash
cd ~/ionwatch && source .venv/bin/activate
rm -rf data/live && python -m ionwatch.cli live --l86 /dev/serial0 --baud 9600 --node-id haldwani-01 --out data/live/
```

In a second terminal:

```bash
watch -n 15 'wc -l data/live/minutes.jsonl; tail -n 3 data/live/live.log; python -c "import json;r=json.load(open(\"data/live/latest.json\"));print(r[\"minute_utc\"],r[\"sats_tracked\"],r[\"warnings\"])"'
```

| Time | Action | Expect |
|---|---|---|
| T+0 | start live mode | `live.log`: `serial /dev/serial0 open` |
| T+1 | – | **First provisional record** ~5 s after the first full minute; `latest.json` is at most ~1 min behind |
| T+1…T+5 | watch | **`minutes.jsonl` grows by 1 provisional line per minute**; from ~T+10 a **final** line per minute follows ~8.6 min after each minute (full 15-min baseline); `sats_tracked` > 0 once fixed; `data/live/raw/nmea_YYYY-MM-DD.nmea` growing |
| T+5 | **unplug the L86** (pull VCC, or the TX wire) for 2 min | process keeps running; after 30 s `live.log` shows `no bytes ... reopening`; records **keep coming, 1 per minute**, with `no_data` / `no_serial_data` warnings |
| T+7 | **replug** | NMEA within seconds (hot start); within ~1 min the provisional records show **`sats_tracked` > 0 again** and `serial_reconnected` in that minute's warnings |
| T+10 | Ctrl-C | clean exit, last lines `live stopped: N records (M final)` |

Then check:

```bash
python -m ionwatch.cli validate data/live/minutes.jsonl          # PASS: 0 invalid, 0 missing minutes, 0 duplicate finals
vcgencmd get_throttled                                            # throttled=0x0
python -m ionwatch.cli run --l86 data/live/raw/nmea_*.nmea --out data/processed/pi-acceptance --no-plots
```

**Pass:** about 9 provisional records for 10 minutes plus ~1 final (finals lag ~8.6 min; the newest minutes
are "pending" in `validate`, which is fine), `validate` says PASS, the recovered minutes have satellites,
the batch re-run of the raw NMEA finishes, and `throttled=0x0`. Leave it running 20 min if you want to see
finals for the unplug minutes.

Service check (2 min):

```bash
sudo systemctl enable --now ionwatch-live
sudo systemctl kill -s SIGKILL ionwatch-live; sleep 8; systemctl is-active ionwatch-live   # -> active (restarted)
```

## 6. If the hardware misbehaves: demo fallback

```bash
python -m ionwatch.cli replay --phone data/raw/gnss_log_2026_10_20_18_58_37.txt --speed 60 --out data/replay/demo/
```

This plays the 20 Oct night through the same live path and writes the same `minutes.jsonl` / `latest.json`.
With the dashboard: `bash deploy/demo.sh` (replay + dashboard on http://<pi-address>:8080).
