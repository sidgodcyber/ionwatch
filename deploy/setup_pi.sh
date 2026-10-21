#!/usr/bin/env bash
# IonWatch: one-time Raspberry Pi setup for live L86 logging.
#
#   cd ~/ionwatch && sudo bash deploy/setup_pi.sh [--user pi] [--disable-bt] [--enable-service]
#
# What it does (idempotent; safe to re-run):
#   1. enables the UART and disables the serial login console (raspi-config, plus config.txt /
#      cmdline.txt edits as a fallback);
#   2. optionally (--disable-bt) moves Bluetooth off the PL011 UART so /dev/serial0 is the stable
#      full UART instead of the mini-UART;
#   3. installs system packages and a Python venv in <repo>/.venv with IonWatch and its deps;
#   4. adds the user to the dialout group (serial port access);
#   5. installs deploy/ionwatch-live.service (and enables + starts it with --enable-service).
# A reboot is needed after the first run (UART / console / group changes).
set -euo pipefail

RUN_USER="${SUDO_USER:-pi}"
DISABLE_BT=0
ENABLE_SERVICE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) RUN_USER="$2"; shift 2 ;;
    --disable-bt) DISABLE_BT=1; shift ;;
    --enable-service) ENABLE_SERVICE=1; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
say() { printf '\n== %s\n' "$*"; }

[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
if ! grep -qi raspberry /proc/device-tree/model 2>/dev/null; then
  echo "WARNING: this does not look like a Raspberry Pi ($(cat /proc/device-tree/model 2>/dev/null || uname -a))"
fi
ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" ]]; then
  echo "WARNING: $ARCH detected. Use 64-bit Raspberry Pi OS: pyarrow/pandas wheels for 32-bit ARM are"
  echo "         missing or very slow to build."
fi

# Boot config lives in /boot/firmware on Bookworm, /boot on older releases.
BOOT=/boot/firmware; [[ -f $BOOT/config.txt ]] || BOOT=/boot
CONFIG="$BOOT/config.txt"; CMDLINE="$BOOT/cmdline.txt"

say "1. UART on, serial console off"
if command -v raspi-config >/dev/null; then
  raspi-config nonint do_serial_cons 1 || true   # 1 = disable login shell on serial
  raspi-config nonint do_serial_hw 0 || true     # 0 = enable serial port hardware
fi
grep -q '^enable_uart=1' "$CONFIG" || echo 'enable_uart=1' >> "$CONFIG"
cp -n "$CMDLINE" "$CMDLINE.ionwatch.bak" || true
sed -i -E 's/console=(serial0|ttyAMA0|ttyS0),[0-9]+ ?//g' "$CMDLINE"
for tty in ttyS0 ttyAMA0 serial0; do
  systemctl disable --now "serial-getty@$tty.service" 2>/dev/null || true
done

if [[ $DISABLE_BT -eq 1 ]]; then
  say "2. Bluetooth off the PL011 UART (dtoverlay=disable-bt)"
  grep -q '^dtoverlay=disable-bt' "$CONFIG" || echo 'dtoverlay=disable-bt' >> "$CONFIG"
  systemctl disable hciuart 2>/dev/null || true
else
  say "2. keeping Bluetooth (serial0 may be the mini-UART; fine at 9600 baud)"
fi

say "3. packages + venv"
apt-get update
apt-get install -y python3-venv python3-pip python3-dev git tmux
sudo -u "$RUN_USER" python3 -m venv "$REPO/.venv"
sudo -u "$RUN_USER" "$REPO/.venv/bin/pip" install --upgrade pip
sudo -u "$RUN_USER" "$REPO/.venv/bin/pip" install -e "$REPO[test]"
sudo -u "$RUN_USER" mkdir -p "$REPO/data/live"

say "4. $RUN_USER -> dialout"
usermod -aG dialout "$RUN_USER"

say "5. systemd unit"
sed -e "s#@USER@#$RUN_USER#g" -e "s#@REPO@#$REPO#g" "$REPO/deploy/ionwatch-live.service" \
  > /etc/systemd/system/ionwatch-live.service
systemctl daemon-reload
if [[ $ENABLE_SERVICE -eq 1 ]]; then
  systemctl enable --now ionwatch-live.service
  echo "service enabled and started: journalctl -u ionwatch-live -f"
else
  echo "installed, not enabled. Start with: sudo systemctl enable --now ionwatch-live"
fi

say "done"
echo "Reboot now (sudo reboot) so the UART, console and dialout changes take effect."
echo "Then: ls -l /dev/serial0 ; stty -F /dev/serial0 9600 raw -echo ; timeout 5 cat /dev/serial0"
