#!/usr/bin/env bash
# IonWatch demo: replay the 20 Oct 2026 night log at 60x through the live path and serve the
# dashboard on the same folder. One command; Ctrl-C stops both.
#
#   bash deploy/demo.sh [--speed 60] [--port 8080] [--log data/raw/gnss_log_2026_10_20_18_58_37.txt]
#
# Open http://<this machine's address>:8080 on a laptop or phone on the same Wi-Fi.
set -euo pipefail

SPEED=60
PORT=8080
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/data/raw/gnss_log_2026_10_20_18_58_37.txt"
OUT="$REPO/data/replay/demo"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --speed) SPEED="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

cd "$REPO"
# First interpreter that can import ionwatch (skips e.g. the Windows Store "python3" stub).
PY=""
for c in "$REPO/.venv/bin/python" "$REPO/.venv/Scripts/python.exe" python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import ionwatch" >/dev/null 2>&1; then PY="$c"; break; fi
done
[[ -n "$PY" ]] || { echo "no Python with IonWatch installed (run deploy/setup_pi.sh or pip install -e .)"; exit 1; }
# Fresh clone: data/ is not in git, so unpack the tracked copy of the 20 Oct log on first use.
PACKED="$REPO/demo/$(basename "$LOG").gz"
if [[ ! -f "$LOG" && -f "$PACKED" ]]; then
  mkdir -p "$(dirname "$LOG")"
  gunzip -c "$PACKED" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
  echo "Unpacked $(basename "$PACKED") -> $LOG"
fi
[[ -f "$LOG" ]] || { echo "log not found: $LOG"; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT"

"$PY" -m ionwatch.cli dashboard --data "$OUT" --port "$PORT" &
DASH=$!
"$PY" -m ionwatch.cli replay --phone "$LOG" --speed "$SPEED" --out "$OUT" > "$OUT/replay.stdout" 2>&1 &
REPLAY=$!
trap 'kill $DASH $REPLAY 2>/dev/null || true' EXIT INT TERM

ADDR="$( (hostname -I 2>/dev/null || true) | awk '{print $1}')"   # Linux; empty elsewhere
echo "Replaying $(basename "$LOG") at ${SPEED}x into $OUT"
echo "Dashboard: http://${ADDR:-localhost}:$PORT   (Ctrl-C to stop)"
wait $REPLAY || true
echo "Replay finished; dashboard still running. Ctrl-C to stop."
wait $DASH
