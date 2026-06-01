#!/bin/bash
# Overnight orchestrator for the shadow-weight ablation grid.
#
# Waits for the in-flight sw_0 run to finish (its .done sentinel), then runs the
# bare grid driver, which skips any weight that already has .done (sw_0) and
# trains the remaining weights {1,2,4,8}.  A background footprint watchdog logs
# per-weight memory and force-stops a runaway train proc above 55 GB — far above
# the proven ~25 GB plateau (the MPS graph-cache leak is fixed; see models.py
# _patch_retinanet_head_loss) — so the 64 GB machine is protected while unattended.
#
# Launch under: caffeinate -i bash phase30/overnight_grid.sh
set -u
cd "/Users/tompitts/Library/Mobile Documents/com~apple~CloudDocs/dphil icloud/CanopyAI"
ROOT=checkpoints/shadow_ablation
LOG="$ROOT/overnight.log"
WLOG="$ROOT/mem_watchdog.tsv"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] orchestrator start (pid $$)" >> "$LOG"

# --- background footprint watchdog over the whole grid ---
(
  printf "ts\tweight\tfootprint_GB\tswap_MB\n" > "$WLOG"
  while true; do
    pid=$(pgrep -f 'phase30/train.py' | head -1)
    if [ -n "$pid" ]; then
      fp=$(vmmap -summary "$pid" 2>/dev/null | awk '/Physical footprint:/{print $3; exit}')
      sw=$(sysctl -n vm.swapusage | awk '{print $7}')
      wt=$(ps -o command= -p "$pid" 2>/dev/null | sed -n 's/.*--shadow-loss-weight \([0-9][0-9]*\).*/\1/p')
      printf "%s\tsw_%s\t%s\t%s\n" "$(ts)" "${wt:-?}" "${fp:-?}" "$sw" >> "$WLOG"
      num=$(printf '%s' "$fp" | sed 's/G.*//')
      if [ -n "$num" ] && awk "BEGIN{exit !($num>55)}" 2>/dev/null; then
        echo "[$(ts)] WATCHDOG: footprint ${fp} > 55G on pid $pid (sw_${wt}) — killing to protect machine" >> "$LOG"
        kill -9 "$pid" 2>/dev/null
        pkill -9 -f 'multiprocessing.spawn' 2>/dev/null
      fi
    fi
    sleep 120
  done
) &
WATCHDOG=$!
trap 'kill "$WATCHDOG" 2>/dev/null' EXIT

# --- wait for the in-flight sw_0 run to finish ---
echo "[$(ts)] waiting for sw_0/.done ..." >> "$LOG"
while [ ! -f "$ROOT/sw_0/.done" ]; do
  if ! pgrep -f 'phase30/train.py' >/dev/null 2>&1; then
    sleep 5
    [ -f "$ROOT/sw_0/.done" ] && break
    echo "[$(ts)] sw_0 train proc gone without .done — grid will retry sw_0" >> "$LOG"
    break
  fi
  sleep 60
done
echo "[$(ts)] sw_0 phase complete (.done=$( [ -f "$ROOT/sw_0/.done" ] && echo yes || echo no ))" >> "$LOG"

# --- run the bare grid (skips weights that already have .done) ---
echo "[$(ts)] launching grid driver for remaining weights" >> "$LOG"
PYTHONUNBUFFERED=1 bash phase30/shadow_ablation.sh >> "$LOG" 2>&1
echo "[$(ts)] grid driver exited (code $?)" >> "$LOG"
echo "[$(ts)] orchestrator done" >> "$LOG"
