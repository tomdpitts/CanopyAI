#!/bin/bash
# Full ±shadow ablation: train+eval each arm. Auto-resumes from runs/<name>/last.pt.
# Strongly prefer a CUDA GPU; on MPS wrap with caffeinate -i and expect it to be slow.
#   caffeinate -i bash run_ablation.sh
set -u
cd "$(dirname "$0")"
PY=${PY:-/tmp/segf_venv/bin/python}
LOG=/tmp/shadow_semseg; mkdir -p "$LOG"

run_arm() {  # name  extra-flags
  echo "[$(date '+%H:%M:%S')] === $1 train ===" | tee -a "$LOG/master.log"
  "$PY" train.py --name "$1" $2 >> "$LOG/$1.train.log" 2>&1
  echo "[$(date '+%H:%M:%S')] === $1 eval ===" | tee -a "$LOG/master.log"
  "$PY" eval.py --name "$1" --weights best.pt >> "$LOG/$1.eval.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $1 done" | tee -a "$LOG/master.log"
}

run_arm semseg_shadow ""
run_arm semseg_noshadow "--no-shadow"
echo "=== ABLATION COMPLETE — compare runs/semseg_{shadow,noshadow}/eval_best.json ===" | tee -a "$LOG/master.log"
