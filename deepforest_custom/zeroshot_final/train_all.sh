#!/usr/bin/env bash
# ============================================================================
# zsfinal — FINAL zero-shot shadow ablation, ONE trainer, ONE data manifest.
#
# All five cells use deepforest_custom/train_deepforest.py with the canonical
# family-A recipe (the phase22_B_L4 recipe): weecology base, 50 epochs,
# EarlyStopping(val mAP, patience 10), lr 1e-3, batch 16, crop-400 only (no
# flip, no photometric), WON bbox-shrink ON. Local MPS. Single variable across
# cells = the shadow loss weight (+ the blind toggle for the controls).
#
#   zsfinal_s1        weight 1  — neutral (no-shadow baseline; ×1 = no reweight)
#   zsfinal_s2        weight 2  — shadow
#   zsfinal_s4        weight 4  — shadow (matches phase22_B_L4's weight)
#   zsfinal_blind_s2  weight 2  + SHADOW_BLIND_CONTROL=1 — specificity control
#   zsfinal_blind_s4  weight 4  + SHADOW_BLIND_CONTROL=1 — specificity control
#
# Resumable: each cell's checkpoint dir is created under checkpoints/zsfinal/;
# the trainer auto-resumes if a checkpoint already exists there. Comment out
# cells you've finished. Run under caffeinate (multi-hour MPS).
#
#   caffeinate -i bash deepforest_custom/zeroshot_final/train_all.sh
# ============================================================================
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

PY=./venv310/bin/python
TRAIN=deepforest_custom/zeroshot_final/train.csv
VAL=deepforest_custom/zeroshot_final/val.csv
OUT=checkpoints/zsfinal
COMMON="--epochs 50 --patience 10 --lr 0.001 --batch_size 16 --accelerator mps --shadow-loss-reweight"

# DF_NUM_WORKERS=0 — macOS MPS DataLoader file-descriptor fix (see trainer header).
export DF_NUM_WORKERS=0

run () {  # run <run_name> <weight> <blind:0|1>
  local name="$1" weight="$2" blind="$3"
  echo -e "\n=== [$(date '+%Y-%m-%d %H:%M:%S %Z')] TRAIN ${name}  weight=${weight} blind=${blind} ==="
  SHADOW_BLIND_CONTROL="${blind}" $PY deepforest_custom/train_deepforest.py \
    --train_csv "$TRAIN" --val_csv "$VAL" \
    --run_name "$name" --output_dir "$OUT" \
    $COMMON --shadow-loss-weight "$weight" \
    2>&1 | tee -a "${OUT}/${name}.train.log"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S %Z')] DONE ${name} rc=${PIPESTATUS[0]} ==="
}

mkdir -p "$OUT"
run zsfinal_s1       1 0
run zsfinal_s2       2 0
run zsfinal_s4       4 0
run zsfinal_blind_s2 2 1
run zsfinal_blind_s4 4 1

echo -e "\nAll cells done. Checkpoints in ${OUT}/zsfinal_*/. Next: eval (see README §Eval)."
