#!/bin/bash
# Zero-shot shadow ablation sweep — train shadow_loss_weight {0,1,2,4,8} on the
# phase22 (BRU/WON/NEON, non-TCD) data with the native deepforest_custom trainer,
# from the weecology base, canopy-free.  Models are evaluated zero-shot on the TCD
# holdout afterwards (phase30/zeroshot/bench.sh).  Resumable via .done sentinels.
#
# Launch:  caffeinate -i bash phase30/zeroshot/sweep.sh
set -u
cd "/Users/tompitts/Library/Mobile Documents/com~apple~CloudDocs/dphil icloud/CanopyAI"
ulimit -n 8192 2>/dev/null || true   # headroom for DataLoader worker FDs on macOS
export DF_NUM_WORKERS=0               # load in-process on macOS (avoids worker Abort trap)
WEIGHTS=(${ZS_WEIGHTS:-0 1 2 4 8})
EPOCHS=${ZS_EPOCHS:-40}
OUT=checkpoints/zeroshot_shadow
mkdir -p "$OUT"
for sw in "${WEIGHTS[@]}"; do
    run="zs_sw_${sw}"
    rdir="$OUT/$run"
    if [ -f "$rdir/.done" ]; then echo "=== SKIP $run (.done) ==="; continue; fi
    mkdir -p "$rdir"
    echo; echo "================================================================"
    echo "  TRAIN $run  shadow_loss_weight=$sw  epochs=$EPOCHS   $(date)"
    echo "================================================================"
    PYTHONUNBUFFERED=1 ./venv310/bin/python phase30/zeroshot/train_one.py \
        --shadow-weight "$sw" --run-name "$run" --epochs "$EPOCHS" \
        2>&1 | tee "$rdir/train.log"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        touch "$rdir/.done"; echo "  ✅ $run complete"
    else
        echo "  ❌ FAILED $run (left un-done for retry)"
    fi
done
echo; echo "=== ZERO-SHOT SWEEP COMPLETE $(date) ==="
for sw in "${WEIGHTS[@]}"; do
    last=$(ls "$OUT/zs_sw_${sw}"/*.ckpt 2>/dev/null | head -1)
    echo "  shadow=${sw} -> ${last:-(no ckpt)}"
done
