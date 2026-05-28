#!/bin/bash
# Shadow-loss-weight ablation: train DeepForest at each shadow_weight,
# record the best val mAP.  Resumable — skips any value whose run dir
# already has a best checkpoint.
#
# Fixed 10-epoch budget, batch 16, no early stopping → every model gets
# identical training for a fair cross-model comparison.
set -u
cd "/Users/tompitts/Library/Mobile Documents/com~apple~CloudDocs/dphil icloud/CanopyAI"
PY=./venv310/bin/python

WEIGHTS=(0 0.5 1 2 4 8 16)
EPOCHS=10
BATCH=16
OUT_ROOT=checkpoints/shadow_ablation
mkdir -p "$OUT_ROOT"

for sw in "${WEIGHTS[@]}"; do
    run_name="sw_${sw}"
    run_dir="${OUT_ROOT}/${run_name}"
    # Resume: skip if a best-map checkpoint already exists
    if ls "${run_dir}"/deepforest-epoch=*.ckpt >/dev/null 2>&1; then
        echo "=== SKIP shadow_weight=${sw} (checkpoint exists) ==="
        continue
    fi
    echo
    echo "================================================================"
    echo "  TRAIN shadow_weight=${sw}   (epochs=${EPOCHS}, batch=${BATCH})"
    echo "  $(date)"
    echo "================================================================"
    mkdir -p "$run_dir"
    $PY phase30/train.py \
        --train-csv phase30/phase30_tcd_train.csv \
        --val-csv   phase30/phase30_tcd_val.csv \
        --checkpoint phase22_B_L4.pth \
        --canopy-polygons phase30/phase30_tcd_canopy_polygons.json \
        --canopy-loss-scale 1.0 \
        --shadow-loss-weight "${sw}" \
        --output-dir "$OUT_ROOT" \
        --run-name "$run_name" \
        --batch-size "$BATCH" --lr 0.0001 --epochs "$EPOCHS" --patience 99 \
        2>&1 | tee "${run_dir}/train.log"
done

echo
echo "================================================================"
echo "  ABLATION SUMMARY — best val mAP per shadow_weight"
echo "================================================================"
for sw in "${WEIGHTS[@]}"; do
    ckpt=$(ls "${OUT_ROOT}/sw_${sw}"/deepforest-epoch=*.ckpt 2>/dev/null | head -1)
    if [ -n "$ckpt" ]; then
        # filename: deepforest-epoch=NN-map=X.XX.ckpt
        base=$(basename "$ckpt")
        echo "  shadow_weight=${sw}  →  ${base}"
    else
        echo "  shadow_weight=${sw}  →  (no checkpoint)"
    fi
done
echo "DONE."
