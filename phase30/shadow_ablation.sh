#!/bin/bash
# Shadow-loss-weight ablation: train DeepForest at each shadow_weight,
# record the best val mAP.  Resumable — skips any value with a .done sentinel.
#
# Fixed 10-epoch budget, batch 16, no early stopping → every model gets
# identical training for a fair cross-model comparison.  Winning weight is then
# re-evaluated on the full 439-tile holdout with the complete foxtrot pipeline.
#
# Full 677-image val (phase30_tcd_val.csv) for a low-noise ranking — val is only
# ~13 min/cycle, cheap relative to training.
#
# Keep persistent_workers=True (the train.py default) — do NOT pass
# --no-persistent-workers (it deadlocks on macOS spawn and fixes nothing).  The
# multi-hour MPS memory blow-up that used to kill canopy runs is fixed in
# models.py _patch_retinanet_head_loss (canopy label-assignment moved to CPU +
# dense fixed-shape focal loss), so footprint now plateaus ~25 GB.  Launch under
# caffeinate -i + PYTHONUNBUFFERED=1 (otherwise the Lightning progress bar
# block-buffers through the tee pipe and the log looks frozen).
#
# Run the whole grid:   bash phase30/shadow_ablation.sh
# Run a single weight:  SHADOW_WEIGHTS=0 bash phase30/shadow_ablation.sh
set -u
cd "/Users/tompitts/Library/Mobile Documents/com~apple~CloudDocs/dphil icloud/CanopyAI"
PY=./venv310/bin/python

# SHADOW_WEIGHTS=N runs a single weight (resumes/extends the grid); default = all.
WEIGHTS=(${SHADOW_WEIGHTS:-0 1 2 4 8})
EPOCHS=10
BATCH=16
OUT_ROOT=checkpoints/shadow_ablation
mkdir -p "$OUT_ROOT"

for sw in "${WEIGHTS[@]}"; do
    run_name="sw_${sw}"
    run_dir="${OUT_ROOT}/${run_name}"
    # Resume: skip only if the run completed cleanly (.done sentinel).  A bare
    # checkpoint is NOT enough — a crashed run can leave a non-final ckpt, which
    # the old logic would have silently treated as "done" and corrupted the grid.
    if [ -f "${run_dir}/.done" ]; then
        echo "=== SKIP shadow_weight=${sw} (.done sentinel present) ==="
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
    # PIPESTATUS[0] is train.py's exit code (not tee's).  Only mark done on
    # success — a failed run stays un-done so a rerun retries it, and the
    # monitor sees the explicit FAILED line instead of a silent hang.
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        touch "${run_dir}/.done"
        echo "  ✅ shadow_weight=${sw} complete"
    else
        echo "  ❌ FAILED shadow_weight=${sw} (exit ${PIPESTATUS[0]}) — leaving un-done for retry"
    fi
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
