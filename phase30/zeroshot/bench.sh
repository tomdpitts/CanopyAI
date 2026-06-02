#!/bin/bash
# Evaluate the zero-shot shadow sweep on the TCD holdout subset.
# Each phase22-trained model (never saw TCD) -> foxtrot detector+SAM -> polygons,
# scored at mAP50 (primary) + Area-F1 (secondary).  One benchmark.py call.
#
# Launch:  caffeinate -i bash phase30/zeroshot/bench.sh
set -u
cd "/Users/tompitts/Library/Mobile Documents/com~apple~CloudDocs/dphil icloud/CanopyAI"
SUBSET=phase30/shadow_eval/subset_tiles.txt
OUT=phase30/shadow_eval/zeroshot
mkdir -p "$OUT"

MODELS=(); NAMES=()
for sw in 0 1 2 4 8; do
    ck=$(ls "checkpoints/zeroshot_shadow/zs_sw_${sw}"/deepforest-epoch=*.ckpt 2>/dev/null | head -1)
    if [ -z "$ck" ]; then echo "WARN: no ckpt for zs_sw_${sw} — skipping"; continue; fi
    MODELS+=("$ck"); NAMES+=("zs_sw_${sw}")
done
echo "Benchmarking ${#MODELS[@]} models on $(wc -l < "$SUBSET") tiles..."

PYTHONUNBUFFERED=1 ./venv310/bin/python phase30/benchmark.py \
    --models "${MODELS[@]}" --names "${NAMES[@]}" \
    --sam-model vit_l --sam-checkpoint sam_vit_l_0b3195.pth \
    --df-confidence 0.05 --max-dets 512 --pred-score-thresh 0.0 \
    --tiles-file "$SUBSET" \
    --output-root "$OUT" 2>&1 | tee "$OUT/bench.log"
