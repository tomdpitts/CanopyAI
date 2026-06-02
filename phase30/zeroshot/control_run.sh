#!/bin/bash
# Negative control + seeds: 2 more REAL-shadow seeds (sw_2) and 3 SHADOW-BLIND
# seeds (weight 2, random equal-count crowns).  Then benchmark all 5 on TCD.
# Combined with existing zs_sw_2 => 3 shadow vs 3 blind, with error bars.
set -u
cd "/Users/tompitts/Library/Mobile Documents/com~apple~CloudDocs/dphil icloud/CanopyAI"
ulimit -n 8192 2>/dev/null || true
export DF_NUM_WORKERS=0
RUNS=("zs_sw_2b:0" "zs_sw_2c:0" "zs_blind_2a:1" "zs_blind_2b:1" "zs_blind_2c:1")
for spec in "${RUNS[@]}"; do
    run="${spec%%:*}"; blind="${spec##*:}"
    rdir="checkpoints/zeroshot_shadow/$run"
    [ -f "$rdir/.done" ] && { echo "SKIP $run"; continue; }
    mkdir -p "$rdir"
    echo "=== TRAIN $run (blind=$blind) $(date) ==="
    SHADOW_BLIND_CONTROL=$blind PYTHONUNBUFFERED=1 ./venv310/bin/python phase30/zeroshot/train_one.py \
        --shadow-weight 2 --run-name "$run" --epochs 40 2>&1 | tee "$rdir/train.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] && touch "$rdir/.done" || echo "FAILED $run"
done
echo "=== BENCHMARK 5 control models on TCD $(date) ==="
MODELS=(); NAMES=()
for run in zs_sw_2b zs_sw_2c zs_blind_2a zs_blind_2b zs_blind_2c; do
    ck=$(ls "checkpoints/zeroshot_shadow/$run"/deepforest-epoch=*.ckpt 2>/dev/null | head -1)
    [ -n "$ck" ] && { MODELS+=("$ck"); NAMES+=("$run"); }
done
PYTHONUNBUFFERED=1 ./venv310/bin/python phase30/benchmark.py \
    --models "${MODELS[@]}" --names "${NAMES[@]}" \
    --sam-model vit_l --sam-checkpoint sam_vit_l_0b3195.pth \
    --df-confidence 0.05 --max-dets 512 --pred-score-thresh 0.0 --skip-existing \
    --tiles-file phase30/shadow_eval/subset_tiles.txt \
    --output-root phase30/shadow_eval/zeroshot 2>&1 | tee phase30/shadow_eval/zeroshot/control_bench.log
echo "=== CONTROL RUN COMPLETE $(date) ==="
