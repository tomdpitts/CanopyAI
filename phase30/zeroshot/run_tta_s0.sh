#!/bin/bash
# ============================================================================
# Full detector-stage multi-scale TTA eval for the best model, ablation_tcd_s0.
#   • foxtrot --df_tta (Restor downscale factors 0.375..1.0), reranker ON, SAM-H — identical
#     to the published-table settings except for the added TTA.
#   • 2-lane tile-split (~92s/tile in smoke → ~6h for 439 across 2 lanes).
#   • Outputs as model "s0_ttaR" inside benchmark_results_holdout so the existing
#     oracle (canopy_aggregation_test.py, PRED_DIR=benchmark_results_holdout/<m>)
#     works unmodified.
#   • Inference scored per-half by each lane (ignored), then ONE final full-439
#     --skip-inference pass writes the correct tree-only map50; then the oracle
#     writes the 2-cat number.  --skip-existing throughout => crash-resumable.
# Launch:  caffeinate -i bash phase30/zeroshot/run_tta_s0.sh
# Compare: tree-only map50 vs no-TTA 0.535 ; 2-cat oracle vs no-TTA 0.447.
# ============================================================================
set -u
cd "/Users/tompitts/Library/Mobile Documents/com~apple~CloudDocs/dphil icloud/CanopyAI"
PY=./venv310/bin/python
CKPT="checkpoints/ablation_tcd_s0/deepforest-epoch=71-map=0.45.ckpt"
NAME=s0_ttaR    # Restor-matched DOWNSCALE-only TTA (apples-to-apples)
ROOT=benchmark_results_holdout
HOLD=data/tcd/images/data/tcd/val
SHADOW=solar/shadow_regression/output/shadow_model_combined_best.pth
RERANK=phase30/cnn_reranker_ens3.pt
SCALES="0.375,0.391,0.439,0.5,0.732,1.0"   # Restor MIN_SIZES/2048 (downscale-only)
LOGDIR=/tmp/tta_s0; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/MASTER.log"
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$MASTER"; }

# Fresh output dir (scale set changed -> do NOT reuse old-scale smoke geojsons).
mkdir -p "$ROOT/$NAME"

# Parity tile-split (balances tree density better than a contiguous halve).
ls "$HOLD"/*.tif | xargs -n1 basename | sed 's/\.tif$//' | sort > "$LOGDIR/all.txt"
awk 'NR%2==1' "$LOGDIR/all.txt" > "$LOGDIR/halfA.txt"
awk 'NR%2==0' "$LOGDIR/all.txt" > "$LOGDIR/halfB.txt"
say "tiles: $(wc -l < $LOGDIR/all.txt) total, A=$(wc -l < $LOGDIR/halfA.txt) B=$(wc -l < $LOGDIR/halfB.txt)"

bench_infer(){  # tiles-file logfile
  "$PY" phase30/benchmark.py \
    --models "$CKPT" --names "$NAME" \
    --holdout-dir "$HOLD" --shadow-model "$SHADOW" \
    --df-confidence 0.05 \
    --sam-model vit_h --sam-checkpoint sam_vit_h_4b8939.pth \
    --reranker-checkpoint "$RERANK" \
    --df-tta --df-tta-scales "$SCALES" \
    --max-dets 512 --pred-score-thresh 0.0 --skip-existing \
    --tiles-file "$1" --output-root "$ROOT" > "$2" 2>&1
}

say "=== TTA inference (2 lanes) START ==="
bench_infer "$LOGDIR/halfA.txt" "$LOGDIR/laneA.log" & PA=$!
bench_infer "$LOGDIR/halfB.txt" "$LOGDIR/laneB.log" & PB=$!
wait $PA; say "lane A done rc=$?"
wait $PB; say "lane B done rc=$?"

say "=== final full-439 scoring (--skip-inference) ==="
"$PY" phase30/benchmark.py \
  --models "$CKPT" --names "$NAME" \
  --holdout-dir "$HOLD" --shadow-model "$SHADOW" \
  --max-dets 512 --pred-score-thresh 0.0 \
  --skip-inference --output-root "$ROOT" > "$LOGDIR/score.log" 2>&1
say "scoring done rc=$?"
cp -f "$ROOT/benchmark_holdout_summary.json" "$ROOT/benchmark_holdout_summary_${NAME}.json" 2>/dev/null || true
tree=$("$PY" -c "import json;d=json.load(open('$ROOT/benchmark_holdout_summary.json'));r=list(d['results'].values())[0] if 'results' in d else list(d.values())[0];print(round(r.get('map50',float('nan')),4))" 2>/dev/null)
say "  TTA tree-only mAP50 = $tree   (no-TTA s0 = 0.535)"

say "=== 2-cat oracle ==="
AGG_MODEL="$NAME" AGG_TREE_REF="0.535" "$PY" phase30/canopy_aggregation_test.py > "$LOGDIR/oracle.log" 2>&1
twocat=$("$PY" -c "import json;d=json.load(open('$ROOT/canopy_aggregation_oracle_${NAME}.json'));print(round(d['thresholds']['T=0.5']['twocat_mean_ap50'],4))" 2>/dev/null)
say "  TTA 2-cat oracle (T=0.5) = $twocat   (no-TTA s0 = 0.447 ; Restor 0.432)"
say "=== TTA s0 RUN COMPLETE ==="
