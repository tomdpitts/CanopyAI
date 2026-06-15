#!/bin/bash
# ============================================================================
# Overnight autonomous runner (2026-06-05 night).  Two jobs, in priority order:
#   1. BLIND EVAL  (priority) — shadow-specificity control on the 439 holdout,
#      reranker ON, identical settings to the published ablation table.
#      Models: ablation_shadow_w4_local (positive), ablation_blind_w4 +
#      ablation_blind_w2 (random-reweight negative controls).
#   2. RERANKER-OFF (only if blind finishes) — core models re-scored with NO
#      reranker, to show the shadow finding survives without the reranker.
#
# Design for unattended reliability:
#   • each model -> its OWN --output-root  => no shared-summary race in 2-lane
#   • --skip-existing on every model        => crash-resumable, lossless
#   • map50 read from each clean per-model benchmark_holdout_summary.json
#     (verified == the oracle tree-only AP50)
#   • 2 lanes max (memory: 2 optimal, 3 oversubscribes the 16-core CPU)
# Launch:  caffeinate -i bash phase30/zeroshot/overnight_blind_rroff.sh
# ============================================================================
set -u
cd "/Users/tompitts/Library/Mobile Documents/com~apple~CloudDocs/dphil icloud/CanopyAI"
PY=./venv310/bin/python
SHADOW=solar/shadow_regression/output/shadow_model_combined_best.pth
RERANK=phase30/cnn_reranker_ens3.pt
HOLD=data/tcd/images/data/tcd/val
LOGDIR=/tmp/blind_overnight
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/MASTER.log"
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$MASTER"; }

# bench <name> <ckpt> <output-root> <extra-args...>
bench(){
  local name="$1" ckpt="$2" root="$3"; shift 3
  "$PY" phase30/benchmark.py \
    --models "$ckpt" --names "$name" \
    --holdout-dir "$HOLD" \
    --shadow-model "$SHADOW" \
    --df-confidence 0.05 \
    --sam-model vit_h --sam-checkpoint sam_vit_h_4b8939.pth \
    --max-dets 512 --pred-score-thresh 0.0 --skip-existing \
    --output-root "$root" "$@"
}

say "================= OVERNIGHT RUN START ================="

# -------------------- JOB 1: BLIND EVAL (reranker ON) --------------------
say "=== JOB 1: BLIND EVAL (reranker ON) ==="

SHW=checkpoints/ablation_shadow_w4_local/deepforest-epoch=23-map=0.47.ckpt
BW4=checkpoints/ablation_blind_w4/deepforest-epoch=23-map=0.48.ckpt
BW2=checkpoints/ablation_blind_w2/deepforest-epoch=23-map=0.48.ckpt

# Lane A + Lane B concurrently
say "launch lane A: ablation_shadow_w4_local"
bench ablation_shadow_w4_local "$SHW" benchmark_results_blind_shadow_w4_local \
      --reranker-checkpoint "$RERANK" > "$LOGDIR/shadow_w4_local.log" 2>&1 &
PA=$!
say "launch lane B: ablation_blind_w4"
bench ablation_blind_w4 "$BW4" benchmark_results_blind_blind_w4 \
      --reranker-checkpoint "$RERANK" > "$LOGDIR/blind_w4.log" 2>&1 &
PB=$!
wait $PA; say "lane A done (ablation_shadow_w4_local) rc=$?"
wait $PB; say "lane B done (ablation_blind_w4) rc=$?"

# Third model on a freed lane
say "launch ablation_blind_w2"
bench ablation_blind_w2 "$BW2" benchmark_results_blind_blind_w2 \
      --reranker-checkpoint "$RERANK" > "$LOGDIR/blind_w2.log" 2>&1 &
PC=$!
wait $PC; say "ablation_blind_w2 done rc=$?"
say "=== JOB 1 BLIND EVAL COMPLETE ==="

# Quick map50 readout
for n in shadow_w4_local blind_w4 blind_w2; do
  f="benchmark_results_blind_${n}/benchmark_holdout_summary.json"
  v=$("$PY" -c "import json,sys;d=json.load(open('$f'));r=list(d['results'].values())[0] if 'results' in d else list(d.values())[0];print(round(r.get('map50',float('nan')),4))" 2>/dev/null)
  say "  BLIND map50  $n = $v"
done

# -------------------- JOB 2: RERANKER-OFF (core models) --------------------
say "=== JOB 2: RERANKER-OFF (core models, resumable) ==="
RM=( "phase21_baseline.pth" \
     "phase22_B_L4.pth" \
     "checkpoints/ablation_tcd_s0/deepforest-epoch=71-map=0.45.ckpt" \
     "checkpoints/ablation_tcd_s2/deepforest-epoch=89-map=0.40.ckpt" \
     "checkpoints/ablation_tcd_s4/deepforest-epoch=92-map=0.41.ckpt" )
RN=( phase21_baseline phase22_B_L4 ablation_tcd_s0 ablation_tcd_s2 ablation_tcd_s4 )

i=0
while [ $i -lt ${#RM[@]} ]; do
  n1="${RN[$i]}"; m1="${RM[$i]}"
  say "launch noRR lane A: $n1"
  bench "$n1" "$m1" "benchmark_results_noRR_${n1}" > "$LOGDIR/noRR_${n1}.log" 2>&1 &
  A=$!
  j=$((i+1))
  if [ $j -lt ${#RM[@]} ]; then
    n2="${RN[$j]}"; m2="${RM[$j]}"
    say "launch noRR lane B: $n2"
    bench "$n2" "$m2" "benchmark_results_noRR_${n2}" > "$LOGDIR/noRR_${n2}.log" 2>&1 &
    B=$!
    wait $A; say "noRR $n1 done rc=$?"
    wait $B; say "noRR $n2 done rc=$?"
  else
    wait $A; say "noRR $n1 done rc=$?"
  fi
  i=$((i+2))
done
say "=== JOB 2 RERANKER-OFF COMPLETE ==="

for n in "${RN[@]}"; do
  f="benchmark_results_noRR_${n}/benchmark_holdout_summary.json"
  v=$("$PY" -c "import json,sys;d=json.load(open('$f'));r=list(d['results'].values())[0] if 'results' in d else list(d.values())[0];print(round(r.get('map50',float('nan')),4))" 2>/dev/null)
  say "  noRR map50  $n = $v"
done

say "================= OVERNIGHT RUN COMPLETE ================="
