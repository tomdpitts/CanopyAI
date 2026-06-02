#!/bin/bash
set -u
cd "/Users/tompitts/Library/Mobile Documents/com~apple~CloudDocs/dphil icloud/CanopyAI"
ROOT=phase30/shadow_eval/zeroshot
RES=$ROOT/pr_curve.tsv
printf "thresh\tweight\tprec\trecall\tf1\n" > "$RES"
MODELS=(); NAMES=()
for sw in 0 1 2 4 8; do
  MODELS+=("$(ls checkpoints/zeroshot_shadow/zs_sw_${sw}/deepforest-epoch=*.ckpt|head -1)")
  NAMES+=("zs_sw_${sw}")
done
for X in 0.05 0.10 0.15 0.20 0.25 0.30 0.35; do
  ./venv310/bin/python phase30/benchmark.py --skip-inference \
     --models "${MODELS[@]}" --names "${NAMES[@]}" \
     --max-dets 512 --pred-score-thresh "$X" \
     --tiles-file phase30/shadow_eval/subset_tiles.txt \
     --output-root "$ROOT" > /dev/null 2>&1
  ./venv310/bin/python - "$X" "$RES" <<'PY'
import csv,sys
X,RES=sys.argv[1],sys.argv[2]
with open(RES,"a") as o:
  for sw in [0,1,2,4,8]:
    TP=FP=FN=0
    for r in csv.DictReader(open(f"phase30/shadow_eval/zeroshot/zs_sw_{sw}_holdout_tiles.csv")):
      TP+=int(r["tp"]);FP+=int(r["fp"]);FN+=int(r["fn"])
    P=TP/(TP+FP) if TP+FP else 0; R=TP/(TP+FN) if TP+FN else 0
    F=2*P*R/(P+R) if P+R else 0
    o.write(f"{X}\t{sw}\t{P:.4f}\t{R:.4f}\t{F:.4f}\n")
PY
  echo "thresh $X done"
done
echo "PR SWEEP COMPLETE"
