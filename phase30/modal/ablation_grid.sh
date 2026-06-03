#!/usr/bin/env bash
# phase30/modal/ablation_grid.sh — launch the 4 ablation trains in dependency order.
#
# Shadow weight held constant across both stages (4->4, 2->2, off->off).
# Stage-1 s0/s4 already exist on the volume (phase21_baseline, phase22_B_L4);
# only ablation_pre_s2 is a new stage-1 train. ablation_tcd_s2 needs pre_s2 first.
#
# This launches with --detach (non-blocking). Watch progress with `modal app list`
# / the Modal dashboard. Run the smoke test (README Step 4) BEFORE this.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CKPT=/checkpoints
S1="--batch-size 16 --lr 0.001   --epochs 50  --patience 10"   # stage-1 (phase22 recipe)
S2="--batch-size 16 --lr 0.00001 --epochs 500 --patience 5"    # stage-2 (kunqi5 history)

echo "### stage-1: shadow-2 pretrain (the only new stage-1 cell) ###"
modal run --detach phase30/modal/train.py --dataset bwn --shadow 2 \
  --run-name ablation_pre_s2 $S1

echo "### stage-2: fine-tunes that DON'T depend on pre_s2 — launch now ###"
modal run --detach phase30/modal/train.py --dataset tcd --shadow 1 \
  --run-name ablation_tcd_s0 --base-checkpoint $CKPT/phase21_baseline/deepforest_final.pth $S2
modal run --detach phase30/modal/train.py --dataset tcd --shadow 4 \
  --run-name ablation_tcd_s4 --base-checkpoint $CKPT/phase22_B_L4/deepforest_final.pth $S2

cat <<'EOF'

### stage-2 s2 — DEPENDS on ablation_pre_s2; launch only after it has finished: ###
modal run --detach phase30/modal/train.py --dataset tcd --shadow 2 \
  --run-name ablation_tcd_s2 \
  --base-checkpoint /checkpoints/ablation_pre_s2/deepforest_final.pth \
  --batch-size 16 --lr 0.00001 --epochs 500 --patience 5
EOF
