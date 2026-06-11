# Zero-shot shadow-loss-weight ablation

> **➡️ For the FINAL paper comparison, start at [`deepforest_custom/zeroshot_final/`](../../deepforest_custom/zeroshot_final/README.md).**
> That `zsfinal` run supersedes the mixed checkpoints described here: it retrains all
> cells through ONE trainer (`deepforest_custom`, WON-shrink on) on one manifest, removing
> the trainer/augmentation/WON-shrink confounds in this older sweep. This folder remains the
> record of the exploratory sweeps (`zs_*` seeds, `pr_sweep`, the area-F1 finding). Rationale:
> `paper/zeroshot_training_provenance.md`.

Tests whether `shadow_loss_weight` is a valid lever in the **zero-shot** regime:
train on the phase22 data (BRU/WON/NEON, non-TCD, dense per-box shadow vectors)
and evaluate **zero-shot on the TCD holdout** — the model never sees TCD in
training. (Fine-tuning on TCD washes the effect out; the win is in generalisation.)

Uses the native `deepforest_custom/` trainer (canopy-free, center-crop val that
handles the small phase22 tiles + the original `shadow_loss_reweight`), from the
weecology/NEON base. macOS needs `DF_NUM_WORKERS=0` (set by the drivers) to avoid
a DataLoader file-descriptor crash.

## Pipeline

```bash
# 1. data manifest (remap paths + drop sub-400px tiles -> *_filt.csv)
./venv310/bin/python phase30/zeroshot/prepare_phase22.py

# 2. train the sweep {0,1,2,4,8}  (0=ignore shadow, 1=neutral, >1=upweight)
caffeinate -i bash phase30/zeroshot/sweep.sh

# 3. evaluate zero-shot on TCD holdout (mAP50-polygon + Area-F1)
caffeinate -i bash phase30/zeroshot/bench.sh

# 4. area precision/recall curves (re-threshold the geojsons, no SAM re-run)
bash phase30/zeroshot/pr_sweep.sh

# 5. negative control: 3 shadow + 3 shadow-blind seeds @ weight 2, then benchmark
caffeinate -i bash phase30/zeroshot/control_run.sh
```

`train_one.py` trains a single weight (used by the drivers).
`SHADOW_BLIND_CONTROL=1` (control_run.sh) keeps the same number of upweighted
crowns per image but picks them at random — if that reproduces the gain, the
effect is generic hard-example upweighting, not shadow.

## Result (74-tile TCD subset, single seed)

| shadow weight | Area Precision | Area Recall | Area F1 |
|---|---|---|---|
| 1 (neutral) | 0.606 | 0.509 | 0.553 |
| 2 (optimum) | 0.590 | 0.551 | 0.570 |

Neutral→2: **recall +4 pts at flat precision (−1.6)**, consistent across 7
confidence thresholds — a genuine outward PR-curve shift. Optimum is sw_2;
sw_4/8 over-fire. Instance mAP50 trends the opposite way (coverage win, not a
detection-count win). Seeded shadow-vs-blind verdict pending.
