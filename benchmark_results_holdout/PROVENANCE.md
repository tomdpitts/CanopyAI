# benchmark_results_holdout — PROVENANCE LEDGER

What generated each prediction dir. **Going forward this is automatic:** `phase30/benchmark.py`
(commit 706bfe8) drops a `summary.json` into every model dir — full resolved args (SAM model,
reranker, score thresholds, …), the scored results block, exact start/end datetimes
(ISO-8601 with timezone) and git SHA. It is written provisionally after inference and
finalised after scoring; `--skip-inference` rescores write `summary_rescore.json` so they
never overwrite the generation record. For dirs created *before* 2026-06-11 there is no
summary, so this file records what we could reconstruct — and, honestly, what we could NOT.

> ⚠️ **Two gotchas that bit us (2026-06-11):**
> 1. **SAM model is NOT pinned for the full-439 table dirs.** The `phase30/zeroshot/`
>    drivers use *different* SAM models — `bench.sh`/`control_run.sh` → **vit_l**,
>    `overnight_shadow.sh` stage-3 → vit_b, stage-5/`run_tta_s0.sh` → vit_h. No single
>    driver unambiguously wrote `phase21_baseline`/`phase22_B_L4`. So their exact SAM
>    is **UNKNOWN**. Do not assume vit_h.
> 2. **Score-threshold floor.** The *current* `benchmark.py → foxtrot.py` path floors
>    detections at the DeepForest RetinaNet default `score_thresh = 0.1`
>    (`foxtrot.py:854`, before `--deepforest_confidence`'s post-filter at `:859`).
>    The table dirs reach scores ~0.0009 → they were made by an **older/different
>    pipeline** that lowered the model's `score_thresh`. Re-running them through
>    today's benchmark.py would NOT reproduce them. **Only certifiable apples-to-apples
>    = re-infer the whole compared set through ONE pinned pipeline and validate
>    phase21 ≈ 0.440.**

## Key dirs (the ALL_models_holdout_table A/B/C rows)

| dir | what | SAM | reranker | score floor | generator (best guess) |
|---|---|---|---|---|---|
| `phase21_baseline` | DeepForest no-shadow, zero-shot, 439 | **UNKNOWN** | off (base ablation) | ~0 (full tail) | older pipeline (NOT current benchmark.py) |
| `phase22_B_L4` | DeepForest shadow×4, zero-shot, 439 | **UNKNOWN** | off | ~0 | older pipeline |
| `ablation_pre_s2` | DeepForest shadow×2, zero-shot, 439 | **UNKNOWN** (memory flagged unconfirmed) | off | ~0 | older pipeline |
| `ablation_tcd_s0/s2/s4` | DeepForest fine-tuned on TCD, 439 | **UNKNOWN** | off | ~0 | older pipeline |
| `detectree2_stock_100m_px` | detectree2 @100m, geo→pixel | n/a (native masks) | n/a | n/a | `infer_detectree2.py` (conda `tcd` env), pixel-transformed |
| `restor_mrcnn` | Restor MRCNN re-inference, TREE-only export | n/a (MRCNN masks) | n/a | n/a | `phase30/restor_baseline/run_restor.py` (CPU detectron2) |
| `s0_ttaR` | ablation_tcd_s0 + TTA + reranker | vit_h | **ON** (`cnn_reranker_ens3.pt`) | — | `phase30/zeroshot/run_tta_s0.sh` |

## DEAD / discarded
- `blindzero_w2`, `blindzero_w4` **geojson dirs** — the 2026-06-11 eval of the blind
  checkpoints. **WRONG SAM for that comparison + score-floored at 0.10** → mAP50
  collapsed to 0.098 (artifact). The geojsons are dead; the underlying CHECKPOINTS
  are sound and important for the ablation (see "TRAINING provenance" below) — they
  need a fresh eval through the pinned pipeline.

## TRAINING provenance — the blind-control (shadow-specificity) checkpoint family
The blind models are the key ablation: same count of upweighted boxes per image as the
shadow logic chose, but picked at RANDOM. If blind ≈ shadow, the gain is generic
hard-example mining; if shadow > blind, the shadow prior is doing real work.

**One recipe, five checkpoints** (script: `phase30/zeroshot/train_blind_local.py`,
DELETED untracked 2026-06-11 — recipe reconstructed here + verifiable in each
checkpoint dir's `train.log` / `version_0/hparams.yaml`):
- delegates to the shared `phase30/lib` trainer (`train_deepforest`); weecology/NEON
  HF base; no canopy; crop400 + flip; MPS; `DF_NUM_WORKERS=0`; Lightning seed 42
- epochs 50, batch 16, lr 0.001, patience 99 (no early stop; top-1 val-map kept)
- train = `phase30/zeroshot/phase22_train_filt.csv` (128 tiles = 112 annotated
  + 16 confirmed-empty, 1386 boxes, BRU/WON/NEON), val = `phase22_val_filt.csv`
- `shadow_loss_reweight=True`, weight 2.0 or 4.0
- `--blind` sets `SHADOW_BLIND_CONTROL=1` before model import →
  `_compute_shadow_gt_weights` (phase30/lib/models.py, commit edee80b): count
  k = #shadow-triggered boxes, reset all weights to 1.0, upweight k boxes chosen
  by an UNSEEDED `np.random.default_rng().choice` (re-randomised every step,
  overlap with true shadow boxes allowed — so blind choice is non-deterministic
  despite the global seed 42)

| checkpoint | arm | wt | trained | best ckpt (val-map) |
|---|---|---|---|---|
| `checkpoints/ablation_shadow_w4_local/` | **real shadow** | 4 | 2026-06-04 | epoch=23 (0.47) |
| `checkpoints/ablation_blind_w4/` | blind | 4 | 2026-06-04 | epoch=23 (0.48) |
| `checkpoints/ablation_blind_w2/` | blind | 2 | 2026-06-04 | epoch=23 (0.48) |
| `checkpoints/blindzero_w2/` | blind | 2 | 2026-06-10 | epoch=29 (0.484) |
| `checkpoints/blindzero_w4/` | blind | 4 | 2026-06-10 | epoch=23 (0.470) |

The 2026-06-04 trio came from the `run_blind_local.sh` driver (3 sequential runs,
log: `checkpoints/run_blind_local.driver.log`); the 2026-06-10 pair from
`caffeinate -i bash -c '... train_blind_local.py --shadow-weight 2 --run-name
blindzero_w2 --blind && ... --shadow-weight 4 --run-name blindzero_w4 --blind'`.
So per weight there are TWO blind seeds; at w4 there is also a TRAINING-MATCHED
real-shadow run (`ablation_shadow_w4_local`).

**Matched-pair caveats for the ablation table:**
- The zero-shot table's shadow rows (`phase21_baseline`, `ablation_pre_s2`,
  `phase22_B_L4`) were trained by the OLDER pipeline — NOT training-matched to the
  blind arm. The only fully matched pair is `ablation_shadow_w4_local` vs
  `{ablation_blind_w4, blindzero_w4}`.
- No matched local real-shadow w2 exists (and no blind w1/blind-none — weight 1.0
  is a no-op ≡ neutral).
- The TRAINING of all five is sound; it was only the earlier blindzero EVAL
  (vit_h + the 0.1 score-floor) that was invalid.

## The canonical zero-shot SHADOW-vs-BLIND control lives ELSEWHERE
Not here — in `phase30/shadow_eval/zeroshot/` (generated by `phase30/zeroshot/control_run.sh`,
SAM **vit_l**, 74-tile subset `phase30/shadow_eval/subset_tiles.txt`). Verdict: shadow ≈ blind
on mAP50 (0.141 vs 0.141). See memory `project_blindzero_control`.

## Everything else (~28 dirs)
`kunqi*` (×11), `phase32*`, `s4_diag`/`s4_local_smoke`/`s4_local_stability`,
`ARCHIVEphase22_B_L4`, `deepforest_stock`, `local_kunqi`, `kunqilocal`, `detectree2_stock_100m`
(pre-pixel-transform) — experimental / superseded / unverified. No `_run.json`, provenance lost.
Trust `_run.json` for anything generated from 2026-06-11 onward.
