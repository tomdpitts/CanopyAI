# Shadow × Reranker ablation — kunqi (0.515 mAP50) pipeline

## Objective
The kunqi series fine-tunes on TCD **from `phase22_B_L4`** (a shadow-pretrained
zero-shot model) with the latest canopy-anchor + smallest-overlap-NMS code, and
reaches **0.515 mAP50** through detector → SAM-H → reranker. This ablation
answers, with proper controls:

1. Does the **shadow** advantage (proven zero-shot: +0.13 Area-F1) **survive TCD
   fine-tuning** into the final reranked system?
2. Is it **shadow-specific**, not generic loss reweighting?
3. What does the **reranker** contribute, and does it **interact** with shadow?

## Design — two factors

### Factor A — Shadow (set by the pretraining checkpoint; kunqi recipe identical)
`shadow_loss_weight` is a per-run modifier (1 = no reweighting). All three arms
use the **same pretrain recipe**, varying only the shadow knob:

| Arm | Pretrain (BRU/WON/NEON, dense shadow) | Role |
|---|---|---|
| **NoShadow** | weight **1** (`phase22_B_L1`) — use `phase21_baseline` | clean no-shadow control |
| **Blind** | weight 4, **random** equal-count reweight (`SHADOW_BLIND_CONTROL=1`) | specificity control — shadow, or any reweighting? |
| **Shadow** | weight 4 (phase22's choice) = `phase22_B_L4` | treatment |

> **`phase21_baseline` ≡ `phase22_B_L1` — confirmed** (not just assumed): see Provenance below.

### Provenance (from `deepforest_custom/modal_deepforest.py` — the Modal/HPC trainer)
Both pretrains were Modal runs from **weecology** (`load_model("weecology/deepforest-tree")`, no `--checkpoint`), identical recipe:
**`--epochs 50 --lr 0.001 --batch-size 16 --patience 10`**, on the **same 129 BRU/WON/NEON tiles** (`phase22_train.csv` = `phase21_train.csv` + shadow vectors; image overlap 129/129).

- **`phase22_B_L4`** = that recipe with `shadow_loss_weight=4`, `phase22_train.csv` (shadow vectors present → shadow fires).
- **`phase21_baseline`** = same recipe, `phase21_train.csv` — and the trainer notes *"Phase 21 B_L4 was silently broken: the train CSV had no shadow vectors,"* so **shadow never fired → it is the no-shadow run** (= `phase22_B_L1`).
- (`phase23` onward fine-tunes on TCD **from `phase22_B_L4`** → the kunqi series.)

So the only difference between Shadow and NoShadow is whether shadow reweighting fired — a clean ablation. To **reproduce faithfully** (for the Blind/recreation arms), use: weecology base, `phase22_train.csv`, **50 epochs, lr 0.001, batch 16, patience 10**.

### Factor B — Reranker (post-hoc, no retraining)
Each detector arm evaluated **with** and **without** the `cnn_reranker_ens3`
rescoring stage.

## Two training stages (both on `main`, current code — no branches/old commits)

| Stage | What | Codebase |
|---|---|---|
| **1. Pretrain variants** | NoShadow / Blind / Shadow on phase22 data | `deepforest_custom/` (shadow logic byte-unchanged; env-gated additions) |
| **2. kunqi fine-tune** | TCD fine-tune **from each** pretrain ckpt, canopy+NMS, the 0.515 recipe — identical & seed-paired across arms | `phase30/lib/` (the 0.515 code, with the MPS fix) |

Hold `shadow_loss_weight` **constant/off during TCD fine-tuning** (TCD shadow
coverage is only ~11% → the pretraining checkpoint is the real lever); log it as
an optional secondary factor.

## Evaluation
- **Pipeline:** detector → SAM-H → {reranker off, on}.
- **Metrics:** **mAP50** (segm/polygons — the 0.515 headline) **and Area-F1 +
  precision/recall** (where shadow shows strongest).
- **Sets:** **439 holdout** (primary, the 0.515 set) **and the harder 100-tile
  set** — the shadow edge is far larger where the baseline is weaker (+0.13 vs
  +0.02). Report both; never cherry-pick.
- **Seeds:** ≥3 per arm, seed-paired across arms; report mean ± SD.

## Deliverable table
`{NoShadow, Blind, Shadow} × {reranker off, on} × {mAP50, Area-F1}` (mean ± SD):

- **Shadow main effect:** Shadow − NoShadow (does pretraining shadow survive fine-tuning?)
- **Specificity:** Shadow − Blind (shadow, or any reweighting?)
- **Reranker main effect:** on − off
- **Shadow × reranker interaction:** does the reranker preserve or erase the edge?

## Controls baked in (hard-won this session)
- **Blind negative control** — without it, "shadow helps" is unfalsifiable (a
  flawed run looked positive until the blind control).
- **Seeds + error bars** — a single seed misled us (lucky 0.570 → 0.562 seeded).
- **Identical recipe, correct base, seed-paired** — fine-tuning from the wrong
  base (weecology vs phase21_baseline) erased the effect entirely; arms must
  differ *only* in the shadow factor.
- **Report the metric where it lives** — mAP50 may move little while Area-F1
  moves a lot; show both.

## Caveats
- **Scale:** 3 arms × ≥3 seeds × 2 stages, with the canopy-heavy TCD fine-tune →
  a **cloud-CUDA campaign**. Local MPS is viable (canopy MPS leak fixed) but slow.
- **Reranker bias:** `cnn_reranker_ens3` was trained on the *shadow* kunqi5's
  detections — patch-based so detector-agnostic and defensible across arms, but
  it slightly favours the shadow arm. For the cleanest reranker × shadow
  interaction, retrain the reranker per arm; for the headline, fix it and note this.
- **Expected honest outcome:** TCD fine-tuning may *subsume* much of the
  pretraining shadow benefit (the model relearns crowns from labels), so shadow
  could help zero-shot strongly but the final reranked kunqi only modestly —
  especially on mAP50. Still a clean, publishable finding ("strong zero-shot
  prior, partially retained after fine-tuning"); the design captures it either way.
