# Technical report — Zero-shot shadow ablation: training provenance & apples-to-apples audit

**Status:** DRAFT (started 2026-06-11). Working document for the paper's methods/ablation section.
**Scope:** the *zero-shot* arm only (models trained on BRU/WON/NEON, evaluated zero-shot on the
OAM-TCD 439-tile holdout). The fine-tuned-on-TCD (kunqi) and dense-semantic (DeepLabV3+) arms are
out of scope here.
**Goal:** establish, with evidence, the exact training recipe and data behind every model in the
zero-shot comparison so the shadow-loss-reweight ablation is a clean, single-variable comparison.

---

## 1. The claim under test

The paper's contribution (B) is a **shadow-based train-time augmentation**: upweight the focal loss
on ground-truth crowns that cast a detectable shadow, baking in the prior "a cast shadow implies a
valid tree." The ablation must therefore isolate **one variable** — the shadow loss weight — across
otherwise identical training. Two controls matter:

- **No-shadow** (weight effectively off) vs **shadow** (weight > 1): does shadow help at all?
- **Blind control**: upweight the *same number* of randomly chosen crowns per image. If "blind"
  matches "shadow", the gain is generic hard-example mining, not the shadow prior.

For these to be valid, the no-shadow / shadow / blind models must differ in *nothing else* —
same trainer, base weights, data, augmentation, optimiser, schedule, and early-stopping.

---

## 2. Definitive training recipe (the family-A "canonical" recipe)

The two established headline models — `phase21_baseline` (no-shadow) and `phase22_B_L4` (shadow ×4) —
were trained with the **`deepforest_custom`** trainer on Modal (A100), from the weecology/NEON
pretrained DeepForest (RetinaNet, ResNet-50-FPN). Verified from
`deepforest_custom/modal_deepforest.py` (run commands in its docstring) and
`phase30/SHADOW_RERANKER_ABLATION.md` (provenance section).

| hyper-parameter | value | source |
|---|---|---|
| base weights | weecology/deepforest-tree (HF), no `--checkpoint` | run cmd: no checkpoint arg |
| max epochs | **50** | `--epochs 50` |
| early stopping | **EarlyStopping(monitor=val `map`, mode=max, patience=10)** | `--patience 10`; callback at `deepforest_custom/train_deepforest.py:937` |
| checkpoint kept | `save_top_k=1` on val `map` (best epoch, not last) | `ModelCheckpoint` |
| learning rate | **0.001** | `--lr 0.001` |
| batch size | **16** | `--batch-size 16` |
| crop | `RandomCrop(400×400)`, `min_visibility=0.5` | `train_deepforest.py:758` |
| augmentation | **crop-400 + ToTensorV2 only — NO flip** (see §5.2) | `train_deepforest.py:776` (wrapper branch) |
| WON bbox shrink | **ON** — WON boxes shrunk to 50 % area (`WON_BBOX_AREA_FRACTION=0.50`) | `train_deepforest.py:559`, `models.py:581` |
| shadow loss | `shadow_loss_reweight=True`, `shadow_loss_weight=4.0` (phase22) / fires-as-1 (phase21) | run cmd |
| val | `_TiledValDataset` (400-px tiling of the val tiles) | `train_deepforest.py:167` |
| accelerator | CUDA on Modal; **MPS locally** (`accelerator=None` → auto) | — |

> **Answer to "do we definitively have the recipe?": yes.** epochs 50, patience 10 (with early
> stopping on val mAP), lr 1e-3, batch 16, weecology base. The IDE-selected `phase22_B_L4` run
> command confirms `--epochs 50 --patience 10 --lr 0.001 --batch-size 16`; `phase21_baseline` is the
> *same* recipe on the phase21 CSV (its CSV lacked shadow angles so the reweight never fired → it is
> the clean no-shadow control; 129/129 image overlap with phase22).

---

## 3. Training data (BRU / WON / NEON)

**One dataset, two on-disk CSVs that must not be confused:**

- `phase22/phase22_train.csv` — the *original* manifest used on Modal. **129 unique images, 1403
  boxes.** Domain split (unique images): **BRU 42, NEON 55, WON 32.** Paths point at the legacy
  `/Users/tompitts/dphil/CanopyAI/...` location (pre-iCloud) and NEON rows reference `.png` crops
  under `manual_annotations/images/` — **these paths do not resolve on this machine.**
- `phase30/zeroshot/phase22_train_filt.csv` — the **local, runnable** manifest:
  `prepare_phase22.py` remapped every path to the current iCloud root **and dropped tiles whose
  shorter side < 400 px** (incompatible with the 400-px RandomCrop). **128 unique images, 1402 boxes.**
  All 128 image paths resolve locally (0 missing). Val: `phase22_val_filt.csv` (36 images vs the
  original 37).

**The only difference between the two is one dropped tile:** `bru_tile_856_4400_rot285.tif`
(verified 400×200 px — a sub-threshold strip), plus path remapping. So the local `*_filt.csv`
is the *same* BRU/WON/NEON data minus one degenerate BRU crop — a negligible 1/129 change, and the
correct manifest to train against locally.

> **Apples-to-apples on data:** family A (phase21/phase22) trained on the full 129; any local
> retrain uses 128. Either accept the 1-tile delta (recommended — it is a strip that RandomCrop
> could not have used intact anyway) or document it. The domain composition (42/55/32) is otherwise
> identical.

---

## 4. The two trainers — `deepforest_custom/` vs `phase30/lib/`

This is the crux of the apples-to-apples problem: **the existing zero-shot models were not all
trained by the same code.**

**`deepforest_custom/train_deepforest.py`** (1138 lines) — the *original* research trainer.
Supports the full shadow-mechanism zoo (4th-channel `shadow_channel`, `shadow_cross_attention`,
`shadow_proposals`, `shadow_luma_only`, `shadow_input_only`) **and** `shadow_loss_reweight`, plus
**WON bbox shrink** (always on). Wraps DeepForest in `ShadowConditionedDeepForest`
(`deepforest_custom/models.py`). **Trained: `phase21_baseline`, `phase22_B_L4`, and the
`zs_*` / `p22re_*` sweep families.**

**`phase30/lib/train_deepforest.py`** (886 lines) — a *standalone derivative*. Its own docstring:
"Based on `deepforest_custom/train_deepforest.py`; `shadow_channel` / `shadow_cross_attention` /
`shadow_proposals` and **WON bbox normalisation have been removed** — only `shadow_loss_reweight`
is retained." Adds the canopy-positive-policy path (for TCD fine-tuning) and the MPS-stability
fixes. **Trained: `ablation_pre_s2` (shadow ×2) and the local blind family
(`ablation_blind_w2/w4`, `blindzero_w2/w4`, `ablation_shadow_w4_local`).**

**Why it matters — two confounds introduced by the trainer fork:**

1. **WON bbox shrink (≈25 % of the training images).** `deepforest_custom` shrinks every WON
   ground-truth box to 50 % of its area before training (`WON_BBOX_AREA_FRACTION=0.50`); `phase30/lib`
   does not. With 32/129 WON images, this is a materially different supervision signal on a quarter
   of the data. So `phase22_B_L4` (shrunk) and the `phase30/lib` blinds (un-shrunk) are **not**
   trained on identical targets.
2. **Augmentation default differs by branch** — see §5.2.

> **Consequence:** the local blind checkpoints (family C, `phase30/lib`) are **not** a clean control
> for `phase22_B_L4` (family A, `deepforest_custom`). They differ in trainer, WON-shrink, and compute
> backend simultaneously. A valid blind control must be trained by **`deepforest_custom`**.

---

## 5. Confound inventory across the existing zero-shot checkpoints

### 5.1 Trainer / compute / schedule

| model | shadow | trainer | compute | patience | base | matched to A? |
|---|---|---|---|---|---|---|
| `phase21_baseline` | none (no-fire) | deepforest_custom | Modal A100 | 10 (early stop) | weecology | **A (ref)** |
| `phase22_B_L4` | ×4 | deepforest_custom | Modal A100 | 10 (early stop) | weecology | **A (ref)** |
| `ablation_pre_s2` | ×2 | phase30/lib | Modal A100 | 10 | weecology | ✗ trainer + aug + WON |
| `ablation_shadow_w4_local` | ×4 | phase30/lib | local MPS | 99 (no stop) | weecology | ✗ trainer + WON + patience |
| `ablation_blind_w2/w4` | blind 2/4 | phase30/lib | local MPS | 99 | weecology | ✗ trainer + WON + patience |
| `blindzero_w2/w4` | blind 2/4 | phase30/lib | local MPS | 99 | weecology | ✗ trainer + WON + patience |

Two further single-recipe sweep families exist (both `deepforest_custom`, local MPS, crop-only):
- **`zs_*`** (`checkpoints/zeroshot_shadow/`): from weecology, **40 epochs**, weights {0,1,2,4,8}
  singles + **3 shadow seeds vs 3 blind seeds at weight 2** (`zs_sw_2{,b,c}` / `zs_blind_2{a,b,c}`).
  Missing blind ×4. Generated by tracked `sweep.sh` + `control_run.sh`.
- **`p22re_*`**: fine-tuned **from `phase21_baseline.pth`** (not from weecology), 25 epochs,
  weights {0,1,2,4,8} + `p22re_blind4`. **Excluded** from the headline ablation — different base
  initialisation makes it a different experiment (Tom, 2026-06-11).

### 5.2 Augmentation — correction to earlier notes

Earlier provenance notes (and `SHADOW_RERANKER_ABLATION.md`) stated family A used "crop-400 + flip".
**This is wrong.** When `shadow_loss_reweight` is active, `use_wrapper=True`, and the trainer takes
the wrapper branch (`train_deepforest.py:776`): **crop-400 + ToTensorV2, no flip.** The
`HorizontalFlip(p=0.5)` is only added in the *no-wrapper* `else` branch (raw baseline), which none of
these models used. Net augmentation by model:

- **crop-only** (no flip, no photometric): `phase21_baseline`, `phase22_B_L4`, all local blinds,
  `zs_*`, `p22re_*`.
- **crop + photometric** (GaussianBlur / RandomBrightnessContrast / HueSaturationValue): **only
  `ablation_pre_s2`** — `phase30/modal/train.py` passes an explicit `AUG` list, which overrides the
  wrapper default.

> So `ablation_pre_s2` is the single odd-one-out on augmentation **and** trainer. The published
> monotonic curve w0 → ×2 → ×4 (0.440 → 0.493 → 0.498) therefore changes *three* things at the ×2
> point, not one. The w0↔×4 endpoints (`phase21_baseline` ↔ `phase22_B_L4`) are the only clean pair.

---

## 6. What is / isn't a clean comparison today (no retraining)

**Clean (single-variable):**
- **Shadow ×4 vs none** — `phase21_baseline` ↔ `phase22_B_L4`. Identical trainer, data, aug, recipe;
  only the shadow weight differs. *This is the load-bearing "shadow helps" result.*
- **Specificity @ weight 2** — `zs_sw_2{,b,c}` ↔ `zs_blind_2{a,b,c}` (3v3 seeds, deepforest_custom,
  40 ep). Self-consistent within the `zs_*` family.

**Not clean (needs work):**
- The **×2 point** (`ablation_pre_s2`) — confounded by trainer + augmentation + WON-shrink.
- The **blind control at the headline recipe** — no blind model exists that is trained by
  `deepforest_custom` at 50 epochs from weecology. The local blinds use the wrong trainer
  (phase30/lib, no WON-shrink) and no-early-stop schedule.

---

## 7. Recommendation for the retrain (matched, local MPS)

To complete a single-variable zero-shot ablation **matched to the `phase22_B_L4` headline**, retrain
the missing cells with **`deepforest_custom/train_deepforest.py`** (NOT phase30/lib), locally on MPS,
using the §2 recipe: weecology base, 50 epochs, early stopping, lr 1e-3, batch 16, crop-only
augmentation, WON-shrink on, on `phase22_train_filt.csv` / `phase22_val_filt.csv`. **One deliberate
deviation from the historical recipe: patience 5 (≈15 epochs) instead of 10**, so early-stop
actually engages on the small bwn set — this changes only *when* training stops, not the
best-checkpoint selection (global-max val-mAP), and is applied identically to all five cells.

| cell to (re)train | weight | `--blind` | purpose |
|---|---|---|---|
| `zs50_sw_2` (shadow ×2) | 2 | no | clean ×2 point to replace `ablation_pre_s2` |
| `zs50_blind_2` | 2 | yes | blind control @ ×2 |
| `zs50_blind_4` | 4 | yes | blind control @ ×4 (matches `phase22_B_L4`) |

(Optionally add seeds for error bars, and a `zs50_sw_4` re-train to confirm the local-MPS run
reproduces the Modal `phase22_B_L4` number — a useful CUDA↔MPS sanity check.)

`phase21_baseline` and `phase22_B_L4` need **no** retrain — they already are the matched reference
pair. Leaving them as-is preserves the established headline numbers; the only residual difference
from the retrained cells is CUDA (Modal) vs MPS (local), which the optional `zs50_sw_4` sanity
re-train quantifies.

### On "retrain everything with photometric augmentation"
Possible, but it **resets the headline**: changing the augmentation on `phase21`/`phase22_B_L4`
invalidates the published 0.440 / 0.498 and every downstream comparison (Restor, detectree2,
semantic). Recommendation: **do not** add photometric to the headline pair. Keep the established
crop-only recipe and match the new cells to it; treat a photometric sweep, if wanted, as a separate
clearly-labelled experiment rather than a redefinition of the baseline. (This also matches Tom's
stated preference to "stick to the recipes used".)

### Runtime estimate (local MPS)
The 2026-06-04 local blind batch (3 models, 50 ep, 128 tiles) completed in roughly 30–45 min per
model. Three matched cells ≈ **1.5–2.5 h**; with a few seeds, an overnight `caffeinate -i` run.
(Heed the MPS-stability guidance: `DF_NUM_WORKERS=0`, `caffeinate`, watch for the graph-cache leak.)

---

## 8. Open items / decisions

- [ ] **Confirm the retrain set** (the three §7 cells, ± seeds, ± `zs50_sw_4` sanity).
- [ ] **WON-shrink decision:** retrain via `deepforest_custom` (shrink ON, matches headline) —
      confirm this is what we want for the blinds. (Strongly recommended for apples-to-apples.)
- [ ] **Eval pin (separate from training):** once trained, all cells re-infer through ONE pinned
      foxtrot→SAM pipeline (vit_h vs vit_l still open) with the score-floor patch, writing the new
      `summary.json` provenance. Tracked separately in `phase30/zeroshot/REINFER_PLAN_vitl.md`.
- [ ] **`ablation_pre_s2` disposition:** drop from the table (replaced by `zs50_sw_2`), or keep with
      an explicit "confounded" footnote.

---

## Appendix — evidence trail
- Recipe: `deepforest_custom/modal_deepforest.py` (docstring run commands, §"Phase 22"/"Phase 21");
  defaults at `deepforest_custom/train_deepforest.py:439`, early-stop `:937`, aug branch `:776`,
  WON-shrink `:559` + `models.py:581`.
- `phase21 ≡ no-shadow`: `phase30/SHADOW_RERANKER_ABLATION.md` (Provenance section).
- Data: `phase22/phase22_train.csv` (129 img, 42/55/32) vs `phase30/zeroshot/phase22_train_filt.csv`
  (128 img, all local); dropped `bru_tile_856_4400_rot285.tif` = 400×200 px.
- Trainer fork: `phase30/lib/train_deepforest.py` docstring (lines 1–9).
- Checkpoint family + per-run logs: `benchmark_results_holdout/PROVENANCE.md`
  ("TRAINING recipes" + "TRAINING provenance" sections) and each `checkpoints/*/train.log`.
