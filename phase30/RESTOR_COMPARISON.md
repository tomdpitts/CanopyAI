# Comparison to Restor OAM-TCD (arXiv 2407.11743)

How our DeepForest→SAM→reranker crown pipeline is positioned against the Restor
OAM-TCD baselines, with the metric definitions verified line-for-line against their
code (`github.com/Restor-Foundation/tcd`). Written so the comparison isn't quietly
apples-to-oranges.

## Two tasks, two models — do not conflate
Restor train a **separate model per task** and report each on the OAM-TCD **holdout**:

| task | their model | their numbers (holdout) |
|---|---|---|
| **instance** segmentation | Mask-RCNN ResNet50 | **mAP50 = 0.432** (COCO segm, maxDets 512) |
| **semantic** segmentation | SegFormer mit-b5 | **IoU 0.876 · F1 0.902 · Acc 0.890** |

"COCO segm task" is *instance* segmentation scored on **masks** (not bbox) — it is **not**
semantic segmentation. Our single pipeline gets scored on both: mAP50 (instance) and,
by rasterising all crowns into one tree/not-tree map, pixel IoU/F1 (semantic).

## Metric definitions — verified against their code
All confirmed identical to ours (`phase30/benchmark.py`):

| metric | Restor | ours | match |
|---|---|---|---|
| semantic **IoU** | torchmetrics `JaccardIndex(multiclass, n=2)` → **macro** ½(bg+tree) | macro ½(bg+tree) | ✅ |
| semantic **F1** | classwise `F1Score`, **tree** column = positive-class Dice | tree Dice | ✅ |
| semantic **Acc** | classwise `Accuracy(average="none")`, tree = **recall** (not overall px acc) | tree recall | ✅ |
| mask threshold | 0.5 (argmax over 2 classes) | 0.5 | ✅ |
| instance **mAP50** | `COCOeval(iouType="segm")`, IoU 0.5, areaRng all, **maxDets 512** | same (`--max-dets 512`, now the default) | ✅ |

→ The **semantic** comparison is apples-to-apples. The instance *task type and settings*
match too — but the **category scope** does not (below).

## The instance-mAP catch: 0.432 is the tree+canopy 2-category MEAN
Code-confirmed (`tools/generate_dataset.py`, `models/instance_segmentation.py`):
the OAM-TCD COCO files have **2 categories** — `tree` (id 2) and `canopy` (id 1) — and
their `COCOEvaluator` applies **no category filter**, so the headline `AP50` is the
**mean of AP-tree and AP-canopy**. **No per-category (tree-only) AP is published anywhere**
(repo / paper / HF card). Our default eval is **tree-only** (canopy as `iscrowd` ignore),
so our 0.515 / 0.535 are **not** the same quantity as their 0.432.

## Why putting canopy in mAP50 is methodologically shaky
Instance AP is for **"things"** (discrete, countable). Canopy is **"stuff"** (amorphous
regions). The instance *decomposition* of a canopy area is arbitrary — one annotator
draws it as a single polygon, another as N sub-polygons, both valid — and **AP is not
invariant to that choice**: the same prediction over the same pixels scores ≈1.0 against
the one-polygon GT but ≈0.5 (or worse) against the N-polygon GT (a single big mask can
only match one GT instance, and IoU with each sub-blob falls toward the 0.5 borderline).
**Semantic IoU is fully decomposition-invariant** (pixel coverage only) — which is exactly
why canopy belongs in the semantic metric. Defensible paper argument: **report instance
mAP tree-only; evaluate canopy semantically.** Restor's 0.432 folds a stable things-AP
together with an ill-posed stuff-AP.

## Our pipeline on the 2-category benchmark (honest, and the oracle ceiling)
We predict only tree crowns (no canopy head). In the proper 2-category framing
(`phase30/canopy_aggregation_test.py`, on our best model `tcd_s0`):

| | tree-AP | canopy-AP | 2-cat mean |
|---|---|---|---|
| no canopy predictions | **0.346** | 0 | **≈0.17** |
| **oracle** canopy aggregation (T=0.5) | 0.346 | 0.548 | **0.447** |
| oracle (T=0.7) | 0.346 | 0.507 | 0.427 |
| _Restor_ | _?_ | _?_ | _0.432_ |

Two facts: (1) tree-AP **drops 0.535→0.346** in the 2-cat framing because crowns landing
on canopy stop being ignored and become tree false-positives; (2) the **oracle**
aggregation — merge crowns with high IoP into each *GT* canopy polygon → one canopy mask —
reaches 0.447, narrowly beating 0.432, but it **uses ground truth** to place canopy, so it
is an **upper bound, not deployable**. A fair version needs GT-agnostic canopy proposals.

## Open item (next): Restor tree-only AP
Run `restor/tcd-mask-rcnn-r50` through Detectron2's `COCOEvaluator` to read the
per-category `AP-tree` / `AP-canopy` it prints — the matched tree-only number to put
against our 0.535. If their learned canopy-AP is high (their model *is* trained on canopy),
their AP-tree could back out to ~0.31, i.e. below ours; if low, they're near 0.43 and we're
modestly ahead. Either way it settles whether canopy is materially in the 0.432.

## Preliminary shadow-ablation results (reranker-ON, full 439, IN PROGRESS as of this commit)
Tree-only mAP50 + semantic IoU/F1/recall. `w1`=no-shadow, `w2`/`w4`=shadow weight.

| regime | shadow | model | mAP50 | IoU | F1 | recall |
|---|---|---|---|---|---|---|
| zero-shot | w4 | `phase22_B_L4` | 0.498 | 0.611 | 0.614 | 0.584 |
| zero-shot | w1 | `phase21_baseline` | _re-scoring_ | — | — | — |
| zero-shot | w2 | `ablation_pre_s2` | _running_ | — | — | — |
| fine-tuned | w1 | `ablation_tcd_s0` | **0.535** | 0.658 | 0.654 | 0.542 |
| fine-tuned | w2 | `ablation_tcd_s2` | 0.500 | 0.664 | 0.665 | 0.562 |
| fine-tuned | w4 | `ablation_tcd_s4` | 0.515 | 0.659 | 0.656 | 0.550 |

Reranker-OFF half + final synthesis to follow. Early read: **shadow gives no edge in the
fine-tuned regime** (w1 ≥ w4); the zero-shot contrast (w1 vs w4) is the remaining test.
