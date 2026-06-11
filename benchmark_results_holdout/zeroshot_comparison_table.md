# Zero-shot comparison — OAM-TCD 439 holdout (Restor-paper metrics, 2026-06-10)

All models scored on the identical `phase30/benchmark.py` harness (columns = Restor TCD paper Table 1).
- **IoU** = macro JaccardIndex (bg+tree avg) · **F1** = tree-class Dice · **Acc** = tree recall · **mAP50/AR@1000** = pycocotools segm, cat=tree, canopy as iscrowd · **IoU-tree** = foreground IoU.
- Detectors run through the DeepForest→SAM(vit_h) pipeline (phase21/22/pre_s2); detectree2 native @ author-spec 100m tiles.

| Model | IoU | F1 | Acc | mAP50 | AR@1000 | IoU-tree |
|---|---:|---:|---:|---:|---:|---:|
| detectree2 (stock, author-spec 100 m) | 0.397 | 0.404 | 0.524 | 0.032 | 0.111 | 0.253 |
| phase21 — DeepForest, NO shadow (wt0) | 0.590 | 0.576 | 0.519 | 0.440 | 0.503 | 0.405 |
| ablation_pre_s2 — shadow ×2 ("shadow 2.0") | 0.610 | 0.611 | 0.575 | 0.493 | 0.561 | 0.440 |
| phase22_B_L4 — shadow ×4 | 0.611 | 0.614 | 0.584 | 0.498 | 0.574 | 0.443 |
| DeepForest (stock NEON) → SAM vit_h | _pending — resume tonight (44/439 done)_ | | | | | |
| **Restor SegFormer mit-b5 (supervised, ref)** | 0.876 | 0.902 | 0.890 | — | — | ~0.81 |
| **Restor Mask-RCNN R50 (supervised, ref)** | — | — | — | 0.432 | — | — |

## Takeaways
- **Shadow weight is monotonically beneficial across EVERY metric**: no-shadow → ×2 → ×4 lifts mAP50 0.440→0.493→0.498, F1 0.576→0.611→0.614, IoU 0.590→0.610→0.611, IoU-tree 0.405→0.440→0.443. Clean paper result.
- **Stock detectree2 transfers poorly to TCD zero-shot** (mAP50 0.032, F1 0.404) — its tropical-forest training + crown-delineation style don't match TCD's diverse global landscapes; it hallucinates crowns on agricultural terrain. We used its **authors' recommended ~100 m tile size** (a 40 m off-spec run scored 0.082 mAP50 — still far below the shadow models, so the conclusion is config-independent).
- Supervised Restor models are the dense/instance ceiling (SegFormer F1 0.902; MRCNN mAP50 0.432) — different (supervised) regime.

## Config provenance / repro
- benchmark.py scoring of existing geojsons: `--models x x x x --names <dirs> --skip-inference --output-root benchmark_results_holdout`.
- detectree2 @100m: `infer_detectree2.py --tile_size 100 --weights detectree2` (tcd conda env), geo→pixel via meta bounds → `detectree2_stock_100m_px/`.
- DeepForest stock: `checkpoints/deepforest_stock/stock.pt` → benchmark.py w/ `--sam-model vit_h` (matches phase21/22). Resume cmd in memory [[project_zeroshot_comparison]].
- BLOCKED on this hardware: StarDist (owncloud weights + needs NIR; TCD is RGB-only), SAM2/OFO (bitsandbytes has no arm64 wheel).
