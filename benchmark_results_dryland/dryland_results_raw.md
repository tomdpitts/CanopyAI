# Dryland benchmark

**Set:** dryland-144 (`data/tcd/images/data/tcd/sparse`→`dryland/`): biomes Desert/Xeric + Mediterranean + TempGrass = Desert/Xeric 27, Mediterranean 80, TempGrass 37 (136 train + 8 holdout).
**Scoring:** `phase30/benchmark.py` scorer (`_eval_tile_worker` + pycocotools `_coco_map50`), `pred_score_thresh=0.01`, `max_dets=512`, **no reranker**, identical for all rows.

| model | pipeline | shadow | IoU | F1 | Acc | mAP50 | AR@1000 | IoU_tree |
|---|---|---|---|---|---|---|---|---|
| manual_s1 | DeepForest+SAM  | none | 0.633 | 0.601 | 0.675 | 0.288 | 0.568 | 0.430 |
| manual_s1.5 | DeepForest+SAM  | ×1.5 | 0.619 | 0.589 | 0.700 | 0.340 | 0.604 | 0.418 |
| manual_s2 | DeepForest+SAM  | ×2 | 0.629 | 0.601 | 0.701 | 0.314 | 0.593 | 0.430 |
| manual_s3 | DeepForest+SAM  | ×3 | 0.633 | 0.606 | 0.701 | 0.307 | 0.591 | 0.435 |
| manual_s4 | DeepForest+SAM  | ×4 | 0.609 | 0.576 | 0.690 | 0.328 | 0.599 | 0.405 |
| detectree2_stock | detectree2 stock (Zenodo `230103_randresize_full.pth`) | — | 0.455 | 0.389 | 0.584 | 0.117 | 0.280 | 0.241 |





**Columns** (Restor TCD Table-1 convention): IoU = macro Jaccard (bg+tree avg) · F1 = tree-class Dice · Acc = tree recall (TP/(TP+FN)) · mAP50, AR@1000 = pycocotools segm, cat=tree, canopy as iscrowd · IoU_tree = foreground IoU.

**Notes**
- manual_s1/s2/s4 = fine-tuned `deepforest_best.pth`; shadow = `shadow_loss_reweight` weight (s1 = off/no-shadow control; the logged `shadow_loss_weight=2.0` is inactive when reweight is off).
- detectree2 = stock weights, own pipeline/confidence; ran in the conda `tcd` env. Output is EPSG:3395 world coords → converted to pixel space (per-tile inverse tif transform) before scoring; pixel-space copies in `detectree2_stock_px/`.
