# OAM-TCD holdout benchmark

**Set:** OAM-TCD val holdout, 439 tiles (`data/tcd/images/data/tcd/val`).
**Scoring:** `phase30/benchmark.py` scorer (`_eval_tile_worker` + pycocotools `_coco_map50`), `pred_score_thresh=0.01`, `max_dets=512`, **no reranker**, identical for all rows.

| model | pipeline | shadow | IoU | F1 | Acc | mAP50 | AR@1000 | IoU_tree |
|---|---|---|---|---|---|---|---|---|
| manual_s1 | DeepForest+SAM  | none | 0.569 | 0.542 | 0.473 | 0.189 | 0.506 | 0.371 |
| manual_s1.5 | DeepForest+SAM  | ×1.5 | 0.564 | 0.543 | 0.498 | 0.203 | 0.518 | 0.373 |
| manual_s2 | DeepForest+SAM  | ×2 | 0.574 | 0.556 | 0.505 | 0.204 | 0.524 | 0.385 |
| manual_s3 | DeepForest+SAM  | ×3 | 0.578 | 0.561 | 0.506 | 0.200 | 0.524 | 0.390 |
| manual_s4 | DeepForest+SAM  | ×4 | 0.549 | 0.524 | 0.483 | 0.181 | 0.498 | 0.355 |
| detectree2_stock | detectree2 stock (Zenodo `230103_randresize_full.pth`) | — | 0.433 | 0.438 | 0.543 | 0.076 | 0.228 | 0.280 |




**Columns** (Restor TCD Table-1 convention): IoU = macro Jaccard (bg+tree avg) · F1 = tree-class Dice · Acc = tree recall (TP/(TP+FN)) · mAP50, AR@1000 = pycocotools segm, cat=tree, canopy as iscrowd · IoU_tree = foreground IoU.

**Notes**
- manual_s* = fine-tuned `deepforest_best.pth`; shadow = `shadow_loss_reweight` weight (s1 = off/no-shadow control; the logged `shadow_loss_weight=2.0` is inactive when reweight is off). DF+SAM: `vit_b`, `df_confidence=0.05`, `max_boxes_sam=512`; self-scored raw in one pass (no reranker, no geojson mutation).
- detectree2 = stock weights, own pipeline/confidence; ran in the conda `tcd` env. Output is EPSG:3395 world coords → converted to pixel space (per-tile inverse tif transform) before scoring; pixel-space copies in `detectree2_stock_px/`. 432 tiles produced detections; 7 zero-detection + 2 nodata tiles were written as empty geojsons so detectree2 is scored on the same full 439 (those count as honest misses).
