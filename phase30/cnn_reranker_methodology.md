# CNN reranker for foxtrot — methodology

## Result

On the OAM-TCD 439-tile holdout, using `kunqi5_epoch98.ckpt` with no
retraining of DeepForest or SAM:

```
                                  mAP50    AR@1000   IoU    F1     Acc
Foxtrot baseline (legacy NMS)     0.258    0.573     0.663  0.663  0.558
+ score-based NMS                 0.347    0.589     0.658  0.656  0.548
+ 3-CNN reranker ensemble         0.515    0.589     0.658  0.656  0.548

Restor Mask-RCNN R50 reference    0.432
Theoretical ranking ceiling       0.583
```

The CNN reranker is trained on a **disjoint 600-tile sample of the
OAM-TCD train split** and applied cold to the holdout — zero holdout
label exposure.

## Two architectural changes

### 1. Score-based NMS (foxtrot pipeline)

Foxtrot's NMS originally ranked candidates by bounding-box area.
Low-confidence large detections were eating high-confidence small ones
at every suppression stage.  `apply_nms` now sorts strictly by
descending confidence score; the survivor with the highest score wins
every NMS / containment decision.

This single change lifts mAP50 from 0.258 to 0.347.

### 2. Per-polygon CNN reranker

For each polygon emitted by foxtrot, classify whether it matches a
ground-truth tree at IoU >= 0.5, and use the classifier's calibrated
TP-probability in place of the upstream `deepforest_score`.

The mAP metric integrates precision along the score-sorted ranking, so
re-scoring the same prediction set with a better-calibrated probability
directly improves mAP without changing recall or pixel-union metrics.

#### Architecture

```
Image patch (96x96, polygon bbox + 20% padding, ImageNet-normalised)
   |
ResNet18 (ImageNet pretrained, fine-tuned end-to-end)
   |
512-d embedding
   |
Linear(512, 256) -> ReLU -> Dropout(0.3)
Linear(256, 64)  -> ReLU -> Dropout(0.3)
Linear(64, 1)
   |
sigmoid -> TP probability
```

#### Training

* **Labels:** TP (1) iff the predicted polygon has IoU >= 0.5 with an
  unmatched cat=2 tree GT under greedy score-sorted matching (same
  protocol as pycocotools COCOeval).
* **IGNORE filtering:** predictions falling inside a cat=1 canopy region
  at IoP >= 0.5 with no tree match are **excluded** from training.  This
  mirrors pycocotools' iscrowd ignore at eval time — without it, ~50% of
  training rows are wrongly labelled FP.
* **Loss:** BCE-with-logits, positive-class weight = (1 - TP_rate) / TP_rate.
* **Optimiser:** AdamW with separate learning rates: head 1e-3, backbone 1e-4.
* **Augmentation:** random horizontal/vertical flip + 90-degree rotation.
* **Validation split:** 10% of training tiles held out for best-state
  checkpointing.
* **Epochs:** 8 (longer training overfits — best validation loss is
  typically reached around epoch 4-7).

#### Ensemble

Train 3 independent runs with different random initialisations on the
same training data.  Average the per-polygon TP-probabilities.  Gives
+0.006 mAP over the best single run.

#### Disjoint train / eval

The reranker is trained on a deterministic 600-tile sample of the OAM-TCD
train split (`data/tcd/images/data/tcd/raw/`), disjoint from the 439-tile
holdout by stem.  The trained model is then frozen and applied to the
holdout's foxtrot predictions.  No GT labels from the holdout are ever
visible to the CNN.

## End-to-end reproducer

There are two workflows: a **one-shot training + inference workflow** that
produces a saved checkpoint, and a **deployment workflow** that re-runs
foxtrot with the saved checkpoint on new tiles.

### A. Train the reranker once and save a checkpoint

```bash
# A1. Run foxtrot on the holdout (eval target).
python phase30/benchmark.py \
    --models kunqi5_epoch98.ckpt \
    --names  kunqi5_baseline \
    --output-root benchmark_results_holdout \
    --df-confidence 0.0 --pred-score-thresh 0.0

# A2. Run foxtrot on a disjoint training sample (~600 tiles works).
python phase30/benchmark.py \
    --models kunqi5_epoch98.ckpt \
    --names  kunqi5_train_for_reranker \
    --holdout-dir data/tcd/images/data/tcd/raw \
    --output-root benchmark_results_train \
    --tiles-file phase30/train_sample_seed42.txt \
    --df-confidence 0.0 --pred-score-thresh 0.0

# A3. Train a 3-CNN ensemble in a single call and save the checkpoint.
python phase30/cnn_reranker.py \
    --src       benchmark_results_holdout/kunqi5_baseline \
    --holdout-dir data/tcd/images/data/tcd/val \
    --train-src benchmark_results_train/kunqi5_train_for_reranker \
    --train-holdout-dir data/tcd/images/data/tcd/raw \
    --dst       benchmark_results_holdout/rs_cnn_ensemble \
    --n-runs 3 --epochs 8 --batch-size 128 \
    --save-checkpoint phase30/cnn_reranker_ens3.pt

# A4. Evaluate (the rescored geojsons are already in rs_cnn_ensemble/).
python phase30/benchmark.py \
    --models kunqi5_epoch98.ckpt --names rs_cnn_ensemble \
    --output-root benchmark_results_holdout \
    --skip-inference --pred-score-thresh 0.0
```

`phase30/cnn_reranker_ens3.pt` is now a self-contained reranker
checkpoint (~130 MB for 3 × ResNet18 state_dicts).

**Note on checkpoint storage:** `.pt` files are gitignored (too large
for GitHub's 100 MB hard limit, no LFS configured).  The canonical
trained ensemble is preserved out-of-band at:

```
~/Library/Mobile Documents/com~apple~CloudDocs/canopyai_artifacts/cnn_reranker_ens3_<date>.pt
```

(iCloud Drive auto-syncs this).  Reproducing the result from scratch
takes ~75 min on M4 Max MPS using the recipe above; the trained file
is just a convenience to skip the retraining step.

### B. Run foxtrot end-to-end with the saved reranker

When `--reranker_checkpoint` is passed, foxtrot becomes a 3-stage
pipeline (DeepForest -> SAM -> CNN reranker) and the geojson's
`deepforest_score` is replaced inline with the calibrated probability.
**Omit the flag to fall back to the 2-stage default.**

```bash
# Direct foxtrot call:
python foxtrot.py \
    --image_path /path/to/new_tile.tif \
    --output_dir ./output \
    --deepforest_model kunqi5_epoch98.ckpt \
    --reranker_checkpoint phase30/cnn_reranker_ens3.pt

# Or through benchmark.py for a whole holdout:
python phase30/benchmark.py \
    --models kunqi5_epoch98.ckpt \
    --names  kunqi5_end2end \
    --output-root benchmark_results_holdout \
    --reranker-checkpoint phase30/cnn_reranker_ens3.pt
```

No separate post-processing step.  The output geojsons already carry
the rescored probabilities.

## What this approach can and cannot do

**Can:** improve the *ranking* of an existing prediction set, lifting mAP
by re-ordering TPs above FPs.  This works because mAP integrates precision
along the score-sorted curve, and the upstream detector's confidence
score correlates with TP-ness but isn't optimal for it.

**Cannot:** improve recall (AR@1000), pixel-union segmentation metrics
(IoU, F1, Acc), or find trees the detector never proposed.  The CNN
reranker is a pure rank-rescoring step on a fixed prediction set.

The theoretical ceiling on mAP achievable by rescoring alone equals the
AR@1000 of the current prediction set (in our case 0.589).  The reranker
captures ~89% of this ceiling.  The remaining ~0.07 of headroom would
require improving the detection / SAM stages (more recall, tighter
polygons), which is out of scope for this work.
