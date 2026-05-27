# Phase 30 — TCD Tree Crown Detection

Fine-tuning a DeepForest (RetinaNet) model on the [Restor TCD dataset](https://huggingface.co/datasets/restor/tcd) (~4,100 tiles) to detect individual tree crowns. Follows the paper split strategy (folds 0–3 training, fold 4 early stopping, holdout test set untouched).

## Folder contents

| File | Purpose |
|---|---|
| `train.py` | Training launcher — call with CSV paths and checkpoint to start training |
| `build_phase30_csvs.py` | Streams TCD from HuggingFace → writes train/val CSVs + canopy polygons JSON |
| `visualise_canopy_policy.py` | Renders an explainer PNG showing how the canopy policy treats ITC vs canopy anchors |
| `evaluate.py` | Evaluates a trained checkpoint against the TCD holdout test set |
| `lib/` | Internal modules: training loop, model definition, utilities, configs |
| `inference/predict.py` | Two-stage inference pipeline: DeepForest → SAM segmentation |
| `requirements.txt` | All Python dependencies |

---

## 1. Environment setup

```bash
conda create -n canopyai python=3.10 -y
conda activate canopyai

# Adjust cu121 to match your GPU driver (check with: nvidia-smi)
# Options: cu117, cu118, cu121, cu124
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

---

## 2. Download starting checkpoint

The model fine-tunes from the phase22_B_L4 checkpoint (BRU/WON/NEON pre-trained with shadow loss).

```bash
# From Modal volume (if you have access):
modal volume get canopyai-deepforest-checkpoints \
    /phase22_B_L4/deepforest_final.pth \
    ./phase22_B_L4.pth

# Or request the .pth file directly from Tom.
```

---

## 3. Set the TCD tile path

By default the build script expects tiles at `data/tcd/images/data/tcd/raw/`. If your TCD data lives elsewhere, edit `TRAIN_DIR` near the top of [phase30/build_phase30_csvs.py](phase30/build_phase30_csvs.py):

```python
TRAIN_DIR = Path("/abs/path/to/your/tcd/tiles")   # must contain tcd_tile_*.tif + tcd_tile_*_meta.json
```

You also need the shadow vectors JSON at `data/tcd/tcd_shadow_vectors_by_id.json` (request from Tom) — the build script reads this at startup even when you aren't using shadow loss.

---

## 4. Build training CSVs + canopy polygons

Writes three files into `phase30/`:

- `phase30_tcd_train.csv`, `phase30_tcd_val.csv` — genuine ITC bboxes only (`category_id=2` in COCO; canopy polygons are **not** written as pseudo-ITC rows).
- `phase30_tcd_canopy_polygons.json` — canopy polygon vertices, consumed by the canopy positive policy at training time.

```bash
# First-time setup — streams ~40 GB of tiles from HuggingFace into TRAIN_DIR:
python phase30/build_phase30_csvs.py

# Re-build when tiles + meta.json are already present locally:
python phase30/build_phase30_csvs.py --from-disk
```

---

## 5. Sanity check (1 batch)

```bash
python phase30/train.py --train-csv phase30/phase30_tcd_train.csv --val-csv phase30/phase30_tcd_val.csv --checkpoint phase22_B_L4.pth --canopy-polygons phase30/phase30_tcd_canopy_polygons.json --fast-dev-run
```

Should complete in under 2 minutes with no errors.

---

## 6. Training: experiment matrix

All runs share: lr=1e-4, `shadow_loss_reweight=True`, shadow_weight=2.0, batch=32, epochs=50, patience=10. Canopy regions are handled exclusively via `--canopy-polygons` + `--canopy-loss-scale`.

**Canopy positive (default)** — anchors with IoP ≥ 0.7 against a canopy polygon are treated as positives (cls target=1, regression suppressed):

```bash
python phase30/train.py --train-csv phase30/phase30_tcd_train.csv --val-csv phase30/phase30_tcd_val.csv --checkpoint phase22_B_L4.pth --canopy-polygons phase30/phase30_tcd_canopy_polygons.json --run-name phase31_canopy
```

**Canopy dampened** — each canopy anchor's contribution halved. Use if canopy swamps ITC:

```bash
python phase30/train.py --train-csv phase30/phase30_tcd_train.csv --val-csv phase30/phase30_tcd_val.csv --checkpoint phase22_B_L4.pth --canopy-polygons phase30/phase30_tcd_canopy_polygons.json --canopy-loss-scale 0.5 --run-name phase31_canopy_scale05
```

**Canopy iscrowd-style ignore** — `--canopy-loss-scale 0.0` strips canopy anchors from the cls loss entirely (and from the denominator):

```bash
python phase30/train.py --train-csv phase30/phase30_tcd_train.csv --val-csv phase30/phase30_tcd_val.csv --checkpoint phase22_B_L4.pth --canopy-polygons phase30/phase30_tcd_canopy_polygons.json --canopy-loss-scale 0.0 --run-name phase31_canopy_ignore
```

**ITC-only ablation** — omit `--canopy-polygons` entirely. Canopy regions become unannotated, so anchors there are trained as negatives. Useful as a worst-case lower bound on the value of modelling canopy at all:

```bash
python phase30/train.py --train-csv phase30/phase30_tcd_train.csv --val-csv phase30/phase30_tcd_val.csv --checkpoint phase22_B_L4.pth --run-name phase31_itc_only
```

Checkpoints land in `checkpoints/<run-name>/`; best `map` is kept. Internal IoP threshold is `ShadowConditionedDeepForest.CANOPY_IOP_THRESH = 0.7`. Val `map` treats canopy regions as `iscrowd` — detections inside a canopy polygon with no matching ITC GT are dropped before scoring, so canopy predictions are neither rewarded nor penalised.

---

## 7. Optional — explainer visualisation

Renders a five-panel figure showing how the canopy positive policy treats representative ITC and canopy anchors in real training tiles. Useful for sharing with collaborators:

```bash
python phase30/visualise_canopy_policy.py
```

Writes [phase30/canopy_policy_explainer.png](phase30/canopy_policy_explainer.png).

---

## 8. Inference

Run the two-stage DeepForest → SAM pipeline on a single tile:

```bash
python inference/predict.py \
    --image_path /path/to/tile.tif \
    --deepforest_model /path/to/checkpoint.pth \
    --output_dir ./output
```

---

## 9. Benchmark

Evaluate a checkpoint against the TCD holdout test set:

```bash
python evaluate.py \
    --checkpoint /path/to/checkpoint.pth \
    --tcd-dir /path/to/tcd/test/tiles \
    --output-root ./benchmark_results
```

---

## 10. CNN reranker (post-detection score recalibration)

A per-polygon image-patch classifier that replaces the foxtrot
`deepforest_score` with a calibrated TP-probability.  Lifts mAP50 on the
OAM-TCD 439-tile holdout from 0.347 (score-NMS baseline) to **0.515**
(3-CNN ensemble, strict-independent), beating Restor's Mask-RCNN R50
reference of 0.432.  No retraining of DeepForest or SAM.

Full methodology and reproducer: [cnn_reranker_methodology.md](cnn_reranker_methodology.md).

```bash
python phase30/cnn_reranker.py \
    --src       benchmark_results_holdout/<eval-folder> \
    --holdout-dir data/tcd/images/data/tcd/val \
    --train-src benchmark_results_train/<train-folder> \
    --train-holdout-dir data/tcd/images/data/tcd/raw \
    --dst       benchmark_results_holdout/<output-folder> \
    --epochs 8 --batch-size 128
```

Ensemble multiple runs with [ensemble_geojsons.py](ensemble_geojsons.py).
Per-tile mAP delta diagnostics: [compare_map50.py](compare_map50.py).

---

## Notes

- **Batch size**: 32 is set for A100-40GB. Reduce to 16 if you OOM.
- **num_workers**: hardcoded to 4 in the DataLoaders — reduce to 0 if you hit multiprocessing issues.
- **W&B**: not configured by default. Pass `wandb_project="your-project"` in `train.py` to enable.
- **CUDA version**: check `nvidia-smi` and install the matching PyTorch wheel.
