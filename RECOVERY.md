# RECOVERY.md — phase30 CNN reranker

Disaster-recovery and reproducibility reference for the CNN reranker work
that lifted foxtrot's mAP50 on the OAM-TCD holdout from 0.258 → 0.515.

## Headline result

**mAP50 = 0.515** (strict-independent) on the 439-tile OAM-TCD holdout
using `kunqi5_epoch98.ckpt`, with no retraining of DeepForest or SAM.

Restor Mask-RCNN R50 reference: 0.432.  Magic-wand ranking ceiling
(perfect ranking of current pred set): 0.583.

## Critical artifacts and where they live

| artifact | path | git? | iCloud backup? | notes |
|---|---|---|---|---|
| Code (foxtrot, benchmark, cnn_reranker, etc.) | repo | ✓ pushed to `origin/main` | (git is the backup) | Six commits land the full pipeline. See `Commit history` below. |
| OAM-TCD train tile list (deterministic seed=42) | `phase30/train_sample_seed42.txt` | ✓ | n/a | 600 stems.  Validate with `phase30/sample_reranker_train_tiles.py`. |
| DeepForest checkpoint | `kunqi5_epoch98.ckpt` (repo root) | ✗ (*.ckpt gitignored) | manual — request from Tom | This is the upstream detection model; not part of this work. |
| SAM checkpoint | `sam_vit_b_01ec64.pth` | ✗ (*.pth gitignored) | n/a — downloadable | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |
| CNN reranker ensemble (3 × ResNet18 weights) | `phase30/cnn_reranker_ens3.pt` | ✗ (*.pt gitignored, 130 MB > GitHub 100 MB limit) | ✓ `~/Library/Mobile Documents/com~apple~CloudDocs/canopyai_artifacts/cnn_reranker_ens3_<date>.pt` | Reproducible from scratch in ~75 min on M4 Max MPS. |
| Foxtrot predictions over train sample (input to reranker training) | `benchmark_results_train/kunqi5_train_for_reranker/` | ✗ (gitignored) | ✓ in `canopyai_0515_artifacts_<date>.tar.gz` | ~3 hours to regenerate via `phase30/benchmark.py` on the 600-tile sample. |
| 0.515 rescored holdout geojsons | `benchmark_results_holdout/rs_cnnonly_ens3/` (also `rs_cnn_canonical/` from the canonical retrain at 0.513) | ✗ (gitignored) | ✓ in the same tarball | Regenerable in ~3 hours via foxtrot end-to-end with the reranker checkpoint. |

## Commit history (most recent first)

```
aa58a57  Sync evaluate.py to use canonical foxtrot.py, delete legacy duplicate
48e5e28  Remove --area-weight flag (always 0 / pure-score sort)
9d218a7  Document CNN reranker checkpoint location, gitignore *.pt
76313e0  Preserve reproducibility anchor: deterministic 600-tile train sample
1caf479  Integrate CNN reranker as optional foxtrot 3rd stage
69cf8aa  Add CNN reranker — lifts mAP50 to 0.515 on OAM-TCD holdout
c5f78b3  Ignore benchmark_results_holdout output artifacts
188fdac  Switch foxtrot NMS to score-based ranking + Restor-paper viz metrics
```

## Reproduce the 0.515 result

### Fast path — with the existing checkpoint

If `phase30/cnn_reranker_ens3.pt` is present:

```bash
./venv310/bin/python phase30/benchmark.py \
    --models kunqi5_epoch98.ckpt \
    --names  kunqi5_e2e \
    --output-root benchmark_results_holdout \
    --df-confidence 0.0 \
    --pred-score-thresh 0.0 \
    --reranker-checkpoint phase30/cnn_reranker_ens3.pt \
    --skip-existing
```

Takes ~3 hours (foxtrot is the bottleneck — DeepForest + SAM per tile).

### Recover the checkpoint from backup

```bash
cp ~/Library/Mobile\ Documents/com~apple~CloudDocs/canopyai_artifacts/cnn_reranker_ens3_*.pt \
   phase30/cnn_reranker_ens3.pt
```

### Recover from scratch (no checkpoint, no backup)

```bash
# Foxtrot inference on the 600 train tiles (~3 hours)
./venv310/bin/python phase30/benchmark.py \
    --models kunqi5_epoch98.ckpt \
    --names  kunqi5_train_for_reranker \
    --holdout-dir data/tcd/images/data/tcd/raw \
    --output-root benchmark_results_train \
    --tiles-file phase30/train_sample_seed42.txt \
    --df-confidence 0.0 --pred-score-thresh 0.0 --skip-existing

# Foxtrot inference on the holdout (~3 hours)
./venv310/bin/python phase30/benchmark.py \
    --models kunqi5_epoch98.ckpt \
    --names  kunqi5_baseline \
    --output-root benchmark_results_holdout \
    --df-confidence 0.0 --pred-score-thresh 0.0 --skip-existing

# Train + save the 3-CNN ensemble (~75 min on M4 Max MPS)
./venv310/bin/python phase30/cnn_reranker.py \
    --src       benchmark_results_holdout/kunqi5_baseline \
    --holdout-dir data/tcd/images/data/tcd/val \
    --train-src benchmark_results_train/kunqi5_train_for_reranker \
    --train-holdout-dir data/tcd/images/data/tcd/raw \
    --dst       benchmark_results_holdout/rs_cnn_ensemble \
    --n-runs 3 --epochs 8 --batch-size 128 \
    --save-checkpoint phase30/cnn_reranker_ens3.pt

# Evaluate
./venv310/bin/python phase30/benchmark.py \
    --models kunqi5_epoch98.ckpt --names rs_cnn_ensemble \
    --output-root benchmark_results_holdout \
    --skip-inference --pred-score-thresh 0.0
```

Expected: mAP50 = 0.51 ± 0.008 single-run variance.

## Run foxtrot on a fresh, unseen image

```bash
./venv310/bin/python foxtrot.py \
    --image_path /path/to/new_tile.tif \
    --output_dir /path/to/output \
    --deepforest_model kunqi5_epoch98.ckpt \
    --reranker_checkpoint phase30/cnn_reranker_ens3.pt
```

The output geojson's `deepforest_score` is the CNN reranker's TP-probability.
Omit `--reranker_checkpoint` for the 2-stage default (mAP50 ≈ 0.347).

## File map

| file | purpose |
|---|---|
| `foxtrot.py` | DeepForest → SAM pipeline.  Optional 3rd stage (reranker) when `--reranker_checkpoint` is set. |
| `phase30/benchmark.py` | Restor-reference benchmark on the OAM-TCD val holdout (pycocotools mAP50). |
| `phase30/evaluate.py` | Cross-biome benchmark on OAM-TCD raw (IoP-AP, trees + canopy pooled). |
| `phase30/cnn_reranker.py` | Per-polygon CNN reranker.  Training, inference, and checkpoint save/load. |
| `phase30/cnn_reranker_methodology.md` | Full methodology writeup. |
| `phase30/ensemble_geojsons.py` | Multi-source probability averaging. |
| `phase30/compare_map50.py` | Per-tile mAP50 delta diagnostic. |
| `phase30/sample_reranker_train_tiles.py` | Validates / regenerates the train tile list. |
| `phase30/train_sample_seed42.txt` | Authoritative 600-tile train sample. |

## Backup locations

- **GitHub**: https://github.com/tomdpitts/CanopyAI on branch `main`.
- **iCloud Drive**: `~/Library/Mobile Documents/com~apple~CloudDocs/canopyai_artifacts/` contains the
  `.pt` checkpoint and the rescored geojsons tarball, both auto-synced to Apple servers.
- **Working directory**: same iCloud-synced folder for the whole repo, so every uncommitted
  file is also being synced.

## Pinning details

- **Python**: 3.10 (venv at `./venv310/`).
- **PyTorch**: 2.4.1 with MPS support.
- **Key packages**: scikit-learn 1.7.2, torchvision 0.19.1, geopandas, rasterio, shapely,
  pycocotools, segment-anything, deepforest.
- **Hardware on which the result was produced**: Apple M4 Max with MPS acceleration.
