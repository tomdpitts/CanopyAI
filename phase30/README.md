# Phase 30 — TCD Tree Crown Detection

Fine-tuning a DeepForest (RetinaNet) model on the [Restor TCD dataset](https://huggingface.co/datasets/restor/tcd) (~4,100 tiles) to detect individual tree crowns. Follows the paper split strategy (folds 0–3 training, fold 4 early stopping, holdout test set untouched).

## Folder contents

| File | Purpose |
|---|---|
| `train.py` | Training launcher — call with CSV paths and checkpoint to start training |
| `prepare_data.py` | Downloads TCD tiles from HuggingFace and generates training/val CSVs |
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

## 3. Build training CSVs

Streams the TCD dataset from HuggingFace and generates training/val CSVs with pseudo-canopy bounding boxes. **Downloads ~40 GB of tiles on first run** — point `--save-dir` at a location with enough disk space.

```bash
# Stream from HuggingFace (first-time setup — downloads all tiles):
python prepare_data.py

# If tiles are already downloaded locally:
python prepare_data.py --from-disk --skip-repair

# Dry run (counts only, no writes):
python prepare_data.py --dry-run
```

Output: `phase26_tcd_train.csv` and `phase26_tcd_val.csv` in the current directory.

---

## 4. Sanity check (1 batch)

Before submitting to the cluster, verify paths and environment on a GPU node:

```bash
python train.py \
    --train-csv phase26_tcd_train.csv \
    --val-csv   phase26_tcd_val.csv \
    --checkpoint phase22_B_L4.pth \
    --fast-dev-run
```

Should complete in under 2 minutes with no errors.

---

## 5. Full training

```bash
python train.py \
    --train-csv phase26_tcd_train.csv \
    --val-csv   phase26_tcd_val.csv \
    --checkpoint phase22_B_L4.pth \
    --output-dir checkpoints
```

Key hyperparameters are fixed in `train.py`: `lr=0.001`, `shadow_loss_weight=2.0`, `epochs=50`, `patience=10`.

Checkpoints are saved to `--output-dir/phase30_tcd_L2/`. The best checkpoint (highest `map` on val) is kept; early stopping patience is 10 epochs.

---

## 6. Inference

Run the two-stage DeepForest → SAM pipeline on a single tile:

```bash
python inference/predict.py \
    --image_path /path/to/tile.tif \
    --deepforest_model /path/to/checkpoint.pth \
    --output_dir ./output
```

---

## 7. Benchmark

Evaluate a checkpoint against the TCD holdout test set:

```bash
python evaluate.py \
    --checkpoint /path/to/checkpoint.pth \
    --tcd-dir /path/to/tcd/test/tiles \
    --output-root ./benchmark_results
```

---

## Notes

- **Batch size**: 32 is set for A100-40GB. Reduce to 16 if you OOM.
- **num_workers**: hardcoded to 4 in the DataLoaders — reduce to 0 if you hit multiprocessing issues.
- **W&B**: not configured by default. Pass `wandb_project="your-project"` in `train.py` to enable.
- **CUDA version**: check `nvidia-smi` and install the matching PyTorch wheel.
