# DeepForest Fine-Tuning for Sparse Arid Landscapes

Complete guide for fine-tuning DeepForest on TCD dataset to dramatically improve tree detection.

## 🎯 Goal

Improve detection from **23 trees** (baseline) to **2000+ trees** in sparse landscapes by fine-tuning DeepForest on your specific data.

## 📋 Prerequisites

- TCD dataset downloaded (via `prepare_data.py`)
- Modal account set up
- Python environment with dependencies installed

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Training Data

Convert TCD annotations to DeepForest format:

```bash
# Convert and split into train/val
python deepforest/prepare_deepforest_data.py \
    --data_dir data/tcd/raw \
    --output deepforest_annotations.csv \
    --split \
    --val_fraction 0.2
```

This creates:
- `deepforest_annotations_train.csv` - Training data
- `deepforest_annotations_val.csv` - Validation data

**Output format** (DeepForest CSV):
```
image_path,xmin,ymin,xmax,ymax,label
/path/to/image.tif,100,200,150,250,Tree
```

### Step 2: Upload Data to Modal

```bash
# Upload training data
modal volume put canopyai-deepforest-data \
    deepforest_annotations_train.csv \
    /annotations_train.csv

# Upload validation data  
modal volume put canopyai-deepforest-data \
    deepforest_annotations_val.csv \
    /annotations_val.csv

# Verify upload
modal run modal_deepforest.py::list_data
```

### Step 3: Train on Modal

```bash
# Standard training (20 epochs, ~2-3 hours on A10G)
cd deepforest && modal run modal_deepforest.py \
    --epochs 20 \
    --batch_size 16 \
    --lr 0.001

# With Weights & Biases logging
cd deepforest && modal run modal_deepforest.py \
    --epochs 20 \
    --batch_size 16 \
    --wandb_project canopyai-deepforest \
    --run_name foxtrot
```

### Step 4: Download & Use Fine-tuned Model

```bash
# Download checkpoint
modal volume get canopyai-deepforest-checkpoints \
    /checkpoints/deepforest_final.pth \
    ./deepforest_finetuned.pth

# List all checkpoints
cd deepforest && modal run modal_deepforest.py::list_checkpoints
```

## 🔧 Using the Fine-tuned Model in Foxtrot

Update `foxtrot.py` to use your fine-tuned model:

```python
# In detect_trees_with_deepforest() function
model = deepforest_main.deepforest()
model.load_model("weecology/deepforest-tree")  # Baseline

# Replace with:
model = deepforest_main.deepforest()
model.model.load_state_dict(torch.load("deepforest_finetuned.pth"))
```

Then run inference:

```bash
python foxtrot.py --image_path data/tcd/bin_liang/tcd_tile_WON.tif
```

## 📊 Expected Results

### Before Fine-tuning (Baseline)
- Detected: 23 trees
- Actual: 2000+ trees
- **Miss rate: ~99%** 😞

### After Fine-tuning (Target)
- Detected: 1500-2000+ trees
- Actual: 2000+ trees
- **Recall: 75-95%** 🎉

## 🎛️ Training Parameters

### Recommended Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--epochs` | 20-30 | More epochs = better but slower |
| `--batch_size` | 8-16 | Depends on GPU memory |
| `--lr` | 0.001 | Standard learning rate |
| `--patience` | 5 | Early stopping patience |

### Advanced Tuning

```bash
# Longer training for maximum performance
cd deepforest && modal run modal_deepforest.py \
    --epochs 50 \
    --batch_size 8 \
    --lr 0.0005 \
    --patience 10

# Fast experimentation
cd deepforest && modal run modal_deepforest.py \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.002
```

## 📁 File Structure

```
canopyAI/
├── deepforest/
│   ├── prepare_deepforest_data.py   # Data conversion script
│   ├── train_deepforest.py           # Training script (runs on Modal)
│   ├── modal_deepforest.py           # Modal deployment
│   └── DEEPFOREST_TRAINING.md        # This guide
│
├── foxtrot.py                        # Inference pipeline (use fine-tuned model here)
│
├── data/tcd/raw/                 # TCD dataset
│   ├── tcd_tile_0.tif
│   ├── tcd_tile_0_meta.json
│   └── ...
│
├── deepforest_annotations_train.csv   # Training data
├── deepforest_annotations_val.csv     # Validation data
└── deepforest_finetuned.pth          # Fine-tuned model (download from Modal)
```

## 🧪 Local Training (Optional)

If you want to test locally before using Modal:

```bash
python deepforest/train_deepforest.py \
    --train_csv deepforest_annotations_train.csv \
    --val_csv deepforest_annotations_val.csv \
    --epochs 5 \
    --batch_size 4 \
    --output_dir ./local_checkpoints
```

**Note**: Local training on CPU/MPS will be much slower than Modal GPU.

## 🔍 Monitoring Training

### With Weights & Biases

1. Create account at [wandb.ai](https://wandb.ai)
2. Login: `wandb login`
3. Add `--wandb_project` to training command
4. Monitor at https://wandb.ai/your-username/canopyai-deepforest

### Metrics to Watch

- **box_recall**: Higher is better (target: >0.7)
- **box_precision**: Higher is better (target: >0.6)
- **classification_loss**: Lower is better

## 🐛 Troubleshooting

### "No annotations found"
```bash
# Check your data directory
ls -la data/tcd/raw/*_meta.json

# Verify annotations exist
python -c "import json; print(json.load(open('data/tcd/raw/tcd_tile_0_meta.json'))['coco_annotations'])"
```

### "CSV not found on Modal"
```bash
# Verify volume contents
cd deepforest && modal run modal_deepforest.py::list_data

# Re-upload if needed
modal volume put canopyai-deepforest-data annotations_train.csv /annotations_train.csv
```

### Out of GPU Memory
```bash
# Reduce batch size
cd deepforest && modal run modal_deepforest.py --batch_size 4
```

## 💰 Cost Estimate

- **A10G GPU on Modal**: ~$1.10/hour
- **Training time**: 2-3 hours for 20 epochs
- **Total cost**: ~$2.20-3.30 per run

## 🎓 Tips for Best Results

1. **Start small**: Train on 3-5 images first to verify pipeline works
2. **Monitor validation**: Watch for overfitting (train loss ↓, val loss ↑)
3. **Early stopping**: Let it stop automatically when no improvement
4. **Try lower confidence**: In `foxtrot.py`, use `--deepforest_confidence 0.05`
5. **Compare baseline vs fine-tuned**: Run both on same image to see improvement

## 📚 Additional Resources

- [DeepForest Documentation](https://deepforest.readthedocs.io/)
- [Modal Documentation](https://modal.com/docs)
- [TCD Dataset](https://huggingface.co/datasets/restor/tcd)

## 🤝 Next Steps

After fine-tuning DeepForest:

1. ✅ Use fine-tuned model in `foxtrot.py`
2. ✅ SAM will segment detected trees automatically
3. 🔬 Optionally: Fine-tune confidence thresholds
4. 📊 Compare results with baseline

Happy training! 🌲🚀
