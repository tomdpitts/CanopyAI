# DeepForest Training Workflows

This document describes how to train DeepForest models on different datasets.

## Prerequisites

1. **Modal Setup**: Ensure you have Modal installed and authenticated
2. **Data Volumes**: The following Modal volumes should exist:
   - `canopyai-deepforest-data` - For training data and CSVs
   - `canopyai-deepforest-checkpoints` - For model checkpoints

## Workflow 1: Train on TCD Only

### 1. Prepare TCD Data

```bash
cd deepforest

# Create train/val CSVs from TCD dataset
python prepare_deepforest_data.py \
    --data_dir ../data/tcd/raw \
    --output deepforest_annotations_train.csv

python prepare_deepforest_data.py \
    --data_dir ../data/tcd/raw_test \
    --output deepforest_annotations_val.csv
```

### 2. Upload TCD Data to Modal

```bash
# Upload TCD images (from project root)
cd ..
tar -czf tcd_images.tar.gz data/tcd/raw/*.tif data/tcd/raw_test/*.tif
modal volume put canopyai-deepforest-data tcd_images.tar.gz /tcd_images.tar.gz --force

# Upload CSVs
cd deepforest
modal volume put canopyai-deepforest-data \
    deepforest_annotations_train.csv /annotations_train.csv --force

modal volume put canopyai-deepforest-data \
    deepforest_annotations_val.csv /annotations_val.csv --force
```

### 3. Train

```bash
modal run modal_deepforest.py \
    --epochs 30 \
    --batch-size 16 \
    --run-name "tcd_only"
```

**Checkpoint location**: `/checkpoints/tcd_only/`

---

## Workflow 2: Train on WON Only

### 1. Prepare WON Data

```bash
cd deepforest

# Create train/val CSVs from WON dataset
python prepare_deepforest_data.py \
    --data_dir ../data/won/train \
    --output won_annotations_train.csv

python prepare_deepforest_data.py \
    --data_dir ../data/won/test \
    --output won_annotations_val.csv
```

### 2. Upload WON Data to Modal

```bash
# Upload WON images (from project root)
cd ..
tar -czf won_images.tar.gz data/won/train/*.tif data/won/test/*.tif
modal volume put canopyai-deepforest-data won_images.tar.gz /won_images.tar.gz --force

# Upload CSVs
cd deepforest
modal volume put canopyai-deepforest-data \
    won_annotations_train.csv /annotations_train.csv --force

modal volume put canopyai-deepforest-data \
    won_annotations_val.csv /annotations_val.csv --force
```

### 3. Train

```bash
modal run modal_deepforest.py \
    --epochs 30 \
    --batch-size 16 \
    --run-name "won_only"
```

**Checkpoint location**: `/checkpoints/won_only/`

---

## Workflow 3: Train on TCD + WON Combined

### 1. Prepare Combined Data

```bash
cd deepforest

# First, generate individual CSVs as shown above for TCD and WON
# Then combine them

# Combine training CSVs
cat deepforest_annotations_train.csv > combined_train.csv
tail -n +2 won_annotations_train.csv >> combined_train.csv

# Combine validation CSVs
cat deepforest_annotations_val.csv > combined_val.csv
tail -n +2 won_annotations_val.csv >> combined_val.csv
```

### 2. Upload Combined Data to Modal

```bash
# Ensure BOTH image tarballs are uploaded (from Workflow 1 & 2)
# Then upload combined CSVs

modal volume put canopyai-deepforest-data \
    combined_train.csv /annotations_train.csv --force

modal volume put canopyai-deepforest-data \
    combined_val.csv /annotations_val.csv --force
```

### 3. Train

```bash
modal run modal_deepforest.py \
    --epochs 30 \
    --batch-size 16 \
    --run-name "tcd_won_combined"
```

**Checkpoint location**: `/checkpoints/tcd_won_combined/`

---

## Download Trained Models

After training completes, download checkpoints:

```bash
# Download specific run
modal volume get canopyai-deepforest-checkpoints \
    /checkpoints/tcd_only/deepforest_final.pth \
    ./tcd_model.pth

# Or download entire run folder
modal volume get canopyai-deepforest-checkpoints \
    /checkpoints/tcd_only \
    ./tcd_checkpoints
```

---

## Notes

- **Run names** create separate checkpoint folders, allowing you to keep all experiments organized
- **Batch size**: Adjust `--batch-size` based on your dataset size and GPU memory
- **Epochs**: 30 is a good starting point; early stopping will prevent overfitting
- **Image extraction**: The Modal script automatically extracts all available tarballs (`tcd_images.tar.gz`, `won_images.tar.gz`)
- **Path handling**: CSV paths are automatically converted to point to extracted images on Modal
