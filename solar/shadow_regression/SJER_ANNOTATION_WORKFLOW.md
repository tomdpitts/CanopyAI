# SJER Annotation Workflow

## Overview
61 SJER images prepared for manual shadow annotation to add training diversity.

## Steps

### 1. Annotate Images

```bash
cd solar/shadow_regression/annotation_data
open annotate.html  # Opens in browser
```

**In the annotation tool:**
- Use arrow keys to navigate through 61 SJER images
- Click and drag on each image to draw shadow direction
- Press 'S' to skip unclear images
- Export CSV when done

### 2. Save Annotations

Save the exported CSV as:
```
solar/shadow_regression/data/sjer_annotations.csv
```

### 3. Train Combined Model

Once annotated, train on both WON003 + SJER:

```bash
cd solar/shadow_regression
source ../../venv310/bin/activate

# Option A: Train on SJER only first
python train_won003.py \
  --image_dir annotation_data/sjer_images \
  --annotation_csv data/sjer_annotations.csv \
  --output_dir output/sjer_model \
  --epochs 20 --batch_size 8

# Option B: Combine WON003 + SJER (requires modifying training script)
# TODO: Create combined dataset loader
```

## Expected Improvements

- **Current**: 49 WON003 images, 215° shadows, desert shrubland
- **After**: +61 SJER images, diverse shadows, different vegetation (oak woodland)
- **Total**: 110 training images from 2 sites with different shadow directions

This should significantly improve generalization to unseen data.

## Data Locations

- **Images**: `solar/shadow_regression/annotation_data/sjer_images/` (61 PNG files)
- **Annotation tool**: `solar/shadow_regression/annotation_data/annotate.html`
- **Save CSV to**: `solar/shadow_regression/data/sjer_annotations.csv`
