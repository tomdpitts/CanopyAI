# Annotation Workflow for Custom Images

## Quick Start

### 1. Prepare Images for Annotation

```bash
cd solar/shadow_regression
source ../../venv310/bin/activate

# Copy all images from WON003
python prepare_annotation_images.py --image_dir ../../input_data/WON003/images

# Or sample 30 random images
python prepare_annotation_images.py --image_dir ../../input_data/WON003/images --n_samples 30
```

This copies images to `annotation_images/` for easy browser loading.

### 2. Annotate Shadow Directions

```bash
open annotate.html
```

1. Click "Choose Files" and select all images from `annotation_images/`
2. For each image, click on where shadows point (arrow draws from center to click)
3. Use "Skip" for images without clear shadows
4. Navigate with Next/Previous buttons
5. Click "Export Annotations" when done
6. Copy the CSV content from the green text area
7. Save as `data/shadow_annotations.csv`

### 3. Train with Annotated Data

Use the custom dataset class in your training script:

```python
from local_dataset import LocalShadowDataset, SimpleTransform, RotationTransform

# For validation (use exact annotations)
val_dataset = LocalShadowDataset(
    image_dir="../../input_data/WON003/images",
    annotation_csv="data/shadow_annotations.csv",
    transform=SimpleTransform()
)

# For training (with rotation augmentation)
train_dataset = LocalShadowDataset(
    image_dir="../../input_data/WON003/images",
    annotation_csv="data/shadow_annotations.csv",
    transform=RotationTransform()
)
```

The CSV format is:
```csv
filename,shadow_azimuth,skipped
image001.png,45.3,false
image002.png,,true
image003.png,123.7,false
```

## Notes

- Azimuth angles: 0° = North, 90° = East, 180° = South, 270° = West (clockwise)
- Skipped images are automatically excluded from the dataset
- The annotation tool stores angles from center click to edge click
