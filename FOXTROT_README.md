# Foxtrot: Two-Stage Tree Detection Pipeline

## Overview

`foxtrot.py` implements a custom 2-stage model for detecting and segmenting trees in sparse arid landscapes:

1. **Stage 1 - DeepForest**: Detects trees and produces bounding boxes
2. **Stage 2 - SAM**: Segments the precise tree canopy within each bounding box

## Installation

First, install DeepForest if not already installed:

```bash
pip install deepforest
```

Make sure you also have the SAM checkpoint downloaded. If not, run:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Usage

Basic usage:

```bash
python foxtrot.py --image_path data/tcd/bin_liang/tcd_tile_WON.tif
```

### Command Line Arguments

- `--image_path` (required): Path to input TIF file
- `--output_dir`: Output directory for results (default: `foxtrot_output`)
- `--sam_checkpoint`: Path to SAM checkpoint (default: `sam_vit_b_01ec64.pth`)
- `--sam_model`: SAM model type - `vit_b`, `vit_l`, or `vit_h` (default: `vit_b`)
- `--device`: Device to run on - `auto`, `cuda`, `cpu`, or `mps` (default: `auto`)
- `--deepforest_confidence`: Minimum confidence threshold for DeepForest (default: 0.3)

### Examples

Using a different output directory:
```bash
python foxtrot.py --image_path data/my_image.tif --output_dir results/
```

Adjusting DeepForest confidence threshold:
```bash
python foxtrot.py --image_path data/my_image.tif --deepforest_confidence 0.5
```

Force CPU usage:
```bash
python foxtrot.py --image_path data/my_image.tif --device cpu
```

## Output

The script generates two files in the output directory:

1. **GeoJSON file** (`{image_name}_deepforest_sam.geojson`): Contains polygon segmentations with properties:
   - `tree_id`: Unique identifier for each tree
   - `deepforest_score`: DeepForest confidence score
   - `area_pixels`: Segmented area in pixels
   - `bbox`: Original bounding box from DeepForest

2. **Visualization image** (`{image_name}_deepforest_sam_visualization.png`): Shows:
   - Bounding boxes from DeepForest (colored rectangles)
   - Segmentation masks from SAM (colored overlay + contours)

## How It Works

1. **Load Image**: Reads the TIF file with georeferencing information
2. **DeepForest Detection**: Runs DeepForest pre-trained model to detect trees and get bounding boxes
3. **SAM Initialization**: Loads SAM model on appropriate device (GPU/MPS/CPU)
4. **SAM Segmentation**: For each bounding box, uses SAM to segment the precise tree canopy
5. **Results Export**: Converts masks to polygons and saves as GeoJSON with visualization

## Advantages of This Approach

- **Focused Segmentation**: SAM only processes regions where trees are likely present (guided by DeepForest)
- **Efficient**: Faster than running SAM on the entire image
- **Accurate**: Combines DeepForest's tree detection expertise with SAM's precise segmentation
- **Sparse Landscapes**: Ideal for arid regions where trees are scattered

## Technical Notes

- The pipeline automatically handles device selection (CUDA > MPS > CPU)
- Masks are simplified to reduce polygon complexity
- Small detections (<50 pixels) are filtered out
- DeepForest uses its latest pre-trained model optimized for aerial imagery
