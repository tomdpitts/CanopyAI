#!/usr/bin/env python3
"""
infer_segformer.py — SegFormer mit-b5 inference on TCD tiles.

Loads restor/tcd-segformer-mit-b5 from HuggingFace, runs tiled semantic
segmentation, converts the tree-class probability map to instance polygons
via connected components, and writes {stem}_canopyai.geojson in pixel space
(matching foxtrot.py output format).

Usage:
    python infer_segformer.py \
        --image_path data/tcd/images/data/tcd/raw/tcd_tile_0.tif \
        --output_dir benchmark_results/segformer
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from shapely.geometry import shape, mapping
from shapely.validation import make_valid
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_ID   = "restor/tcd-segformer-mit-b5"
TILE_SIZE  = 512
OVERLAP    = 64
MIN_AREA   = 50     # minimum pixel area to keep an instance
TREE_CLASS = 1      # class index for tree crown in the TCD model


# ── Model ─────────────────────────────────────────────────────────────────────

_processor = _model = _device = None

def get_model():
    global _processor, _model, _device
    if _model is not None:
        return _processor, _model, _device
    _device = ("mps"  if torch.backends.mps.is_available()  else
               "cuda" if torch.cuda.is_available()           else "cpu")
    print(f"  Loading {MODEL_ID} on {_device} ...")
    _processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    _model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    _model.to(_device).eval()
    return _processor, _model, _device


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_prob_map(img_np):
    """
    Tiled inference on a (H, W, 3) uint8 image.
    Returns a (H, W) float32 probability map for the tree class.
    Tiles overlap by OVERLAP pixels to avoid edge artefacts.
    """
    processor, model, device = get_model()
    H, W = img_np.shape[:2]
    prob_sum = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)

    stride   = TILE_SIZE - OVERLAP
    y_starts = sorted(set(list(range(0, max(1, H - TILE_SIZE), stride)) + [max(0, H - TILE_SIZE)]))
    x_starts = sorted(set(list(range(0, max(1, W - TILE_SIZE), stride)) + [max(0, W - TILE_SIZE)]))

    with torch.no_grad():
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + TILE_SIZE, H)
                x1 = min(x0 + TILE_SIZE, W)
                tile   = img_np[y0:y1, x0:x1]
                inputs = processor(images=Image.fromarray(tile), return_tensors="pt").to(device)
                logits = model(**inputs).logits          # (1, C, H/4, W/4)
                logits_up = F.interpolate(
                    logits, size=(y1 - y0, x1 - x0),
                    mode="bilinear", align_corners=False,
                )
                probs = torch.softmax(logits_up, dim=1)[0, TREE_CLASS].cpu().numpy()
                prob_sum[y0:y1, x0:x1] += probs
                count[y0:y1, x0:x1]    += 1

    return prob_sum / np.maximum(count, 1)


# ── Instance extraction ────────────────────────────────────────────────────────

def prob_map_to_instances(prob_map, threshold=0.5):
    """
    Threshold → connected components → polygonise each instance.
    Returns list of (shapely_polygon, score) in pixel coordinates.
    Score = mean tree probability within the instance region.
    """
    from skimage.measure import label, regionprops
    import rasterio.features

    binary = (prob_map >= threshold).astype(np.uint8)
    labeled = label(binary, connectivity=2)

    instances = []
    for region in regionprops(labeled):
        if region.area < MIN_AREA:
            continue
        mask = (labeled == region.label).astype(np.uint8)
        raw_shapes = list(rasterio.features.shapes(mask, mask=mask))
        polys = [shape(geom) for geom, val in raw_shapes if val == 1]
        if not polys:
            continue
        geom = max(polys, key=lambda g: g.area)
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_empty or geom.area < MIN_AREA:
            continue
        score = float(prob_map[labeled == region.label].mean())
        instances.append((geom, score))

    return instances


# ── Output ────────────────────────────────────────────────────────────────────

def save_geojson(instances, out_path):
    features = [
        {
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {"score": score},
        }
        for geom, score in instances
    ]
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for tree class (default: 0.5)")
    return ap.parse_args()


def main():
    args  = parse_args()
    tif   = Path(args.image_path)
    out   = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{tif.stem}_canopyai.geojson"

    with rasterio.open(tif) as src:
        img = src.read([1, 2, 3])
    img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)

    print(f"{tif.name}  ({img.shape[1]}×{img.shape[0]})")
    prob_map  = predict_prob_map(img)
    instances = prob_map_to_instances(prob_map, threshold=args.threshold)
    print(f"  {len(instances)} instances  →  {out_file}")

    save_geojson(instances, out_file)


if __name__ == "__main__":
    main()
