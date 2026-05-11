#!/usr/bin/env python3
"""
Quick eval of phase25 checkpoint on 5 val tiles.
Runs sliding-window inference, computes area F1 vs GT polygons, visualises.

Usage:
    source venv310/bin/activate
    python phase25/eval_phase25.py
"""

import json, sys, random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from pathlib import Path
from shapely.geometry import box as shapely_box, Polygon, shape
from shapely.ops import unary_union

ROOT        = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deepforest_custom"))

VAL_CSV     = ROOT / "phase24/phase24_tcd_val.csv"
TCD_RAW     = ROOT / "data/tcd/images/data/tcd/raw"
CKPT        = ROOT / "phase25_epoch00.ckpt"
OUT_PATH    = ROOT / "benchmark_results/phase25_mini_eval.png"
N_TILES     = 5
SEED        = 42
SCORE_THRESH = 0.25
PATCH_SIZE   = 400
PATCH_OVERLAP = 0.05


def load_model():
    from deepforest_custom.models import ShadowConditionedDeepForest
    model = ShadowConditionedDeepForest(shadow_loss_reweight=True)
    ckpt  = torch.load(CKPT, map_location="cpu")
    # Strip Lightning prefix if present
    sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
          for k, v in ckpt["state_dict"].items()
          if not k.startswith("mAP_metric")}
    model.model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")
    return model


def predict_tile(model, tif_path):
    """Sliding-window inference, returns DataFrame of detections."""
    from deepforest import main as df_main
    # Use DeepForest's main model wrapper for predict_tile
    df = df_main.deepforest()
    # Swap in our fine-tuned weights
    our_sd = model.state_dict()
    # DeepForest main.model wraps RetinaNetHub; map keys
    inner_sd = {k: v for k, v in our_sd.items() if not k.startswith("mAP_metric")}
    df.model.load_state_dict(inner_sd, strict=False)
    df.model.eval()
    df.config["score_thresh"] = SCORE_THRESH
    result = df.predict_tile(
        path=str(tif_path),
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
    )
    if result is None:
        return pd.DataFrame(columns=["xmin","ymin","xmax","ymax","score"])
    return result.reset_index(drop=True)


def load_gt_polygons(meta_path):
    """Load GT polygons (both ITC and canopy) from meta.json."""
    meta = json.loads(Path(meta_path).read_text())
    anns = meta.get("coco_annotations", [])
    if isinstance(anns, str):
        try: anns = json.loads(anns)
        except: anns = []
    polys = []
    for ann in anns:
        if isinstance(ann, str):
            try: ann = json.loads(ann)
            except: continue
        if not isinstance(ann, dict): continue
        seg = ann.get("segmentation")
        if not seg: continue
        try:
            if isinstance(seg, list) and seg and isinstance(seg[0], list):
                coords = np.array(seg[0]).reshape(-1, 2)
                if len(coords) >= 3:
                    p = Polygon(coords)
                    if p.is_valid and not p.is_empty:
                        polys.append(p)
            elif isinstance(seg, dict) and "counts" in seg:
                import pycocotools.mask as mask_utils
                from rasterio.features import shapes
                mask = mask_utils.decode(seg)
                for geom, val in shapes(mask.astype(np.uint8), mask > 0):
                    if val == 1:
                        polys.append(shape(geom))
        except Exception:
            pass
    return polys


def area_f1(pred_boxes, gt_polys, img_w, img_h):
    """Compute area precision, recall, F1 using union of pred bboxes vs GT polygons."""
    if (len(pred_boxes) == 0) and not gt_polys:
        return 1.0, 1.0, 1.0
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0
    if not gt_polys:
        return 0.0, 0.0, 0.0

    pred_polys = [shapely_box(r.xmin, r.ymin, r.xmax, r.ymax) for _, r in pred_boxes.iterrows()]
    pred_union = unary_union(pred_polys)
    gt_union   = unary_union(gt_polys)

    intersection = pred_union.intersection(gt_union).area
    precision    = intersection / pred_union.area if pred_union.area > 0 else 0.0
    recall       = intersection / gt_union.area   if gt_union.area   > 0 else 0.0
    f1           = (2 * precision * recall / (precision + recall)
                    if precision + recall > 0 else 0.0)
    return precision, recall, f1


def load_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read([1, 2, 3]).transpose(1, 2, 0)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr.astype(float) - lo) / max(hi - lo, 1), 0, 1)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Pick N_TILES from val set that exist locally
    df     = pd.read_csv(VAL_CSV)
    tiles  = df["image_path"].unique()
    rng    = random.Random(SEED)
    chosen = []
    for p in rng.sample(list(tiles), min(30, len(tiles))):
        stem = Path(p).stem
        tif  = TCD_RAW / f"{stem}.tif"
        meta = TCD_RAW / f"{stem}_meta.json"
        if tif.exists() and tif.stat().st_size > 0 and meta.exists():
            chosen.append((stem, tif, meta))
        if len(chosen) == N_TILES:
            break

    print(f"Running inference on {len(chosen)} tiles...")
    model = load_model()

    fig, axes = plt.subplots(len(chosen), 2, figsize=(16, 5 * len(chosen)))

    results = []
    for row, (stem, tif, meta) in enumerate(chosen):
        print(f"  [{row+1}/{len(chosen)}] {stem}...")
        img     = load_rgb(tif)
        H, W    = img.shape[:2]
        preds   = predict_tile(model, tif)
        gt_polys = load_gt_polygons(meta)

        prec, rec, f1 = area_f1(preds, gt_polys, W, H)
        results.append({"tile": stem, "n_preds": len(preds), "n_gt_polys": len(gt_polys),
                        "precision": prec, "recall": rec, "f1": f1})
        print(f"    preds={len(preds)}  gt_polys={len(gt_polys)}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

        # Panel 1: RGB + predictions
        ax1 = axes[row][0]
        ax1.imshow(img)
        for _, r in preds.iterrows():
            ax1.add_patch(mpatches.Rectangle(
                (r.xmin, r.ymin), r.xmax-r.xmin, r.ymax-r.ymin,
                lw=0.5, edgecolor="#00ff88", facecolor="none", alpha=0.8))
        ax1.set_title(f"{stem}\n{len(preds)} preds  F1={f1:.3f}", fontsize=7)
        ax1.axis("off")

        # Panel 2: GT polygons
        ax2 = axes[row][1]
        ax2.imshow(img)
        for poly in gt_polys[:500]:
            try:
                xs, ys = poly.exterior.xy
                ax2.fill(xs, ys, alpha=0.25, fc="#4488ff", ec="#4488ff", lw=0.4)
            except Exception:
                pass
        ax2.set_title(f"GT polygons ({len(gt_polys)})  P={prec:.3f}  R={rec:.3f}", fontsize=7)
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"\nSaved → {OUT_PATH}")

    df_res = pd.DataFrame(results)
    print(f"\nResults:\n{df_res.to_string(index=False)}")
    print(f"\nMean area F1: {df_res['f1'].mean():.3f}")
    print(f"Mean precision: {df_res['precision'].mean():.3f}")
    print(f"Mean recall: {df_res['recall'].mean():.3f}")


if __name__ == "__main__":
    main()
