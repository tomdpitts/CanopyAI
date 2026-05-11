#!/usr/bin/env python3
"""
Side-by-side comparison: epoch 0 (initial phase22) vs phase25 final checkpoint.

Shows GT polygons + predictions from both checkpoints on the same 5 val tiles
to verify training is actually improving things.

Usage:
    source venv310/bin/activate
    python phase25/eval_compare.py
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

ROOT     = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deepforest_custom"))

VAL_CSV    = ROOT / "phase24/phase24_tcd_val.csv"
TCD_RAW    = ROOT / "data/tcd/images/data/tcd/raw"
CKPT_E0    = ROOT / "phase25_epoch00.ckpt"          # Lightning ckpt
PTH_FINAL  = ROOT / "phase25_final.pth"             # End-of-train state_dict
OUT_PATH   = ROOT / "benchmark_results/phase25_compare.png"

N_TILES      = 5
SEED         = 42
SCORE_THRESH = 0.25
PATCH_SIZE   = 400
PATCH_OVERLAP = 0.05


def load_model_from_lightning_ckpt(ckpt_path):
    """Build a DeepForest model + load state dict from a Lightning .ckpt."""
    from deepforest import main as df_main
    df = df_main.deepforest()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = {k.replace("model.", "", 1): v
          for k, v in ckpt["state_dict"].items()
          if k.startswith("model.")}
    df.model.load_state_dict(sd, strict=False)
    df.model.eval()
    df.config["score_thresh"] = SCORE_THRESH
    return df


def load_model_from_pth(pth_path):
    """Build a DeepForest model + load state dict from a plain .pth."""
    from deepforest import main as df_main
    df = df_main.deepforest()
    sd = torch.load(pth_path, map_location="cpu")
    # Strip "model." prefix if present
    sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
          for k, v in sd.items()}
    df.model.load_state_dict(sd, strict=False)
    df.model.eval()
    df.config["score_thresh"] = SCORE_THRESH
    return df


def predict_tile(df, tif_path):
    result = df.predict_tile(
        path=str(tif_path),
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
    )
    if result is None:
        return pd.DataFrame(columns=["xmin","ymin","xmax","ymax","score"])
    return result.reset_index(drop=True)


def load_gt_polygons(meta_path):
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
                from rasterio.features import shapes as rio_shapes
                mask = mask_utils.decode(seg)
                for geom, val in rio_shapes(mask.astype(np.uint8), mask > 0):
                    if val == 1:
                        polys.append(shape(geom))
        except Exception:
            pass
    return polys


def area_f1(pred_boxes, gt_polys):
    if len(pred_boxes) == 0 and not gt_polys:
        return 1.0, 1.0, 1.0
    if len(pred_boxes) == 0 or not gt_polys:
        return 0.0, 0.0, 0.0
    pred_polys = [shapely_box(r.xmin, r.ymin, r.xmax, r.ymax) for _, r in pred_boxes.iterrows()]
    pred_union = unary_union(pred_polys)
    gt_union   = unary_union(gt_polys)
    inter = pred_union.intersection(gt_union).area
    p = inter / pred_union.area if pred_union.area > 0 else 0.0
    r = inter / gt_union.area   if gt_union.area   > 0 else 0.0
    f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
    return p, r, f1


def load_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read([1, 2, 3]).transpose(1, 2, 0)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr.astype(float) - lo) / max(hi - lo, 1), 0, 1)


def draw_preds(ax, img, preds, gt_polys, title):
    ax.imshow(img)
    for poly in gt_polys[:600]:
        try:
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=0.18, fc="#4488ff", ec="#4488ff", lw=0.3)
        except Exception:
            pass
    for _, r in preds.iterrows():
        ax.add_patch(mpatches.Rectangle(
            (r.xmin, r.ymin), r.xmax-r.xmin, r.ymax-r.ymin,
            lw=0.4, edgecolor="#00ff88", facecolor="none", alpha=0.75))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_val = pd.read_csv(VAL_CSV)
    tiles  = df_val["image_path"].unique()
    rng    = random.Random(SEED)
    chosen = []
    for p in rng.sample(list(tiles), min(40, len(tiles))):
        stem = Path(p).stem
        tif  = TCD_RAW / f"{stem}.tif"
        meta = TCD_RAW / f"{stem}_meta.json"
        if tif.exists() and tif.stat().st_size > 0 and meta.exists():
            chosen.append((stem, tif, meta))
        if len(chosen) == N_TILES:
            break

    print(f"Loading epoch 0 model from {CKPT_E0.name}...")
    df_e0 = load_model_from_lightning_ckpt(CKPT_E0)
    print(f"Loading final model from {PTH_FINAL.name}...")
    df_final = load_model_from_pth(PTH_FINAL)

    fig, axes = plt.subplots(N_TILES, 2, figsize=(16, 5 * N_TILES))
    results = []
    for row, (stem, tif, meta) in enumerate(chosen):
        print(f"  [{row+1}/{N_TILES}] {stem}...")
        img      = load_rgb(tif)
        gt_polys = load_gt_polygons(meta)

        preds_e0    = predict_tile(df_e0, tif)
        preds_final = predict_tile(df_final, tif)
        p0, r0, f0  = area_f1(preds_e0, gt_polys)
        p1, r1, f1  = area_f1(preds_final, gt_polys)
        results.append({"tile": stem,
                        "n_pred_e0": len(preds_e0), "p_e0": p0, "r_e0": r0, "f1_e0": f0,
                        "n_pred_final": len(preds_final), "p_final": p1, "r_final": r1, "f1_final": f1})
        print(f"    e0:    n={len(preds_e0)}  P={p0:.3f}  R={r0:.3f}  F1={f0:.3f}")
        print(f"    final: n={len(preds_final)}  P={p1:.3f}  R={r1:.3f}  F1={f1:.3f}")

        draw_preds(axes[row][0], img, preds_e0, gt_polys,
                   f"{stem}  EPOCH 0\nn={len(preds_e0)}  P={p0:.3f}  R={r0:.3f}  F1={f0:.3f}")
        draw_preds(axes[row][1], img, preds_final, gt_polys,
                   f"{stem}  FINAL (epoch ~10)\nn={len(preds_final)}  P={p1:.3f}  R={r1:.3f}  F1={f1:.3f}")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"\nSaved → {OUT_PATH}")

    df_res = pd.DataFrame(results)
    print(f"\nResults:\n{df_res.to_string(index=False)}")
    print(f"\nMean F1  — epoch 0: {df_res['f1_e0'].mean():.3f}   final: {df_res['f1_final'].mean():.3f}")
    print(f"Mean prec — epoch 0: {df_res['p_e0'].mean():.3f}   final: {df_res['p_final'].mean():.3f}")
    print(f"Mean rec  — epoch 0: {df_res['r_e0'].mean():.3f}   final: {df_res['r_final'].mean():.3f}")
    print(f"Mean #preds — epoch 0: {df_res['n_pred_e0'].mean():.0f}   final: {df_res['n_pred_final'].mean():.0f}")


if __name__ == "__main__":
    main()
