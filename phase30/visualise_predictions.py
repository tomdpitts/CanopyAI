#!/usr/bin/env python3
"""
visualise_predictions.py — Side-by-side GT vs model predictions on 7 hardcoded
OAM-TCD holdout tiles. Polygon outlines only (no fills) so the underlying
RGB stays readable.

Predictions are classified using the same instance-matching logic as benchmark.py:
  • TP     — pred IoU≥0.5 against an unmatched cat=2 tree GT (green outline)
  • FP     — pred matched nothing, AND does not lie inside a canopy region (red)
  • IGNORE — pred IoP≥0.5 inside a cat=1 canopy blob (grey, dashed)
  • FN     — cat=2 tree GT that no pred matched (orange dashed, on GT panel only)

GT panel shows cat=2 trees in solid yellow, cat=1 canopy blobs in cyan dashed.

Usage:
    python phase30/visualise_predictions.py \\
        --models kunqi_epoch6 phase22_B_L4 \\
        --names  kunqi_epoch6 phase22_B_L4 \\
        --output-root benchmark_results_holdout \\
        --save-dir phase30/viz_predictions
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import (
    _load_predictions,
    _parse_coco_annotations,
    _rasterize,
    _seg_to_polygons,
)
import geopandas as gpd
from shapely.geometry import box as shapely_box

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOLDOUT_DIR = REPO_ROOT / "data" / "tcd" / "images" / "data" / "tcd" / "val"

# Hardcoded for reproducibility — spread across the 0..438 holdout range,
# plus three extra in the 100-164 band for closer side-by-side inspection.
TILE_INDICES = [105, 111, 117, 123, 128, 132, 145]

IOU_THRESH = 0.5
IOP_THRESH = 0.5


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _safe(g):
    if g is None or g.is_empty:
        return g
    return g if g.is_valid else make_valid(g)


def _iou(a, b):
    a, b = _safe(a), _safe(b)
    if a is None or b is None or not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def _iop(pred, region):
    pred, region = _safe(pred), _safe(region)
    if pred is None or region is None or pred.area <= 0 or not pred.intersects(region):
        return 0.0
    return pred.intersection(region).area / pred.area


def _classify(preds, tree_gts, canopy_gts,
              iou_thresh=IOU_THRESH, iop_thresh=IOP_THRESH):
    """
    Greedy-match preds against tree GTs by IoU≥iou_thresh.  Unmatched preds
    that lie inside any canopy GT at IoP≥iop_thresh are 'IGNORE'.
    Returns (pred_classes, unmatched_tree_indices).
    """
    # Sort preds by score (descending) when available — not strictly needed for
    # visualisation, but keeps it consistent with COCOeval's matching priority.
    matched = [False] * len(tree_gts)
    classes = []
    for p, _score in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(tree_gts):
            if matched[j]:
                continue
            iou = _iou(p, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            matched[best_j] = True
            classes.append("TP")
            continue
        if canopy_gts and any(_iop(p, c) >= iop_thresh for c in canopy_gts):
            classes.append("IGNORE")
        else:
            classes.append("FP")
    fn_indices = [j for j, m in enumerate(matched) if not m]
    return classes, fn_indices


# ── Plotting ──────────────────────────────────────────────────────────────────

def _load_pred_bboxes(pred_path):
    """Read det_bbox property out of the geojson (foxtrot's raw DeepForest boxes,
    before SAM polygonisation). Returns a list of shapely box Polygons aligned
    with the polygon list from `_load_predictions` (same order, same length).
    Predictions without a det_bbox property → None at that index."""
    if not pred_path.exists():
        return []
    gdf = gpd.read_file(str(pred_path))
    if gdf.empty or "det_bbox" not in gdf.columns:
        return []
    out = []
    for raw in gdf["det_bbox"]:
        if raw is None:
            out.append(None); continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                out.append(None); continue
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            out.append(shapely_box(*raw))
        else:
            out.append(None)
    return out


def _iter_polygons(geom):
    """Yield only Polygon parts of a geometry, recursing into MultiPolygon /
    GeometryCollection. make_valid() can return mixed-type collections (lines
    on shared edges, etc.) — those are skipped silently."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
        return
    geoms = getattr(geom, "geoms", None)
    if geoms is None:
        return
    for g in geoms:
        yield from _iter_polygons(g)


def _poly_to_path(poly):
    """Convert a shapely Polygon/MultiPolygon/GeometryCollection to a matplotlib Path."""
    verts, codes = [], []
    for p in _iter_polygons(poly):
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            if len(xy) < 3:
                continue
            verts.extend(xy.tolist())
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    if not verts:
        return None
    return MplPath(verts, codes)


def _draw_polys(ax, polys, edgecolor, linestyle="-", linewidth=1.2, alpha=0.95):
    for poly in polys:
        if poly is None or poly.is_empty:
            continue
        path = _poly_to_path(poly)
        if path is None:
            continue
        ax.add_patch(PathPatch(path, edgecolor=edgecolor, facecolor="none",
                               linestyle=linestyle, linewidth=linewidth, alpha=alpha))


def _draw_panel(ax, img, polys_groups, title):
    """polys_groups: list of (polys, edgecolor, linestyle, linewidth)."""
    ax.imshow(img)
    for polys, edge, ls, lw in polys_groups:
        _draw_polys(ax, polys, edgecolor=edge, linestyle=ls, linewidth=lw)
    ax.set_title(title, fontsize=9, family="monospace", loc="left")
    ax.set_xticks([]); ax.set_yticks([])


def _load_tile(holdout_dir, idx):
    stem = f"tcd_val_tile_{idx}"
    tif = holdout_dir / f"{stem}.tif"
    meta_path = holdout_dir / f"{stem}_meta.json"
    with rasterio.open(tif) as src:
        img = src.read([1, 2, 3])
    img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img / max(1, img.max()) * 255).astype(np.uint8)
    meta = json.load(open(meta_path))
    H, W = int(meta["height"]), int(meta["width"])

    tree_gts, canopy_gts = [], []
    for cat, seg, *_ in _parse_coco_annotations(meta):
        for p in _seg_to_polygons(seg, H, W):
            (tree_gts if cat == 2 else canopy_gts).append(p)
    return stem, img, tree_gts, canopy_gts, H, W


def visualise_tile(idx, model_dirs, model_names, holdout_dir, save_dir,
                   score_thresh):
    stem, img, tree_gts, canopy_gts, H, W = _load_tile(holdout_dir, idx)

    n_cols = 1 + len(model_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6.5))
    if n_cols == 1:
        axes = [axes]

    # GT panel
    _draw_panel(
        axes[0], img,
        [
            (canopy_gts, "#00d8ff", "--", 1.0),
            (tree_gts,   "#ffd400", "-",  1.2),
        ],
        title=f"{stem}  —  GT  (trees={len(tree_gts)}, canopy={len(canopy_gts)})",
    )

    # Pre-rasterise GT once for the binary metric (shared across all models)
    gt_mask_bin = _rasterize(tree_gts + canopy_gts, H, W)

    # Model panels
    for ax, mdir, name in zip(axes[1:], model_dirs, model_names):
        pred_path = mdir / f"{stem}_canopyai.geojson"

        # Distinguish "model predicted nothing" (file present, empty) from
        # "inference hasn't run yet" (file missing). The viz would otherwise
        # render identically for both cases.
        if not pred_path.exists():
            ax.imshow(img)
            ax.text(0.5, 0.5, "NO PREDICTION FILE\n(inference not run yet)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=18, color="white", weight="bold",
                    bbox=dict(facecolor="#e53935", alpha=0.85, edgecolor="white",
                              boxstyle="round,pad=0.6"))
            ax.set_title(f"{name}\n(missing: {pred_path.name})",
                         fontsize=9, family="monospace", loc="left")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        preds = _load_predictions(pred_path, H, W, score_thresh)
        bboxes = _load_pred_bboxes(pred_path)
        if len(bboxes) != len(preds):
            # Score-thresh dropped some preds; can't align — disable bbox drawing.
            bboxes = [None] * len(preds)
        classes, fn_idx = _classify(preds, tree_gts, canopy_gts)

        tp_polys     = [preds[i][0] for i, c in enumerate(classes) if c == "TP"]
        fp_polys     = [preds[i][0] for i, c in enumerate(classes) if c == "FP"]
        ignore_polys = [preds[i][0] for i, c in enumerate(classes) if c == "IGNORE"]
        fn_polys     = [tree_gts[j] for j in fn_idx]

        tp_bboxes     = [bboxes[i] for i, c in enumerate(classes) if c == "TP" and bboxes[i] is not None]
        fp_bboxes     = [bboxes[i] for i, c in enumerate(classes) if c == "FP" and bboxes[i] is not None]
        ignore_bboxes = [bboxes[i] for i, c in enumerate(classes) if c == "IGNORE" and bboxes[i] is not None]

        # ── Instance metrics @ IoU=0.5 (iscrowd-ignored preds excluded from denom)
        n_tp, n_fp, n_fn = len(tp_polys), len(fp_polys), len(fn_polys)
        inst_p  = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
        inst_r  = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
        inst_f1 = 2 * inst_p * inst_r / (inst_p + inst_r) if (inst_p + inst_r) else 0.0

        # ── Binary semantic-seg metrics (per-tile pixel counts)
        pred_mask_bin = _rasterize([p for p, _ in preds], H, W)
        tp_pix = int(np.count_nonzero(pred_mask_bin & gt_mask_bin))
        fp_pix = int(np.count_nonzero(pred_mask_bin & ~gt_mask_bin))
        fn_pix = int(np.count_nonzero(~pred_mask_bin & gt_mask_bin))
        tn_pix = pred_mask_bin.size - tp_pix - fp_pix - fn_pix
        bin_iou = tp_pix / (tp_pix + fp_pix + fn_pix) if (tp_pix + fp_pix + fn_pix) else 0.0
        bin_f1  = (2 * tp_pix) / (2 * tp_pix + fp_pix + fn_pix) if (2 * tp_pix + fp_pix + fn_pix) else 0.0
        bin_acc = (tp_pix + tn_pix) / pred_mask_bin.size

        title = (
            f"{name}\n"
            f"inst @IoU0.5  P={inst_p:.2f}  R={inst_r:.2f}  F1={inst_f1:.2f}   "
            f"(TP={n_tp} FP={n_fp} IGN={len(ignore_polys)} FN={n_fn})\n"
            f"binary       IoU={bin_iou:.3f}  F1={bin_f1:.3f}  Acc={bin_acc:.3f}"
        )
        _draw_panel(
            ax, img,
            [
                # Raw DeepForest bboxes (under polygons): thin same-colour dashed lines
                (ignore_bboxes, "#9aa0a6", "--", 0.6),
                (fp_bboxes,     "#e53935", "--", 0.7),
                (tp_bboxes,     "#22c55e", "--", 0.7),
                # SAM-refined polygons on top: thicker solid lines
                (fn_polys,      "#ff9500", "--", 1.0),   # missed trees from GT
                (ignore_polys,  "#9aa0a6", ":",  1.0),   # in-canopy → ignored
                (fp_polys,      "#e53935", "-",  1.4),   # genuine false positives
                (tp_polys,      "#22c55e", "-",  1.4),   # matched trees
            ],
            title=title,
        )

    legend = [
        Line2D([0], [0], color="#ffd400", lw=1.5, label="GT tree (cat=2)"),
        Line2D([0], [0], color="#00d8ff", lw=1.5, ls="--", label="GT canopy (cat=1)"),
        Line2D([0], [0], color="#22c55e", lw=1.5, label="TP polygon  (solid=SAM, dashed=raw bbox)"),
        Line2D([0], [0], color="#e53935", lw=1.5, label="FP polygon  (solid=SAM, dashed=raw bbox)"),
        Line2D([0], [0], color="#9aa0a6", lw=1.5, ls=":", label="IGNORE  (pred IoP≥0.5 in canopy → ignored by mAP50)"),
        Line2D([0], [0], color="#ff9500", lw=1.5, ls="--", label="FN  (unmatched tree GT)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_path = save_dir / f"{stem}.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--models", nargs="+", required=True,
                   help="Folder names (subdirs of --output-root) containing *_canopyai.geojson.")
    p.add_argument("--names", nargs="+", default=None,
                   help="Display names (default: same as --models).")
    p.add_argument("--holdout-dir", default=str(DEFAULT_HOLDOUT_DIR))
    p.add_argument("--output-root", default="benchmark_results_holdout",
                   help="Root containing per-model prediction folders.")
    p.add_argument("--save-dir", default="phase30/viz_predictions",
                   help="Where PNGs are written.")
    p.add_argument("--pred-score-thresh", type=float, default=0.0)
    p.add_argument("--tile-indices", nargs="+", type=int, default=None,
                   help=f"Override hardcoded indices (default: {TILE_INDICES}).")
    return p.parse_args()


def main():
    args = parse_args()
    names = args.names or list(args.models)
    if len(names) != len(args.models):
        print("❌ --names must match --models length"); sys.exit(1)

    holdout_dir = Path(args.holdout_dir)
    output_root = Path(args.output_root)
    save_dir    = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    indices = args.tile_indices if args.tile_indices is not None else TILE_INDICES
    model_dirs = [output_root / m for m in args.models]
    missing = [d for d in model_dirs if not d.is_dir()]
    if missing:
        print(f"❌ Missing prediction folder(s): {missing}"); sys.exit(1)

    print(f"Tiles: {indices}")
    print(f"Models: {list(zip(names, [str(d) for d in model_dirs]))}")
    print(f"Saving to: {save_dir}\n")

    for idx in indices:
        try:
            path = visualise_tile(idx, model_dirs, names, holdout_dir, save_dir,
                                  args.pred_score_thresh)
            print(f"  ✓ tile {idx:>3}  →  {path}")
        except FileNotFoundError as e:
            print(f"  ✗ tile {idx:>3}  missing file: {e}")


if __name__ == "__main__":
    main()
