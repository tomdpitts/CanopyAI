#!/usr/bin/env python3
"""
benchmark_tcd.py — Evaluate multiple TCD models on the Restor public dataset.

Computes proper AP (greedy matching, sorted by confidence) and plots PR curves.
Single unified metric: all GT annotations (individual trees cat=2 + canopy blobs cat=1)
are pooled together. A prediction is TP if IoP ≥ iop_thresh against any GT polygon.

Supported model specifiers:
  weecology          → weecology/deepforest NEON pretrained   (foxtrot.py → SAM polygon)
  detectree2         → Detectree2 Mask R-CNN baseline         (infer.py   → polygon)
  <path>.pth         → ShadowConditionedDeepForest checkpoint  (foxtrot.py → SAM polygon)

Usage:
    # Step 1: download any missing checkpoints
    modal volume get canopyai-deepforest-checkpoints \\
        /phase16_A_baseline/deepforest_final.pth phase16_A_baseline.pth

    # Step 2: run benchmark
    python benchmark_tcd.py \\
        --models weecology detectree2 phase16_A_baseline.pth phase16_D_shadow_channel.pth \\
        --names  weecology detectree2 phase16_A phase16_D \\
        --shadow-model solar/shadow_regression/output/shadow_model_combined_best.pth \\
        --output-root benchmark_results

    # Step 3: re-evaluate only (skip inference)
    python benchmark_tcd.py --skip-inference --output-root benchmark_results \\
        --models weecology detectree2 phase16_A_baseline.pth phase16_D_shadow_channel.pth \\
        --names  weecology detectree2 phase16_A phase16_D
"""

import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rasterio.transform import from_bounds
from shapely.affinity import affine_transform
from shapely.geometry import box as shapely_box, Polygon
from shapely.validation import make_valid
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).parent))

# ── Biome grouping ─────────────────────────────────────────────────────────────
# Maps raw biome_name → display group label.
# Edit this dict freely — groups are applied at evaluation time, no re-inference needed.
# Any biome_name not listed here falls into "Other".
BIOME_GROUPS = {
    # Arid / dryland (model's target domain)
    "West Sudanian savanna":               "Arid / dryland",
    "East Sudanian savanna":               "Arid / dryland",
    "Zambezian and Mopane woodlands":      "Arid / dryland",
    "Chilean matorral":                    "Arid / dryland",
    "Sechura desert":                      "Arid / dryland",
    "Low Monte":                           "Arid / dryland",
    "Sonoran desert":                      "Arid / dryland",
    "Central Mexican matorral":            "Arid / dryland",
    "Gulf of Oman desert and semi-desert": "Arid / dryland",
    # African tropical forest
    "Northern Zanzibar-Inhambane coastal forest mosaic": "African tropical forest",
    "Southern Zanzibar-Inhambane coastal forest mosaic": "African tropical forest",
    "Eastern Guinean forests":             "African tropical forest",
    "Western Guinean lowland forests":     "African tropical forest",
    "Nigerian lowland forests":            "African tropical forest",
    "Northern Congolian forest-savanna mosaic": "African tropical forest",
    "Central African mangroves":           "African tropical forest",
    "East African montane forests":        "African tropical forest",
    "Maputaland coastal forest mosaic":    "African tropical forest",
    # European forest
    "Central European mixed forests":      "European forest",
    "Sarmatic mixed forests":              "European forest",
    "East European forest steppe":         "European forest",
    "Western European broadleaf forests":  "European forest",
    "Balkan mixed forests":                "European forest",
    "Celtic broadleaf forests":            "European forest",
    "Caucasus mixed forests":              "European forest",
    "Atlantic mixed forests":              "European forest",
    "Baltic mixed forests":                "European forest",
    "English Lowlands beech forests":      "European forest",
    "Pannonian mixed forests":             "European forest",
    "Rodope montane mixed forests":        "European forest",
    "Alps conifer and mixed forests":      "European forest",
    # SE Asian & Pacific rainforest
    "Luzon rain forests":                  "SE Asian & Pacific rainforest",
    "Mindanao-Eastern Visayas rain forests": "SE Asian & Pacific rainforest",
    "Sulawesi lowland rain forests":       "SE Asian & Pacific rainforest",
    "Lesser Sundas deciduous forests":     "SE Asian & Pacific rainforest",
    "Greater Negros-Panay rain forests":   "SE Asian & Pacific rainforest",
    "Sumatran lowland rain forests":       "SE Asian & Pacific rainforest",
    "Eastern Java-Bali rain forests":      "SE Asian & Pacific rainforest",
    "Palawan rain forests":                "SE Asian & Pacific rainforest",
    "Timor and Wetar deciduous forests":   "SE Asian & Pacific rainforest",
    "Taiwan subtropical evergreen forests": "SE Asian & Pacific rainforest",
    "Mizoram-Manipur-Kachin rain forests": "SE Asian & Pacific rainforest",
    "Hispaniolan moist forests":           "SE Asian & Pacific rainforest",
    "Tongan tropical moist forests":       "SE Asian & Pacific rainforest",
    # N. American forest
    "Sierra Nevada forests":               "N. American forest",
    "Southern Great Lakes forests":        "N. American forest",
    "Eastern Great Lakes lowland forests": "N. American forest",
    "Southeastern mixed forests":          "N. American forest",
    "Northeastern coastal forests":        "N. American forest",
    "Upper Midwest forest-savanna transition": "N. American forest",
    "North Central Rockies forests":       "N. American forest",
    "Appalachian-Blue Ridge forests":      "N. American forest",
    "New England-Acadian forests":         "N. American forest",
    "Central U.S. hardwood forests":       "N. American forest",
    "Southeastern conifer forests":        "N. American forest",
    "Willamette Valley forests":           "N. American forest",
    "Northern California coastal forests": "N. American forest",
    # Latin American forest
    "Madeira-Tapajós moist forests":       "Latin American forest",
    "Cerrado":                             "Latin American forest",
    "Mato Grosso seasonal forests":        "Latin American forest",
    "Petén-Veracruz moist forests":        "Latin American forest",
    "Bahia coastal forests":               "Latin American forest",
    "Serra do Mar coastal forests":        "Latin American forest",
    "Peruvian Yungas":                     "Latin American forest",
    "Uatuma-Trombetas moist forests":      "Latin American forest",
    "Ucayali moist forests":               "Latin American forest",
    "Bolivian montane dry forests":        "Latin American forest",
    "Northwestern Andean montane forests": "Latin American forest",
}

def biome_group(biome_name):
    return BIOME_GROUPS.get(biome_name, "Other")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--names",  nargs="+", required=True)
    p.add_argument("--tiles",  nargs="+", default=None,
                   help="Restrict to these tile stems (e.g. tcd_tile_3 tcd_tile_7). "
                        "Applied to both inference and evaluation so comparisons are fair.")
    p.add_argument("--tcd-dir", default="data/tcd/images/data/tcd/raw")
    p.add_argument("--shadow-model",
                   default="solar/shadow_regression/output/shadow_model_combined_best.pth")
    p.add_argument("--output-root", default="benchmark_results")
    p.add_argument("--iop-thresh", type=float, default=0.4,
                   help="IoP threshold for TP: pred area inside any GT polygon (default 0.4)")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip tiles that already have a geojson in the output dir")
    p.add_argument("--abs-luma-max", type=float, default=None,
                   help="Shadow map luma ceiling passed to foxtrot (None → default 71)")
    p.add_argument("--pr-save", default="benchmark_pr_curves.png",
                   help="Path to save PR curve figure")
    return p.parse_args()


def model_type(spec):
    s = spec.lower()
    if s in ("weecology", "weecology/deepforest", "default"):
        return "weecology"
    if s == "detectree2":
        return "detectree2"
    if s == "segformer":
        return "segformer"
    return "checkpoint"


# ── Inference ─────────────────────────────────────────────────────────────────

def run_foxtrot(model_spec, mtype, image_path, out_dir, shadow_model, abs_luma_max=None):
    cmd = [sys.executable, "foxtrot.py",
           "--image_path", str(image_path),
           "--output_dir", str(out_dir),
           "--shadow_model", str(shadow_model),
           "--no_viz"]
    if mtype == "checkpoint":
        cmd += ["--deepforest_model", model_spec]
    if abs_luma_max is not None:
        cmd += ["--abs_luma_max", str(abs_luma_max)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"      ⚠  foxtrot failed: {r.stderr[-300:]}")
        return False
    return True


def run_segformer(image_path, out_dir):
    cmd = [sys.executable, "infer_segformer.py",
           "--image_path", str(image_path),
           "--output_dir", str(out_dir)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"      ⚠  infer_segformer failed: {r.stderr[-300:]}")
        return False
    return True


def run_detectree2(image_path, out_dir):
    # infer.py writes: {infer_root}/tiles_pred/{stem}_chips/predictions_geo/{stem}_merged.geojson
    infer_root = Path(out_dir) / "_detectree2_working"
    infer_root.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "infer.py",
           "--image_path", str(image_path),
           "--output_root", str(infer_root)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"      ⚠  infer.py failed: {r.stderr[-500:]}")
        return False
    stem = image_path.stem
    merged = infer_root / "tiles_pred" / f"{stem}_chips" / "predictions_geo" / f"{stem}_merged.geojson"
    if not merged.exists():
        print(f"      ⚠  merged GeoJSON not found for {stem} (expected: {merged})")
        return False
    shutil.copy(merged, out_dir / f"{stem}_canopyai.geojson")
    return True


def run_inference(model_spec, mtype, tcd_dir, out_dir, shadow_model, abs_luma_max=None,
                  skip_existing=False, tile_filter=None):
    tifs = sorted(Path(tcd_dir).glob("*.tif"))
    if tile_filter is not None:
        tifs = [t for t in tifs if t.stem in tile_filter]
    ok = 0
    skipped = 0
    for tif in tifs:
        out_file = Path(out_dir) / f"{tif.stem}_canopyai.geojson"
        if skip_existing and out_file.exists():
            skipped += 1
            ok += 1
            continue
        print(f"    {tif.name} ... ", end="", flush=True)
        if mtype == "detectree2":
            success = run_detectree2(tif, out_dir)
        elif mtype == "segformer":
            success = run_segformer(tif, out_dir)
        else:
            success = run_foxtrot(model_spec, mtype, tif, out_dir, shadow_model, abs_luma_max)
        print("✓" if success else "✗")
        ok += success
    if skipped:
        print(f"  {skipped} tiles skipped (already exist)")
    print(f"  {ok}/{len(tifs)} successful")
    return ok > 0


# ── GT loading ────────────────────────────────────────────────────────────────

def load_gt(meta, tif_path=None):
    """
    Parse COCO annotations from meta dict.
    Returns (gt_polys_world, gt_cats) — world-coordinate shapely polygons and
    integer category labels (1=canopy, 2=tree).
    """
    from shapely.geometry import shape
    try:
        from pycocotools import mask as mask_utils
        import rasterio.features
    except ImportError:
        mask_utils = None

    coco = meta.get("coco_annotations", [])
    if isinstance(coco, str):
        coco = json.loads(coco)
    if not coco:
        return [], []

    width, height = meta["width"], meta["height"]
    bounds = meta["bounds"]
    tf = from_bounds(*bounds, width=width, height=height)
    coeffs = [tf.a, tf.b, tf.d, tf.e, tf.c, tf.f]

    polys, cats = [], []
    for ann in coco:
        cat = int(ann.get("category_id", 2))
        segs = ann.get("segmentation")
        if not segs:
            continue
        raw = []
        if isinstance(segs, list) and isinstance(segs[0], list):
            try:
                coords = np.array(segs[0]).reshape(-1, 2)
                raw = [Polygon(coords)]
            except Exception:
                pass
        elif isinstance(segs, dict) and "counts" in segs and mask_utils:
            try:
                rle = mask_utils.frPyObjects(segs, segs["size"][0], segs["size"][1])
                mask = mask_utils.decode(rle)
                raw = [shape(g) for g, v in rasterio.features.shapes(
                    mask.astype(np.uint8), mask > 0) if v == 1]
            except Exception:
                pass
        for p in raw:
            if not p.is_valid:
                p = make_valid(p)
            if p.is_valid and p.area > 0:
                polys.append(affine_transform(p, coeffs))
                cats.append(cat)
    return polys, cats


def score_column(gdf):
    """Return the confidence-score column name from a foxtrot/detectree2 GeoJSON."""
    for col in ("deepforest_score", "score", "Confidence", "confidence"):
        if col in gdf.columns:
            return col
    return None


def load_predictions(pred_path, meta):
    """
    Load predictions from *_canopyai.geojson, transform to world coords.
    Returns (polys, scores, box_polys) — world-coordinate shapely geometries,
    confidence floats, and axis-aligned bounding box polygons.
    """
    gdf = gpd.read_file(str(pred_path))
    if gdf.empty:
        return [], [], []

    width, height = meta["width"], meta["height"]
    bounds = meta["bounds"]
    tf = from_bounds(*bounds, width=width, height=height)
    coeffs = [tf.a, tf.b, tf.d, tf.e, tf.c, tf.f]

    # Detect pixel-space coords: foxtrot writes the world CRS label but pixel coordinates.
    # Check the centroid of the first geometry — if it falls within pixel bounds it needs
    # transforming regardless of what CRS is declared.
    first_centroid = gdf.geometry.iloc[0].centroid
    in_pixel_space = (0 <= first_centroid.x <= width) and (0 <= first_centroid.y <= height)
    if gdf.crs is None or in_pixel_space:
        # Pixel-space output (foxtrot/weecology) — transform to world coords
        gdf["geometry"] = gdf.geometry.apply(lambda g: affine_transform(g, coeffs))
    gdf = gdf.set_crs(meta["crs"], allow_override=True)

    sc = score_column(gdf)
    scores = gdf[sc].values.astype(float) if sc else np.ones(len(gdf))

    # Bounding-box polygons (for box-IoU comparison if needed)
    box_polys = [shapely_box(*g.bounds) if g else None for g in gdf.geometry]

    return list(gdf.geometry), list(scores), box_polys


# ── AP computation (greedy, standard VOC/COCO) ────────────────────────────────

def _valid(g):
    if g is None or g.is_empty:
        return g
    if not g.is_valid:
        g = make_valid(g)
    return g


def _iou(a, b):
    a, b = _valid(a), _valid(b)
    inter = a.intersection(b).area
    return inter / (a.union(b).area + 1e-10) if inter > 0 else 0.0


def _iop(a, b):
    """Intersection over Prediction area."""
    a, b = _valid(a), _valid(b)
    if a.area <= 0:
        return 0.0
    return a.intersection(b).area / a.area


def compute_ap(
    all_pred_polys,   # list of (shapely geom, confidence, tile_id)
    gt_by_tile,       # {tile_id: [geom, ...]}  — all GT regardless of category
    thresh,           # IoP threshold for TP
):
    """
    Proper greedy AP computation using IoP against all GT polygons (trees + canopy pooled).
    Sorts all predictions by confidence descending, greedily assigns each to the best
    unmatched GT in its tile.

    Returns (ap, precisions, recalls, confidences)
    """
    n_gt_total = sum(len(gts) for gts in gt_by_tile.values())
    if n_gt_total == 0:
        return 0.0, np.array([]), np.array([]), np.array([])

    # Build per-tile spatial index and matched-GT tracking
    tile_trees = {}
    tile_gt    = {}
    for tid, geoms in gt_by_tile.items():
        if not geoms:
            continue
        tile_trees[tid] = STRtree(geoms)
        tile_gt[tid]    = {"geoms": geoms, "matched": [False] * len(geoms)}

    sorted_preds = sorted(all_pred_polys, key=lambda x: -x[1])

    tp_list, fp_list, conf_list = [], [], []

    for pred_geom, conf, tid in sorted_preds:
        if pred_geom is None or pred_geom.is_empty:
            continue
        conf_list.append(conf)

        gt_info = tile_gt.get(tid)
        if gt_info is None:
            tp_list.append(0); fp_list.append(1)
            continue

        try:
            cand_idx = tile_trees[tid].query(pred_geom, predicate="intersects")
        except TypeError:
            cand_idx = [j for j, g in enumerate(gt_info["geoms"])
                        if pred_geom.intersects(g)]

        best_iop = 0.0
        best_j   = -1
        for j in cand_idx:
            if gt_info["matched"][j]:
                continue
            iop = _iop(pred_geom, gt_info["geoms"][j])
            if iop > best_iop:
                best_iop = iop
                best_j   = j

        if best_iop >= thresh and best_j >= 0:
            gt_info["matched"][best_j] = True
            tp_list.append(1); fp_list.append(0)
        else:
            tp_list.append(0); fp_list.append(1)

    if not tp_list:
        return 0.0, np.array([]), np.array([]), np.array([])

    cum_tp = np.cumsum(tp_list)
    cum_fp = np.cumsum(fp_list)
    prec   = cum_tp / (cum_tp + cum_fp)
    rec    = cum_tp / n_gt_total
    conf   = np.array(conf_list)

    prec_env = prec.copy()
    for i in range(len(prec_env) - 2, -1, -1):
        prec_env[i] = max(prec_env[i], prec_env[i + 1])

    ap = max(float(np.trapz(prec_env, rec)) if len(rec) > 1 else 0.0, 0.0)
    return ap, prec_env, rec, conf


def f1_optimal_point(prec, rec, conf):
    """Return (precision, recall, f1, threshold) at the F1-maximising point."""
    if len(prec) == 0:
        return 0.0, 0.0, 0.0, 0.0
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = np.argmax(f1)
    return float(prec[idx]), float(rec[idx]), float(f1[idx]), float(conf[idx])


# ── Evaluation orchestration ───────────────────────────────────────────────────

def evaluate_model(name, out_dir, tcd_dir, iop_thresh, tile_filter=None):
    """
    Loads all predictions + GT for a model.
    GT pools all annotations (cat=1 canopy + cat=2 trees) into one flat list per tile.

    Returns (all_preds, gt_by_tile, tile_biomes) where:
      all_preds   — [(geom, score, tile_id)]
      gt_by_tile  — {tile_id: [geom, ...]}   (all GT, no category split)
      tile_biomes — {tile_id: biome_name}
    """
    tcd_dir = Path(tcd_dir)
    pred_files = sorted(Path(out_dir).glob("*_canopyai.geojson"))
    if tile_filter is not None:
        pred_files = [p for p in pred_files if p.stem.replace("_canopyai", "") in tile_filter]
    if not pred_files:
        print(f"  ⚠  No predictions in {out_dir}")
        return None

    gt_by_tile  = {}
    all_preds   = []
    tile_biomes = {}

    for i, pred_path in enumerate(pred_files):
        stem = pred_path.stem.replace("_canopyai", "")
        meta_path = tcd_dir / f"{stem}_meta.json"
        tif_path  = tcd_dir / f"{stem}.tif"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        if "width" not in meta or "height" not in meta:
            import rasterio
            if tif_path.exists():
                with rasterio.open(tif_path) as src:
                    meta["width"], meta["height"] = src.width, src.height
            else:
                continue

        tile_biomes[i] = meta.get("biome_name", "unknown")

        gt_polys, _ = load_gt(meta, tif_path)   # pool all categories
        gt_by_tile[i] = gt_polys

        polys, scores, _ = load_predictions(pred_path, meta)
        for geom, score in zip(polys, scores):
            all_preds.append((geom, score, i))

    return all_preds, gt_by_tile, tile_biomes


# ── PR curve plotting ──────────────────────────────────────────────────────────

def plot_pr_curves(model_results, iop_thresh, save_path):
    """
    model_results: {name: (ap, prec, rec, conf)}
    Single unified metric: IoP ≥ iop_thresh against all GT (trees + canopy).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    for ci, (name, res) in enumerate(model_results.items()):
        if res is None:
            continue
        ap, prec, rec, conf = res
        if len(rec) == 0:
            continue
        colour = colours[ci % len(colours)]
        ax.plot(rec, prec, label=f"{name}  AP={ap:.3f}", color=colour, lw=1.8)
        p_opt, r_opt, f1_opt, c_opt = f1_optimal_point(prec, rec, conf)
        ax.scatter([r_opt], [p_opt], color=colour, s=60, zorder=5)
        ax.annotate(f"F1={f1_opt:.2f}@{c_opt:.2f}",
                    xy=(r_opt, p_opt), xytext=(r_opt + 0.02, p_opt - 0.05),
                    fontsize=7, color=colour)

    ax.set_xlabel("Recall",    fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(f"All GT (trees + canopy)  IoP ≥ {iop_thresh}", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Restor TCD Benchmark — Precision-Recall Curves", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 PR curves saved to {save_path}")
    plt.close(fig)


# ── Results table ──────────────────────────────────────────────────────────────

def print_table(model_results, iop_thresh):
    print("\n" + "═" * 65)
    print("  BENCHMARK — Restor TCD  |  AP + F1-optimal operating point")
    print(f"  All GT (trees + canopy pooled)  |  IoP ≥ {iop_thresh:.2f}  (* = F1-optimal)")
    print("═" * 65)

    hdr = f"  {'Model':<18}  {'AP':>6}  {'Prec*':>6}  {'Rec*':>6}  {'F1*':>6}  {'Thr*':>6}"
    sep = "  " + "-"*18 + "  " + "  ".join(["-"*6]*5)
    row = "  {:<18}  {:6.3f}  {:6.3f}  {:6.3f}  {:6.3f}  {:6.3f}"
    print(hdr); print(sep)

    for name, res in model_results.items():
        if res is None:
            print(f"  {name:<18}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}")
            continue
        ap, prec, rec, conf = res
        p_opt, r_opt, f1_opt, c_opt = f1_optimal_point(prec, rec, conf)
        print(row.format(name, ap, p_opt, r_opt, f1_opt, c_opt))

    print("═" * 65 + "\n")


# ── Per-biome results table ────────────────────────────────────────────────────

def compute_biome_results(all_preds, gt_by_tile, tile_biomes, iop_thresh, use_groups=False):
    """
    Splits preds/GT by biome (or biome group) and computes AP+F1 per slice.
    Returns {biome_label: {"result": (ap, p, r, c), "n_tiles": int}}
    """
    from collections import defaultdict

    label_fn = biome_group if use_groups else (lambda b: b)

    label_to_tiles = defaultdict(set)
    for tid, bname in tile_biomes.items():
        label_to_tiles[label_fn(bname)].add(tid)

    results = {}
    for label, tile_ids in sorted(label_to_tiles.items()):
        sub_preds = [(g, s, t) for g, s, t in all_preds   if t in tile_ids]
        sub_gt    = {t: v      for t, v      in gt_by_tile.items() if t in tile_ids}
        if not sub_gt:
            continue
        ap, p, r, c = compute_ap(sub_preds, sub_gt, iop_thresh)
        results[label] = {"result": (ap, p, r, c), "n_tiles": len(tile_ids)}
    return results


def print_biome_table(model_biome_results, iop_thresh, use_groups=False):
    level = "Group" if use_groups else "Biome"
    print("\n" + "═" * 75)
    print(f"  PER-{level.upper()} BREAKDOWN  |  IoP ≥ {iop_thresh:.2f}  (trees + canopy pooled)")
    print("═" * 75)

    hdr = f"  {'Model':<18}  {level:<30}  {'N':>4}  {'AP':>7}  {'F1*':>7}"
    sep = "  " + "-"*18 + "  " + "-"*30 + "  " + "  ".join(["-"*4, "-"*7, "-"*7])
    print(hdr); print(sep)

    for name, biome_results in model_biome_results.items():
        if biome_results is None:
            continue
        for label, res in sorted(biome_results.items()):
            ap, p, r, c = res["result"]
            _, _, f1, _ = f1_optimal_point(p, r, c)
            n = res["n_tiles"]
            print(f"  {name:<18}  {label:<30}  {n:>4}  {ap:>7.3f}  {f1:>7.3f}")
        print(sep)

    print("═" * 75 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if len(args.models) != len(args.names):
        print("❌ --models and --names must match in length"); sys.exit(1)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Inference ──────────────────────────────────────────────────────
    for model_spec, name in zip(args.models, args.names):
        mtype = model_type(model_spec)
        out_dir = output_root / name
        out_dir.mkdir(exist_ok=True)

        if args.skip_inference:
            continue

        print(f"\n{'─'*60}")
        print(f"  Inference: {name}  [{mtype}]")
        print(f"{'─'*60}")
        run_inference(model_spec, mtype, args.tcd_dir, out_dir, args.shadow_model,
                      abs_luma_max=args.abs_luma_max,
                      skip_existing=args.skip_existing,
                      tile_filter=set(args.tiles) if args.tiles else None)

    # ── Step 2: AP computation ────────────────────────────────────────────────
    model_results       = {}
    model_biome_results = {}

    for model_spec, name in zip(args.models, args.names):
        out_dir = output_root / name
        print(f"\n  Evaluating {name} ...")

        data = evaluate_model(name, out_dir, args.tcd_dir, args.iop_thresh,
                              tile_filter=set(args.tiles) if args.tiles else None)
        if data is None:
            model_results[name] = None
            model_biome_results[name] = None
            continue

        all_preds, gt_by_tile, tile_biomes = data

        ap, prec, rec, conf = compute_ap(all_preds, gt_by_tile, args.iop_thresh)
        model_results[name] = (ap, prec, rec, conf)
        model_biome_results[name] = compute_biome_results(
            all_preds, gt_by_tile, tile_biomes, args.iop_thresh)

        print(f"    AP={ap:.3f}")

    # ── Step 3: Output ────────────────────────────────────────────────────────
    print_table(model_results, args.iop_thresh)
    print_biome_table(model_biome_results, args.iop_thresh, use_groups=False)
    print_biome_table(model_biome_results, args.iop_thresh, use_groups=True)
    plot_pr_curves(model_results, args.iop_thresh, args.pr_save)

    # Save raw AP values
    summary = {}
    for name, res in model_results.items():
        if res is None:
            summary[name] = None
            continue
        ap, prec, rec, conf = res
        summary[name] = {
            "ap": float(ap),
            "f1_optimal": dict(zip(["precision", "recall", "f1", "threshold"],
                                   f1_optimal_point(prec, rec, conf))),
        }
    with open(output_root / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to {output_root}/benchmark_summary.json")


if __name__ == "__main__":
    main()
