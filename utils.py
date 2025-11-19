"""
utils.py — Shared logic for downloading TCD tiles with streaming.

Used by both main.py (inference) and train.py (data preparation).
"""

from pathlib import Path
import numpy as np
import requests
import rasterio
from rasterio.transform import from_bounds
from datasets import load_dataset
from pathlib import Path
from datasets import load_dataset
import requests
import json
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, shape
from shapely.validation import make_valid
from shapely.affinity import affine_transform
from shapely.strtree import STRtree
from pycocotools import mask as mask_utils
from rasterio.transform import from_bounds

import rasterio
from rasterio.transform import from_bounds
from shapely.affinity import affine_transform

from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import json, cv2
from detectree2.preprocessing.tiling import tile_data



def is_australia(x):
    return (
        -44 <= x["lat"] <= -10 and
        112 <= x["lon"] <= 154
    )
aus_tiles = [
1207, 4347, 4159, 4893, 4406, 2100, 1104, 3956, 2859, 3684, 5001, 3469,
5012, 4660, 536, 4315, 4506, 3624, 4127, 4963, 423, 1703, 3016, 4643,
922, 4221, 4955, 4909, 3219, 1671, 195, 4923, 4556, 4086, 1969, 3611,
4336, 2581, 1033, 314, 2491, 4720, 4421, 5005, 4435, 2654, 62, 4593,
5057, 1612, 1417, 1278, 2403, 2270, 367, 1339, 1117, 4507, 4040, 577,
439, 2888, 4326, 1875, 760, 678, 3456, 4108, 1029, 2515, 4996, 876,
4639, 4933, 2031, 750, 1248, 743, 2293, 3277, 3875, 1077
]

def is_really_australia(x):
    return x["image_id"] in aus_tiles

rangeland = [
    "West Sudanian savanna",
    "East Sudanian savanna",
    "Northern Congolian forest-savanna mosaic",
    "Northern mixed grasslands",
    "Central and Southern mixed grasslands",
    "Pontic steppe",
    "East European forest steppe",
    "Snake-Columbia shrub steppe",
    "Chilean matorral",
    "Central Mexican matorral",
    "Sechura desert",
    "Sonoran desert",
    "Gulf of Oman desert and semi-desert",
    "Low Monte",
    "Central Andean puna",
    "Central Andean dry puna",
    "Zambezian and Mopane woodlands"
]

arid_rangeland = [
    "Sechura desert",
    "Sonoran desert",
    "Gulf of Oman desert and semi-desert",
    "Chilean matorral",
    "Central Mexican matorral",
    "Low Monte",
    "Zambezian and Mopane woodlands",
    "East Sudanian savanna",
    "West Sudanian savanna"
]

def is_rangeland(x):
    return x["biome_name"] in arid_rangeland



def download_tcd_tiles_streaming(save_dir: Path, max_images: int = 3):
    """
    Download up to `max_images` TCD tiles (filtered by is_rangeland),
    save each as a GeoTIFF plus a sidecar *_meta.json file containing
    all COCO + geo metadata needed later for training/validation.

    Output per tile:
      save_dir / f"tcd_tile_{i}.tif"
      save_dir / f"tcd_tile_{i}_meta.json"
    """
    print("📦 Loading TCD dataset in streaming mode...")
    ds = load_dataset("restor/tcd", split="train", streaming=True)

    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for image_info in ds:
        if not is_rangeland(image_info):
            print("Nope!")
            continue

        if count >= max_images:
            break

        image_id = image_info["image_id"]
        print(f"📸 Downloading Rangeland tile {count}: {image_id}")

        img_path = save_dir / f"tcd_tile_{count}.tif"
        meta_path = save_dir / f"tcd_tile_{count}_meta.json"

        # ------------------
        # Save image (.tif)
        # ------------------
        img = np.array(image_info["image"])
        h, w = img.shape[:2]
        crs = image_info["crs"]
        bounds = image_info["bounds"]

        transform = from_bounds(*bounds, width=w, height=h)

        with rasterio.open(
            img_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=3,
            dtype=img.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for b in range(3):
                dst.write(img[:, :, b], b + 1)

        # ------------------
        # Save metadata JSON
        # ------------------
        # We store all the important geo/COCO info in a JSON-serializable form
        meta = {
            "image_id": image_id,
            "bounds": bounds,
            "crs": str(crs),
            "width": w,
            "height": h,
            "coco_annotations": image_info.get("coco_annotations", []),
            # keep some extra context if you like:
            "biome": image_info.get("biome"),
            "country": image_info.get("country"),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        print(f"✅ Saved Australian tile → {img_path}")
        print(f"📄 Saved metadata → {meta_path}")

        count += 1

    print(f"🎯 Finished downloading {count} TCD tiles.")

def has_geodata(tif_path: str | Path) -> bool:
    """Return True if GeoTIFF has valid CRS and affine transform."""
    with rasterio.open(tif_path) as ds:
        # CRS and transform must both be defined and not identity
        has_crs = ds.crs is not None
        has_transform = ds.transform != rasterio.Affine.identity()
        return has_crs and has_transform
    

def compute_final_metric(scores, thresh, n_pred, n_gt):
    scores = np.asarray(scores, dtype=float)
    n_tp = int(np.sum(scores >= thresh))
    precision = n_tp / n_pred if n_pred else 0.0
    recall = n_tp / n_gt if n_gt else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    mean_overlap = float(np.mean(scores)) if len(scores) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_overlap": mean_overlap,
        "n_tp": n_tp,
        "n_pred": n_pred,
        "n_gt": n_gt
    }



def clean_validate_predictions_vs_tcd_segments(
    pred_geojson_path,
    image_tif,
    iou_thresh_tree=0.5,
    iop_thresh_canopy=0.7,
):
    """
    Validate Detectree2 predictions against TCD 'segments' polygons.
    Robust to all COCO segmentation formats (Polygon, RLE string/list),
    and complex geometry types (MultiPolygon, GeometryCollection).
    Returns: (metrics_all, pred_gdf, gt_gdf, (scores_trees, scores_canopy), coco_annotations)
    """

    # ---------------------------
    # Helper: flatten geometry parts
    # ---------------------------
    def polygon_parts(geom):
        """Return list of Polygon parts from any shapely geometry."""
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, Polygon):
            return [geom]
        if isinstance(geom, MultiPolygon):
            return list(geom.geoms)
        if isinstance(geom, GeometryCollection):
            out = []
            for g in geom.geoms:
                out.extend(polygon_parts(g))
            return out
        return []

    # ---------------------------
    # Helper: pixel → world transform
    # ---------------------------
    def pixel_to_world_geom(geom, transform):
        """Apply rasterio Affine transform to shapely geometry."""
        coeffs = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]
        return affine_transform(geom, coeffs)

    # ---------------------------
    # Metric helpers
    # ---------------------------
    def iou(a, b):
        inter = a.intersection(b).area
        if inter <= 0.0:
            return 0.0
        union = a.union(b).area
        return inter / union if union > 0 else 0.0

    def iop(a, b):
        inter = a.intersection(b).area
        return inter / a.area if a.area > 0 else 0.0

    def compute_metrics(scores, thresh, n_pred, n_gt):
        scores = np.asarray(scores, dtype=float)
        n_tp = int(np.sum(scores >= thresh))
        precision = n_tp / n_pred if n_pred else 0.0
        recall = n_tp / n_gt if n_gt else 0.0
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        mean_score = float(np.mean(scores)) if scores.size > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mean_overlap": mean_score,
            "n_tp": n_tp,
        }

    # ---------------------------
    # 1) Load predictions
    # ---------------------------
    print("📂 Loading Detectree2 predictions ...")
    pred = gpd.read_file(pred_geojson_path)
    print(f"  → {len(pred)} predicted polygons")

    # ---------------------------
    # 2) Parse COCO annotations
    # ---------------------------
    coco_annotations = image_tif.get("coco_annotations", [])
    if isinstance(coco_annotations, str):
        coco_annotations = json.loads(coco_annotations)
    if not isinstance(coco_annotations, list) or not coco_annotations:
        print(f"❌ coco_annotations: {coco_annotations}")
        return None, None, None, None, None

    gt_polys_px = []
    gt_cats = []

    for ann in coco_annotations:
        segs = ann.get("segmentation", None)
        cat = int(ann.get("category_id"))  # 1=canopy, 2=tree

        if not segs:
            continue

        polys = []

        # Case A: Polygon list (e.g. [[x1,y1,...]])
        if isinstance(segs, list) and isinstance(segs[0], list):
            try:
                coords = np.array(segs[0], dtype=float).reshape(-1, 2)
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                polys = polygon_parts(poly)
            except Exception as e:
                print(f"⚠️ Skipping invalid polygon segmentation: {e}")
                polys = []

        # Case B: RLE dict (either string or list counts)
        elif isinstance(segs, dict) and "counts" in segs and "size" in segs:
            try:
                mask = mask_utils.decode(segs)  # works for both string and list counts
                shapes = rasterio.features.shapes(mask.astype(np.uint8), mask > 0)
                polys = [shape(geom) for geom, val in shapes if val == 1]
            except Exception as e:
                print(f"⚠️ Failed to decode RLE segmentation: {e}")
                polys = []

        else:
            print(f"⚠️ Unknown segmentation format in annotation id={ann.get('id')} — skipping.")
            polys = []

        for p in polys:
            if p.is_valid and p.area > 0:
                gt_polys_px.append(p)
                gt_cats.append(cat)

    print(f" GT Categories — Canopy: {gt_cats.count(1)}, Trees: {gt_cats.count(2)}")

    # ---------------------------
    # 3) Pixel → world transform
    # ---------------------------
    width, height = image_tif["width"], image_tif["height"]
    bounds = image_tif["bounds"]
    transform = from_bounds(*bounds, width=width, height=height)

    gt_world_parts = [pixel_to_world_geom(p, transform) for p in gt_polys_px]
    gt = gpd.GeoDataFrame({"geometry": gt_world_parts, "category": gt_cats}, crs=image_tif["crs"])
    print(f"  → {len(gt)} ground-truth polygons (after normalization)")

    # ---------------------------
    # 4) CRS alignment
    # ---------------------------
    if pred.crs != gt.crs:
        print(f"Aligning CRS: {pred.crs} → {gt.crs}")
        pred = pred.to_crs(gt.crs)

    # ---------------------------
    # 5) Overlap evaluation via STRtree
    # ---------------------------
    gt_geoms = list(gt.geometry)
    gt_cats_arr = np.asarray(gt["category"], dtype=int)
    gt_tree = STRtree(gt_geoms)

    n_pred = len(pred)
    scores_trees = np.zeros(n_pred, dtype=float)
    scores_canopy = np.zeros(n_pred, dtype=float)

    for i, p in enumerate(pred.geometry):
        if p is None or p.is_empty or not p.is_valid:
            continue
        try:
            cand_idx = gt_tree.query(p, predicate="intersects")
        except TypeError:
            cand_idx = [j for j, ggt in enumerate(gt_geoms) if p.intersects(ggt)]
        if len(cand_idx) == 0:
            continue

        best_tree, best_canopy = 0.0, 0.0
        for j in cand_idx:
            ggt = gt_geoms[j]
            if not p.intersects(ggt):
                continue
            if gt_cats_arr[j] == 2:
                best_tree = max(best_tree, iou(p, ggt))
            else:
                best_canopy = max(best_canopy, iop(p, ggt))
        scores_trees[i] = best_tree
        scores_canopy[i] = best_canopy

    # ---------------------------
    # 6) Metrics (+F1)
    # ---------------------------
    n_gt = len(gt)
    n_gt_trees = int(np.sum(gt_cats_arr == 2))
    n_gt_canopy = int(np.sum(gt_cats_arr == 1))

    metrics_trees = compute_metrics(scores_trees, iou_thresh_tree, n_pred, n_gt_trees)
    metrics_canopy = compute_metrics(scores_canopy, iop_thresh_canopy, n_pred, n_gt_canopy)

    metrics_all = {
        "trees": metrics_trees,
        "canopy": metrics_canopy,
        "n_pred": n_pred,
        "n_gt_total": n_gt,
        "n_gt_trees": n_gt_trees,
        "n_gt_canopy": n_gt_canopy,
    }

    # ---------------------------
    # 7) Summary
    # ---------------------------
    print("\n📊 Validation Results:")
    print("  🌳 Trees (IoU):")
    for k, v in metrics_trees.items():
        print(f"    {k:12s}: {v:.3f}" if isinstance(v, float) else f"    {k:12s}: {v}")
    print("  🌿 Canopy (IoP):")
    for k, v in metrics_canopy.items():
        print(f"    {k:12s}: {v:.3f}" if isinstance(v, float) else f"    {k:12s}: {v}")
    print(f"\n  Total predictions: {n_pred}")
    print(f"  Total GT polygons: {n_gt} (Trees: {n_gt_trees}, Canopy: {n_gt_canopy})")

    return metrics_all, pred, gt, (scores_trees, scores_canopy), coco_annotations

def visualize_validation_results(pred, gt, ious, coco_anns=None, 
                                 iou_thresh_tree=0.5, iop_thresh_canopy=0.7,
                                 site_path=None, rgb_path=None,
                                 tile_name=None, image_id=None):
    """
    Visualize Detectree2 vs TCD polygons over RGB base image.
    """

    # === 1. Output path ===
    out_dir = Path(site_path) / "overlays_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path_constructor = ["validate"]
    if tile_name:
        path_constructor.append(tile_name)
    if image_id:
        path_constructor.append(f"tcd{image_id}")

    out_path = out_dir / f"{'_'.join(path_constructor)}.png"

    # === 2. Load background image ===
    img, extent = None, None
    if rgb_path and Path(rgb_path).exists():
        with rasterio.open(rgb_path) as src:
            img = src.read([1, 2, 3])
            img = np.moveaxis(img, 0, -1)
            img = (img - img.min()) / (img.max() - img.min() + 1e-9)
            extent = [src.bounds.left, src.bounds.right,
                      src.bounds.bottom, src.bounds.top]

    # === 3. Plot setup ===
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Detectree2 vs TCD — IoU ≥ {iou_thresh_tree}, IoP ≥ {iop_thresh_canopy}")
    ax.set_aspect("equal")

    if img is not None:
        ax.imshow(img, extent=extent, origin="upper")

    # === 4. Draw predictions ===

    def draw_pred_outline(ax, geom, color):
        from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        if geom.is_empty: 
            return
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=1.2, alpha=0.9)
        elif isinstance(geom, MultiPolygon):
            for sub in geom.geoms:
                x, y = sub.exterior.xy
                ax.plot(x, y, color=color, linewidth=1.2, alpha=0.9)
        elif isinstance(geom, GeometryCollection):
            for sub in geom.geoms:
                draw_pred_outline(ax, sub, color)


    for i, p in enumerate(pred.geometry):
        if not p.is_valid or p.is_empty:
            continue

        # --- Extract per-prediction score ---
        score_tree = score_canopy = 0.0
        if isinstance(ious, tuple) and len(ious) == 2:
            scores_trees, scores_canopy = ious
            if i < len(scores_trees):
                score_tree = scores_trees[i]
            if i < len(scores_canopy):
                score_canopy = scores_canopy[i]
        elif isinstance(ious, list):
            if i < len(ious):
                score_tree = ious[i]  # fallback

        # --- Pick color based on which test passes ---
        if score_tree >= iou_thresh_tree:
            color = "#00F0FF"      # bright teal  → good tree (IoU)
        elif score_canopy >= iop_thresh_canopy:
            color = "#00FF9D"      # neon green  → good canopy (IoP)
        else:
            color = "#FF8800"      # orange      → low-overlap / FP

        draw_pred_outline(ax, p, color)

    # === 5. Draw ground-truth crowns & canopy ===
    for idx, g in enumerate(gt.geometry):
        if not g.is_valid or g.is_empty:
            continue
        # Match color by category (1=tree, 2=canopy)
        cat = None
        if coco_anns and idx < len(coco_anns):
            cat = coco_anns[idx].get("category_id", 1)
        color = "#C266FF" if cat == 1 else "#0a20ad"  # purple vs dark-blue
        if isinstance(g, Polygon):
            geoms = [g]
        elif isinstance(g, MultiPolygon):
            geoms = list(g.geoms)
        elif isinstance(g, GeometryCollection):
            geoms = [geom for geom in g.geoms if isinstance(geom, (Polygon, MultiPolygon))]
        else:
            geoms = []

        for geom in geoms:
            if isinstance(geom, Polygon):
                x, y = geom.exterior.xy
                ax.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
                ax.plot(x, y, color=color, linewidth=0.8, alpha=0.9)
            elif isinstance(geom, MultiPolygon):
                for sub in geom.geoms:
                    x, y = sub.exterior.xy
                    ax.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
                    ax.plot(x, y, color=color, linewidth=0.8, alpha=0.9)

    # === 6. Legend ===
    import matplotlib.patches as mpatches
    legend_elems = [
        mpatches.Patch(color="#00F0FF", label="Tree TP (IoU ≥ 0.5)"),
        mpatches.Patch(color="#00FF9D", label="Canopy TP (IoP ≥ 0.7)"),
        mpatches.Patch(color="#FF8800", label="False Positive / Low Overlap"),
        mpatches.Patch(color="#C266FF", label="Ground Truth — Canopy"),
        mpatches.Patch(color="#0a20ad", label="Ground Truth — Tree")
    ]

    ax.legend(handles=legend_elems, loc="lower right", frameon=True, fontsize=8)

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close(fig)

    print(f"🖼️  Saved validation overlay → {out_path}")

    # === 7. Also save pure Ground Truth visualization for comparison ===

    # fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
    # ax_gt.set_title(f"TCD Ground Truth (image_id={image_id})")
    # ax_gt.set_aspect("equal")

    # if img is not None:
    #     ax_gt.imshow(img, extent=extent, origin="upper")

    # for idx, g in enumerate(gt.geometry):
    #     if g.is_empty:
    #         continue

    #     # Category colouring
    #     cat = None
    #     if coco_anns and idx < len(coco_anns):
    #         cat = coco_anns[idx].get("category_id", 1)
    #     color = "#C266FF" if cat == 1 else "#0a20ad"  # purple=canopy, blue=tree

    #     # Handle all geometry types safely
    #     if isinstance(g, Polygon):
    #         geoms = [g]
    #     elif isinstance(g, MultiPolygon):
    #         geoms = list(g.geoms)
    #     elif isinstance(g, GeometryCollection):
    #         geoms = [geom for geom in g.geoms if isinstance(geom, (Polygon, MultiPolygon))]
    #     else:
    #         geoms = []

    #     for geom in geoms:
    #         if isinstance(geom, Polygon):
    #             x, y = geom.exterior.xy
    #             ax_gt.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
    #             ax_gt.plot(x, y, color=color, linewidth=0.8, alpha=0.9)
    #         elif isinstance(geom, MultiPolygon):
    #             for sub in geom.geoms:
    #                 x, y = sub.exterior.xy
    #                 ax_gt.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
    #                 ax_gt.plot(x, y, color=color, linewidth=0.8, alpha=0.9)

    # ax_gt.set_xlabel("Easting")
    # ax_gt.set_ylabel("Northing")

    # out_gt_path = out_path.with_name(out_path.stem + "_groundtruth.png")
    # plt.tight_layout()
    # plt.savefig(out_gt_path, dpi=250)
    # plt.close(fig_gt)

    # print(f"🗺️  Saved pure ground-truth overlay → {out_gt_path}")
    
def filter_raw_predictions(pred_dir: Path, score_thresh: float = 0.8, overwrite=True) -> None:
    """
    Filter raw Detectron2 prediction JSONs by confidence score.

    If overwrite=True, original JSONs are replaced in place so
    Detectree2.project_to_geojson() automatically picks them up.
    """

    pred_dir = Path(pred_dir)
    json_files = sorted(pred_dir.glob("Prediction_*.json"))
    if not json_files:
        raise FileNotFoundError(f"❌ No Detectron2 predictions found in {pred_dir}")

    for fpath in json_files:
        with open(fpath, "r") as f:
            preds = json.load(f)

        before = len(preds)
        preds = [p for p in preds if p.get("score", 0) >= score_thresh]
        after = len(preds)

        if overwrite:
            out_path = fpath
        else:
            out_path = fpath.with_name(f"{fpath.stem}_filtered_{score_thresh}.json")

        with open(out_path, "w") as f:
            json.dump(preds, f)

        print(f"📊 {fpath.name}: kept {after}/{before} predictions (≥ {score_thresh})")

    print(f"✅ Filtering complete — overwrote {len(json_files)} files at ≥ {score_thresh}.")
    

def load_tcd_meta_for_tile(img_path: Path):
    """
    Load TCD metadata for a given .tif tile, if available.

    Returns:
      dict with keys: width, height, bounds, crs, coco_annotations, ...
      or None if no _meta.json exists (in which case we skip validation).
    """
    meta_path = img_path.with_name(img_path.stem + "_meta.json")
    if not meta_path.exists():
        print(f"ℹ️  No metadata found for {img_path.name} (no _meta.json) — skipping validation.")
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


# ============================================================
#   COCO → GeoDataFrame
# ============================================================

def coco_meta_to_geodf(meta: dict) -> gpd.GeoDataFrame:
    width = meta["width"]
    height = meta["height"]
    bounds = meta["bounds"]
    crs_str = meta["crs"]
    coco_annotations = meta.get("coco_annotations", [])

    if isinstance(coco_annotations, str):
        coco_annotations = json.loads(coco_annotations)
    if not isinstance(coco_annotations, list):
        coco_annotations = []

    transform = from_bounds(*bounds, width=width, height=height)

    def px_to_world(geom):
        coeffs = [
            transform.a, transform.b,
            transform.d, transform.e,
            transform.c, transform.f,
        ]
        return affine_transform(geom, coeffs)

    world_geoms = []
    cats = []

    for ann in coco_annotations:
        seg = ann.get("segmentation")
        cat = ann.get("category_id")

        if not seg:
            continue

        polys = []

        # Polygon list
        if isinstance(seg, list) and isinstance(seg[0], list):
            try:
                coords = np.array(seg[0], dtype=float).reshape(-1, 2)
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                polys = [poly]
            except Exception:
                pass

        # RLE mask
        elif isinstance(seg, dict) and "counts" in seg:
            try:
                mask = mask_utils.decode(seg).astype(np.uint8)
                shapes = rasterio.features.shapes(mask, mask > 0)
                polys = [shape(g) for g, val in shapes if val == 1]
            except Exception:
                pass

        for p in polys:
            if p.is_valid and p.area > 0:
                world_geoms.append(px_to_world(p))
                cats.append(cat)

    return gpd.GeoDataFrame({"geometry": world_geoms, "category": cats}, crs=crs_str)


# -------------------------------------------------------------------
# Tiling
# -------------------------------------------------------------------

def tile_all_tcd_tiles(raw_dir: Path, tiles_root: Path,
                       buffer: int = 30,
                       tile_width: int = 40,
                       tile_height: int = 40,
                       threshold: float = 0.0):
    """
    For each .tif + _meta.json in raw_dir:

      - load metadata
      - convert COCO segs to polygons (GeoDataFrame)
      - tile with Detectree2 tile_data (with crowns)

    Output: one chips folder per tile under tiles_root.
    """
    tif_files = sorted(raw_dir.glob("tcd_tile_*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No tcd_tile_*.tif files found in {raw_dir}")

    tiles_root.mkdir(parents=True, exist_ok=True)

    print(f"📸 Found {len(tif_files)} TCD tiles to tile")

    for img_path in tif_files:
        stem = img_path.stem  # e.g. "tcd_tile_0"
        meta_path = raw_dir / f"{stem}_meta.json"

        if not meta_path.exists():
            print(f"⚠️  Missing metadata {meta_path}, skipping {img_path}")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Convert COCO segs → polygons
        crowns_gdf = coco_meta_to_geodf(meta)
        if crowns_gdf.empty:
            print(f"⚠️  No polygons found in metadata for {stem}, skipping.")
            continue

        chips_dir = tiles_root / f"{stem}_chips"
        print(f"🧩 Tiling {img_path.name} → {chips_dir}")

        tile_data(
            img_path=str(img_path),
            out_dir=str(chips_dir),
            buffer=buffer,
            tile_width=tile_width,
            tile_height=tile_height,
            crowns=crowns_gdf,
            threshold=threshold,
            mode="rgb",
        )

        print(f"✅ Finished tiling → {chips_dir}")