"""
utils.py — Shared logic for downloading TCD tiles with streaming.

Used by both infer.py (inference) and train.py (data preparation).
"""

from pathlib import Path
import numpy as np
import requests
import rasterio
from rasterio.transform import from_bounds
import json
import geopandas as gpd
import rasterio.features
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, shape
from shapely.validation import make_valid
from shapely.affinity import affine_transform
from shapely.strtree import STRtree
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt
import cv2



# ── Shadow probability map ─────────────────────────────────────────────────────
# Algorithm: shadow_tuner Slot 2 (tuned for BRU/WON aerials).
# Public API (backward compatible):
#   compute_shadow_normalization_stats(image_rgb, angle)  →  (norm_dg, None, None)
#   generate_shadow_map(img_rgb, angle, dg_scale, ...)    →  H×W float32
#
# Key design choices vs the old algorithm:
#   - Short + long range DG (8-35px and 35-150px) weighted equally.
#   - Max-brightness ray accumulation with anti-sun directional penalty
#     (replaces old Gaussian-weighted average of anti−sun difference).
#   - Absolute luma gate (abs_luma_max=71): pixels brighter than this cannot
#     be shadows — kills bare-ground false positives entirely.
#   - Lateral edge penalty (Sobel): suppresses road edges running parallel
#     to the shadow direction.
#   - Fixed sigmoid centre (otsu_ctr=0.35) and activation floor (0.2):
#     no per-tile dynamic threshold, so maps are globally consistent.
#   - Speckle removal (min 150px²): removes isolated noise blobs.

# Slot 2 defaults
_SM_D_SHORT_MIN  = 8
_SM_D_SHORT_MAX  = 35
_SM_BLUR_SHORT   = 2.0
_SM_D_LONG_MIN   = 35
_SM_D_LONG_MAX   = 150
_SM_BLUR_LONG    = 4.0
_SM_SHORT_WEIGHT = 0.5
_SM_ABS_LUMA_MAX = 71
_SM_OTSU_CTR     = 0.35
_SM_SIGMOID_K    = 12.0
_SM_ACT_FLOOR    = 0.2
_SM_SPECKLE_MIN  = 150


def _sm_vectors(shadow_angle_deg: float):
    """Return (adx, ady, sdx, sdy): anti-shadow and shadow unit-vector components."""
    ang = np.radians(shadow_angle_deg)
    sx, sy = np.sin(ang), -np.cos(ang)
    return -sx, -sy, sx, sy


def _sm_valid_mask(gray: np.ndarray) -> np.ndarray:
    """Binary valid-pixel mask: excludes black (nodata=0) and white (nodata=255) fill."""
    valid = ((gray > 15) & (gray < 250)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.erode(valid, k).astype(np.float32)


def _sm_dg(gray, valid, adx, ady, sdx, sdy, d_min, d_max, n_steps, blur):
    """
    Directional gradient: max brightness seen toward sun minus an anti-sun penalty.
    Uses np.maximum accumulation (not Gaussian-weighted average) so the signal
    fires as soon as any bright object is seen in the sun direction.
    """
    h, w = gray.shape
    if d_min >= d_max:
        return np.zeros((h, w), dtype=np.float32)
    ds = np.unique(
        np.round(np.linspace(d_min, d_max, max(n_steps, 2))).astype(int)
    ).astype(float)
    sun_max  = gray.copy()
    anti_max = gray.copy()
    for d in ds:
        Ma = np.float32([[1, 0, round(d * adx)], [0, 1, round(d * ady)]])
        Ms = np.float32([[1, 0, round(d * sdx)], [0, 1, round(d * sdy)]])
        sun_side  = cv2.warpAffine(gray,  Ma, (w, h), borderMode=cv2.BORDER_REFLECT)
        anti_side = cv2.warpAffine(gray,  Ms, (w, h), borderMode=cv2.BORDER_REFLECT)
        va = cv2.warpAffine(valid, Ma, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        vs = cv2.warpAffine(valid, Ms, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sun_side  *= (va > 0.5)
        anti_side *= (vs > 0.5)
        sun_max  = np.maximum(sun_max,  sun_side)
        anti_max = np.maximum(anti_max, anti_side)
    base_diff = np.clip(sun_max - gray, 0, None)
    anti_diff = np.clip(anti_max - gray, 0, None)
    diff = np.clip(base_diff - anti_diff * 0.4, 0, None)
    if blur > 0:
        diff = cv2.GaussianBlur(diff, (0, 0), sigmaX=float(blur))
    return diff * valid


def _sm_remove_speckle(prob: np.ndarray, min_area: int, threshold: float = 0.2) -> np.ndarray:
    """Remove connected components smaller than min_area pixels."""
    if min_area <= 0:
        return prob
    binary = (prob >= max(threshold, 0.01)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary)
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == lab] = 1
    return prob * mask.astype(np.float32)


def compute_shadow_normalization_stats(
    image_rgb: np.ndarray,
    shadow_angle_deg: float,
    max_side: int = 2000,
) -> tuple:
    """
    Compute norm_dg: 99.5th-percentile short-range directional gradient over
    the full image (or domain mosaic).  Call ONCE before tiling and pass the
    result to every generate_shadow_map call so that all tiles share the same
    normalization scale.

    Returns (norm_dg, None, None) — 3-tuple for backward compatibility with
    callers that unpack as (dg_s, dark_s, otsu_s).  Only the first element
    is used by generate_shadow_map; the other two are ignored.
    """
    h, w = image_rgb.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        nh, nw = int(h * scale), int(w * scale)
        img_s = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        img_s = image_rgb

    gray  = cv2.cvtColor(img_s.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    valid = _sm_valid_mask(gray)
    mask  = valid > 0

    if mask.sum() < 100:
        return 30.0, None, None

    adx, ady, sdx, sdy = _sm_vectors(shadow_angle_deg)
    dg = _sm_dg(gray, valid, adx, ady, sdx, sdy,
                _SM_D_SHORT_MIN, _SM_D_SHORT_MAX, 8, _SM_BLUR_SHORT)
    norm_dg = max(float(np.percentile(dg[mask], 99.5)), 30.0)
    return norm_dg, None, None


def generate_shadow_map(
    img_rgb: np.ndarray,
    shadow_angle_deg: float,
    dg_scale: float | None = None,    # norm_dg from compute_shadow_normalization_stats()
    dark_scale: float | None = None,  # unused — kept for API compatibility
    otsu_ctr: float | None = None,    # if None uses Slot 2 default (0.35)
) -> np.ndarray:
    """
    Generate a shadow probability map using the Slot 2 algorithm.

    Args:
        img_rgb:          H×W×3 uint8 numpy array (RGB).
        shadow_angle_deg: Sun azimuth in degrees (0=North, 90=East, CW).
        dg_scale:         norm_dg from compute_shadow_normalization_stats().
                          When None, falls back to per-tile max (not recommended).
        dark_scale:       Ignored. Kept so existing callers don't break.
        otsu_ctr:         Sigmoid centre. None → uses Slot 2 default (0.35).

    Returns:
        shadow_map: H×W float32 array in [0, 1].
    """
    gray  = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    valid = _sm_valid_mask(gray)
    adx, ady, sdx, sdy = _sm_vectors(shadow_angle_deg)

    # ── Short + long range directional gradient ────────────────────────────────
    dg_s = _sm_dg(gray, valid, adx, ady, sdx, sdy,
                  _SM_D_SHORT_MIN, _SM_D_SHORT_MAX, 8,  _SM_BLUR_SHORT)
    dg_l = _sm_dg(gray, valid, adx, ady, sdx, sdy,
                  _SM_D_LONG_MIN,  _SM_D_LONG_MAX,  12, _SM_BLUR_LONG)
    dg   = (_SM_SHORT_WEIGHT * dg_s + (1.0 - _SM_SHORT_WEIGHT) * dg_l) * valid

    # ── Absolute luma gate: pixels brighter than abs_luma_max cannot be shadows ─
    luma_penalty = np.clip((_SM_ABS_LUMA_MAX - gray) / 20.0, 0.0, 1.0)

    # ── Lateral edge penalty: suppress edges running parallel to shadow dir ─────
    ang = np.radians(shadow_angle_deg)
    sun_dx, sun_dy = np.sin(ang), -np.cos(ang)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_sun  = cv2.GaussianBlur(np.abs(gx * sun_dx  + gy * sun_dy),  (5, 5), 2)
    grad_perp = cv2.GaussianBlur(np.abs(gx * (-sun_dy) + gy * sun_dx), (5, 5), 2)
    lateral_penalty = np.clip(1.5 - (grad_perp / (grad_sun + 1e-3)) * 0.5, 0.0, 1.0)

    # ── Normalise and combine ──────────────────────────────────────────────────
    norm_dg = max(float(dg_scale), 1.0) if dg_scale is not None else max(float(dg.max()), 30.0)
    dg_n    = np.clip(dg / norm_dg, 0.0, 1.0)
    base    = dg_n * luma_penalty * lateral_penalty

    if base.max() <= 0:
        return np.zeros(img_rgb.shape[:2], dtype=np.float32)

    # ── Sigmoid + activation floor ─────────────────────────────────────────────
    ctr = float(otsu_ctr) if otsu_ctr is not None else _SM_OTSU_CTR
    sig = 1.0 / (1.0 + np.exp(-_SM_SIGMOID_K * (base - ctr)))
    sig[sig < _SM_ACT_FLOOR] = 0.0
    sig = _sm_remove_speckle(sig * valid, min_area=_SM_SPECKLE_MIN, threshold=_SM_ACT_FLOOR)
    return sig.astype(np.float32)

def is_australia(x):
    return -44 <= x["lat"] <= -10 and 112 <= x["lon"] <= 154


aus_tiles = [
    1207,
    4347,
    4159,
    4893,
    4406,
    2100,
    1104,
    3956,
    2859,
    3684,
    5001,
    3469,
    5012,
    4660,
    536,
    4315,
    4506,
    3624,
    4127,
    4963,
    423,
    1703,
    3016,
    4643,
    922,
    4221,
    4955,
    4909,
    3219,
    1671,
    195,
    4923,
    4556,
    4086,
    1969,
    3611,
    4336,
    2581,
    1033,
    314,
    2491,
    4720,
    4421,
    5005,
    4435,
    2654,
    62,
    4593,
    5057,
    1612,
    1417,
    1278,
    2403,
    2270,
    367,
    1339,
    1117,
    4507,
    4040,
    577,
    439,
    2888,
    4326,
    1875,
    760,
    678,
    3456,
    4108,
    1029,
    2515,
    4996,
    876,
    4639,
    4933,
    2031,
    750,
    1248,
    743,
    2293,
    3277,
    3875,
    1077,
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
    "Zambezian and Mopane woodlands",
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
    "West Sudanian savanna",
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

    Note: We disable image decoding in the dataset to avoid Pillow's
    JPEG-compressed TIFF issues, and decode manually with cv2 instead.
    """
    from datasets import load_dataset, Image
    from io import BytesIO
    from PIL import Image as PILImage

    print("📦 Loading TCD dataset in streaming mode...")
    # Disable automatic image decoding to avoid Pillow TIFF/JPEG issues
    ds = load_dataset("restor/tcd", split="train", streaming=True).cast_column(
        "image", Image(decode=False)
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for image_info in ds:
        if not is_rangeland(image_info):
            continue

        if count >= max_images:
            break

        image_id = image_info["image_id"]
        print(f"📸 Downloading Rangeland tile {count}: {image_id}")

        img_path = save_dir / f"tcd_tile_{count}.tif"
        meta_path = save_dir / f"tcd_tile_{count}_meta.json"

        # ------------------
        # Decode image manually to avoid Pillow TIFF/JPEG issues
        # ------------------
        try:
            # Get raw image bytes
            img_bytes = image_info["image"]["bytes"]

            # Decode with cv2 (more robust for JPEG-compressed TIFFs)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                # Fallback to PIL if cv2 fails
                print("  ⚠️  cv2 decode failed, trying PIL...")
                pil_img = Image.open(BytesIO(img_bytes))
                img = np.array(pil_img)
                if img.ndim == 3 and img.shape[2] == 3:
                    # PIL loads as RGB, cv2 expects BGR
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Convert BGR to RGB for rasterio
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"  ❌ Failed to decode image {image_id}: {e}")
            print(f"  Skipping tile {count}")
            continue

        h, w = img.shape[:2]
        crs = image_info["crs"]
        bounds = image_info["bounds"]

        transform = from_bounds(*bounds, width=w, height=h)

        # ------------------
        # Save as GeoTIFF
        # ------------------
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
        meta = {
            "image_id": image_id,
            "bounds": bounds,
            "crs": str(crs),
            "width": w,
            "height": h,
            "coco_annotations": image_info.get("coco_annotations", []),
            "biome": image_info.get("biome"),
            "biome_name": image_info.get("biome_name"),
            "country": image_info.get("country"),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        print(f"✅ Saved tile → {img_path}")
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
        "n_gt": n_gt,
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
        coeffs = [
            transform.a,
            transform.b,
            transform.d,
            transform.e,
            transform.c,
            transform.f,
        ]
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
            print(
                f"⚠️ Unknown segmentation format in annotation id={ann.get('id')} — skipping."
            )
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
    gt = gpd.GeoDataFrame(
        {"geometry": gt_world_parts, "category": gt_cats}, crs=image_tif["crs"]
    )
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
    metrics_canopy = compute_metrics(
        scores_canopy, iop_thresh_canopy, n_pred, n_gt_canopy
    )

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


def visualize_validation_results(
    pred,
    gt=None,
    ious=None,
    coco_anns=None,
    iou_thresh_tree=0.5,
    iop_thresh_canopy=0.7,
    site_path=None,
    rgb_path=None,
    tile_name=None,
    image_id=None,
):
    """
    Visualize Detectree2 vs TCD polygons over RGB base image.
    Handles cases with no Ground Truth (gt=None).
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
            extent = [
                src.bounds.left,
                src.bounds.right,
                src.bounds.bottom,
                src.bounds.top,
            ]

    # === 3. Plot setup ===
    # Calculate figsize to preserve resolution (assuming dpi=100 for calculation)
    # We want 1 px in image ~= 1 px in output
    dpi = 100
    if img is not None:
        h, w = img.shape[:2]
        figsize = (w / dpi, h / dpi)
    else:
        figsize = (10, 10)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    title = f"Detectree2 Predictions (IoU ≥ {iou_thresh_tree})"
    if gt is not None:
        title += f" vs TCD (IoP ≥ {iop_thresh_canopy})"
    ax.set_title(title)
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
        if ious:
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
        if gt is None:
            color = "#00F0FF"  # Default teal for predictions when no GT
        elif score_tree >= iou_thresh_tree:
            color = "#00F0FF"  # bright teal  → good tree (IoU)
        elif score_canopy >= iop_thresh_canopy:
            color = "#00FF9D"  # neon green  → good canopy (IoP)
        else:
            color = "#FF8800"  # orange      → low-overlap / FP

        draw_pred_outline(ax, p, color)

        # --- Small score label at centroid ---
        if gt is not None and (score_tree > 0 or score_canopy > 0):
            best_score = score_tree if score_tree >= score_canopy else score_canopy
            metric_name = "IoU" if score_tree >= score_canopy else "IoP"
            centroid = p.centroid
            ax.text(
                centroid.x,
                centroid.y,
                f"{metric_name}:{best_score:.2f}",
                ha="center",
                va="center",
                fontsize=5,
                color="white",
                alpha=0.85,
                clip_on=True,
            )

    # === 5. Draw ground-truth crowns & canopy (if available) ===
    if gt is not None:
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
                geoms = [
                    geom
                    for geom in g.geoms
                    if isinstance(geom, (Polygon, MultiPolygon))
                ]
            else:
                geoms = []

            for geom in geoms:
                if isinstance(geom, Polygon):
                    x, y = geom.exterior.xy
                    ax.fill(
                        x,
                        y,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.8,
                        alpha=0.25,
                    )
                    ax.plot(x, y, color=color, linewidth=0.8, alpha=0.9)
                elif isinstance(geom, MultiPolygon):
                    for sub in geom.geoms:
                        x, y = sub.exterior.xy
                        ax.fill(
                            x,
                            y,
                            facecolor=color,
                            edgecolor=color,
                            linewidth=0.8,
                            alpha=0.25,
                        )
                        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.9)

    # === 6. Legend ===
    import matplotlib.patches as mpatches

    legend_elems = [
        mpatches.Patch(color="#00F0FF", label="Tree TP (IoU ≥ 0.5)"),
        mpatches.Patch(color="#00FF9D", label="Canopy TP (IoP ≥ 0.7)"),
        mpatches.Patch(color="#FF8800", label="False Positive / Low Overlap"),
        mpatches.Patch(color="#C266FF", label="Ground Truth — Canopy"),
        mpatches.Patch(color="#0a20ad", label="Ground Truth — Tree"),
    ]

    ax.legend(handles=legend_elems, loc="lower right", frameon=True, fontsize=8)

    # Determine axis labels based on CRS
    xlabel, ylabel = "Easting", "Northing"
    if rgb_path and Path(rgb_path).exists():
        with rasterio.open(rgb_path) as src:
            if src.crs and src.crs.is_geographic:
                xlabel, ylabel = "Longitude", "Latitude"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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


def filter_raw_predictions(
    pred_dir: Path, score_thresh: float = 0.8, overwrite=True
) -> None:
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

    print(
        f"✅ Filtering complete — overwrote {len(json_files)} files at ≥ {score_thresh}."
    )


def load_tcd_meta_for_tile(tile_path: Path):
    """Load metadata JSON for a given TCD tile (if available)."""
    # Expected metadata path: data/tcd/raw/tcd_tile_X_meta.json
    # Or just look for any .json with same stem
    meta_path = tile_path.with_name(f"{tile_path.stem}_meta.json")

    if not meta_path.exists():
        # Try finding *any* json that matches
        candidates = list(tile_path.parent.glob(f"{tile_path.stem}*.json"))
        if candidates:
            meta_path = candidates[0]

    if not meta_path.exists():
        # Return minimal default metadata for unannotated images
        return {"image_id": "unknown", "biome_name": "Unknown", "coco_annotations": []}

    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load metadata {meta_path}: {e}")
        return {"image_id": "unknown", "biome_name": "Unknown", "coco_annotations": []}


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
            transform.a,
            transform.b,
            transform.d,
            transform.e,
            transform.c,
            transform.f,
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



# Non Maximum Suppression (NMS) for GeoJSON predictions
def apply_nms_to_geojson(geojson_path: Path, iou_threshold: float = 0.3) -> Path:
    """
    Remove duplicate/overlapping predictions using NMS (Non-Maximum Suppression).
    Keeps highest-confidence predictions and removes lower-confidence overlaps.
    """

    gdf = gpd.read_file(geojson_path)

    if len(gdf) == 0:
        print(f"⚠️ No predictions in {geojson_path}")
        return geojson_path

    print(f"📊 NMS input: {len(gdf)} polygons")

    # Fix invalid geometries (e.g. self-intersections) that crash shapely
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: make_valid(g) if not g.is_valid else g
    )
    # If make_valid returns a Collection, filter to keep only Polygons/MultiPolygons
    # Assumes make_valid returns a geometry compatible with intersection operations
    # A safer approach for NMS is to explode if it becomes a collection, but usually it's fine

    gdf = gdf.sort_values("Confidence_score", ascending=False).reset_index(drop=True)

    keep_indices = set(range(len(gdf)))  # Start with all indices
    removed = 0

    for i in range(len(gdf)):
        if i not in keep_indices:
            continue  # Already removed

        geom_i = gdf.loc[i, "geometry"]
        conf_i = gdf.loc[i, "Confidence_score"]

        for j in range(i + 1, len(gdf)):
            if j not in keep_indices:
                continue  # Already removed

            geom_j = gdf.loc[j, "geometry"]
            conf_j = gdf.loc[j, "Confidence_score"]

            intersection = geom_i.intersection(geom_j).area
            union = geom_i.union(geom_j).area
            iou = intersection / union if union > 0 else 0

            # Remove LOWER-confidence polygon (j, since sorted descending)
            if iou > iou_threshold:
                keep_indices.discard(j)  # REMOVE j from keep set
                removed += 1
                # print(f"  Removing poly_{j} (conf={conf_j:.3f}) — overlaps with poly_{i} (conf={conf_i:.3f}, IoU={iou:.3f})")

    print(f"📊 NMS output: {len(keep_indices)} polygons (removed {removed})")

    gdf_dedup = gdf.loc[list(keep_indices)].copy()
    gdf_dedup.to_file(geojson_path, driver="GeoJSON")

    return geojson_path
