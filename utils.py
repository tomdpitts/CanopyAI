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



# ── Shadow probability map (Phase 6) ──────────────────────────────────────────

def generate_shadow_map(img_rgb: np.ndarray, shadow_angle_deg: float) -> np.ndarray:
    """
    Generate a shadow probability map for a tile.

    Uses a directional asymmetry (DirGrad) × local darkness formulation:
      - DirGrad: for each pixel, accumulate max(0, brightness_toward_anti_shadow
                                                  - brightness_toward_shadow)
                 over offset distances d = 8..35 px (Gaussian weighted).
                 This fires where one side of a pixel, along the shadow axis,
                 is distinctly brighter than the other — the shadow/ground boundary.
      - Darkness: how much darker each pixel is than its 30-px-sigma local mean,
                  normalised to the 95th percentile so contrast is image-relative.
      - Product: DirGrad × darkness → high only where the pixel is both genuinely
                 dark AND sits at a directionally asymmetric boundary.
      - Otsu sigmoid (k=12): sharpens the map using the image's own bimodal
                 histogram split as the sigmoid centre.
      - Boundary mask (+3px erode): zeros out black rotation corners
                 and the 1-2px warpAffine interpolation bleed.

    Args:
        img_rgb:          H×W×3 uint8 numpy array (RGB order)
        shadow_angle_deg: Sun azimuth in degrees (0=North, 90=East, CW convention).
                          Shadows point in direction (180 + azimuth) mod 360 from the tree.

    Returns:
        shadow_map: H×W float32 array in [0, 1]
    """
    D_MIN, D_MAX = 8, 35

    # Convert to grayscale float
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # Shadow vector: shadow points FROM tree IN this direction
    # Convention: azimuth 0=North CW, image y increases downward
    ang_rad = np.radians(shadow_angle_deg)
    sx =  np.sin(ang_rad)   # shadow x-component (image right)
    sy = -np.cos(ang_rad)   # shadow y-component (image down, so negate cos)
    # Anti-shadow direction (toward crown)
    adx, ady = -sx, -sy
    # Shadow direction (beyond shadow tip)
    sdx, sdy =  sx,  sy

    # ── 1. Directional gradient ────────────────────────────────────────────
    ds  = np.arange(D_MIN, D_MAX + 1, 3).astype(np.float32)
    d_c = (D_MIN + D_MAX) / 2.0
    wts = np.exp(-0.5 * ((ds - d_c) / 8.0) ** 2)
    wts /= wts.sum()

    dg = np.zeros((h, w), dtype=np.float32)
    for d, wt in zip(ds, wts):
        Ma = np.float32([[1, 0, int(round(d * adx))], [0, 1, int(round(d * ady))]])
        Ms = np.float32([[1, 0, int(round(d * sdx))], [0, 1, int(round(d * sdy))]])
        anti = cv2.warpAffine(gray, Ma, (w, h), borderMode=cv2.BORDER_REFLECT)
        shad = cv2.warpAffine(gray, Ms, (w, h), borderMode=cv2.BORDER_REFLECT)
        dg  += wt * np.clip((anti - shad).astype(np.float32), 0, None)
    dg = cv2.GaussianBlur(dg, (0, 0), sigmaX=4)
    dg /= (dg.max() + 1e-6)

    # ── 2. Percentile-normalised darkness ──────────────────────────────────
    lm   = cv2.GaussianBlur(gray, (0, 0), sigmaX=30)
    dark = np.clip(lm - gray, 0, None).astype(np.float32)
    dark /= (np.percentile(dark, 95) + 1e-6)
    dark  = np.clip(dark, 0, 1)

    # ── 3. Product ─────────────────────────────────────────────────────────
    base = dg * dark
    base /= (base.max() + 1e-6)

    # ── 4. Otsu sigmoid (image-adaptive threshold) ─────────────────────────
    m_u8 = (base * 255).astype(np.uint8)
    thresh, _ = cv2.threshold(m_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ctr = float(np.clip(thresh / 255.0, 0.10, 0.70))
    k   = 12.0
    sig = 1.0 / (1.0 + np.exp(-k * (base - ctr)))
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-6)
    sig = sig.astype(np.float32)

    # ── 5. Boundary mask: zero out black corners + 3-px bleed ──────────────
    valid = (gray > 15).astype(np.uint8)
    k3    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 3px erode
    valid = cv2.erode(valid, k3).astype(np.float32)

    result = sig * valid
    mx = result.max()
    return (result / mx).astype(np.float32) if mx > 0 else result


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
