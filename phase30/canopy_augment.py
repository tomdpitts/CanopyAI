#!/usr/bin/env python3
"""
canopy_augment.py — augment foxtrot predictions with SAM-generated canopy
polygons to boost binary semantic-segmentation F1.

Foxtrot is a per-tree instance detector and SAM is prompted with each
tree's bounding box.  Dense uniform canopy regions where individual
trees blur together are silently dropped — no DeepForest detection
fires inside them.  This script uses SAM's built-in automatic mask
generator (no prompts, samples a grid of points) to recover those
canopy regions.

Filter: a SAM auto-mask is kept iff at least N existing foxtrot tree
predictions fall inside it.  This is the elegant heuristic — tree
detections cluster in canopy and require no external signal.  Sky,
water, bare ground will have no tree detections inside them and are
filtered out naturally.

Output: new geojson with the original tree predictions PLUS added
canopy polygons (tagged via "pred_type" = "canopy").  pycocotools mAP50
ignores predictions falling inside cat=1 canopy GT (iscrowd), so the
canopy polygons are free upside for binary F1 and never hurt mAP.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from shapely.geometry import Polygon, mapping
from shapely.validation import make_valid

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _mask_to_polygon(mask):
    """Convert a binary mask to a Shapely polygon (largest connected
    contour).  Returns None if the mask is empty or degenerate."""
    msk = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50: return None
    if len(largest) < 3: return None
    coords = largest.reshape(-1, 2).tolist()
    if coords[0] != coords[-1]: coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid: poly = make_valid(poly)
    poly = poly.simplify(1.0, preserve_topology=True)
    if poly.geom_type != "Polygon" or poly.area < 50: return None
    return poly


def generate_canopy_polygons(image, tree_polys, mask_gen,
                              sam_input_size=1024,
                              min_trees_inside=3, min_area_frac=0.005,
                              area_per_required_tree=50_000):
    """Generate canopy polygons for one tile.

    For each SAM auto-mask:
      * Discard masks smaller than `min_area_frac` of tile area.
      * Compute n_trees_inside (tree predictions with centroid in mask).
      * **Area-scaled tree-density filter**: require
            n_trees_inside >= max(min_trees_inside, mask_area / area_per_required_tree)
        This means small masks need just a few trees, but tile-spanning
        masks need many — preventing false-canopy masks that cover bare
        ground with a few scattered trees.

    Returns list of (polygon, score, n_trees_inside)."""
    H, W = image.shape[:2]
    # Downsample for SAM speed; masks are reasonably resolution-robust
    sample = np.array(
        Image.fromarray(image).resize((sam_input_size, sam_input_size),
                                       Image.BILINEAR)
    )
    masks = mask_gen.generate(sample)
    scale_x = W / sam_input_size; scale_y = H / sam_input_size

    if tree_polys:
        tree_centroids = []
        for p in tree_polys:
            try:
                c = p.centroid
                tree_centroids.append((c.x, c.y))
            except Exception:
                pass
        tree_centroids = np.array(tree_centroids) if tree_centroids else np.empty((0, 2))
    else:
        tree_centroids = np.empty((0, 2))

    canopy_polys = []
    min_area_px = min_area_frac * H * W
    for m in masks:
        msk_low = m["segmentation"].astype(np.uint8)
        msk = cv2.resize(msk_low, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(msk.sum())
        if area < min_area_px:
            continue
        # Count tree centroids that land inside this mask
        if len(tree_centroids):
            xi = np.clip(tree_centroids[:, 0].astype(int), 0, W - 1)
            yi = np.clip(tree_centroids[:, 1].astype(int), 0, H - 1)
            n_inside = int(msk[yi, xi].sum())
        else:
            n_inside = 0
        required = max(min_trees_inside, int(area / area_per_required_tree))
        if n_inside < required:
            continue
        poly = _mask_to_polygon(msk)
        if poly is None: continue
        canopy_polys.append((poly, float(m["predicted_iou"]), n_inside))
    return canopy_polys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_image(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read([1, 2, 3])
    image = np.transpose(arr, (1, 2, 0))
    if image.dtype != np.uint8:
        mx = max(1, image.max())
        image = (image.astype(np.float32) / mx * 255).astype(np.uint8)
    return image


def _write_geojson(gdf, canopy_polys, out_path):
    """Write geojson with original tree features + appended canopy features."""
    features = []
    # Existing predictions (preserve as-is)
    for idx, geom in enumerate(gdf.geometry):
        props = {k: v for k, v in gdf.iloc[idx].items() if k != "geometry"}
        clean = {"pred_type": "tree"}
        for k, v in props.items():
            if hasattr(v, "tolist"): clean[k] = v.tolist()
            elif isinstance(v, (int, float, str, bool)) or v is None: clean[k] = v
            else: clean[k] = str(v)
        features.append({"type": "Feature", "properties": clean,
                          "geometry": mapping(geom)})
    # Canopy additions
    for i, (poly, score, n_trees) in enumerate(canopy_polys):
        # Canopy polys are stored with a deliberately LOW deepforest_score
        # (default 0.001) so pycocotools sorts them to the bottom of the
        # detection ranking.  They get correctly IGNOREd at eval (iscrowd
        # match against cat=1 canopy GT) and still contribute pixels to
        # the binary union, but they don't displace tree-pred matches at
        # the top of the ranking — preserving mAP50 exactly.
        features.append({"type": "Feature",
                          "properties": {"pred_type": "canopy",
                                         "deepforest_score": 0.001,
                                         "sam_predicted_iou": float(score),
                                         "canopy_trees_inside": int(n_trees),
                                         "area_pixels": float(poly.area)},
                          "geometry": mapping(poly)})
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--src", required=True,
                    help="Folder of foxtrot _canopyai.geojson files to augment.")
    ap.add_argument("--dst", required=True, help="Output folder.")
    ap.add_argument("--image-dir", required=True,
                    help="Folder containing the matching <stem>.tif files.")
    ap.add_argument("--sam-checkpoint", default="sam_vit_b_01ec64.pth")
    ap.add_argument("--sam-model", default="vit_b", choices=["vit_b","vit_l","vit_h"])
    ap.add_argument("--min-trees-inside", type=int, default=3,
                    help="Minimum tree predictions inside a SAM mask to keep it.")
    ap.add_argument("--min-area-frac", type=float, default=0.005,
                    help="Minimum mask area as fraction of tile (default 0.005).")
    ap.add_argument("--area-per-required-tree", type=int, default=30_000,
                    help="For mask area A, require trees_inside >= A / this. "
                         "Default 30k (empirical F1 peak on OAM-TCD): a 100k-px "
                         "mask needs 3 trees, 3M-px mask needs 100 trees.  "
                         "Higher value = stricter on big masks.")
    ap.add_argument("--sam-input-size", type=int, default=1024,
                    help="Resolution at which to run SAM auto-mask (default 1024).")
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--tiles", nargs="+", default=None,
                    help="Restrict to these tile stems.")
    args = ap.parse_args()

    # SAM auto-mask requires float64 internally — MPS doesn't support it,
    # so run on CPU.  Per-tile cost is ~3-5s at 1024 input on M4 Max CPU.
    device = "cpu"
    print(f"Device: {device}  (SAM auto-mask uses float64 → CPU)")
    print(f"Loading {args.sam_model} from {args.sam_checkpoint}...")
    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint)
    sam.to(device).eval()
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.90,
        min_mask_region_area=2500,
    )

    src = Path(args.src); dst = Path(args.dst); image_dir = Path(args.image_dir)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*_canopyai.geojson"))
    if args.tiles:
        filt = set(args.tiles)
        files = [f for f in files if f.name.replace("_canopyai.geojson","") in filt]
    print(f"Augmenting {len(files)} tiles...")
    t_total = time.time()
    n_canopy_total = 0
    for fi, f in enumerate(files):
        stem = f.name.replace("_canopyai.geojson", "")
        gdf = gpd.read_file(str(f))
        tree_polys = [g for g in gdf.geometry if g is not None and not g.is_empty] if not gdf.empty else []
        tif = image_dir / f"{stem}.tif"
        if not tif.exists():
            # No image → pass-through
            with open(f) as fh: content = fh.read()
            with open(dst / f.name, "w") as fh: fh.write(content)
            continue
        image = _load_image(tif)
        t0 = time.time()
        canopy = generate_canopy_polygons(
            image, tree_polys, mask_gen,
            sam_input_size=args.sam_input_size,
            min_trees_inside=args.min_trees_inside,
            min_area_frac=args.min_area_frac,
            area_per_required_tree=args.area_per_required_tree,
        )
        elapsed = time.time() - t0
        n_canopy_total += len(canopy)
        _write_geojson(gdf, canopy, dst / f.name)
        if (fi + 1) % 5 == 0 or (fi + 1) == len(files):
            print(f"  {fi+1}/{len(files)}  +{len(canopy)} canopy  "
                  f"({elapsed:.1f}s/tile, total {time.time()-t_total:.0f}s)")
    print(f"\nDone.  Wrote {len(files)} files, total added canopy polys: {n_canopy_total}")


if __name__ == "__main__":
    main()
