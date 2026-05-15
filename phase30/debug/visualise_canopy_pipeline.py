#!/usr/bin/env python3
"""
visualise_canopy_pipeline.py — single-image debug trace of the phase30 canopy
GT handling pipeline.

Re-implements the dataset crop + IoP machinery LOCALLY (not imported from
train_deepforest/models) so any divergence between viz and training is itself
a useful signal.

Five panels written to phase30/debug/output/<image_stem>.png:
  A — full image + all canopy polygons + GT trees + crop window
  B — 400x400 crop with clipped polygons + clipped GT boxes
  C — crop + rasterised canopy mask overlay
  D — crop + anchors passing IoP>=0.7 (post-transform space)
  E — crop + anchor-centre scatter: grey (none), blue (canopy+), green (GT+),
        red (both)

Usage:
    python phase30/debug/visualise_canopy_pipeline.py \\
        --image-path <tile.tif> \\
        --train-csv  <phase30_tcd_train.csv> \\
        --canopy-polygons <phase30_tcd_canopy_polygons.json> \\
        --crop-seed 42

Optional --checkpoint <path.pth> to load a specific phase30 checkpoint instead
of HF pretrained weights.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

CROP_SIZE  = 400
MIN_VIS    = 0.5
IOP_THRESH = 0.7   # must match CANOPY_IOP_THRESH in phase30/lib/models.py


# ───────────────────────────────────────────────────────────────────────────
# Data loading
# ───────────────────────────────────────────────────────────────────────────

def load_image(path):
    """Load an image as HxWx3 uint8 RGB. Tries rasterio first, falls back to PIL."""
    p = Path(path)
    try:
        import rasterio
        with rasterio.open(p) as src:
            arr = src.read([1, 2, 3])
        img = np.transpose(arr, (1, 2, 0))
    except Exception:
        from PIL import Image
        img = np.array(Image.open(p).convert("RGB"))
    if img.dtype != np.uint8:
        m = max(1, int(img.max()))
        img = (img.astype(np.float32) / m * 255.0).astype(np.uint8)
    return img


def load_gt_boxes(train_csv, image_path):
    """Return (N,4) xyxy array for rows matching image_path (exact or basename)."""
    df = pd.read_csv(train_csv)
    matches = df[df["image_path"].astype(str) == str(image_path)]
    if matches.empty:
        bn = Path(image_path).name
        matches = df[df["image_path"].astype(str).apply(
            lambda p: Path(p).name == bn
        )]
    matches = matches.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
    if matches.empty:
        return np.zeros((0, 4), np.float32)
    return matches[["xmin", "ymin", "xmax", "ymax"]].to_numpy(np.float32)


def load_canopy_polygons(json_path, image_path):
    """Look up polygons for the given image via Path().name basename match.
    Returns list of (V,2) float32 vertex arrays."""
    raw = json.loads(Path(json_path).read_text())
    key = Path(image_path).name
    out = []
    for flat in raw.get(key, []):
        arr = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
        if len(arr) >= 3:
            out.append(arr)
    return out


# ───────────────────────────────────────────────────────────────────────────
# Pipeline replication (local copies of the training-time logic)
# ───────────────────────────────────────────────────────────────────────────

def random_crop_offset(rng, H, W, cs):
    """Same logic as _CanopyAwareTrainDataset.__getitem__ crop selection."""
    if H > cs and W > cs:
        cy = int(rng.integers(0, H - cs + 1))
        cx = int(rng.integers(0, W - cs + 1))
    else:
        cy = cx = 0
    ch = min(cs, H)
    cw = min(cs, W)
    return cy, cx, ch, cw


def clip_polygons(polys_full, cy_off, cx_off, ch, cw):
    """Mirror the shapely clipping in _CanopyAwareTrainDataset."""
    if not polys_full:
        return []
    from shapely.geometry import box as shapely_box, Polygon
    window = shapely_box(cx_off, cy_off, cx_off + cw, cy_off + ch)
    out = []
    for verts in polys_full:
        if len(verts) < 3:
            continue
        try:
            poly = Polygon(verts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            clipped = poly.intersection(window)
            if clipped.is_empty or clipped.area < 4.0:
                continue
            if clipped.geom_type == "Polygon":
                parts = [clipped]
            elif clipped.geom_type == "MultiPolygon":
                parts = list(clipped.geoms)
            else:
                parts = []
            for g in parts:
                if g.is_empty or g.area < 4.0:
                    continue
                ext = np.asarray(g.exterior.coords, dtype=np.float32)
                ext[:, 0] -= cx_off
                ext[:, 1] -= cy_off
                out.append(ext)
        except Exception:
            continue
    return out


def clip_boxes(boxes, cy_off, cx_off, ch, cw, min_vis=MIN_VIS):
    """Mirror the GT box clipping in _CanopyAwareTrainDataset."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32)
    cx1 = np.clip(boxes[:, 0] - cx_off, 0, cw)
    cy1 = np.clip(boxes[:, 1] - cy_off, 0, ch)
    cx2 = np.clip(boxes[:, 2] - cx_off, 0, cw)
    cy2 = np.clip(boxes[:, 3] - cy_off, 0, ch)
    bw  = boxes[:, 2] - boxes[:, 0]
    bh  = boxes[:, 3] - boxes[:, 1]
    vis = np.where(
        bw * bh > 0,
        np.maximum(cx2 - cx1, 0) * np.maximum(cy2 - cy1, 0) / (bw * bh),
        0,
    )
    keep = vis >= min_vis
    if not keep.any():
        return np.zeros((0, 4), np.float32)
    return np.stack([cx1[keep], cy1[keep], cx2[keep], cy2[keep]],
                    axis=1).astype(np.float32)


def build_canopy_integral(polygons, post_h, post_w, scale_w, scale_h):
    """Mirror models.py:_build_canopy_integral. Returns (integral, mask) or None."""
    if not polygons:
        return None
    from PIL import Image, ImageDraw
    img = Image.new("L", (post_w, post_h), 0)
    drw = ImageDraw.Draw(img)
    for verts in polygons:
        if len(verts) < 3:
            continue
        pts = [(float(v[0]) * scale_w, float(v[1]) * scale_h) for v in verts]
        drw.polygon(pts, fill=1)
    mask = np.asarray(img, dtype=np.int64)
    if mask.sum() == 0:
        return None
    integral = np.zeros((post_h + 1, post_w + 1), dtype=np.float32)
    integral[1:, 1:] = mask.cumsum(0).cumsum(1).astype(np.float32)
    return integral, mask


def build_canopy_integrals_per_poly(polygons, post_h, post_w, scale_w, scale_h):
    """Mirror models.py:_build_canopy_integrals_per_poly.  Returns list of
    (integral, mask) tuples — one per polygon — or None.  Used so the IoP
    test in this viz matches the per-polygon training semantics: an anchor
    qualifies only if it's >=thresh inside a SINGLE polygon.
    """
    if not polygons:
        return None
    from PIL import Image, ImageDraw
    out = []
    for verts in polygons:
        if len(verts) < 3:
            continue
        img = Image.new("L", (post_w, post_h), 0)
        drw = ImageDraw.Draw(img)
        pts = [(float(v[0]) * scale_w, float(v[1]) * scale_h) for v in verts]
        drw.polygon(pts, fill=1)
        mask = np.asarray(img, dtype=np.int64)
        if mask.sum() == 0:
            continue
        integral = np.zeros((post_h + 1, post_w + 1), dtype=np.float32)
        integral[1:, 1:] = mask.cumsum(0).cumsum(1).astype(np.float32)
        out.append((integral, mask))
    return out if out else None


def anchor_iop_from_integral(anchors, integral_t, post_h, post_w):
    """Mirror models.py:_anchor_iop_from_integral."""
    x1 = anchors[:, 0].clamp(0, post_w).long()
    y1 = anchors[:, 1].clamp(0, post_h).long()
    x2 = anchors[:, 2].clamp(0, post_w).long()
    y2 = anchors[:, 3].clamp(0, post_h).long()
    A = integral_t[y2, x2]
    B = integral_t[y2, x1]
    C = integral_t[y1, x2]
    D = integral_t[y1, x1]
    canopy_area = A - B - C + D
    anchor_w = (anchors[:, 2] - anchors[:, 0]).clamp(min=1.0)
    anchor_h = (anchors[:, 3] - anchors[:, 1]).clamp(min=1.0)
    return (canopy_area / (anchor_w * anchor_h)).clamp(0.0, 1.0)


# ───────────────────────────────────────────────────────────────────────────
# Model machinery (anchors + matcher) — runs the actual deepforest RetinaNet
# ───────────────────────────────────────────────────────────────────────────

def load_retinanet(checkpoint=None):
    """deepforest model with anchor_generator / transform / matcher ready."""
    from deepforest import main as deepforest_main
    m = deepforest_main.deepforest()
    if checkpoint and Path(checkpoint).exists():
        try:
            sd = torch.load(checkpoint, map_location="cpu")
            if any(k.startswith("model.") for k in sd):
                m.load_state_dict(sd, strict=False)
            else:
                m.model.load_state_dict(sd, strict=False)
            print(f"   Loaded checkpoint: {checkpoint}")
        except Exception as e:
            print(f"   ⚠️  Couldn't load checkpoint ({e}); falling back to pretrained")
            m.load_model("weecology/deepforest-tree")
    else:
        m.load_model("weecology/deepforest-tree")
    m.model.eval()
    return m


def run_anchor_pipeline(rnet_model, patch_np):
    """Take HxWx3 uint8 patch, return (anchors, post_h, post_w, pre_h, pre_w).
    Anchors are in post-transform coordinates — same space the loss runs in."""
    pre_h, pre_w = patch_np.shape[:2]
    patch_t = torch.from_numpy(patch_np).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        image_list, _ = rnet_model.transform([patch_t], None)
        post_h = int(image_list.image_sizes[0][0])
        post_w = int(image_list.image_sizes[0][1])
        features  = rnet_model.backbone(image_list.tensors)
        feat_list = list(features.values())
        anchors   = rnet_model.anchor_generator(image_list, feat_list)[0]
    return anchors, post_h, post_w, pre_h, pre_w


def match_gt_to_anchors(rnet_model, gt_boxes_post, anchors):
    """Replicate RetinaNet anchor-to-GT matching. -1=bg, -2=between, >=0=fg."""
    from torchvision.ops import box_iou
    if gt_boxes_post.shape[0] == 0:
        return torch.full((anchors.shape[0],), -1, dtype=torch.long,
                          device=anchors.device)
    iou = box_iou(gt_boxes_post, anchors)
    return rnet_model.proposal_matcher(iou)


# ───────────────────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────────────────

def _outline_polys(ax, polys, color, ls="-", lw=1.2, alpha=0.95):
    for verts in polys:
        ax.plot(verts[:, 0], verts[:, 1], color=color, ls=ls, lw=lw, alpha=alpha)


def _draw_boxes(ax, boxes, color, lw=1.2, alpha=0.95):
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               edgecolor=color, facecolor="none",
                               lw=lw, alpha=alpha))


def make_figure(out_path, image_stem,
                full_img, full_polys, full_boxes,
                cy_off, cx_off, ch, cw,
                patch, polys_crop, boxes_crop,
                mask, anchors_post, iop, fg_mask, canopy_mask,
                post_h, post_w, scale_w, scale_h):
    fig, axes = plt.subplots(1, 5, figsize=(30, 6.5))

    # Panel A
    axes[0].imshow(full_img)
    _outline_polys(axes[0], full_polys, "#00d8ff", lw=1.0)
    _draw_boxes(axes[0], full_boxes, "#ffd400", lw=1.0)
    axes[0].add_patch(Rectangle((cx_off, cy_off), cw, ch,
                                edgecolor="#ff2a2a", facecolor="none", lw=2.0))
    axes[0].set_title(f"A — full image\n{image_stem}\n"
                      f"polys={len(full_polys)}  boxes={len(full_boxes)}",
                      fontsize=9, family="monospace", loc="left")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Panel B
    axes[1].imshow(patch)
    _outline_polys(axes[1], polys_crop, "#00d8ff", lw=1.4)
    _draw_boxes(axes[1], boxes_crop, "#ffd400", lw=1.4)
    axes[1].set_title(f"B — crop (clipped)\n"
                      f"polys_kept={len(polys_crop)}  boxes_kept={len(boxes_crop)}",
                      fontsize=9, family="monospace", loc="left")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Panel C — rasterised mask, resized from (post_h, post_w) back to (ch, cw)
    axes[2].imshow(patch)
    if mask is not None:
        from PIL import Image as PILImage
        mask_pil  = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_disp = np.asarray(mask_pil.resize((cw, ch), PILImage.NEAREST))
        masked    = np.ma.masked_where(mask_disp == 0, mask_disp)
        axes[2].imshow(masked, cmap="cool", alpha=0.4, vmin=0, vmax=255)
    _outline_polys(axes[2], polys_crop, "#00d8ff", lw=0.8, alpha=0.9)
    axes[2].set_title(f"C — rasterised mask  (post={post_w}x{post_h})\n"
                      f"scale=({scale_w:.3f},{scale_h:.3f})",
                      fontsize=9, family="monospace", loc="left")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    # Panel D — anchors passing IoP threshold (drawn in crop-local coords)
    axes[3].imshow(patch)
    if anchors_post is not None and canopy_mask.any():
        anchors_local = anchors_post.cpu().numpy().copy()
        anchors_local[:, [0, 2]] /= max(1e-6, scale_w)
        anchors_local[:, [1, 3]] /= max(1e-6, scale_h)
        kept = anchors_local[canopy_mask.cpu().numpy()]
        areas = (kept[:, 2] - kept[:, 0]) * (kept[:, 3] - kept[:, 1])
        if len(areas):
            q1, q2 = np.percentile(areas, [33, 66])
        else:
            q1 = q2 = 0
        for (x1, y1, x2, y2), a in zip(kept, areas):
            c = "#4daf4a" if a < q1 else ("#377eb8" if a < q2 else "#e41a1c")
            axes[3].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        edgecolor=c, facecolor="none",
                                        lw=0.4, alpha=0.55))
    n_pass = int(canopy_mask.sum().item()) if canopy_mask is not None else 0
    n_tot  = int(anchors_post.shape[0]) if anchors_post is not None else 0
    axes[3].set_title(f"D — anchors IoP>={IOP_THRESH}\n"
                      f"passing={n_pass} / total={n_tot}\n"
                      f"green=small  blue=mid  red=large",
                      fontsize=9, family="monospace", loc="left")
    axes[3].set_xticks([]); axes[3].set_yticks([])

    # Panel E — anchor-centre scatter
    axes[4].imshow(patch)
    if anchors_post is not None:
        a_local = anchors_post.cpu().numpy().copy()
        a_local[:, [0, 2]] /= max(1e-6, scale_w)
        a_local[:, [1, 3]] /= max(1e-6, scale_h)
        cx = (a_local[:, 0] + a_local[:, 2]) / 2
        cy = (a_local[:, 1] + a_local[:, 3]) / 2
        cmask = canopy_mask.cpu().numpy()
        fmask = fg_mask.cpu().numpy()
        both   = cmask & fmask
        only_c = cmask & ~both
        only_f = fmask & ~both
        none   = ~(cmask | fmask)
        rng_sub = np.random.default_rng(0)
        none_idx = np.where(none)[0]
        if len(none_idx) > 5000:
            none_idx = rng_sub.choice(none_idx, 5000, replace=False)
        axes[4].scatter(cx[none_idx], cy[none_idx], s=0.5, c="#cccccc", alpha=0.3)
        axes[4].scatter(cx[only_c], cy[only_c], s=4, c="#377eb8", alpha=0.7,
                        label=f"canopy+ ({only_c.sum()})")
        axes[4].scatter(cx[only_f], cy[only_f], s=4, c="#4daf4a", alpha=0.7,
                        label=f"GT fg ({only_f.sum()})")
        axes[4].scatter(cx[both], cy[both], s=6, c="#e41a1c", alpha=0.85,
                        label=f"both ({both.sum()})")
        axes[4].legend(loc="upper right", fontsize=7, framealpha=0.85)
    _outline_polys(axes[4], polys_crop, "#00d8ff", lw=0.6, alpha=0.7)
    _draw_boxes(axes[4], boxes_crop, "#ffd400", lw=0.6, alpha=0.7)
    axes[4].set_title(f"E — anchor cls targets\n(grey=none, blue=canopy+, "
                      f"green=GT fg, red=both)",
                      fontsize=9, family="monospace", loc="left")
    axes[4].set_xticks([]); axes[4].set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--image-path",      required=True)
    p.add_argument("--train-csv",       required=True)
    p.add_argument("--canopy-polygons", required=True)
    p.add_argument("--crop-seed",       type=int, default=42)
    p.add_argument("--checkpoint",      default=None)
    p.add_argument("--save-dir",        default="phase30/debug/output")
    return p.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    stem       = image_path.stem
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path   = save_dir / f"{stem}.png"

    print(f"\n=== Canopy pipeline trace ===")
    print(f"image: {image_path}")

    full_img   = load_image(image_path)
    H, W       = full_img.shape[:2]
    full_polys = load_canopy_polygons(args.canopy_polygons, image_path)
    full_boxes = load_gt_boxes(args.train_csv, image_path)

    print(f"image size: {W}x{H}")
    print(f"polygons in JSON: {len(full_polys)}")
    print(f"GT boxes in CSV:  {len(full_boxes)}")

    # Step 1: deterministic random crop
    rng = np.random.default_rng(args.crop_seed)
    cy_off, cx_off, ch, cw = random_crop_offset(rng, H, W, CROP_SIZE)
    patch = full_img[cy_off:cy_off + ch, cx_off:cx_off + cw]
    print(f"crop window: x=[{cx_off},{cx_off+cw}]  y=[{cy_off},{cy_off+ch}]  "
          f"(seed={args.crop_seed})")

    # Step 2: clip GT + polygons to the crop window
    polys_crop = clip_polygons(full_polys, cy_off, cx_off, ch, cw)
    boxes_crop = clip_boxes(full_boxes, cy_off, cx_off, ch, cw)
    print(f"polygons surviving crop clip: {len(polys_crop)}")
    print(f"GT boxes surviving crop clip: {len(boxes_crop)}")

    # Step 3: instantiate the deepforest RetinaNet for anchors + matcher
    print(f"\n=== Loading deepforest model (anchor generator + matcher only) ===")
    rnet  = load_retinanet(args.checkpoint)
    inner = rnet.model

    anchors, post_h, post_w, pre_h, pre_w = run_anchor_pipeline(inner, patch)
    scale_w = post_w / max(1, pre_w)
    scale_h = post_h / max(1, pre_h)
    print(f"pre=({pre_h},{pre_w})  post=({post_h},{post_w})  "
          f"scale=({scale_w:.3f},{scale_h:.3f})")

    # Step 4: rasterise canopy at post-transform size
    out = build_canopy_integral(polys_crop, post_h, post_w, scale_w, scale_h)
    if out is None:
        integral, mask, integral_t = None, None, None
    else:
        integral, mask = out
        integral_t = torch.from_numpy(integral).to(anchors.device)

    # Step 5: IoP per anchor
    if integral_t is not None:
        iop = anchor_iop_from_integral(anchors, integral_t, post_h, post_w)
        canopy_mask = iop >= IOP_THRESH
    else:
        iop = torch.zeros(anchors.shape[0])
        canopy_mask = torch.zeros(anchors.shape[0], dtype=torch.bool)

    # Step 6: GT-to-anchor matching (boxes_crop scaled into post-transform space)
    gt_post = torch.from_numpy(boxes_crop).float() if len(boxes_crop) else torch.zeros((0, 4))
    if gt_post.shape[0]:
        gt_post[:, [0, 2]] *= scale_w
        gt_post[:, [1, 3]] *= scale_h
    matched = match_gt_to_anchors(inner, gt_post.to(anchors.device), anchors)
    fg_mask = matched >= 0

    # Stats
    print(f"\n=== Anchor stats (post-transform space) ===")
    print(f"total anchors:           {anchors.shape[0]}")
    print(f"GT-matched foreground:   {int(fg_mask.sum().item())}")
    print(f"canopy-positive (IoP>={IOP_THRESH}): {int(canopy_mask.sum().item())}")
    print(f"overlap (fg AND canopy+): {int((canopy_mask & fg_mask).sum().item())}")

    if integral_t is not None and canopy_mask.any():
        iop_pos = iop[iop > 0]
        if len(iop_pos):
            p50, p75, p90, mx = np.percentile(iop_pos.cpu().numpy(), [50, 75, 90, 100])
            print(f"\nIoP distribution among anchors with any canopy overlap:")
            print(f"  p50={p50:.3f}  p75={p75:.3f}  p90={p90:.3f}  max={mx:.3f}")

    n_c = int(canopy_mask.sum().item())
    n_f = int(fg_mask.sum().item())
    if n_c + n_f > 0:
        print(f"\n=== Loss-budget signal ===")
        print(f"scale=1.0  canopy contributes ~{n_c/(n_c+n_f):.2%} of cls denominator")
        if n_c:
            for s in (0.5, 0.25, 0.1):
                print(f"scale={s}  canopy contributes ~{s*n_c/(n_f+s*n_c):.2%}")

    if n_c == 0:
        print("\n⚠️  WARNING: no anchor cleared IoP>=0.7 on this image.")
        print("   The user hypothesis (multi-scale anchors clear 0.7) is")
        print("   falsified for this tile.  Either pick a more canopy-heavy")
        print("   tile or revisit the threshold.")

    # Plot
    make_figure(out_path, stem, full_img, full_polys, full_boxes,
                cy_off, cx_off, ch, cw, patch, polys_crop, boxes_crop,
                mask, anchors, iop, fg_mask, canopy_mask,
                post_h, post_w, scale_w, scale_h)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
