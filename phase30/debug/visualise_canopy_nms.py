#!/usr/bin/env python3
"""
visualise_canopy_nms.py — show what the area-weighted NMS pass keeps on a
sample of canopy-heavy training tiles.

Reuses the pipeline helpers from visualise_canopy_pipeline.py (random crop,
polygon clip, anchor generation, IoP integral image), then runs the same
greedy NMS with anchor area as the score (matches CANOPY_NMS_IOU in
phase30/lib/models.py).  Writes one PNG per tile to phase30/debug/output/nms/.

Each figure has three panels:
  A — crop + canopy polygons + ITC GT boxes (context)
  B — raw IoP>=0.7 anchor centres (one dot per positive, sub-sampled to <=5000
       for legibility) coloured by FPN-level proxy (anchor area)
  C — NMS survivors drawn as boxes, coloured by size bucket
       (green=small, blue=mid, red=large). Stats panel inset.

Usage:
    python phase30/debug/visualise_canopy_nms.py \\
        --train-csv       phase30/phase30_tcd_train.csv \\
        --canopy-polygons phase30/phase30_tcd_canopy_polygons.json \\
        --images-root     data/tcd/images/data/tcd/raw \\
        --n 10  --nms-iou 0.2  --crop-seed 42

The tile sampling mirrors the smoke-test workflow: pick the top-N
polygon-count tiles from the canopy JSON so figures are non-empty.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualise_canopy_pipeline import (  # noqa: E402
    CROP_SIZE, IOP_THRESH,
    anchor_iop_from_integral, build_canopy_integrals_per_poly,
    clip_boxes, clip_polygons,
    load_canopy_polygons, load_gt_boxes, load_image, load_retinanet,
    random_crop_offset, run_anchor_pipeline,
)


# ───────────────────────────────────────────────────────────────────────────
# NMS — mirror models.py:_patch_retinanet_head_loss
# ───────────────────────────────────────────────────────────────────────────

def canopy_nms(anchors, canopy_mask, ios_thresh):
    """Greedy size-descending suppression with Intersection-over-Smaller
    (IoS) as the overlap metric.  Mirrors models.py:_ios_greedy_suppress.

    Survivor invariant: for any pair, the smaller box has at most
    ``ios_thresh`` of its area inside the larger.  Plain IoU NMS would
    leave small-inside-large redundants alive because IoU(small,large) is
    bounded by small_area/large_area, which is often far below the
    threshold.
    """
    survivors = torch.zeros_like(canopy_mask)
    if not canopy_mask.any():
        return survivors
    pool_idx   = canopy_mask.nonzero(as_tuple=True)[0]
    pool_boxes = anchors[pool_idx]
    pool_areas = (
        (pool_boxes[:, 2] - pool_boxes[:, 0]).clamp(min=1.0)
        * (pool_boxes[:, 3] - pool_boxes[:, 1]).clamp(min=1.0)
    )
    order = pool_areas.argsort(descending=True)
    sorted_boxes = pool_boxes[order]
    sorted_areas = pool_areas[order]
    N = sorted_boxes.shape[0]
    alive = torch.ones(N, dtype=torch.bool, device=anchors.device)
    keep_sorted = []
    for k in range(N):
        if not alive[k]:
            continue
        keep_sorted.append(k)
        tail = slice(k + 1, N)
        box = sorted_boxes[k]
        x1 = torch.maximum(sorted_boxes[tail, 0], box[0])
        y1 = torch.maximum(sorted_boxes[tail, 1], box[1])
        x2 = torch.minimum(sorted_boxes[tail, 2], box[2])
        y2 = torch.minimum(sorted_boxes[tail, 3], box[3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        ios = inter / sorted_areas[tail]
        alive[tail] &= ~(ios > ios_thresh)
    keep_sorted_t = torch.tensor(keep_sorted, dtype=torch.long,
                                 device=anchors.device)
    survivors[pool_idx[order[keep_sorted_t]]] = True
    return survivors


# ───────────────────────────────────────────────────────────────────────────
# Plot
# ───────────────────────────────────────────────────────────────────────────

def _outline_polys(ax, polys, color, lw=1.2, alpha=0.95):
    for verts in polys:
        ax.plot(verts[:, 0], verts[:, 1], color=color, lw=lw, alpha=alpha)


def _draw_boxes(ax, boxes, color, lw=1.0, alpha=0.95):
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               edgecolor=color, facecolor="none",
                               lw=lw, alpha=alpha))


def make_figure(out_path, image_stem, patch, polys_crop, boxes_crop,
                anchors_post, raw_mask, survivor_mask,
                scale_w, scale_h, nms_iou):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3))

    # ── Panel A — context
    axes[0].imshow(patch)
    _outline_polys(axes[0], polys_crop, "#00d8ff", lw=1.2)
    _draw_boxes(axes[0], boxes_crop, "#ffd400", lw=1.2)
    axes[0].set_title(f"A — context\n"
                      f"polys_in_crop={len(polys_crop)}  itc_gt={len(boxes_crop)}",
                      fontsize=9, family="monospace", loc="left")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Anchors back into crop-local coords for drawing
    a_local = anchors_post.cpu().numpy().copy()
    a_local[:, [0, 2]] /= max(1e-6, scale_w)
    a_local[:, [1, 3]] /= max(1e-6, scale_h)
    areas_local = (a_local[:, 2] - a_local[:, 0]) * (a_local[:, 3] - a_local[:, 1])
    cx = (a_local[:, 0] + a_local[:, 2]) / 2
    cy = (a_local[:, 1] + a_local[:, 3]) / 2

    # ── Panel B — raw IoP>=0.7 (centres, sub-sampled, coloured by anchor area)
    axes[1].imshow(patch)
    raw_idx = raw_mask.cpu().numpy().nonzero()[0]
    if len(raw_idx) > 5000:
        rng = np.random.default_rng(0)
        raw_idx = rng.choice(raw_idx, 5000, replace=False)
    if len(raw_idx):
        sc = axes[1].scatter(cx[raw_idx], cy[raw_idx], c=np.sqrt(areas_local[raw_idx]),
                             cmap="viridis", s=2, alpha=0.45)
        cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("sqrt(anchor area), px", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
    _outline_polys(axes[1], polys_crop, "#00d8ff", lw=0.8, alpha=0.7)
    axes[1].set_title(f"B — raw IoP>={IOP_THRESH} anchor centres\n"
                      f"total={int(raw_mask.sum().item())}  "
                      f"(plotted {min(len(raw_idx), 5000)})",
                      fontsize=9, family="monospace", loc="left")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # ── Panel C — NMS survivors as actual boxes
    axes[2].imshow(patch)
    surv_idx = survivor_mask.cpu().numpy().nonzero()[0]
    surv_boxes = a_local[surv_idx]
    surv_areas = areas_local[surv_idx]
    if len(surv_areas) >= 3:
        q1, q2 = np.percentile(surv_areas, [33, 66])
    elif len(surv_areas):
        q1 = q2 = float(surv_areas.max())
    else:
        q1 = q2 = 0
    # Sort largest-first so smaller boxes render on top and stay visible
    order = np.argsort(-surv_areas)
    for i in order:
        x1, y1, x2, y2 = surv_boxes[i]
        a = surv_areas[i]
        c = "#4daf4a" if a < q1 else ("#377eb8" if a < q2 else "#e41a1c")
        axes[2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    edgecolor=c, facecolor="none",
                                    lw=0.6, alpha=0.7))
    _outline_polys(axes[2], polys_crop, "#00d8ff", lw=0.7, alpha=0.7)
    _draw_boxes(axes[2], boxes_crop, "#ffd400", lw=1.0, alpha=0.95)
    ratio = (len(surv_idx) / max(1, int(raw_mask.sum().item()))) * 100
    axes[2].set_title(
        f"C — NMS survivors  (iou={nms_iou})\n"
        f"survivors={len(surv_idx)} ({ratio:.1f}% of raw)\n"
        f"green=small  blue=mid  red=large",
        fontsize=9, family="monospace", loc="left",
    )
    axes[2].set_xticks([]); axes[2].set_yticks([])

    fig.suptitle(image_stem, fontsize=11, family="monospace", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
# Tile sampling
# ───────────────────────────────────────────────────────────────────────────

def pick_canopy_heavy_tiles(canopy_json_path, n):
    """Top-N tiles by polygon count from the canopy JSON."""
    raw = json.loads(Path(canopy_json_path).read_text())
    counts = sorted(((k, len(v)) for k, v in raw.items()),
                    key=lambda kv: -kv[1])
    return [k for k, _ in counts[:n]]


def resolve_image_path(images_root, basename):
    """Find a tile by basename under images_root."""
    p = Path(images_root) / basename
    if p.exists():
        return p
    matches = list(Path(images_root).rglob(basename))
    return matches[0] if matches else None


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--train-csv",       required=True)
    p.add_argument("--canopy-polygons", required=True)
    p.add_argument("--images-root",     required=True,
                   help="Root directory containing the .tif tiles")
    p.add_argument("--n",               type=int, default=10)
    p.add_argument("--nms-iou",         type=float, default=0.2)
    p.add_argument("--crop-seed",       type=int, default=42)
    p.add_argument("--checkpoint",      default=None)
    p.add_argument("--save-dir",        default="phase30/debug/output/nms")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Picking top-{args.n} canopy-heavy tiles ...")
    basenames = pick_canopy_heavy_tiles(args.canopy_polygons, args.n)

    print("Loading deepforest model ...")
    rnet = load_retinanet(args.checkpoint)
    inner = rnet.model

    rng_master = np.random.default_rng(args.crop_seed)

    print(f"\n{'tile':<28s} {'raw':>6s} {'kept':>5s}  {'pct':>5s}  {'gt':>4s}")
    print("-" * 56)

    for bn in basenames:
        image_path = resolve_image_path(args.images_root, bn)
        if image_path is None:
            print(f"{bn:<28s}   ?? image not found under {args.images_root}")
            continue

        full_img   = load_image(image_path)
        H, W       = full_img.shape[:2]
        full_polys = load_canopy_polygons(args.canopy_polygons, image_path)
        full_boxes = load_gt_boxes(args.train_csv, image_path)

        seed = int(rng_master.integers(0, 2**31 - 1))
        rng  = np.random.default_rng(seed)
        cy_off, cx_off, ch, cw = random_crop_offset(rng, H, W, CROP_SIZE)
        patch = full_img[cy_off:cy_off + ch, cx_off:cx_off + cw]

        polys_crop = clip_polygons(full_polys, cy_off, cx_off, ch, cw)
        boxes_crop = clip_boxes(full_boxes, cy_off, cx_off, ch, cw)

        anchors, post_h, post_w, pre_h, pre_w = run_anchor_pipeline(inner, patch)
        scale_w = post_w / max(1, pre_w)
        scale_h = post_h / max(1, pre_h)

        per_poly = build_canopy_integrals_per_poly(
            polys_crop, post_h, post_w, scale_w, scale_h
        )
        if per_poly is None:
            print(f"{bn:<28s}  (no polygons survived crop, skipping)")
            continue
        # Per-polygon IoP: anchor qualifies iff >=IOP_THRESH inside a SINGLE
        # polygon (max IoP across polygons), matching the training-time test.
        max_iop = torch.zeros(anchors.shape[0], device=anchors.device)
        for integral, _mask in per_poly:
            integral_t = torch.from_numpy(integral).to(anchors.device)
            iop_p = anchor_iop_from_integral(anchors, integral_t, post_h, post_w)
            max_iop = torch.maximum(max_iop, iop_p)
        raw_mask = max_iop >= IOP_THRESH

        # Run the same area-weighted NMS as the head loss patch.  Here we have
        # no GT-matched mask to exclude (this script is illustrative, not the
        # exact training pool), so NMS runs over all raw canopy positives.
        survivor_mask = canopy_nms(anchors, raw_mask, args.nms_iou)

        n_raw  = int(raw_mask.sum().item())
        n_surv = int(survivor_mask.sum().item())
        pct    = (n_surv / max(1, n_raw)) * 100
        print(f"{bn:<28s} {n_raw:>6d} {n_surv:>5d}  {pct:>4.1f}%  {len(boxes_crop):>4d}")

        out_path = save_dir / f"{Path(bn).stem}.png"
        make_figure(out_path, Path(bn).stem, patch, polys_crop, boxes_crop,
                    anchors, raw_mask, survivor_mask,
                    scale_w, scale_h, args.nms_iou)

    print(f"\nWrote figures to {save_dir}/")


if __name__ == "__main__":
    main()
