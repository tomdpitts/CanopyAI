#!/usr/bin/env python3
"""Render an explainer figure for the canopy positive policy.

For 5 random training tiles that contain both ITC bbox annotations and a
canopy polygon, draw a 600 px crop with:
  - ITC GT bboxes (gold)
  - the canopy polygon (cyan, semi-transparent fill)
  - one ITC anchor (rose, dashed) annotated with its loss treatment
  - one canopy anchor (violet, dashed) annotated with its loss treatment

Output: phase30/canopy_policy_explainer.png
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle, FancyBboxPatch
from matplotlib.patheffects import withStroke
from PIL import Image
from shapely.geometry import Polygon, box as shapely_box
from shapely.affinity import translate as shapely_translate

ROOT        = Path(__file__).resolve().parent.parent
TRAIN_CSV   = ROOT / "phase30/phase30_tcd_train.csv"
CANOPY_JSON = ROOT / "phase30/phase30_tcd_canopy_polygons.json"
OUT_PNG     = ROOT / "phase30/canopy_policy_explainer.png"

CROP_SIZE   = 600           # px of tile shown per row
IOP_THRESH  = 0.7   # matches ShadowConditionedDeepForest.CANOPY_IOP_THRESH
SEED        = 7

C_BG          = "#0f172a"   # slate-900
C_PANEL       = "#1e293b"   # slate-800
C_TEXT        = "#e2e8f0"
C_ITC_GT      = "#fbbf24"
C_CANOPY      = "#5eead4"
C_ITC_ANCHOR  = "#fb7185"
C_CAN_ANCHOR  = "#a78bfa"


# ---------------------------------------------------------------- helpers

def pick_crop_window(poly_full_tile, W, H, size):
    """Centre the crop on the polygon centroid, clamp to image bounds."""
    cx, cy = poly_full_tile.centroid.x, poly_full_tile.centroid.y
    x0 = int(np.clip(cx - size / 2, 0, max(0, W - size)))
    y0 = int(np.clip(cy - size / 2, 0, max(0, H - size)))
    return x0, y0


def _iop(anchor_xyxy, poly):
    a = shapely_box(*anchor_xyxy)
    return poly.intersection(a).area / a.area if a.area else 0.0


def make_canopy_anchor_positive(poly_local, size=85, fit_size=CROP_SIZE):
    """Place a square anchor at the polygon centroid (deep inside → high IoP)."""
    cx, cy = poly_local.centroid.x, poly_local.centroid.y
    for dx, dy in [(0, 0), (-25, 0), (25, 0), (0, -25), (0, 25), (-40, 0), (40, 0)]:
        x1, y1 = cx - size / 2 + dx, cy - size / 2 + dy
        if x1 < 8 or y1 < 8 or x1 + size > fit_size - 8 or y1 + size > fit_size - 8:
            continue
        xyxy = (x1, y1, x1 + size, y1 + size)
        iop = _iop(xyxy, poly_local)
        if iop >= IOP_THRESH:
            return xyxy, iop
    # fallback: any anchor at the centroid
    x1, y1 = cx - size / 2, cy - size / 2
    xyxy = (x1, y1, x1 + size, y1 + size)
    return xyxy, _iop(xyxy, poly_local)


def make_canopy_anchor_boundary(poly_local, size=85, fit_size=CROP_SIZE):
    """Place an anchor centred near the polygon exterior — typically yields
    IoP < IOP_THRESH (a negative case)."""
    try:
        boundary = poly_local.exterior
        perim = boundary.length
    except Exception:
        return None
    # Aim for an anchor whose IoP is clearly below the threshold but not zero,
    # so the negative case is visually persuasive (the anchor still overlaps
    # the polygon, but not enough to count).
    target_iop = max(0.15, IOP_THRESH - 0.25)
    best = None  # (xyxy, iop, distance-from-target)
    for frac in np.linspace(0.05, 0.95, 19):
        pt = boundary.interpolate(frac * perim)
        cx, cy = pt.x, pt.y
        x1, y1 = cx - size / 2, cy - size / 2
        if x1 < 8 or y1 < 8 or x1 + size > fit_size - 8 or y1 + size > fit_size - 8:
            continue
        xyxy = (x1, y1, x1 + size, y1 + size)
        iop = _iop(xyxy, poly_local)
        if 0.10 <= iop < IOP_THRESH:
            score = abs(iop - target_iop)
            if best is None or score < best[2]:
                best = (xyxy, iop, score)
    return (best[0], best[1]) if best else None


def make_itc_anchor(box):
    """Anchor slightly larger than an ITC GT box (mimics a matched FPN anchor)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * 1.18, (y2 - y1) * 1.18
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    b_area = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def draw_box(ax, xyxy, *, edge, lw=1.5, ls="-", alpha=1.0):
    x1, y1, x2, y2 = xyxy
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=edge,
        facecolor="none", linestyle=ls, alpha=alpha,
    ))


def draw_label_card(ax, anchor_xyxy, text, colour, corner):
    """Draw a label card pinned to a corner of the panel and an arrow to the anchor.

    corner ∈ {"tr", "tl", "br", "bl"}.
    """
    margin = 18
    if corner == "tr":
        card_xy = (CROP_SIZE - margin, margin)
        ha, va  = "right", "top"
        anchor_pt = (anchor_xyxy[0], (anchor_xyxy[1] + anchor_xyxy[3]) / 2)
    elif corner == "tl":
        card_xy = (margin, margin)
        ha, va  = "left", "top"
        anchor_pt = (anchor_xyxy[2], (anchor_xyxy[1] + anchor_xyxy[3]) / 2)
    elif corner == "br":
        card_xy = (CROP_SIZE - margin, CROP_SIZE - margin)
        ha, va  = "right", "bottom"
        anchor_pt = (anchor_xyxy[0], (anchor_xyxy[1] + anchor_xyxy[3]) / 2)
    else:  # "bl"
        card_xy = (margin, CROP_SIZE - margin)
        ha, va  = "left", "bottom"
        anchor_pt = (anchor_xyxy[2], (anchor_xyxy[1] + anchor_xyxy[3]) / 2)

    t = ax.text(
        card_xy[0], card_xy[1], text,
        ha=ha, va=va, fontsize=10.5, color=C_TEXT,
        family="DejaVu Sans",
        bbox=dict(boxstyle="round,pad=0.55",
                  fc=C_PANEL, ec=colour, lw=1.6, alpha=0.92),
        zorder=10,
    )
    # Build leader line from the edge of the card toward the anchor.
    ax.annotate(
        "", xy=anchor_pt, xytext=card_xy,
        arrowprops=dict(
            arrowstyle="-", color=colour, lw=1.4, alpha=0.85,
            connectionstyle="arc3,rad=0.0",
            shrinkA=14, shrinkB=4,
        ),
        zorder=9,
    )


# ---------------------------------------------------------------- main

def main():
    df  = pd.read_csv(TRAIN_CSV).dropna(subset=["xmin"])
    can = json.loads(CANOPY_JSON.read_text())

    candidate_paths = [
        p for p in df["image_path"].unique()
        if Path(p).name in can and Path(p).exists()
    ]
    random.seed(SEED)
    chosen = random.sample(candidate_paths, 5)

    fig, axes = plt.subplots(
        5, 1, figsize=(9.0, 9.0 * 5),
        gridspec_kw=dict(hspace=0.10, left=0.03, right=0.97,
                         top=0.965, bottom=0.025),
    )
    fig.patch.set_facecolor(C_BG)

    for row_idx, (ax, img_path) in enumerate(zip(axes, chosen)):
        img = np.array(Image.open(img_path))
        H, W = img.shape[:2]
        rows = df[df["image_path"] == img_path]
        itc_full = rows[["xmin", "ymin", "xmax", "ymax"]].values.astype(float)

        polys_flat = can[Path(img_path).name]
        polys = [np.asarray(p, dtype=float).reshape(-1, 2) for p in polys_flat]
        polys = [Polygon(p) for p in polys if len(p) >= 3]
        polys = [p.buffer(0) if not p.is_valid else p for p in polys]
        if not polys:
            continue
        poly_full = max(polys, key=lambda g: g.area)

        # Crop window
        x0, y0 = pick_crop_window(poly_full, W, H, CROP_SIZE)
        crop = img[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]

        # Crop-local polygon
        crop_window = shapely_box(x0, y0, x0 + CROP_SIZE, y0 + CROP_SIZE)
        poly_local  = poly_full.intersection(crop_window)
        if poly_local.is_empty:
            continue
        if poly_local.geom_type == "GeometryCollection":
            parts = [g for g in poly_local.geoms
                     if g.geom_type in ("Polygon", "MultiPolygon")]
            if not parts:
                continue
            poly_local = max(parts, key=lambda g: g.area)
        if poly_local.geom_type == "MultiPolygon":
            poly_local = max(poly_local.geoms, key=lambda g: g.area)
        poly_local  = shapely_translate(poly_local, xoff=-x0, yoff=-y0)
        poly_coords = np.asarray(poly_local.exterior.coords)

        # Crop-local ITC boxes
        itc_local = itc_full.copy()
        itc_local[:, [0, 2]] -= x0
        itc_local[:, [1, 3]] -= y0
        keep = (
            (itc_local[:, 2] > 0) & (itc_local[:, 0] < CROP_SIZE) &
            (itc_local[:, 3] > 0) & (itc_local[:, 1] < CROP_SIZE)
        )
        itc_local = itc_local[keep]
        itc_local[:, [0, 2]] = np.clip(itc_local[:, [0, 2]], 0, CROP_SIZE)
        itc_local[:, [1, 3]] = np.clip(itc_local[:, [1, 3]], 0, CROP_SIZE)

        # ---- draw ----
        ax.imshow(crop)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(C_PANEL); s.set_linewidth(2)

        # Translucent fill
        ax.add_patch(MplPolygon(
            poly_coords, closed=True,
            facecolor=C_CANOPY, edgecolor="none",
            alpha=0.22,
        ))
        # Bright, opaque border on top (with a thin dark stroke so it reads on
        # busy canopy imagery)
        border = MplPolygon(
            poly_coords, closed=True,
            facecolor="none", edgecolor=C_CANOPY,
            alpha=1.0, linewidth=2.6,
        )
        border.set_path_effects([withStroke(linewidth=4.2, foreground=C_BG, alpha=0.6)])
        ax.add_patch(border)
        for box in itc_local:
            draw_box(ax, box, edge=C_ITC_GT, lw=1.3, alpha=0.9)

        # Choose ITC anchor — prefer one in the side AWAY from the canopy poly,
        # so the two label cards don't overlap.
        if len(itc_local):
            cx_poly = poly_local.centroid.x
            distances = np.abs((itc_local[:, 0] + itc_local[:, 2]) / 2 - cx_poly)
            order = np.argsort(distances)[::-1]
            # Pick first one that is fully inside the panel and has decent size
            picked_itc = None
            for idx in order:
                b = itc_local[idx]
                if (b[2] - b[0] >= 25 and b[3] - b[1] >= 25
                        and b[0] > 30 and b[2] < CROP_SIZE - 30
                        and b[1] > 30 and b[3] < CROP_SIZE - 30):
                    picked_itc = b
                    break
            if picked_itc is None and len(itc_local):
                picked_itc = itc_local[order[0]]
        else:
            picked_itc = None

        if picked_itc is not None:
            itc_anchor = make_itc_anchor(picked_itc)
            iou_itc    = iou_xyxy(itc_anchor, tuple(picked_itc))
            draw_box(ax, itc_anchor,
                     edge=C_ITC_ANCHOR, lw=2.2, ls="--", alpha=0.95)
            corner_itc = "tr" if cx_poly < CROP_SIZE / 2 else "tl"
            draw_label_card(
                ax, itc_anchor,
                f"ITC anchor   IoU = {iou_itc:.2f} with GT\n"
                f"→ POSITIVE   (regression ON)",
                colour=C_ITC_ANCHOR, corner=corner_itc,
            )

        # Canopy anchor: for variety, force a boundary (negative) example on
        # the last two panels so colleagues see the threshold in action.
        force_negative = row_idx >= 3
        result = make_canopy_anchor_boundary(poly_local) if force_negative else None
        if result is None:
            canopy_anchor, iop = make_canopy_anchor_positive(poly_local)
        else:
            canopy_anchor, iop = result

        is_positive = iop >= IOP_THRESH
        verdict_txt = "POSITIVE   (regression OFF)" if is_positive else "NEGATIVE   (background)"
        thr_hint    = f"≥ {IOP_THRESH}" if is_positive else f"< {IOP_THRESH}"

        draw_box(ax, canopy_anchor,
                 edge=C_CAN_ANCHOR, lw=2.2, ls="--", alpha=0.95)
        if picked_itc is not None and corner_itc.startswith("t"):
            corner_can = "br" if corner_itc.endswith("r") else "bl"
        else:
            corner_can = "br"
        draw_label_card(
            ax, canopy_anchor,
            f"Canopy anchor   IoP = {iop:.2f}   ({thr_hint})\n"
            f"→ {verdict_txt}",
            colour=C_CAN_ANCHOR, corner=corner_can,
        )

        ax.set_xlim(0, CROP_SIZE); ax.set_ylim(CROP_SIZE, 0)
        # Row label
        ax.text(8, 8, f"  tile {Path(img_path).stem}  ",
                color=C_TEXT, fontsize=9.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc=C_PANEL,
                          ec="none", alpha=0.85), zorder=10,
                path_effects=[withStroke(linewidth=2.5, foreground=C_BG)])

    # ---- legend + title ----
    fig.suptitle(
        "Canopy positive policy — anchor-level loss treatment",
        color="white", fontsize=15, fontweight="bold", y=0.985,
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="none", ec=C_ITC_GT,     lw=1.6, label="ITC GT bbox"),
        plt.Rectangle((0, 0), 1, 1, fc=C_CANOPY, ec=C_CANOPY,   lw=2.4, alpha=0.55, label="Canopy polygon"),
        plt.Rectangle((0, 0), 1, 1, fc="none", ec=C_ITC_ANCHOR, lw=2.2, ls="--", label="ITC anchor (matched)"),
        plt.Rectangle((0, 0), 1, 1, fc="none", ec=C_CAN_ANCHOR, lw=2.2, ls="--", label=f"Canopy anchor (IoP ≥ {IOP_THRESH})"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.973), ncol=4, frameon=False,
        fontsize=10.5, labelcolor="white", handlelength=2.4,
    )

    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
