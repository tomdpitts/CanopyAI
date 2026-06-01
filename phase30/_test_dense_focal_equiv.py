"""Standalone equivalence check: OLD (gather) vs NEW (dense-masked) cls-loss.

Proves the refactor in models.py `_patch_retinanet_head_loss` is numerically
identical to the gather-based original, so the canopy numbers need no
re-validation.  Run: ./venv310/bin/python phase30/_test_dense_focal_equiv.py
"""
import torch
from torchvision.ops import sigmoid_focal_loss

BETWEEN = -2  # torchvision Matcher.BETWEEN_THRESHOLDS


DTYPE = torch.float32  # overridden in main()


def make_case(N, C, G, seed, with_shadow, scale):
    g = torch.Generator().manual_seed(seed)
    cls_logits = torch.randn(N, C, generator=g).to(DTYPE)

    # matched_idxs in {-2 (between), -1 (bg), 0..G-1 (fg)}
    pick = torch.randint(0, 4, (N,), generator=g)
    matched = torch.full((N,), -1, dtype=torch.long)
    matched[pick == 0] = BETWEEN
    fg_sel = pick >= 2
    if G > 0:
        matched[fg_sel] = torch.randint(0, G, (int(fg_sel.sum()),), generator=g)
    else:
        matched[fg_sel] = -1  # no GT → no foreground

    foreground_idxs = matched >= 0
    valid_idxs = matched != BETWEEN

    # canopy masks: survivors & suppressed are both disjoint from foreground
    # and from each other (mirrors the code's construction).
    non_fg = ~foreground_idxs
    r = torch.rand(N, generator=g)
    canopy_positive_mask = non_fg & (r < 0.10)
    canopy_suppressed_mask = non_fg & (r >= 0.10) & (r < 0.40)

    labels = torch.randint(0, C, (max(G, 1),), generator=g)  # per-GT class
    shadow = (torch.rand(G, generator=g) * 3.0).to(DTYPE) if (with_shadow and G > 0) else (
        torch.empty(0, dtype=DTYPE) if with_shadow else None)

    return dict(cls_logits=cls_logits, matched=matched, foreground_idxs=foreground_idxs,
                valid_idxs=valid_idxs, canopy_positive_mask=canopy_positive_mask,
                canopy_suppressed_mask=canopy_suppressed_mask, labels=labels,
                shadow=shadow, scale=scale, N=N)


def build_targets(c):
    cls_logits = c["cls_logits"]
    gt = torch.zeros_like(cls_logits)
    fg = c["foreground_idxs"]
    if fg.any():
        gt[fg, c["labels"][c["matched"][fg]]] = 1.0
    scale = c["scale"]
    canopy_acts_as_ignore = (scale == 0.0 and c["canopy_positive_mask"].any())
    if c["canopy_positive_mask"].any() and not canopy_acts_as_ignore:
        gt[c["canopy_positive_mask"], 0] = 1.0
    return gt, canopy_acts_as_ignore


def old_loss(c):
    cls_logits = c["cls_logits"]; matched = c["matched"]
    foreground_idxs = c["foreground_idxs"]; valid_idxs = c["valid_idxs"]
    canopy_positive_mask = c["canopy_positive_mask"]
    canopy_suppressed_mask = c["canopy_suppressed_mask"]
    scale = c["scale"]; shadow = c["shadow"]
    num_foreground = int(foreground_idxs.sum().item())
    gt, ignore = build_targets(c)

    if ignore:
        effective_valid = valid_idxs & ~canopy_positive_mask & ~canopy_suppressed_mask
    else:
        effective_valid = valid_idxs & ~canopy_suppressed_mask

    per_anchor_loss = sigmoid_focal_loss(
        cls_logits[effective_valid], gt[effective_valid], reduction="none").sum(dim=-1)
    anchor_weights = torch.ones_like(per_anchor_loss)
    if shadow is not None and len(shadow) > 0:
        w = shadow
        fg_in_valid = foreground_idxs[effective_valid]
        matched_gt = matched[effective_valid][fg_in_valid]
        valid_match = matched_gt < len(w)
        fg_valid_pos = fg_in_valid.nonzero(as_tuple=True)[0]
        anchor_weights[fg_valid_pos[valid_match]] = w[matched_gt[valid_match]]

    if ignore:
        n_canopy_pos = 0
        total = (per_anchor_loss * anchor_weights).sum()
    else:
        positive_in_valid = canopy_positive_mask[effective_valid]
        n_canopy_pos = int(positive_in_valid.sum().item())
        if n_canopy_pos > 0 and scale != 1.0:
            non_canopy = ~positive_in_valid
            ncs = (per_anchor_loss[non_canopy] * anchor_weights[non_canopy]).sum()
            cs = (per_anchor_loss[positive_in_valid] * anchor_weights[positive_in_valid]).sum()
            total = ncs + cs * scale
        else:
            total = (per_anchor_loss * anchor_weights).sum()
    norm = max(1, num_foreground + n_canopy_pos)
    return (total / norm).item()


def new_loss(c):
    cls_logits = c["cls_logits"]; matched = c["matched"]
    foreground_idxs = c["foreground_idxs"]; valid_idxs = c["valid_idxs"]
    canopy_positive_mask = c["canopy_positive_mask"]
    canopy_suppressed_mask = c["canopy_suppressed_mask"]
    scale = c["scale"]; shadow = c["shadow"]
    num_foreground = int(foreground_idxs.sum().item())
    gt, ignore = build_targets(c)

    if ignore:
        effective_valid = valid_idxs & ~canopy_positive_mask & ~canopy_suppressed_mask
    else:
        effective_valid = valid_idxs & ~canopy_suppressed_mask

    valid_f = effective_valid.to(cls_logits.dtype)
    per_anchor_loss = sigmoid_focal_loss(
        cls_logits, gt, reduction="none").sum(dim=-1) * valid_f
    anchor_weights = torch.ones_like(per_anchor_loss)
    if shadow is not None and len(shadow) > 0:
        w = shadow
        matched_gt = matched.clamp(min=0)
        in_range = foreground_idxs & (matched_gt < len(w))
        gathered = w[matched_gt.clamp(max=len(w) - 1)]
        anchor_weights = torch.where(in_range, gathered, anchor_weights)

    weighted = per_anchor_loss * anchor_weights
    if ignore:
        n_canopy_pos = 0
        total = weighted.sum()
    else:
        positive_in_valid = canopy_positive_mask & effective_valid
        n_canopy_pos = int(positive_in_valid.sum().item())
        if n_canopy_pos > 0 and scale != 1.0:
            scale_vec = torch.where(positive_in_valid,
                                    weighted.new_full((), scale), weighted.new_ones(()))
            total = (weighted * scale_vec).sum()
        else:
            total = weighted.sum()
    norm = max(1, num_foreground + n_canopy_pos)
    return (total / norm).item()


def run(dtype, tol):
    global DTYPE
    DTYPE = dtype
    cases = 0; worst = 0.0
    for seed in range(60):
        for C in (1, 2):
            for G in (0, 1, 5, 20):
                for with_shadow in (True, False):
                    for scale in (1.0, 0.5, 0.0):
                        c = make_case(800, C, G, seed * 97 + C * 7 + G, with_shadow, scale)
                        a = old_loss(c); b = new_loss(c)
                        # relative diff (loss magnitudes ~100s; float32 sum noise)
                        d = abs(a - b) / max(1.0, abs(a))
                        worst = max(worst, d); cases += 1
                        if d > tol:
                            print(f"  MISMATCH seed={seed} C={C} G={G} shadow={with_shadow} "
                                  f"scale={scale}: old={a:.10f} new={b:.10f} rel={d:.2e}")
    print(f"{dtype}: {cases} cases, worst relative |old-new| = {worst:.2e} "
          f"-> {'PASS' if worst < tol else 'FAIL'}")
    return worst < tol


def main():
    torch.set_grad_enabled(False)
    # float32: matches the real model; expect tiny summation-order noise (~1e-7 rel).
    ok32 = run(torch.float32, tol=1e-5)
    # float64: removes summation noise; if logic is identical the gap collapses.
    ok64 = run(torch.float64, tol=1e-12)
    print("\nOVERALL:", "PASS — logic identical, float32 gap is summation noise"
          if (ok32 and ok64) else "FAIL")


if __name__ == "__main__":
    main()
