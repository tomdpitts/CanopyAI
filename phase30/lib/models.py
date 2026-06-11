import json
from pathlib import Path

import os
import torch
import torch.nn.functional as F
import numpy as np
from deepforest import main as deepforest_main

try:
    from .utils import generate_shadow_map
except ImportError:
    from utils import generate_shadow_map


class ShadowConditionedDeepForest(deepforest_main.deepforest):
    """
    DeepForest fine-tuned with shadow loss reweighting and (optionally) a
    polygon-precise canopy positive policy.

    Shadow reweighting:
        For GT boxes in images that have a shadow_x/shadow_y annotation, the
        focal loss is upweighted for boxes where shadow evidence is detected
        downstream of the crown.

    Canopy positive policy (active when ``canopy_polygons_path`` is supplied):
        An anchor is canopy-positive iff its own-area share inside one single
        canopy polygon clears ``CANOPY_IOP_THRESH`` (0.7).  Size-aware
        IoSmaller suppression at ``CANOPY_NMS_IOU`` (0.2) then keeps the
        largest anchor in each overlap cluster and ignores the rest.  GT-
        matched anchors are exempt from the suppression pool.  Survivors are
        treated as positive examples for the classification focal loss
        (cls target=1) and their regression target is the survivor anchor
        itself (zero offset) — at inference this emits the survivor's own
        geometry, so dense canopy is covered by the large boxes the NMS pass
        chose rather than ITC-shaped boxes from an unsupervised reg head.
        The summed canopy contribution to both cls and reg losses can be
        dampened by ``canopy_loss_scale`` (≤1.0).
    """

    CANOPY_IOP_THRESH = 0.7
    # Intersection-over-Smaller threshold for size-aware suppression among
    # canopy positives: among any cluster of overlapping canopy anchors, the
    # largest is kept and a smaller one is suppressed if its own-area share
    # inside the larger exceeds this value.  Suppressed anchors are ignored
    # by the cls loss (not labelled background), and GT-matched anchors are
    # excluded from the suppression pool.
    CANOPY_NMS_IOU = 0.2

    def __init__(
        self,
        train_csv=None,
        val_csv=None,
        shadow_loss_reweight=False,
        shadow_loss_weight=2.0,
        canopy_polygons_path=None,
        canopy_loss_scale=1.0,
        config=None,
        **kwargs,
    ):
        self.shadow_loss_reweight = shadow_loss_reweight
        self.shadow_loss_weight   = shadow_loss_weight

        self.canopy_enabled    = canopy_polygons_path is not None
        self.canopy_loss_scale = float(canopy_loss_scale)

        deepforest_main.deepforest.__init__(self, config=config, **kwargs)

        # Override mAP to IoU=0.4 (better for aerial tree crown detection than COCO 0.5–0.95)
        from torchmetrics.detection.mean_ap import MeanAveragePrecision as _MAP
        self.mAP_metric = _MAP(iou_thresholds=[0.4])

        # Cap proposals to avoid OOM on 2048×2048 TCD tiles
        try:
            inner = self.model.model if hasattr(self.model, "model") else self.model
            if hasattr(inner, "rpn"):
                inner.rpn.nms_thresh = 0.7
                inner.rpn._post_nms_top_n = {"training": 1000, "testing": 1000}
            if hasattr(inner, "roi_heads"):
                inner.roi_heads.detections_per_img = 500
        except Exception:
            pass

        # Per-image shadow lookup: image_path -> np.array([shadow_x, shadow_y])
        self.shadow_lookup = {}
        for csv_path, label in [(train_csv, "train"), (val_csv, "val")]:
            if csv_path is None:
                continue
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                if "shadow_x" in df.columns and "shadow_y" in df.columns:
                    before = len(self.shadow_lookup)
                    for _, row in df.drop_duplicates("image_path").iterrows():
                        sx, sy = row["shadow_x"], row["shadow_y"]
                        if pd.isna(sx) or pd.isna(sy):
                            continue
                        self.shadow_lookup[row["image_path"]] = np.array(
                            [float(sx), float(sy)], dtype=np.float32
                        )
                    n_loaded  = len(self.shadow_lookup) - before
                    n_skipped = df["image_path"].nunique() - n_loaded
                    print(f"   Loaded {n_loaded} shadow vectors from {label} CSV "
                          f"({n_skipped} skipped — no shadow annotation)")
                else:
                    print(f"   No shadow_x/shadow_y in {label} CSV — loss reweighting inactive")
            except Exception as e:
                print(f"   Warning: could not load {label} shadow lookup: {e}")

        self._cls_loss_patched          = False
        self._head_loss_patched         = False
        self._transform_capture_patched = False
        self._head_loss_invoked         = False  # set True on first patched-head call
        self._current_shadow_gt_weights = None
        self._current_canopy_polygons   = None   # list[list[np.ndarray(V,2)]] per batch
        self._current_image_sizes_pre   = None   # list[(H,W)] pre-transform sizes
        self._current_image_sizes_post  = None   # list[(H,W)] post-transform sizes

        # Canopy polygons lookup: {tile_basename: [np.ndarray(V,2), ...]}.
        # Loaded from JSON written by build_phase30_csvs.py.
        self.canopy_polygons = {}
        if self.canopy_enabled:
            cp_path = Path(canopy_polygons_path)
            if not cp_path.exists():
                raise FileNotFoundError(f"canopy_polygons file not found: {cp_path}")
            raw = json.loads(cp_path.read_text())
            n_polys = 0
            for key, flat_list in raw.items():
                polys = []
                for flat in flat_list:
                    arr = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
                    if len(arr) >= 3:
                        polys.append(arr)
                if polys:
                    self.canopy_polygons[key] = polys
                    n_polys += len(polys)
            print(f"   Loaded {n_polys} canopy polygons across "
                  f"{len(self.canopy_polygons)} tiles from {cp_path.name}")

        if self.shadow_loss_reweight:
            print(f"   ✅ Shadow Loss Reweight: ENABLED  weight={self.shadow_loss_weight}x")
        else:
            print("   Shadow Loss Reweight: DISABLED")

        if self.canopy_enabled:
            print(f"   ✅ Canopy Positive Policy: ENABLED  "
                  f"iop_thresh={self.CANOPY_IOP_THRESH}  "
                  f"scale={self.canopy_loss_scale}")
        else:
            print("   Canopy Positive Policy: DISABLED")

    # ------------------------------------------------------------------
    # Shadow map
    # ------------------------------------------------------------------

    def _compute_shadow_map(self, img_t, shadow_vector):
        """Compute shadow probability map for one [3,H,W] float tensor. Returns [1,H,W]."""
        img_np    = (img_t.permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        angle_deg = float(np.degrees(np.arctan2(float(shadow_vector[0]), float(shadow_vector[1]))))
        shadow_np = generate_shadow_map(img_np, angle_deg)
        return torch.from_numpy(shadow_np).unsqueeze(0)   # [1, H, W]

    # ------------------------------------------------------------------
    # Shadow loss reweighting
    # ------------------------------------------------------------------

    _SLR_PROBE_FRACTIONS = (0.26, 0.54, 0.80, 1.12, 1.52, 2.0)
    _SLR_PROBE_MIN_PX    = 5
    _SLR_SHADOW_THRESH   = 0.35
    _SLR_PROBE_RADIUS    = 2

    def _compute_shadow_gt_weights(self, images, image_paths, targets):
        """
        For each GT box probe the shadow map downstream of the crown.
        Returns a list of (N_gt,) float32 weight tensors.
        Boxes in images without a shadow vector get weight 1.0 (no reweighting).
        """
        result = []
        for img_t, path, target in zip(images, image_paths, targets):
            boxes   = target["boxes"]
            N_gt    = boxes.shape[0]
            weights = torch.ones(N_gt, dtype=torch.float32)

            if N_gt == 0:
                result.append(weights)
                continue

            sv = self.shadow_lookup.get(path)
            if sv is None:
                result.append(weights)
                continue

            sv  = np.array(sv, dtype=np.float32)
            sv  = sv / (np.linalg.norm(sv) + 1e-8)
            sdx =  float(sv[0])
            sdy = -float(sv[1])

            shadow_t = self._compute_shadow_map(img_t, sv)
            sm_np    = shadow_t[0].float().cpu().numpy()
            H, W     = sm_np.shape

            # .float() guards against bf16-mixed autocast — numpy has no bfloat16
            cx = ((boxes[:, 0] + boxes[:, 2]) / 2).float().cpu().numpy()
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2).float().cpu().numpy()
            bw = (boxes[:, 2] - boxes[:, 0]).float().cpu().numpy()
            bh = (boxes[:, 3] - boxes[:, 1]).float().cpu().numpy()

            r = self._SLR_PROBE_RADIUS
            for i in range(N_gt):
                # Ray-box intersection: exact distance from crown centre to box edge
                t_x = ((float(bw[i]) / 2) / abs(sdx)) if abs(sdx) > 1e-6 else float("inf")
                t_y = ((float(bh[i]) / 2) / abs(sdy)) if abs(sdy) > 1e-6 else float("inf")
                edge_dist   = max(min(t_x, t_y), self._SLR_PROBE_MIN_PX)
                probe_dists = [max(f * edge_dist, self._SLR_PROBE_MIN_PX)
                               for f in self._SLR_PROBE_FRACTIONS]
                for d in probe_dists:
                    px  = int(round(cx[i] + d * sdx))
                    py  = int(round(cy[i] + d * sdy))
                    y0c = max(py - r, 0);  y1c = min(py + r + 1, H)
                    x0c = max(px - r, 0);  x1c = min(px + r + 1, W)
                    if y1c > y0c and x1c > x0c:
                        if sm_np[y0c:y1c, x0c:x1c].max() >= self._SLR_SHADOW_THRESH:
                            weights[i] = self.shadow_loss_weight
                            break

            # Negative control (SHADOW_BLIND_CONTROL=1): keep the same NUMBER of
            # upweighted boxes per image as the shadow logic chose, but pick them
            # at RANDOM. If this reproduces the shadow gain, the effect is generic
            # hard-example upweighting, not shadow-specific. (Mirrors the existing
            # control in deepforest_custom/models.py.)
            if os.environ.get("SHADOW_BLIND_CONTROL") == "1":
                k = int((weights != 1.0).sum().item())
                weights = torch.ones(N_gt, dtype=torch.float32)
                if k > 0:
                    idx = np.random.default_rng().choice(N_gt, size=min(k, N_gt),
                                                         replace=False)
                    weights[idx] = self.shadow_loss_weight
                # One-time sanity: confirm the random-reweight branch fired and
                # that it upweights the SAME count of boxes the shadow logic chose
                # (k), so the control is genuinely comparable — not a silent no-op.
                if not getattr(self, "_blind_control_announced", False):
                    import sys as _sys
                    n_up = int((weights != 1.0).sum().item())
                    print(f"   🎲 SHADOW_BLIND_CONTROL active: random reweight branch "
                          f"taken (shadow-selected k={k}, randomly upweighted={n_up}/"
                          f"{N_gt} boxes @ weight={self.shadow_loss_weight}x)",
                          file=_sys.stderr, flush=True)
                    self._blind_control_announced = True

            result.append(weights)
        return result

    # ------------------------------------------------------------------
    # Canopy region handling
    # ------------------------------------------------------------------

    def _build_canopy_integral(self, polygons, post_h, post_w, scale_w, scale_h):
        """Rasterise polygons (pre-transform pixel coords) into a (post_h,post_w)
        uint8 mask scaled by (scale_w, scale_h), then return the (H+1, W+1)
        integral image as a float32 numpy array.  Returns None if no polygons
        or rasterisation fails.
        """
        if not polygons:
            return None
        from PIL import Image, ImageDraw
        try:
            img = Image.new("L", (post_w, post_h), 0)
            drw = ImageDraw.Draw(img)
            for verts in polygons:
                if len(verts) < 3:
                    continue
                pts = [(float(v[0]) * scale_w, float(v[1]) * scale_h) for v in verts]
                drw.polygon(pts, fill=1)
            mask = np.asarray(img, dtype=np.int64)
        except Exception:
            return None
        if mask.sum() == 0:
            return None
        integral = np.zeros((post_h + 1, post_w + 1), dtype=np.float32)
        integral[1:, 1:] = mask.cumsum(0).cumsum(1).astype(np.float32)
        return integral

    def _build_canopy_integral_stack(self, polygons, post_h, post_w,
                                     scale_w, scale_h):
        """Return a single (P, H+1, W+1) float32 numpy stack of integral
        images, one per polygon that rasterises non-empty, or None.

        The head-loss patch takes the max IoP across the P slices, so an
        anchor is canopy-positive iff its own-area share inside a single
        continuous polygon clears ``CANOPY_IOP_THRESH``.  Stacking lets the
        per-polygon integrals reach the GPU in one transfer and lets the IoP
        computation run as a single batched kernel.
        """
        if not polygons:
            return None
        from PIL import Image, ImageDraw
        slices = []
        for verts in polygons:
            if len(verts) < 3:
                continue
            try:
                img = Image.new("L", (post_w, post_h), 0)
                drw = ImageDraw.Draw(img)
                pts = [(float(v[0]) * scale_w, float(v[1]) * scale_h) for v in verts]
                drw.polygon(pts, fill=1)
                mask = np.asarray(img, dtype=np.int64)
            except Exception:
                continue
            if mask.sum() == 0:
                continue
            integral = np.zeros((post_h + 1, post_w + 1), dtype=np.float32)
            integral[1:, 1:] = mask.cumsum(0).cumsum(1).astype(np.float32)
            slices.append(integral)
        if not slices:
            return None
        return np.stack(slices, axis=0)

    @staticmethod
    def _ios_greedy_suppress(boxes, ios_thresh):
        """Greedy size-aware suppression on ``boxes`` (N,4) using
        Intersection-over-Smaller as the overlap metric.  Anchors are
        processed largest area first; a smaller box is suppressed if its
        own-area share inside any kept box exceeds ``ios_thresh``.

        Survivor invariant: for any surviving pair, the smaller box has at
        most ``ios_thresh`` of its area inside the larger.  Returns a long
        tensor of indices into ``boxes`` (matching its device) to keep.

        The greedy pass runs on CPU/numpy.  The loop reads a scalar bool per
        iteration; on GPU that would force a stream sync each step, which
        dominates wall-clock for typical canopy pool sizes (tens of
        thousands of anchors per image).
        """
        N = boxes.shape[0]
        if N == 0:
            return boxes.new_zeros(0, dtype=torch.long)

        device = boxes.device
        boxes_np = boxes.detach().float().cpu().numpy()   # .float(): numpy has no bf16
        areas_np = (
            np.clip(boxes_np[:, 2] - boxes_np[:, 0], 1.0, None)
            * np.clip(boxes_np[:, 3] - boxes_np[:, 1], 1.0, None)
        )
        order = np.argsort(-areas_np)
        sorted_boxes = boxes_np[order]
        sorted_areas = areas_np[order]

        alive = np.ones(N, dtype=bool)
        keep_sorted: list[int] = []
        for k in range(N):
            if not alive[k]:
                continue
            keep_sorted.append(k)
            if k + 1 >= N:
                break
            tail_b = sorted_boxes[k + 1:]
            box_k  = sorted_boxes[k]
            x1 = np.maximum(tail_b[:, 0], box_k[0])
            y1 = np.maximum(tail_b[:, 1], box_k[1])
            x2 = np.minimum(tail_b[:, 2], box_k[2])
            y2 = np.minimum(tail_b[:, 3], box_k[3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            # IoSmaller — the tail box is the smaller (or equal) one because
            # the pool is sorted by area descending.
            ios = inter / sorted_areas[k + 1:]
            alive[k + 1:] &= ~(ios > ios_thresh)

        keep_idx = order[np.asarray(keep_sorted, dtype=np.int64)]
        return torch.from_numpy(keep_idx).to(device=device, dtype=torch.long)

    @staticmethod
    def _anchor_iop_from_integral(anchors, integral_t, post_h, post_w):
        """Vectorised intersection-over-anchor-area via the integral image
        summed-area-table trick.  ``anchors`` is (N,4) on device in post-
        transform pixel coords.  Returns (N,) float tensor in [0,1].
        """
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
        anchor_area = anchor_w * anchor_h
        return (canopy_area / anchor_area).clamp(0.0, 1.0)

    @staticmethod
    def _anchor_max_iop_from_stack(anchors, integral_stack_t, post_h, post_w):
        """Batched form of ``_anchor_iop_from_integral`` over a stack of P
        polygon integrals.  ``integral_stack_t`` is (P, H+1, W+1) on device.
        Returns (N,) — the max IoP across polygons for each anchor.
        """
        x1 = anchors[:, 0].clamp(0, post_w).long()
        y1 = anchors[:, 1].clamp(0, post_h).long()
        x2 = anchors[:, 2].clamp(0, post_w).long()
        y2 = anchors[:, 3].clamp(0, post_h).long()
        # Fancy indexing with (y, x) returns (P, N) since integral_stack_t is (P, H, W).
        A = integral_stack_t[:, y2, x2]
        B = integral_stack_t[:, y2, x1]
        C = integral_stack_t[:, y1, x2]
        D = integral_stack_t[:, y1, x1]
        canopy_area = A - B - C + D                       # (P, N)
        anchor_w = (anchors[:, 2] - anchors[:, 0]).clamp(min=1.0)
        anchor_h = (anchors[:, 3] - anchors[:, 1]).clamp(min=1.0)
        anchor_area = anchor_w * anchor_h                 # (N,)
        iop = (canopy_area / anchor_area).clamp(0.0, 1.0)  # (P, N) via broadcasting
        return iop.max(dim=0).values

    # ------------------------------------------------------------------
    # Transform capture (record post-resize image sizes for canopy IoP)
    # ------------------------------------------------------------------

    def _patch_transform_capture(self):
        """Record pre/post-resize image sizes for the current batch so the
        head loss patch can scale canopy polygons into anchor coordinates.
        """
        if self._transform_capture_patched:
            return
        inner = self.model
        if hasattr(inner, "model"):
            inner = inner.model
        if not hasattr(inner, "transform"):
            return
        transform = inner.transform
        orig_forward = transform.forward
        model_ref    = self

        def patched_forward(images, targets=None):
            pre = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
            image_list, targets_out = orig_forward(images, targets)
            post = [(int(h), int(w)) for h, w in image_list.image_sizes]
            model_ref._current_image_sizes_pre  = pre
            model_ref._current_image_sizes_post = post
            return image_list, targets_out

        transform.forward = patched_forward
        self._transform_capture_patched = True

    def _patch_retinanet_cls_loss(self):
        """Monkey-patch RetinaNet focal loss to apply per-GT shadow weights."""
        if self._cls_loss_patched:
            return

        inner = self.model
        if hasattr(inner, "model"):
            inner = inner.model
        if not hasattr(inner, "head") or not hasattr(inner.head, "classification_head"):
            print("   ⚠️  Cannot find RetinaNet classification_head — shadow loss reweight disabled")
            self.shadow_loss_reweight = False
            self._cls_loss_patched    = True
            return

        cls_head  = inner.head.classification_head
        model_ref = self

        def patched_compute_loss(targets, head_outputs, matched_idxs):
            from torchvision.ops import sigmoid_focal_loss

            losses     = []
            cls_logits = head_outputs["cls_logits"]
            gt_weights = model_ref._current_shadow_gt_weights

            for img_idx, (targets_per_image, cls_logits_per_image, matched_idxs_per_image) in enumerate(
                zip(targets, cls_logits, matched_idxs)
            ):
                foreground_idxs = matched_idxs_per_image >= 0
                num_foreground  = foreground_idxs.sum()
                valid_idxs      = matched_idxs_per_image != cls_head.BETWEEN_THRESHOLDS

                gt_classes_target = torch.zeros_like(cls_logits_per_image)
                gt_classes_target[
                    foreground_idxs,
                    targets_per_image["labels"][matched_idxs_per_image[foreground_idxs]],
                ] = 1.0

                per_anchor_loss = sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs],
                    gt_classes_target[valid_idxs],
                    reduction="none",
                ).sum(dim=-1)

                anchor_weights = torch.ones(
                    valid_idxs.sum(), dtype=per_anchor_loss.dtype, device=per_anchor_loss.device
                )
                if gt_weights is not None and img_idx < len(gt_weights):
                    w            = gt_weights[img_idx].to(per_anchor_loss.device)
                    fg_in_valid  = foreground_idxs[valid_idxs]
                    matched_gt   = matched_idxs_per_image[valid_idxs][fg_in_valid]
                    valid_match  = matched_gt < len(w)
                    fg_valid_pos = fg_in_valid.nonzero(as_tuple=True)[0]
                    anchor_weights[fg_valid_pos[valid_match]] = w[matched_gt[valid_match]]

                losses.append(
                    (per_anchor_loss * anchor_weights).sum() / max(1, num_foreground)
                )

            return sum(losses) / len(targets)

        cls_head.compute_loss = patched_compute_loss
        self._cls_loss_patched = True
        print(f"   ✅ RetinaNet cls loss patched — shadow GT weight={self.shadow_loss_weight}x")

    # ------------------------------------------------------------------
    # Unified head loss patch (shadow + canopy)
    # ------------------------------------------------------------------

    def _patch_retinanet_head_loss(self):
        """Replace RetinaNetHead.compute_loss with a unified version that
        applies shadow GT weighting AND polygon-precise canopy positive
        handling.  Operates at the head level (not classification_head) so
        it has access to ``anchors`` for per-anchor IoP computation.
        """
        if self._head_loss_patched:
            return

        inner = self.model
        if hasattr(inner, "model"):
            inner = inner.model
        if not hasattr(inner, "head") or not hasattr(inner.head, "classification_head"):
            print("   ⚠️  Cannot find RetinaNet head — canopy policy disabled")
            self.canopy_enabled    = False
            self._head_loss_patched = True
            return

        head      = inner.head
        cls_head  = head.classification_head
        reg_head  = head.regression_head
        model_ref = self
        iop_thr   = self.CANOPY_IOP_THRESH
        printed_once = [False]   # closure-captured one-shot diagnostic flag

        def patched_head_compute_loss(targets, head_outputs, anchors, matched_idxs):
            from torchvision.ops import sigmoid_focal_loss

            cls_logits_all = head_outputs["cls_logits"]
            bbox_reg_all   = head_outputs["bbox_regression"]
            cls_losses     = []
            reg_losses     = []

            # One-shot diagnostic accumulators
            diag_n_anchors      = 0
            diag_n_canopy_raw   = 0   # anchors passing IoP>=iop_thr (pre-NMS)
            diag_n_canopy_pos   = 0   # NMS survivors that actually train as positive
            diag_n_foreground   = 0

            shadow_gt_weights  = model_ref._current_shadow_gt_weights
            canopy_polys_batch = model_ref._current_canopy_polygons
            sizes_pre          = model_ref._current_image_sizes_pre  or []
            sizes_post         = model_ref._current_image_sizes_post or []
            canopy_on          = model_ref.canopy_enabled

            for img_idx, (targets_per_image, cls_logits_per_image,
                          bbox_reg_per_image, anchors_per_image,
                          matched_idxs_per_image) in enumerate(zip(
                targets, cls_logits_all, bbox_reg_all, anchors, matched_idxs
            )):
                device    = cls_logits_per_image.device
                N_anchors = anchors_per_image.shape[0]

                foreground_idxs = matched_idxs_per_image >= 0
                num_foreground  = int(foreground_idxs.sum().item())
                valid_idxs      = matched_idxs_per_image != cls_head.BETWEEN_THRESHOLDS

                # ----- Per-anchor canopy IoP & positive mask -----
                canopy_positive_mask   = torch.zeros(N_anchors, dtype=torch.bool, device=device)
                canopy_raw_mask        = torch.zeros(N_anchors, dtype=torch.bool, device=device)
                canopy_suppressed_mask = torch.zeros(N_anchors, dtype=torch.bool, device=device)

                if (canopy_on and canopy_polys_batch is not None
                        and img_idx < len(canopy_polys_batch)
                        and canopy_polys_batch[img_idx]
                        and img_idx < len(sizes_pre) and img_idx < len(sizes_post)):
                    pre_h, pre_w  = sizes_pre[img_idx]
                    post_h, post_w = sizes_post[img_idx]
                    scale_w = post_w / max(1, pre_w)
                    scale_h = post_h / max(1, pre_h)
                    integral_stack = model_ref._build_canopy_integral_stack(
                        canopy_polys_batch[img_idx], post_h, post_w, scale_w, scale_h
                    )
                    if integral_stack is not None:
                        # Anchor qualifies as canopy-positive iff its own-area
                        # share inside a single polygon clears iop_thr — the max
                        # IoP across per-polygon integral images.
                        #
                        # This whole determination is non-differentiable label
                        # assignment, so it runs on CPU — and that is essential,
                        # not incidental.  The IoP kernel's (P, N) shape carries a
                        # data-dependent P (polygons-per-tile) and the NMS-pool
                        # gather is data-dependent too; on MPS every new shape
                        # forces a fresh Metal graph compile + cache entry,
                        # leaking unified memory until jetsam.  CPU torch is eager
                        # (no per-shape compilation), and only the final
                        # fixed-shape (N,) masks cross back to the device.
                        anchors_cpu = anchors_per_image.detach().to("cpu")
                        fg_cpu      = foreground_idxs.detach().to("cpu")
                        integral_stack_t = torch.from_numpy(integral_stack)   # stays on CPU
                        max_iop = model_ref._anchor_max_iop_from_stack(
                            anchors_cpu, integral_stack_t, post_h, post_w
                        )
                        raw_cpu = max_iop >= iop_thr
                        pos_cpu = torch.zeros_like(raw_cpu)
                        sup_cpu = torch.zeros_like(raw_cpu)

                        # Size-aware IoSmaller suppression: keep the largest
                        # anchor in each overlap cluster, suppress smaller
                        # redundants.  GT-matched anchors stay out of the pool
                        # so labelled ITCs keep their fg supervision.
                        nms_pool = raw_cpu & ~fg_cpu
                        if nms_pool.any():
                            pool_idx   = nms_pool.nonzero(as_tuple=True)[0]
                            pool_boxes = anchors_cpu[pool_idx]
                            keep_in_pool = model_ref._ios_greedy_suppress(
                                pool_boxes, model_ref.CANOPY_NMS_IOU
                            )
                            pos_cpu[pool_idx[keep_in_pool]] = True
                            sup_cpu = raw_cpu & ~fg_cpu & ~pos_cpu

                        canopy_raw_mask        = raw_cpu.to(device)
                        canopy_positive_mask   = pos_cpu.to(device)
                        canopy_suppressed_mask = sup_cpu.to(device)

                diag_n_anchors    += N_anchors
                diag_n_canopy_raw += int(canopy_raw_mask.sum().item())
                diag_n_canopy_pos += int(canopy_positive_mask.sum().item())
                diag_n_foreground += num_foreground

                # ===== Classification loss =====
                # canopy_loss_scale == 0 short-circuits to iscrowd semantics:
                # canopy anchors are dropped from cls + denominator entirely.
                canopy_acts_as_ignore = (
                    model_ref.canopy_loss_scale == 0.0 and canopy_positive_mask.any()
                )
                # NMS-suppressed canopy anchors are excluded from the cls
                # loss (never count as positives nor negatives).
                if canopy_acts_as_ignore:
                    effective_valid = (
                        valid_idxs & ~canopy_positive_mask & ~canopy_suppressed_mask
                    )
                else:
                    effective_valid = valid_idxs & ~canopy_suppressed_mask

                gt_classes_target = torch.zeros_like(cls_logits_per_image)
                gt_classes_target[
                    foreground_idxs,
                    targets_per_image["labels"][matched_idxs_per_image[foreground_idxs]],
                ] = 1.0
                # Treat canopy-positive anchors as positives for the (single) tree
                # class — unless scale=0.0, in which case they're ignored above.
                if canopy_positive_mask.any() and not canopy_acts_as_ignore:
                    gt_classes_target[canopy_positive_mask, 0] = 1.0

                # Dense focal loss over ALL anchors → fixed shape (N, C), then
                # masked to zero outside effective_valid.  This is exactly equal
                # to gathering cls_logits[effective_valid] first: the masked-out
                # terms contribute 0 to the sum and carry 0 gradient.  But the
                # tensor shape is now CONSTANT across tiles, so MPS compiles the
                # loss graph once instead of recompiling per data-dependent
                # valid-anchor count (the count swings because canopy suppression
                # removes a variable number of anchors → the MPS graph-cache leak).
                valid_f = effective_valid.to(cls_logits_per_image.dtype)   # (N,)
                per_anchor_loss = sigmoid_focal_loss(
                    cls_logits_per_image,
                    gt_classes_target,
                    reduction="none",
                ).sum(dim=-1) * valid_f                                     # (N,)

                anchor_weights = torch.ones_like(per_anchor_loss)          # (N,)

                # Shadow weights for foreground anchors.  All GT-matched anchors
                # are in effective_valid (foreground ⊆ valid, never suppressed),
                # so weighting them full-length is exact.  No variable gather.
                if shadow_gt_weights is not None and img_idx < len(shadow_gt_weights):
                    w = shadow_gt_weights[img_idx].to(device)
                    if len(w) > 0:
                        matched_gt = matched_idxs_per_image.clamp(min=0)   # (N,)
                        in_range   = foreground_idxs & (matched_gt < len(w))
                        gathered   = w[matched_gt.clamp(max=len(w) - 1)]   # (N,)
                        anchor_weights = torch.where(in_range, gathered, anchor_weights)

                weighted = per_anchor_loss * anchor_weights                # (N,)

                # Canopy contribution can be dampened by canopy_loss_scale (≤1.0)
                # to stop large polygons from outvoting ITC anchors.
                if canopy_acts_as_ignore:
                    n_canopy_pos = 0  # excluded from both numerator and denominator
                    total_loss   = weighted.sum()
                else:
                    positive_in_valid = canopy_positive_mask & effective_valid
                    n_canopy_pos      = int(positive_in_valid.sum().item())

                    if n_canopy_pos > 0 and model_ref.canopy_loss_scale != 1.0:
                        # Scale only the canopy-positive anchors' loss; ×1 elsewhere
                        # (masked-out anchors are already 0, so ×1 keeps them 0).
                        scale_vec = torch.where(
                            positive_in_valid,
                            weighted.new_full((), model_ref.canopy_loss_scale),
                            weighted.new_ones(()),
                        )
                        total_loss = (weighted * scale_vec).sum()
                    else:
                        total_loss = weighted.sum()

                norm = max(1, num_foreground + n_canopy_pos)
                cls_losses.append(total_loss / norm)

                # ===== Regression loss =====
                # GT-matched anchors regress to their matched ITC box.  Canopy
                # survivors regress to themselves (zero offset) so the model
                # emits the survivor anchor's own geometry at inference —
                # under canopy that means large boxes (the IoSmaller NMS keeps
                # the largest in each cluster), not arbitrary ITC-shaped boxes
                # from an unsupervised reg head.  Canopy reg is dampened by
                # canopy_loss_scale and dropped entirely when scale=0.
                fg_idxs_pos = foreground_idxs.nonzero(as_tuple=True)[0]
                if canopy_acts_as_ignore:
                    canopy_reg_idxs = anchors_per_image.new_zeros(
                        0, dtype=torch.long
                    )
                else:
                    canopy_reg_idxs = canopy_positive_mask.nonzero(as_tuple=True)[0]

                n_reg = fg_idxs_pos.numel() + canopy_reg_idxs.numel()
                if n_reg == 0:
                    reg_losses.append(cls_logits_per_image.new_zeros(()))
                    continue

                reg_sum = cls_logits_per_image.new_zeros(())

                if fg_idxs_pos.numel() > 0:
                    matched_gt_boxes = targets_per_image["boxes"][
                        matched_idxs_per_image[fg_idxs_pos]
                    ]
                    anchors_fg  = anchors_per_image[fg_idxs_pos]
                    reg_pred_fg = bbox_reg_per_image[fg_idxs_pos]
                    target_fg   = reg_head.box_coder.encode_single(
                        matched_gt_boxes, anchors_fg
                    )
                    reg_sum = reg_sum + F.l1_loss(
                        reg_pred_fg, target_fg, reduction="sum"
                    )

                if canopy_reg_idxs.numel() > 0:
                    # target_box == anchor_box → encoded offsets are all zero;
                    # no need to call encode_single.
                    reg_pred_canopy = bbox_reg_per_image[canopy_reg_idxs]
                    canopy_reg_sum  = F.l1_loss(
                        reg_pred_canopy,
                        torch.zeros_like(reg_pred_canopy),
                        reduction="sum",
                    )
                    reg_sum = reg_sum + canopy_reg_sum * model_ref.canopy_loss_scale

                reg_losses.append(reg_sum / n_reg)

            if not printed_once[0]:
                printed_once[0] = True
                model_ref._head_loss_invoked = True
                print(f"   🔬 head loss patch INVOKED  batch={len(targets)}  "
                      f"canopy_on={canopy_on}  scale={model_ref.canopy_loss_scale}")
                print(f"      anchors={diag_n_anchors}  "
                      f"iop>={iop_thr}={diag_n_canopy_raw}  "
                      f"after_nms({model_ref.CANOPY_NMS_IOU})={diag_n_canopy_pos}  "
                      f"foreground_gt={diag_n_foreground}")

            N_batch = max(1, len(targets))
            return {
                "classification":  sum(cls_losses) / N_batch,
                "bbox_regression": sum(reg_losses) / N_batch,
            }

        head.compute_loss        = patched_head_compute_loss
        self._head_loss_patched  = True
        # When the head patch is active, classification_head.compute_loss is no
        # longer invoked, so the shadow-only cls patch is unnecessary.
        self._cls_loss_patched   = True
        print(f"   ✅ RetinaNet head loss patched — canopy={self.canopy_enabled}  "
              f"shadow={self.shadow_loss_reweight}")

    # ------------------------------------------------------------------
    # Lightning overrides
    # ------------------------------------------------------------------

    def on_train_start(self):
        if self.canopy_enabled:
            self._patch_transform_capture()
            self._patch_retinanet_head_loss()
        elif self.shadow_loss_reweight and not self._cls_loss_patched:
            self._patch_retinanet_cls_loss()
        if hasattr(super(), "on_train_start"):
            super().on_train_start()

    def on_train_epoch_end(self):
        if self.canopy_enabled and not self._head_loss_invoked:
            print("\n❌ WARNING: head loss patch was NEVER invoked — "
                  "canopy supervision is INACTIVE. Confirm that "
                  "RetinaNet head.compute_loss is reached during training.")
        if hasattr(super(), "on_train_epoch_end"):
            super().on_train_epoch_end()

    def training_step(self, batch, batch_idx):
        images      = batch[0]
        targets     = batch[1]
        image_paths = batch[2] if len(batch) > 2 else [None] * len(images)

        if self.shadow_loss_reweight:
            self._current_shadow_gt_weights = self._compute_shadow_gt_weights(
                images, image_paths, targets
            )
        else:
            self._current_shadow_gt_weights = None

        if self.canopy_enabled:
            self._current_canopy_polygons = [
                t.get("canopy_polygons", []) if isinstance(t, dict) else []
                for t in targets
            ]
        else:
            self._current_canopy_polygons = None

        return super().training_step(batch, batch_idx)

    # ------------------------------------------------------------------
    # Canopy-aware validation
    # ------------------------------------------------------------------

    _VAL_ITC_IOU_THRESH = 0.4   # IoU above which a prediction counts as matched to ITC GT

    def validation_step(self, batch, batch_idx):
        """If canopy is enabled, replicate the base validation_step but filter
        predictions before updating mAP / IoU metrics: drop detections that fall
        substantially inside a canopy polygon AND have no matching ITC GT
        bbox.  Such detections are correct under the canopy positive policy and
        should not be charged as FPs.

        The train-mode forward used by the base validation_step to log
        ``val_loss`` is deliberately omitted — no callback / scheduler consumes
        it any more (LR scheduler now monitors train_loss_epoch; EarlyStopping
        and ModelCheckpoint both monitor map).  Skipping it ~halves val cost.
        """
        # bf16-mixed autocast makes predictions bf16, which DeepForest's
        # format_boxes / torchmetrics convert via .numpy() — numpy has no bfloat16.
        # Validation is cheap and infrequent; run it in fp32 (autocast off on CUDA).
        if not self.canopy_enabled:
            if self.device.type == "cuda":
                with torch.autocast("cuda", enabled=False):
                    return super().validation_step(batch, batch_idx)
            return super().validation_step(batch, batch_idx)

        from torchvision.ops import box_iou
        from deepforest import utilities as _df_utilities

        images, targets, image_names = batch

        # Prediction pass (eval-mode forward) — fp32, see note above.
        self.model.eval()
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast("cuda", enabled=False):
                    preds = self.model.forward(images, targets)
            else:
                preds = self.model.forward(images, targets)

        # Filter predictions: drop in-canopy detections with no ITC GT match.
        iop_thr = self.CANOPY_IOP_THRESH
        iou_thr = self._VAL_ITC_IOU_THRESH
        filtered_preds   = []
        filtered_targets = []
        for img_i, (target, pred, img_t) in enumerate(zip(targets, preds, images)):
            if target["boxes"].shape[0] == 0:
                continue   # base validation_step skips empty-GT — preserve that

            pred_boxes = pred["boxes"]
            polys      = target.get("canopy_polygons", []) if isinstance(target, dict) else []

            if pred_boxes.numel() == 0 or not polys:
                filtered_preds.append(pred)
                filtered_targets.append(target)
                continue

            device = pred_boxes.device
            gt_boxes = target["boxes"].to(device)

            # IoU of each prediction with each ITC GT bbox
            iou = box_iou(pred_boxes, gt_boxes)
            max_iou_per_pred = iou.max(dim=1).values
            matched_to_gt    = max_iou_per_pred >= iou_thr

            # IoP of each (unmatched) prediction against the canopy polygon mask.
            # Predictions live in patch-local pixel coords (RetinaNet's transform
            # rescales them back to the input image size before returning), so
            # the canopy mask is built at the patch size with no scale factor.
            H, W = int(img_t.shape[-2]), int(img_t.shape[-1])
            integral = self._build_canopy_integral(polys, H, W, 1.0, 1.0)
            if integral is None:
                filtered_preds.append(pred)
                filtered_targets.append(target)
                continue

            integral_t = torch.from_numpy(integral).to(device)
            iop = self._anchor_iop_from_integral(pred_boxes, integral_t, H, W)

            # Drop predictions that are mostly inside canopy AND not an ITC match
            drop = (iop >= iop_thr) & (~matched_to_gt)
            if drop.any():
                keep = ~drop
                pred = {
                    "boxes":  pred_boxes[keep],
                    "scores": pred["scores"][keep],
                    "labels": pred["labels"][keep],
                }
            filtered_preds.append(pred)
            filtered_targets.append(target)

        if filtered_targets:
            if hasattr(self, "iou_metric"):
                self.iou_metric.update(filtered_preds, filtered_targets)
            self.mAP_metric.update(filtered_preds, filtered_targets)

        # Preserve the base behaviour of logging unfiltered predictions so any
        # downstream evaluator sees what the model actually predicted.
        for i, result in enumerate(preds):
            try:
                formatted_result = _df_utilities.format_geometry(result)
            except Exception:
                formatted_result = None
            if formatted_result is not None:
                formatted_result["image_path"] = image_names[i]
                self.predictions.append(formatted_result)

        # Lightning accepts None — no callback/scheduler reads the return value here.
        return None
