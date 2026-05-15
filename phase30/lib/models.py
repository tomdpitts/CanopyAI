import json
from pathlib import Path

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
        Anchors whose intersection-over-anchor-area against the canopy
        polygons exceeds ``CANOPY_IOP_THRESH`` (0.7) are treated as normal
        positive examples for the classification focal loss (cls target=1,
        no per-anchor reweight), and their regression contribution is
        suppressed — there is no precise crown box to regress to.  The
        summed canopy contribution can optionally be dampened by
        ``canopy_loss_scale`` (≤1.0) to stop large canopy polygons from
        outvoting ITC anchors.
    """

    CANOPY_IOP_THRESH = 0.7
    # Size-aware suppression over canopy positives: process anchors largest
    # first; suppress every smaller anchor whose own-area share inside the
    # kept anchor exceeds this threshold (Intersection-over-Smaller).  Plain
    # IoU NMS does NOT enforce this — a 32px anchor fully inside a 128px
    # anchor has IoU≈0.06, so all the small redundants survive.  Suppressed
    # anchors become IGNORED in the cls loss (not negative), so the model is
    # not trained to predict background under canopy.  GT-matched anchors
    # are excluded from the pool — labelled ITCs always keep their fg
    # supervision.
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
        img_np    = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
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
            sm_np    = shadow_t[0].cpu().numpy()
            H, W     = sm_np.shape

            cx = ((boxes[:, 0] + boxes[:, 2]) / 2).cpu().numpy()
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2).cpu().numpy()
            bw = (boxes[:, 2] - boxes[:, 0]).cpu().numpy()
            bh = (boxes[:, 3] - boxes[:, 1]).cpu().numpy()

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

    def _build_canopy_integrals_per_poly(self, polygons, post_h, post_w,
                                         scale_w, scale_h):
        """Per-polygon integral images.  Returns a list of (H+1, W+1) float32
        numpy arrays — one per polygon that rasterises non-empty — or None.

        Used by the head-loss patch so the canopy IoP test is run against a
        SINGLE continuous polygon at a time rather than the union of all
        polygons.  An anchor straddling the boundary of two adjacent polygons
        no longer qualifies just because its union-IoP is high.
        """
        if not polygons:
            return None
        from PIL import Image, ImageDraw
        out = []
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
            out.append(integral)
        return out if out else None

    @staticmethod
    def _ios_greedy_suppress(boxes, ios_thresh):
        """Greedy 'NMS' using Intersection-over-Smaller (IoS) as the overlap
        metric instead of IoU.  Boxes are processed largest area first; every
        smaller box whose own-area share inside the kept box exceeds
        ``ios_thresh`` is suppressed.  Survivor invariant: for any pair the
        smaller box has at most ``ios_thresh`` of its area inside the larger.

        ``boxes`` is (N,4) on device.  Returns a long tensor of indices into
        ``boxes`` to keep.
        """
        N = boxes.shape[0]
        if N == 0:
            return boxes.new_zeros(0, dtype=torch.long)

        areas = (
            (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
            * (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
        )
        order = areas.argsort(descending=True)
        sorted_boxes = boxes[order]
        sorted_areas = areas[order]

        # alive[k]=True means index k (in sorted order) is still a candidate.
        alive = torch.ones(N, dtype=torch.bool, device=boxes.device)
        keep_sorted: list[int] = []
        # The number of survivors is small relative to the pool, so a Python
        # loop over kept-only is cheap; each iteration is fully vectorised.
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
            # IoSmaller — denominator is the smaller box's own area (always
            # the tail box here, since tail is strictly smaller-or-equal).
            ios = inter / sorted_areas[tail]
            alive[tail] &= ~(ios > ios_thresh)

        keep_sorted_t = torch.tensor(keep_sorted, dtype=torch.long,
                                     device=boxes.device)
        return order[keep_sorted_t]

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
                    integrals = model_ref._build_canopy_integrals_per_poly(
                        canopy_polys_batch[img_idx], post_h, post_w, scale_w, scale_h
                    )
                    if integrals is not None:
                        # Per-polygon IoP: anchor qualifies iff its own-area
                        # share inside ANY SINGLE polygon exceeds iop_thr.
                        # Computed as max IoP across polygons — straddling two
                        # adjacent polygons no longer qualifies.
                        max_iop = torch.zeros(
                            N_anchors, dtype=torch.float32, device=device
                        )
                        for integral in integrals:
                            integral_t = torch.from_numpy(integral).to(device)
                            iop_p = model_ref._anchor_iop_from_integral(
                                anchors_per_image, integral_t, post_h, post_w
                            )
                            max_iop = torch.maximum(max_iop, iop_p)
                        canopy_raw_mask = max_iop >= iop_thr

                        # Size-aware suppression over canopy-only anchors
                        # (GT-matched anchors excluded so labelled ITCs always
                        # keep their fg supervision).  IoSmaller is used as
                        # the overlap metric — plain IoU NMS fails here
                        # because a small anchor fully inside a large anchor
                        # has IoU = small_area / large_area, often well below
                        # the threshold, so the small redundants survive.
                        nms_pool = canopy_raw_mask & ~foreground_idxs
                        if nms_pool.any():
                            pool_idx   = nms_pool.nonzero(as_tuple=True)[0]
                            pool_boxes = anchors_per_image[pool_idx]
                            keep_in_pool = model_ref._ios_greedy_suppress(
                                pool_boxes, model_ref.CANOPY_NMS_IOU
                            )
                            survivors = torch.zeros_like(canopy_raw_mask)
                            survivors[pool_idx[keep_in_pool]] = True
                            canopy_positive_mask   = survivors
                            canopy_suppressed_mask = (
                                canopy_raw_mask & ~foreground_idxs & ~survivors
                            )

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
                # NMS-suppressed canopy anchors are ALWAYS ignored from the
                # cls loss regardless of canopy_loss_scale — that is the whole
                # point of running NMS over the canopy pool.
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

                per_anchor_loss = sigmoid_focal_loss(
                    cls_logits_per_image[effective_valid],
                    gt_classes_target[effective_valid],
                    reduction="none",
                ).sum(dim=-1)

                anchor_weights = torch.ones_like(per_anchor_loss)

                # Shadow weights for foreground anchors
                if shadow_gt_weights is not None and img_idx < len(shadow_gt_weights):
                    w           = shadow_gt_weights[img_idx].to(device)
                    fg_in_valid = foreground_idxs[effective_valid]
                    matched_gt  = matched_idxs_per_image[effective_valid][fg_in_valid]
                    valid_match = matched_gt < len(w)
                    fg_valid_pos = fg_in_valid.nonzero(as_tuple=True)[0]
                    anchor_weights[fg_valid_pos[valid_match]] = w[matched_gt[valid_match]]

                # Canopy contribution can be dampened by canopy_loss_scale (≤1.0)
                # to stop large polygons from outvoting ITC anchors.
                if canopy_acts_as_ignore:
                    n_canopy_pos = 0  # excluded from both numerator and denominator
                    total_loss   = (per_anchor_loss * anchor_weights).sum()
                else:
                    positive_in_valid = canopy_positive_mask[effective_valid]
                    n_canopy_pos      = int(positive_in_valid.sum().item())

                    if n_canopy_pos > 0 and model_ref.canopy_loss_scale != 1.0:
                        non_canopy = ~positive_in_valid
                        non_canopy_sum = (per_anchor_loss[non_canopy] *
                                          anchor_weights[non_canopy]).sum()
                        canopy_sum     = (per_anchor_loss[positive_in_valid] *
                                          anchor_weights[positive_in_valid]).sum()
                        total_loss = non_canopy_sum + canopy_sum * model_ref.canopy_loss_scale
                    else:
                        total_loss = (per_anchor_loss * anchor_weights).sum()

                norm = max(1, num_foreground + n_canopy_pos)
                cls_losses.append(total_loss / norm)

                # ===== Regression loss =====
                if num_foreground == 0:
                    reg_losses.append(cls_logits_per_image.new_zeros(()))
                    continue

                fg_idxs_pos = foreground_idxs.nonzero(as_tuple=True)[0]
                # Suppress regression for foreground anchors that are also canopy-
                # positive: pseudo-canopy GT boxes are not precise crown targets.
                if canopy_on and canopy_positive_mask.any():
                    keep = ~canopy_positive_mask[fg_idxs_pos]
                    fg_idxs_pos = fg_idxs_pos[keep]

                if fg_idxs_pos.numel() == 0:
                    reg_losses.append(cls_logits_per_image.new_zeros(()))
                    continue

                matched_gt_boxes  = targets_per_image["boxes"][
                    matched_idxs_per_image[fg_idxs_pos]
                ]
                anchors_fg        = anchors_per_image[fg_idxs_pos]
                reg_pred_fg       = bbox_reg_per_image[fg_idxs_pos]
                target_regression = reg_head.box_coder.encode_single(
                    matched_gt_boxes, anchors_fg
                )
                reg_loss_img = F.l1_loss(
                    reg_pred_fg, target_regression, reduction="sum"
                ) / max(1, num_foreground)
                reg_losses.append(reg_loss_img)

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
                  "canopy supervision is INACTIVE.")
            print("   Likely cause: torch.compile interference. Try removing the "
                  "torch.compile call in train_deepforest.py:792, or apply the "
                  "head patch BEFORE compile.")
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
        if not self.canopy_enabled:
            return super().validation_step(batch, batch_idx)

        from torchvision.ops import box_iou
        from deepforest import utilities as _df_utilities

        images, targets, image_names = batch

        # Prediction pass (eval-mode forward).
        self.model.eval()
        with torch.no_grad():
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
