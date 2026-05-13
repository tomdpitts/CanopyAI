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
    polygon-precise canopy region loss policy.

    Shadow reweighting:
        For GT boxes in images that have a shadow_x/shadow_y annotation, the
        focal loss is upweighted for boxes where shadow evidence is detected
        downstream of the crown.

    Canopy modes (controlled by ``canopy_mode``):
        "ignore"   — anchors whose intersection-over-anchor-area against a
                     canopy polygon ≥ ``canopy_iop_ignore_thresh`` are excluded
                     from the classification loss (neither rewarded nor
                     penalised), mimicking ``iscrowd`` but polygon-precise.
        "upweight" — anchors with IoP ≥ ``canopy_iop_upweight_thresh`` are
                     treated as positives in the focal loss and upweighted by
                     ``canopy_loss_weight``; their regression contribution is
                     suppressed (no precise crown box to regress to).  The
                     summed canopy contribution is scaled by
                     ``canopy_loss_scale`` before being added to the per-image
                     loss, to prevent canopy regions from dominating training.
        "both"     — partial overlap (ignore_thresh ≤ IoP < upweight_thresh) is
                     ignored; full overlap (IoP ≥ upweight_thresh) is
                     upweighted.
    """

    def __init__(
        self,
        train_csv=None,
        val_csv=None,
        shadow_loss_reweight=False,
        shadow_loss_weight=2.0,
        canopy_mode=None,
        canopy_polygons_path=None,
        canopy_loss_weight=2.0,
        canopy_loss_scale=0.5,
        canopy_iop_ignore_thresh=0.1,
        canopy_iop_upweight_thresh=0.4,
        config=None,
        **kwargs,
    ):
        self.shadow_loss_reweight = shadow_loss_reweight
        self.shadow_loss_weight   = shadow_loss_weight

        if canopy_mode not in (None, "ignore", "upweight", "both"):
            raise ValueError(
                f"canopy_mode must be None, 'ignore', 'upweight', or 'both' "
                f"(got {canopy_mode!r})"
            )
        self.canopy_mode                = canopy_mode
        self.canopy_loss_weight         = float(canopy_loss_weight)
        self.canopy_loss_scale          = float(canopy_loss_scale)
        self.canopy_iop_ignore_thresh   = float(canopy_iop_ignore_thresh)
        self.canopy_iop_upweight_thresh = float(canopy_iop_upweight_thresh)

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
        self._current_shadow_gt_weights = None
        self._current_canopy_polygons   = None   # list[list[np.ndarray(V,2)]] per batch
        self._current_image_sizes_pre   = None   # list[(H,W)] pre-transform sizes
        self._current_image_sizes_post  = None   # list[(H,W)] post-transform sizes

        # Canopy polygons lookup: {tile_basename: [np.ndarray(V,2), ...]}.
        # Loaded from JSON written by build_phase30_csvs.py.
        self.canopy_polygons = {}
        if self.canopy_mode is not None:
            if canopy_polygons_path is None:
                raise ValueError(
                    "canopy_mode is set but canopy_polygons_path was not provided"
                )
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

        if self.canopy_mode is not None:
            print(
                f"   ✅ Canopy Loss Policy: mode={self.canopy_mode}  "
                f"weight={self.canopy_loss_weight}x  scale={self.canopy_loss_scale}  "
                f"iop_thresh=ignore≥{self.canopy_iop_ignore_thresh}/"
                f"upweight≥{self.canopy_iop_upweight_thresh}"
            )
        else:
            print("   Canopy Loss Policy: DISABLED")

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
        applies shadow GT weighting AND polygon-precise canopy region
        handling.  Operates at the head level (not classification_head) so it
        has access to ``anchors`` for per-anchor IoP computation.
        """
        if self._head_loss_patched:
            return

        inner = self.model
        if hasattr(inner, "model"):
            inner = inner.model
        if not hasattr(inner, "head") or not hasattr(inner.head, "classification_head"):
            print("   ⚠️  Cannot find RetinaNet head — canopy loss policy disabled")
            self.canopy_mode = None
            self._head_loss_patched = True
            return

        head      = inner.head
        cls_head  = head.classification_head
        reg_head  = head.regression_head
        model_ref = self

        def patched_head_compute_loss(targets, head_outputs, anchors, matched_idxs):
            from torchvision.ops import sigmoid_focal_loss

            cls_logits_all = head_outputs["cls_logits"]
            bbox_reg_all   = head_outputs["bbox_regression"]
            cls_losses     = []
            reg_losses     = []

            shadow_gt_weights  = model_ref._current_shadow_gt_weights
            canopy_polys_batch = model_ref._current_canopy_polygons
            sizes_pre          = model_ref._current_image_sizes_pre  or []
            sizes_post         = model_ref._current_image_sizes_post or []
            ignore_thr   = model_ref.canopy_iop_ignore_thresh
            upweight_thr = model_ref.canopy_iop_upweight_thresh
            mode         = model_ref.canopy_mode

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

                # ----- Per-anchor canopy IoP & masks -----
                canopy_ignore_mask   = torch.zeros(N_anchors, dtype=torch.bool, device=device)
                canopy_upweight_mask = torch.zeros(N_anchors, dtype=torch.bool, device=device)

                if (mode is not None and canopy_polys_batch is not None
                        and img_idx < len(canopy_polys_batch)
                        and canopy_polys_batch[img_idx]
                        and img_idx < len(sizes_pre) and img_idx < len(sizes_post)):
                    pre_h, pre_w  = sizes_pre[img_idx]
                    post_h, post_w = sizes_post[img_idx]
                    scale_w = post_w / max(1, pre_w)
                    scale_h = post_h / max(1, pre_h)
                    integral = model_ref._build_canopy_integral(
                        canopy_polys_batch[img_idx], post_h, post_w, scale_w, scale_h
                    )
                    if integral is not None:
                        integral_t = torch.from_numpy(integral).to(device)
                        iop = model_ref._anchor_iop_from_integral(
                            anchors_per_image, integral_t, post_h, post_w
                        )
                        if mode in ("upweight", "both"):
                            canopy_upweight_mask = iop >= upweight_thr
                        if mode in ("ignore", "both"):
                            canopy_ignore_mask = iop >= ignore_thr
                        if mode == "both":
                            # full-overlap → upweight; partial → ignore (exclusive)
                            canopy_ignore_mask = canopy_ignore_mask & ~canopy_upweight_mask

                # ===== Classification loss =====
                gt_classes_target = torch.zeros_like(cls_logits_per_image)
                gt_classes_target[
                    foreground_idxs,
                    targets_per_image["labels"][matched_idxs_per_image[foreground_idxs]],
                ] = 1.0
                # Treat canopy-upweight anchors as positives for the (single) tree class.
                if canopy_upweight_mask.any():
                    gt_classes_target[canopy_upweight_mask, 0] = 1.0

                effective_valid = valid_idxs & ~canopy_ignore_mask

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

                # Canopy upweighting
                upweight_in_valid = canopy_upweight_mask[effective_valid]
                n_canopy_up       = int(upweight_in_valid.sum().item())
                if n_canopy_up > 0:
                    anchor_weights = anchor_weights.clone()
                    anchor_weights[upweight_in_valid] = (
                        anchor_weights[upweight_in_valid] * model_ref.canopy_loss_weight
                    )

                # Apply canopy_loss_scale to the canopy contribution only.
                if n_canopy_up > 0 and model_ref.canopy_loss_scale != 1.0:
                    non_canopy = ~upweight_in_valid
                    non_canopy_sum = (per_anchor_loss[non_canopy] *
                                      anchor_weights[non_canopy]).sum()
                    canopy_sum     = (per_anchor_loss[upweight_in_valid] *
                                      anchor_weights[upweight_in_valid]).sum()
                    total_loss = non_canopy_sum + canopy_sum * model_ref.canopy_loss_scale
                else:
                    total_loss = (per_anchor_loss * anchor_weights).sum()

                norm = max(1, num_foreground + n_canopy_up)
                cls_losses.append(total_loss / norm)

                # ===== Regression loss =====
                if num_foreground == 0:
                    reg_losses.append(cls_logits_per_image.new_zeros(()))
                    continue

                fg_idxs_pos = foreground_idxs.nonzero(as_tuple=True)[0]
                # Suppress regression for foreground anchors that are also canopy-
                # upweighted: pseudo-canopy GT boxes are not precise crown targets.
                if mode in ("upweight", "both") and canopy_upweight_mask.any():
                    keep = ~canopy_upweight_mask[fg_idxs_pos]
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
        print(f"   ✅ RetinaNet head loss patched — canopy_mode={self.canopy_mode}  "
              f"shadow={self.shadow_loss_reweight}")

    # ------------------------------------------------------------------
    # Lightning overrides
    # ------------------------------------------------------------------

    def on_train_start(self):
        if self.canopy_mode is not None:
            self._patch_transform_capture()
            self._patch_retinanet_head_loss()
        elif self.shadow_loss_reweight and not self._cls_loss_patched:
            self._patch_retinanet_cls_loss()
        if hasattr(super(), "on_train_start"):
            super().on_train_start()

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

        if self.canopy_mode is not None:
            self._current_canopy_polygons = [
                t.get("canopy_polygons", []) if isinstance(t, dict) else []
                for t in targets
            ]
        else:
            self._current_canopy_polygons = None

        return super().training_step(batch, batch_idx)
