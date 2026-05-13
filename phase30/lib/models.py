import torch
import numpy as np
from deepforest import main as deepforest_main

try:
    from .utils import generate_shadow_map
except ImportError:
    from utils import generate_shadow_map


class ShadowConditionedDeepForest(deepforest_main.deepforest):
    """
    DeepForest fine-tuned with shadow loss reweighting.

    For GT boxes in images that have a shadow_x/shadow_y annotation, the focal
    loss is upweighted for boxes where shadow evidence is detected downstream of
    the crown (shadow_loss_weight× for shadow-casting boxes, 1.0 otherwise).

    Shadow vectors are loaded from the training/validation CSVs via the
    shadow_x/shadow_y columns.  Images without those columns or with NaN values
    receive uniform weights (no reweighting) and are trained as normal.
    """

    def __init__(
        self,
        train_csv=None,
        val_csv=None,
        shadow_loss_reweight=False,
        shadow_loss_weight=2.0,
        config=None,
        **kwargs,
    ):
        self.shadow_loss_reweight = shadow_loss_reweight
        self.shadow_loss_weight   = shadow_loss_weight

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
        self._current_shadow_gt_weights = None

        if self.shadow_loss_reweight:
            print(f"   ✅ Shadow Loss Reweight: ENABLED  weight={self.shadow_loss_weight}x")
        else:
            print("   Shadow Loss Reweight: DISABLED")

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
    # Lightning overrides
    # ------------------------------------------------------------------

    def on_train_start(self):
        if self.shadow_loss_reweight and not self._cls_loss_patched:
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
        return super().training_step(batch, batch_idx)
