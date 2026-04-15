import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from deepforest import main as deepforest_main

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from .utils import generate_shadow_map, generate_luma_darkness_map   # package import
except ImportError:
    from utils import generate_shadow_map, generate_luma_darkness_map   # script import


# ---------------------------------------------------------------------------
# Shadow Anticipation
# ---------------------------------------------------------------------------

class ShadowAnticipation(nn.Module):
    """
    Geometrically-explicit shadow feature injection at layer2 (H/8) and
    layer3 (H/16).

    Replaces ShadowCrossAttention. Instead of learning a Q-K attention
    mechanism to find crown-shadow correspondences (which requires too much
    data to converge and cannot represent variable tree heights with a single
    scalar), this module directly implements the geometric reasoning:

        "If there is shadow at position p + d * shadow_dir_image,
         there is likely a tree crown at position p."

    For each spatial position the shadow map is sampled at a range of offsets
    d ∈ offsets_px covering the expected range of tree heights. This produces
    a multi-channel 'crown evidence' feature map. A zero-initialised 1×1 conv
    mixes the evidence into the backbone feature channels.

    Gradient chain:  loss → FPN → layer residual → gate × conv(evidence)
    Two hops to the parameters — vastly shorter than cross-attention.
    No dir_scale, no Q-K embedding, no learned geometry.

    offsets_px covers ~3–18 m trees at ~50° sun elevation (0.1 m/px):
        shadow_px ≈ height / tan(elevation)
        3 m → 25 px, 5 m → 42 px, 8 m → 67 px,
        12 m → 101 px, 15 m → 126 px, 18 m → 151 px
    """

    DEFAULT_OFFSETS = (12, 20, 30, 42, 55, 67)  # red/orange zone: 12–67px (~1.4–8m at 50°)

    def __init__(self, offsets_px=None, l2_channels=512, l3_channels=1024):
        super().__init__()
        self.offsets_px = tuple(offsets_px or self.DEFAULT_OFFSETS)
        n = len(self.offsets_px)

        # Layer2 injection: n evidence channels → 512 (zero-init → identity at init)
        self.l2_conv = nn.Conv2d(n, l2_channels, kernel_size=1)
        nn.init.zeros_(self.l2_conv.weight)
        nn.init.zeros_(self.l2_conv.bias)
        self.gate_l2 = nn.Parameter(torch.ones(1) * 0.01)

        # Layer3 injection: n evidence channels → 1024 (zero-init → identity at init)
        self.l3_conv = nn.Conv2d(n, l3_channels, kernel_size=1)
        nn.init.zeros_(self.l3_conv.weight)
        nn.init.zeros_(self.l3_conv.bias)
        self.gate_l3 = nn.Parameter(torch.ones(1) * 0.01)

    def _warp_stack(self, shadow_map, shadow_dir):
        """
        For each offset d, warp shadow_map so that output[p] = shadow_map[p + d*dir].
        Returns (B, n_offsets, H, W).

        shadow_map:  (B, 1, H, W)  in [0, 1]
        shadow_dir:  (B, 2)  (sin_az, cos_az) — geographic convention
        """
        B, _, H, W = shadow_map.shape
        dev   = shadow_map.device
        dtype = shadow_map.dtype

        # Convert to image coordinates: x-right, y-down → flip y component
        sdx = shadow_dir[:, 0]        # (B,)  x
        sdy = -shadow_dir[:, 1]       # (B,)  y  (cos_az flipped)

        # Base identity grid — computed once, reused for all offsets
        theta     = torch.eye(2, 3, device=dev, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        base_grid = F.affine_grid(theta, (B, 1, H, W), align_corners=True)  # (B, H, W, 2)

        slices = []
        for d in self.offsets_px:
            # Shift in normalised [-1,1] coords so output[h,w] samples input[h+dy, w+dx]
            dx_n = sdx * d * 2.0 / max(W - 1, 1)   # (B,)
            dy_n = sdy * d * 2.0 / max(H - 1, 1)   # (B,)
            shift = torch.stack([dx_n, dy_n], dim=-1)[:, None, None, :]  # (B,1,1,2)

            warped = F.grid_sample(
                shadow_map, base_grid + shift,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )  # (B, 1, H, W)
            slices.append(warped)

        return torch.cat(slices, dim=1)  # (B, n_offsets, H, W)

    def forward_l2(self, rgb_feat, shadow_map, shadow_dir):
        """
        rgb_feat:   (B, 512,  H/8,  W/8)   — ResNet layer2 output
        shadow_map: (B, 1,    H,    W)      — full-resolution shadow probability
        shadow_dir: (B, 2)
        Returns:    (B, 512,  H/8,  W/8)    — zero-perturbation at init
        """
        evidence = self._warp_stack(shadow_map, shadow_dir)            # (B, n, H, W)
        _, _, Hf, Wf = rgb_feat.shape
        evidence = F.interpolate(evidence, size=(Hf, Wf),
                                 mode='bilinear', align_corners=False)  # (B, n, H/8, W/8)
        return rgb_feat + self.gate_l2 * self.l2_conv(evidence)

    def forward_l3(self, rgb_feat, shadow_map, shadow_dir):
        """
        rgb_feat:   (B, 1024, H/16, W/16)  — ResNet layer3 output
        shadow_map: (B, 1,    H,    W)      — full-resolution shadow probability
        shadow_dir: (B, 2)
        Returns:    (B, 1024, H/16, W/16)   — zero-perturbation at init
        """
        evidence = self._warp_stack(shadow_map, shadow_dir)            # (B, n, H, W)
        _, _, Hf, Wf = rgb_feat.shape
        evidence = F.interpolate(evidence, size=(Hf, Wf),
                                 mode='bilinear', align_corners=False)  # (B, n, H/16, W/16)
        return rgb_feat + self.gate_l3 * self.l3_conv(evidence)




# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class ShadowConditionedDeepForest(deepforest_main.deepforest):
    """
    DeepForest fine-tuned with optional shadow awareness.

    Two orthogonal shadow mechanisms, independently togglable:

    1. shadow_channel=True  — shadow probability map is prepended as a 4th
       input channel to conv1 (widen_first_conv_for_shadow_channel must be
       called after weight loading in train_deepforest.py).

    2. shadow_cross_attention=True — ShadowAnticipation is hooked into ResNet
       layer2 (H/8) and layer3 (H/16). For each backbone position p, the shadow
       map is sampled at p + d*shadow_dir for a range of offsets d covering 3–18 m
       trees. This directly implements the geometric reasoning "shadow at expected
       offset → crown here" without learned Q-K attention or dir_scale.

    Both mechanisms can be combined (Run D in the ablation study).
    """

    def __init__(
        self,
        shadow_angle_deg=215.0,
        train_csv=None,
        val_csv=None,
        freeze_backbone=False,
        shadow_channel=False,
        shadow_cross_attention=False,
        shadow_luma_only=False,   # ablation: replace directional shadow map with luma darkness map
        shadow_input_only=False,  # ablation F: replace RGB entirely with shadow map (tiled ×3)
        config=None,
        **kwargs,
    ):
        self.shadow_channel = shadow_channel
        self.shadow_cross_attention = shadow_cross_attention
        self.shadow_luma_only = shadow_luma_only
        self.shadow_input_only = shadow_input_only

        # Initialize DeepForest (LightningModule)
        deepforest_main.deepforest.__init__(self, config=config, **kwargs)

        # Override DeepForest's default COCO mAP metric (averages IoU 0.5→0.95) with
        # IoU=0.4 to match evaluate_boxes threshold and suit aerial tree crown detection,
        # where predictions tighter than the GT box are still valid detections.
        from torchmetrics.detection.mean_ap import MeanAveragePrecision as _MAP
        self.mAP_metric = _MAP(iou_thresholds=[0.4])

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            # Freeze only the ResNet body — NOT the FPN or detection head.
            # The FPN and head must remain trainable so they can adapt to the modified
            # layer3 features that SCA produces. Freezing everything blocks all gradient
            # signal to SCA and prevents any learning.
            frozen = False
            try:
                if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "body"):
                    body = self.model.backbone.body
                elif (hasattr(self.model, "model") and
                      hasattr(self.model.model, "backbone") and
                      hasattr(self.model.model.backbone, "body")):
                    body = self.model.model.backbone.body
                else:
                    body = None

                if body is not None:
                    for param in body.parameters():
                        param.requires_grad = False
                    print("   ❄️  Froze ResNet body only (FPN + detection head remain trainable)")
                    frozen = True
            except Exception as e:
                print(f"   ⚠️  Could not resolve ResNet body for freezing: {e}")

            if not frozen:
                # Fallback: freeze everything (old behaviour)
                for param in self.model.parameters():
                    param.requires_grad = False
                print("   ❄️  Fallback: froze entire model.model (could not resolve body)")

        # Store base shadow angle for fallback
        self.shadow_angle_deg = shadow_angle_deg
        shadow_angle_rad = np.radians(shadow_angle_deg)
        self.base_shadow_vector = torch.tensor(
            [np.sin(shadow_angle_rad), np.cos(shadow_angle_rad)], dtype=torch.float32
        )

        # Per-image shadow lookup: image_path -> np.array([shadow_x, shadow_y])
        self.shadow_lookup = {}
        # Per-image global normalization stats: image_path -> (dg_scale, dark_scale, otsu_ctr)
        self.norm_stats_lookup = {}
        # Per-image domain lookup: image_path -> domain string (e.g. "BRU", "WON")
        self.domain_lookup = {}
        for csv_path, label in [(train_csv, "train"), (val_csv, "val")]:
            if csv_path is None:
                continue
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                if "shadow_x" in df.columns and "shadow_y" in df.columns:
                    before = len(self.shadow_lookup)
                    for _, row in df.drop_duplicates("image_path").iterrows():
                        self.shadow_lookup[row["image_path"]] = np.array(
                            [row["shadow_x"], row["shadow_y"]], dtype=np.float32
                        )
                    print(f"   Loaded {len(self.shadow_lookup)-before} shadow vectors from {label} CSV")
                else:
                    print(f"   No shadow_x/shadow_y in {label} CSV — will use angle fallback")
                if all(c in df.columns for c in ["dg_scale", "dark_scale", "otsu_ctr"]):
                    before = len(self.norm_stats_lookup)
                    for _, row in df.drop_duplicates("image_path").iterrows():
                        import pandas as _pd
                        self.norm_stats_lookup[row["image_path"]] = (
                            float(row["dg_scale"]) if _pd.notna(row["dg_scale"]) else None,
                            float(row["dark_scale"]) if _pd.notna(row["dark_scale"]) else None,
                            float(row["otsu_ctr"]) if _pd.notna(row["otsu_ctr"]) else None,
                        )
                    print(f"   Loaded {len(self.norm_stats_lookup)-before} shadow norm stats from {label} CSV")
                else:
                    print(f"   No dg_scale/dark_scale/otsu_ctr in {label} CSV — shadow maps will use per-tile stats")
                if "domain" in df.columns:
                    for _, row in df.drop_duplicates("image_path").iterrows():
                        self.domain_lookup[row["image_path"]] = str(row["domain"]).upper()
                    domains_found = sorted(set(self.domain_lookup.values()))
                    print(f"   Loaded domain labels from {label} CSV: {domains_found}")
            except Exception as e:
                print(f"   Warning: could not load {label} shadow lookup: {e}")

        # Per-domain MAP accumulators — populated lazily from domain_lookup on first val batch
        self._domain_map_metrics = {}   # domain -> torchmetrics.MeanAveragePrecision
        self._domain_val_preds   = {}   # domain -> list of prediction dicts (accumulated)
        self._domain_val_targets = {}   # domain -> list of target dicts (accumulated)

        # Shadow Anticipation modules
        if self.shadow_cross_attention:
            self.shadow_anticipation = ShadowAnticipation()
            print("   ✅ Shadow Anticipation: ENABLED (layer2/H8 + layer3/H16, identity at init)")
        else:
            print("   Shadow Anticipation: DISABLED")

        # Flag to track hook registration
        self._sca_hook_injected = False
        self._current_shadow_map = None  # (B, 1, H, W) raw shadow maps, set per forward pass
        self._current_shadow_dir = None  # (B, 2) shadow direction vectors, set per forward pass

        # Shadow channel flag is serialised in checkpoint for inference detection
        self.shadow_channel = shadow_channel

    # ------------------------------------------------------------------
    # Shadow map computation helpers
    # ------------------------------------------------------------------

    # Per-domain luma ceiling for shadow map generation.
    # WON: arid sandy terrain — shadows on pale soil have higher luma than BRU.
    # BRU: darker soil/vegetation — default threshold (71) is appropriate.
    _SHADOW_ABS_LUMA_MAX = {"WON": 150, "BRU": 71}
    _SHADOW_ABS_LUMA_MAX_DEFAULT = 71

    def _compute_shadow_map(self, img_t, shadow_vector, norm_stats=None, domain=None):
        """
        Compute a single shadow probability map for one [3,H,W] float image tensor.

        - Uses the shadow_x/shadow_y vector to derive the angle.
        - norm_stats: (dg_scale, dark_scale, otsu_ctr) pre-computed from the full source
          orthomosaic so that normalization is globally consistent across tiles.
          When None, falls back to per-tile stats (not recommended for training).
        - domain: 'WON' or 'BRU' — selects per-domain abs_luma_max.
        Returns a [1, H, W] float32 tensor in [0, 1].
        """
        img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        luma_max = self._SHADOW_ABS_LUMA_MAX.get(
            str(domain).upper() if domain else "",
            self._SHADOW_ABS_LUMA_MAX_DEFAULT,
        )

        if self.shadow_luma_only:
            # Ablation: direction-free darkness map — no shadow vector used.
            # Isolates the contribution of the directional shadow angle.
            shadow_np = generate_luma_darkness_map(img_np, abs_luma_max=luma_max)
        else:
            angle = float(np.degrees(np.arctan2(shadow_vector[0], shadow_vector[1])))
            if norm_stats is not None:
                dg_scale, dark_scale, otsu_ctr = norm_stats
                shadow_np = generate_shadow_map(img_np, angle,
                                                dg_scale=dg_scale,
                                                dark_scale=dark_scale,
                                                otsu_ctr=otsu_ctr,
                                                abs_luma_max=luma_max)
            else:
                shadow_np = generate_shadow_map(img_np, angle, abs_luma_max=luma_max)
        return torch.from_numpy(shadow_np).unsqueeze(0)   # [1, H, W]

    def _prepend_shadow_channel(self, images, image_paths):
        """
        Prepend a shadow probability map as a 4th channel.
        Images: list of [3, H, W] float tensors in [0,1].
        Returns: list of [4, H, W] float tensors.
        """
        result = []
        for img_t, path in zip(images, image_paths):
            sv = self.shadow_lookup.get(path)
            if sv is None:
                shadow_t = torch.zeros(1, img_t.shape[1], img_t.shape[2],
                                       dtype=img_t.dtype, device=img_t.device)
            else:
                norm_stats = self.norm_stats_lookup.get(path) or self.norm_stats_lookup.get(None)
                domain = self.domain_lookup.get(path)
                shadow_t = self._compute_shadow_map(img_t, sv, norm_stats=norm_stats, domain=domain)
                shadow_t = shadow_t.to(img_t.device, dtype=img_t.dtype)
            result.append(torch.cat([img_t, shadow_t], dim=0))  # [4, H, W]
        return result

    def _replace_with_shadow_input(self, images, image_paths):
        """
        Ablation F: discard RGB entirely, replace each image with the shadow map
        tiled 3× across channels.  Input to backbone is still (3, H, W) so no
        architecture change is needed.
        """
        result = []
        for img_t, path in zip(images, image_paths):
            sv = self.shadow_lookup.get(path)
            if sv is None:
                shadow_t = torch.zeros(1, img_t.shape[1], img_t.shape[2],
                                       dtype=img_t.dtype, device=img_t.device)
            else:
                norm_stats = self.norm_stats_lookup.get(path) or self.norm_stats_lookup.get(None)
                domain = self.domain_lookup.get(path)
                shadow_t = self._compute_shadow_map(img_t, sv, norm_stats=norm_stats, domain=domain)
                shadow_t = shadow_t.to(img_t.device, dtype=img_t.dtype)
            result.append(shadow_t.repeat(3, 1, 1))  # [3, H, W]
        return result

    def _compute_shadow_dir_batch(self, images, image_paths):
        """
        Return (B, 2) float32 tensor of normalised shadow direction unit vectors
        (sin_az, cos_az) for each image in the batch, sourced from shadow_lookup
        or the angle fallback.
        """
        device = images[0].device
        dirs = []
        for path in image_paths:
            sv = self.shadow_lookup.get(path)
            if sv is None:
                angle_rad = np.radians(self.shadow_angle_deg)
                sv = np.array([np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float32)
            sv = np.array(sv, dtype=np.float32)
            sv = sv / (np.linalg.norm(sv) + 1e-8)
            dirs.append(torch.from_numpy(sv))
        return torch.stack(dirs).to(device)  # (B, 2)

    def _compute_shadow_map_batch(self, images, image_paths):
        """
        Compute raw shadow probability maps for a batch.
        Returns tensor (B, 1, H, W) float32 in [0, 1] on the same device as images.
        All maps are resized to match the first image's spatial dimensions.
        """
        device = images[0].device
        target_h, target_w = images[0].shape[1], images[0].shape[2]
        shadow_maps = []
        for img_t, path in zip(images, image_paths):
            sv = self.shadow_lookup.get(path)
            if sv is None:
                sv = np.array([np.sin(np.radians(self.shadow_angle_deg)),
                               np.cos(np.radians(self.shadow_angle_deg))], dtype=np.float32)
            norm_stats = self.norm_stats_lookup.get(path) or self.norm_stats_lookup.get(None)
            domain = self.domain_lookup.get(path)
            shadow_t = self._compute_shadow_map(img_t, sv, norm_stats=norm_stats, domain=domain)  # [1, H_i, W_i]
            if shadow_t.shape[1] != target_h or shadow_t.shape[2] != target_w:
                shadow_t = F.interpolate(
                    shadow_t.unsqueeze(0), size=(target_h, target_w),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
            shadow_maps.append(shadow_t)
        return torch.stack(shadow_maps).to(device)  # (B, 1, H, W)

    # ------------------------------------------------------------------
    # Hook injection for Shadow Cross-Attention
    # ------------------------------------------------------------------

    def _inject_sca_hook(self):
        """Register forward hooks on ResNet layer2 and layer3 for shadow anticipation."""
        if self._sca_hook_injected:
            return

        def layer2_hook(module, input, output):
            if self._current_shadow_map is not None and self._current_shadow_dir is not None:
                return self.shadow_anticipation.forward_l2(
                    output, self._current_shadow_map, self._current_shadow_dir
                )
            return output

        def layer3_hook(module, input, output):
            if self._current_shadow_map is not None and self._current_shadow_dir is not None:
                return self.shadow_anticipation.forward_l3(
                    output, self._current_shadow_map, self._current_shadow_dir
                )
            return output

        try:
            body = None
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "body"):
                body = self.model.backbone.body
            elif (hasattr(self.model, "model") and
                  hasattr(self.model.model, "backbone") and
                  hasattr(self.model.model.backbone, "body")):
                body = self.model.model.backbone.body

            hooked = []
            if body is not None:
                if hasattr(body, "layer2"):
                    body.layer2.register_forward_hook(layer2_hook)
                    hooked.append("layer2 (H/8)")
                if hasattr(body, "layer3"):
                    body.layer3.register_forward_hook(layer3_hook)
                    hooked.append("layer3 (H/16)")
            if hooked:
                print(f"   ✅ Shadow Anticipation hooks injected on {', '.join(hooked)}")
            else:
                print("   ⚠️  Could not find layer2/layer3 to inject shadow hooks")
        except Exception as e:
            print(f"   ⚠️  Shadow hook injection failed: {e}")

        self._sca_hook_injected = True

    # ------------------------------------------------------------------
    # Lightning lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_start(self):
        if self.shadow_cross_attention and not self._sca_hook_injected:
            self._inject_sca_hook()
        if hasattr(super(), "on_train_start"):
            super().on_train_start()

    def on_validation_start(self):
        if self.shadow_cross_attention and not self._sca_hook_injected:
            self._inject_sca_hook()
        if hasattr(super(), "on_validation_start"):
            super().on_validation_start()

    def on_predict_start(self):
        if self.shadow_cross_attention and not self._sca_hook_injected:
            self._inject_sca_hook()
        if hasattr(super(), "on_predict_start"):
            super().on_predict_start()

    def on_train_epoch_end(self):
        """Log Shadow Anticipation diagnostic scalars once per epoch."""
        if self.shadow_cross_attention:
            sa = self.shadow_anticipation

            # Gate values: grow from 0.01 as the anticipation residuals become useful.
            self.log("sca/gate_l2", sa.gate_l2.item(), prog_bar=True)
            self.log("sca/gate_l3", sa.gate_l3.item(), prog_bar=False)

            # Gradient norms: confirm the gradient chain is live at layer2 and layer3.
            # Both should be non-zero from epoch 1 (very short chain: loss→FPN→gate×conv).
            g_l2 = sa.gate_l2.grad
            g_l3 = sa.gate_l3.grad
            w_l2 = sa.l2_conv.weight.grad
            w_l3 = sa.l3_conv.weight.grad
            self.log("sca/grad_gate_l2",   g_l2.norm().item() if g_l2 is not None else 0.0, prog_bar=False)
            self.log("sca/grad_gate_l3",   g_l3.norm().item() if g_l3 is not None else 0.0, prog_bar=False)
            self.log("sca/grad_l2_conv",   w_l2.norm().item() if w_l2 is not None else 0.0, prog_bar=False)
            self.log("sca/grad_l3_conv",   w_l3.norm().item() if w_l3 is not None else 0.0, prog_bar=False)

        if hasattr(super(), "on_train_epoch_end"):
            super().on_train_epoch_end()

    # ------------------------------------------------------------------
    # WON bbox normalisation
    # ------------------------------------------------------------------

    # WON annotations were drawn to include the cast shadow as well as the crown,
    # making them substantially larger than equivalent BRU annotations which are
    # tight to the canopy only. This inconsistency confuses both the detector and
    # the ShadowAnticipation module (which expects the shadow to lie OUTSIDE the box).
    #
    # We correct for this at training/validation time by shrinking WON boxes to
    # WON_BBOX_AREA_FRACTION of their original area, preserving aspect ratio and
    # centre. This is applied in code only — the source CSVs are NOT modified so
    # the transform is fully reversible and auditable.
    #
    # 0.50 → sides scaled by √0.5 ≈ 0.707  (chosen by visual inspection)
    WON_BBOX_AREA_FRACTION = 0.50

    def _maybe_shrink_won_targets(self, targets, image_paths):
        """
        For WON images, shrink bbox coordinates to WON_BBOX_AREA_FRACTION of
        their original area, keeping aspect ratio and centre unchanged.

        targets:     list of dicts with "boxes" key, each (N, 4) tensor [x1,y1,x2,y2]
        image_paths: list of str (or None) aligned with targets

        Returns a new list of target dicts — originals are not mutated.
        """
        if not self.domain_lookup:
            return targets  # no domain info loaded, skip

        scale = self.WON_BBOX_AREA_FRACTION ** 0.5  # ≈ 0.707
        result = []
        for target, path in zip(targets, image_paths):
            if self.domain_lookup.get(path, "").upper() != "WON":
                result.append(target)
                continue

            boxes = target["boxes"]  # (N, 4)
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            hw = (boxes[:, 2] - boxes[:, 0]) * scale / 2  # half-width
            hh = (boxes[:, 3] - boxes[:, 1]) * scale / 2  # half-height
            new_boxes = torch.stack(
                [cx - hw, cy - hh, cx + hw, cy + hh], dim=1
            )
            result.append({**target, "boxes": new_boxes})

        return result

    # ------------------------------------------------------------------
    # Training / Validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        images      = batch[0]
        targets     = batch[1]
        image_paths = batch[2] if len(batch) > 2 else [None] * len(images)

        # Normalise WON bbox sizes before loss computation
        targets = self._maybe_shrink_won_targets(targets, image_paths)

        # Compute raw shadow maps and direction vectors (read by layer2/layer3 hooks)
        if self.shadow_cross_attention:
            self._current_shadow_map = self._compute_shadow_map_batch(images, image_paths)
            self._current_shadow_dir = self._compute_shadow_dir_batch(images, image_paths)
        else:
            self._current_shadow_map = None
            self._current_shadow_dir = None

        # Replace RGB with shadow map (ablation F) or prepend as 4th channel
        if self.shadow_input_only:
            images = self._replace_with_shadow_input(images, image_paths)
        elif self.shadow_channel:
            images = self._prepend_shadow_channel(images, image_paths)

        batch = (images, targets, *batch[2:])
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        images      = batch[0]
        targets     = batch[1]
        image_paths = batch[2] if len(batch) > 2 else [None] * len(images)

        # Normalise WON bbox sizes before metric computation (same transform as training)
        targets = self._maybe_shrink_won_targets(targets, image_paths)

        if self.shadow_cross_attention:
            self._current_shadow_map = self._compute_shadow_map_batch(images, image_paths)
            self._current_shadow_dir = self._compute_shadow_dir_batch(images, image_paths)
        else:
            self._current_shadow_map = None
            self._current_shadow_dir = None

        if self.shadow_input_only:
            images = self._replace_with_shadow_input(images, image_paths)
        elif self.shadow_channel:
            images = self._prepend_shadow_channel(images, image_paths)

        batch  = (images, targets, *batch[2:])
        result = super().validation_step(batch, batch_idx)

        # Per-domain prediction accumulation — runs model a second time in eval mode.
        # Only active when domain_lookup is populated (CSV has a 'domain' column).
        if self.domain_lookup:
            try:
                self.eval()
                with torch.no_grad():
                    raw_preds = self.model(images)
                for pred, target, path in zip(raw_preds, targets, image_paths):
                    domain = self.domain_lookup.get(path)
                    if domain is None:
                        continue
                    if domain not in self._domain_val_preds:
                        self._domain_val_preds[domain]   = []
                        self._domain_val_targets[domain] = []
                    # torchmetrics MAP format: boxes, scores, labels
                    self._domain_val_preds[domain].append({
                        "boxes":  pred["boxes"].cpu(),
                        "scores": pred["scores"].cpu(),
                        "labels": pred["labels"].cpu(),
                    })
                    self._domain_val_targets[domain].append({
                        "boxes":  target["boxes"].cpu(),
                        "labels": target["labels"].cpu(),
                    })
            except Exception:
                pass  # non-fatal — don't break training over a diagnostic

        return result

    def on_validation_epoch_end(self):
        """Compute and log per-domain mAP from accumulated validation predictions."""
        if not self._domain_val_preds:
            return

        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            for domain, preds in self._domain_val_preds.items():
                targets = self._domain_val_targets[domain]
                # Use IoU=0.4 to match DeepForest's evaluate_boxes threshold.
                # COCO default (0.5-0.95 average) is too strict for aerial tree crown
                # detection where predictions are often tighter than GT annotations.
                metric  = MeanAveragePrecision(iou_thresholds=[0.4])
                metric.update(preds, targets)
                result = metric.compute()
                map_val = result.get("map", torch.tensor(0.0)).item()
                self.log(f"map_{domain.lower()}", map_val,
                         prog_bar=True, on_epoch=True, sync_dist=False)
        except Exception:
            pass
        finally:
            self._domain_val_preds.clear()
            self._domain_val_targets.clear()

        if hasattr(super(), "on_validation_epoch_end"):
            super().on_validation_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (list, tuple)) and len(batch) > 2:
            images      = batch[0]
            image_paths = batch[2]
        elif isinstance(batch, list):
            images      = batch
            image_paths = [None] * len(images)
        else:
            images      = list(batch)
            image_paths = [None] * len(images)

        if self.shadow_cross_attention:
            self._current_shadow_map = self._compute_shadow_map_batch(images, image_paths)
            self._current_shadow_dir = self._compute_shadow_dir_batch(images, image_paths)
        else:
            self._current_shadow_map = None
            self._current_shadow_dir = None

        if self.shadow_channel:
            prepended = self._prepend_shadow_channel(images, image_paths)
            if isinstance(batch, (list, tuple)) and len(batch) > 2:
                batch = (prepended, batch[1], *batch[2:])
            else:
                batch = prepended

        return super().predict_step(batch, batch_idx, dataloader_idx)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        backbone_lr = self.config.train.lr if self.config else 0.001
        sca_lr      = backbone_lr * 10   # SCA modules need to adapt faster than pretrained backbone

        if self.freeze_backbone:
            extra = []
            if self.shadow_cross_attention:
                extra += list(self.shadow_anticipation.parameters())
            params = [{"params": extra, "lr": sca_lr}]
            print(f"   Optimizer: Adam  sca_lr={sca_lr} (backbone FROZEN, ShadowAnticipation only)")
        else:
            params = [{"params": self.model.parameters(), "lr": backbone_lr}]
            if self.shadow_cross_attention:
                params += [
                    {"params": self.shadow_anticipation.parameters(), "lr": sca_lr},
                ]
                print(f"   Optimizer: SGD  backbone_lr={backbone_lr}  sca_lr={sca_lr} (10x backbone)")
            else:
                print(f"   Optimizer: SGD  backbone_lr={backbone_lr}")

        if self.freeze_backbone and self.shadow_cross_attention:
            # Stage 2: Adam for shadow-only training (short gradient chain, benefits from
            # per-parameter adaptive LR)
            optimizer = torch.optim.Adam(params, lr=sca_lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(params, lr=backbone_lr, momentum=0.9, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "map",
                             "interval": "epoch", "frequency": 1},
        }
