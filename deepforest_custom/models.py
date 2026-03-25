import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from deepforest import main as deepforest_main

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from .utils import generate_shadow_map   # package import
except ImportError:
    from utils import generate_shadow_map     # script import


# ---------------------------------------------------------------------------
# Shadow Encoder
# ---------------------------------------------------------------------------

class ShadowEncoder(nn.Module):
    """
    Lightweight CNN that maps a 1-channel shadow probability map to
    feature tokens at the same spatial scale as ResNet layer4
    (i.e. H/32 x W/32 for a standard crop).

    Output is zero-initialized so the module starts fully silent.
    """
    def __init__(self, out_channels=256):
        super().__init__()
        # 5x stride-2 convolutions bring H -> H/32
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
        )
        # Zero-init output conv so encoder is silent at epoch 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # x: (B, 1, H, W) shadow map in [0,1]
        return self.net(x)   # -> (B, out_channels, H/32, W/32)


# ---------------------------------------------------------------------------
# Shadow Cross-Attention
# ---------------------------------------------------------------------------

class ShadowCrossAttention(nn.Module):
    """
    Cross-attention module inserted after ResNet layer4.

    RGB layer4 features are used as Queries.
    Shadow encoder tokens are Keys and Values.

    The learned gate scalar is initialized to 0 so the module is provably
    identity at epoch 0 — oscar50 weights are completely unaffected initially.
    """
    def __init__(self, rgb_channels=2048, shadow_channels=256, num_heads=8):
        super().__init__()
        self.q_proj   = nn.Linear(rgb_channels, shadow_channels)
        self.attn     = nn.MultiheadAttention(shadow_channels, num_heads, batch_first=True)
        self.out_proj  = nn.Linear(shadow_channels, rgb_channels)
        self.norm      = nn.LayerNorm(rgb_channels)
        # Gate init: tiny nonzero value so gradients flow from step 1.
        # out_proj stays zero-init so at epoch 0 the output is ~0 * attn + rgb = rgb (identity).
        # The small gate gives a real gradient signal to start learning from.
        self.gate = nn.Parameter(torch.ones(1) * 0.01)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, rgb_feat, shadow_feat):
        """
        rgb_feat:    (B, 2048, H, W) from ResNet layer4
        shadow_feat: (B, 256,  H, W) from ShadowEncoder
        Returns:     (B, 2048, H, W) — same shape, zero-perturbation at init
        """
        B, C, H, W = rgb_feat.shape

        # Flatten spatial dims to token sequences
        q  = rgb_feat.flatten(2).permute(0, 2, 1)     # (B, H*W, 2048)
        kv = shadow_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)

        # Project Q down to shadow_channels
        q = self.q_proj(q)                             # (B, H*W, 256)

        # Cross-attention: RGB queries attend to Shadow keys/values
        attn_out, _ = self.attn(q, kv, kv)            # (B, H*W, 256)

        # Project back up to RGB channel width
        attn_out = self.out_proj(attn_out)             # (B, H*W, 2048)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        # Residual with zero-init gate — starts as identity, grows via gradient
        out = rgb_feat + self.gate * attn_out
        # LayerNorm applied over channel dim
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


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

    2. shadow_cross_attention=True — a ShadowCrossAttention module is grafted
       after ResNet layer4 via a forward hook. The shadow map becomes K/V
       tokens that RGB features (Q) can attend to.

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
        config=None,
        **kwargs,
    ):
        self.shadow_channel = shadow_channel
        self.shadow_cross_attention = shadow_cross_attention

        # Initialize DeepForest (LightningModule)
        deepforest_main.deepforest.__init__(self, config=config, **kwargs)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            print("   ❄️  Freezing backbone parameters")
            for param in self.model.parameters():
                param.requires_grad = False

        # Store base shadow angle for fallback
        self.shadow_angle_deg = shadow_angle_deg
        shadow_angle_rad = np.radians(shadow_angle_deg)
        self.base_shadow_vector = torch.tensor(
            [np.sin(shadow_angle_rad), np.cos(shadow_angle_rad)], dtype=torch.float32
        )

        # Per-image shadow lookup: image_path -> np.array([shadow_x, shadow_y])
        self.shadow_lookup = {}
        # Per-image global normalization stats: image_path -> (dg_scale, dark_scale, otsu_ctr)
        # These are computed from the full source orthomosaic before tiling so that all tiles
        # share the same normalization scale (identical to the inference-time computation in foxtrot.py).
        self.norm_stats_lookup = {}
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
            except Exception as e:
                print(f"   Warning: could not load {label} shadow lookup: {e}")

        # Shadow Cross-Attention modules
        if self.shadow_cross_attention:
            self.shadow_encoder = ShadowEncoder(out_channels=256)
            self.shadow_cross_attn = ShadowCrossAttention(
                rgb_channels=2048, shadow_channels=256, num_heads=8
            )
            print("   ✅ Shadow Cross-Attention: ENABLED (gate=0, identity at init)")
        else:
            print("   Shadow Cross-Attention: DISABLED")

        # Flag to track hook registration
        self._sca_hook_injected = False
        self._current_shadow_enc = None  # set per forward pass

        # Shadow channel flag is serialised in checkpoint for inference detection
        self.shadow_channel = shadow_channel

    # ------------------------------------------------------------------
    # Shadow map computation helpers
    # ------------------------------------------------------------------

    def _compute_shadow_map(self, img_t, shadow_vector, norm_stats=None):
        """
        Compute a single shadow probability map for one [3,H,W] float image tensor.

        - Uses the shadow_x/shadow_y vector to derive the angle.
        - norm_stats: (dg_scale, dark_scale, otsu_ctr) pre-computed from the full source
          orthomosaic so that normalization is globally consistent across tiles.
          When None, falls back to per-tile stats (not recommended for training).
        Returns a [1, H, W] float32 tensor in [0, 1].
        """
        img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        angle  = float(np.degrees(np.arctan2(shadow_vector[0], shadow_vector[1])))

        if norm_stats is not None:
            dg_scale, dark_scale, otsu_ctr = norm_stats
            shadow_np = generate_shadow_map(img_np, angle,
                                            dg_scale=dg_scale,
                                            dark_scale=dark_scale,
                                            otsu_ctr=otsu_ctr)
        else:
            shadow_np = generate_shadow_map(img_np, angle)
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
                shadow_t = self._compute_shadow_map(img_t, sv, norm_stats=norm_stats)
                shadow_t = shadow_t.to(img_t.device, dtype=img_t.dtype)
            result.append(torch.cat([img_t, shadow_t], dim=0))  # [4, H, W]
        return result

    def _compute_shadow_enc_batch(self, images, image_paths):
        """
        Compute ShadowEncoder features for a batch for use in cross-attention.
        Returns tensor (B, 256, H/32, W/32) on the same device as images.
        """
        device = images[0].device
        # Target spatial size: use first image's H/W so all maps are uniform.
        # Batches can contain variable-size crops; stack requires equal shapes.
        target_h, target_w = images[0].shape[1], images[0].shape[2]
        shadow_maps = []
        for img_t, path in zip(images, image_paths):
            sv = self.shadow_lookup.get(path)
            if sv is None:
                sv = np.array([np.sin(np.radians(self.shadow_angle_deg)),
                                np.cos(np.radians(self.shadow_angle_deg))], dtype=np.float32)
            norm_stats = self.norm_stats_lookup.get(path) or self.norm_stats_lookup.get(None)
            shadow_t = self._compute_shadow_map(img_t, sv, norm_stats=norm_stats)  # [1, H_i, W_i]
            if shadow_t.shape[1] != target_h or shadow_t.shape[2] != target_w:
                shadow_t = F.interpolate(
                    shadow_t.unsqueeze(0), size=(target_h, target_w),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
            shadow_maps.append(shadow_t)
        shadow_batch = torch.stack(shadow_maps).to(device)   # (B, 1, H, W)
        return self.shadow_encoder(shadow_batch)              # (B, 256, H/32, W/32)

    # ------------------------------------------------------------------
    # Hook injection for Shadow Cross-Attention
    # ------------------------------------------------------------------

    def _inject_sca_hook(self):
        """Register a forward hook on ResNet layer4 to apply cross-attention."""
        if self._sca_hook_injected:
            return

        def layer4_hook(module, input, output):
            if self._current_shadow_enc is not None:
                return self.shadow_cross_attn(output, self._current_shadow_enc)
            return output

        try:
            # Navigate to layer4 through the DeepForest model wrappers
            body = None
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "body"):
                body = self.model.backbone.body
            elif (hasattr(self.model, "model") and
                  hasattr(self.model.model, "backbone") and
                  hasattr(self.model.model.backbone, "body")):
                body = self.model.model.backbone.body

            if body and hasattr(body, "layer4"):
                body.layer4.register_forward_hook(layer4_hook)
                print("   ✅ Shadow Cross-Attention hook injected on layer4")
            else:
                print("   ⚠️  Could not find layer4 to inject SCA hook")
        except Exception as e:
            print(f"   ⚠️  SCA hook injection failed: {e}")

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

    # ------------------------------------------------------------------
    # Training / Validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        images      = batch[0]
        targets     = batch[1]
        image_paths = batch[2] if len(batch) > 2 else [None] * len(images)

        # Compute shadow encoder features (set on self so layer4 hook can read them)
        if self.shadow_cross_attention:
            self._current_shadow_enc = self._compute_shadow_enc_batch(images, image_paths)
        else:
            self._current_shadow_enc = None

        # Prepend shadow as 4th input channel if enabled
        if self.shadow_channel:
            images = self._prepend_shadow_channel(images, image_paths)
            batch  = (images, targets, *batch[2:])

        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        images      = batch[0]
        targets     = batch[1]
        image_paths = batch[2] if len(batch) > 2 else [None] * len(images)

        if self.shadow_cross_attention:
            self._current_shadow_enc = self._compute_shadow_enc_batch(images, image_paths)
        else:
            self._current_shadow_enc = None

        if self.shadow_channel:
            images = self._prepend_shadow_channel(images, image_paths)
            batch  = (images, targets, *batch[2:])

        return super().validation_step(batch, batch_idx)

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
            self._current_shadow_enc = self._compute_shadow_enc_batch(images, image_paths)
        else:
            self._current_shadow_enc = None

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
                extra += list(self.shadow_encoder.parameters())
                extra += list(self.shadow_cross_attn.parameters())
            params = [{"params": extra, "lr": sca_lr}]
            print(f"   Optimizer: SGD  sca_lr={sca_lr} (backbone FROZEN, SCA only)")
        else:
            params = [{"params": self.model.parameters(), "lr": backbone_lr}]
            if self.shadow_cross_attention:
                params += [
                    {"params": self.shadow_encoder.parameters(),   "lr": sca_lr},
                    {"params": self.shadow_cross_attn.parameters(), "lr": sca_lr},
                ]
                print(f"   Optimizer: SGD  backbone_lr={backbone_lr}  sca_lr={sca_lr} (10x backbone)")
            else:
                print(f"   Optimizer: SGD  backbone_lr={backbone_lr}")

        optimizer = torch.optim.SGD(params, lr=backbone_lr, momentum=0.9, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "map",
                             "interval": "epoch", "frequency": 1},
        }
