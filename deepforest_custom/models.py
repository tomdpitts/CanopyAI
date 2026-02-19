import torch
import torch.nn as nn
import numpy as np
from deepforest import main as deepforest_main


class SolarAttentionBlock(nn.Module):
    """
    Solar-Gated Spatial Attention for FiLM conditioning.

    Injects global shadow vector into FPN feature maps.
    Learns to upweight features consistent with shadow direction.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Project shadow vector to channel-wise modulation (FiLM)
        self.sun_mlp = nn.Sequential(
            nn.Linear(2, channels // 4), nn.ReLU(), nn.Linear(channels // 4, channels)
        )

        # Spatial gating convolution
        self.gate_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Learnable residual scale initialized to 0.
        # CRITICAL: Even with zero-init on sun_mlp[2] output layer, the first
        # optimizer step updates sun_mlp[2] weights by ±lr. Combined with random
        # upstream hidden states (h~±5.6 norm), gamma immediately grows to ~0.21
        # — a 21% feature perturbation across all 5 FPN levels simultaneously.
        # The residual scale ensures: out = x + scale*(film_out - x).
        # When scale=0, the output is EXACTLY x regardless of gamma/gate.
        # Scale grows gradually via its own gradient, allowing FiLM to "fade in".
        self.output_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, sun_vector, return_attn=False):
        """
        Args:
            x: Feature map (B, C, H, W)
            sun_vector: Shadow direction (B, 2) - unit vector
            return_attn: If True, return attention map for visualization

        Returns:
            modulated_features: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1. FiLM: Channel-wise modulation from shadow vector
        gamma = self.sun_mlp(sun_vector)  # (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # Apply FiLM modulation (affine transformation)
        x_modulated = x * (1 + gamma)  # Scale features based on shadow

        # 2. Spatial Gating: Learn where to apply shadow-conditioned features
        gate = self.sigmoid(self.gate_conv(x_modulated))  # (B, 1, H, W)

        # 3. Blend: Weighted combination of original and modulated features
        film_out = x * (1 - gate) + x_modulated * gate

        # 4. Residual gate: blend FiLM output with identity using learnable scale.
        # scale=0 → output=x exactly; scale grows as FiLM learns to be useful.
        out = x + self.output_scale * (film_out - x)

        if return_attn:
            return out, gate
        return out


class ShadowConditionedDeepForest(deepforest_main.deepforest):
    """
    DeepForest with FiLM conditioning on shadow vectors.

    Uses a base shadow direction (215° for WON003) that gets rotated during
    augmentation. Rotation provides natural shadow diversity: rotating the image
    by θ degrees also rotates the shadow by θ degrees.
    """

    def __init__(
        self,
        shadow_angle_deg=215.0,
        train_csv=None,
        film_lr=1e-4,
        config=None,
        **kwargs,
    ):
        # Initialize DeepForest (LightningModule)
        # Explicit call to avoid MRO/super() issues with keyword args
        deepforest_main.deepforest.__init__(self, config=config, **kwargs)

        # Add FiLM blocks for each FPN level (P3-P7)
        # DeepForest uses RetinaNet with 256 channels at each FPN level
        self.film_blocks = nn.ModuleDict(
            {
                f"P{level}": SolarAttentionBlock(channels=256)
                for level in [3, 4, 5, 6, 7]
            }
        )

        # Store FiLM-specific learning rate for differential lr
        self.film_lr = film_lr

        # Store base shadow angle (fallback when no CSV lookup available)
        self.shadow_angle_deg = shadow_angle_deg
        shadow_angle_rad = np.radians(shadow_angle_deg)
        self.base_shadow_vector = torch.tensor(
            [np.sin(shadow_angle_rad), np.cos(shadow_angle_rad)], dtype=torch.float32
        )

        # Build per-image shadow lookup from training CSV if provided.
        # Maps image_path -> np.array([shadow_x, shadow_y]) using azimuth convention.
        self.shadow_lookup = {}
        if train_csv is not None:
            try:
                import pandas as pd

                df = pd.read_csv(train_csv)
                if "shadow_x" in df.columns and "shadow_y" in df.columns:
                    for _, row in df.drop_duplicates("image_path").iterrows():
                        self.shadow_lookup[row["image_path"]] = np.array(
                            [row["shadow_x"], row["shadow_y"]], dtype=np.float32
                        )
                    print(
                        f"   Loaded shadow vectors for {len(self.shadow_lookup)} images from CSV"
                    )
                else:
                    print(
                        "   No shadow_x/shadow_y in CSV — using base vector with rotation fallback"
                    )
            except Exception as e:
                print(f"   Warning: could not load shadow lookup: {e}")

        if self.shadow_lookup:
            print(
                f"   Per-image FiLM vectors: ENABLED ({len(self.shadow_lookup)} images)"
            )
        else:
            print(
                f"   Per-image FiLM vectors: DISABLED (base {shadow_angle_deg} deg + rotation)"
            )

        # Track whether FiLM hooks are injected
        self._film_injected = False
        self.current_shadow_vector = None

    def training_step(self, batch, batch_idx):
        """
        Training step with per-image FiLM shadow conditioning.

        Reads shadow_x/shadow_y from the CSV lookup built at init (keyed by image path).
        The CSV is expected to have pre-augmented images with matching shadow vectors
        (as generated by augment_with_shadows.py). No random rotation is applied here.

        Raises RuntimeError if shadow lookup is missing or any image path is not found.
        """
        images = batch[0]
        targets = batch[1]
        # batch[2] = list of image paths (from DeepForest dataloader collate_fn)
        image_paths = batch[2] if len(batch) > 2 else None

        if not self.shadow_lookup:
            raise RuntimeError(
                "ShadowConditionedDeepForest: shadow_lookup is empty. "
                "Pass train_csv= at init so shadow_x/shadow_y can be read from the CSV. "
                "The CSV must have 'shadow_x' and 'shadow_y' columns."
            )

        if image_paths is None:
            raise RuntimeError(
                "ShadowConditionedDeepForest: batch[2] (image paths) is missing. "
                "Expected DeepForest dataloader to provide image paths as batch[2]."
            )

        # Look up per-image shadow vectors — fail loudly on missing paths
        missing = [p for p in image_paths if p not in self.shadow_lookup]
        if missing:
            raise RuntimeError(
                f"ShadowConditionedDeepForest: {len(missing)} image path(s) not found in shadow_lookup.\n"
                f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
                "Check that image paths in the batch match those in the training CSV exactly."
            )

        shadow_vectors_np = np.array(
            [self.shadow_lookup[p] for p in image_paths], dtype=np.float32
        )
        shadow_vectors = torch.from_numpy(shadow_vectors_np).to(images[0].device)
        self.current_shadow_vector = shadow_vectors

        # Images and boxes are already correctly oriented — augment_with_shadows.py
        # pre-generates the 4 rotations with matching shadow vectors in the CSV.
        return super().training_step(batch, batch_idx)

    def configure_optimizers(self):
        """
        Configure optimizers with differential learning rates.

        Uses oscar50's proven SGD+momentum=0.9 for the backbone. This is now
        safe because the output_scale residual gate in SolarAttentionBlock
        ensures FiLM is exactly identity at init — backbone trains as if FiLM
        doesn't exist. FiLM blocks use a lower lr so output_scale grows slowly.
        """
        backbone_lr = self.config.train.lr if self.config else 0.001
        film_lr = self.film_lr

        params = [
            {"params": self.model.parameters(), "lr": backbone_lr},
            {"params": self.film_blocks.parameters(), "lr": film_lr},
        ]

        print(f"   Optimizer: SGD(momentum=0.9)  backbone_lr={backbone_lr}  film_lr={film_lr}")

        optimizer = torch.optim.SGD(
            params, lr=backbone_lr, momentum=0.9, weight_decay=1e-4
        )
        return optimizer

    def set_shadow_vector(self, shadow_vector):
        """Set the current shadow vector (called by dataloader during augmentation)."""
        self.current_shadow_vector = shadow_vector

    def _inject_film_hooks(self):
        """
        Inject forward hooks into DeepForest's FPN to apply FiLM conditioning.
        """
        if self._film_injected:
            return

        def make_hook(level_name):
            def hook(module, input, output):
                if self.current_shadow_vector is not None:
                    # Apply FiLM conditioning
                    return self.film_blocks[level_name](
                        output, self.current_shadow_vector
                    )
                return output

            return hook

        # Register hooks on FPN outputs
        try:
            # Try to access FPN. Path might vary depending on DeepForest/Torchvision version.
            # Usually: self.model (RetinaNet) -> backbone -> fpn
            # Or: self.model (RetinaNet) -> model (if wrapped) -> backbone -> fpn
            fpn = None
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "fpn"):
                fpn = self.model.backbone.fpn
            elif hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
                # Handle case where RetinaNet is wrapped or double-nested
                fpn = self.model.model.backbone.fpn

            if fpn:
                for level in [3, 4, 5, 6, 7]:
                    layer = getattr(fpn, f"fpn_layer{level}", None)
                    if layer is not None:
                        layer.register_forward_hook(make_hook(f"P{level}"))
                print("   ✅ FiLM hooks injected into FPN")
            else:
                print("   ⚠️  Could not find FPN to inject hooks.")

        except AttributeError:
            print(
                "⚠️  Warning: Could not inject FiLM hooks. FPN structure may have changed."
            )
            print("   FiLM conditioning will be skipped.")

        self._film_injected = True

    def on_train_start(self):
        """DeepForest hook: called when training starts."""
        if not self._film_injected:
            self._inject_film_hooks()
        # Call super just in case
        if hasattr(super(), "on_train_start"):
            super().on_train_start()

    def on_validation_start(self):
        """DeepForest hook: called when validation starts."""
        if not self._film_injected:
            self._inject_film_hooks()
        if hasattr(super(), "on_validation_start"):
            super().on_validation_start()

    def on_predict_start(self):
        """DeepForest hook: called when prediction starts."""
        if not self._film_injected:
            self._inject_film_hooks()
        if hasattr(super(), "on_predict_start"):
            super().on_predict_start()

    def forward(self, x, targets=None, shadow_vectors=None):
        """
        Forward pass with shadow conditioning.

        Args:
            x: Input images - List[(C, H, W)] or (B, C, H, W)
            targets: Optional targets for training
            shadow_vectors: Optional per-sample shadow vectors (B, 2)
                          If None, uses base shadow vector

        Returns:
            DeepForest predictions (losses dict during training, detections during eval)
        """
        # Use provided shadow vectors or default to base
        if shadow_vectors is not None:
            self.current_shadow_vector = shadow_vectors
        else:
            # Create shadow vector batch from base
            if isinstance(x, list):
                batch_size = len(x)
                device = x[0].device
            else:
                batch_size = x.shape[0]
                device = x.device

            self.current_shadow_vector = (
                self.base_shadow_vector.to(device).unsqueeze(0).repeat(batch_size, 1)
            )

        # Inject hooks on first forward pass if not already done
        if not self._film_injected:
            self._inject_film_hooks()

        # Forward through DeepForest (FiLM hooks will activate)
        # self.model is the backbone+head in DeepForest class
        if targets is not None:
            # Training mode
            return self.model(x, targets)
        else:
            # Inference mode
            return self.model(x)
