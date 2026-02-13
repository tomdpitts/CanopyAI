
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
        out = x * (1 - gate) + x_modulated * gate

        if return_attn:
            return out, gate
        return out


class ShadowConditionedDeepForest(nn.Module):
    """
    DeepForest with FiLM conditioning on shadow vectors.

    Uses a base shadow direction (215° for WON003) that gets rotated during
    augmentation. Rotation provides natural shadow diversity: rotating the image
    by θ degrees also rotates the shadow by θ degrees.
    """

    def __init__(self, shadow_angle_deg=215.0, config=None, **kwargs):
        super().__init__()

        # Create base DeepForest model
        self.deepforest = deepforest_main.deepforest(config=config, **kwargs)

        # Add FiLM blocks for each FPN level (P3-P7)
        # DeepForest uses RetinaNet with 256 channels at each FPN level
        self.film_blocks = nn.ModuleDict(
            {
                f"P{level}": SolarAttentionBlock(channels=256)
                for level in [3, 4, 5, 6, 7]
            }
        )

        # Store base shadow angle (will be rotated during augmentation)
        self.shadow_angle_deg = shadow_angle_deg

        # Convert to (sin, cos) representation
        shadow_angle_rad = np.radians(shadow_angle_deg)
        self.base_shadow_vector = torch.tensor(
            [np.sin(shadow_angle_rad), np.cos(shadow_angle_rad)], dtype=torch.float32
        )

        print(f"   ℹ️  Base shadow direction: {shadow_angle_deg}°")
        print(f"   ℹ️  Rotation will provide shadow diversity")

        # Track whether FiLM hooks are injected
        self._film_injected = False
        self.current_shadow_vector = None

        # Patch DeepForest's training_step for rotation augmentation
        self._patch_training_step()

    def _patch_training_step(self):
        """Patch DeepForest's training_step to apply rotation augmentation."""
        original_training_step = self.deepforest.training_step
        base_shadow_vector = self.base_shadow_vector
        film_model = self  # Reference to this ShadowConditionedDeepForest instance

        def training_step_with_rotation(batch, batch_idx):
            # Import here to avoid circular dependencies
            try:
                from .rotation_augmentation import (
                    rotate_image_and_boxes,
                    rotate_shadow_vector,
                )
            except ImportError:
                # If running as script
                from rotation_augmentation import (
                    rotate_image_and_boxes,
                    rotate_shadow_vector,
                )

            # DeepForest batch can be (images, targets) or (images, targets, image_ids)
            images = batch[0]
            targets = batch[1]

            # Apply rotation to each sample
            rotated_images = []
            rotated_targets = []
            rotated_shadows = []

            for image, target in zip(images, targets):
                # Random rotation angle (±180°)
                angle = np.random.uniform(-180, 180)

                # Convert tensor to numpy for rotation
                img_np = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
                boxes_np = target["boxes"].cpu().numpy()

                # Rotate image and boxes
                rotated_img, rotated_boxes = rotate_image_and_boxes(
                    img_np, boxes_np, angle
                )

                # Rotate shadow vector
                base_shadow_np = base_shadow_vector.cpu().numpy()
                rotated_shadow = rotate_shadow_vector(base_shadow_np, angle)

                # Convert back to tensors
                rotated_img_tensor = (
                    torch.from_numpy(rotated_img).permute(2, 0, 1).to(image.device)
                )  # (H, W, C) -> (C, H, W)
                rotated_boxes_tensor = (
                    torch.from_numpy(rotated_boxes).float().to(target["boxes"].device)
                )

                # Update target
                target["boxes"] = rotated_boxes_tensor

                rotated_images.append(rotated_img_tensor)
                rotated_targets.append(target)
                rotated_shadows.append(rotated_shadow)

            # Store shadow vectors for forward pass
            shadow_vectors = (
                torch.from_numpy(np.array(rotated_shadows)).float().to(images[0].device)
            )
            film_model.current_shadow_vector = shadow_vectors  # Set for hooks

            # Call original training step with rotated data
            # The original training step expects (images, targets, image_ids)
            # We reconstruct the batch with rotated images/targets and original image_ids
            if len(batch) > 2:
                new_batch = (rotated_images, rotated_targets) + batch[2:]
            else:
                new_batch = (rotated_images, rotated_targets)

            return original_training_step(new_batch, batch_idx)

        self.deepforest.training_step = training_step_with_rotation

    def set_shadow_vector(self, shadow_vector):
        """Set the current shadow vector (called by dataloader during augmentation)."""
        self.current_shadow_vector = shadow_vector

    def _inject_film_hooks(self):
        """
        Inject forward hooks into DeepForest's FPN to apply FiLM conditioning.

        This modifies the model's forward pass to condition features on shadow vectors.
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
        # Note: This assumes DeepForest's internal structure
        try:
            fpn = self.deepforest.model.model.backbone.fpn
            for level in [3, 4, 5, 6, 7]:
                layer = getattr(fpn, f"fpn_layer{level}", None)
                if layer is not None:
                    layer.register_forward_hook(make_hook(f"P{level}"))
        except AttributeError:
            print(
                "⚠️  Warning: Could not inject FiLM hooks. FPN structure may have changed."
            )
            print("   FiLM conditioning will be skipped.")

        self._film_injected = True

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

        # Inject hooks on first forward pass
        if not self._film_injected:
            self._inject_film_hooks()

        # Forward through DeepForest (FiLM hooks will activate)
        if targets is not None:
            # Training mode
            return self.deepforest.model(x, targets)
        else:
            # Inference mode
            return self.deepforest.model(x)

    def __getattr__(self, name):
        """Delegates attributes to the wrapped DeepForest model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.deepforest, name)
