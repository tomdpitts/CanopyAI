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
        # Learnable residual scale initialized to 0.1 (was 0).
        # CRITICAL: We initialize to 0.1 to ensure the shadow pathway has
        # non-zero gradients immediately, forcing the model to use it.
        # If 0, it might get stuck in a "ignore shadows" local minimum.
        self.output_scale = nn.Parameter(torch.tensor([0.1]))

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
        val_csv=None,
        film_lr=1e-4,
        freeze_backbone=False,
        aux_loss_weight=1.0,
        shadow_channel=False,   # If True, widen first conv to 4 channels
        config=None,
        **kwargs,
    ):
        self.shadow_channel = shadow_channel
        # Initialize DeepForest (LightningModule)
        # Explicit call to avoid MRO/super() issues with keyword args
        deepforest_main.deepforest.__init__(self, config=config, **kwargs)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            print("   ❄️  Freezing backbone parameters (training FiLM only)")
            for param in self.model.parameters():
                param.requires_grad = False
                # Also set eval mode for batchnorm? Usually keeping them in train mode is better 
                # if existing stats are not representative, but for "freezing" typically we want eval mode 
                # or at least no grad. DeepForest handles train/eval mode. 
                # Lightning will call .train() which sets training=True.
                # If we want strict freezing, we might want to force eval mode on backbone, 
                # but let's stick to requires_grad=False for now.

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
                print(f"   Warning: could not load shadow lookup from train_csv: {e}")

        # Also load from val_csv if provided
        if val_csv is not None:
             try:
                import pandas as pd
                df_val = pd.read_csv(val_csv)
                if "shadow_x" in df_val.columns and "shadow_y" in df_val.columns:
                    count_before = len(self.shadow_lookup)
                    for _, row in df_val.drop_duplicates("image_path").iterrows():
                         # Only add if not present (or overwrite? overwrite is fine)
                         self.shadow_lookup[row["image_path"]] = np.array(
                            [row["shadow_x"], row["shadow_y"]], dtype=np.float32
                         )
                    print(f"   Loaded {len(self.shadow_lookup) - count_before} new shadow vectors from val CSV")
             except Exception as e:
                 print(f"   Warning: could not load shadow lookup from val_csv: {e}")

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

        # Auxiliary head: Predict shadow vector from FPN P5 features
        # This "mathematically induces" the features to contain shadow info.
        # P5 is high-level (semantic), good for global shadow direction.
        # AdaptiveAvgPool ensures fixed size regardless of input image size.
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Predict (sin, cos)
        )
        # Initialize aux head logic for loss weighting
        self.aux_loss_weight = aux_loss_weight

        # Shadow channel: 4th input channel (shadow probability map)
        # We store the flag but actually widen the conv AFTER pretrained weights are
        # loaded externally (see train_deepforest.py: widen_first_conv_for_shadow_channel).
        # This flag is serialised in the checkpoint so inference can detect it.
        self.shadow_channel = shadow_channel

    def _prepend_shadow_channel(self, images, image_paths):
        """
        For each image in the batch, compute a shadow probability map and
        concatenate it as a 4th channel.  Images are expected as a list of
        [3, H, W] float tensors in [0,1] range (DeepForest default).
        Returns a list of [4, H, W] float tensors.
        """
        result = []
        for img_t, path in zip(images, image_paths):
            sv = self.shadow_lookup.get(path)
            if sv is None:
                # Fallback: zero channel (no shadow info)
                shadow_t = torch.zeros(1, img_t.shape[1], img_t.shape[2],
                                       dtype=img_t.dtype, device=img_t.device)
            else:
                # img_t is [3, H, W] float [0,1] — convert to uint8 RGB for generate_shadow_map
                img_np  = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                angle   = float(np.degrees(np.arctan2(sv[0], sv[1])))  # sx,sy → degrees
                shadow_np = generate_shadow_map(img_np, angle)          # H×W float32
                shadow_t  = torch.from_numpy(shadow_np).unsqueeze(0)    # [1,H,W]
                shadow_t  = shadow_t.to(img_t.device, dtype=img_t.dtype)
            result.append(torch.cat([img_t, shadow_t], dim=0))          # [4,H,W]
        return result

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

        # Prepend shadow map as 4th channel if enabled
        if self.shadow_channel:
            batch = (self._prepend_shadow_channel(images, image_paths),
                     targets,
                     *batch[2:])

        # 1. Main DeepForest loss (box regression + classification)
        loss_result = super().training_step(batch, batch_idx)
        
        # 2. Auxiliary Shadow Loss
        aux_loss = 0.0
        
        # Retrieve captured features from the hook
        if hasattr(self, "_last_p5_features") and self._last_p5_features is not None:
            # Predict shadow vector from features
            pred_shadow = self.aux_head(self._last_p5_features)  # (B, 2)
            
            # Ground truth
            gt_shadow = self.current_shadow_vector  # (B, 2)
            
            # Computed MSE loss
            aux_loss = F.mse_loss(pred_shadow, gt_shadow)
            
            # Log it
            self.log("aux_loss", aux_loss, on_step=True, on_epoch=True, prog_bar=True)
            
            # Clear for next step
            self._last_p5_features = None
            
        # Combine losses
        # Check if loss_result is a dict (DeepForest < 2.0?) or Tensor (DeepForest >= 2.0 via Lightning?)
        if isinstance(loss_result, dict):
            loss_result["loss"] += self.aux_loss_weight * aux_loss
            return loss_result
        else:
            # It's a scalar tensor
            total_loss = loss_result + self.aux_loss_weight * aux_loss
            return total_loss

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

        if self.freeze_backbone:
            params = [
                {"params": self.film_blocks.parameters(), "lr": film_lr},
                 # Aux head is part of the new logic, should be trained too
                {"params": self.aux_head.parameters(), "lr": film_lr},
            ]
            print(f"   Optimizer: SGD(momentum=0.9)  film_lr={film_lr} (Backbone FROZEN)")
        else:
            params = [
                {"params": self.model.parameters(), "lr": backbone_lr},
                {"params": self.film_blocks.parameters(), "lr": film_lr},
                {"params": self.aux_head.parameters(), "lr": film_lr},
            ]
            print(f"   Optimizer: SGD(momentum=0.9)  backbone_lr={backbone_lr}  film_lr={film_lr}")

        optimizer = torch.optim.SGD(
            params, lr=backbone_lr, momentum=0.9, weight_decay=1e-4
        )

        # Add scheduler: Reduce LR when mAP plateaus
        # Patience=5 matches early stopping (but early stopping is 15 in run args, so maybe 5 is good for scheduler)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "map",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def set_shadow_vector(self, shadow_vector):
        """Set the current shadow vector (called by dataloader during augmentation)."""
        self.current_shadow_vector = shadow_vector

    def _inject_film_hooks(self):
        """
        Inject forward hooks into DeepForest's backbone to apply FiLM conditioning.
        We hook the backbone output dict directly.
        """
        if self._film_injected:
            return

        def backbone_hook(module, input, output):
            """
            Input: Tensor/List of images
            Output: OrderedDict({'0': P3, '1': P4, '2': P5, 'p6': P6, 'p7': P7})
            """
            if self.current_shadow_vector is None:
                return output
                
            # Mapping from backbone keys to FiLM block names
            # RetinaNet BackboneWithFPN: '0'->P3, '1'->P4, '2'->P5, 'p6'->P6, 'p7'->P7 or similar
            # DeepForest/Torchvision convention:
            # ResNet FPN usually: 0,1,2,pool -> P3,P4,P5,P6,P7
            # We verified keys are: '0', '1', '2', 'p6', 'p7'
            key_map = {
                '0': 'P3',
                '1': 'P4',
                '2': 'P5',
                'p6': 'P6',
                'p7': 'P7'
            }
            
            # Modify features in-place (or return new dict)
            # We must return the modified dict for the head to use.
            
            # Note: `output` is usually an OrderedDict.
            # We shouldn't modify it in-place if it affects other things, but here it's fine.
            
            for key, film_name in key_map.items():
                if key in output:
                    feat = output[key]
                    
                    # Apply FiLM
                    if film_name in self.film_blocks:
                        modulated = self.film_blocks[film_name](feat, self.current_shadow_vector)
                        output[key] = modulated
                        
                        # Capture P5 for aux head
                        if film_name == 'P5':
                            self._last_p5_features = modulated
            
            return output

        # Register hook on backbone
        try:
            # Handle wrapping: self.model -> (optional .model) -> .backbone
            backbone = None
            if hasattr(self.model, "backbone"):
                backbone = self.model.backbone
            elif hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
                backbone = self.model.model.backbone
            
            if backbone:
                backbone.register_forward_hook(backbone_hook)
                print("   ✅ FiLM hooks injected into Backbone (output dict)")
            else:
                print("   ⚠️  Could not find Backbone to inject hooks.")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not inject FiLM hooks: {e}")
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

    def validation_step(self, batch, batch_idx):
        """
        Validation step with shadow conditioning.
        Ensures shadow vectors are correctly set for the validation batch.
        """
        # Set shadow vector for this batch
        self._set_batch_shadow_vector(batch)
        
        # Run base validation (forward pass)
        return super().validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step with shadow conditioning.
        """
        self._set_batch_shadow_vector(batch)
        return super().predict_step(batch, batch_idx, dataloader_idx)

    def _set_batch_shadow_vector(self, batch):
        """Helper to look up and set shadow vector for a batch."""
        # Check if batch has image paths (index 2)
        # DeepForest batches are usually: (images, targets, image_paths)
        if len(batch) > 2:
            image_paths = batch[2]
            
            # Look up vectors
            # Use base vector as fallback for missing paths (e.g. if val set incomplete)
            vectors = []
            device = batch[0][0].device if isinstance(batch[0], list) else batch[0].device
            
            for p in image_paths:
                if p in self.shadow_lookup:
                    vectors.append(self.shadow_lookup[p])
                else:
                    # Fallback to base shadow vector
                    vectors.append(self.base_shadow_vector.numpy())
            
            shadow_vectors_np = np.array(vectors, dtype=np.float32)
            self.current_shadow_vector = torch.from_numpy(shadow_vectors_np).to(device)
        else:
             # No paths? Use base shadow vector for entire batch
             # This happens if dataloader doesn't return paths
             if isinstance(batch[0], list):
                 batch_size = len(batch[0])
                 device = batch[0][0].device
             else:
                 batch_size = batch[0].shape[0]
                 device = batch[0].device
                 
             self.current_shadow_vector = (
                 self.base_shadow_vector.to(device).unsqueeze(0).repeat(batch_size, 1)
             )


