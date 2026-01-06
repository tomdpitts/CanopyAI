import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from deepforest.main import deepforest
from solar_adapter import SolarAttentionBlock


class SolarRetinaNet(RetinaNet):
    """
    Modified RetinaNet that accepts a global Sun Vector.
    It injects 'Solar Attention' into the FPN features before passing them
    to the regression/classification heads.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        if backbone is None:
            # Standard ResNet50-FPN backbone
            backbone = resnet_fpn_backbone("resnet50", pretrained=True)

        super().__init__(backbone, num_classes, **kwargs)

        # Inject Solar Gates into the FPN output channels (usually 256)
        out_channels = backbone.out_channels

        # Create a dictionary of attention blocks, one for each FPN level
        # FPN levels: '0', '1', '2', '3', 'pool' (standard torchvision names)
        self.solar_gates = nn.ModuleDict(
            {
                "0": SolarAttentionBlock(out_channels),
                "1": SolarAttentionBlock(out_channels),
                "2": SolarAttentionBlock(out_channels),
                "3": SolarAttentionBlock(out_channels),
                "pool": SolarAttentionBlock(out_channels),
            }
        )

    def _compute_guided_attention_loss(self, attention_maps, shadow_masks):
        """
        Compute MSE loss between attention maps and shadow masks.

        Args:
            attention_maps: Dict[str, Tensor] - Attention maps per FPN level (B, 1, H, W)
            shadow_masks: List[Tensor] or Tensor - Shadow masks at original resolution

        Returns:
            Tensor: Scalar attention guidance loss
        """
        total_loss = 0.0
        num_levels = 0

        # Stack shadow masks if passed as list
        if isinstance(shadow_masks, list):
            # Assume all masks are same size, stack to (B, 1, H_orig, W_orig)
            shadow_masks = torch.stack(
                [m.unsqueeze(0) if m.dim() == 2 else m for m in shadow_masks]
            )
        if shadow_masks.dim() == 3:
            shadow_masks = shadow_masks.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        for key, attn_map in attention_maps.items():
            # Downsample shadow mask to match attention map spatial size
            _, _, H, W = attn_map.shape
            target_mask = F.interpolate(
                shadow_masks.float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            # Normalize target to [0, 1] if not already
            target_mask = target_mask.clamp(0, 1)

            # MSE loss: attention should correlate with shadow regions
            level_loss = F.mse_loss(attn_map, target_mask)
            total_loss += level_loss
            num_levels += 1

        return total_loss / num_levels if num_levels > 0 else torch.tensor(0.0)

    def forward(self, images, sun_vectors, targets=None, shadow_masks=None):
        """
        Modified forward pass.

        Args:
            images: List[Tensor]
            sun_vectors: Tensor (B, 2) - The global context
            targets: List[Dict] - Standard detection targets
            shadow_masks: List[Tensor] - Optional for "Level 2" Guided Attention
        """
        # 1. Get Transform (Standard internal pre-processing)
        if self.training:
            original_image_sizes = [img.shape[-2:] for img in images]
            images, targets = self.transform(images, targets)
        else:
            original_image_sizes = [img.shape[-2:] for img in images]
            images, _ = self.transform(images)

        # 2. Backbone + FPN
        # features is a Dict[str, Tensor] typically keys '0', '1', '2', '3', 'pool'
        features = self.backbone(images.tensors)

        # 3. Apply Solar Gating (The Innovation)
        gated_features = {}
        attention_maps = {}  # For Level 2 Supervision

        # Extract attention maps during training for guided attention loss
        extract_attn = self.training and shadow_masks is not None

        for key, feature_map in features.items():
            if key in self.solar_gates:
                if extract_attn:
                    # Real implementation: extract attention maps for supervision
                    out, attn = self.solar_gates[key](
                        feature_map, sun_vectors, return_attn=True
                    )
                    gated_features[key] = out
                    attention_maps[key] = attn
                else:
                    # Inference or training without shadow supervision
                    gated_features[key] = self.solar_gates[key](
                        feature_map, sun_vectors
                    )
            else:
                gated_features[key] = feature_map

        # 4. Standard Head Processing
        if isinstance(features, torch.Tensor):
            features = [features]
        if isinstance(gated_features, torch.Tensor):
            gated_features = [gated_features]

        head_outputs = self.head(list(gated_features.values()))

        # 5. Loss Calculation
        losses = {}
        detections = []

        if self.training:
            # Standard RetinaNet Losses (Box + Class)
            losses = self.compute_loss(targets, head_outputs, kwargs={})

            # --- LEVEL 2: GUIDED ATTENTION LOSS ---
            if shadow_masks is not None and attention_maps:
                attention_loss = self._compute_guided_attention_loss(
                    attention_maps, shadow_masks
                )
                losses["loss_solar_attention"] = attention_loss

        else:
            # Inference
            detections = self.postprocess_detections(
                head_outputs, original_image_sizes, images.image_sizes
            )
            return detections

        return losses


class SolarDeepForest(deepforest):
    """
    Wrapper class for the Solar-Gated Detector.
    Inherits from DeepForest to keep all the utility methods (predict_tile, etc.)
    but swaps the engine.
    """

    def create_model(self, backbone="resnet50", num_classes=1):
        """
        Override to load SolarRetinaNet instead of standard RetinaNet
        """
        self.model = SolarRetinaNet(
            num_classes=num_classes,
            backbone=None,  # Loads default FPN
            min_size=self.config["min_size"],
            max_size=self.config["max_size"],
            nms_thresh=self.config["nms_thresh"],
            score_thresh=self.config["score_thresh"],
        )

    def train(self, images, annotations, sun_vectors, padding=True):
        """
        Modified train loop to accept sun_vectors
        """
        # This would require rewriting the PyTorch Lightning module
        # or the custom training loop to pass the extra argument.
        pass

    def predict_tile(self, image, sun_vector, return_plot=False):
        """
        Modified inference
        """
        self.model.eval()
        # Transform image to tensor...
        # Pass (image, sun_vector) to model
        pass
