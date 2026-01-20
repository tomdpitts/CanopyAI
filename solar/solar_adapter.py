import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GlobalContextEncoder(nn.Module):
    """
    Stage 0: Unsupervised 'Sun Vector' Estimator.

    Takes a 500x500 image tile and outputs a normalized
    2D vector representing the global scene directionality (Sun Azimuth).

    Training Objective: Self-Supervised Rotational Equivariance.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Use a lightweight backbone
        resnet = models.resnet18(pretrained=pretrained)

        # Remove the classification head (fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # New head: Regress to 2D vector (x, y)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Output (x, y)
        )

    def forward(self, x):
        # x: (B, 3, H, W) - typically 500x500 tiles
        features = self.backbone(x)  # (B, 512, 1, 1)
        features = torch.flatten(features, 1)  # (B, 512)

        vector = self.projection_head(features)  # (B, 2)

        # Normalize to unit vector (direction only)
        vector = F.normalize(vector, p=2, dim=1)
        return vector


class SolarAttentionBlock(nn.Module):
    """
    Solar-Gated Spatial Attention.

    Injects the global 'Sun Vector' into local feature maps (from FPN).
    Learns to upweight features that are consistent with the solar direction.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 1. Project global vector to channel interactions
        # We want the sun vector to shift the feature distribution
        self.sun_mlp = nn.Sequential(
            nn.Linear(2, channels // 4), nn.ReLU(), nn.Linear(channels // 4, channels)
        )

        # 2. Spatial Gating Convolution
        # A 3x3 conv that learns to look at 'offsets' dictated by the features
        self.gate_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, sun_vector, return_attn=False):
        """
        x: Feature map (B, C, H, W)
        sun_vector: Global context (B, 2)
        return_attn: If True, return (output, attention_map) tuple
        """
        batch_size, C, H, W = x.shape

        # --- FiLM (Feature-wise Linear Modulation) style conditioning ---

        # Project sun vector to channel dimensions
        sun_embedding = self.sun_mlp(sun_vector)  # (B, C)
        sun_embedding = sun_embedding.view(batch_size, C, 1, 1)  # (B, C, 1, 1)

        # Modulate features (Additive injection of context)
        # This tells the conv kernels "The sun is roughly in this configuration"
        conditioned_features = x + sun_embedding

        # --- Spatial Attention Map ---

        # Compute attention score based on conditioned features
        # The conv weights will learn to detect "Shadow-like" patterns
        # The sun_embedding bias pushes the activation towards the correct shadow orientation
        attn_map = self.gate_conv(conditioned_features)  # (B, 1, H, W)
        attn_map = self.sigmoid(attn_map)

        # --- Gating ---

        # Apply soft attention mask to original features
        # "Dim the lights" on features that don't match our shadow expectation
        out = x * attn_map

        if return_attn:
            return out, attn_map
        return out
