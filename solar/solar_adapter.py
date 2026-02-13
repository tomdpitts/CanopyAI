import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def compute_shadow_channel(rgb_images):
    """
    Compute a 'shadow likeness' channel from RGB images.
    Shadows are characterized by low luminance and high blue/red ratio (Rayleigh scattering).

    Args:
        rgb_images: (B, 3, H, W) tensor, normalized or not (assumes roughly [0,1] or standard standardization)

    Returns:
        shadow_channel: (B, 1, H, W) tensor
    """
    # If images are standardized (e.g. ImageNet mean/std), we might need to un-standardize to get color ratios?
    # For now, let's work on the assumption that relative differences persist.
    # Typically resnet inputs are normalized.
    # Let's assume input is standard tensor.

    r, g, b = rgb_images[:, 0], rgb_images[:, 1], rgb_images[:, 2]

    # 1. Luminance (darkness)
    # Standard coeffs: 0.299R + 0.587G + 0.114B
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    # Inverted: higher value = darker = more likely shadow
    darkness = 1.0 - luminance

    # 2. Blue Ratio / Blue Shift
    # Shadows are illuminated by the blue sky, so they are bluer than direct sunlit areas.
    # Simple proxy: (B - R). Normalized to roughly [0, 1] range if possible, or just used as is.
    blue_shift = b - r

    # Combine: Shadow = Dark AND Blue-ish
    # Using sigmoid to squash range or just raw multiplication
    # Let's keep it simple and differentiable
    shadow_score = darkness + blue_shift

    # Add Gaussian Blur to suppress high-frequency noise (leaf texture)
    # and focus on structural shadows (tree casting shadow)
    # Kernel size 7, sigma 3 seems reasonable for 10cm/px resolution
    blur = torch.nn.Sequential(
        torch.nn.ReflectionPad2d(3),
        torch.nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=0, bias=False, groups=1),
    )
    # Initialize with Gaussian kernel
    with torch.no_grad():
        sigma = 3.0
        k = 7
        x = torch.arange(k).float() - (k - 1) / 2
        y = torch.arange(k).float() - (k - 1) / 2
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        blur[1].weight.data = kernel.view(1, 1, k, k)
        # Move to correct device
        if rgb_images.device.type != "cpu":
            blur.to(rgb_images.device)

    shadow_score = shadow_score.unsqueeze(1)  # (B, 1, H, W)
    shadow_score = blur(shadow_score)

    return shadow_score


class GlobalContextEncoder(nn.Module):
    """
    Stage 0: Unsupervised 'Sun Vector' Estimator.

    Takes a 500x500 image tile (RGB or RGB+Shadow) and outputs a normalized
    2D vector representing the global scene directionality (Sun Azimuth).

    Training Objective: Self-Supervised Rotational Equivariance.
    """

    def __init__(self):
        super().__init__()
        # Use a lightweight backbone
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )  # this model is trained on ImageNet-1K.

        # Standard ResNet18 accepts 3 channels (RGB)
        # We use the backbone as-is, removing the classification head (fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # New head: Regress to 2D vector (x, y)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Output (x, y)
        )

    def forward(self, x):
        # x: (B, 3, H, W) - Standard RGB

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
