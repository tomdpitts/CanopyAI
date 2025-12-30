"""
Noise Augmentation Module for Tree Detection Training.

Provides two modes:
1. Standard Albumentations noise (for Papa model)
2. Shadow-Aware Noise Augmentation / SANA (for Quebec model)

Usage in training:
    from noise_augmentation import apply_noise, get_albumentations_pipeline

Author: CanopyAI
"""

import numpy as np
import cv2

try:
    import albumentations as A

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# =============================================================================
# Standard Albumentations Pipeline (Papa model)
# =============================================================================


def get_albumentations_pipeline() -> "A.Compose":
    """
    Standard noise augmentation pipeline using Albumentations.

    Includes:
    - GaussNoise: Gaussian noise with configurable variance
    - ISONoise: Camera sensor noise simulation
    - MultiplicativeNoise: Speckle-like noise

    Returns:
        Albumentations Compose pipeline
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for standard noise pipeline")

    return A.Compose(
        [
            # Use std_range for Albumentations 2.x (var_limit is deprecated)
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2),
        ]
    )


def apply_albumentations_noise(image: np.ndarray) -> np.ndarray:
    """
    Apply standard Albumentations noise to an image.

    Args:
        image: RGB image as numpy array (H, W, C), uint8

    Returns:
        Noisy image as numpy array, uint8
    """
    pipeline = get_albumentations_pipeline()
    result = pipeline(image=image)
    return result["image"]


# =============================================================================
# Shadow-Aware Noise Augmentation / SANA (Quebec model)
# =============================================================================


def detect_shadows(
    image: np.ndarray,
    threshold_ratio: float = 0.7,
    blur_size: int = 51,
    morph_size: int = 5,
) -> np.ndarray:
    """
    Detect shadow regions using adaptive luminance thresholding.

    Shadows are defined as pixels significantly darker than their local
    neighborhood. This is robust to varying lighting across the image.

    Args:
        image: RGB image as numpy array (H, W, C), uint8
        threshold_ratio: Pixels below (local_mean * ratio) are shadows
        blur_size: Gaussian blur kernel size for local mean computation
        morph_size: Morphological operation kernel size for cleanup

    Returns:
        Binary shadow mask (H, W), uint8 with 1=shadow, 0=non-shadow
    """
    # Convert to LAB and extract luminance channel
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # Compute local mean luminance
    local_mean = cv2.GaussianBlur(L, (blur_size, blur_size), 0)

    # Threshold: pixels significantly darker than local mean are shadows
    shadow_mask = (L < local_mean * threshold_ratio).astype(np.uint8)

    # Morphological cleanup to remove noise and fill small holes
    kernel = np.ones((morph_size, morph_size), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    return shadow_mask


def apply_sana(
    image: np.ndarray,
    shadow_noise_std: float = 5.0,
    ground_noise_std: float = 30.0,
    threshold_ratio: float = 0.7,
    blur_size: int = 51,
) -> np.ndarray:
    """
    Shadow-Aware Noise Augmentation (SANA).

    Applies different noise levels to shadow vs non-shadow regions:
    - Low noise on shadows (preserves discriminative feature)
    - High noise on ground/scrub (forces model to ignore texture)

    This is designed for rangeland imagery where trees and scrub have
    similar appearance, but shadows indicate vertical objects (trees).

    Args:
        image: RGB image as numpy array (H, W, C), uint8
        shadow_noise_std: Gaussian noise std for shadow regions (low)
        ground_noise_std: Gaussian noise std for non-shadow regions (high)
        threshold_ratio: Shadow detection threshold (see detect_shadows)
        blur_size: Shadow detection blur size (see detect_shadows)

    Returns:
        Noisy image as numpy array, uint8
    """
    # Detect shadow regions
    shadow_mask = detect_shadows(
        image,
        threshold_ratio=threshold_ratio,
        blur_size=blur_size,
    )

    # Generate noise fields for each region
    shadow_noise = np.random.normal(0, shadow_noise_std, image.shape)
    ground_noise = np.random.normal(0, ground_noise_std, image.shape)

    # Expand mask to 3 channels for broadcasting
    mask_3d = shadow_mask[:, :, np.newaxis].astype(np.float32)

    # Blend noise based on shadow mask
    # shadow regions get low noise, non-shadow gets high noise
    combined_noise = shadow_noise * mask_3d + ground_noise * (1 - mask_3d)

    # Apply noise and clip to valid range
    noisy = image.astype(np.float32) + combined_noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


# =============================================================================
# Unified Interface
# =============================================================================


def apply_noise(
    image: np.ndarray,
    mode: str = "none",
    **kwargs,
) -> np.ndarray:
    """
    Apply noise augmentation to an image.

    Args:
        image: RGB image as numpy array (H, W, C), uint8
        mode: Noise mode - "none", "albumentations", or "sana"
        **kwargs: Additional arguments passed to the noise function

    Returns:
        Augmented image as numpy array, uint8
    """
    if mode == "none":
        return image
    elif mode == "albumentations":
        return apply_albumentations_noise(image)
    elif mode == "sana":
        return apply_sana(image, **kwargs)
    else:
        raise ValueError(f"Unknown noise mode: {mode}")


def get_shadow_mask(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Utility function to get shadow mask for visualization/debugging.

    Args:
        image: RGB image as numpy array (H, W, C), uint8
        **kwargs: Arguments passed to detect_shadows

    Returns:
        Binary shadow mask (H, W), uint8 with 255=shadow, 0=non-shadow
    """
    mask = detect_shadows(image, **kwargs)
    return mask * 255  # Scale for visualization


# =============================================================================
# Detectron2 Transform Wrapper
# =============================================================================


class NoiseAugmentation:
    """
    Detectron2-compatible transform for noise augmentation.

    Usage in train.py:
        from noise_augmentation import NoiseAugmentation
        augs.append(NoiseAugmentation(mode="sana"))
    """

    def __init__(self, mode: str = "none", **kwargs):
        self.mode = mode
        self.kwargs = kwargs

    def get_transform(self, image):
        """Return a transform that applies noise."""
        from detectron2.data.transforms import Transform

        class _NoiseTransform(Transform):
            def __init__(self, mode, kwargs):
                super().__init__()
                self._mode = mode
                self._kwargs = kwargs

            def apply_image(self, img):
                return apply_noise(img, mode=self._mode, **self._kwargs)

            def apply_coords(self, coords):
                return coords  # Noise doesn't affect coordinates

            def apply_segmentation(self, segmentation):
                return segmentation  # Noise doesn't affect masks

        return _NoiseTransform(self.mode, self.kwargs)

    def __call__(self, aug_input):
        """Apply the transform to an AugInput."""
        transform = self.get_transform(aug_input.image)
        aug_input.image = transform.apply_image(aug_input.image)
        return transform


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test noise augmentation")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--mode", choices=["albumentations", "sana"], default="sana")
    parser.add_argument("--output", type=str, default="noise_test_output")
    parser.add_argument("--show_mask", action="store_true", help="Save shadow mask")
    args = parser.parse_args()

    # Load image
    img_path = Path(args.image)
    if img_path.suffix.lower() in [".tif", ".tiff"]:
        import rasterio

        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3]).transpose(1, 2, 0)
    else:
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Loaded image: {image.shape}")

    # Apply noise
    noisy = apply_noise(image, mode=args.mode)

    # Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original and noisy
    cv2.imwrite(
        str(output_dir / "original.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        str(output_dir / f"noisy_{args.mode}.png"),
        cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR),
    )

    if args.show_mask and args.mode == "sana":
        mask = get_shadow_mask(image)
        cv2.imwrite(str(output_dir / "shadow_mask.png"), mask)
        print(f"Saved shadow mask to {output_dir / 'shadow_mask.png'}")

    print(f"Saved outputs to {output_dir}")
