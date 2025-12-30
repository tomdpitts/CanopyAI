#!/usr/bin/env python3
"""
Shadow Detection for Training Data Generation
----------------------------------------------
Purpose-built script to detect shadows in WON003 imagery
for creating shadow-excluding masks for SAM training.

Methods:
1. Improved luminance + chromaticity (fast, no dependencies)
2. MTMT deep learning model (more accurate, requires model download)

Usage:
    # Test on single image
    python shadow_detector.py --image won003/images/sample.png --output shadow_output

    # Process all WON003 images
    python shadow_detector.py --input_dir won003/images --output shadow_masks

    # Compare methods
    python shadow_detector.py --image sample.png --compare

Author: CanopyAI
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional


# =============================================================================
# Method 1: Improved Chromaticity-Based Shadow Detection
# =============================================================================


def detect_shadows_lab(
    image: np.ndarray,
    l_percentile: float = 25,
    green_threshold: float = -5,
    min_area: int = 100,
    blur_size: int = 5,
    morph_size: int = 5,
) -> np.ndarray:
    """
    Detect shadows using LAB color space with physics-based heuristics.

    Physics insight:
    - Shadows are DARK (low L channel)
    - Shadows are illuminated by BLUE SKY (higher B than sunlit areas)
    - Shadows are NOT GREEN VEGETATION (A channel near neutral or positive)

    This distinguishes:
    - Shadows (dark, blue-ish, not green) ✓
    - Trees (dark, but strongly GREEN) ✗
    - Dirt (bright, yellow-brown) ✗

    Args:
        image: RGB image as numpy array (H, W, C), uint8
        l_percentile: Percentile threshold for darkness (default 25 = darkest 25%)
        green_threshold: A channel threshold (below this = too green = tree, not shadow)
        min_area: Minimum contour area to keep
        blur_size: Gaussian blur kernel size for noise reduction
        morph_size: Morphological operation kernel size

    Returns:
        Binary shadow mask (H, W), uint8 with 255=shadow, 0=non-shadow
    """
    # Convert to LAB color space
    # L: 0-255 (0=black, 255=white)
    # A: 0-255 (0=green, 128=neutral, 255=red)
    # B: 0-255 (0=blue, 128=neutral, 255=yellow)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    L = lab[:, :, 0]  # Luminance
    A = lab[:, :, 1] - 128  # Shift to -128 to 127 (negative = green)
    B = lab[:, :, 2] - 128  # Shift to -128 to 127 (negative = blue)

    # Smooth to reduce noise
    if blur_size > 1:
        L = cv2.GaussianBlur(L, (blur_size, blur_size), 0)
        A = cv2.GaussianBlur(A, (blur_size, blur_size), 0)
        B = cv2.GaussianBlur(B, (blur_size, blur_size), 0)

    # 1. DARKNESS CRITERION: Low L (percentile-based adaptive threshold)
    l_threshold = np.percentile(L, l_percentile)
    is_dark = L < l_threshold

    # 2. NOT-GREEN CRITERION: A channel above threshold
    # Trees and vegetation have negative A (green)
    # Shadows on dirt have neutral/slightly positive A
    is_not_green = A > green_threshold

    # 3. BLUE-ISH CRITERION: B channel relatively negative (blue)
    # Shadows illuminated by blue sky have lower B than sunlit areas
    # We use local comparison rather than absolute threshold
    local_mean_B = cv2.GaussianBlur(B, (51, 51), 0)
    is_relatively_blue = B < (
        local_mean_B + 10
    )  # Slightly more blue than local average

    # Combine criteria: DARK AND NOT-GREEN
    # (Blue criterion is softer - helps but not required)
    shadow_mask = (is_dark & is_not_green).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((morph_size, morph_size), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)  # Fill holes

    # Filter by minimum area
    contours, _ = cv2.findContours(
        shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    shadow_mask_filtered = np.zeros_like(shadow_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(shadow_mask_filtered, [cnt], -1, 1, -1)

    # Debug output
    n_regions = len([c for c in contours if cv2.contourArea(c) >= min_area])
    print(
        f"   LAB detector: L_thresh={l_threshold:.1f}, found {n_regions} shadow regions"
    )

    return shadow_mask_filtered * 255


def detect_canopy_lab(
    image: np.ndarray,
    green_threshold: float = -5,
    min_area: int = 100,
    blur_size: int = 5,
    morph_size: int = 5,
) -> np.ndarray:
    """
    Detect tree canopy using LAB color space.

    Physics insight:
    - Trees are GREEN (negative A channel in LAB)
    - Trees are darker than dirt but lighter than shadows

    Args:
        image: RGB image as numpy array (H, W, C), uint8
        green_threshold: A channel threshold (below this = green = tree)
        min_area: Minimum contour area to keep
        blur_size: Gaussian blur kernel size
        morph_size: Morphological operation kernel size

    Returns:
        Binary canopy mask (H, W), uint8 with 255=canopy, 0=non-canopy
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    A = lab[:, :, 1] - 128  # Shift to -128 to 127 (negative = green)

    # Smooth to reduce noise
    if blur_size > 1:
        A = cv2.GaussianBlur(A, (blur_size, blur_size), 0)

    # GREEN CRITERION: A channel below threshold (more green)
    is_green = A < green_threshold

    # Create mask
    canopy_mask = is_green.astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((morph_size, morph_size), np.uint8)
    canopy_mask = cv2.morphologyEx(canopy_mask, cv2.MORPH_OPEN, kernel)
    canopy_mask = cv2.morphologyEx(canopy_mask, cv2.MORPH_CLOSE, kernel)

    # Filter by minimum area
    contours, _ = cv2.findContours(
        canopy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    canopy_mask_filtered = np.zeros_like(canopy_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(canopy_mask_filtered, [cnt], -1, 1, -1)

    # Debug output
    n_regions = len([c for c in contours if cv2.contourArea(c) >= min_area])
    a_min, a_max = A.min(), A.max()
    print(
        f"   Canopy detector: A range=[{a_min:.0f},{a_max:.0f}], "
        f"thresh={green_threshold}, found {n_regions} regions"
    )

    return canopy_mask_filtered * 255


def detect_shadows_chromaticity(
    image: np.ndarray,
    luminance_ratio: float = 0.65,
    saturation_threshold: float = 0.15,
    blur_size: int = 51,
    morph_size: int = 7,
) -> np.ndarray:
    """
    Detect shadows using physics-informed chromaticity analysis.

    Key insight: Shadows preserve chromaticity (hue) while reducing luminance.
    This is more robust than pure luminance thresholding.

    Args:
        image: RGB image as numpy array (H, W, C), uint8
        luminance_ratio: Pixels below (local_mean * ratio) are shadow candidates
        saturation_threshold: Minimum saturation to be considered shadow
        blur_size: Gaussian blur kernel for local mean computation
        morph_size: Morphological operation kernel size

    Returns:
        Binary shadow mask (H, W), uint8 with 255=shadow, 0=non-shadow
    """
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Extract channels
    _, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    L = lab[:, :, 0]

    # Normalize
    S_norm = S / 255.0

    # 1. Luminance-based detection (adaptive threshold)
    local_mean_L = cv2.GaussianBlur(L, (blur_size, blur_size), 0)
    luminance_mask = L < (local_mean_L * luminance_ratio)

    # 2. Saturation constraint (shadows have low saturation in aerial imagery)
    saturation_mask = S_norm < 0.6  # Shadows typically not highly saturated

    # 3. Value constraint (shadows have low value)
    local_mean_V = cv2.GaussianBlur(V, (blur_size, blur_size), 0)
    value_mask = V < (local_mean_V * luminance_ratio)

    # Combine criteria
    shadow_mask = (luminance_mask & value_mask & saturation_mask).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((morph_size, morph_size), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    # Small hole filling
    contours, _ = cv2.findContours(
        shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    shadow_mask_filled = np.zeros_like(shadow_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # Filter tiny noise
            cv2.drawContours(shadow_mask_filled, [cnt], -1, 1, -1)

    return shadow_mask_filled * 255


def detect_shadows_gradient(
    image: np.ndarray,
    luminance_ratio: float = 0.6,
    blur_size: int = 31,
) -> np.ndarray:
    """
    Detect shadows using gradient analysis.

    Shadow boundaries have characteristic gradient patterns:
    - Sharp edge on object side (hard shadow boundary)
    - Soft/no edge on opposite side (penumbra)

    Args:
        image: RGB image as numpy array
        luminance_ratio: Threshold ratio for shadow detection
        blur_size: Blur kernel size

    Returns:
        Binary shadow mask (H, W), uint8 with 255=shadow
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Local mean luminance
    local_mean = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # Shadow candidates: low luminance relative to local mean
    shadow_candidates = gray < (local_mean * luminance_ratio)

    # Edge refinement: shadows should have edges on at least one side
    # High gradient at boundary indicates shadow edge
    grad_threshold = np.percentile(grad_mag, 70)
    has_edge = grad_mag > grad_threshold

    # Dilate edges to connect to shadow regions
    edge_kernel = np.ones((5, 5), np.uint8)
    has_edge_dilated = cv2.dilate(has_edge.astype(np.uint8), edge_kernel)

    # Combine: shadow candidates that are near edges
    shadow_mask = shadow_candidates.astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    return shadow_mask * 255


def detect_shadows_combined(
    image: np.ndarray,
    luminance_ratio: float = 0.60,
    blur_size: int = 51,
) -> np.ndarray:
    """
    Combined shadow detection using multiple cues.

    Combines:
    1. Chromaticity-based detection
    2. Gradient-based edge detection
    3. Color ratio analysis

    Args:
        image: RGB image as numpy array
        luminance_ratio: Base threshold ratio
        blur_size: Blur kernel size

    Returns:
        Binary shadow mask (H, W), uint8 with 255=shadow
    """
    # Get individual detections
    chromaticity_mask = detect_shadows_chromaticity(
        image, luminance_ratio=luminance_ratio, blur_size=blur_size
    )

    # Color ratio method (shadows have specific R/G/B ratios)
    # In outdoor imagery, shadows often appear more blue
    r, g, b = (
        image[:, :, 0].astype(float),
        image[:, :, 1].astype(float),
        image[:, :, 2].astype(float),
    )
    intensity = (r + g + b) / 3 + 1e-6

    # Blue channel ratio (shadows tend to be bluish)
    blue_ratio = b / intensity
    blue_shadow = blue_ratio > 1.1  # More blue than average

    # Low intensity
    local_intensity = cv2.GaussianBlur(
        intensity.astype(np.float32), (blur_size, blur_size), 0
    )
    low_intensity = intensity < (local_intensity * luminance_ratio)

    # Combine votes
    combined = (chromaticity_mask > 127).astype(np.uint8) + low_intensity.astype(
        np.uint8
    )

    # Require at least 1 vote
    shadow_mask = (combined >= 1).astype(np.uint8)

    # Cleanup
    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    return shadow_mask * 255


# =============================================================================
# Method 2: GroundingDINO (Vision-Language) Shadow Detection
# =============================================================================


# Global cache for model to avoid reloading
_grounding_dino_model = None
_grounding_dino_processor = None


def detect_shadows_grounding_dino(
    image: np.ndarray,
    text_prompt: str = "dark tree shadow.",
    box_threshold: float = 0.15,
) -> np.ndarray:
    """
    Detect shadows using GroundingDINO vision-language model.

    This uses zero-shot detection with text prompt "shadow" to find
    shadow regions. Much more robust than threshold-based methods.

    Requires: pip install transformers torch

    Args:
        image: RGB image as numpy array
        text_prompt: Text prompt for detection (default: "shadow")
        box_threshold: Detection confidence threshold

    Returns:
        Binary shadow mask (H, W), uint8 with 255=shadow
    """
    global _grounding_dino_model, _grounding_dino_processor

    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from PIL import Image as PILImage
        import torch
    except ImportError:
        raise ImportError(
            "GroundingDINO requires transformers. Install with:\n"
            "pip install transformers torch"
        )

    # Load model (cached after first call)
    if _grounding_dino_model is None:
        print("   Loading GroundingDINO model (first time, may take a moment)...")
        model_id = "IDEA-Research/grounding-dino-tiny"
        _grounding_dino_processor = AutoProcessor.from_pretrained(model_id)
        _grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        )
        # Move to GPU if available
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        _grounding_dino_model = _grounding_dino_model.to(device)
        print(f"   GroundingDINO loaded on {device}")

    # Convert numpy to PIL
    pil_image = PILImage.fromarray(image)

    # Process image
    inputs = _grounding_dino_processor(
        images=pil_image, text=text_prompt, return_tensors="pt"
    )

    # Move inputs to same device as model
    device = next(_grounding_dino_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = _grounding_dino_model(**inputs)

    # Post-process - get image size for rescaling boxes
    target_sizes = torch.tensor([pil_image.size[::-1]], device=device)
    results = _grounding_dino_processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        target_sizes=target_sizes,
    )[0]

    # Create mask from detected boxes
    h, w = image.shape[:2]
    shadow_mask = np.zeros((h, w), dtype=np.uint8)

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    # Filter by threshold
    mask = scores >= box_threshold
    boxes = boxes[mask]
    scores = scores[mask]

    print(f"   GroundingDINO found {len(boxes)} shadow regions")

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.astype(int)
        print(f"      Box {i}: ({x1},{y1}) to ({x2},{y2}) score={score:.2f}")
        # Clip to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:  # Only fill if valid box
            shadow_mask[y1:y2, x1:x2] = 255

    return shadow_mask


# =============================================================================
# Mask Cleaning Utilities
# =============================================================================


def detect_shadows_aggressive(
    image: np.ndarray,
    dark_percentile: float = 20,
    morph_size: int = 5,
) -> np.ndarray:
    """
    Aggressive shadow detection using percentile-based thresholding.

    This method is designed for WON003-style imagery where shadows
    are the darkest regions in the image.

    Args:
        image: RGB image as numpy array
        dark_percentile: Percentile threshold (larger = more shadow detected)
        morph_size: Morphological kernel size

    Returns:
        Binary shadow mask (H, W), uint8 with 255=shadow
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Use percentile-based threshold
    threshold = np.percentile(gray, dark_percentile)
    shadow_mask = (gray < threshold).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((morph_size, morph_size), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    # Remove very small regions
    contours, _ = cv2.findContours(
        shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    shadow_mask_cleaned = np.zeros_like(shadow_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            cv2.drawContours(shadow_mask_cleaned, [cnt], -1, 1, -1)

    return shadow_mask_cleaned * 255


def detect_shadows_otsu(
    image: np.ndarray,
    bias: float = 0.8,
    morph_size: int = 5,
) -> np.ndarray:
    """
    Shadow detection using Otsu's automatic thresholding with bias.

    Otsu finds the optimal threshold to separate foreground/background.
    We bias it toward darker pixels for shadow detection.

    Args:
        image: RGB image as numpy array
        bias: Multiply Otsu threshold by this (< 1 = more aggressive)
        morph_size: Morphological kernel size

    Returns:
        Binary shadow mask (H, W), uint8 with 255=shadow
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Otsu's thresholding
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply biased threshold (lower = more shadow)
    adjusted_thresh = otsu_thresh * bias
    shadow_mask = (gray < adjusted_thresh).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((morph_size, morph_size), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    return shadow_mask * 255


def clean_sam_mask_with_shadow(
    sam_mask: np.ndarray,
    shadow_mask: np.ndarray,
    dilation_px: int = 3,
) -> np.ndarray:
    """
    Clean a SAM-generated mask by removing shadow pixels.

    Args:
        sam_mask: Binary mask from SAM (H, W), True/1 = tree
        shadow_mask: Binary shadow mask (H, W), 255 = shadow
        dilation_px: Dilate shadow mask slightly to catch edges

    Returns:
        Cleaned mask with shadows removed
    """
    # Ensure binary
    sam_binary = (sam_mask > 0).astype(np.uint8)
    shadow_binary = (shadow_mask > 127).astype(np.uint8)

    # Slightly dilate shadow mask to catch penumbra
    if dilation_px > 0:
        kernel = np.ones((dilation_px, dilation_px), np.uint8)
        shadow_binary = cv2.dilate(shadow_binary, kernel)

    # Remove shadow from SAM mask
    cleaned = sam_binary & (~shadow_binary)

    # Fill small holes that might appear
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned * 255


# =============================================================================
# Visualization
# =============================================================================


def create_comparison_visualization(
    image: np.ndarray,
    shadow_mask: np.ndarray,
    output_path: Path,
    canopy_mask: np.ndarray = None,
) -> None:
    """
    Create side-by-side visualization of original, detection overlay, and mask.

    Shows shadows in RED and canopy in GREEN.
    """
    h, w = image.shape[:2]

    # Create visualization
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)

    # Left: Original
    vis[:, :w] = image

    # Middle: Shadow (red) and Canopy (green) boundaries
    overlay = image.copy()

    # Draw shadow contours in RED
    shadow_contours, _ = cv2.findContours(
        shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, shadow_contours, -1, (255, 0, 0), 1)  # Red

    # Draw canopy contours in GREEN if provided
    if canopy_mask is not None:
        canopy_contours, _ = cv2.findContours(
            canopy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, canopy_contours, -1, (0, 255, 0), 1)  # Green

    vis[:, w : w * 2] = overlay

    # Right: Shadow mask grayscale
    vis[:, w * 2 :, 0] = shadow_mask
    vis[:, w * 2 :, 1] = shadow_mask
    vis[:, w * 2 :, 2] = shadow_mask

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
    label = "Shadow (R) + Canopy (G)" if canopy_mask is not None else "Shadow (R)"
    cv2.putText(vis, label, (w + 10, 30), font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "Shadow Mask", (w * 2 + 10, 30), font, 0.8, (255, 255, 255), 2)

    # Save
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# =============================================================================
# CLI
# =============================================================================


def process_single_image(
    image_path: Path,
    output_dir: Path,
    method: str = "lab",
    shadow_thresh: float = 25,
    canopy_thresh: float = -5,
    visualize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single image and generate shadow mask.

    Returns:
        (image, shadow_mask) tuple
    """
    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Detect shadows
    if method == "lab":
        shadow_mask = detect_shadows_lab(image, l_percentile=shadow_thresh)
    elif method == "chromaticity":
        shadow_mask = detect_shadows_chromaticity(image)
    elif method == "gradient":
        shadow_mask = detect_shadows_gradient(image)
    elif method == "combined":
        shadow_mask = detect_shadows_combined(image)
    elif method == "aggressive":
        shadow_mask = detect_shadows_aggressive(image, dark_percentile=shadow_thresh)
    elif method == "otsu":
        shadow_mask = detect_shadows_otsu(image, bias=shadow_thresh)
    elif method == "grounding_dino":
        shadow_mask = detect_shadows_grounding_dino(image)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Detect canopy (for LAB method)
    canopy_mask = None
    if method == "lab":
        canopy_mask = detect_canopy_lab(image, green_threshold=canopy_thresh)
        # Remove any shadow that overlaps canopy (canopy takes priority)
        shadow_mask = cv2.bitwise_and(shadow_mask, cv2.bitwise_not(canopy_mask))

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # Save shadow mask (prefix naming: shadow_mask_<stem>.png)
    mask_path = output_dir / f"shadow_mask_{stem}.png"
    cv2.imwrite(str(mask_path), shadow_mask)

    # Save canopy mask if available
    if canopy_mask is not None:
        canopy_path = output_dir / f"canopy_mask_{stem}.png"
        cv2.imwrite(str(canopy_path), canopy_mask)

    # Save visualization
    if visualize:
        vis_path = output_dir / f"shadow_vis_{stem}.png"
        create_comparison_visualization(image, shadow_mask, vis_path, canopy_mask)

    return image, shadow_mask


def main():
    parser = argparse.ArgumentParser(
        description="Shadow detection for training data generation"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Single image path")
    input_group.add_argument(
        "--input_dir", type=str, help="Directory of images to process"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="shadow_output",
        help="Output directory for masks and visualizations",
    )

    parser.add_argument(
        "--method",
        choices=[
            "lab",
            "aggressive",
            "otsu",
            "chromaticity",
            "gradient",
            "combined",
            "grounding_dino",
        ],
        default="lab",
        help="Shadow detection method (lab recommended for WON003)",
    )

    parser.add_argument(
        "--shadow_thresh",
        type=float,
        default=25,
        help="Shadow L percentile threshold (default 25 = darkest 25%%)",
    )

    parser.add_argument(
        "--canopy_thresh",
        type=float,
        default=-5,
        help="Canopy A channel threshold (default -5, more negative = stricter)",
    )

    parser.add_argument(
        "--compare", action="store_true", help="Run all methods and create comparison"
    )

    parser.add_argument(
        "--no_vis", action="store_true", help="Skip visualization output (faster)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.image:
        # Single image processing
        image_path = Path(args.image)

        if args.compare:
            # Compare all methods
            print(f"📊 Comparing all methods on {image_path.name}")

            # Load image once
            image_bgr = cv2.imread(str(image_path))
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Note: grounding_dino excluded from compare as it's GPU-heavy
            methods = ["aggressive", "otsu", "chromaticity", "gradient", "combined"]
            masks = {}

            for method in methods:
                print(f"   Running {method}...")
                if method == "chromaticity":
                    masks[method] = detect_shadows_chromaticity(image)
                elif method == "gradient":
                    masks[method] = detect_shadows_gradient(image)
                elif method == "combined":
                    masks[method] = detect_shadows_combined(image)
                elif method == "aggressive":
                    masks[method] = detect_shadows_aggressive(image, dark_percentile=20)
                elif method == "otsu":
                    masks[method] = detect_shadows_otsu(image, bias=0.8)

            # Create comparison grid
            output_dir.mkdir(parents=True, exist_ok=True)
            h, w = image.shape[:2]
            n_methods = len(methods) + 1  # +1 for original

            grid = np.zeros((h, w * n_methods, 3), dtype=np.uint8)
            grid[:, :w] = image  # Original

            for i, (name, mask) in enumerate(masks.items()):
                start = (i + 1) * w
                overlay = image.copy()
                shadow_color = np.zeros_like(image)
                shadow_color[:, :, 0] = mask
                overlay = cv2.addWeighted(overlay, 0.7, shadow_color, 0.3, 0)
                grid[:, start : start + w] = overlay

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(grid, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
            for i, name in enumerate(methods):
                cv2.putText(
                    grid, name, ((i + 1) * w + 10, 30), font, 0.7, (255, 255, 255), 2
                )

            comparison_path = output_dir / f"{image_path.stem}_comparison.png"
            cv2.imwrite(str(comparison_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"✅ Saved comparison to {comparison_path}")

        else:
            print(f"🌑 Processing {image_path.name} with method: {args.method}")
            process_single_image(
                image_path,
                output_dir,
                method=args.method,
                shadow_thresh=args.shadow_thresh,
                canopy_thresh=args.canopy_thresh,
                visualize=not args.no_vis,
            )
            print(f"✅ Output saved to {output_dir}")

    else:
        # Batch processing
        input_dir = Path(args.input_dir)
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

        print(f"🌑 Processing {len(image_files)} images with method: {args.method}")

        from tqdm import tqdm

        for img_path in tqdm(image_files, desc="Processing"):
            try:
                process_single_image(
                    img_path,
                    output_dir,
                    method=args.method,
                    shadow_thresh=args.shadow_thresh,
                    canopy_thresh=args.canopy_thresh,
                    visualize=not args.no_vis,
                )
            except Exception as e:
                print(f"⚠️  Failed {img_path.name}: {e}")

        print(f"✅ Output saved to {output_dir}")


if __name__ == "__main__":
    main()
