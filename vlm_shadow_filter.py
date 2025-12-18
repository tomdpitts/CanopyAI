"""
VLM Shadow Filter Module
------------------------
Uses Moondream 2 VLM to identify and filter shadow false positives
from tree detection results.

Usage:
    from vlm_shadow_filter import filter_shadows
    shadow_indices = filter_shadows(image, bboxes)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Lazy-loaded model
_vlm_model = None


def load_vlm_model():
    """
    Lazy load Moondream 2 VLM model.
    Only loads on first call, subsequent calls return cached model.
    """
    global _vlm_model
    if _vlm_model is not None:
        return _vlm_model

    print("\n🤖 Loading Moondream 2 VLM for shadow detection...")
    try:
        import moondream as md

        _vlm_model = md.vl(model="moondream-2b-int8")
        print("   ✅ VLM loaded successfully")
        return _vlm_model
    except ImportError:
        raise ImportError("Moondream not installed. Run: pip install moondream")
    except Exception as e:
        raise RuntimeError(f"Failed to load VLM: {e}")


def create_detection_overlay(image, bboxes, max_boxes=50):
    """
    Create image with numbered bounding boxes overlaid for VLM analysis.

    Args:
        image: RGB numpy array (H, W, 3)
        bboxes: List of [xmin, ymin, xmax, ymax]
        max_boxes: Maximum boxes to include (VLM context limit)

    Returns:
        PIL Image with numbered boxes drawn
    """
    # Convert to PIL
    pil_image = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(pil_image)

    # Use default font (works cross-platform)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    # Limit boxes for VLM context
    boxes_to_draw = bboxes[:max_boxes]

    for i, bbox in enumerate(boxes_to_draw):
        xmin, ymin, xmax, ymax = [int(c) for c in bbox]

        # Draw box outline (yellow for visibility)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="yellow", width=2)

        # Draw number label
        label = str(i + 1)  # 1-indexed for human readability
        text_bbox = draw.textbbox((xmin, ymin - 16), label, font=font)
        draw.rectangle(text_bbox, fill="yellow")
        draw.text((xmin, ymin - 16), label, fill="black", font=font)

    return pil_image


def query_shadow_detection(model, image_pil, num_boxes):
    """
    Query VLM to identify which detections are shadows.

    Args:
        model: Loaded Moondream model
        image_pil: PIL Image with numbered boxes
        num_boxes: Total number of boxes in image

    Returns:
        List of box indices (0-indexed) identified as shadows
    """
    prompt = f"""This aerial image shows {num_boxes} detected regions marked with numbered yellow boxes.
Some detections may be tree SHADOWS on the ground rather than actual tree canopies.

Shadows are characterized by:
1. Dark patches on bare ground (not textured canopy)
2. Shape is sheared/elongated in a consistent direction across the image
3. Located adjacent to an actual tree, on the opposite side from the shear direction
4. All shadows point the same direction (determined by sun angle)

Looking at this image, list ONLY the box numbers that appear to be shadows, not actual trees.
If no boxes appear to be shadows, respond with just "NONE".

Respond with ONLY comma-separated numbers or "NONE". No explanation needed."""

    try:
        response = model.query(image_pil, prompt)
        return parse_vlm_response(response, num_boxes)
    except Exception as e:
        print(f"   ⚠️ VLM query failed: {e}")
        return []


def parse_vlm_response(response, num_boxes):
    """
    Parse VLM response to extract shadow box indices.

    Args:
        response: Raw VLM response string
        num_boxes: Total number of boxes (for validation)

    Returns:
        List of 0-indexed box indices identified as shadows
    """
    response = response.strip().upper()

    if "NONE" in response or not response:
        return []

    shadow_indices = []
    # Extract numbers from response
    import re

    numbers = re.findall(r"\d+", response)

    for num_str in numbers:
        try:
            box_num = int(num_str)
            # Convert from 1-indexed (display) to 0-indexed (internal)
            if 1 <= box_num <= num_boxes:
                shadow_indices.append(box_num - 1)
        except ValueError:
            continue

    return shadow_indices


def filter_shadows(image, bboxes, scores=None, max_boxes_per_query=50):
    """
    Main entry point: Use VLM to identify and filter shadow false positives.

    Args:
        image: RGB numpy array (H, W, 3)
        bboxes: List of [xmin, ymin, xmax, ymax]
        scores: Optional confidence scores (used to prioritize which boxes to check)
        max_boxes_per_query: Maximum boxes per VLM query (context limit)

    Returns:
        shadow_indices: List of indices to remove (0-indexed)
    """
    if len(bboxes) == 0:
        return []

    print(f"\n🔍 Running VLM shadow detection on {len(bboxes)} detections...")

    # Load model
    model = load_vlm_model()

    all_shadow_indices = []

    # Process in batches if too many boxes
    if len(bboxes) <= max_boxes_per_query:
        # Single query
        overlay = create_detection_overlay(image, bboxes)
        shadow_indices = query_shadow_detection(model, overlay, len(bboxes))
        all_shadow_indices.extend(shadow_indices)
    else:
        # Multiple queries for different regions
        # For simplicity, just process first batch in POC
        print(f"   Processing first {max_boxes_per_query} boxes (POC limit)")
        overlay = create_detection_overlay(image, bboxes[:max_boxes_per_query])
        shadow_indices = query_shadow_detection(model, overlay, max_boxes_per_query)
        all_shadow_indices.extend(shadow_indices)

    if all_shadow_indices:
        print(
            f"   🌑 Identified {len(all_shadow_indices)} likely shadows: "
            f"boxes {[i + 1 for i in all_shadow_indices]}"
        )
    else:
        print("   ✅ No shadows detected")

    return all_shadow_indices
