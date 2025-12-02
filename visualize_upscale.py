import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio


def visualize_upscale(input_path, target_size=1024):
    # Load original chip with rasterio
    try:
        with rasterio.open(input_path) as src:
            # Read first 3 bands (RGB)
            img = src.read([1, 2, 3])
            # Move channels to last dimension: (C, H, W) -> (H, W, C)
            img = np.moveaxis(img, 0, -1)

            # Normalize to 0-255 for visualization if needed
            if img.dtype != np.uint8:
                img = (img - img.min()) / (img.max() - img.min()) * 255.0
                img = img.astype(np.uint8)
    except Exception as e:
        print(f"Could not read {input_path}: {e}")
        return
    h, w = img.shape[:2]

    # Upscale to target size (simulating Detectron2 ResizeShortestEdge)
    # If shortest edge < target_size, it scales up
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img)
    axes[0].set_title(f"Original Chip ({w}x{h} px)")
    axes[0].axis("off")

    axes[1].imshow(upscaled)
    axes[1].set_title(f"Upscaled Input to Model ({new_w}x{new_h} px)")
    axes[1].axis("off")

    plt.tight_layout()
    out_path = "upscale_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    # Find a sample chip
    chips = list(Path("data/tcd/tiles_pred").glob("**/*.tif"))
    if chips:
        visualize_upscale(chips[0])
    else:
        print("No chips found to visualize.")
