import torch
import cv2
import numpy as np
import os
import sys

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prototypes.solar_adapter import GlobalContextEncoder
from prototypes.solar_deepforest import SolarDeepForest
from segment_anything import sam_model_registry


def run_pipeline(
    image_path, ortho_path, context_weights, detector_weights, sam_checkpoint
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # STAGE 0: GLOBAL CONTEXT
    # =========================================================================
    print("--- Stage 0: Inferring Global Sun Vector ---")

    # Load Context Model
    context_model = GlobalContextEncoder().to(device)
    context_model.load_state_dict(torch.load(context_weights, map_location=device))
    context_model.eval()

    # Process Orthomosaic (Simulated as random crops logic)
    # Ideally: Load big ortho -> random crops -> consensus
    # Here: Just resize image for prototype
    ortho = cv2.imread(ortho_path)
    ortho_resized = cv2.resize(ortho, (512, 512))
    ortho_tensor = (
        torch.from_numpy(ortho_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
        / 255.0
    )

    with torch.no_grad():
        sun_vector = context_model(ortho_tensor)

    print(f"Global Sun Vector: {sun_vector.cpu().numpy()}")

    # =========================================================================
    # STAGE 1: SOLAR-GATED DETECTION
    # =========================================================================
    print("--- Stage 1: Solar-Gated Detection ---")

    solar_df = SolarDeepForest()
    solar_df.create_model()
    solar_df.model.load_state_dict(torch.load(detector_weights, map_location=device))
    solar_df.model.to(device)
    solar_df.model.eval()

    # Load high-res tile for detection
    tile = cv2.imread(image_path)
    # Preprocess (deepforest expects numpy, handles transform internally)
    # But our modifies predict_tile needs explicit manual handling in this raw script
    # Simulating the internal call:
    tile_tensor = (
        torch.from_numpy(tile).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    )

    with torch.no_grad():
        # Solar Retinanet Forward
        # Returns List[Dict] (boxes, labels, scores)
        detections = solar_df.model(tile_tensor, sun_vector)

    boxes = detections[0]["boxes"]
    print(f"Detected {len(boxes)} trees.")

    # =========================================================================
    # STAGE 2: SAM SEGMENTATION (Refined)
    # =========================================================================
    print("--- Stage 2: SAM Segmentation ---")

    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device)

    # Pass boxes to SAM...
    # (Standard Foxtrot Logic)
    print("Pipeline Complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ortho", required=True)
    parser.add_argument("--context_ckpt", default="solar_context.pth")
    parser.add_argument("--detector_ckpt", default="solar_deepforest.pth")
    parser.add_argument("--sam_ckpt", default="sam_vit_b_01ec64.pth")
    args = parser.parse_args()

    run_pipeline(
        args.image, args.ortho, args.context_ckpt, args.detector_ckpt, args.sam_ckpt
    )
