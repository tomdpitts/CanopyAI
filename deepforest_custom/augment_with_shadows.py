import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import math

# Configuration
INPUT_CSV = "deepforest_custom/train_phase3_final_split.csv"
OUTPUT_CSV = "deepforest_custom/train_phase3_augmented.csv"
WON_ANGLE = 215  # Shadow azimuth (0=North CW): SSW — as annotated in local_dataset.py
BRU_ANGLE = 118  # Shadow azimuth (0=North CW): ESE/SEE. Original 332° was mathematical (0=East CCW); (90-332)%360=118°


def rotate_box(box, width, height, angle_blocks):
    """
    Rotate bbox (xmin, ymin, xmax, ymax) by 90*angle_blocks degrees COUNTER-CLOCKWISE.
    """
    xmin, ymin, xmax, ymax = box

    for _ in range(angle_blocks):
        # 90 deg CCW
        # new_x = y
        # new_y = width - x
        # new_width = height
        # new_height = width

        # Points:
        # (x, y) -> (y, W - x)

        p1 = (xmin, ymin)
        p2 = (xmax, ymax)
        p3 = (xmin, ymax)
        p4 = (xmax, ymin)

        points = [p1, p2, p3, p4]
        new_points = []
        for x, y in points:
            new_x = y
            new_y = (
                width - x
            )  # 0-indexed? Dimensions are usually 1-to-N, but pixel coords 0..N-1
            new_points.append((new_x, new_y))

        xs = [p[0] for p in new_points]
        ys = [p[1] for p in new_points]

        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)

        # Swap dimension for next iteration
        width, height = height, width

    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def augment_data():
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Get unique images
    unique_images = df["image_path"].unique()
    print(f"Found {len(unique_images)} unique images.")

    augmented_rows = []

    # Pre-calculate files to avoid re-processing if multiple rows reference same file
    # Actually we just iterate unique images and filter DF

    for img_path in tqdm(unique_images):
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        # Determine Source and Angle
        if "won" in img_path.lower():
            base_angle = WON_ANGLE
        else:
            base_angle = BRU_ANGLE

        # Load Image
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                original_w, original_h = im.size

                # Get all annotations for this image
                img_rows = df[df["image_path"] == img_path]
                boxes = img_rows[["xmin", "ymin", "xmax", "ymax", "label"]].values

                # We need to generate 0, 90, 180, 270
                # 0 is original
                # Each rotation is +90 deg CCW

                for i in range(4):
                    # i=0: 0 deg
                    # i=1: 90 deg CCW
                    # i=2: 180 deg
                    # i=3: 270 deg

                    # Rotated Image Name
                    if i == 0:
                        save_path = img_path
                        # Current dimensions
                        curr_w, curr_h = original_w, original_h
                    else:
                        base, ext = os.path.splitext(img_path)
                        save_path = f"{base}_rot{i * 90}{ext}"

                        # Rotate image
                        # PIL rotate is CCW
                        rot_img = im.rotate(i * 90, expand=True)
                        rot_img.save(save_path)

                        curr_w, curr_h = rot_img.size

                    # Update Angle
                    # If we rotate image CCW by 90, the sun vector (fixed in world)
                    # rotates CW by 90 relative to image frame.
                    # Wait.
                    # Angle 0 (East) -> (1, 0).
                    # Image Rotated 90 CCW (North is now Left).
                    # East (original Right) is now Down (270).
                    # So +90 Image Rot = -90 Vector Rot.
                    current_angle = (base_angle - (i * 90)) % 360

                    # Calculate vector components (azimuth convention: 0=North, CW)
                    # Matches local_dataset.py: target = [sin(azimuth), cos(azimuth)]
                    rad = math.radians(current_angle)
                    sx = math.sin(rad)
                    sy = math.cos(rad)

                    # Transform Boxes
                    for box_row in boxes:
                        box = box_row[:4]
                        label = box_row[4]

                        if i == 0:
                            new_box = box
                        elif (
                            box[0] == 0 and box[1] == 0 and box[2] == 0 and box[3] == 0
                        ):
                            # Negative sample (no box) - keep as is
                            new_box = box
                        else:
                            # rotate_box handles cumulative 90 deg turns
                            new_box = rotate_box(box, original_w, original_h, i)

                        # Append to list
                        augmented_rows.append(
                            {
                                "image_path": save_path,
                                "xmin": new_box[0],
                                "ymin": new_box[1],
                                "xmax": new_box[2],
                                "ymax": new_box[3],
                                "label": label,
                                "shadow_angle": current_angle,
                                "shadow_x": sx,
                                "shadow_y": sy,
                            }
                        )

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Save new CSV
    new_df = pd.DataFrame(augmented_rows)
    new_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Created {OUTPUT_CSV} with {len(new_df)} rows (4x augmentation).")


if __name__ == "__main__":
    augment_data()
