"""
Modal deployment for DeepForest fine-tuning

Usage:
    # Prepare data first (locally)
    python prepare_deepforest_data.py --data_dir data/tcd/raw --output annotations.csv --split
    
    # Upload to Modal volume
    modal volume put canopyai-deepforest-data annotations_train.csv /annotations_train.csv
    modal volume put canopyai-deepforest-data annotations_val.csv /annotations_val.csv
    
    # Run training on Modal (from deepforest/ directory)
    cd deepforest && modal run modal_deepforest.py \
        --epochs 20 \
        --batch-size 16 \
        --lr 0.001 \
        --patience 5 \
        --run-name "my_experiment_v1"
    
    # With Weights & Biases
    cd deepforest && modal run modal_deepforest.py \
        --epochs 20 \
        --wandb-project canopyai-deepforest \
        --run-name "tcd_finetune_v1"
    
    # Download checkpoints
    modal volume get canopyai-deepforest-checkpoints /checkpoints/deepforest_final.pth ./deepforest_finetuned.pth
"""

import modal
import os
import sys

app = modal.App("canopyai-deepforest-training")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0", "git", "gdal-bin", "libgdal-dev")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "pytorch-lightning==1.9.5",  # Compatible with deepforest 1.3.x
        "rasterio",
        "geopandas",
        "shapely",
        "opencv-python",
        "pandas",
        "numpy",
        "pillow",
        "pycocotools",
        "albumentations==1.3.1",  # Pinned - newer versions removed functional module
        "wandb",
        "deepforest==1.3.1",  # Pinned version compatible with PL 1.9.5
    )
    .add_local_dir(
        "..",  # Parent directory (canopyAI)
        remote_path="/root/canopyAI",
        ignore=[
            "data/",
            "won003*/",
            "venv*/",
            "*.tif",
            "__pycache__",
            ".git",
            "*.pth",
            "foxtrot_output/",
            "sam_output/",
            "train_outputs/",
        ],
    )
)

# Create volumes
checkpoint_volume = modal.Volume.from_name(
    "canopyai-deepforest-checkpoints", create_if_missing=True
)
data_volume = modal.Volume.from_name("canopyai-deepforest-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/checkpoints": checkpoint_volume,
        "/data": data_volume,
    },
    timeout=86400,  # 24 hours
)
def train_deepforest_modal(
    epochs=20,
    batch_size=16,
    lr=0.001,
    patience=5,
    wandb_project=None,
    run_name=None,
):
    """Train DeepForest on Modal with TCD data."""

    os.chdir("/root/canopyAI")
    sys.path.insert(0, "/root/canopyAI")
    sys.path.insert(0, "/root/canopyAI/deepforest")

    # Import our custom training script (not the deepforest package)
    import train_deepforest
    import pandas as pd
    import tarfile

    print("🚀 Starting DeepForest training on Modal")
    print(f"   GPU: A10G")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")

    # Extract TCD images if they exist in the volume
    tcd_archive = "/data/tcd_images.tar.gz"
    if os.path.exists(tcd_archive):
        print("\n📦 Extracting TCD images from volume...")
        os.makedirs("/data/images", exist_ok=True)
        with tarfile.open(tcd_archive, "r:gz") as tar:
            tar.extractall("/data/images")
        print("   ✅ Images extracted to /data/images/")

    # Training and validation CSVs should be in data volume
    train_csv = "/data/annotations_train.csv"
    val_csv = "/data/annotations_val.csv"

    # Check files exist
    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"Training CSV not found: {train_csv}\n"
            "Upload with: modal volume put canopyai-deepforest-data annotations_train.csv /annotations_train.csv"
        )

    # Fix image paths in CSVs to point to extracted images
    print("\n🔧 Fixing image paths in CSVs...")
    for csv_path in [train_csv, val_csv]:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Update paths: /Users/.../data/tcd/raw/image.tif -> /data/images/data/tcd/raw/image.tif
            df["image_path"] = df["image_path"].apply(
                lambda x: x.replace("/Users/tompitts/dphil/canopyAI/", "/data/images/")
            )
            df.to_csv(csv_path, index=False)
            print(f"   ✅ Fixed paths in {csv_path}")

    if not os.path.exists(val_csv):
        print(f"⚠️  Validation CSV not found: {val_csv}")
        print("   Training without validation data")
        val_csv = None

    # Train
    model, results = train_deepforest.train_deepforest(
        train_csv=train_csv,
        val_csv=val_csv,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        output_dir="/checkpoints",
        pretrained=True,
        wandb_project=wandb_project,
        run_name=run_name,
    )

    # Commit checkpoint volume
    checkpoint_volume.commit()

    print("\n✅ Training complete! Checkpoints saved to Modal volume.")
    print("\nTo download:")
    print(
        "modal volume get canopyai-deepforest-checkpoints /checkpoints/deepforest_final.pth ./deepforest_finetuned.pth"
    )

    return results


@app.local_entrypoint()
def main(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.001,
    patience: int = 5,
    wandb_project: str = None,
    run_name: str = None,
):
    """Launch DeepForest training on Modal."""

    print("☁️  Submitting DeepForest training job to Modal...")

    results = train_deepforest_modal.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        wandb_project=wandb_project,
        run_name=run_name,
    )

    if results:
        print("\n📊 Final validation results:")
        for k, v in results.items():
            print(f"   {k}: {v:.4f}")


@app.function(volumes={"/checkpoints": checkpoint_volume})
def list_checkpoints():
    """List saved checkpoints."""
    import os

    print("📂 DeepForest checkpoints:")
    for root, dirs, files in os.walk("/checkpoints"):
        for file in files:
            path = os.path.join(root, file)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {path} ({size_mb:.1f} MB)")

    print("\nTo download:")
    print("modal volume get canopyai-deepforest-checkpoints <remote_path> <local_path>")


@app.function(volumes={"/data": data_volume})
def list_data():
    """List uploaded data files."""
    import os

    print("📂 Training data:")
    for root, dirs, files in os.walk("/data"):
        for file in files:
            path = os.path.join(root, file)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {path} ({size_mb:.1f} MB)")

    if not any(os.listdir("/data")):
        print("  (Volume is empty)")
        print("\nUpload data with:")
        print(
            "modal volume put canopyai-deepforest-data annotations_train.csv /annotations_train.csv"
        )
