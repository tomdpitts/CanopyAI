"""
Modal deployment for DeepForest fine-tuning

Usage:
    # Prepare data first (locally)
    python prepare_deepforest_data.py --data_dir data/tcd/raw --output annotations.csv --split
    
    # Upload to Modal volume
    modal volume put canopyai-deepforest-data annotations_train.csv /annotations_train.csv
    modal volume put canopyai-deepforest-data annotations_val.csv /annotations_val.csv
    
    # Run training on Modal (from deepforest_custom/ directory)
    cd deepforest_custom && modal run modal_deepforest.py \
        --epochs 20 \
        --batch-size 16 \
        --lr 0.001 \
        --patience 5 \
        --run-name "my_experiment_v1"
    
    # With Weights & Biases
    cd deepforest_custom && modal run modal_deepforest.py \
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
        "pytorch-lightning>=2.1.0,<3.0.0",  # Updated for DeepForest 2.0.0 compatibility
        "rasterio",
        "geopandas",
        "shapely",
        "opencv-python",
        "pandas",
        "numpy",
        "pillow",
        "pycocotools",
        "albumentations>=2.0.0",  # Updated for DeepForest 2.0.0 compatibility
        "wandb",
        "deepforest==2.0.0",  # Upgraded to match local env and get load_model() API
    )
    # Mount specific files and directories instead of the whole root
    # This prevents accidental upload of large data folders
    .add_local_file("../pyproject.toml", remote_path="/root/canopyAI/pyproject.toml")
    .add_local_dir(
        ".",
        remote_path="/root/canopyAI/deepforest_custom",
        ignore=["__pycache__", "lightning_logs/", "wandb/", "data/", "checkpoints/"],
    )
    .add_local_dir("../configs", remote_path="/root/canopyAI/configs")
    .add_local_file("../utils.py", remote_path="/root/canopyAI/utils.py")
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
    dataset="tcd",
    shadow_conditioning=False,  # Enable FiLM conditioning with constant 215° shadow
    checkpoint=None,  # Optional checkpoint path
):
    """
    Train DeepForest on Modal. Auto-resumes from checkpoint if exists.

    Args:
        shadow_conditioning: If True, use FiLM conditioning with constant 215° shadow + rotation augmentation
        checkpoint: Optional path to checkpoint file in Modal volume
    """

    os.chdir("/root/canopyAI")
    sys.path.insert(0, "/root/canopyAI")
    sys.path.insert(0, "/root/canopyAI/deepforest_custom")

    # Import our custom training script (not the deepforest package)
    import train_deepforest
    import pandas as pd
    import tarfile

    print("🚀 Starting DeepForest training on Modal")
    print(f"   GPU: A10G")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")

    # Extract image tarballs from volume if they exist
    image_archives = [
        "/data/tcd_images.tar.gz",
        "/data/won_images.tar.gz",
    ]

    print("\n📦 Extracting images from volume...")
    os.makedirs("/data/images", exist_ok=True)

    for archive in image_archives:
        if os.path.exists(archive):
            dataset_name = os.path.basename(archive).replace("_images.tar.gz", "")
            print(f"   Extracting {dataset_name}...")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall("/data/images")
            print(f"   ✅ {dataset_name} extracted")
        else:
            print(f"   ⚠️  {os.path.basename(archive)} not found, skipping")

    print("   All available images extracted to /data/images/")

    # Select dataset based on flag
    print(f"\n📊 Dataset: {dataset}")

    if dataset == "tcd":
        train_csv = "/data/tcd_train.csv"
        val_csv = "/data/tcd_val.csv"
    elif dataset == "won":
        train_csv = "/data/won_train.csv"
        val_csv = "/data/won_val.csv"
    elif dataset == "both":
        # Combine TCD and WON datasets
        print("   Combining TCD and WON datasets...")
        import pandas as pd

        tcd_train = pd.read_csv("/data/tcd_train.csv")
        won_train = pd.read_csv("/data/won_train.csv")
        combined_train = pd.concat([tcd_train, won_train], ignore_index=True)
        train_csv = "/tmp/combined_train.csv"
        combined_train.to_csv(train_csv, index=False)

        tcd_val = pd.read_csv("/data/tcd_val.csv")
        won_val = pd.read_csv("/data/won_val.csv")
        combined_val = pd.concat([tcd_val, won_val], ignore_index=True)
        val_csv = "/tmp/combined_val.csv"
        combined_val.to_csv(val_csv, index=False)

        print(f"   ✅ Combined: {len(combined_train)} train + {len(combined_val)} val")
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Must be 'tcd', 'won', or 'both'")

    # Check if CSVs exist
    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"Training CSV not found: {train_csv}\n"
            f"Upload with: modal volume put canopyai-deepforest-data <local-file> {train_csv}"
        )

    if not os.path.exists(val_csv):
        print(f"   ⚠️  Validation CSV not found: {val_csv}")
        print("   Training without validation")
        val_csv = None

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

    # Train (auto-resumes from checkpoint if exists)
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
        shadow_conditioning=shadow_conditioning,  # NEW
        checkpoint=checkpoint,  # NEW
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
    dataset: str = "tcd",
    shadow_conditioning: bool = False,
    checkpoint: str = None,
):
    """
    Launch DeepForest training on Modal.

    Automatically resumes from latest checkpoint if found in run_name directory.

    Args:
        dataset: Which dataset to train on ('tcd', 'won', or 'both')
        shadow_conditioning: Enable FiLM conditioning with constant 215° shadow + rotation augmentation
        checkpoint: Path to checkpoint file in Modal volume (e.g., 'models/model_oscar50.pth')
    """

    print("☁️  Submitting DeepForest training job to Modal...")

    results = train_deepforest_modal.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        wandb_project=wandb_project,
        run_name=run_name,
        dataset=dataset,
        shadow_conditioning=shadow_conditioning,  # NEW
        checkpoint=checkpoint,  # NEW
    )

    if results:
        print("\n📊 Final validation results:")
        for k, v in results.items():
            if isinstance(v, (int, float)):
                print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")


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
