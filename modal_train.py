"""
modal_train.py — Modal wrapper for CanopyAI training.

This script defines the Modal infrastructure (GPU, container, volumes) and
runs the training logic defined in `train.py`.

Usage:
    modal run modal_train.py --preset fast --weights baseline
    modal run modal_train.py::list_checkpoints
"""

import modal
import os
import sys

app = modal.App("canopyai-training")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim()
    # System dependencies for OpenCV and GDAL
    .apt_install("libgl1", "libglib2.0-0", "git", "gdal-bin", "libgdal-dev")
    # Python dependencies from requirements.txt
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "rasterio",
        "geopandas",
        "shapely",
        "fiona",
        "pyproj",
        "rtree",
        "affine",
        "opencv-python",
        "wget",
        "requests",
        "matplotlib",
        "cython",
        "pycocotools",
        "duckdb",
        "pyarrow",
        "scikit-image==0.20.0",
        "tqdm",
        "datasets",
        "Pillow==9.5.0",
    )
    # Detectron2 installation. Standard pip install is attempted first.
    # Alternative: use run_commands for specific flags if needed.
    # Standard pip install often works in Modal's clean env)
    .pip_install(
        "git+https://github.com/facebookresearch/detectron2.git@v0.6",
        "git+https://github.com/PatBall1/detectree2.git",
    )
    .add_local_dir(
        ".",
        remote_path="/root/canopyAI",
        ignore=[
            "data/",
            "venv*/",
            "*.pth",
            "*.tif",
            "__pycache__",
            ".git",
            ".DS_Store",
        ],
    )
)

# Create volume for checkpoints
volume = modal.Volume.from_name("canopyai-checkpoints", create_if_missing=True)

# Create volume for training data (persistent across runs)
data_volume = modal.Volume.from_name("canopyai-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/checkpoints": volume,
        "/data": data_volume,  # Persistent data storage
    },
    timeout=86400,  # 24 hours
)
def run_training(args):
    """Remote function that imports and runs the training logic."""
    # Change to the project directory
    os.chdir("/root/canopyAI")

    # Add to Python path so imports work
    sys.path.insert(0, "/root/canopyAI")

    # Import and run training
    # Import in function scope to ensure execution in remote container
    from train import main_worker

    print(f"🚀 Starting training on Modal with args: {vars(args)}")

    # Output directory creation is handled by train.py

    main_worker(0, args)


@app.local_entrypoint()
def main(
    preset: str = "fast",
    weights: str = "baseline",
    max_images: int = 3,
    already_downloaded: bool = False,
    train_split: float = 0.8,
):
    """
    Run training on Modal.

    Args:
        preset: 'tiny', 'fast', or 'full'
        weights: 'baseline', 'finetuned', or path
        max_images: Number of images to download/use
        already_downloaded: Skip download step
        train_split: Train/test split ratio (default 0.8)
    """

    # Simple class to mimic argparse Namespace
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Args(
        preset=preset,
        weights=weights,
        max_images=max_images,
        already_downloaded=already_downloaded,
        train_split=train_split,
    )

    print(f"☁️  Submitting training job to Modal...")
    run_training.remote(args)


@app.function(volumes={"/checkpoints": volume})
def list_checkpoints():
    """List files in the checkpoint volume."""
    import os

    print("📂 Listing checkpoints in volume '/checkpoints':")
    found = False
    for root, dirs, files in os.walk("/checkpoints"):
        for file in files:
            print(f" - {os.path.join(root, file)}")
            found = True

    if not found:
        print("   (Volume is empty)")

    print("\nTo download a file:")
    print("modal volume get canopyai-checkpoints <remote_path> <local_path>")
