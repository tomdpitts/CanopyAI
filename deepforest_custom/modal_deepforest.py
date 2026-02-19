"""
Modal deployment for DeepForest fine-tuning
============================================

## Phase 3 training (new, direct-CSV mode)

Step 1 — Upload images (phase3 data goes under /phase3/ to avoid touching existing won/tcd data)

    cd deepforest_custom
    tar -czf /tmp/phase3_images.tar.gz won/raw won/raw_test annotation_tiles
    modal volume put canopyai-deepforest-data /tmp/phase3_images.tar.gz /phase3/phase3_images.tar.gz
    modal volume put canopyai-deepforest-data train_phase3_augmented.csv /phase3/train_phase3_augmented.csv
    modal volume put canopyai-deepforest-data val_phase3_combined.csv /phase3/val_phase3_combined.csv

Step 2 — Launch baseline training (no shadow conditioning)

    cd deepforest_custom && modal run modal_deepforest.py \\
        --train-csv /data/phase3/train_phase3_augmented.csv \\
        --val-csv /data/phase3/val_phase3_combined.csv \\
        --run-name phase3_runQ_baseline_no_shadow \\
        --epochs 50 --patience 15 --lr 0.001 --batch-size 16

Step 3 — Launch FiLM shadow-conditioned training

    cd deepforest_custom && modal run modal_deepforest.py \\
        --train-csv /phase3/train_phase3_augmented.csv \\
        --val-csv /phase3/val_phase3_combined.csv \\
        --run-name phase3_runQ_film_shadow \\
        --epochs 50 --patience 15 --lr 0.001 --film-lr 1e-4 --batch-size 16 \\
        --shadow-conditioning

Step 4 — Download checkpoint

    modal volume get canopyai-deepforest-checkpoints \\
        /checkpoints/phase3_runQ_baseline_no_shadow/deepforest_final.pth \\
        ./phase3_runQ_baseline_no_shadow.pth

## Legacy dataset mode (tcd / won / both) — unchanged, existing data untouched

    cd deepforest_custom && modal run modal_deepforest.py \\
        --dataset tcd --epochs 20 --run-name tcd_v1

## Utility commands

    modal run modal_deepforest.py::list_checkpoints
    modal run modal_deepforest.py::list_data
"""

import modal
import os
import sys

app = modal.App("canopyai-deepforest-training")

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0", "git", "gdal-bin", "libgdal-dev")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "pytorch-lightning>=2.1.0,<3.0.0",
        "rasterio",
        "geopandas",
        "shapely",
        "opencv-python",
        "pandas",
        "numpy",
        "pillow",
        "pycocotools",
        "albumentations>=2.0.0",
        "wandb",
        "deepforest==2.0.0",
    )
    # Mount specific files and directories instead of the whole root.
    # Large data folders are intentionally excluded — data lives on Modal volumes.
    .add_local_file("../pyproject.toml", remote_path="/root/canopyAI/pyproject.toml")
    .add_local_dir(
        ".",
        remote_path="/root/canopyAI/deepforest_custom",
        ignore=["__pycache__", "lightning_logs/", "wandb/", "data/", "checkpoints/", "won/", "annotation_tiles/"],
    )
    .add_local_dir("../configs", remote_path="/root/canopyAI/configs")
    .add_local_file("../utils.py", remote_path="/root/canopyAI/utils.py")
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
checkpoint_volume = modal.Volume.from_name(
    "canopyai-deepforest-checkpoints", create_if_missing=True
)
data_volume = modal.Volume.from_name("canopyai-deepforest-data", create_if_missing=True)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100",       # A100-40GB; swap to "A10G" if quota / cost is a concern
    volumes={
        "/checkpoints": checkpoint_volume,
        "/data": data_volume,
    },
    timeout=86400,    # 24 hours
)
def train_deepforest_modal(
    # ── Direct-CSV mode (phase3 and future runs) ──────────────────────────
    train_csv: str = None,   # Absolute path inside /data volume, e.g. /phase3/train_phase3_augmented.csv
    val_csv: str = None,     # Absolute path inside /data volume, e.g. /phase3/val_phase3_combined.csv
    # ── Legacy dataset mode ────────────────────────────────────────────────
    dataset: str = None,     # "tcd" | "won" | "both" — used when train_csv is not set
    # ── Training hyper-parameters ──────────────────────────────────────────
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.001,
    patience: int = 5,
    run_name: str = "default",
    # ── FiLM / shadow conditioning ─────────────────────────────────────────
    shadow_conditioning: bool = False,  # Enable FiLM conditioning
    film_lr: float = 1e-4,             # LR for FiLM blocks (lower than backbone)
    shadow_angle_deg: float = None,    # Auto-derived from CSV if None
    # ── Misc ───────────────────────────────────────────────────────────────
    wandb_project: str = None,
    checkpoint: str = None,            # Optional path to initial weights in /data volume
    dry_run: bool = False,             # If True, print config and exit without training
):
    """
    Train DeepForest on Modal GPU. Auto-resumes from checkpoint if one exists
    in the run_name subdirectory of /checkpoints.

    Two modes:
      • Direct-CSV: pass train_csv / val_csv as absolute /data paths (phase3+)
      • Legacy:     pass dataset="tcd"|"won"|"both" to use pre-uploaded tarballs
    """
    os.chdir("/root/canopyAI")
    sys.path.insert(0, "/root/canopyAI")
    sys.path.insert(0, "/root/canopyAI/deepforest_custom")

    import tarfile
    import train_deepforest as trainer

    gpu_type = os.environ.get("MODAL_GPU_TYPE", "A100")
    print("🚀 DeepForest training on Modal")
    print(f"   GPU           : {gpu_type}")
    print(f"   Epochs        : {epochs}")
    print(f"   Batch size    : {batch_size}")
    print(f"   LR            : {lr}")
    print(f"   Film LR       : {film_lr}")
    print(f"   Shadow cond.  : {shadow_conditioning}")
    print(f"   Run name      : {run_name}")

    # ------------------------------------------------------------------
    # Resolve CSVs (direct mode vs legacy dataset mode)
    # ------------------------------------------------------------------
    if train_csv is not None:
        # ── Direct-CSV mode ──────────────────────────────────────────────
        print(f"\n📄 Direct-CSV mode:")
        print(f"   train_csv : {train_csv}")
        print(f"   val_csv   : {val_csv}")

        # Verify the CSV exists
        if not os.path.exists(train_csv):
            raise FileNotFoundError(
                f"Training CSV not found at {train_csv}\n"
                "Upload with:\n"
                "  modal volume put canopyai-deepforest-data <local>.csv <remote path>"
            )

        if val_csv and not os.path.exists(val_csv):
            print(f"   ⚠️  Validation CSV not found: {val_csv} — training without validation")
            val_csv = None

        # Extract images: look for ANY tar.gz under /data matching common patterns.
        # Phase3 images are stored under /phase3/phase3_images.tar.gz (or any *.tar.gz there).
        print("\n📦 Extracting image archives from volume...")
        os.makedirs("/data/images", exist_ok=True)

        # Discover all tar.gz files under /data (includes /phase3/ and legacy root)
        import glob
        archives = sorted(glob.glob("/data/**/*.tar.gz", recursive=True))
        if not archives:
            archives = sorted(glob.glob("/data/*.tar.gz"))
        for archive in archives:
            marker_file = archive + ".extracted"
            if os.path.exists(marker_file):
                print(f"   Skipping {archive} (already extracted)")
                continue

            print(f"   Extracting {archive}...")
            try:
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall("/data/images")
                # Create marker file
                with open(marker_file, "w") as f:
                    f.write("extracted")
                print(f"   ✅ Done")
            except Exception as e:
                print(f"   ⚠️  Failed to extract {archive}: {e}")

        # Rewrite absolute local paths in the CSVs to /data/images/...
        # Local prefix (the machine root) is stripped; the remainder becomes the
        # sub-path under /data/images.
        # e.g.  /Users/tompitts/dphil/CanopyAI/deepforest_custom/won/raw/x.tif
        #   ->  /data/images/won/raw/x.tif
        import pandas as pd
        _local_prefix = "/Users/tompitts/dphil/CanopyAI/deepforest_custom/"
        _modal_prefix = "/data/images/"

        def _rewrite_csv(csv_path):
            if not os.path.exists(csv_path):
                return
            df = pd.read_csv(csv_path)
            if "image_path" in df.columns:
                df["image_path"] = df["image_path"].str.replace(
                    _local_prefix, _modal_prefix, regex=False
                )
                df.to_csv(csv_path, index=False)
                print(f"   ✅ Rewrote paths in {csv_path}")

        print("\n🔧 Rewriting image paths in CSVs...")
        _rewrite_csv(train_csv)
        if val_csv:
            _rewrite_csv(val_csv)

    else:
        # ── Legacy dataset mode ───────────────────────────────────────────
        dataset = dataset or "tcd"
        print(f"\n📊 Legacy dataset mode: {dataset}")

        os.makedirs("/data/images", exist_ok=True)

        # Extract legacy tarballs
        image_archives = [
            "/data/tcd_images.tar.gz",
            "/data/won_images.tar.gz",
        ]
        for archive in image_archives:
            if os.path.exists(archive):
                dataset_name = os.path.basename(archive).replace("_images.tar.gz", "")
                print(f"   Extracting {dataset_name}...")
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall("/data/images")
                print(f"   ✅ {dataset_name} extracted")
            else:
                print(f"   ⚠️  {os.path.basename(archive)} not found, skipping")

        import pandas as pd

        if dataset == "tcd":
            train_csv = "/data/tcd_train.csv"
            val_csv = "/data/tcd_val.csv"
        elif dataset == "won":
            train_csv = "/data/won_train.csv"
            val_csv = "/data/won_val.csv"
        elif dataset == "both":
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

        if not os.path.exists(train_csv):
            raise FileNotFoundError(
                f"Training CSV not found: {train_csv}\n"
                f"Upload with: modal volume put canopyai-deepforest-data <local> {train_csv}"
            )

        if val_csv and not os.path.exists(val_csv):
            print(f"   ⚠️  Validation CSV not found: {val_csv}")
            val_csv = None

        # Fix paths from old local prefix
        for csv_path in filter(None, [train_csv, val_csv]):
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["image_path"] = df["image_path"].apply(
                    lambda x: x.replace(
                        "/Users/tompitts/dphil/canopyAI/", "/data/images/"
                    )
                )
                df.to_csv(csv_path, index=False)
                print(f"   ✅ Fixed paths in {csv_path}")

    # ------------------------------------------------------------------
    # Dry-run exit
    # ------------------------------------------------------------------
    if dry_run:
        print("\n🧪 DRY RUN — exiting before training")
        print(f"   train_csv : {train_csv}")
        print(f"   val_csv   : {val_csv}")
        return None

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model, results = trainer.train_deepforest(
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
        shadow_conditioning=shadow_conditioning,
        film_lr=film_lr,
        shadow_angle_deg=shadow_angle_deg,
        checkpoint=checkpoint,
        accelerator="gpu",   # Force CUDA — we always have an NVIDIA GPU on Modal
    )

    # Persist checkpoints
    checkpoint_volume.commit()

    print("\n✅ Training complete! Checkpoints saved to Modal volume.")
    print("\nTo download:")
    print(
        f"  modal volume get canopyai-deepforest-checkpoints "
        f"/{run_name}/deepforest_final.pth ./{run_name}.pth"
    )

    return results


# ---------------------------------------------------------------------------
# Local entrypoint (CLI)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    # ── Direct-CSV mode ────────────────────────────────────────────────────
    train_csv: str = None,
    val_csv: str = None,
    # ── Legacy dataset mode ────────────────────────────────────────────────
    dataset: str = None,
    # ── Training hypers ────────────────────────────────────────────────────
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.001,
    patience: int = 5,
    run_name: str = "default",
    # ── FiLM / shadow conditioning ─────────────────────────────────────────
    shadow_conditioning: bool = False,
    film_lr: float = 1e-4,
    shadow_angle_deg: float = None,
    # ── Misc ───────────────────────────────────────────────────────────────
    wandb_project: str = None,
    checkpoint: str = None,
    dry_run: bool = False,
):
    """
    Launch DeepForest training on Modal GPU.

    Phase3 baseline example:
        modal run modal_deepforest.py \\
            --train-csv /phase3/train_phase3_augmented.csv \\
            --val-csv /phase3/val_phase3_combined.csv \\
            --run-name phase3_runQ_baseline_no_shadow \\
            --epochs 50 --patience 15 --lr 0.001 --batch-size 16

    Phase3 FiLM example:
        modal run modal_deepforest.py \\
            --train-csv /phase3/train_phase3_augmented.csv \\
            --val-csv /phase3/val_phase3_combined.csv \\
            --run-name phase3_runQ_film_shadow \\
            --epochs 50 --patience 15 --lr 0.001 --film-lr 1e-4 --batch-size 16 \\
            --shadow-conditioning

    Legacy mode:
        modal run modal_deepforest.py --dataset tcd --epochs 20 --run-name tcd_v1
    """
    print("☁️  Submitting DeepForest training job to Modal...")

    results = train_deepforest_modal.remote(
        train_csv=train_csv,
        val_csv=val_csv,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        run_name=run_name,
        shadow_conditioning=shadow_conditioning,
        film_lr=film_lr,
        shadow_angle_deg=shadow_angle_deg,
        wandb_project=wandb_project,
        checkpoint=checkpoint,
        dry_run=dry_run,
    )

    if results:
        print("\n📊 Final validation results:")
        for k, v in results.items():
            if isinstance(v, (int, float)):
                print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
@app.function(volumes={"/checkpoints": checkpoint_volume})
def list_checkpoints():
    """List saved checkpoints in the volume."""
    import os

    print("📂 DeepForest checkpoints:")
    for root, dirs, files in os.walk("/checkpoints"):
        for file in files:
            path = os.path.join(root, file)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {path} ({size_mb:.1f} MB)")

    print("\nTo download:")
    print("  modal volume get canopyai-deepforest-checkpoints <remote_path> <local_path>")


@app.function(volumes={"/data": data_volume})
def list_data():
    """List all files in the data volume (shows both legacy and phase3 data)."""
    import os

    print("📂 Training data volume contents:")
    for root, dirs, files in os.walk("/data"):
        # Skip the extracted images directory (too many files)
        dirs[:] = [d for d in dirs if d != "images"]
        for file in files:
            path = os.path.join(root, file)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {path} ({size_mb:.1f} MB)")

    if not any(os.walk("/data")):
        print("  (Volume is empty)")
        print("\nUpload data with:")
        print("  modal volume put canopyai-deepforest-data <local_file> <remote_path>")
