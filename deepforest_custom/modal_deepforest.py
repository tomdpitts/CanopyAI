"""
Modal deployment for DeepForest fine-tuning
============================================

## Phase 8 — Shadow Cross-Attention Ablation Study

Phase 5 training data is already in Modal storage:
    phase5_train_aug.csv  (152 KiB)
    phase5_val_aug.csv    (38.5 KiB)
    phase5_images.tar.gz

### Run A — Baseline (no shadow)

    source venv310/bin/activate && cd deepforest_custom && modal run --detach modal_deepforest.py \\
        --train-csv /data/phase5_train_aug.csv \\
        --val-csv /data/phase5_val_aug.csv \\
        --run-name phase8_A_baseline \\
        --epochs 50 --patience 10 --lr 0.001 --batch-size 16

### Run B — Shadow 4th input channel

    source venv310/bin/activate && cd deepforest_custom && modal run --detach modal_deepforest.py \\
        --train-csv /data/phase5_train_aug.csv \\
        --val-csv /data/phase5_val_aug.csv \\
        --run-name phase8_B_shadow_channel \\
        --epochs 50 --patience 10 --lr 0.001 --batch-size 16 \\
        --shadow-channel

### Run C — Shadow cross-attention only (no 4th channel)

    source venv310/bin/activate && cd deepforest_custom && modal run --detach modal_deepforest.py \\
        --train-csv /data/phase5_train_aug.csv \\
        --val-csv /data/phase5_val_aug.csv \\
        --run-name phase8_C_shadow_crossattn \\
        --epochs 50 --patience 10 --lr 0.001 --batch-size 16 \\
        --shadow-cross-attention

### Run D — Both (channel + cross-attention)

    source venv310/bin/activate && cd deepforest_custom && modal run --detach modal_deepforest.py \\
        --train-csv /data/phase5_train_aug.csv \\
        --val-csv /data/phase5_val_aug.csv \\
        --run-name phase8_D_full \\
        --epochs 50 --patience 10 --lr 0.001 --batch-size 16 \\
        --shadow-channel --shadow-cross-attention

## Utility commands

    modal run modal_deepforest.py::list_checkpoints
    modal run modal_deepforest.py::list_data

## Download checkpoint after run

    modal volume get canopyai-deepforest-checkpoints \\
        /phase8_A_baseline/deepforest_final.pth \\
        ./phase8_A_baseline.pth
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
    .add_local_file("pyproject.toml", remote_path="/root/canopyAI/pyproject.toml")
    .add_local_dir(
        "deepforest_custom",
        remote_path="/root/canopyAI/deepforest_custom",
        ignore=["__pycache__", "lightning_logs/", "wandb/", "data/", "checkpoints/",
                "won/", "annotation_tiles/", "bru_tiles/", "phase5_tiles/",
                "won_visualizations/", "*.tif", "*.tiff", "*.png", "*.jpg"],
    )
    .add_local_dir("configs", remote_path="/root/canopyAI/configs")
    .add_local_file("utils.py", remote_path="/root/canopyAI/utils.py")
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
    # ── Direct-CSV mode ──────────────────────────────────────────────
    train_csv: str = None,
    val_csv: str = None,
    # ── Legacy dataset mode ───────────────────────────────────────
    dataset: str = None,
    # ── Training hyper-parameters ────────────────────────────────
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.001,
    patience: int = 5,
    run_name: str = "default",
    # ── Shadow mechanisms ───────────────────────────────────────
    shadow_channel: bool = False,         # Run B/D: 4th input channel
    shadow_cross_attention: bool = False,  # Run C/D: cross-attn after layer4
    shadow_angle_deg: float = None,
    # ── Misc ───────────────────────────────────────────────────
    wandb_project: str = None,
    checkpoint: str = None,
    dry_run: bool = False,
    freeze_backbone: bool = False,
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
    print(f"   GPU                  : {gpu_type}")
    print(f"   Epochs               : {epochs}")
    print(f"   Batch size           : {batch_size}")
    print(f"   LR                   : {lr}")
    print(f"   Shadow channel       : {shadow_channel}")
    print(f"   Shadow cross-attn    : {shadow_cross_attention}")
    print(f"   Run name             : {run_name}")

    # ------------------------------------------------------------------
    # Resolve CSVs (direct mode vs legacy dataset mode)
    # ------------------------------------------------------------------
    # Phase5 uses direct-CSV mode so the auto-glob tar extraction picks up
    # /phase5_images.tar.gz (legacy mode only knows won/tcd tarballs).
    if dataset == "phase5" and train_csv is None:
        train_csv = "/data/phase5_train_aug.csv"
        val_csv   = "/data/phase5_val_aug.csv"
        print("📄 Phase5 → direct-CSV mode (images extracted via glob)")

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

        # ------------------------------------------------------------------
        # DEBUG: Verify images exist and have correct dimensions
        # ------------------------------------------------------------------
        print("\n🕵️‍♀️ Verifying image paths and dimensions (first 5)...")
        import cv2
        for csv_path in filter(None, [train_csv, val_csv]):
            try:
                df = pd.read_csv(csv_path)
                for i, row in df.head(5).iterrows():
                    img_path = row["image_path"]
                    if not os.path.exists(img_path):
                        print(f"   ❌ Missing: {img_path}")
                    else:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"   ❌ Failed to load: {img_path}")
                        else:
                            h, w = img.shape[:2]
                            print(f"   ✅ Found: {img_path} ({w}x{h})")
                            
                            # Check bbox
                            xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
                            if xmax > w or ymax > h:
                                print(f"      ⚠️  BBox out of bounds: [{xmin}, {ymin}, {xmax}, {ymax}] vs {w}x{h}")
            except Exception as e:
                print(f"   ⚠️  Verification failed for {csv_path}: {e}")

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
            train_csv = "/data/won_train_pruned.csv"
            val_csv = "/data/won_val.csv"
        elif dataset == "phase5":
            train_csv = "/data/phase5_train_aug.csv"
            val_csv = "/data/phase5_val_aug.csv"
        elif dataset == "both":
            tcd_train = pd.read_csv("/data/tcd_train.csv")
            won_train = pd.read_csv("/data/won_train_pruned.csv")
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
        shadow_angle_deg=shadow_angle_deg,
        checkpoint=checkpoint,
        accelerator="gpu",
        freeze_backbone=freeze_backbone,
        shadow_channel=shadow_channel,
        shadow_cross_attention=shadow_cross_attention,
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
    # ── Direct-CSV mode ──────────────────────────────────────────────
    train_csv: str = None,
    val_csv: str = None,
    # ── Legacy dataset mode ───────────────────────────────────────
    dataset: str = None,
    # ── Training hypers ──────────────────────────────────────────
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.001,
    patience: int = 5,
    run_name: str = "default",
    # ── Shadow mechanisms ───────────────────────────────────────
    shadow_channel: bool = False,
    shadow_cross_attention: bool = False,
    shadow_angle_deg: float = None,
    # ── Misc ───────────────────────────────────────────────────
    wandb_project: str = None,
    checkpoint: str = None,
    dry_run: bool = False,
    freeze_backbone: bool = False,
):
    """
Modal deployment for DeepForest fine-tuning
============================================

## Phase 8 — Shadow Cross-Attention Ablation Study

Phase 5 training data is already in Modal storage:
    phase5_train_aug.csv  (152 KiB)
    phase5_val_aug.csv    (38.5 KiB)
    phase5_images.tar.gz

### Run A — Baseline (no shadow)

    cd deepforest_custom && modal run modal_deepforest.py \
        --train-csv /data/phase5_train_aug.csv \
        --val-csv /data/phase5_val_aug.csv \
        --run-name phase8_A_baseline \
        --epochs 50 --patience 10 --lr 0.001 --batch-size 16 \
        --checkpoint /checkpoints/model_oscar50.pth

### Run B — Shadow 4th input channel

    cd deepforest_custom && modal run modal_deepforest.py \
        --train-csv /data/phase5_train_aug.csv \
        --val-csv /data/phase5_val_aug.csv \
        --run-name phase8_B_shadow_channel \
        --epochs 50 --patience 10 --lr 0.001 --batch-size 16 \
        --shadow-channel \
        --checkpoint /checkpoints/model_oscar50.pth

### Run C — Shadow cross-attention only (no 4th channel)

    cd deepforest_custom && modal run modal_deepforest.py \
        --train-csv /data/phase5_train_aug.csv \
        --val-csv /data/phase5_val_aug.csv \
        --run-name phase8_C_shadow_crossattn \
        --epochs 50 --patience 10 --lr 0.001 --batch-size 16 \
        --shadow-cross-attention \
        --checkpoint /checkpoints/model_oscar50.pth

### Run D — Both (channel + cross-attention)

    cd deepforest_custom && modal run modal_deepforest.py \
        --train-csv /data/phase5_train_aug.csv \
        --val-csv /data/phase5_val_aug.csv \
        --run-name phase8_D_full \
        --epochs 50 --patience 10 --lr 0.001 --batch-size 16 \
        --shadow-channel --shadow-cross-attention \
        --checkpoint /checkpoints/model_oscar50.pth

## Utility commands

    modal run modal_deepforest.py::list_checkpoints
    modal run modal_deepforest.py::list_data

## Download checkpoint after run

    modal volume get canopyai-deepforest-checkpoints \
        /phase8_A_baseline/deepforest_final.pth \
        ./phase8_A_baseline.pth
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
        shadow_channel=shadow_channel,
        shadow_cross_attention=shadow_cross_attention,
        shadow_angle_deg=shadow_angle_deg,
        wandb_project=wandb_project,
        checkpoint=checkpoint,
        dry_run=dry_run,
        freeze_backbone=freeze_backbone,
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
