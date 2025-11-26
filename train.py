#!/usr/bin/env python3
"""
train.py — Fine-tune Detectree2 using tiny/fast/full YAML presets.

Usage:
    python train.py --preset tiny   --weights baseline --already_downloaded
    python train.py --preset fast   --weights baseline
    python train.py --preset full   --weights baseline

for finetuning from previous run:
    python train.py --weights finetuned
"""

import argparse
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp
from detectree2.models.train import register_train_data, MyTrainer
from detectree2.models.train import setup_cfg
from prepare_data import run_preparation
import wget


def is_running_on_modal():
    """Detect if running on Modal by checking environment variables."""
    return os.environ.get("MODAL_ENVIRONMENT") is not None


# ============================================================
#   Load preset YAML + override weights + device
# ============================================================


def load_preset_cfg(preset: str, weights: str, output_dir: Path):
    cfg = setup_cfg(update_model="230103_randresize_full.pth")

    # Preset selection
    preset_map = {
        "tiny": "configs/tiny_debug.yaml",
        "fast": "configs/fast_train.yaml",
        "full": "configs/full_train.yaml",
    }
    cfg.merge_from_file(preset_map[preset])
    cfg.IMGMODE = "rgb"

    # ---- Set model weights ----
    if weights == "baseline":
        # Determine weights path (use persistent storage on Modal)
        if is_running_on_modal():
            weights_path = Path("/checkpoints/230103_randresize_full.pth")
        else:
            weights_path = Path("230103_randresize_full.pth")

        # Download if missing
        if not weights_path.exists():
            url = "https://zenodo.org/records/10522461/files/230103_randresize_full.pth"
            print(f"📦 Downloading baseline weights to {weights_path}")
            wget.download(url, out=str(weights_path))
            print("\n✅ Model download complete.")
        else:
            print(f"✅ Using cached weights: {weights_path}")

        cfg.MODEL.WEIGHTS = str(weights_path)
    elif weights == "finetuned":
        cfg.MODEL.WEIGHTS = str(output_dir / "model_final.pth")
    else:
        cfg.MODEL.WEIGHTS = weights  # custom path

    # ---- Device selection ----
    # if torch.backends.mps.is_available():
    #     cfg.MODEL.DEVICE = "mps"
    #     print("💻 Using Apple MPS backend")
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        print("🚀 Using CUDA GPU")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("🧠 Using CPU")

    cfg.OUTPUT_DIR = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return cfg


# ============================================================
#   TRAINING
# ============================================================


def get_next_nato_codename(output_dir: Path) -> str:
    """Get next available NATO phonetic alphabet codename."""
    nato_alphabet = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "yankee",
        "zulu",
    ]

    existing_runs = [
        d.name for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]

    for codename in nato_alphabet:
        if f"run_{codename}" not in existing_runs:
            return codename

    # Fallback if all NATO names used
    import random

    return f"run_{random.randint(1000, 9999)}"


def find_recent_incomplete_run(output_dir: Path, time_window_minutes: int = 5) -> Path:
    """Find most recent incomplete run within time window (for preemption recovery)."""
    from datetime import datetime, timedelta

    if not output_dir.exists():
        return None

    cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
    incomplete_runs = []

    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue

        # Check if run is incomplete (no model_final.pth)
        if (run_dir / "model_final.pth").exists():
            continue

        # Check if created within time window
        created_time = datetime.fromtimestamp(run_dir.stat().st_ctime)
        if created_time > cutoff_time:
            incomplete_runs.append((created_time, run_dir))

    if incomplete_runs:
        # Return most recent
        incomplete_runs.sort(reverse=True)
        return incomplete_runs[0][1]

    return None


def train_detectree2(
    chips_root: Path, output_dir: Path, preset: str, weights: str, run_name: str = None
):
    print("📚 Registering training dataset…")

    site_name = "TCD"
    val_fold = 1
    register_train_data(str(chips_root), site_name, val_fold)

    train_name = f"{site_name}_train"
    val_name = f"{site_name}_val"

    # Detect Modal and override output directory if needed
    if is_running_on_modal():
        output_dir = Path("/checkpoints")
        print(f"☁️  Running on Modal: Saving checkpoints to {output_dir}")
    else:
        print(f"💾 Running locally: Saving checkpoints to {output_dir}")

    cfg = load_preset_cfg(preset, weights, output_dir)

    # Explicitly set correct datasets
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)

    # Enable COCO evaluation for early stopping
    # Set eval period to match checkpoint period to avoid too frequent evaluation
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD

    # Determine run directory
    if run_name:
        # Explicit run name provided
        run_dir = output_dir / f"run_{run_name}"
        print(f"📁 Using run name: {run_name}")
    else:
        # Auto-detect recent incomplete run (preemption recovery)
        recent_incomplete = find_recent_incomplete_run(
            output_dir, time_window_minutes=5
        )
        if recent_incomplete:
            run_dir = recent_incomplete
            print(f"🔄 Detected recent incomplete run: {run_dir.name}")
            print(f"   (created within last 5 minutes, likely preemption)")
        else:
            # Create new auto-codename
            codename = get_next_nato_codename(output_dir)
            run_dir = output_dir / f"run_{codename}"
            print(f"🆕 Creating new run: {codename}")

    cfg.OUTPUT_DIR = str(run_dir)

    # Freeze config
    cfg.freeze()

    # Clean up eval cache to avoid stale COCO annotations
    import shutil

    eval_dir = Path("eval")
    if eval_dir.exists():
        print(f"🧹 Cleaning eval cache: {eval_dir}")
        shutil.rmtree(eval_dir)

    print("🚀 Training starting…")
    print(f"✅ COCO evaluation enabled (every {cfg.TEST.EVAL_PERIOD} iters)")
    print(f"⏱️  Early stopping patience: 5 evaluations")

    # Check for existing checkpoints to resume from
    checkpoint_dir = Path(cfg.OUTPUT_DIR)
    last_checkpoint_file = checkpoint_dir / "last_checkpoint"
    resume_from_checkpoint = False

    if last_checkpoint_file.exists():
        with open(last_checkpoint_file, "r") as f:
            last_checkpoint = f.read().strip()
        checkpoint_path = checkpoint_dir / last_checkpoint
        if checkpoint_path.exists():
            print(f"🔄 Found existing checkpoint: {last_checkpoint}")
            print(
                f"   Resuming training from iteration {last_checkpoint.split('_')[-1].replace('.pth', '')}"
            )
            resume_from_checkpoint = True
        else:
            print(f"⚠️  Checkpoint file listed but not found: {checkpoint_path}")
            print(f"   Starting fresh training")
    else:
        print(f"📝 No existing checkpoints found, starting fresh training")

    trainer = MyTrainer(cfg, patience=5)
    trainer.resume_or_load(resume=resume_from_checkpoint)
    trainer.train()


# ============================================================
#   Worker run by launch()
# ============================================================


def main_worker(rank, args):
    # Detect Modal and use persistent volume paths
    if is_running_on_modal():
        data_root = Path("/data/tcd")
        output_dir = Path("/checkpoints")
        print(f"☁️  Running on Modal:")
        print(f"   Data: {data_root} (persistent volume)")
        print(f"   Checkpoints: {output_dir} (persistent volume)")
    else:
        data_root = Path("data/tcd")
        output_dir = data_root / "train_outputs"
        print(f"💾 Running locally:")
        print(f"   Data: {data_root}")
        print(f"   Checkpoints: {output_dir}")

    chips_root = data_root / "chips"

    # Create directories
    data_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download
    # 1. Prepare Data (Download + Tile + Split)
    if not args.already_downloaded:
        print("🏗️  Running data preparation pipeline...")
        run_preparation(args)
    else:
        # Nothing to download, nothing to tile
        print("⏭️ Skipping download and tiling (using existing chips)")

    # 2. Train
    train_detectree2(
        chips_root,
        output_dir,
        preset=args.preset,
        weights=args.weights,
        run_name=args.run_name,
    )


# ============================================================
#   CLI
# ============================================================


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--already_downloaded", action="store_true")
    p.add_argument("--max_images", type=int, default=3)
    p.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (rest for test)",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run (e.g., 'experiment-v1'). If not provided, auto-generates NATO codename.",
    )

    p.add_argument("--preset", default="tiny", choices=["tiny", "fast", "full"])

    p.add_argument(
        "--weights",
        default="baseline",
        help="'baseline' | 'finetuned' | /path/to/model.pth",
    )

    return p.parse_args()


# ============================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    # Determine number of available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1

    # Multi-GPU → spawn workers
    if num_gpus > 1:
        mp.spawn(main_worker, nprocs=num_gpus, args=(args,))
    else:
        # Single GPU or CPU/MPS
        main_worker(0, args)
