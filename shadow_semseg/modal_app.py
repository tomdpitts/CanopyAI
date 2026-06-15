"""modal_app.py — 2048-res DeepLabV3+ semantic training on Modal (A100/H100 80GB).

Native 2048 training: full resolution AND whole-tile global context in training,
matching the 2048 holdout test exactly — the one thing 64 GB MPS can't do.

ONE-TIME SETUP
  pip install modal && modal token new
  # upload the phase22 backbone-init checkpoint to the checkpoint volume:
  modal volume put canopyai-deepforest-checkpoints \
      "checkpoints/phase22_B_L4/deepforest-epoch=15-map=0.54.ckpt" phase22/phase22.ckpt

RUN (one seed)
  modal run shadow_semseg/modal_app.py --epochs 30 --batch-size 2 --seed 0

FETCH the trained model for local eval
  modal volume get canopyai-deepforest-checkpoints \
      shadow_semseg/semseg_2048_s0/best.pt saved/2048_s0/best.pt
  # then: ensemble_eval.py / eval.py locally (MPS) as usual

NOTES
  - restor/tcd is ALREADY cached on /data (hf_cache); the run is forced HF-offline so it
    loads from the warm arrow cache and never re-downloads.
  - Auto-resumes from last.pt on the volume if a run is interrupted (no --fresh).
  - If 2048/bs2 OOMs even on 80 GB, drop to --batch-size 1 (BN gets noisy) or we add
    gradient checkpointing; A100-80GB / H100 should hold bs2.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

app = modal.App("shadow-semseg-2048")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")          # opencv runtime libs
    .pip_install(
        "torch", "torchvision", "transformers", "datasets", "torchmetrics",
        "opencv-python-headless", "pycocotools", "omegaconf", "pillow",
        "numpy", "safetensors",
    )
    .add_local_dir(str(HERE), remote_path="/root/shadow_semseg",
                   ignore=["__pycache__", "runs", "saved", "*.pt", ".DS_Store"])
)

data_volume = modal.Volume.from_name("canopyai-deepforest-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("canopyai-deepforest-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    timeout=86400,
)
def train_2048(epochs: int = 50, crop: int = 2048, batch_size: int = 2, seed: int = 0,
               scale_min: float = 0.75, scale_max: float = 1.25, loss: str = "ce_lovasz",
               grad_accum: int = 4, label_smoothing: float = 0.05, aspp_rates: str = "18,36,54"):
    import os
    import subprocess
    # TCD is ALREADY cached on the volume (datasets/restor___tcd + hub/datasets--restor--tcd,
    # all 6 train shards + test arrow). Point HF at it AND force offline so we never re-fetch
    # from the hub — load straight from the warm arrow cache.
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.chdir("/root/shadow_semseg")
    name = f"semseg_2048_s{seed}"
    cmd = [
        "python", "train.py", "--name", name, "--v2", "--no-shadow",
        "--loss", loss, "--crop", str(crop), "--batch-size", str(batch_size),
        "--epochs", str(epochs), "--seed", str(seed),
        "--scale-min", str(scale_min), "--scale-max", str(scale_max),
        # 2048 is small-batch -> freeze BN (stable stats) + grad-accum (effective batch)
        "--freeze-bn", "--grad-accum", str(grad_accum),
        # loaded recipe: deep-supervision aux head + label smoothing (noisy GT) + wide ASPP
        "--aux-loss", "--label-smoothing", str(label_smoothing), "--aspp-rates", aspp_rates,
        "--phase22-ckpt", "/checkpoints/phase22/phase22.ckpt",
        "--out-dir", "/checkpoints/shadow_semseg",
        # no --fresh: auto-resume from last.pt on the volume if interrupted
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    checkpoint_volume.commit()
    data_volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    timeout=86400,
)
def train_canopy_region(epochs: int = 30, crop: int = 512, batch_size: int = 8, seed: int = 0,
                        scale_min: float = 0.5, scale_max: float = 1.25, loss: str = "ce_lovasz",
                        grad_accum: int = 1, label_smoothing: float = 0.05,
                        limit_train: int = 0, warm_start: bool = True, fresh: bool = False):
    """Canopy-REGION (cat-1) segmenter: predicts the closed-canopy class, CC -> canopy
    instances downstream. Same DeepLabV3+/phase22 machinery, retargeted labels
    (--target canopy_region). Warm-starts from the v3 cover model if present."""
    import os
    import subprocess
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.chdir("/root/shadow_semseg")
    name = f"canopyreg_s{seed}" + ("_smoke" if limit_train else "")
    cmd = [
        "python", "train.py", "--name", name, "--v2", "--no-shadow",
        "--target", "canopy_region",
        "--loss", loss, "--crop", str(crop), "--batch-size", str(batch_size),
        "--epochs", str(epochs), "--seed", str(seed),
        "--scale-min", str(scale_min), "--scale-max", str(scale_max),
        "--grad-accum", str(grad_accum),
        "--aux-loss", "--label-smoothing", str(label_smoothing),
        "--phase22-ckpt", "/checkpoints/phase22/phase22.ckpt",
        "--out-dir", "/checkpoints/shadow_semseg",
    ]
    if limit_train:
        cmd += ["--limit-train", str(limit_train), "--limit-eval", "48"]
    if fresh or limit_train:
        cmd += ["--fresh"]
    init = "/checkpoints/shadow_semseg_init/v3.pt"
    if warm_start and os.path.exists(init):
        cmd += ["--init-ckpt", init]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    checkpoint_volume.commit()
    data_volume.commit()


@app.local_entrypoint()
def main(epochs: int = 50, crop: int = 2048, batch_size: int = 2, seed: int = 0):
    train_2048.remote(epochs=epochs, crop=crop, batch_size=batch_size, seed=seed)


@app.local_entrypoint()
def canopy(epochs: int = 30, crop: int = 512, batch_size: int = 8, seed: int = 0,
           limit_train: int = 0, warm_start: bool = True):
    train_canopy_region.remote(epochs=epochs, crop=crop, batch_size=batch_size, seed=seed,
                               limit_train=limit_train, warm_start=warm_start)
