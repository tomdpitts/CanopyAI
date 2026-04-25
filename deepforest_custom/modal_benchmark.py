"""
Modal batch inference for TCD benchmark
========================================

Streams the full restor/tcd HuggingFace dataset (~4169 tiles), runs foxtrot.py
inference on each tile, and saves GeoJSON predictions to a Modal volume.

Volumes
-------
  canopyai-deepforest-checkpoints  → /checkpoints   (model .pth files)
  canopyai-benchmark-results       → /results        (output GeoJSONs per model)

Setup (one-time)
----------------
    # Upload shadow model if not already on checkpoints volume:
    modal volume put canopyai-deepforest-checkpoints \\
        solar/shadow_regression/output/shadow_model_combined_best.pth \\
        shadow_model_combined_best.pth

    # Download SAM checkpoint to checkpoints volume:
    modal run deepforest_custom/modal_benchmark.py::download_sam

Run inference
-------------
    # Weecology baseline
    source venv310/bin/activate
    modal run deepforest_custom/modal_benchmark.py \\
        --model weecology --name weecology

    # Phase21 baseline
    modal run deepforest_custom/modal_benchmark.py \\
        --model /checkpoints/phase21_baseline/deepforest_final.pth \\
        --name phase21_baseline

    # Phase21 B λ4
    modal run deepforest_custom/modal_benchmark.py \\
        --model /checkpoints/phase21_B_λ4/deepforest_final.pth \\
        --name phase21_B_λ4

Pull results back locally
-------------------------
    modal volume get canopyai-benchmark-results /weecology       benchmark_results/weecology
    modal volume get canopyai-benchmark-results /phase21_baseline benchmark_results/phase21_baseline
    modal volume get canopyai-benchmark-results /phase21_B_λ4    benchmark_results/phase21_B_λ4

Then evaluate locally (no GPU needed):
    python benchmark_tcd.py --skip-inference \\
        --models weecology phase21_baseline.pth phase21_B_λ4.pth \\
        --names  weecology phase21_baseline phase21_B_λ4
"""

import os
import sys
import json
import io
import tempfile
import subprocess
from pathlib import Path

import modal

app = modal.App("canopyai-benchmark")

# ---------------------------------------------------------------------------
# Container image — same base as training + SAM + HuggingFace datasets
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0", "git", "gdal-bin", "libgdal-dev", "wget")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "rasterio",
        "geopandas",
        "shapely",
        "opencv-python",
        "pandas",
        "numpy",
        "pillow",
        "pycocotools",
        "deepforest==2.0.0",
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        "datasets",
        "huggingface_hub",
        "tqdm",
    )
    .add_local_file("foxtrot.py",   remote_path="/root/canopyAI/foxtrot.py")
    .add_local_file("utils.py",     remote_path="/root/canopyAI/utils.py")
    .add_local_dir("solar",         remote_path="/root/canopyAI/solar",
                   ignore=["__pycache__", "*.png", "*.jpg"])
    .add_local_dir("deepforest_custom", remote_path="/root/canopyAI/deepforest_custom",
                   ignore=["__pycache__", "lightning_logs/", "wandb/", "data/",
                           "checkpoints/", "*.tif", "*.tiff", "*.png", "*.jpg"])
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
checkpoint_volume = modal.Volume.from_name(
    "canopyai-deepforest-checkpoints", create_if_missing=True
)
results_volume = modal.Volume.from_name(
    "canopyai-benchmark-results", create_if_missing=True
)

CHECKPOINTS_DIR = Path("/checkpoints")
RESULTS_DIR     = Path("/results")
SAM_CHECKPOINT  = CHECKPOINTS_DIR / "sam_vit_b_01ec64.pth"
SHADOW_MODEL    = CHECKPOINTS_DIR / "shadow_model_combined_best.pth"


# ---------------------------------------------------------------------------
# One-time: download SAM checkpoint to volume
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=600,
)
def download_sam():
    """Download SAM ViT-B checkpoint to the checkpoints volume."""
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    out = str(SAM_CHECKPOINT)
    if SAM_CHECKPOINT.exists():
        print(f"SAM checkpoint already exists at {out}")
        return
    print(f"Downloading SAM checkpoint → {out}")
    urllib.request.urlretrieve(url, out)
    checkpoint_volume.commit()
    print("Done.")


# ---------------------------------------------------------------------------
# Per-tile inference worker
# ---------------------------------------------------------------------------
def _save_tile_tif(item: dict, tmp_dir: str) -> tuple[str, dict]:
    """Save a HuggingFace TCD item to a GeoTIFF. Returns (tif_path, meta_dict)."""
    import numpy as np
    import cv2
    import rasterio
    from rasterio.transform import from_bounds
    from PIL import Image as PILImage

    img_bytes = item["image"]["bytes"]
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        pil_img = PILImage.open(io.BytesIO(img_bytes))
        img = np.array(pil_img)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    bounds = item["bounds"]
    crs    = item["crs"]
    transform = from_bounds(*bounds, width=w, height=h)

    # Use image_id as filename to avoid collisions across containers
    image_id = str(item.get("image_id", "tile"))
    tif_path = os.path.join(tmp_dir, f"tile_{image_id}.tif")

    with rasterio.open(tif_path, "w", driver="GTiff", height=h, width=w,
                       count=3, dtype=img.dtype, crs=crs, transform=transform) as dst:
        for b in range(3):
            dst.write(img[:, :, b], b + 1)

    meta = {
        "image_id":         item.get("image_id"),
        "bounds":           bounds,
        "crs":              str(crs),
        "width":            w,
        "height":           h,
        "coco_annotations": item.get("coco_annotations", []),
        "biome":            item.get("biome"),
        "biome_name":       item.get("biome_name"),
        "country":          item.get("country"),
    }
    return tif_path, meta


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/checkpoints": checkpoint_volume,
        "/results":     results_volume,
    },
    timeout=7200,
    memory=16384,
)
def infer_shard(
    parquet_url: str,
    model_spec: str,
    model_name: str,
    skip_existing: bool = True,
):
    """
    Process all tiles in one HuggingFace parquet shard.
    Downloads only that shard file — no streaming past unwanted rows.
    """
    os.chdir("/root/canopyAI")
    sys.path.insert(0, "/root/canopyAI")

    import pyarrow.parquet as pq
    import requests

    out_dir = RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {SAM_CHECKPOINT}. "
            "Run: modal run deepforest_custom/modal_benchmark.py::download_sam"
        )
    shadow_arg = ["--shadow_model", str(SHADOW_MODEL)] if SHADOW_MODEL.exists() else []
    if not SHADOW_MODEL.exists():
        print(f"⚠  Shadow model not found — running without shadow")

    model_arg = []
    if model_spec.lower() not in ("weecology", "weecology/deepforest", "default"):
        if not Path(model_spec).exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_spec}")
        model_arg = ["--deepforest_model", model_spec]

    # Download this shard directly — no scanning past other shards
    shard_name = parquet_url.split("/")[-1]
    local_parquet = f"/tmp/{shard_name}"
    print(f"Downloading shard {shard_name} ...")
    r = requests.get(parquet_url, stream=True)
    r.raise_for_status()
    with open(local_parquet, "wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
    print(f"  Downloaded {os.path.getsize(local_parquet) / 1e6:.1f} MB")

    table = pq.read_table(local_parquet)
    processed = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for row_idx in range(len(table)):
            row = {col: table[col][row_idx].as_py() for col in table.column_names}

            image_id = str(row.get("image_id", f"{shard_name}_{row_idx}"))
            out_path = out_dir / f"tile_{image_id}_canopyai.geojson"

            if skip_existing and out_path.exists():
                processed += 1
                continue

            try:
                tif_path, meta = _save_tile_tif(row, tmp_dir)
                meta_path = tif_path.replace(".tif", "_meta.json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f)

                cmd = [
                    sys.executable, "foxtrot.py",
                    "--image_path", tif_path,
                    "--output_dir", str(out_dir),
                    "--no_viz",
                    "--sam_checkpoint", str(SAM_CHECKPOINT),
                    *shadow_arg,
                    *model_arg,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"  ✗ tile {image_id}: {result.stderr[-200:]}")
                else:
                    stem = Path(tif_path).stem
                    produced = out_dir / f"{stem}_canopyai.geojson"
                    if produced.exists() and produced != out_path:
                        produced.rename(out_path)
                    results_volume.commit()
                    print(f"  ✓ tile {image_id}  [{row.get('biome_name', '?')}]")
            except Exception as e:
                print(f"  ✗ tile {image_id}: {e}")

            processed += 1

    return processed


# ---------------------------------------------------------------------------
# Orchestrator: one container per parquet shard
# ---------------------------------------------------------------------------
def _get_shard_urls() -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files("restor/tcd", repo_type="dataset")
    parquet_files = sorted(f for f in files if "train" in f and f.endswith(".parquet"))
    base = "https://huggingface.co/datasets/restor/tcd/resolve/main"
    return [f"{base}/{f}" for f in parquet_files]


def _run_one_model(model: str, name: str, skip_existing: bool):
    shard_urls = _get_shard_urls()
    print(f"\n🚀 {name}: {len(shard_urls)} shards → {len(shard_urls)} containers")
    results = list(infer_shard.starmap(
        [(url, model, name, skip_existing) for url in shard_urls]
    ))
    total = sum(results)
    print(f"✅ {name} done: {total} tiles processed")


@app.local_entrypoint()
def main(
    model: str = "weecology",
    name: str = "weecology",
    skip_existing: bool = True,
):
    """Run a single model. Use run_all to run all three sequentially."""
    _run_one_model(model, name, skip_existing)
    print(f"\nPull results:\n"
          f"  modal volume get canopyai-benchmark-results /{name} benchmark_results/{name}")


@app.local_entrypoint()
def run_all(
    skip_existing: bool = True,
):
    """
    Run all 3 benchmark models sequentially so all 10 GPUs go to one model at a time.

        modal run --detach deepforest_custom/modal_benchmark.py::run_all

    Resume safely at any time — completed tiles are skipped.
    """
    models = [
        ("weecology",                                              "weecology"),
        ("/checkpoints/phase21_baseline/deepforest_final.pth",    "phase21_baseline"),
        ("/checkpoints/phase21_B_λ4/deepforest_final.pth",        "phase21_B_λ4"),
    ]
    for model, name in models:
        _run_one_model(model, name, skip_existing)

    print("\n🎉 All models complete. Pull results:")
    for _, name in models:
        print(f"  modal volume get canopyai-benchmark-results /{name} benchmark_results/{name}")
