# Restor Mask-RCNN baseline — matched tree-only AP (LOCAL, CPU)

**Purpose.** Get Restor's per-category instance AP (`AP-tree` / `AP-canopy`) for
`restor/tcd-mask-rcnn-r50` on our 439-tile OAM-TCD holdout, so our crown pipeline's
tree-only mAP50 (0.535) can be compared to a **matched Restor tree-only number** — the
one quantity missing from the comparison (their paper only publishes the tree+canopy
2-category **mean** = 0.432; no per-category AP anywhere). See `../RESTOR_COMPARISON.md`.

Kept **isolated** here so their Detectron2 stack never touches our DeepForest/SAM
`venv310`. Everything Restor-specific lives in this folder + a dedicated
`venv_restor/` (auto-gitignored by the `venv*` rule).

## What it does (`run_restor.py`)
1. Load `restor/tcd-mask-rcnn-r50` (Detectron2 Mask-RCNN R50-FPN) on **CPU**
   from the HF `config.yaml` + `model.pth` (Detectron2's MPS support is patchy).
   Classes: **canopy = id 0, tree = id 1** (`Vegetation` IntEnum; class list
   `["canopy", "tree"]`) → OAM-TCD COCO `canopy = cat 1`, `tree = cat 2`.
2. Run instance inference on the 439 holdout tiles (2048×2048 RGB, fed at full
   size — `MIN_SIZE_TEST=0`, `MAX_SIZE_TEST=2048`, `SCORE_THRESH_TEST=0.2`,
   `DETECTIONS_PER_IMAGE=512`, all Restor defaults).
3. Evaluate with **pycocotools `COCOeval` segm**, IoU 0.5, areaRng "all",
   maxDets 512, over **both** categories → prints overall **AP50** (mean,
   sanity target ≈ 0.432) plus **AP-tree** and **AP-canopy** separately.
   (Here both categories are real positives — the Restor 0.432 framing — unlike
   `../benchmark.py`, which demotes canopy to an ignore region for *its*
   tree-only number.)
4. Export the **tree-category** predictions as `{stem}_canopyai.geojson`
   (pixel-coord polygons + `score`) into `../../benchmark_results_restor/`
   (gitignored), so they can be scored through `../benchmark.py`'s tree-only
   convention for a like-for-like number vs our 0.535.

## Run

```bash
# smoke test (first N tiles) — keep N small; CPU inference ~1.5 s/tile
phase30/restor_baseline/venv_restor/bin/python \
    phase30/restor_baseline/run_restor.py --tiles 3

# full 439 (run when no heavy MPS eval is competing for the CPU)
phase30/restor_baseline/venv_restor/bin/python \
    phase30/restor_baseline/run_restor.py
```

Outputs land in `benchmark_results_restor/`: one `*_canopyai.geojson` per tile
(tree crowns only) plus `restor_ap_summary.json` (the AP numbers).

Useful flags: `--tiles N` / `--limit N` (smoke subset), `--max-dets` (default
512, Restor's setting), `--score-thresh` (override the 0.2 test threshold),
`--skip-existing`, `--device` (default `cpu`).

## Status: INSTALL + SMOKE VERIFIED (full 439 pending, awaits free CPU)
Model loads on CPU, inference works (~1.5 s/tile), per-category COCO eval prints
AP-tree / AP-canopy / overall, and the exported geojsons are consumed cleanly by
`../benchmark._load_predictions`. Numbers from the 4-tile smoke are noisy by
design — they only prove the pipeline RUNS.

---

## INSTALL (the finicky part — exact working recipe)

Target: **Apple Silicon (M4 Max)**, **Python 3.10** (Homebrew
`/opt/homebrew/bin/python3.10`), **CPU** torch. Apple clang + macOS SDK as of
2026-06 (clang-2100 / `MacOSX26.sdk`). Xcode Command Line Tools must be
installed (`xcode-select -p` → a CLT path).

Pinned end-state in `requirements_restor.txt`. **Order and build flags matter**
— do not `pip install -r` it blindly. The working sequence:

```bash
cd "<repo root>"

# 0. Separate venv (name matches the venv* gitignore rule — DO NOT use venv310)
/opt/homebrew/bin/python3.10 -m venv phase30/restor_baseline/venv_restor
V=phase30/restor_baseline/venv_restor/bin/python
$V -m pip install --upgrade pip wheel

# 1. CPU torch / torchvision that Detectron2 0.6 builds against, + numpy<2
$V -m pip install "numpy<2" "torch==2.1.2" "torchvision==0.16.2"

# 2. Geo / COCO deps (NOTE: this re-pulls numpy 2 and a too-new opencv)
$V -m pip install pycocotools "opencv-python==4.10.0.84" rasterio shapely \
    geopandas huggingface_hub

# 3. Re-pin numpy<2 (torch 2.1.2 + the compiled detectron2 need the numpy-1 ABI)
$V -m pip install "numpy==1.26.4"

# 4. Downgrade setuptools so torch's cpp_extension can import
#    `from pkg_resources import packaging` (removed in setuptools 70+)
$V -m pip install "setuptools==69.5.1"

# 5. Build Detectron2 from source for CPU.
#    --no-build-isolation so it uses the pinned torch/setuptools above.
#    The CXXFLAGS suppress a HARD compile error from the new macOS libc++:
#    PyTorch 2.1's bundled strong_type.h specializes std::is_arithmetic, which
#    the SDK's libc++ marks _LIBCPP_NO_SPECIALIZATIONS -> fatal without this flag.
CXXFLAGS="-Wno-error=invalid-specialization -Wno-invalid-specialization" \
CFLAGS="-Wno-error=invalid-specialization -Wno-invalid-specialization" \
CC=clang CXX=clang++ MACOSX_DEPLOYMENT_TARGET=11.0 \
$V -m pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git"

# 6. Detectron2 pulls matplotlib which re-pulls numpy 2 — pin back one more time
$V -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84"

# 7. Verify
$V -c "import torch, detectron2, cv2, rasterio, geopandas, numpy; \
       from pycocotools.cocoeval import COCOeval; \
       print('OK', numpy.__version__, torch.__version__, detectron2.__version__)"
```

### Gotchas (each one bit during setup)
- **`ModuleNotFoundError: No module named 'pkg_resources'`** building detectron2
  → setuptools >= 70 removed `pkg_resources.packaging`, which torch 2.1's
  `cpp_extension.py` imports. Fix: `setuptools==69.5.1` (step 4).
- **`error: 'is_arithmetic' cannot be specialized … _LIBCPP_NO_SPECIALIZATIONS`**
  → new Apple SDK libc++ vs old PyTorch headers. Fix: the `-Wno-...invalid-
  specialization` `CXXFLAGS` in step 5.
- **numpy keeps jumping back to 2.x** → rasterio/geopandas (step 2) and
  matplotlib (pulled by detectron2, step 5) both depend on numpy>=2. The
  compiled detectron2 + torch 2.1.2 need the numpy-1 ABI, so re-pin
  `numpy==1.26.4` **after** each of those installs (steps 3 and 6).
- **opencv-python >= 4.11 requires numpy>=2** → use `opencv-python==4.10.0.84`,
  which is happy on numpy 1.26.
- Build is CPU-only (no CUDA on this Mac); `torch.cuda.is_available()` is
  `False` and that is correct — the runner forces `MODEL.DEVICE=cpu`.

### Model assets
Auto-downloaded on first run via `huggingface_hub.hf_hub_download` and cached
under `~/.cache/huggingface/`:
- `config.yaml` (full standalone Detectron2 dump; we override only DEVICE→cpu
  and WEIGHTS→the local pth),
- `model.pth` (~351 MB).

### Working versions (recorded end state)
torch 2.1.2 · torchvision 0.16.2 · detectron2 0.6 (git) · numpy 1.26.4 ·
opencv-python 4.10.0.84 · pycocotools 2.0.11 · rasterio 1.4.4 · shapely 2.1.2 ·
geopandas 1.1.3 · huggingface_hub 1.17.0 · setuptools 69.5.1 · Python 3.10.20.
