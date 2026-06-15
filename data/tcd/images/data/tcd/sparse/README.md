# TCD "sparse" test subset (biome-filtered)

A **180-tile subset of the 439-tile OAM-TCD holdout** (`../val/`), filtered to
non-tropical, biome-labelled tiles. Built to focus evaluation on
temperate/dryland scenes with more individual-tree structure (vs closed tropical
rainforest canopy), for sparse-distributed individual-tree detection.

> ⚠️ Naming: called "sparse" as a project label, but **no canopy-density cap is
> applied** — it includes dense-canopy temperate tiles. The filter is purely
> *biome* + the existing 439 set. Per-tile `canopy_frac` is recorded in
> `_SUBSET_PROVENANCE.json` if you later want to add a canopy cut.

## Selection criteria (exact)
Source: `data/tcd/images/data/tcd/val` (the 439-tile OAM-TCD holdout). A tile is
**included** iff:
1. **Biome NOT in** `{1, 2, 3, 7, -1}` — i.e. exclude all tropical biomes
   (TropMoistForest 1, TropDryForest 2, TropConifForest 3, TropSavanna 7) and
   Unknown (−1, biome metadata missing).
2. **No canopy restriction** (all densities).
3. **Any tree count** (including 0 — 26 of the 180 have no individual-tree GT;
   these are neutral to mAP via the canopy `iscrowd` ignore, but count as
   foreground in the semantic F1/IoU).

Result: **180 tiles.** Funnel: 439 → 180 (− 259 tropical/Unknown).

## Methodology notes (gotchas that matter for recreation)
- **`meta.json` GT:** `coco_annotations` is a **JSON-encoded string** (must
  `json.loads()` it again). Categories: **cat=1 = canopy regions, cat=2 =
  individual tree crowns**.
- **`biome` field = WWF/RESOLVE biome code (1–14, −1=unknown):**
  `1 TropMoistForest · 2 TropDryForest · 3 TropConifForest · 4 TempBroadleaf ·
  5 TempConifer · 6 Boreal · 7 TropSavanna · 8 TempGrass · 9 FloodedGrass ·
  10 MontaneGrass · 11 Tundra · 12 Mediterranean · 13 Desert/Xeric · 14 Mangrove`.
- **Segmentation is MIXED polygon + RLE.** `canopy_frac` (recorded, not filtered
  here) is the union area of cat=1 polygons ÷ tile area, decoded **RLE-aware via
  pycocotools** — a polygon-only rasterizer silently counts RLE regions as 0.
- **Stems preserved as `tcd_val_tile_N`** (symlinks to `../val/`), so existing
  prediction geojsons can be re-scored on this subset with **no re-inference**.

## Composition (180)
- Biome: TempBroadleaf 137 · TempConifer 31 · Desert/Xeric 4 · Boreal 3 ·
  TempGrass 2 · Mediterranean 2 · Mangrove 1
- Trees: 154 tiles with ≥1 tree; 26 zero-tree.

## Recreate
```bash
./venv310/bin/python data/tcd/images/data/tcd/sparse/build_sparse_subset.py
```
Deterministic from `../val/` GT — re-running reproduces the same 180 symlinks,
`sparse_tiles.txt`, and `_SUBSET_PROVENANCE.json`.

## Files
- `<stem>.tif`, `<stem>_meta.json` — symlinks into `../val/` (180 each)
- `_SUBSET_PROVENANCE.json` — criteria + per-tile canopy_frac / trees / biome
- `build_sparse_subset.py` — the exact builder
- `sparse_tiles.txt` (repo root) — 180-stem manifest (git-tracked; this folder is
  under `data/*` which is gitignored)

## Usage
```bash
# self-contained:
./venv310/bin/python phase30/benchmark.py --holdout-dir data/tcd/images/data/tcd/sparse ...

# OR reuse existing predictions (no re-inference), scoring only these stems:
./venv310/bin/python phase30/benchmark.py --models x --names <existing_dir> \
    --skip-inference --tiles-file sparse_tiles.txt \
    --max-dets 512 --pred-score-thresh 0.0 --output-root benchmark_results_holdout
```

### Many models, fresh inference → offload DF+SAM to Modal (CUDA), rerank locally
DF+SAM on Modal (the holdout tiles are already on `canopyai-deepforest-data:/holdout`,
no upload), then rerank+score in one local command. `benchmark.py --skip-inference
--reranker-checkpoint` now reranks the geojsons in place before scoring. See
`deepforest_custom/modal_benchmark.py` (`run_sparse_subset` entrypoint) for the exact 2-step.
