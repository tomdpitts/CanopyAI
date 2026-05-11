#!/usr/bin/env python3
"""
setup_by_id.py — Create image_id-keyed tile directory and shadow vectors JSON.

Creates data/tcd/images/data/tcd/by_id/ with:
  tcd_{image_id}.tif       → symlink to original tcd_tile_N.tif
  tcd_{image_id}_meta.json → copy of original meta (with stem updated)

Creates data/tcd/tcd_shadow_vectors_by_id.json:
  - Re-keyed from tcd_tile_N → tcd_{image_id}
  - All 620 manually_reviewed entries reset to False (annotations were
    linked to wrong images due to download-order ID mismatch)
  - Auto-estimated entries preserved as-is

The original data/tcd/images/data/tcd/raw/ is untouched.

Usage:
    source venv310/bin/activate
    python phase23/setup_by_id.py
"""

import json
import os
from pathlib import Path

RAW_DIR    = Path("data/tcd/images/data/tcd/raw")
BY_ID_DIR  = Path("data/tcd/images/data/tcd/by_id")
OLD_JSON   = Path("data/tcd/tcd_shadow_vectors.json")
NEW_JSON   = Path("data/tcd/tcd_shadow_vectors_by_id.json")


def main():
    BY_ID_DIR.mkdir(parents=True, exist_ok=True)

    # Build mapping: tcd_tile_N → image_id
    stem_to_id = {}
    missing_meta = 0
    for meta_p in sorted(RAW_DIR.glob("tcd_tile_*_meta.json")):
        try:
            meta = json.loads(meta_p.read_text())
            iid  = meta.get("image_id")
            if iid is not None:
                stem_to_id[meta_p.stem.replace("_meta", "")] = int(iid)
        except Exception:
            missing_meta += 1

    print(f"Stems with image_id: {len(stem_to_id)}  (missing meta: {missing_meta})")

    # Create symlinks and meta copies
    created = skipped = 0
    for stem, iid in stem_to_id.items():
        src_tif  = RAW_DIR  / f"{stem}.tif"
        src_meta = RAW_DIR  / f"{stem}_meta.json"
        dst_tif  = BY_ID_DIR / f"tcd_{iid}.tif"
        dst_meta = BY_ID_DIR / f"tcd_{iid}_meta.json"

        if not src_tif.exists() or src_tif.stat().st_size == 0:
            continue

        # Symlink for tif (relative path)
        if not dst_tif.exists():
            rel = os.path.relpath(src_tif, BY_ID_DIR)
            dst_tif.symlink_to(rel)
            created += 1
        else:
            skipped += 1

        # Copy meta with updated stem key
        if not dst_meta.exists():
            meta = json.loads(src_meta.read_text())
            dst_meta.write_text(json.dumps(meta))

    print(f"Symlinks created: {created}  already existed: {skipped}")

    # Re-key shadow vectors JSON
    old_vecs = json.loads(OLD_JSON.read_text())
    new_vecs = {}
    n_reset  = 0

    for old_stem, v in old_vecs.items():
        iid = stem_to_id.get(old_stem)
        if iid is None:
            continue
        new_key = f"tcd_{iid}"
        entry   = dict(v)
        if entry.get("manually_reviewed"):
            entry["manually_reviewed"] = False
            n_reset += 1
        new_vecs[new_key] = entry

    NEW_JSON.write_text(json.dumps(new_vecs, indent=2))
    print(f"Shadow vectors re-keyed: {len(new_vecs)}")
    print(f"  manually_reviewed reset to False: {n_reset}")
    print(f"Saved → {NEW_JSON}")
    print(f"\nNext: run the review tool")
    print(f"  python deepforest_custom/tcd_shadow/review_tcd_shadows.py --filter unreviewed")


if __name__ == "__main__":
    main()
