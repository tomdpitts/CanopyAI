#!/usr/bin/env python3
"""
Download NEON AOP camera tiles (DP3.30010.001) for new environment types.

Tropical:  LAJA(4) GUAN(3) PUUM(3)
Wetland:   LENO(4) DELA(3) DSNY(3)
Savanna:   CLBJ(4) KONZ(3) JERC(3)
"""
import json, random, sys, time
import urllib.request
from pathlib import Path

OUT_DIR = Path("phase18/neon_samples")
SEED    = 42

TARGETS = [
    # (site, month, n_tiles, environment)
    ("LAJA", "2022-11", 4, "tropical"),
    ("GUAN", "2022-10", 3, "tropical"),
    ("PUUM", "2025-01", 3, "tropical"),
    ("LENO", "2024-05", 4, "wetland"),
    ("DELA", "2024-05", 3, "wetland"),
    ("DSNY", "2025-05", 3, "wetland"),
    ("CLBJ", "2025-05", 4, "savanna"),
    ("KONZ", "2024-09", 3, "savanna"),
    ("JERC", "2024-05", 3, "savanna"),
]

API = "https://data.neonscience.org/api/v0/data/DP3.30010.001"


def get_tile_urls(site, month):
    url = f"{API}/{site}/{month}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    files = [f for f in data["data"]["files"] if f["name"].endswith("_image.tif")]
    return files


def download_file(url, dest, label):
    dest = Path(dest)
    if dest.exists():
        print(f"  already exists: {dest.name}")
        return
    tmp = dest.with_suffix(".tmp")
    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb  = downloaded / 1e6
                        sys.stdout.write(f"\r  {label}  {mb:.0f}/{total/1e6:.0f}MB  {pct:.0f}%")
                        sys.stdout.flush()
        tmp.rename(dest)
        elapsed = time.time() - start
        print(f"\r  {label}  {total/1e6:.0f}MB  done in {elapsed:.0f}s")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"\r  {label}  FAILED: {e}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    total_tiles = sum(n for _, _, n, _ in TARGETS)
    print(f"Downloading {total_tiles} tiles (~{total_tiles * 330:.0f}MB) to {OUT_DIR}/\n")

    done = 0
    for site, month, n, env in TARGETS:
        print(f"[{env}] {site} {month} — fetching index...")
        try:
            files = get_tile_urls(site, month)
        except Exception as e:
            print(f"  ERROR fetching index: {e}")
            continue

        # Skip tiles already downloaded
        existing = {p.stem for p in OUT_DIR.glob("*.tif")}
        files = [f for f in files if Path(f["name"]).stem not in existing]

        selected = rng.sample(files, min(n, len(files)))
        print(f"  {len(files)} available → downloading {len(selected)}")

        for f in selected:
            dest = OUT_DIR / f["name"]
            download_file(f["url"], dest, f["name"])
            done += 1

    print(f"\nDone. {done} tiles downloaded to {OUT_DIR}/")


if __name__ == "__main__":
    main()
