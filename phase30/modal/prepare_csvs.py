#!/usr/bin/env python3
"""
phase30/modal/prepare_csvs.py — rewrite the phase30 TCD CSVs for the Modal volume.

The 2-stage ablation needs only the TCD stage-2 CSVs rebuilt:

    tcd_train.csv / tcd_val.csv   full ~3.5k TCD   image_path -> /data/images/data/tcd/raw/<basename>

(Stage-1 reuses the on-volume /data/phase22_{train,val}.csv as-is — those already
carry correct volume paths + shadow vectors, so no bwn CSV is generated here.)

The prefix matches the *working* convention already on the volume
(`phase26_tcd_train.csv` references `/data/images/data/tcd/raw/tcd_tile_0.tif`),
so the tiles are reused in place — no tile upload.

Usage:
    ./venv310/bin/python phase30/modal/prepare_csvs.py
Then upload data_csvs/tcd_*.csv + the canopy json to the volume (see README).
"""
import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "phase30" / "modal" / "data_csvs"
COMMON = ["image_path", "xmin", "ymin", "xmax", "ymax", "label",
          "shadow_angle", "shadow_x", "shadow_y"]

SOURCES = {
    "tcd": (REPO / "phase30" / "phase30_tcd_train.csv",
            REPO / "phase30" / "phase30_tcd_val.csv"),
}


def _rows(path, prefix):
    """Yield rows with image_path → <prefix>/<basename>, restricted to COMMON cols."""
    with open(path) as f:
        for row in csv.DictReader(f):
            bn = row["image_path"].rsplit("/", 1)[-1]
            out = {c: row.get(c, "") for c in COMMON}
            out["image_path"] = f"{prefix}/{bn}"
            yield out


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMMON)
        w.writeheader()
        n = 0
        for row in rows:
            w.writerow(row); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-prefix", default="/data/images/data/tcd/raw",
                    help="volume dir the TCD tiles live in (matches the on-volume convention)")
    a = ap.parse_args()
    pre = a.remote_prefix.rstrip("/")

    for split in ("train", "val"):
        idx = 0 if split == "train" else 1
        n = _write(OUT / f"tcd_{split}.csv", _rows(SOURCES["tcd"][idx], pre))
        print(f"tcd_{split}.csv: {n} boxes")
    print(f"\nwrote → {OUT}  (image_path prefix = {pre})")
    print("Next: upload tcd_{train,val}.csv + canopy json to /data/phase30/ (see README.md)")


if __name__ == "__main__":
    main()
