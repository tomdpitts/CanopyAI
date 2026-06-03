"""
phase30/modal/train.py — Modal training for the shadow ablation (2-stage, matched).

One `modal run` per cell; checkpoints land on the canopyai-deepforest-checkpoints
volume. Shadow weight is held CONSTANT across both stages (4→4, 2→2, off→off).

    Stage 1  --dataset bwn  (BRU/WON/NEON, ~100% shadow)   phase22 recipe: bs16 lr1e-3 50ep p10
             s0 = phase21_baseline (exists)  s4 = phase22_B_L4 (exists)  s2 = NEW
    Stage 2  --dataset tcd  (4k TCD, canopy on, ~11% shadow), --base-checkpoint <stage-1>
             kunqi5 recipe (colleague's history): bs16 lr1e-5 500ep patience5 canopy1.0

Examples
--------
    # smoke test first (1 epoch, tiny — verify the path end-to-end):
    modal run phase30/modal/train.py --dataset tcd --shadow 2 --run-name smoke \
        --base-checkpoint /checkpoints/phase22_B_L4/deepforest_final.pth --fast-dev-run

    # stage-1 shadow-2 pretrain (the only new stage-1 cell)
    modal run --detach phase30/modal/train.py --dataset bwn --shadow 2 --run-name ablation_pre_s2 \
        --batch-size 16 --lr 0.001 --epochs 50 --patience 10
    # stage-2 fine-tune from a matching stage-1 ckpt (shadow held constant)
    modal run --detach phase30/modal/train.py --dataset tcd --shadow 4 --run-name ablation_tcd_s4 \
        --base-checkpoint /checkpoints/phase22_B_L4/deepforest_final.pth \
        --batch-size 16 --lr 0.00001 --epochs 500 --patience 5

Data expected on the `canopyai-deepforest-data` volume (see README):
    /data/phase22_{train,val}.csv          stage-1 (already on the volume)
    /data/phase30/tcd_{train,val}.csv      stage-2 (upload from prepare_csvs)
    /data/phase30/canopy_polygons.json     TCD canopy (upload, ~292 MB)
"""
import modal

APP = "canopyai-phase30-train"
app = modal.App(APP)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "git", "gdal-bin", "libgdal-dev")
    .pip_install_from_requirements("phase30/requirements.txt")
    .add_local_file("foxtrot.py", remote_path="/root/canopyAI/foxtrot.py")
    .add_local_file("utils.py",   remote_path="/root/canopyAI/utils.py")
    .add_local_dir("phase30/lib", remote_path="/root/canopyAI/phase30/lib")
    .add_local_dir("deepforest_custom", remote_path="/root/canopyAI/deepforest_custom",
                   ignore=["*.pth", "*.tif", "__pycache__"])
)

data_vol  = modal.Volume.from_name("canopyai-deepforest-data", create_if_missing=True)
ckpt_vol  = modal.Volume.from_name("canopyai-deepforest-checkpoints", create_if_missing=True)

# dataset key -> (train_csv, val_csv, uses_canopy)
#   bwn = stage-1 pretrain, reuses the on-volume phase22 CSVs as-is (correct paths + shadow vectors)
#   tcd = stage-2 fine-tune, the rewritten phase30 TCD CSVs (uploaded to /data/phase30/)
DATASETS = {
    "bwn": ("/data/phase22_train.csv",     "/data/phase22_val.csv",     False),
    "tcd": ("/data/phase30/tcd_train.csv", "/data/phase30/tcd_val.csv", True),
}
CANOPY_JSON = "/data/phase30/canopy_polygons.json"


@app.function(image=image, gpu="A100", timeout=86400,
              volumes={"/data": data_vol, "/checkpoints": ckpt_vol})
def train(dataset: str, shadow: float, run_name: str,
          epochs: int = 50, lr: float = 0.001, batch_size: int = 16,
          patience: int = 10, base_checkpoint: str = None,
          blind: bool = False, fast_dev_run: bool = False):
    import os, sys
    os.chdir("/root/canopyAI")
    sys.path.insert(0, "/root/canopyAI/phase30/lib")
    if blind:
        os.environ["SHADOW_BLIND_CONTROL"] = "1"   # specificity control
    from train_deepforest import train_deepforest

    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}")
    train_csv, val_csv, uses_canopy = DATASETS[dataset]

    AUG = [
        {"GaussianBlur":             {"blur_limit": [3, 7], "p": 0.3}},
        {"RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5}},
        {"HueSaturationValue":       {"hue_shift_limit": 10, "sat_shift_limit": 20,
                                      "val_shift_limit": 20, "p": 0.5}},
    ]
    print(f"🌲 phase30 train | dataset={dataset} shadow={shadow} blind={blind} "
          f"canopy={uses_canopy} epochs={epochs} lr={lr} bs={batch_size}", flush=True)

    train_deepforest(
        train_csv=train_csv,
        val_csv=val_csv,
        checkpoint=base_checkpoint,            # None -> weecology base
        output_dir="/checkpoints",
        run_name=run_name,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
        shadow_loss_reweight=True,             # weight=1 => no reweighting (clean "no-shadow")
        shadow_loss_weight=shadow,
        canopy_polygons_path=(CANOPY_JSON if uses_canopy else None),
        augmentations=AUG,
        accelerator="gpu",                     # CUDA on Modal — no MPS issues
        fast_dev_run=fast_dev_run,
    )
    ckpt_vol.commit()
    print(f"✅ done → /checkpoints/{run_name}", flush=True)


@app.local_entrypoint()
def main(dataset: str, shadow: float, run_name: str,
         epochs: int = 50, lr: float = 0.001, batch_size: int = 16,
         patience: int = 10, base_checkpoint: str = None,
         blind: bool = False, fast_dev_run: bool = False):
    train.remote(dataset=dataset, shadow=shadow, run_name=run_name,
                 epochs=epochs, lr=lr, batch_size=batch_size, patience=patience,
                 base_checkpoint=base_checkpoint, blind=blind, fast_dev_run=fast_dev_run)
