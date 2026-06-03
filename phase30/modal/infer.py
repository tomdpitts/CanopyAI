"""
phase30/modal/infer.py — Modal inference/benchmark for the experiment.

Runs the 0.515 pipeline (detector → SAM → optional reranker) on the TCD holdout
for one or more trained checkpoints, with **configurable SAM (default vit_h)** and
**reranker on/off**, and reports the Restor metrics (IoU / F1 / mAP50 / Area-F1).

Examples
--------
    # one-time: pull SAM weights onto the checkpoints volume
    modal run phase30/modal/infer.py::download_sam --variant vit_h

    # full 0.515-style pipeline (vit_h + reranker) on two trained runs
    modal run --detach phase30/modal/infer.py \
        --models comb_s2,bwn_s4 --sam vit_h --reranker true --names comb_s2_full,bwn_s4_full

    # reranker-off ablation
    modal run --detach phase30/modal/infer.py --models comb_s2 --sam vit_h --reranker false

    # pull results
    modal volume get canopyai-benchmark-results /comb_s2_full benchmark_results_holdout/comb_s2_full

Data expected on volumes (see README):
    /data/holdout/*.tif + *_meta.json            (the 439-tile TCD holdout)
    /checkpoints/<run_name>/deepforest-*.ckpt    (from train.py)
    /checkpoints/sam_vit_{b,l,h}_*.pth           (download_sam)
    /checkpoints/cnn_reranker_ens3.pt            (uploaded; for --reranker true)
"""
import modal

app = modal.App("canopyai-phase30-infer")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "git", "gdal-bin", "libgdal-dev")
    .pip_install_from_requirements("phase30/requirements.txt")
    .pip_install("segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
                 "wget")
    .add_local_file("foxtrot.py", remote_path="/root/canopyAI/foxtrot.py")
    .add_local_file("utils.py",   remote_path="/root/canopyAI/utils.py")
    .add_local_dir("phase30", remote_path="/root/canopyAI/phase30",
                   ignore=["**/__pycache__", "shadow_eval", "zeroshot", "modal/data_csvs",
                           "*.md", "*.pt", "*.pth", "*.tif", "*.npy", "*.png",
                           "*_canopy_polygons.json", "phase30_tcd_*.csv", "phase22_*.csv"])
    .add_local_dir("deepforest_custom", remote_path="/root/canopyAI/deepforest_custom",
                   ignore=["*.pth", "*.tif", "__pycache__"])
)

data_vol = modal.Volume.from_name("canopyai-deepforest-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("canopyai-deepforest-checkpoints", create_if_missing=True)
res_vol  = modal.Volume.from_name("canopyai-benchmark-results", create_if_missing=True)

SAM_URLS = {
    "vit_b": ("sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
    "vit_l": ("sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
    "vit_h": ("sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
}


@app.function(image=image, timeout=3600, volumes={"/checkpoints": ckpt_vol})
def download_sam(variant: str = "vit_h"):
    import os, urllib.request
    fname, url = SAM_URLS[variant]
    out = f"/checkpoints/{fname}"
    if os.path.exists(out):
        print(f"already present: {out}"); return
    print(f"downloading {variant} → {out}")
    urllib.request.urlretrieve(url, out)
    ckpt_vol.commit()
    print(f"✅ {out} ({os.path.getsize(out)/1e9:.1f} GB)")


@app.function(image=image, gpu="A100", timeout=86400,
              volumes={"/data": data_vol, "/checkpoints": ckpt_vol, "/results": res_vol})
def benchmark(models: str, sam: str = "vit_h", reranker: bool = True,
              names: str = None, df_confidence: float = 0.05,
              max_dets: int = 512, pred_score_thresh: float = 0.0,
              holdout_dir: str = "/data/holdout"):
    import os, glob, subprocess, sys
    os.chdir("/root/canopyAI")

    # resolve each run-name to its checkpoint (or accept a literal path)
    def resolve(m):
        if os.path.exists(m):
            return m
        hits = sorted(glob.glob(f"/checkpoints/{m}/deepforest-*.ckpt")) or \
               sorted(glob.glob(f"/checkpoints/{m}/*.ckpt")) or \
               sorted(glob.glob(f"/checkpoints/{m}*.pth"))
        if not hits:
            raise FileNotFoundError(f"no checkpoint for '{m}' under /checkpoints")
        return hits[0]

    model_list = [resolve(m.strip()) for m in models.split(",") if m.strip()]
    name_list  = [n.strip() for n in names.split(",")] if names else \
                 [m.strip() for m in models.split(",") if m.strip()]
    sam_ckpt = f"/checkpoints/{SAM_URLS[sam][0]}"

    cmd = ["python", "phase30/benchmark.py",
           "--models", *model_list, "--names", *name_list,
           "--sam-model", sam, "--sam-checkpoint", sam_ckpt,
           "--df-confidence", str(df_confidence),
           "--max-dets", str(max_dets), "--pred-score-thresh", str(pred_score_thresh),
           "--holdout-dir", holdout_dir, "--output-root", "/results"]
    if reranker:
        cmd += ["--reranker-checkpoint", "/checkpoints/cnn_reranker_ens3.pt"]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    res_vol.commit()


@app.local_entrypoint()
def main(models: str, sam: str = "vit_h", reranker: bool = True, names: str = None,
         df_confidence: float = 0.05, max_dets: int = 512, pred_score_thresh: float = 0.0):
    benchmark.remote(models=models, sam=sam, reranker=reranker, names=names,
                     df_confidence=df_confidence, max_dets=max_dets,
                     pred_score_thresh=pred_score_thresh)
