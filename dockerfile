# canopyAI Dockerfile (HPC-friendly, CPU-only, no conda)
# Build for x86_64 from Apple Silicon with:
#   docker build --platform=linux/amd64 -t canopyai .

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System deps ----
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-dev \
    wget \
    curl \
    # for OpenCV + matplotlib
    libglib2.0-0 \
    libgl1 \
    # (optional but tiny, helps some wheels)
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Python tooling ----
RUN pip install --upgrade pip setuptools wheel

# ---- Geospatial + scientific stack (wheels, no system GDAL) ----
# Versions chosen for stability & manylinux wheels on py3.10
RUN pip install \
    numpy \
    rasterio==1.3.10 \
    fiona==1.9.5 \
    geopandas==0.14.4 \
    shapely==2.0.4 \
    pyproj==3.6.1 \
    rtree==1.3.0 \
    affine

# ---- Core ML stack ----
# CPU-only PyTorch (fine for HPC if you don't need GPUs from inside container)
RUN pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# ---- Detectron2 (CPU-compatible prebuilt wheel) ----
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html

# ---- Utilities & project deps ----
RUN pip install \
    opencv-python \
    wget \
    requests \
    matplotlib \
    cython \
    pycocotools \
    duckdb \
    pyarrow \
    scikit-image \
    tqdm \
    detectree2

# ---- Copy canopyAI repo and install in editable mode ----
WORKDIR /workspace
COPY . /workspace

# If you have a requirements.txt you still want to use, you *could* do:
# RUN pip install -r requirements.txt
# but most of it is already handled above, avoiding version chaos.

RUN pip install -e .

ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]