# canopyAI Dockerfile (HPC-friendly, CUDA-enabled for GPU training)
# Build for x86_64 from Apple Silicon with:
#   docker build --platform=linux/amd64 -t canopyai .
# Run with GPU support:
#   docker run --gpus all -it canopyai

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- Install Python 3.10 ----
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

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

# ---- Core ML stack (CUDA 11.8) ----
RUN pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ---- Detectron2 (build from source for cross-platform compatibility) ----
# Prebuilt wheels don't work for --platform=linux/amd64 builds from Mac
RUN pip install --no-build-isolation \
    git+https://github.com/facebookresearch/detectron2.git@v0.6

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
    tqdm

# ---- Detectree2 (build from source) ----
RUN pip install --no-build-isolation \
    git+https://github.com/PatBall1/detectree2.git

# ---- Copy canopyAI repo and install in editable mode ----
WORKDIR /workspace
COPY . /workspace

# If you have a requirements.txt you still want to use, you *could* do:
# RUN pip install -r requirements.txt
# but most of it is already handled above, avoiding version chaos.

RUN pip install -e .

ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]