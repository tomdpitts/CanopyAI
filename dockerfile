# Base image with PyTorch and CUDA
# To run: 
    # docker build -t detectree2:tcd .
# On cluster node:
    # docker run --gpus all -it -v /data:/data detectree2:tcd /bin/bash


FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# System deps for rasterio/GDAL/shapely etc.
RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Optional: set GDAL envs
ENV GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj

# Create workspace
WORKDIR /workspace

# Copy project
COPY . /workspace

# Python deps
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
    "torchmetrics" "tensorboard" \
    "rasterio" "geopandas" "shapely" "pycocotools" "datasets" "wget"

# Install detectron2 (matching CUDA/PyTorch)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Install detectree2
RUN pip install --no-cache-dir 'git+https://github.com/PatBall1/detectree2.git'

# Default: just drop into bash
CMD ["/bin/bash"]
