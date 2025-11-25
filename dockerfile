# ============================================================
# canopyAI Dockerfile (HPC-ready & Singularity compatible)
# ============================================================

# To build:
# docker build -t canopyai .

# Hardcoded to Linux x86_64 for HPC env
FROM --platform=linux/amd64 continuumio/miniconda3

# --- System deps ---
RUN apt-get update && apt-get install -y \
    git build-essential wget \
    libgl1 libglib2.0-0 libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# --- Conda env ---
RUN conda create -y -n canopyAI python=3.10
SHELL ["conda", "run", "-n", "canopyAI", "/bin/bash", "-c"]

# --- Geospatial stack ---
RUN conda install -y -c conda-forge \
    gdal=3.10.* \
    rasterio=1.4.* \
    fiona=1.9.* \
    geopandas=0.14.* \
    shapely pyproj rtree affine \
    && conda clean -afy

# --- PyTorch CPU ---
RUN pip install torch torchvision torchaudio

# --- Detectron2 ---
RUN pip install "git+https://github.com/facebookresearch/detectron2.git"

# --- Utilities ---
RUN pip install \
    opencv-python wget requests matplotlib cython pycocotools \
    duckdb pyarrow scikit-image tqdm

# --- Copy canopyAI repo ---
WORKDIR /workspace
COPY . /workspace

# --- Install canopyAI package from the repo ---
RUN pip install -e .

ENV PYTHONPATH=/workspace
ENV PATH=/opt/conda/envs/canopyAI/bin:$PATH

CMD ["/bin/bash"]