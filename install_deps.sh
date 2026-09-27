#!/usr/bin/env bash
# TRELLIS dependency install for torch 2.4.0 + CUDA 12.1 (pins follow microsoft/TRELLIS setup.sh).
# Wheel-only where wheels exist; the two CUDA-extension packages (nvdiffrast JIT, diff-gaussian-
# rasterization AOT via TORCH_CUDA_ARCH_LIST) build without a GPU present.
set -euxo pipefail

pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
    rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph "transformers>=4.40,<5" safetensors
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install spconv-cu120

git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
pip install /tmp/nvdiffrast && rm -rf /tmp/nvdiffrast

git clone --recursive https://github.com/autonomousvision/mip-splatting.git /tmp/mip
pip install --no-build-isolation /tmp/mip/submodules/diff-gaussian-rasterization && rm -rf /tmp/mip

pip install fastapi uvicorn pydantic
