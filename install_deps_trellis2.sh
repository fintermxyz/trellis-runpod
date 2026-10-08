#!/usr/bin/env bash
# TRELLIS.2 dependency install for torch 2.6.0 + CUDA 12.4 on system Python (no conda).
# Mirrors microsoft/TRELLIS.2 setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh
# --o-voxel --flexgemm, with every git dependency pinned. Runs without a GPU: the CUDA extensions
# compile for TORCH_CUDA_ARCH_LIST (set by the Dockerfile) instead of probing the device.
set -euxo pipefail

T2_DIR=${T2_DIR:-/app/TRELLIS.2}
CUMESH_SHA=12289e1062f0603f2f0d0771b02e1395d247f26f
FLEXGEMM_SHA=6dd94a859c26ee8246888502eada3dd8ad85532e
NVDIFFREC_SHA=b296927cc7fd01c2ac1087c8065c4d7248f72da4   # JeffreyXiang/nvdiffrec, branch renderutils
NVDIFFRAST_TAG=v0.4.0
UTILS3D_SHA=9a4eb15e4021b67b12c460c7057d642626897ec8

# --- basic (setup.sh --basic, minus gradio/tensorboard/lpips which only the demo app and training use;
# plain pillow instead of pillow-simd, which needs a source build) ---
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh pandas zstandard plyfile \
    "transformers==4.57.1" "huggingface_hub>=0.34,<1.0" safetensors kornia timm
pip install "git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_SHA}"

# --- flash-attn: prebuilt wheel matching torch 2.6 / cu12 / this Python / torch's C++ ABI ---
PYTAG=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
ABI=$(python -c 'import torch; print("TRUE" if torch.compiled_with_cxx11_abi() else "FALSE")')
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"

# --- CUDA extensions (AOT for TORCH_CUDA_ARCH_LIST) ---
mkdir -p /tmp/ext
git clone -b "${NVDIFFRAST_TAG}" --depth 1 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast
pip install --no-build-isolation /tmp/ext/nvdiffrast

git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec
git -C /tmp/ext/nvdiffrec checkout "${NVDIFFREC_SHA}"
pip install --no-build-isolation /tmp/ext/nvdiffrec

git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh
git -C /tmp/ext/CuMesh checkout "${CUMESH_SHA}"
git -C /tmp/ext/CuMesh submodule update --init --recursive
pip install --no-build-isolation /tmp/ext/CuMesh

git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM
git -C /tmp/ext/FlexGEMM checkout "${FLEXGEMM_SHA}"
git -C /tmp/ext/FlexGEMM submodule update --init --recursive
pip install --no-build-isolation /tmp/ext/FlexGEMM

cp -r "${T2_DIR}/o-voxel" /tmp/ext/o-voxel
pip install --no-build-isolation /tmp/ext/o-voxel

rm -rf /tmp/ext

# --- server deps (same roles as install_deps.sh / install_deps_hunyuan.sh) ---
# diffusers/accelerate: SDXL concept image for text mode. rembg+onnxruntime: u2net for /cutout and bg="u2net".
# pyrender/PyOpenGL: preview turntable (EGL). pymeshfix: watertight repair fallback for print exports.
pip install "diffusers==0.35.2" "accelerate==1.10.1" rembg onnxruntime "pyrender==0.1.45" PyOpenGL pymeshfix scikit-image \
    fastapi uvicorn pydantic scipy

# Fail the build here, not on the pod, if a compiled extension doesn't import. (Only imports that need
# no GPU at import time; the CUDA kernels themselves can only be exercised on the pod.)
# nvdiffrec_render links libcuda directly, which only exists on a GPU host: checked for presence only.
python - <<'EOF'
import importlib, importlib.util
import numpy as np
if not hasattr(np, "infty"):
    np.infty = np.inf  # pyrender 0.1.45 predates numpy 2
for m in ["torch", "flash_attn", "nvdiffrast.torch", "cumesh", "flex_gemm", "o_voxel",
          "utils3d", "transformers", "diffusers", "pyrender", "pymeshfix", "rembg"]:
    importlib.import_module(m)
    print("import ok:", m)
assert importlib.util.find_spec("nvdiffrec_render") is not None, "nvdiffrec_render missing"
from transformers import DINOv3ViTModel  # TRELLIS.2's image-cond model needs transformers >= 4.56
EOF
