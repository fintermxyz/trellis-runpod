# 3D generation server for RunPod pods: TRELLIS + Hunyuan3D-2.0 (BACKEND=hunyuan, the default).
# Everything is installed and compiled at build time, so a pod from this image only
# downloads model weights on first start instead of running bootstrap.sh (~15 min).
# torch 2.4.0 + CUDA 12.1 is the combination microsoft/TRELLIS's setup.sh pins its wheels to.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libegl1 libgles2 libglvnd0 libopengl0 libgomp1 libusb-1.0-0 libx11-6 libxext6 libsm6 libice6 libxrender1 libdrm2 \
    && rm -rf /var/lib/apt/lists/*

# Pinned TRELLIS checkout (MIT licensed, (c) Microsoft).
ARG TRELLIS_SHA=442aa1e1afb9014e80681d3bf604e8d728a86ee7
RUN git clone https://github.com/microsoft/TRELLIS.git /app/TRELLIS \
    && git -C /app/TRELLIS checkout ${TRELLIS_SHA} \
    && git -C /app/TRELLIS submodule update --init --recursive

# CUDA extensions are compiled ahead of time without a GPU: A100 (8.0), A5000-A40 / RTX 30xx
# (8.6), 4090 / L40S (8.9) and H100 (9.0), + PTX for anything newer. nvdiffrast JIT-compiles
# on the pod at first use.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
COPY install_deps.sh /app/install_deps.sh
RUN bash /app/install_deps.sh

# Hunyuan3D-2.0 shape + texture pipelines (custom_rasterizer is a CUDAExtension and honours
# TORCH_CUDA_ARCH_LIST above).
COPY install_deps_hunyuan.sh /app/install_deps_hunyuan.sh
RUN bash /app/install_deps_hunyuan.sh && rm -rf /root/.cache

COPY server.py /app/server.py
ENV PYTHONPATH=/app/TRELLIS \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    ATTN_BACKEND=xformers \
    SPCONV_ALGO=native \
    PYOPENGL_PLATFORM=egl \
    BACKEND=hunyuan \
    HF_HOME=/workspace/hf \
    OUT_DIR=/workspace/out

WORKDIR /app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
