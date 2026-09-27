#!/usr/bin/env bash
# Boots a BARE pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel pod into the TRELLIS generation
# server — the fallback when the prebuilt ghcr.io/fintermxyz/trellis-runpod image can't be
# pulled. Idempotent-ish; takes ~10-15 min on first boot.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

for i in 1 2 3; do
    apt-get update && apt-get install -y --no-install-recommends         git curl ca-certificates libgl1 libegl1 libgles2 libglvnd0 libopengl0 libgomp1         libusb-1.0-0 libx11-6 libxext6 libsm6 libice6 libxrender1 libdrm2 && break
    echo "apt attempt $i failed, retrying" >&2; sleep 10
done
rm -rf /var/lib/apt/lists/*

TRELLIS_SHA=442aa1e1afb9014e80681d3bf604e8d728a86ee7
mkdir -p /app
if [ -d /app/repo ]; then git -C /app/repo pull --ff-only || true
else git clone https://github.com/fintermxyz/trellis-runpod.git /app/repo; fi
if [ ! -d /app/TRELLIS ]; then
    git clone https://github.com/microsoft/TRELLIS.git /app/TRELLIS
    git -C /app/TRELLIS checkout "$TRELLIS_SHA"
fi

bash /app/repo/install_deps.sh
cp /app/repo/server.py /app/server.py

export PYTHONPATH=/app/TRELLIS
export ATTN_BACKEND=xformers SPCONV_ALGO=native
export HF_HOME="${HF_HOME:-/workspace/hf}"
cd /app
exec python -m uvicorn server:app --host 0.0.0.0 --port 8000
