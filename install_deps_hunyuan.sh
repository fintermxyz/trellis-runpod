#!/usr/bin/env bash
# Hunyuan3D-2.0 (hy3dgen) on top of the TRELLIS stack. Only run when
# BACKEND=hunyuan. Deps are installed by hand instead of the repo's unpinned
# requirements.txt, which would upgrade torch/diffusers/transformers and break
# everything this stack has pinned.
set -euxo pipefail

# pyrender renders the preview turntable (the repo has no server-side renderer).
pip install einops omegaconf pymeshlab pygltflib "pyrender==0.1.45" PyOpenGL

[ -d /app/Hunyuan3D-2 ] || git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /app/Hunyuan3D-2
pip install -e /app/Hunyuan3D-2 --no-deps

# Texture components: a CUDA rasterizer (arch autodetected from the pod's GPU)
# and a small pybind11 CPU extension.
python -c "import custom_rasterizer" 2>/dev/null || {
    cd /app/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && python3 setup.py install
}
python -c "import mesh_processor" 2>/dev/null || {
    cd /app/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && python3 setup.py install
}
