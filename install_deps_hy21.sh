#!/usr/bin/env bash
# Hunyuan3D-2.1 SHAPE model (package hy3dshape) next to Hunyuan3D-2.0 (hy3dgen), for A/B tests
# (request field shape_model="2.1"). Shape only: the 2.1 PBR paint model is not installed.
#
# hy3dshape has no setup.py and is pure Python: no CUDA extension to build (marching cubes runs
# through scikit-image, like hy3dgen 2.0; the optional DMC path needs `diso`, which the server never
# selects). Nothing is pip-installed: the repo's requirements.txt pins torch 2.5.1 / diffusers 0.30 /
# transformers 4.46 / numpy 1.24 etc., which would break the stack this image pins (torch 2.4.0 cu121,
# diffusers 0.32.2, transformers <4.48). Every module the 2.1 shape pipeline imports (torch>=2.4 for
# nn.RMSNorm, transformers' Dinov2Model, diffusers, timm, einops, omegaconf, cv2, skimage, pymeshlab,
# trimesh) is already here for hy3dgen/TRELLIS/BiRefNet; check_hy21.py proves it at build time.
set -euxo pipefail

# Pinned Tencent-Hunyuan/Hunyuan3D-2.1 commit (2025-10-17, "Update LICENSE"; hy3dshape last changed 2025-09-24).
HY21_SHA=${HY21_SHA:-82920d643c0dc2f7bfd7255f45f62d386edfe60c}
SRC=/tmp/hy21-src
DEST=/app/Hunyuan3D-2.1

pip freeze | sort > /tmp/pip-before.txt

rm -rf "$SRC" && mkdir -p "$SRC"
git -C "$SRC" init -q
git -C "$SRC" remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
git -C "$SRC" fetch -q --depth 1 origin "$HY21_SHA"
git -C "$SRC" checkout -q FETCH_HEAD
test "$(git -C "$SRC" rev-parse HEAD)" = "$HY21_SHA"

# Keep only the inference package and the licence files. Its own directory goes on sys.path through a
# .pth file, so only `hy3dshape` becomes importable (not the repo's top-level main.py, demo.py, ...).
rm -rf "$DEST" && mkdir -p "$DEST"
cp -r "$SRC/hy3dshape/hy3dshape" "$DEST/hy3dshape"
cp "$SRC/LICENSE" "$SRC/Notice.txt" "$DEST/"
cp "$SRC/hy3dshape/LICENSE" "$DEST/hy3dshape/LICENSE"
echo "$HY21_SHA" > "$DEST/COMMIT"
find "$DEST" -name __pycache__ -prune -exec rm -rf {} +
rm -rf "$SRC"

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$DEST" > "$SITE/hunyuan3d_21_shape.pth"

# Nothing above may have changed an installed package.
pip freeze | sort > /tmp/pip-after.txt
diff /tmp/pip-before.txt /tmp/pip-after.txt
rm -f /tmp/pip-before.txt /tmp/pip-after.txt
