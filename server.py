"""Tiny generation API around microsoft/TRELLIS, made for RunPod pods.

GET  /health                     -> {ok, gpu, loaded: [modes]}
POST /generate  (X-Token)        -> {name, mode, prompt|image, seed, simplify, texture_size,
                                     ss_steps?, ss_cfg?, slat_steps?, slat_cfg?}
                                    => GLB + 4-view preview PNG
GET  /asset/<name>.glb|.png (X-Token)

Two modes:
  mode="text"  -> TRELLIS-text-xlarge, prompt -> 3D
  mode="image" -> TRELLIS-image-large, single object photo (data URL / base64,
                  rembg strips the background) -> 3D. Markedly higher fidelity;
                  this is the Tripo-style path (generate/choose an image first).

The two pipelines don't fit a 24 GB card together, so switching modes evicts
the other pipeline (first request in a new mode pays the load).

Single-flight: one generation at a time (the GPU is fully busy anyway).
The GLB is written before the preview PNG — clients may treat the PNG's
existence as "the GLB is complete"; keep that ordering.
"""
import base64
import io
import os
import re
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

OUT = Path(os.environ.get("OUT_DIR", "/workspace/out"))
OUT.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ.get("GEN_TOKEN", "")

app = FastAPI()
_lock = threading.Lock()
_pipes = {}

MODELS = {
    "text": os.environ.get("TEXT_MODEL", "microsoft/TRELLIS-text-xlarge"),
    "image": os.environ.get("IMAGE_MODEL", "microsoft/TRELLIS-image-large"),
}


def _check(x_token: str) -> None:
    if not TOKEN or x_token != TOKEN:
        raise HTTPException(401, "bad token")


def _get_pipe(mode: str):
    global _pipes
    if mode not in _pipes:
        # Evict the other pipeline first: both don't fit in 24 GB.
        for other in [k for k in _pipes if k != mode]:
            del _pipes[other]
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        if mode == "image":
            from trellis.pipelines import TrellisImageTo3DPipeline

            p = TrellisImageTo3DPipeline.from_pretrained(MODELS["image"])
        else:
            from trellis.pipelines import TrellisTextTo3DPipeline

            p = TrellisTextTo3DPipeline.from_pretrained(MODELS["text"])
        p.cuda()
        _pipes[mode] = p
    return _pipes[mode]


def _decode_image(data: str):
    from PIL import Image

    b64 = data.split(",", 1)[1] if data.startswith("data:") else data
    # RGBA: an alpha channel is used as the object mask; otherwise the
    # pipeline's own preprocessing runs rembg.
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")


@app.get("/health")
def health():
    import torch

    return {
        "ok": True,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded": sorted(_pipes.keys()),
        "model_loaded": bool(_pipes),  # back-compat
    }


class GenReq(BaseModel):
    name: str
    mode: str = "text"  # "text" | "image"
    prompt: str = ""
    image: str | None = None  # data URL or raw base64, mode="image"
    seed: int = 0
    simplify: float = 0.9
    texture_size: int = 1024
    ss_steps: int | None = None
    ss_cfg: float | None = None
    slat_steps: int | None = None
    slat_cfg: float | None = None


@app.post("/generate")
def generate(req: GenReq, x_token: str = Header(default="")):
    _check(x_token)
    if not re.fullmatch(r"[a-z0-9_]{1,40}", req.name):
        raise HTTPException(400, "bad name")
    if req.mode not in ("text", "image"):
        raise HTTPException(400, "mode must be 'text' or 'image'")
    if req.mode == "image" and not req.image:
        raise HTTPException(400, "image required for mode='image'")
    if req.mode == "text" and not req.prompt.strip():
        raise HTTPException(400, "prompt required for mode='text'")
    if not _lock.acquire(timeout=2):
        raise HTTPException(409, "busy")
    try:
        t0 = time.time()
        pipe = _get_pipe(req.mode)

        # Sampler overrides: only what the caller sets; pipeline defaults otherwise.
        sampler = {}
        ss = {}
        if req.ss_steps:
            ss["steps"] = int(req.ss_steps)
        if req.ss_cfg:
            ss["cfg_strength"] = float(req.ss_cfg)
        if ss:
            sampler["sparse_structure_sampler_params"] = ss
        slat = {}
        if req.slat_steps:
            slat["steps"] = int(req.slat_steps)
        if req.slat_cfg:
            slat["cfg_strength"] = float(req.slat_cfg)
        if slat:
            sampler["slat_sampler_params"] = slat

        subject = _decode_image(req.image) if req.mode == "image" else req.prompt
        out = pipe.run(subject, seed=req.seed, formats=["gaussian", "mesh"], **sampler)
        gaussian, mesh = out["gaussian"][0], out["mesh"][0]

        from trellis.utils import postprocessing_utils, render_utils

        glb = postprocessing_utils.to_glb(
            gaussian, mesh, simplify=req.simplify, texture_size=req.texture_size, verbose=False
        )
        glb.export(str(OUT / f"{req.name}.glb"))

        import imageio
        import numpy as np

        frames = render_utils.render_video(gaussian, resolution=512, num_frames=4)["color"]
        grid = np.concatenate(
            [np.concatenate(frames[0:2], axis=1), np.concatenate(frames[2:4], axis=1)], axis=0
        )
        imageio.imwrite(str(OUT / f"{req.name}.png"), grid)

        if hasattr(glb, "vertices"):
            verts = int(glb.vertices.shape[0])
        else:  # trimesh.Scene
            verts = int(sum(g.vertices.shape[0] for g in glb.geometry.values()))
        return {"ok": True, "mode": req.mode, "ms": int((time.time() - t0) * 1000), "verts": verts}
    except HTTPException:
        raise
    except Exception as e:  # surface the real error to the client
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        _lock.release()


@app.get("/asset/{fname}")
def asset(fname: str, x_token: str = Header(default="")):
    _check(x_token)
    if not re.fullmatch(r"[a-z0-9_]{1,40}\.(glb|png)", fname):
        raise HTTPException(400, "bad file")
    p = OUT / fname
    if not p.exists():
        raise HTTPException(404, "missing")
    return FileResponse(str(p))
