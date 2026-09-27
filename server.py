"""Tiny generation API around microsoft/TRELLIS text-to-3D, made for RunPod pods.

GET  /health                     -> {ok, gpu, model_loaded}
POST /generate  (X-Token)        -> {name, prompt, seed, simplify, texture_size} => GLB + 4-view preview PNG
GET  /asset/<name>.glb|.png (X-Token)

Single-flight: one generation at a time (the GPU is fully busy anyway).
"""
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
_pipe = None


def _check(x_token: str) -> None:
    if not TOKEN or x_token != TOKEN:
        raise HTTPException(401, "bad token")


def _get_pipe():
    global _pipe
    if _pipe is None:
        from trellis.pipelines import TrellisTextTo3DPipeline

        p = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
        p.cuda()
        _pipe = p
    return _pipe


@app.get("/health")
def health():
    import torch

    return {
        "ok": True,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": _pipe is not None,
    }


class GenReq(BaseModel):
    name: str
    prompt: str
    seed: int = 0
    simplify: float = 0.95
    texture_size: int = 1024


@app.post("/generate")
def generate(req: GenReq, x_token: str = Header(default="")):
    _check(x_token)
    if not re.fullmatch(r"[a-z0-9_]{1,40}", req.name):
        raise HTTPException(400, "bad name")
    if not _lock.acquire(timeout=2):
        raise HTTPException(409, "busy")
    try:
        t0 = time.time()
        pipe = _get_pipe()
        out = pipe.run(req.prompt, seed=req.seed, formats=["gaussian", "mesh"])
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
        return {"ok": True, "ms": int((time.time() - t0) * 1000), "verts": verts}
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
