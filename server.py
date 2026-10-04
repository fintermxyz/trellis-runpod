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
_last_used = time.time()  # boot counts as activity: gives a fresh pod a grace period
# Changes on every server start. The worker compares it across dispatches: a different id means the
# server restarted (crash, OOM kill) and any job it was running is gone, so it re-sends at once.
BOOT_ID = os.urandom(6).hex()
# Per-job stage progress, polled by clients via GET /progress/<name>.
# Stages run in order; pct is overall 0-100.
_progress: dict = {}
STAGES = {
    "loading": (0, 5, "Loading models"),
    "concept": (5, 20, "Drawing the concept image"),
    "shape": (20, 55, "Building the 3D shape"),
    "cleanup": (55, 62, "Cleaning up the mesh"),
    "texture": (62, 94, "Painting textures"),
    "export": (94, 100, "Exporting GLB"),
    "done": (100, 100, "Done"),
}


def _stage(name: str, stage: str, frac: float = 0.0) -> None:
    """Record that job `name` is in `stage`, `frac` of the way through it."""
    lo, hi, label = STAGES[stage]
    _progress[name] = {
        "stage": stage,
        "label": label,
        "pct": int(lo + (hi - lo) * max(0.0, min(1.0, frac))),
        "stage_end": hi,
        "updated": time.time(),
    }


def _fail(name: str, err: BaseException) -> None:
    """Keep a job's error on its progress record: the caller's HTTP request has usually been cut by the
    RunPod proxy long before a late failure, so /progress is the only place the worker can read it."""
    p = _progress.get(name, {"stage": "failed", "pct": 0})
    _progress[name] = {**p, "error": f"{type(err).__name__}: {err}"[:500], "failed": True, "updated": time.time()}


# Set when the boot warm-up fails, shown on /health (container logs aren't reachable from the worker).
_warm_error: str | None = None

MODELS = {
    "text": os.environ.get("TEXT_MODEL", "microsoft/TRELLIS-text-xlarge"),
    "image": os.environ.get("IMAGE_MODEL", "microsoft/TRELLIS-image-large"),
}
# Text requests go text -> SDXL concept image -> image-to-3D by default (the
# Tripo architecture; markedly better than direct text-to-3D). Set
# TEXT_VIA_IMAGE=0 to fall back to TRELLIS-text-xlarge directly.
TEXT_VIA_IMAGE = os.environ.get("TEXT_VIA_IMAGE", "1") != "0"
# RealVisXL: ungated SDXL finetune with far better anatomy than base SDXL.
SDXL_MODEL = os.environ.get("SDXL_MODEL", "SG161222/RealVisXL_V5.0")
# Concept-image model. FLUX.1-schnell is the best option but its HF repo is
# gated: it only works when the pod env carries an HF_TOKEN that has accepted
# the FLUX terms (huggingface_hub picks the token up automatically).
CONCEPT_MODEL = os.environ.get("CONCEPT_MODEL", "flux" if os.environ.get("HF_TOKEN") else "sdxl")
FLUX_MODEL = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
# 3D backend: "trellis" (default) or "hunyuan" (Hunyuan3D-2.0, image-conditioned
# only — text prompts go through the SDXL concept stage first). Hunyuan needs
# BACKEND=hunyuan in the pod env so bootstrap installs hy3dgen.
BACKEND = os.environ.get("BACKEND", "trellis")
HUNYUAN_MODEL = os.environ.get("HUNYUAN_MODEL", "tencent/Hunyuan3D-2")
# hy3dgen looks here first (<dir>/<repo>/<subfolder>) and only falls back to downloading whole subfolders
# from the Hub when it's missing. On the pod volume, so a restart doesn't fetch again.
HY3DGEN_DIR = Path(os.environ.setdefault("HY3DGEN_MODELS", "/workspace/hy3dgen"))
HUNYUAN_PAINT = os.environ.get("HUNYUAN_PAINT", "1") != "0"


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


# Sexual content is refused. The web app filters prompts before a job is created; this is the second
# line for anything that reaches the pod directly. The concept negatives below steer SDXL away from
# nudity for prompts that slip past both.
BLOCKED_PROMPT = re.compile(
    r"\b(nsfw|nude|nudes|nudity|naked|topless|bottomless|undress\w*|porn\w*|hentai|xxx|sex|sexy|sexual\w*|"
    r"erotic\w*|fetish\w*|bdsm|bondage|lingerie|genital\w*|penis|penises|cock|cocks|dick|dicks|vagina\w*|"
    r"pussy|pussies|boob|boobs|tits|titties|breasts|nipples?|areola\w*|butt\s*naked|asshole|anus|"
    r"orgasm\w*|masturbat\w*|blowjob|handjob|cum|semen|stripper|strip\s*tease|onlyfans|"
    r"loli|lolicon|shota|shotacon)\b",
    re.IGNORECASE,
)
SAFE_NEGATIVE = "nsfw, nude, naked, nudity, sexual, erotic, exposed breasts, nipples, genitals, underwear, lingerie"


def _gpu_total_gb() -> float:
    import torch

    return torch.cuda.get_device_properties(0).total_memory / 1e9


CONCEPT_SUFFIX = (
    ", single object, perfectly centered, symmetrical front view, neutral standing pose, "
    "full object in frame, limbs and tail clearly separated and visible, 3D render style, "
    "plain light gray studio background, soft even lighting, highly detailed"
)
CONCEPT_NEGATIVE = (
    "cropped, cut off, multiple objects, duplicated limbs, extra tail, collage, side view, "
    "rear view, dynamic pose, motion blur, text, watermark, busy background, scenery, frame, border, human hands"
)

# Long, rigid things (vehicles, furniture, buildings...) need the opposite of
# the creature template: a straight front view shows none of their length, so
# the 3D stage guesses the depth and the model comes out squat (a bus as deep
# as it is wide). A raised three-quarter view shows front and side together.
OBJECT_SUFFIX = (
    ", single object, three-quarter view from slightly above showing the front and one full side, "
    "true real-world proportions and full length visible, entire object in frame with margin, "
    "3D render style, plain light gray studio background, soft even lighting, highly detailed"
)
OBJECT_NEGATIVE = (
    "cropped, cut off, multiple objects, front view only, head-on view, flat orthographic view, "
    "squashed, stubby, chibi, toy-like proportions, foreshortened, distorted proportions, fisheye, "
    "motion blur, text, watermark, busy background, scenery, road, frame, border, human hands"
)
OBJECT_WORDS = re.compile(
    r"\b(bus|buses|car|cars|truck|lorry|van|tractor|train|tram|locomotive|carriage|wagon|cart|"
    r"bike|bicycle|motorbike|motorcycle|scooter|boat|ship|yacht|canoe|kayak|submarine|plane|"
    r"aeroplane|airplane|aircraft|jet|helicopter|rocket|spaceship|tank|taxi|cab|limo|limousine|"
    r"ambulance|jeep|suv|forklift|excavator|digger|bulldozer|crane|caravan|trailer|sled|sleigh|"
    r"skateboard|surfboard|sofa|couch|bench|table|desk|bed|cabinet|wardrobe|dresser|bookshelf|"
    r"shelf|piano|guitar|violin|house|building|cottage|castle|tower|bridge|barn|shed|church|"
    r"sword|rifle|gun|keyboard|laptop|bottle|shoe|boot|chair)\b",
    re.IGNORECASE,
)


def _concept_template(prompt: str) -> tuple[str, str]:
    """(suffix, negative) for the prompt: the raised three-quarter view for long,
    rigid objects, the symmetric front view for creatures and everything else."""
    return (OBJECT_SUFFIX, OBJECT_NEGATIVE) if OBJECT_WORDS.search(prompt) else (CONCEPT_SUFFIX, CONCEPT_NEGATIVE)


def _load_concept(keep: bool):
    """The concept (text -> image) diffusion pipeline, cached in _pipes when `keep`."""
    import torch

    key = f"concept_{CONCEPT_MODEL}"
    pipe = _pipes.get(key)
    if pipe is None:
        from diffusers import DiffusionPipeline

        if CONCEPT_MODEL == "flux":
            pipe = DiffusionPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16).to("cuda")
        else:
            # Finetune repos often ship no fp16 variant files; fall back cleanly.
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    SDXL_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                ).to("cuda")
            except Exception:
                pipe = DiffusionPipeline.from_pretrained(
                    SDXL_MODEL, torch_dtype=torch.float16, use_safetensors=True
                ).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        if keep:
            _pipes[key] = pipe
    return pipe


def _concept_image(prompt: str, seed: int, name: str = ""):
    """Concept render of the prompt: one centered object on a plain backdrop.
    Creatures get a symmetric front view (the 3D stage sees ONE view and
    hallucinates the back, so an ambiguous pose becomes mirrored limbs and
    doubled tails); long, rigid objects get a raised three-quarter view so
    their length is visible (see _concept_template). FLUX.1-schnell by default (far better anatomy
    than SDXL base); the 3D stage's rembg strips the background after."""
    import gc

    import torch

    # Small (24 GB) cards can't hold the concept model next to a 3D pipeline;
    # big cards keep everything resident and skip the reload tax.
    small_card = _gpu_total_gb() < 30
    if small_card:
        _pipes.clear()
        gc.collect()
        torch.cuda.empty_cache()

    pipe = _load_concept(keep=not small_card)
    total_steps = 4 if CONCEPT_MODEL == "flux" else 30

    def on_step(_pipe, step, _t, kwargs):
        if name:
            _stage(name, "concept", (step + 1) / total_steps)
        return kwargs

    suffix, negative = _concept_template(prompt)
    negative = f"{negative}, {SAFE_NEGATIVE}"
    if name:
        _stage(name, "concept", 0.0)
    try:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        if CONCEPT_MODEL == "flux":
            # schnell is distilled: 4 steps, guidance 0, no negative prompt.
            img = pipe(
                prompt=f"{prompt}{suffix}",
                num_inference_steps=4,
                guidance_scale=0.0,
                width=1024,
                height=1024,
                generator=gen,
                callback_on_step_end=on_step,
            ).images[0]
        else:
            img = pipe(
                prompt=f"{prompt}{suffix}",
                negative_prompt=negative,
                num_inference_steps=30,
                guidance_scale=7.0,
                width=1024,
                height=1024,
                generator=gen,
                callback_on_step_end=on_step,
            ).images[0]
    finally:
        if small_card:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    return img


_weights_ready = False


def _fetch_used_weights() -> None:
    """Download only the Hunyuan files the pipelines actually load into HY3DGEN_DIR.

    Left to itself hy3dgen downloads whole subfolders: the shape model's folder holds the same 4.9 GB
    weights five times over (.ckpt, fp16 .ckpt, .safetensors, ...) = 24.6 GB, and the paint folders keep
    every UNet/VAE as both .bin and .safetensors. Fetching just config.yaml + model.fp16.safetensors for
    shape, and one format per paint/delight component, cuts a first boot from ~50 GB to ~20 GB.
    """
    global _weights_ready
    if _weights_ready:
        return
    import inspect

    from huggingface_hub import hf_hub_download, list_repo_files
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    # The paint subfolder hy3dgen will ask for (turbo in current versions).
    sig = inspect.signature(Hunyuan3DPaintPipeline.from_pretrained)
    paint = sig.parameters["subfolder"].default if "subfolder" in sig.parameters else "hunyuan3d-paint-v2-0"
    files = list_repo_files(HUNYUAN_MODEL)
    want = ["hunyuan3d-dit-v2-0/config.yaml", "hunyuan3d-dit-v2-0/model.fp16.safetensors"]
    if HUNYUAN_PAINT:
        for folder in (paint, "hunyuan3d-delight-v2-0"):
            in_folder = [f for f in files if f.startswith(folder + "/")]
            # diffusers prefers .safetensors: drop the .bin twin wherever a component has one.
            has_safe = {os.path.dirname(f) for f in in_folder if f.endswith(".safetensors")}
            want += [f for f in in_folder if not (f.endswith(".bin") and os.path.dirname(f) in has_safe)]
    dest = HY3DGEN_DIR / HUNYUAN_MODEL
    t0 = time.time()
    for f in want:
        if not (dest / f).exists():
            hf_hub_download(HUNYUAN_MODEL, f, local_dir=str(dest))
    print(f"weights ready in {time.time() - t0:.0f}s: {len(want)} files under {dest}", flush=True)
    _weights_ready = True

def _load_paint():
    """Hunyuan3DPaintPipeline, working around a first-boot import failure.

    The paint model's multiview UNet ships its own Python file (hunyuan3d-paint-v2-0/unet/modules.py).
    diffusers copies it into $HF_HOME/modules/diffusers_modules/local/ and imports it from there. On a
    fresh volume that package doesn't exist when the server starts, Python's import system caches the
    miss, and the import fails with "No module named 'diffusers_modules.local.modules'" for the rest of
    the process (only a restart used to fix it). Creating the package up front, and retrying once with the
    import caches cleared, makes the first boot work.
    """
    import importlib
    import sys

    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    def prepare() -> None:
        from diffusers.utils.constants import HF_MODULES_CACHE
        from diffusers.utils.dynamic_modules_utils import init_hf_modules

        init_hf_modules()  # creates HF_MODULES_CACHE with __init__.py and puts it on sys.path
        local = Path(HF_MODULES_CACHE) / "diffusers_modules" / "local"
        local.mkdir(parents=True, exist_ok=True)
        for pkg in (local.parent, local):
            (pkg / "__init__.py").touch(exist_ok=True)
        for name in [m for m in sys.modules if m.startswith("diffusers_modules")]:
            del sys.modules[name]
        importlib.invalidate_caches()

    _fetch_used_weights()
    prepare()
    try:
        return Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN_MODEL)
    except Exception as e:
        # hy3dgen re-raises the ModuleNotFoundError as a generic "Something wrong while loading" error.
        chain, err = [], e
        while err is not None and len(chain) < 5:
            chain.append(repr(err))
            err = err.__cause__ or err.__context__
        if not any("diffusers_modules" in c for c in chain):
            raise
        print(f"paint load hit the dynamic-module import miss, retrying: {e}", flush=True)
        prepare()  # modules.py has been copied by now; clear the cached miss and import again
        return Hunyuan3DPaintPipeline.from_pretrained(HUNYUAN_MODEL)


def _hunyuan_generate(img, req) -> int:
    """Hunyuan3D-2.0: image -> shape DiT -> cleanup -> optional texture paint.
    Writes <name>.glb then <name>.png (keep that order). Returns vertex count."""
    import gc

    import torch

    # Small cards: evict non-Hunyuan pipelines (shape+paint alone need ~16 GB).
    # Big cards keep the concept model resident too.
    if _gpu_total_gb() < 30:
        for k in [k for k in list(_pipes) if not k.startswith("hy_")]:
            del _pipes[k]
        gc.collect()
        torch.cuda.empty_cache()

    from hy3dgen.rembg import BackgroundRemover

    alpha = img.getextrema()[3] if img.mode == "RGBA" else (255, 255)
    if alpha[0] == 255:  # no real transparency: strip the background
        img = BackgroundRemover()(img.convert("RGB"))

    if "hy_shape" not in _pipes:
        _stage(req.name, "loading", 0.5)
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        _fetch_used_weights()
        _pipes["hy_shape"] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(HUNYUAN_MODEL)
    _stage(req.name, "shape", 0.0)
    mesh = _pipes["hy_shape"](
        image=img,
        num_inference_steps=int(req.ss_steps or 30),
        guidance_scale=float(req.ss_cfg or 5.0),
        octree_resolution=380,
        generator=torch.manual_seed(req.seed),
    )[0]

    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

    _stage(req.name, "cleanup", 0.0)
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    # simplify 0.9 -> 40k faces (the repo's own texturing budget), 0.85 -> 60k.
    facenum = max(20000, min(120000, int(400000 * (1.0 - req.simplify))))
    mesh = FaceReducer()(mesh, max_facenum=facenum)

    if HUNYUAN_PAINT and req.paint:
        if "hy_paint" not in _pipes:
            _stage(req.name, "texture", 0.05)
            _pipes["hy_paint"] = _load_paint()
        _stage(req.name, "texture", 0.1)
        mesh = _pipes["hy_paint"](mesh, image=img)

    _stage(req.name, "export", 0.0)
    mesh.export(str(OUT / f"{req.name}.glb"))
    grid = _preview_grid(mesh)
    import imageio

    imageio.imwrite(str(OUT / f"{req.name}.png"), grid)
    return int(mesh.vertices.shape[0])


def _preview_grid(mesh):
    """2x2 turntable of a trimesh via pyrender EGL (the repo ships no renderer)."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import numpy as np

    if not hasattr(np, "infty"):
        np.infty = np.inf  # pyrender 0.1.45 predates numpy 2
    import pyrender

    m = mesh.copy()
    m.apply_translation(-m.bounding_box.centroid)
    m.apply_scale(1.0 / max(float(max(m.extents)), 1e-6))

    frames = []
    for yaw in (0, 90, 180, 270):
        scene = pyrender.Scene(bg_color=[0, 0, 0, 255], ambient_light=[0.35, 0.35, 0.35])
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        theta, elev, r = np.deg2rad(yaw), np.deg2rad(20), 1.9
        eye = np.array([r * np.cos(elev) * np.sin(theta), r * np.sin(elev), r * np.cos(elev) * np.cos(theta)])
        z = eye / np.linalg.norm(eye)
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
        scene.add(pyrender.PerspectiveCamera(yfov=float(np.deg2rad(40))), pose=pose)
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
        renderer = pyrender.OffscreenRenderer(512, 512)
        color, _ = renderer.render(scene)
        renderer.delete()
        frames.append(color)
    return np.concatenate(
        [np.concatenate(frames[0:2], axis=1), np.concatenate(frames[2:4], axis=1)], axis=0
    )


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
        "backend": BACKEND,
        "concept_model": CONCEPT_MODEL,
        "text_via_image": TEXT_VIA_IMAGE,
        "model_loaded": bool(_pipes),  # back-compat
        # Seconds since the last /generate (or boot). An external janitor can
        # stop the pod when this grows large — the pod is the one place that
        # sees traffic from every client, so idle is judged here.
        "idle_s": int(time.time() - _last_used),
        "boot_id": BOOT_ID,
        "warm_error": _warm_error,
        "busy": _lock.locked(),
    }


class GenReq(BaseModel):
    name: str
    mode: str = "text"  # "text" | "image"
    prompt: str = ""
    image: str | None = None  # data URL or raw base64, mode="image"
    seed: int = 0
    simplify: float = 0.9
    texture_size: int = 1024
    # Draft tier: skip the texture-paint pass (fast, untextured preview).
    paint: bool = True
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
    if BLOCKED_PROMPT.search(req.prompt):
        raise HTTPException(422, "prompt_blocked: sexual content is not allowed")
    if not _lock.acquire(timeout=2):
        raise HTTPException(409, "busy")
    global _last_used
    _last_used = time.time()
    try:
        t0 = time.time()
        skip_early_load = BACKEND == "hunyuan" or (req.mode == "text" and TEXT_VIA_IMAGE)
        pipe = None if skip_early_load else _get_pipe(req.mode)

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

        if req.mode == "image":
            subject = _decode_image(req.image)
        elif TEXT_VIA_IMAGE:
            # Tripo-style: render a concept image first, then lift it to 3D.
            concept = _concept_image(req.prompt, req.seed, req.name)
            concept.save(str(OUT / f"{req.name}_src.png"))
            subject = concept.convert("RGBA")
        else:
            if BACKEND == "hunyuan":
                raise HTTPException(400, "hunyuan backend is image-conditioned; enable TEXT_VIA_IMAGE")
            subject = req.prompt

        if BACKEND == "hunyuan":
            verts = _hunyuan_generate(subject, req)
            _stage(req.name, "done", 1.0)
            return {
                "ok": True,
                "mode": req.mode,
                "backend": "hunyuan",
                "ms": int((time.time() - t0) * 1000),
                "verts": verts,
            }

        if pipe is None:
            pipe = _get_pipe("image")
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
        _fail(req.name, e)
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        _last_used = time.time()
        _lock.release()


@app.get("/progress/{name}")
def progress(name: str, x_token: str = Header(default="")):
    _check(x_token)
    p = _progress.get(name)
    if p is None:
        # Not started here: either waiting behind another job or unknown.
        return {"stage": "waiting" if _lock.locked() else "unknown", "pct": 0, "busy": _lock.locked()}
    return {**p, "busy": _lock.locked(), "age_s": int(time.time() - p["updated"])}


@app.get("/asset/{fname}")
def asset(fname: str, x_token: str = Header(default="")):
    _check(x_token)
    if not re.fullmatch(r"[a-z0-9_]{1,44}\.(glb|png)", fname):
        raise HTTPException(400, "bad file")
    p = OUT / fname
    if not p.exists():
        raise HTTPException(404, "missing")
    return FileResponse(str(p))


# --- Warm-up on boot --------------------------------------------------------
# A cold pod used to download and load every model on the first job, so that
# visitor waited minutes. Now the server does it in the background as soon as
# it starts; /generate answers 409 "busy" meanwhile, which the worker retries.
WARM_ON_BOOT = os.environ.get("WARM_ON_BOOT", "1") != "0"


def _warm() -> None:
    with _lock:
        t0 = time.time()
        try:
            big_card = _gpu_total_gb() >= 30
            if BACKEND == "hunyuan":
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

                _fetch_used_weights()
                _pipes["hy_shape"] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(HUNYUAN_MODEL)
                if HUNYUAN_PAINT:
                    _pipes["hy_paint"] = _load_paint()
            else:
                _get_pipe("image")
            if TEXT_VIA_IMAGE:
                if big_card:
                    _load_concept(keep=True)  # resident alongside the 3D pipeline
                else:
                    # 24 GB cards swap per job; at least have the weights on disk.
                    from huggingface_hub import snapshot_download

                    snapshot_download(FLUX_MODEL if CONCEPT_MODEL == "flux" else SDXL_MODEL)
            print(f"warm-up done in {time.time() - t0:.0f}s: {sorted(_pipes)}", flush=True)
        except Exception as e:  # never keep the server from serving
            global _warm_error
            _warm_error = f"{type(e).__name__}: {e}"[:500]
            traceback.print_exc()
            print(f"warm-up failed after {time.time() - t0:.0f}s: {e}", flush=True)


@app.on_event("startup")
def _start_warm() -> None:
    if WARM_ON_BOOT:
        threading.Thread(target=_warm, daemon=True).start()
