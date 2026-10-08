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

BACKEND=hunyuan: Hunyuan3D shape (+ 2.0 paint). shape_model="2.0" (default) or "2.1" picks the shape
DiT per request; SHAPE_MODEL_DEFAULT sets the default for the pod.

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
HUNYUAN_PAINT = os.environ.get("HUNYUAN_PAINT", "1") != "0"
# Shape model (BACKEND=hunyuan only), per request via GenReq.shape_model:
#   "2.0" -> Hunyuan3D-2.0 shape DiT (hy3dgen, tencent/Hunyuan3D-2): the production model.
#   "2.1" -> Hunyuan3D-2.1 shape DiT (hy3dshape, tencent/Hunyuan3D-2.1), under A/B test. Loaded on the
#            first request that asks for it: that request first downloads ~7.4 GB of weights into HF_HOME.
# SHAPE_MODEL_DEFAULT picks the model for requests that don't say (and the one warmed at boot), so a pod can
# be switched wholesale. Texture paint is the 2.0 paint model either way.
SHAPE_MODELS = ("2.0", "2.1")
SHAPE_MODEL_DEFAULT = os.environ.get("SHAPE_MODEL_DEFAULT", "2.1").strip()  # 2.1 since 2026-10-08 (lab: better shapes)
if SHAPE_MODEL_DEFAULT not in SHAPE_MODELS:
    print(f"SHAPE_MODEL_DEFAULT={SHAPE_MODEL_DEFAULT!r} is not one of {SHAPE_MODELS}; using 2.0", flush=True)
    SHAPE_MODEL_DEFAULT = "2.0"
HY21_MODEL = os.environ.get("HY21_MODEL", "tencent/Hunyuan3D-2.1")
# Pinned HF revision of tencent/Hunyuan3D-2.1 (set HY21_REVISION=main to follow the repo).
HY21_REVISION = os.environ.get("HY21_REVISION", "0b94677654c57bb9a6b6845cd7b704ccf551d327")
HY21_SUBFOLDER = "hunyuan3d-dit-v2-1"


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


# "Front view" without "symmetrical": asking for symmetry made the image model mirror whatever the subject
# holds ("a man with a fish" came out holding two identical fish, one per hand, 2026-10-05).
CONCEPT_SUFFIX = (
    ", single subject, perfectly centered, front view facing the camera, neutral standing pose, "
    "full object in frame, limbs and tail clearly separated and visible, 3D render style, "
    "plain light gray studio background, soft even lighting, highly detailed"
)
CONCEPT_NEGATIVE = (
    "cropped, cut off, multiple objects, duplicated limbs, extra tail, collage, side view, "
    "duplicate objects, two of the same object, mirrored copies, identical pair, twin objects, "
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
    "duplicate objects, two of the same object, mirrored copies, identical pair, "
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


def _load_hy20_shape():
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(HUNYUAN_MODEL)


def _load_hy21_shape():
    """Hunyuan3D-2.1 shape pipeline (DiT 3.0B + VAE 0.3B + DINOv2-L 0.3B in one 7.4 GB fp16 checkpoint).

    hy3dshape's own from_pretrained downloads into ~/.cache/hy3dgen (the container disk, lost with the pod),
    so the weights are fetched here into the HF cache (HF_HOME, on the volume) and loaded from that file.
    """
    import torch
    from huggingface_hub import snapshot_download
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    root = snapshot_download(HY21_MODEL, revision=HY21_REVISION, allow_patterns=[f"{HY21_SUBFOLDER}/*"])
    d = Path(root) / HY21_SUBFOLDER
    return Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        str(d / "model.fp16.ckpt"), str(d / "config.yaml"), device="cuda", dtype=torch.float16
    )


# shape_model -> (_pipes key, loader, GB of free VRAM wanted before loading it next to the other one).
# Both keys start with "hy" so _hunyuan_generate's small-card eviction keeps them.
SHAPE_PIPES = {
    "2.0": ("hy_shape", _load_hy20_shape, 8.0),
    "2.1": ("hy21_shape", _load_hy21_shape, 12.0),
}


def _shape_pipe(version: str, name: str = ""):
    """The shape pipeline for `version`, loading it on first use. The other shape model is dropped first on
    small cards, or when the GPU is too full to hold both (big cards normally keep both resident)."""
    import gc

    import torch

    key, load, need_gb = SHAPE_PIPES[version]
    if key not in _pipes:
        if name:
            _stage(name, "loading", 0.5)
        others = [k for v, (k, _, _) in SHAPE_PIPES.items() if v != version and k in _pipes]
        if others:
            gc.collect()
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            if _gpu_total_gb() < 30 or free_gb < need_gb:
                for k in others:
                    del _pipes[k]
                gc.collect()
                torch.cuda.empty_cache()
                print(f"shape {version}: dropped {others} to make room ({free_gb:.1f} GB was free)", flush=True)
        t0 = time.time()
        _pipes[key] = load()
        print(f"shape {version} loaded in {time.time() - t0:.0f}s", flush=True)
    return _pipes[key]


def _birefnet_cut(img):
    """Background removal with BiRefNet (MIT, ZhengPeng7/BiRefNet): returns RGBA. Loaded once, kept on the GPU (~1 GB)."""
    import numpy as np
    import torch
    from torchvision import transforms

    if "hy_birefnet" not in _pipes:  # hy_ prefix: survives the small-card eviction in _hunyuan_generate
        from transformers import AutoModelForImageSegmentation

        m = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        _pipes["hy_birefnet"] = m.eval().to("cuda")
    rgb = img.convert("RGB")
    tf = transforms.Compose(
        [transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    with torch.no_grad():
        pred = _pipes["hy_birefnet"](tf(rgb).unsqueeze(0).to("cuda"))[-1].sigmoid()[0, 0].float().cpu().numpy()
    from PIL import Image

    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(rgb.size, Image.BILINEAR)
    out = rgb.convert("RGBA")
    out.putalpha(mask)
    return out


def _hunyuan_generate(img, req) -> int:
    """Hunyuan3D: image -> shape DiT (2.0 or 2.1, req.shape_model) -> cleanup -> optional 2.0 texture paint.
    Writes <name>.glb then <name>.png (keep that order). Returns vertex count."""
    import gc

    import torch

    # Small cards: evict non-Hunyuan pipelines (shape+paint alone need ~16 GB).
    # Big cards keep the concept model resident too.
    if _gpu_total_gb() < 30:
        for k in [k for k in list(_pipes) if not k.startswith(("hy_", "hy21_"))]:
            del _pipes[k]
        gc.collect()
        torch.cuda.empty_cache()

    from hy3dgen.rembg import BackgroundRemover

    alpha = img.getextrema()[3] if img.mode == "RGBA" else (255, 255)
    if alpha[0] == 255:  # no real transparency: strip the background
        # BiRefNet by default (2026-10-07 lab: same or better masks, and jobs 35-55% faster because it stays
        # loaded, where BackgroundRemover() reloaded u2net every job). bg="u2net" keeps the old path; any
        # BiRefNet failure falls back to it rather than failing the job.
        if req.bg == "u2net":
            img = BackgroundRemover()(img.convert("RGB"))
        else:
            try:
                img = _birefnet_cut(img)
            except Exception as e:
                print(f"birefnet failed, using u2net: {type(e).__name__}: {e}", flush=True)
                img = BackgroundRemover()(img.convert("RGB"))

    # Same inputs and sampler settings for 2.0 and 2.1, so an A/B differs only in the model.
    version = req.shape_model or SHAPE_MODEL_DEFAULT
    shape = _shape_pipe(version, req.name)
    _stage(req.name, "shape", 0.0)
    mesh = shape(
        image=img,
        num_inference_steps=int(req.ss_steps or 30),
        guidance_scale=float(req.ss_cfg or 5.0),
        octree_resolution=int(req.octree_resolution or 380),
        **({"num_chunks": int(req.num_chunks)} if req.num_chunks else {}),
        generator=torch.manual_seed(req.seed),
    )[0]
    if mesh is None:  # marching cubes found no surface (both pipelines return None rather than raise)
        raise RuntimeError(f"Hunyuan {version} shape model produced no surface")

    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

    _stage(req.name, "cleanup", 0.0)
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    # simplify 0.9 -> 40k faces (the repo's own texturing budget), 0.85 -> 60k.
    facenum = max(20000, min(120000, int(400000 * (1.0 - req.simplify))))
    if req.max_faces:  # explicit budget (print exports want more than paint's 40k)
        facenum = max(20000, min(500000, int(req.max_faces)))
    mesh = FaceReducer()(mesh, max_facenum=facenum)

    if HUNYUAN_PAINT and req.paint:
        if "hy_paint" not in _pipes:
            _stage(req.name, "texture", 0.05)
            # The background warm-up may be loading it right now: wait for that rather than load twice.
            if WARM_ON_BOOT and not _paint_ready.is_set():
                _paint_ready.wait(timeout=15 * 60)
            if "hy_paint" not in _pipes:
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


# Explicit photos are refused: the studio's photo check (/cutout) blocks them on upload, and /generate refuses
# them too, for anything that skips the studio. A small ViT classifier (normal / nsfw); scores at or above
# NSFW_BLOCK are blocked.
NSFW_MODEL = os.environ.get("NSFW_MODEL", "Falconsai/nsfw_image_detection")
NSFW_BLOCK = float(os.environ.get("NSFW_BLOCK", "0.7"))
_nsfw = None
_nsfw_lock = threading.Lock()


def _nsfw_score(img) -> float:
    """Probability that the image is sexually explicit (0..1)."""
    global _nsfw
    with _nsfw_lock:
        if _nsfw is None:
            from transformers import pipeline

            _nsfw = pipeline("image-classification", model=NSFW_MODEL, device=-1)
        out = _nsfw(img.convert("RGB"))
    return float(next((o["score"] for o in out if o["label"].lower() == "nsfw"), 0.0))


class ModerateReq(BaseModel):
    image: str


@app.post("/moderate")
def moderate(req: ModerateReq, x_token: str = Header(default="")):
    """NSFW score for one image (used to check uploads already stored)."""
    _check(x_token)
    score = _nsfw_score(_decode_image(req.image))
    return {"nsfw": round(score, 4), "blocked": score >= NSFW_BLOCK}


@app.get("/health")
def health():
    import torch

    return {
        "ok": True,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded": sorted(_pipes.keys()),
        "backend": BACKEND,
        "shape_model_default": SHAPE_MODEL_DEFAULT,
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
    # Shape-quality options under test (unset = today's behaviour).
    octree_resolution: int | None = None  # Hunyuan marching-cubes grid; 380 by default
    num_chunks: int | None = None
    max_faces: int | None = None  # face budget after cleanup, instead of the simplify mapping
    bg: str | None = None  # background removal: BiRefNet by default; "u2net" for the old remover
    shape_model: str | None = None  # Hunyuan shape model, "2.0" | "2.1"; unset = SHAPE_MODEL_DEFAULT ("2.1")


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
    if req.shape_model is not None and req.shape_model not in SHAPE_MODELS:
        raise HTTPException(400, "shape_model must be '2.0' or '2.1'")
    if req.shape_model == "2.1" and BACKEND != "hunyuan":
        raise HTTPException(400, "shape_model='2.1' needs BACKEND=hunyuan")
    if BLOCKED_PROMPT.search(req.prompt):
        raise HTTPException(422, "prompt_blocked: sexual content is not allowed")
    if req.mode == "image" and _nsfw_score(_decode_image(req.image)) >= NSFW_BLOCK:
        raise HTTPException(422, "image_blocked: explicit images are not allowed")
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
            req.shape_model = req.shape_model or SHAPE_MODEL_DEFAULT
            verts = _hunyuan_generate(subject, req)
            _stage(req.name, "done", 1.0)
            return {
                "ok": True,
                "mode": req.mode,
                "backend": "hunyuan",
                "shape_model": req.shape_model,
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


class CutoutReq(BaseModel):
    image: str  # data URL or raw base64


_remover = None
_remover_lock = threading.Lock()


@app.post("/cutout")
def cutout(req: CutoutReq, x_token: str = Header(default="")):
    """What the 3D step will see for an uploaded photo: the same background removal generation uses, plus a
    verdict the studio turns into advice before anyone spends a draft. Runs beside the generation lock (rembg
    is light) so a check never queues behind a model being made.

    verdict: "ok" | "cut_off" (the object runs off the photo's edges) | "background" (nearly the whole photo
    was kept: the background wasn't separated, or the object fills the frame) | "empty" (nothing was found).
    """
    _check(x_token)
    import numpy as np
    from PIL import Image

    global _remover
    t0 = time.time()
    img = _decode_image(req.image)
    img.thumbnail((1024, 1024))
    nsfw = _nsfw_score(img)
    if nsfw >= NSFW_BLOCK:
        # Never echo an explicit image back; the studio refuses the upload.
        return {"verdict": "explicit", "coverage": 0, "edges_touching": [], "cutout": "", "nsfw": round(nsfw, 3)}
    t1 = time.time()
    if img.getextrema()[3][0] < 255:  # already transparent: use its own alpha
        cut = img
    else:
        with _remover_lock:
            if _remover is None:
                # The same rembg model hy3dgen's BackgroundRemover uses (u2net), but pinned to the CPU: the
                # default provider list on this image cost a fixed ~24 s per call.
                from rembg import new_session

                _remover = new_session("u2net", providers=["CPUExecutionProvider"])
            from rembg import remove

            cut = remove(img.convert("RGB"), session=_remover, bgcolor=[255, 255, 255, 0]).convert("RGBA")
    t2 = time.time()
    a = np.asarray(cut.getchannel("A")) > 128
    h, w = a.shape
    coverage = float(a.mean())
    band = max(2, int(min(h, w) * 0.01))  # a thin strip along each edge
    edges = {
        "top": float(a[:band, :].mean()),
        "bottom": float(a[-band:, :].mean()),
        "left": float(a[:, :band].mean()),
        "right": float(a[:, -band:].mean()),
    }
    touching = [k for k, v in edges.items() if v > 0.05]
    if coverage < 0.02:
        verdict = "empty"
    elif coverage > 0.85:
        verdict = "background"
    elif len(touching) >= 2 or any(v > 0.25 for v in edges.values()):
        verdict = "cut_off"
    else:
        verdict = "ok"
    thumb = cut.copy()
    thumb.thumbnail((384, 384))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG", optimize=True)
    return {
        "verdict": verdict,
        "coverage": round(coverage, 3),
        "edges_touching": touching,
        "cutout": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "ms": {"decode": int((t1 - t0) * 1000), "remove": int((t2 - t1) * 1000), "total": int((time.time() - t0) * 1000)},
    }


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


# Set once the paint model is loaded (or its background load has given up), so a textured job that
# arrives while it's still loading waits for it instead of loading a second copy.
_paint_ready = threading.Event()


def _warm_paint() -> None:
    """Load the paint model in the background, outside the generation lock: free drafts don't use it, so
    they run while it loads (a cold start used to block every job for the whole ~3-5 min warm-up)."""
    t0 = time.time()
    try:
        if "hy_paint" not in _pipes:
            _pipes["hy_paint"] = _load_paint()
        print(f"paint ready in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        global _warm_error
        _warm_error = f"{type(e).__name__}: {e}"[:500]
        traceback.print_exc()
        print(f"paint warm-up failed after {time.time() - t0:.0f}s: {e}", flush=True)
    finally:
        _paint_ready.set()


def _warm() -> None:
    with _lock:
        t0 = time.time()
        try:
            big_card = _gpu_total_gb() >= 30
            if BACKEND == "hunyuan":
                # Only the default shape model; the other one loads on the first request that asks for it.
                _shape_pipe(SHAPE_MODEL_DEFAULT)
                try:  # the default background remover, so the first photo job doesn't wait for it
                    from PIL import Image as _Img

                    _birefnet_cut(_Img.new("RGB", (64, 64), "gray"))
                except Exception as e:
                    print(f"birefnet warm-up failed (jobs fall back to u2net): {e}", flush=True)
                if HUNYUAN_PAINT and not big_card:
                    # 24 GB cards: paint swaps with the concept model per job, keep the old order.
                    _pipes["hy_paint"] = _load_paint()
                    _paint_ready.set()
            else:
                _get_pipe("image")
            if TEXT_VIA_IMAGE:
                if big_card:
                    _load_concept(keep=True)  # resident alongside the 3D pipeline
                else:
                    # 24 GB cards swap per job; at least have the weights on disk.
                    from huggingface_hub import snapshot_download

                    snapshot_download(FLUX_MODEL if CONCEPT_MODEL == "flux" else SDXL_MODEL)
            # Upload checks (photo check + explicit-image block) answer in seconds from the first upload.
            try:
                from PIL import Image as _Img

                _nsfw_score(_Img.new("RGB", (64, 64), "gray"))
                global _remover
                with _remover_lock:
                    if _remover is None:
                        from rembg import new_session

                        _remover = new_session("u2net", providers=["CPUExecutionProvider"])
            except Exception as e:
                print(f"upload-check warm-up failed: {e}", flush=True)
            print(f"warm-up done in {time.time() - t0:.0f}s: {sorted(_pipes)} (drafts can run)", flush=True)
        except Exception as e:  # never keep the server from serving
            global _warm_error
            _warm_error = f"{type(e).__name__}: {e}"[:500]
            traceback.print_exc()
            print(f"warm-up failed after {time.time() - t0:.0f}s: {e}", flush=True)
    # Big cards: shape + concept are resident and the lock is free, so drafts run now; paint follows.
    if BACKEND == "hunyuan" and HUNYUAN_PAINT and not _paint_ready.is_set():
        threading.Thread(target=_warm_paint, daemon=True).start()


@app.on_event("startup")
def _start_warm() -> None:
    if WARM_ON_BOOT:
        threading.Thread(target=_warm, daemon=True).start()
