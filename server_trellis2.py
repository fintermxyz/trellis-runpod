"""Generation API around microsoft/TRELLIS.2 (image -> 3D, 4B), for A/B tests against server.py (Hunyuan3D-2.0).

Same HTTP surface as server.py, so the worker and the shape lab talk to it unchanged:

GET  /health                     -> {ok, gpu, loaded, backend: "trellis2", busy, boot_id, warm_error, ...}
POST /generate  (X-Token)        -> {name, mode, prompt|image, seed, simplify, texture_size, paint, ...}
                                    => <name>.glb, then <name>.png (2x2 preview grid)
GET  /progress/<name> (X-Token)  -> {stage, label, pct, ...}; same stages as server.py
GET  /asset/<name>.glb|.png (X-Token)
POST /cutout, /moderate (X-Token) -> unchanged from server.py

TRELLIS.2 specifics:
  - paint=false -> shape only: the texture latent is never sampled; the GLB carries no material.
  - paint=true  -> PBR-textured GLB baked by o_voxel.postprocess.to_glb (texture_size from the request).
  - pipeline_type: "512" | "1024" | "1024_cascade" (default, T2_PIPELINE_TYPE) | "1536_cascade".
  - watertight=true -> one closed, manifold, outward-facing shell for print exports (voxel flood-fill remesh;
    thin sheets become >= 1 voxel thick). Geometry only: a watertight request is never textured.
  - Background removal is BiRefNet (ZhengPeng7/BiRefNet, MIT). The pipeline config names briaai/RMBG-2.0
    (non-commercial); its loader is replaced before from_pretrained, so RMBG-2.0 is never downloaded or run.
  - The image-cond model facebook/dinov3-vitl16-pretrain-lvd1689m is gated: the pod env needs HF_TOKEN
    from an account that accepted its licence. The DINOv3 licence bans weapons uses, so firearm prompts are
    refused (422 prompt_blocked) alongside the sexual-content filter.
  - Text mode renders a concept image (SDXL by default) and lifts it to 3D, as server.py does.

Single-flight: one generation at a time. The GLB is written before the preview PNG: clients may treat the
PNG's existence as "the GLB is complete"; keep that ordering.
"""
import base64
import io
import os
import re
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

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
# Changes on every server start: the worker re-sends a running job when it sees a new boot id.
BOOT_ID = os.urandom(6).hex()
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
    if not name:
        return
    lo, hi, label = STAGES[stage]
    _progress[name] = {
        "stage": stage,
        "label": label,
        "pct": int(lo + (hi - lo) * max(0.0, min(1.0, frac))),
        "stage_end": hi,
        "updated": time.time(),
    }


def _fail(name: str, err: BaseException) -> None:
    """Keep a job's error on its progress record (the proxy usually cut the HTTP request long before)."""
    p = _progress.get(name, {"stage": "failed", "pct": 0})
    _progress[name] = {**p, "error": f"{type(err).__name__}: {err}"[:500], "failed": True, "updated": time.time()}


_warm_error: str | None = None

BACKEND = "trellis2"
T2_MODEL = os.environ.get("T2_MODEL", "microsoft/TRELLIS.2-4B")
PIPELINE_TYPES = ("512", "1024", "1024_cascade", "1536_cascade")
T2_PIPELINE_TYPE = os.environ.get("T2_PIPELINE_TYPE", "1024_cascade")
# "auto": keep every model on the GPU on cards with >= 40 GB, else let the pipeline swap them per stage.
T2_LOW_VRAM = os.environ.get("T2_LOW_VRAM", "auto")
T2_WATERTIGHT_RES = int(os.environ.get("T2_WATERTIGHT_RES", "512"))
BIREFNET_MODEL = os.environ.get("BIREFNET_MODEL", "ZhengPeng7/BiRefNet")
TEXT_VIA_IMAGE = os.environ.get("TEXT_VIA_IMAGE", "1") != "0"
SDXL_MODEL = os.environ.get("SDXL_MODEL", "SG161222/RealVisXL_V5.0")
# SDXL by default even when HF_TOKEN is set: this image needs HF_TOKEN for DINOv3, and that token
# may not have accepted the (gated) FLUX.1-schnell terms. CONCEPT_MODEL=flux opts in.
CONCEPT_MODEL = os.environ.get("CONCEPT_MODEL", "sdxl")
FLUX_MODEL = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")


def _check(x_token: str) -> None:
    if not TOKEN or x_token != TOKEN:
        raise HTTPException(401, "bad token")


def _gpu_total_gb() -> float:
    import torch

    return torch.cuda.get_device_properties(0).total_memory / 1e9


def _low_vram() -> bool:
    if T2_LOW_VRAM in ("0", "1"):
        return T2_LOW_VRAM == "1"
    return _gpu_total_gb() < 40


# --- Content filters -----------------------------------------------------------------------------------
# Sexual content is refused (same list as server.py): the web app filters first; this is the second line.
BLOCKED_PROMPT = re.compile(
    r"\b(nsfw|nude|nudes|nudity|naked|topless|bottomless|undress\w*|porn\w*|hentai|xxx|sex|sexy|sexual\w*|"
    r"erotic\w*|fetish\w*|bdsm|bondage|lingerie|genital\w*|penis|penises|cock|cocks|dick|dicks|vagina\w*|"
    r"pussy|pussies|boob|boobs|tits|titties|breasts|nipples?|areola\w*|butt\s*naked|asshole|anus|"
    r"orgasm\w*|masturbat\w*|blowjob|handjob|cum|semen|stripper|strip\s*tease|onlyfans|"
    r"loli|lolicon|shota|shotacon)\b",
    re.IGNORECASE,
)
# Guns and firearms are refused: the DINOv3 licence (TRELLIS.2's image encoder) prohibits weapons uses.
# Toy, water and replica guns are refused too (a printed replica is still a gun-shaped object). Tools that
# are called guns (glue gun, nail gun...) are allowed: they're stripped before the check.
TOOL_GUNS = re.compile(
    r"\b(hot\s*glue|glue|heat|staple|nail|spray|paint|caulk\w*|massage|grease|soldering|price|label|tape|"
    r"foam|rivet|solder|sealant|silicone|tattoo|radar|speed)\s*-?\s*guns?\b|\bpistol\s*shrimps?\b",
    re.IGNORECASE,
)
BLOCKED_WEAPONS = re.compile(
    r"\b(guns?|gunman|gunmen|gunslingers?|gunfighters?|handguns?|pistols?|revolvers?|rifles?|shotguns?|firearms?|carbines?|muskets?|"
    r"blunderbuss\w*|derringers?|flintlocks?|matchlocks?|machine\s*-?\s*guns?|submachine\s*-?\s*guns?|smgs?|"
    r"assault\s+weapons?|snipers?|ar\s*-?\s*15s?|ak\s*-?\s*47s?|ak\s*-?\s*74s?|m\s*-?\s*16s?|m4a1|mp5|"
    r"uzis?|glocks?|berettas?|kalashnikovs?|lugers?|colt\s*45|tommy\s*guns?|gatling\w*|miniguns?|"
    r"grenade\s*launchers?|rocket\s*launchers?|bazookas?|ammo|ammunition|silencers?|suppressors?|"
    r"gunstocks?|gun\s*barrels?|ghost\s*guns?|3d\s*printed\s*guns?|liberator\s*pistols?)\b",
    re.IGNORECASE,
)
SAFE_NEGATIVE = "nsfw, nude, naked, nudity, sexual, erotic, exposed breasts, nipples, genitals, underwear, lingerie"
WEAPON_NEGATIVE = "gun, firearm, pistol, rifle, weapon"


def _prompt_block_reason(prompt: str) -> str | None:
    if BLOCKED_PROMPT.search(prompt):
        return "prompt_blocked: sexual content is not allowed"
    if BLOCKED_WEAPONS.search(TOOL_GUNS.sub(" ", prompt)):
        return "prompt_blocked: guns and firearms are not allowed"
    return None


# --- Concept image (text mode), as in server.py -----------------------------------------------------------
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
    r"sword|keyboard|laptop|bottle|shoe|boot|chair)\b",
    re.IGNORECASE,
)


def _concept_template(prompt: str) -> tuple[str, str]:
    return (OBJECT_SUFFIX, OBJECT_NEGATIVE) if OBJECT_WORDS.search(prompt) else (CONCEPT_SUFFIX, CONCEPT_NEGATIVE)


def _load_concept(keep: bool):
    import torch

    key = f"concept_{CONCEPT_MODEL}"
    pipe = _pipes.get(key)
    if pipe is None:
        from diffusers import DiffusionPipeline

        if CONCEPT_MODEL == "flux":
            pipe = DiffusionPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16).to("cuda")
        else:
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    SDXL_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                ).to("cuda")
            except Exception:
                pipe = DiffusionPipeline.from_pretrained(SDXL_MODEL, torch_dtype=torch.float16, use_safetensors=True).to(
                    "cuda"
                )
        pipe.set_progress_bar_config(disable=True)
        if keep:
            _pipes[key] = pipe
    return pipe


def _concept_image(prompt: str, seed: int, name: str = ""):
    """Concept render of the prompt: one centered object on a plain backdrop (see server.py)."""
    import gc

    import torch

    # < 30 GB cards: load the concept model for this job only. TRELLIS.2 itself runs in low-VRAM mode there
    # (its models wait on the CPU), so it doesn't need evicting.
    small_card = _gpu_total_gb() < 30
    pipe = _load_concept(keep=not small_card)
    total_steps = 4 if CONCEPT_MODEL == "flux" else 30

    def on_step(_pipe, step, _t, kwargs):
        _stage(name, "concept", (step + 1) / total_steps)
        return kwargs

    suffix, negative = _concept_template(prompt)
    negative = f"{negative}, {SAFE_NEGATIVE}, {WEAPON_NEGATIVE}"
    _stage(name, "concept", 0.0)
    try:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        if CONCEPT_MODEL == "flux":
            img = pipe(
                prompt=f"{prompt}{suffix}", num_inference_steps=4, guidance_scale=0.0, width=1024, height=1024,
                generator=gen, callback_on_step_end=on_step,
            ).images[0]
        else:
            img = pipe(
                prompt=f"{prompt}{suffix}", negative_prompt=negative, num_inference_steps=30, guidance_scale=7.0,
                width=1024, height=1024, generator=gen, callback_on_step_end=on_step,
            ).images[0]
    finally:
        if small_card:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    return img


# --- Background removal ---------------------------------------------------------------------------------
def _birefnet_cut(img):
    """Background removal with BiRefNet (MIT, ZhengPeng7/BiRefNet): returns RGBA. Loaded once, kept on the GPU."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    if "birefnet" not in _pipes:
        from transformers import AutoModelForImageSegmentation

        m = AutoModelForImageSegmentation.from_pretrained(BIREFNET_MODEL, trust_remote_code=True)
        _pipes["birefnet"] = m.eval().to("cuda")
    rgb = img.convert("RGB")
    tf = transforms.Compose(
        [transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    with torch.no_grad():
        pred = _pipes["birefnet"](tf(rgb).unsqueeze(0).to("cuda"))[-1].sigmoid()[0, 0].float().cpu().numpy()
    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(rgb.size, Image.BILINEAR)
    out = rgb.convert("RGBA")
    out.putalpha(mask)
    return out


class _BiRefNetForPipeline:
    """Stands in for trellis2.pipelines.rembg.BiRefNet. pipeline.json asks for briaai/RMBG-2.0 (CC BY-NC);
    this ignores the requested model and cuts with the server's shared BiRefNet instead. The server passes
    pre-cut RGBA anyway, so the pipeline only calls this for a photo whose cut came back fully opaque."""

    def __init__(self, model_name: str = "", **_kw):
        self.requested = model_name

    def to(self, device):
        pass

    def cuda(self):
        pass

    def cpu(self):
        pass

    def __call__(self, image):
        return _birefnet_cut(image)


def _u2net_cut(img):
    from rembg import remove

    global _remover
    with _remover_lock:
        if _remover is None:
            from rembg import new_session

            _remover = new_session("u2net", providers=["CPUExecutionProvider"])
        return remove(img.convert("RGB"), session=_remover, bgcolor=[255, 255, 255, 0]).convert("RGBA")


# --- TRELLIS.2 pipeline ---------------------------------------------------------------------------------
def _get_t2():
    if "t2" not in _pipes:
        import trellis2.pipelines.rembg as t2_rembg

        # Must happen before from_pretrained: it instantiates getattr(rembg, "BiRefNet")(model_name=...).
        t2_rembg.BiRefNet = _BiRefNetForPipeline
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        try:
            p = Trellis2ImageTo3DPipeline.from_pretrained(T2_MODEL)
        except Exception as e:
            if any(s in f"{type(e).__name__} {e}" for s in ("Gated", "gated", "401", "403", "dinov3")):
                raise RuntimeError(
                    "TRELLIS.2 load failed; facebook/dinov3-vitl16-pretrain-lvd1689m is gated: set HF_TOKEN in the "
                    f"pod env to a token whose account accepted the DINOv3 licence ({type(e).__name__}: {e})"
                ) from e
            raise
        p.low_vram = _low_vram()
        p.cuda()  # low_vram: only records the device; models move per stage
        _pipes["t2"] = p
    return _pipes["t2"]


class _StepProgress:
    """Counts sampler steps (sampler.sample_once calls) into a stage's progress fraction."""

    def __init__(self, pipe, name: str, stage: str, lo: float, hi: float, total: int):
        self.samplers = [pipe.sparse_structure_sampler, pipe.shape_slat_sampler, pipe.tex_slat_sampler]
        self.name, self.stage, self.lo, self.hi, self.total, self.done = name, stage, lo, hi, max(1, total), 0

    def retarget(self, stage: str, lo: float, hi: float, total: int):
        self.stage, self.lo, self.hi, self.total, self.done = stage, lo, hi, max(1, total), 0

    def __enter__(self):
        for s in {id(s): s for s in self.samplers}.values():
            orig = type(s).sample_once.__get__(s)

            def wrapped(*a, _orig=orig, **k):
                r = _orig(*a, **k)
                self.done += 1
                _stage(self.name, self.stage, self.lo + (self.hi - self.lo) * min(1.0, self.done / self.total))
                return r

            s.sample_once = wrapped
        return self

    def __exit__(self, *exc):
        for s in {id(s): s for s in self.samplers}.values():
            s.__dict__.pop("sample_once", None)


def _t2_run(pipe, image_rgba, req, paint: bool):
    """Pipeline.run() split into stages (so shape-only jobs never sample the texture latent, and progress is
    reported per sampler step). Returns (mesh, resolution); mesh is a MeshWithVoxel when painted."""
    import torch
    from trellis2.representations import MeshWithVoxel

    ptype = req.pipeline_type or T2_PIPELINE_TYPE
    ss_p, shape_p, tex_p = {}, {}, {}
    if req.ss_steps:
        ss_p["steps"] = int(req.ss_steps)
    if req.ss_cfg:
        ss_p["guidance_strength"] = float(req.ss_cfg)
    if req.slat_steps:
        shape_p["steps"] = int(req.slat_steps)
    if req.slat_cfg:
        shape_p["guidance_strength"] = float(req.slat_cfg)
    if req.tex_steps:
        tex_p["steps"] = int(req.tex_steps)
    if req.tex_cfg:
        tex_p["guidance_strength"] = float(req.tex_cfg)
    n_ss = ss_p.get("steps", pipe.sparse_structure_sampler_params["steps"])
    n_shape = shape_p.get("steps", pipe.shape_slat_sampler_params["steps"]) * (2 if "cascade" in ptype else 1)
    n_tex = tex_p.get("steps", pipe.tex_slat_sampler_params["steps"])
    max_tokens = int(req.max_num_tokens or 49152)
    m = pipe.models

    with torch.no_grad(), _StepProgress(pipe, req.name, "shape", 0.0, 0.3, n_ss) as prog:
        image = pipe.preprocess_image(image_rgba)
        torch.manual_seed(req.seed)
        cond_512 = pipe.get_cond([image], 512)
        cond_1024 = pipe.get_cond([image], 1024) if ptype != "512" else None
        ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[ptype]
        coords = pipe.sample_sparse_structure(cond_512, ss_res, 1, ss_p)
        prog.retarget("shape", 0.3, 0.9, n_shape)
        if ptype == "512":
            shape_slat, res = pipe.sample_shape_slat(cond_512, m["shape_slat_flow_model_512"], coords, shape_p), 512
            tex_cond, tex_model = cond_512, m["tex_slat_flow_model_512"]
        elif ptype == "1024":
            shape_slat, res = pipe.sample_shape_slat(cond_1024, m["shape_slat_flow_model_1024"], coords, shape_p), 1024
            tex_cond, tex_model = cond_1024, m["tex_slat_flow_model_1024"]
        else:
            shape_slat, res = pipe.sample_shape_slat_cascade(
                cond_512, cond_1024, m["shape_slat_flow_model_512"], m["shape_slat_flow_model_1024"],
                512, 1536 if ptype == "1536_cascade" else 1024, coords, shape_p, max_tokens,
            )
            tex_cond, tex_model = cond_1024, m["tex_slat_flow_model_1024"]
        _stage(req.name, "shape", 0.92)
        torch.cuda.empty_cache()
        meshes, subs = pipe.decode_shape_slat(shape_slat, res)
        mesh = meshes[0]
        mesh.fill_holes()
        if not paint:
            del subs
            return mesh, res

        prog.retarget("texture", 0.0, 0.8, n_tex)
        _stage(req.name, "texture", 0.0)
        tex_slat = pipe.sample_tex_slat(tex_cond, tex_model, shape_slat, tex_p)
        _stage(req.name, "texture", 0.85)
        v = pipe.decode_tex_slat(tex_slat, subs)[0]
        return (
            MeshWithVoxel(
                mesh.vertices, mesh.faces, origin=[-0.5, -0.5, -0.5], voxel_size=1 / res, coords=v.coords[:, 1:],
                attrs=v.feats, voxel_shape=torch.Size([*v.shape, *v.spatial_shape]), layout=pipe.pbr_attr_layout,
            ),
            res,
        )


def _face_budget(req) -> int:
    """Same mapping as server.py: simplify 0.9 -> 40k faces, 0.85 -> 60k; max_faces overrides (print)."""
    if req.max_faces:
        return max(20000, min(1000000, int(req.max_faces)))
    return max(20000, min(120000, int(400000 * (1.0 - req.simplify))))


def _to_y_up(vertices):
    """TRELLIS.2 meshes are Z-up; GLB is Y-up (the same swap o_voxel.postprocess.to_glb applies)."""
    import numpy as np

    v = np.array(vertices, dtype=np.float32, copy=True)
    v[:, 1], v[:, 2] = v[:, 2].copy(), -v[:, 1].copy()
    return v


def _clean_shape(vertices, faces, target: int):
    """Shape-only path: to_glb's non-remesh cleanup (simplify, dedupe, non-manifold repair, small parts,
    small holes, consistent winding), without UV unwrapping or baking."""
    import cumesh

    cm = cumesh.CuMesh()
    cm.init(vertices.cuda().float().contiguous(), faces.cuda().int().contiguous())
    cm.fill_holes(max_hole_perimeter=3e-2)
    cm.simplify(target * 3)
    for final in (False, True):
        cm.remove_duplicate_faces()
        cm.repair_non_manifold_edges()
        cm.remove_small_connected_components(1e-5)
        cm.fill_holes(max_hole_perimeter=3e-2)
        if not final:
            cm.simplify(target)
    cm.unify_face_orientations()
    return cm.read()


def _watertight_mesh(vertices, faces, target: int, res: int):
    """One closed, manifold shell for printing: unsigned distance to the mesh on a dense grid (CuMesh BVH),
    flood-fill the outside from the grid border (scipy), then marching cubes on the signed field at a ~1 voxel
    offset, so open sheets (leaves, cloth) become thin solids instead of vanishing. Internal cavities and
    hidden inner surfaces are filled. Simplified to `target` faces; pymeshfix repairs anything left open."""
    import cumesh
    import numpy as np
    import torch
    import trimesh
    from scipy import ndimage
    from skimage import measure

    v = vertices.cuda().float().contiguous()
    f = faces.cuda().int().contiguous()
    bvh = cumesh.cuBVH(v, f)
    lo, hi = v.min(0).values, v.max(0).values
    h = float((hi - lo).max()) / res
    pad = 3
    dims = (((hi - lo) / h).ceil().long() + 2 * pad + 1).tolist()
    origin = lo - pad * h
    udf = torch.empty(dims, dtype=torch.float32)
    ys = torch.arange(dims[1], device="cuda")
    zs = torch.arange(dims[2], device="cuda")
    step = max(1, (8 << 20) // (dims[1] * dims[2]))  # ~8M points per BVH query
    for x0 in range(0, dims[0], step):
        xs = torch.arange(x0, min(x0 + step, dims[0]), device="cuda")
        pts = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3).float() * h + origin
        udf[x0 : x0 + len(xs)] = bvh.unsigned_distance(pts)[0].reshape(len(xs), dims[1], dims[2]).cpu()
    del bvh
    udf = udf.numpy()
    t = 0.87 * h
    labels, _ = ndimage.label(udf >= t)
    border = np.unique(
        np.concatenate([labels[[0, -1]].ravel(), labels[:, [0, -1]].ravel(), labels[:, :, [0, -1]].ravel()])
    )
    is_outside = np.zeros(int(labels.max()) + 1, dtype=bool)  # lookup table: np.isin is slow on 1e8 voxels
    is_outside[border[border > 0]] = True
    outside = is_outside[labels]
    del labels
    sdf = np.where(outside, udf, -udf)
    del udf, outside
    mv, mf, _, _ = measure.marching_cubes(sdf, level=t, spacing=(h, h, h))
    del sdf
    mv = torch.from_numpy(mv.astype(np.float32)).cuda() + origin
    cm = cumesh.CuMesh()
    cm.init(mv.contiguous(), torch.from_numpy(mf.astype(np.int32)).cuda().contiguous())
    cm.simplify(target)
    cm.remove_duplicate_faces()
    cm.unify_face_orientations()
    v2, f2 = cm.read()
    tm = trimesh.Trimesh(v2.cpu().numpy(), f2.cpu().numpy(), process=True)
    if not tm.is_watertight:
        import pymeshfix

        fix = pymeshfix.MeshFix(tm.vertices, tm.faces)
        fix.repair()
        pv, pf = (fix.v, fix.f) if hasattr(fix, "v") else (fix.points, fix.faces)
        tm = trimesh.Trimesh(np.asarray(pv), np.asarray(pf).reshape(-1, 3), process=True)
    trimesh.repair.fix_normals(tm)  # outward-facing (positive volume)
    return tm


def _shape_glb(mesh, req, res: int):
    """Shape-only trimesh (Y-up) for paint=false or watertight=true."""
    import trimesh

    target = _face_budget(req)
    if req.watertight:
        tm = _watertight_mesh(mesh.vertices, mesh.faces, target, int(T2_WATERTIGHT_RES))
        tm.vertices = _to_y_up(tm.vertices)
        return tm
    v, f = _clean_shape(mesh.vertices, mesh.faces, target)
    return trimesh.Trimesh(_to_y_up(v.cpu().numpy()), f.cpu().numpy(), process=False)


def _textured_glb(mesh, req):
    import o_voxel

    return o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=_face_budget(req),
        texture_size=max(512, min(4096, int(req.texture_size))),
        # Dual-contouring remesh is opt-in: it wraps closed parts in an outer and a hidden inner shell,
        # which spends half the face budget on faces nobody sees.
        remesh=bool(req.remesh),
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )


# --- Preview grid ---------------------------------------------------------------------------------------
def _cam_pose(yaw_deg: float, elev_deg: float = 20, r: float = 1.9):
    import numpy as np

    theta, elev = np.deg2rad(yaw_deg), np.deg2rad(elev_deg)
    eye = np.array([r * np.cos(elev) * np.sin(theta), r * np.sin(elev), r * np.cos(elev) * np.cos(theta)])
    z = eye / np.linalg.norm(eye)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def _preview_mesh(mesh):
    """Unit-size, centered copy; metallic factor 0 so PBR textures read under plain preview lights."""
    m = mesh.copy()
    m.apply_translation(-m.bounding_box.centroid)
    m.apply_scale(1.0 / max(float(max(m.extents)), 1e-6))
    mat = getattr(getattr(m, "visual", None), "material", None)
    if mat is not None and hasattr(mat, "metallicFactor"):
        mat.metallicFactor = 0.0
    return m


def _preview_grid_pyrender(mesh):
    """2x2 turntable via pyrender EGL: the same renderer and camera as server.py's Hunyuan previews."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import numpy as np

    if not hasattr(np, "infty"):
        np.infty = np.inf  # pyrender 0.1.45 predates numpy 2
    import pyrender

    m = _preview_mesh(mesh)
    frames = []
    renderer = pyrender.OffscreenRenderer(512, 512)
    try:
        for yaw in (0, 90, 180, 270):
            scene = pyrender.Scene(bg_color=[0, 0, 0, 255], ambient_light=[0.35, 0.35, 0.35])
            scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
            pose = _cam_pose(yaw)
            scene.add(pyrender.PerspectiveCamera(yfov=float(np.deg2rad(40))), pose=pose)
            scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
            color, _ = renderer.render(scene)
            frames.append(color[..., :3])
    finally:
        renderer.delete()
    return np.concatenate([np.concatenate(frames[0:2], axis=1), np.concatenate(frames[2:4], axis=1)], axis=0)


def _preview_grid_cuda(mesh):
    """Fallback when EGL isn't available in the container: same views, rasterized with nvdiffrast's CUDA
    rasterizer (no OpenGL), flat-shaded with a camera light; base-color texture when present."""
    import numpy as np
    import nvdiffrast.torch as dr
    import torch

    m = _preview_mesh(mesh)
    v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device="cuda")
    f = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device="cuda")
    fn = torch.tensor(np.asarray(m.face_normals), dtype=torch.float32, device="cuda")
    uv = tex = None
    vis = getattr(m, "visual", None)
    if getattr(vis, "uv", None) is not None and getattr(vis, "material", None) is not None:
        img = getattr(vis.material, "baseColorTexture", None) or getattr(vis.material, "image", None)
        if img is not None:
            uv = torch.tensor(np.asarray(vis.uv), dtype=torch.float32, device="cuda")
            uv[:, 1] = 1 - uv[:, 1]
            tex = torch.tensor(np.asarray(img.convert("RGB")), dtype=torch.float32, device="cuda")[None] / 255
    fov = np.deg2rad(40)
    near, far = 0.1, 10.0
    p = 1 / np.tan(fov / 2)
    proj = torch.tensor(
        [[p, 0, 0, 0], [0, p, 0, 0], [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]],
        dtype=torch.float32, device="cuda",
    )
    ctx = dr.RasterizeCudaContext()
    vh = torch.cat([v, torch.ones_like(v[:, :1])], 1)
    frames = []
    for yaw in (0, 90, 180, 270):
        pose = torch.tensor(_cam_pose(yaw), dtype=torch.float32, device="cuda")
        clip = (vh @ (proj @ torch.linalg.inv(pose)).T)[None].contiguous()
        rast, _ = dr.rasterize(ctx, clip, f, (512, 512))
        tri = rast[0, ..., 3].long() - 1
        hit = tri >= 0
        light = pose[:3, 2]  # toward the camera
        shade = 0.35 + 0.65 * (fn[tri.clamp(min=0)] @ light).abs()
        color = torch.full((512, 512, 3), 0.8, device="cuda") * shade[..., None]
        if tex is not None:
            tuv, _ = dr.interpolate(uv[None].contiguous(), rast, f)
            color = dr.texture(tex.contiguous(), tuv, filter_mode="linear")[0] * shade[..., None]
        color = torch.where(hit[..., None], color, torch.zeros_like(color))
        frames.append((color.clamp(0, 1).flip(0).cpu().numpy() * 255).astype(np.uint8))
    return np.concatenate([np.concatenate(frames[0:2], axis=1), np.concatenate(frames[2:4], axis=1)], axis=0)


_preview_backend = os.environ.get("PREVIEW_RENDERER", "pyrender")  # pyrender | cuda


def _preview_grid(mesh):
    global _preview_backend
    if _preview_backend == "pyrender":
        try:
            return _preview_grid_pyrender(mesh)
        except Exception as e:  # no EGL device in this container: use the CUDA rasterizer from now on
            print(f"pyrender preview failed, switching to nvdiffrast: {type(e).__name__}: {e}", flush=True)
            _preview_backend = "cuda"
    return _preview_grid_cuda(mesh)


# --- Generation -----------------------------------------------------------------------------------------
def _trellis2_generate(img, req) -> dict:
    """Image -> TRELLIS.2 -> GLB (+ preview PNG, written after the GLB). Returns stats for the response."""
    import numpy as np

    _stage(req.name, "loading", 0.2)
    img.thumbnail((2048, 2048))
    alpha = img.getextrema()[3] if img.mode == "RGBA" else (255, 255)
    if alpha[0] == 255:  # no real transparency: strip the background (BiRefNet; u2net on request or failure)
        if req.bg == "u2net":
            img = _u2net_cut(img)
        else:
            try:
                img = _birefnet_cut(img)
            except Exception as e:
                print(f"birefnet failed, using u2net: {type(e).__name__}: {e}", flush=True)
                img = _u2net_cut(img)
    if int((np.asarray(img.getchannel("A")) > 204).sum()) < 64:
        raise ValueError("no object found in the image (background removal left nothing)")

    pipe = _get_t2()
    _stage(req.name, "shape", 0.0)
    paint = bool(req.paint) and not req.watertight
    mesh, res = _t2_run(pipe, img, req, paint)

    if paint:
        _stage(req.name, "texture", 0.9)
        glb = _textured_glb(mesh, req)
    else:
        _stage(req.name, "cleanup", 0.0)
        glb = _shape_glb(mesh, req, res)
    del mesh

    _stage(req.name, "export", 0.0)
    glb.export(str(OUT / f"{req.name}.glb"))
    _stage(req.name, "export", 0.5)
    import imageio

    imageio.imwrite(str(OUT / f"{req.name}.png"), _preview_grid(glb))
    import torch

    torch.cuda.empty_cache()
    return {
        "verts": int(glb.vertices.shape[0]),
        "faces": int(glb.faces.shape[0]),
        "textured": paint,
        "watertight": bool(getattr(glb, "is_watertight", False)) if req.watertight else None,
        "resolution": res,
        "pipeline_type": req.pipeline_type or T2_PIPELINE_TYPE,
    }


def _decode_image(data: str):
    from PIL import Image

    b64 = data.split(",", 1)[1] if data.startswith("data:") else data
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")


# Explicit photos are refused (same classifier and threshold as server.py).
NSFW_MODEL = os.environ.get("NSFW_MODEL", "Falconsai/nsfw_image_detection")
NSFW_BLOCK = float(os.environ.get("NSFW_BLOCK", "0.7"))
_nsfw = None
_nsfw_lock = threading.Lock()


def _nsfw_score(img) -> float:
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
        "model": T2_MODEL,
        "pipeline_type": T2_PIPELINE_TYPE,
        "low_vram": getattr(_pipes.get("t2"), "low_vram", None),
        "preview_renderer": _preview_backend,
        "concept_model": CONCEPT_MODEL,
        "text_via_image": TEXT_VIA_IMAGE,
        "model_loaded": "t2" in _pipes,
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
    paint: bool = True  # false: shape only (no texture latent, no material)
    ss_steps: int | None = None  # sparse-structure sampler steps (TRELLIS.2 default 12)
    ss_cfg: float | None = None  # sparse-structure guidance strength (default 7.5)
    slat_steps: int | None = None  # shape latent steps (default 12, per cascade pass)
    slat_cfg: float | None = None  # shape latent guidance (default 7.5)
    tex_steps: int | None = None  # texture latent steps (default 12)
    tex_cfg: float | None = None  # texture latent guidance (default 1.0)
    pipeline_type: str | None = None  # 512 | 1024 | 1024_cascade (default) | 1536_cascade
    max_num_tokens: int | None = None  # cascade token cap (default 49152; lower = less VRAM, coarser)
    watertight: bool = False  # one closed shell for printing (geometry only)
    remesh: bool = False  # textured: dual-contouring remesh before UV unwrap (to_glb remesh=True)
    max_faces: int | None = None  # face budget, instead of the simplify mapping (20k..1M)
    bg: str | None = None  # background removal: BiRefNet by default; "u2net" for the old remover
    # Hunyuan-only knobs, accepted so the same lab configs run against both servers; ignored here.
    octree_resolution: int | None = None
    num_chunks: int | None = None


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
    if req.mode == "text" and not TEXT_VIA_IMAGE:
        raise HTTPException(400, "trellis2 is image-conditioned; enable TEXT_VIA_IMAGE")
    if req.pipeline_type and req.pipeline_type not in PIPELINE_TYPES:
        raise HTTPException(400, f"pipeline_type must be one of {', '.join(PIPELINE_TYPES)}")
    reason = _prompt_block_reason(req.prompt)
    if reason:
        raise HTTPException(422, reason)
    if req.mode == "image" and _nsfw_score(_decode_image(req.image)) >= NSFW_BLOCK:
        raise HTTPException(422, "image_blocked: explicit images are not allowed")
    if not _lock.acquire(timeout=2):
        raise HTTPException(409, "busy")
    global _last_used
    _last_used = time.time()
    try:
        t0 = time.time()
        _progress.pop(req.name, None)
        if req.mode == "image":
            subject = _decode_image(req.image)
        else:
            concept = _concept_image(req.prompt, req.seed, req.name)
            concept.save(str(OUT / f"{req.name}_src.png"))
            subject = concept.convert("RGBA")
        stats = _trellis2_generate(subject, req)
        _stage(req.name, "done", 1.0)
        return {"ok": True, "mode": req.mode, "backend": BACKEND, "ms": int((time.time() - t0) * 1000), **stats}
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
    """Photo check for the studio, identical to server.py: u2net cut on the CPU + a verdict.

    verdict: "ok" | "cut_off" | "background" | "empty" | "explicit".
    """
    _check(x_token)
    import numpy as np

    t0 = time.time()
    img = _decode_image(req.image)
    img.thumbnail((1024, 1024))
    nsfw = _nsfw_score(img)
    if nsfw >= NSFW_BLOCK:
        return {"verdict": "explicit", "coverage": 0, "edges_touching": [], "cutout": "", "nsfw": round(nsfw, 3)}
    t1 = time.time()
    cut = img if img.getextrema()[3][0] < 255 else _u2net_cut(img)
    t2 = time.time()
    a = np.asarray(cut.getchannel("A")) > 128
    h, w = a.shape
    coverage = float(a.mean())
    band = max(2, int(min(h, w) * 0.01))
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


# --- Warm-up on boot ------------------------------------------------------------------------------------
# Downloads and loads everything in the background as soon as the server starts (/generate answers 409
# "busy" meanwhile), then runs one tiny shape-only generation so the first real job doesn't pay for
# Triton/FlexGEMM kernel compilation, and so a broken stack shows up in /health's warm_error at boot.
WARM_ON_BOOT = os.environ.get("WARM_ON_BOOT", "1") != "0"
WARM_RUN = os.environ.get("T2_WARM_RUN", "1") != "0"


def _warm_image():
    """A synthetic RGBA subject (a shaded disc on transparency) for the warm-up run."""
    import numpy as np
    from PIL import Image

    yy, xx = np.mgrid[0:512, 0:512]
    d = np.hypot(xx - 256, yy - 256) / 180
    rgba = np.zeros((512, 512, 4), np.uint8)
    rgba[..., :3] = (np.clip(1.1 - d, 0.2, 1.0)[..., None] * np.array([200, 120, 80])).astype(np.uint8)
    rgba[..., 3] = np.where(d < 1, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _warm() -> None:
    global _warm_error
    with _lock:
        t0 = time.time()
        try:
            from PIL import Image as _Img

            try:  # the default background remover, so the first photo job doesn't wait for it
                _birefnet_cut(_Img.new("RGB", (64, 64), "gray"))
            except Exception as e:
                print(f"birefnet warm-up failed (jobs fall back to u2net): {e}", flush=True)
            _get_t2()
            print(f"trellis2 loaded in {time.time() - t0:.0f}s (low_vram={_pipes['t2'].low_vram})", flush=True)
            if WARM_RUN:
                t1 = time.time()
                req = GenReq(name="", mode="image", image="", seed=0, paint=False, pipeline_type="512")
                mesh, _res = _t2_run(_pipes["t2"], _warm_image(), req, paint=False)
                del mesh
                print(f"warm-up generation ok in {time.time() - t1:.0f}s", flush=True)
            if TEXT_VIA_IMAGE:
                if _gpu_total_gb() >= 30:
                    _load_concept(keep=True)
                else:
                    from huggingface_hub import snapshot_download

                    snapshot_download(FLUX_MODEL if CONCEPT_MODEL == "flux" else SDXL_MODEL)
            try:
                _nsfw_score(_Img.new("RGB", (64, 64), "gray"))
                _u2net_cut(_Img.new("RGB", (64, 64), "gray"))
            except Exception as e:
                print(f"upload-check warm-up failed: {e}", flush=True)
            print(f"warm-up done in {time.time() - t0:.0f}s: {sorted(_pipes)}", flush=True)
        except Exception as e:  # never keep the server from serving
            _warm_error = f"{type(e).__name__}: {e}"[:500]
            traceback.print_exc()
            print(f"warm-up failed after {time.time() - t0:.0f}s: {e}", flush=True)
        finally:
            import torch

            torch.cuda.empty_cache()


@app.on_event("startup")
def _start_warm() -> None:
    if WARM_ON_BOOT:
        threading.Thread(target=_warm, daemon=True).start()
