# trellis-runpod

A [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) text-to-3D generation server
packaged for [RunPod](https://runpod.io) GPU pods. Built to batch-generate game assets
(GLB + preview PNG) over a tiny authenticated HTTP API.

Image: `ghcr.io/fintermxyz/trellis-runpod:latest` (built by the GitHub Action in this repo).

## API

| Route | Auth | What |
|---|---|---|
| `GET /health` | – | `{ok, gpu, model_loaded}` |
| `POST /generate` | `X-Token` | `{name, mode: "text"\|"image", prompt?, image?, seed, simplify, texture_size, ss_steps?, ss_cfg?, slat_steps?, slat_cfg?}` → renders `<name>.glb` + 4-view `<name>.png` |
| `GET /asset/<name>.glb\|.png` | `X-Token` | download a result |

Set `GEN_TOKEN` in the pod env. `mode:"image"` (single object photo as data URL /
base64; rembg strips the background) runs `TRELLIS-image-large` — markedly higher
fidelity than text mode (`TRELLIS-text-xlarge`), and the Tripo-style path: make an
image first, then lift it to 3D. The two pipelines don't share a 24 GB card, so
switching modes reloads (~1 min from a warm HF cache, ~10 min first ever). Warm
generations take ~1–3 min on an RTX 4090.

## RunPod

Create a pod from the image with 60 GB container disk, port `8000/http`, env
`GEN_TOKEN=<secret>`. If the ghcr image can't be pulled, boot a bare
`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` pod whose start command runs
[`bootstrap.sh`](bootstrap.sh) from this repo — same result, slower first boot.

## Licenses

TRELLIS is MIT licensed, © Microsoft Corporation. This repo's glue code is MIT as well.
Generated models inherit the licenses of TRELLIS's released weights.

## TRELLIS.2 image (A/B test build, branch `trellis2`)

`Dockerfile.trellis2` + `server_trellis2.py` serve [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
(4B, image → 3D) behind the **same HTTP API** as `server.py`, so the worker and the shape lab talk to it
unchanged. Built by `.github/workflows/build-trellis2.yml` on pushes to `trellis2` and published only as
`ghcr.io/fintermxyz/trellis-runpod:t2-<sha>` and `:t2-latest` (never `:latest`).

Pod env:

| Var | Needed | What |
|---|---|---|
| `GEN_TOKEN` | yes | X-Token for every authenticated route |
| `HF_TOKEN` | yes | read token from an HF account that accepted the licence of the gated `facebook/dinov3-vitl16-pretrain-lvd1689m` (TRELLIS.2's image encoder). Never bake it into the image. |
| `HF_HOME` | preset `/workspace/hf` | weights persist on the pod volume (~25 GB first download incl. SDXL) |
| `T2_PIPELINE_TYPE` | optional | default `1024_cascade`; `512`, `1024`, `1536_cascade` |
| `T2_LOW_VRAM` | optional | `auto` (default: models stay on the GPU on ≥ 40 GB cards), `0`, `1` |
| `T2_WATERTIGHT_RES` | optional | grid for `watertight=true` exports, default 512 |
| `CONCEPT_MODEL` | optional | `sdxl` (default here, even with HF_TOKEN set) or `flux` |
| `PREVIEW_RENDERER` | optional | `pyrender` (EGL, default; falls back automatically) or `cuda` (nvdiffrast) |
| `T2_WARM_RUN` | optional | `1` (default): one tiny generation at boot to compile kernels and surface errors in `/health.warm_error` |

Extra `/generate` fields (all optional; the rest are as in `server.py`): `pipeline_type`, `tex_steps`,
`tex_cfg`, `max_num_tokens`, `watertight` (one closed manifold shell for printing, geometry only),
`remesh` (textured: dual-contouring remesh before UV unwrap). `ss_steps`/`ss_cfg` drive the sparse-structure
sampler (default 12 steps, guidance 7.5), `slat_steps`/`slat_cfg` the shape sampler. `octree_resolution`
and `num_chunks` are Hunyuan-only and ignored. `paint=false` skips the texture latent entirely.
Prompts naming guns/firearms are refused (422 `prompt_blocked`), because the DINOv3 licence bans weapons uses.

Shape lab (`shapelab/lab.py`) against a TRELLIS.2 pod runs unchanged; it sends `ss_steps: 15`, which here
means 15 sparse-structure steps. Pass `null` to use TRELLIS.2's defaults, e.g.

```
LAB_URL=https://<pod>-8000.proxy.runpod.net LAB_TOKEN=... \
LAB_CONF='{"T2_1024c":{"ss_steps":null},"T2_512":{"ss_steps":null,"pipeline_type":"512"},"T2_1536c":{"ss_steps":null,"pipeline_type":"1536_cascade"}}' \
python lab.py
```

Use a separate `out/` directory (or move `out/results.json` aside) so results don't merge with the Hunyuan runs.
