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

## Hunyuan3D-2.1 shape A/B (branch `hy21`, image tags `h21-*`)

With `BACKEND=hunyuan`, `POST /generate` takes `shape_model: "2.0" | "2.1"` (unset = the pod's
`SHAPE_MODEL_DEFAULT`, which defaults to `"2.0"`, the production model). `"2.1"` runs the
[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) shape DiT (`hy3dshape`, pinned
commit in `install_deps_hy21.sh`) with the same BiRefNet cutout, `seed`, `ss_steps`, `ss_cfg`,
`octree_resolution`, `num_chunks`, cleanup, face budget and GLB/preview output; `paint: true` paints
it with the 2.0 paint model. The response says which model ran (`shape_model`). The first 2.1 request
on a pod downloads ~7.4 GB (`tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt`) into
`HF_HOME`; set `SHAPE_MODEL_DEFAULT=2.1` to switch a whole pod (and warm 2.1 at boot instead).
Built by `.github/workflows/build-hy21.yml` as `ghcr.io/fintermxyz/trellis-runpod:h21-<sha>` and
`:h21-latest` only.

## RunPod

Create a pod from the image with 60 GB container disk, port `8000/http`, env
`GEN_TOKEN=<secret>`. If the ghcr image can't be pulled, boot a bare
`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` pod whose start command runs
[`bootstrap.sh`](bootstrap.sh) from this repo — same result, slower first boot.

## Licenses

TRELLIS is MIT licensed, © Microsoft Corporation. This repo's glue code is MIT as well.
Hunyuan3D-2.0 and 2.1 are under Tencent's Hunyuan 3D Community Licenses, which exclude the EU,
UK and South Korea from their territory (see `/app/Hunyuan3D-2.1/LICENSE` in the image).
Generated models inherit the licenses of TRELLIS's released weights.
