# trellis-runpod

A [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) text-to-3D generation server
packaged for [RunPod](https://runpod.io) GPU pods. Built to batch-generate game assets
(GLB + preview PNG) over a tiny authenticated HTTP API.

Image: `ghcr.io/fintermxyz/trellis-runpod:latest` (built by the GitHub Action in this repo).

## API

| Route | Auth | What |
|---|---|---|
| `GET /health` | – | `{ok, gpu, model_loaded}` |
| `POST /generate` | `X-Token` | `{name, prompt, seed, simplify, texture_size}` → renders `<name>.glb` + 4-view `<name>.png` |
| `GET /asset/<name>.glb\|.png` | `X-Token` | download a result |

Set `GEN_TOKEN` in the pod env. First `/generate` loads `microsoft/TRELLIS-text-xlarge`
(~10 min incl. download); after that each asset takes ~1–3 min on an RTX 4090.

## RunPod

Create a pod from the image with 60 GB container disk, port `8000/http`, env
`GEN_TOKEN=<secret>`. If the ghcr image can't be pulled, boot a bare
`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` pod whose start command runs
[`bootstrap.sh`](bootstrap.sh) from this repo — same result, slower first boot.

## Licenses

TRELLIS is MIT licensed, © Microsoft Corporation. This repo's glue code is MIT as well.
Generated models inherit the licenses of TRELLIS's released weights.
