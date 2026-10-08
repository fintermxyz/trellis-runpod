"""Build-time check for the Hunyuan3D-2.1 shape stack (hy3dshape). Needs no GPU and no weights.

1. Imports every hy3dshape module the 2.1 shape pipeline touches, plus hy3dgen (2.0) next to it.
2. Builds the real 2.1 DiT / VAE / DINO conditioner from the real config.yaml (copied below from
   tencent/Hunyuan3D-2.1 @ HY21_REVISION, hunyuan3d-dit-v2-1/config.yaml) on the meta device, so a
   config/code/library mismatch (e.g. a kwarg the installed transformers or torch doesn't accept) fails
   the build instead of the first 2.1 job.
3. Runs the whole 2.1 pipeline end to end on the CPU with a tiny random-weight model of the same
   architecture: from_single_file (torch.load weights_only) -> image preprocessing -> DINO -> flow-matching
   DiT with MoE -> VAE decode -> marching cubes -> trimesh, then the hy3dgen cleanup the server applies.
   This proves the code path runs on this torch/diffusers/transformers/timm; it says nothing about speed,
   VRAM or output quality, which need a GPU and the real weights.

Run: python check_hy21.py   (exits non-zero on any failure)
"""
import os
import sys
import tempfile

import numpy as np
import torch
import yaml
from PIL import Image

REAL_CONFIG = """
model:
  target: hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain
  params:
    input_size: &num_latents 4096
    in_channels: 64
    hidden_size: 2048
    context_dim: 1024
    depth: 21
    num_heads: 16
    qk_norm: true
    text_len: 1370
    with_decoupled_ca: false
    use_attention_pooling: false
    qk_norm_type: 'rms'
    qkv_bias: false
    use_pos_emb: false
    num_moe_layers: 6
    num_experts: 8
    moe_top_k: 2
vae:
  target: hy3dshape.models.autoencoders.ShapeVAE
  params:
    num_latents: *num_latents
    embed_dim: 64
    num_freqs: 8
    include_pi: false
    heads: 16
    width: 1024
    num_encoder_layers: 8
    num_decoder_layers: 16
    qkv_bias: false
    qk_norm: true
    scale_factor: 1.0039506158752403
    geo_decoder_mlp_expand_ratio: 4
    geo_decoder_downsample_ratio: 1
    geo_decoder_ln_post: true
    point_feats: 4
    pc_size: 81920
    pc_sharpedge_size: 0
conditioner:
  target: hy3dshape.models.conditioner.SingleImageEncoder
  params:
    main_image_encoder:
        type: DinoImageEncoder # dino large
        kwargs:
            config:
              attention_probs_dropout_prob: 0.0
              drop_path_rate: 0.0
              hidden_act: gelu
              hidden_dropout_prob: 0.0
              hidden_size: 1024
              image_size: 518
              initializer_range: 0.02
              layer_norm_eps: 1.e-6
              layerscale_value: 1.0
              mlp_ratio: 4
              model_type: dinov2
              num_attention_heads: 16
              num_channels: 3
              num_hidden_layers: 24
              patch_size: 14
              qkv_bias: true
              torch_dtype: float32
              use_swiglu_ffn: false
            image_size: 518
            use_cls_token: true
scheduler:
  target: hy3dshape.schedulers.FlowMatchEulerDiscreteScheduler
  params:
    num_train_timesteps: 1000
image_processor:
  target: hy3dshape.preprocessors.ImageProcessorV2
  params:
    size: 512
    border_ratio: 0.15
pipeline:
  target: hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline
"""


def tiny_config() -> dict:
    """Same targets and options as the real config, toy sizes (runs in seconds on a CPU)."""
    c = yaml.safe_load(REAL_CONFIG)
    c["model"]["params"].update(
        input_size=16, in_channels=8, hidden_size=64, context_dim=32, depth=3, num_heads=4,
        text_len=5, num_moe_layers=1, num_experts=2, moe_top_k=1,
    )
    c["vae"]["params"].update(
        num_latents=16, embed_dim=8, heads=4, width=32, num_encoder_layers=1, num_decoder_layers=1, pc_size=64,
    )
    enc = c["conditioner"]["params"]["main_image_encoder"]["kwargs"]
    enc["config"].update(hidden_size=32, image_size=28, num_attention_heads=4, num_hidden_layers=1)
    enc["image_size"] = 28  # (28 // 14)^2 + cls = 5 tokens = text_len
    c["image_processor"]["params"]["size"] = 64
    return c


def main() -> None:
    print("torch", torch.__version__, "cuda build", torch.version.cuda)

    # 1. imports
    import hy3dshape
    from hy3dshape.models.autoencoders import ShapeVAE, SurfaceExtractors  # noqa: F401
    from hy3dshape.models.conditioner import SingleImageEncoder  # noqa: F401
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain  # noqa: F401
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline, instantiate_from_config
    from hy3dshape.preprocessors import ImageProcessorV2  # noqa: F401
    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: F401

    print("hy3dshape from", os.path.dirname(hy3dshape.__file__))
    try:
        import hy3dgen.shapegen  # noqa: F401  the 2.0 stack must still import next to it

        have_hy3dgen = True
    except ImportError as e:
        if os.environ.get("REQUIRE_HY3DGEN", "1") == "1":
            raise
        print("hy3dgen not installed here, skipping the 2.0 checks:", e)
        have_hy3dgen = False

    # 2. the real architecture, on the meta device (no memory, no weights)
    real = yaml.safe_load(REAL_CONFIG)
    with torch.device("meta"):
        counts = {}
        for part in ("model", "vae", "conditioner"):
            m = instantiate_from_config(real[part])
            counts[part] = sum(p.numel() for p in m.parameters())
            del m
    total = sum(counts.values())
    print("real 2.1 parameter counts:", {k: f"{v / 1e9:.2f}B" for k, v in counts.items()},
          f"total {total / 1e9:.2f}B (~{total * 2 / 1e9:.1f} GB in fp16)")
    instantiate_from_config(real["scheduler"])
    instantiate_from_config(real["image_processor"])

    # 3. tiny end-to-end run on the CPU
    torch.manual_seed(0)
    cfg = tiny_config()
    with tempfile.TemporaryDirectory() as d:
        cfg_path, ckpt_path = os.path.join(d, "config.yaml"), os.path.join(d, "model.fp16.ckpt")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        state = {k: instantiate_from_config(cfg[k]).state_dict() for k in ("model", "vae", "conditioner")}
        torch.save(state, ckpt_path)
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path, cfg_path, device="cpu", dtype=torch.float32
        )

    # An RGBA cutout like the one BiRefNet hands the server: an opaque disc on a transparent background.
    yy, xx = np.mgrid[:96, :96]
    rgba = np.zeros((96, 96, 4), np.uint8)
    disc = (yy - 48) ** 2 + (xx - 48) ** 2 < 30**2
    rgba[disc] = (200, 120, 60, 255)
    img = Image.fromarray(rgba, "RGBA")

    # Random weights give a random field: shift the iso level to its median so marching cubes finds a surface.
    with torch.inference_mode():
        lat = pipe(image=img, num_inference_steps=2, guidance_scale=5.0, octree_resolution=24, num_chunks=4000,
                   generator=torch.manual_seed(0), output_type="latent", enable_pbar=False)
        lat = pipe.vae(1.0 / pipe.vae.scale_factor * lat)
        grid = pipe.vae.volume_decoder(lat, pipe.vae.geo_decoder, bounds=1.01, num_chunks=4000, octree_resolution=24,
                                       enable_pbar=False)
    level = float(grid.float().median())
    mesh = pipe._export(
        pipe(image=img, num_inference_steps=2, guidance_scale=5.0, octree_resolution=24, num_chunks=4000,
             generator=torch.manual_seed(0), output_type="latent", enable_pbar=False),
        "trimesh", 1.01, level, 4000, 24, None, enable_pbar=False,
    )[0]
    if mesh is None:
        sys.exit("tiny 2.1 pipeline produced no mesh")
    print(f"tiny 2.1 pipeline mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # The exact call the server makes (default mc_level), to prove the kwargs are accepted.
    out = pipe(image=img, num_inference_steps=2, guidance_scale=5.0, octree_resolution=24, num_chunks=4000,
               generator=torch.manual_seed(0), enable_pbar=False)
    assert isinstance(out, list) and len(out) == 1, out

    if have_hy3dgen:
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

        m = FloaterRemover()(mesh)
        m = DegenerateFaceRemover()(m)
        m = FaceReducer()(m, max_facenum=20000)
        print(f"hy3dgen cleanup on the 2.1 mesh: {len(m.faces)} faces")
    print("hy21 check OK")


if __name__ == "__main__":
    main()
