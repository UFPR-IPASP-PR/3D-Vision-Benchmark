#!/usr/bin/env python3
"""
extract_hunyuan_features.py

Multi-stage feature extraction from the Hunyuan3D pipeline
(Hunyuan3DDiTFlowMatchingPipeline).  

Feature files are written under --output_dir, one sub-folder per feature type:

  geometry_tokens/  – VAE-decoded shape token set (post_kl + Transformer)
                      → (num_latents, width)               float32  .npy

  occupancy_field/  – Dense 3-D occupancy scalar field from VolumeDecoder
                      → (R+1, R+1, R+1)                    float32  .npy
                        where R = --octree_resolution (default 64 for speed)

  mesh/             – Triangle mesh from Marching Cubes
                      → .npz  with keys:
                           'vertices' : (V, 3) float32
                           'faces'    : (F, 3) int32

Each feature is saved as a separate  <image_stem>.<ext>  file inside its folder.
No large in-memory accumulators are used.

The three features are derived from the same denoised latent, so exactly ONE
diffusion pass is run per image regardless of how many feature types are saved.

Usage example
─────────────
python extract_hunyuan_features.py \
    --ann_json     data/annotations.json \
    --img_dir      data/images \
    --model_path   tencent/Hunyuan3D-2 \
    --output_dir   features/ \
    --octree_resolution 64          # use 384 for production quality

# Skip the mesh if you only care about tokens + field:
python extract_hunyuan_features.py ... --no_mesh

# Deterministic runs:
python extract_hunyuan_features.py ... --seed 42
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure the local hy3dshape source tree takes priority over any installed
# version of the package.  gradio_app.py does the same with sys.path.insert.
# Using an absolute path derived from this file's location so the script
# works regardless of the current working directory.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "hy3dshape"))

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# retrieve_timesteps moved between diffusers versions; try all known locations.
try:
    from diffusers.schedulers.scheduling_utils import retrieve_timesteps
except ImportError:
    try:
        from diffusers import retrieve_timesteps
    except ImportError:
        # Inline fallback: matches the diffusers reference implementation.
        def retrieve_timesteps(scheduler, num_inference_steps=None, device=None,
                               timesteps=None, sigmas=None, **kwargs):
            if timesteps is not None and sigmas is not None:
                raise ValueError("Pass at most one of `timesteps` or `sigmas`.")
            if sigmas is not None:
                scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            elif timesteps is not None:
                scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            else:
                scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            return scheduler.timesteps, len(scheduler.timesteps)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Arguments
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description="Extract geometry tokens, occupancy field, and mesh from Hunyuan3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ──────────────────────────────────────────────────────────────────
    p.add_argument("--ann_json",   required=True,
                   help="Path to the annotation JSON file")
    p.add_argument("--img_dir",    required=True,
                   help="Root directory containing the images")
    p.add_argument("--format",     default="jbcs", choices=["jbcs", "lplc"],
                   help="Annotation format: 'jbcs' (flat list) or 'lplc' (nested dict)")
    p.add_argument("--output_dir", required=True,
                   help="Root directory under which per-feature-type folders are created")

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model_path", default="tencent/Hunyuan3D-2.1",
                   help="HuggingFace model path or local directory for from_pretrained()")
    p.add_argument("--subfolder",  default="hunyuan3d-dit-v2-1",
                   help="Subfolder inside model_path containing the DiT weights")
    p.add_argument("--dtype",      default="fp16", choices=["fp16", "bf16", "fp32"],
                   help="Model dtype")

    # ── diffusion ─────────────────────────────────────────────────────────────
    p.add_argument("--num_inference_steps", type=int,  default=50)
    p.add_argument("--guidance_scale",      type=float, default=5.0)
    p.add_argument("--seed",                type=int,  default=None,
                   help="Fixed RNG seed for reproducibility (None = random per image)")

    # ── volume decode ─────────────────────────────────────────────────────────
    p.add_argument("--octree_resolution", type=int, default=64,
                   help="Grid resolution R for the occupancy field (R+1)^3. "
                        "64 is fast for pre-processing; use 384 for mesh quality.")
    p.add_argument("--mc_level",          type=float, default=0.0,
                   help="Iso-surface level for Marching Cubes")
    p.add_argument("--num_chunks",        type=int, default=8000,
                   help="Number of query points per geo_decoder chunk (tune to VRAM)")
    p.add_argument("--box_v",             type=float, default=1.01,
                   help="Half-size of the bounding cube passed to the volume decoder")

    # ── feature switches ──────────────────────────────────────────────────────
    p.add_argument("--no_geometry_tokens", action="store_true",
                   help="Skip saving geometry tokens")
    p.add_argument("--no_occupancy_field", action="store_true",
                   help="Skip saving occupancy field (also skips mesh since it "
                        "requires the field to have been computed)")
    p.add_argument("--no_mesh",            action="store_true",
                   help="Skip saving mesh (field is still computed)")

    # ── execution ─────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Annotation loaders  (same contract as no_pool.py / extract_features.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_samples_jbcs(ann_json: str, img_dir: str):
    """Flat list format: [{"filename": "…"}, …]"""
    with open(ann_json) as f:
        data = json.load(f)
    samples = []
    for entry in data:
        fname = entry.get("filename", "")
        if fname:
            samples.append({
                "img_path": os.path.join(img_dir, fname),
                "filename": os.path.basename(fname),
            })
    return samples


def load_samples_lplc(ann_json: str, img_dir: str):
    """Nested-dict format: {"filename": {…}, …}"""
    with open(ann_json) as f:
        data = json.load(f)
    samples = []
    for fname in data:
        samples.append({
            "img_path": os.path.join(img_dir, fname),
            "filename": os.path.basename(fname),
        })
    return samples


def deduplicate(samples):
    seen, out = set(), []
    for s in samples:
        if s["filename"] not in seen:
            out.append(s)
            seen.add(s["filename"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_feature_dirs(output_dir: str, keys: list) -> dict:
    """Create one sub-folder per feature key; return a {key: Path} mapping."""
    dirs = {}
    for k in keys:
        d = Path(output_dir) / k
        d.mkdir(parents=True, exist_ok=True)
        dirs[k] = d
    return dirs


def save_npy(feat_dirs: dict, key: str, stem: str, arr: np.ndarray):
    """Save arr as  <feat_dirs[key]>/<stem>.npy"""
    np.save(feat_dirs[key] / f"{stem}.npy", arr)


def save_mesh_npz(feat_dirs: dict, stem: str, vertices: np.ndarray, faces: np.ndarray):
    """Save mesh as  <feat_dirs['mesh']>/<stem>.npz  with 'vertices' and 'faces' keys."""
    np.savez_compressed(
        feat_dirs["mesh"] / f"{stem}.npz",
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(model_path: str, subfolder: str, dtype_str: str, device: str):
    """
    Load Hunyuan3DDiTFlowMatchingPipeline via from_pretrained.
    Returns the pipeline object (already moved to device + dtype).
    """
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[dtype_str]

    print(f"[INFO] Loading Hunyuan3D pipeline from '{model_path}' (subfolder='{subfolder}') …")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder=subfolder,
        device=device,
        dtype=dtype,
    )
    pipeline.model.eval()
    pipeline.vae.eval()
    pipeline.conditioner.eval()
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Per-image extraction
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def extract_one(
    img_path:            str,
    pipeline,                       # Hunyuan3DDiTFlowMatchingPipeline
    args,
    save_geometry_tokens: bool,
    save_occupancy_field: bool,
    save_mesh:            bool,
) -> dict:
    """
    Run one image through the full pipeline and return a dict of numpy arrays:

      "geometry_tokens"  shape (num_latents, width)      float32
      "occupancy_field"  shape (R+1, R+1, R+1)           float32
      "mesh_vertices"    shape (V, 3)                     float32   ┐ only if
      "mesh_faces"       shape (F, 3)                     int32     ┘ save_mesh

    Intermediate representations explained
    ──────────────────────────────────────
    1. Diffusion sampling produces  `latents` of shape (1, num_latents, embed_dim).
       This is the noise-free point in the latent space of the shape VAE.

    2. VAE decode:  latents / scale_factor → post_kl (linear) + Transformer
       → `geometry_tokens` of shape (num_latents, width).
       These are num_latents learned feature vectors that jointly encode the 3D
       shape in a distributed, order-independent way (no explicit geometry yet).

    3. Volume decode:  geometry_tokens → CrossAttentionDecoder evaluated on a
       dense (R+1)^3 grid of 3D query points → `occupancy_field` of shape
       (R+1, R+1, R+1).  Each scalar is a signed occupancy logit:
         >0 → inside the surface,  <0 → outside.

    4. Marching Cubes on occupancy_field at iso-level mc_level → `mesh`
       (vertices, faces).
    """
    results = {}
    device = pipeline.device
    dtype  = pipeline.dtype

    # ── Step 1: run the diffusion loop to get the denoised latent ─────────────
    # We replicate the __call__ body up to _export so we can intercept `latents`.
    pil_image = Image.open(img_path).convert("RGB")

    # Preprocess: recenters the object, resizes to 512×512, returns image+mask tensors
    cond_inputs = pipeline.prepare_image(pil_image)
    image = cond_inputs.pop("image")

    # Encode condition: CLIP / DINOv2 patch tokens → (2B, N_tokens, D)
    do_cfg = args.guidance_scale >= 0 and not (
        hasattr(pipeline.model, "guidance_embed") and pipeline.model.guidance_embed is True
    )
    cond = pipeline.encode_cond(
        image=image,
        additional_cond_inputs=cond_inputs,
        do_classifier_free_guidance=do_cfg,
        dual_guidance=False,
    )

    # Set up sigmas / timesteps
    sigmas = np.linspace(0, 1, args.num_inference_steps)
    timesteps, _ = retrieve_timesteps(pipeline.scheduler, args.num_inference_steps, device, sigmas=sigmas)

    # Initialise latent as pure Gaussian noise  (1, num_latents, embed_dim)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = pipeline.prepare_latents(batch_size=1, dtype=dtype, device=device, generator=generator)

    guidance = None
    if hasattr(pipeline.model, "guidance_embed") and pipeline.model.guidance_embed is True:
        guidance = torch.tensor([args.guidance_scale], device=device, dtype=dtype)

    # Denoising loop
    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep = t.expand(latent_model_input.shape[0]).to(dtype)
        timestep = timestep / pipeline.scheduler.config.num_train_timesteps

        noise_pred = pipeline.model(latent_model_input, timestep, cond, guidance=guidance)

        if do_cfg:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        outputs = pipeline.scheduler.step(noise_pred, t, latents)
        latents = outputs.prev_sample
    # latents: (1, num_latents, embed_dim)

    # ── Step 2: VAE decode → geometry tokens ──────────────────────────────────
    # Replicate _export: de-scale then run the VAE forward pass (post_kl + Transformer).
    # vae.forward() == vae.decode() == post_kl(latents) + transformer(...)
    vae = pipeline.vae
    scaled_latents = latents / vae.scale_factor               # (1, num_latents, embed_dim)
    geo_tokens = vae(scaled_latents)                          # (1, num_latents, width)

    if save_geometry_tokens:
        results["geometry_tokens"] = (
            geo_tokens.squeeze(0).float().cpu().numpy()       # (num_latents, width)
        )

    # ── Steps 3 & 4: volume decode → occupancy field → mesh ───────────────────
    if save_occupancy_field or save_mesh:
        grid_logits = vae.volume_decoder(
            latents=geo_tokens,
            geo_decoder=vae.geo_decoder,
            bounds=args.box_v,
            mc_level=args.mc_level,
            num_chunks=args.num_chunks,
            octree_resolution=args.octree_resolution,
            enable_pbar=False,
        )
        # grid_logits: (1, R+1, R+1, R+1)  — signed occupancy logits on a dense grid

        if save_occupancy_field:
            results["occupancy_field"] = (
                grid_logits.squeeze(0).float().cpu().numpy()  # (R+1, R+1, R+1)
            )

        if save_mesh:
            # surface_extractor runs Marching Cubes and returns a Latent2MeshOutput
            # per batch item (list of length B).
            mesh_outputs = vae.surface_extractor(
                grid_logits,
                mc_level=args.mc_level,
                bounds=args.box_v,
                octree_resolution=args.octree_resolution,
            )
            mesh_out = mesh_outputs[0]  # Latent2MeshOutput for this image
            if mesh_out is None or mesh_out.mesh_v is None:
                results["mesh_vertices"] = None
                results["mesh_faces"]    = None
            else:
                results["mesh_vertices"] = mesh_out.mesh_v.astype(np.float32)  # (V, 3)
                results["mesh_faces"]    = mesh_out.mesh_f.astype(np.int32)    # (F, 3)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()

    # ── Annotations ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading annotations  (format={args.format}) …")
    load_fn = load_samples_jbcs if args.format == "jbcs" else load_samples_lplc
    samples = deduplicate(load_fn(args.ann_json, args.img_dir))
    print(f"[INFO] {len(samples)} unique images to process.")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    pipeline = build_pipeline(
        args.model_path, args.subfolder, args.dtype, args.device
    )

    # ── Decide which feature types to save ───────────────────────────────────
    save_geometry_tokens = not args.no_geometry_tokens
    save_occupancy_field = not args.no_occupancy_field
    # mesh requires the field computation (grid_logits) so disable both together
    save_mesh = (not args.no_mesh) and save_occupancy_field

    active_keys = []
    if save_geometry_tokens:
        active_keys.append("geometry_tokens")
    if save_occupancy_field:
        active_keys.append("occupancy_field")
    if save_mesh:
        active_keys.append("mesh")

    if not active_keys:
        print("[WARN] All feature types are disabled (--no_* flags). Nothing to do.")
        return

    # ── Output folders ────────────────────────────────────────────────────────
    feat_dirs = make_feature_dirs(args.output_dir, active_keys)
    print(f"[INFO] Feature folders under  {args.output_dir}:")
    for k, d in feat_dirs.items():
        print(f"         {k:20s}  →  {d}")

    # ── Extraction loop ───────────────────────────────────────────────────────
    n_ok = n_skip = n_empty_mesh = 0

    for sample in tqdm(samples, desc="Extracting"):
        img_path = sample["img_path"]
        filename = sample["filename"]
        stem     = Path(filename).stem

        if not os.path.exists(img_path):
            print(f"[WARN] Skipping {filename} – image not found at {img_path}")
            n_skip += 1
            continue

        try:
            feats = extract_one(
                img_path=img_path,
                pipeline=pipeline,
                args=args,
                save_geometry_tokens=save_geometry_tokens,
                save_occupancy_field=save_occupancy_field,
                save_mesh=save_mesh,
            )
        except Exception as e:
            import traceback
            print(f"[WARN] Skipping {filename} – extraction failed: {e}")
            traceback.print_exc()
            n_skip += 1
            continue

        # ── Persist each feature immediately (no in-memory accumulation) ──
        if save_geometry_tokens:
            save_npy(feat_dirs, "geometry_tokens", stem, feats["geometry_tokens"])

        if save_occupancy_field:
            save_npy(feat_dirs, "occupancy_field", stem, feats["occupancy_field"])

        if save_mesh:
            if feats["mesh_vertices"] is None:
                print(f"[WARN] {filename} produced an empty mesh – skipping mesh save.")
                n_empty_mesh += 1
            else:
                save_mesh_npz(feat_dirs, stem, feats["mesh_vertices"], feats["mesh_faces"])

        n_ok += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[INFO] Done.  {n_ok} images saved, {n_skip} skipped, "
          f"{n_empty_mesh} empty meshes.")
    print("[INFO] Output layout:")
    for k, d in feat_dirs.items():
        ext = ".npz" if k == "mesh" else ".npy"
        saved = list(d.glob(f"*{ext}"))
        if saved:
            if k != "mesh":
                shape_example = np.load(saved[0]).shape
                print(f"         {k:20s}  {len(saved):5d} files   "
                      f"shape example: {shape_example}")
            else:
                ex = np.load(saved[0])
                print(f"         {k:20s}  {len(saved):5d} files   "
                      f"vertices example: {ex['vertices'].shape}  "
                      f"faces example: {ex['faces'].shape}")


if __name__ == "__main__":
    main()

