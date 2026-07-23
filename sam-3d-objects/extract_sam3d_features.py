#!/usr/bin/env python3
"""
extract_sam3d_features.py
Multi-stage feature extraction from the SAM-3D (InferencePipelinePointMap) pipeline.
Mirrors the structure of the DINOv2 extract_features.py but taps into the intermediate
representations produced at each stage of the 3D reconstruction pipeline.

Feature files are written under --output_dir, one sub-folder per feature type:
  pointmap/   – depth-model XYZ pointmap, unpooled                             → (3, H, W)
  ss/         – sparse-structure preprocessor image, unpooled                  → (C_ss, H_ss, W_ss)
  slat/       – structured-latent preprocessor image, unpooled                 → (C_slat, H_slat, W_slat)
  concat/     – np.concatenate([flatten(pointmap), flatten(slat)])             → (3*H*W + C_slat*H_slat*W_slat,)
  shape_latent/  [--extract_latents]  pre-threshold shape latent               → (4096, 8)
  latent/        [--extract_latents]  per-voxel sparse latent matrix           → (N_voxels, D)

Each feature is saved as a separate  <image_stem>.npy  file inside its folder,
so no large in-memory accumulator is ever needed.

Usage example
─────────────
# Fast (no diffusion passes):
python extract_sam3d_features.py \
    --ann_json   data/annotations.json \
    --img_dir    data/images \
    --mask_dir   data/masks \
    --config     checkpoints/hf/pipeline.yaml \
    --output_dir features/

# With sparse latent features (runs full stage-1 + stage-2 diffusion – slow):
python extract_sam3d_features.py ... --extract_latents --seed 42
"""
import os
import sys
import json
import argparse
from pathlib import Path

# SAM-3D puts its inference helpers under notebook/
sys.path.append("notebook")

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

# inference.py lives in notebook/
from inference import Inference, load_mask


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Arguments
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description="Extract multi-stage features from the SAM-3D pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── data ──────────────────────────────────────────────────────────────────
    p.add_argument("--ann_json",  required=True,
                   help="Path to the annotation JSON file")
    p.add_argument("--img_dir",   required=True,
                   help="Root directory containing the images")
    p.add_argument("--mask_dir",  default=None,
                   help="Optional directory with per-image mask PNGs (same basename as "
                        "image). A full-image (all-ones) mask is used when a file is "
                        "absent or this flag is omitted.")
    p.add_argument("--format",    default="jbcs", choices=["jbcs", "lplc"],
                   help="Annotation format: 'jbcs' (flat list) or 'lplc' (nested dict)")
    p.add_argument("--output_dir", required=True,
                   help="Root directory under which per-feature-type folders are created")
    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--config",   default="checkpoints/hf/pipeline.yaml",
                   help="Path to pipeline.yaml used by Inference()")
    p.add_argument("--compile",  action="store_true",
                   help="Torch-compile the pipeline (faster after warmup)")
    # ── feature options ───────────────────────────────────────────────────────
    p.add_argument("--extract_latents", action="store_true",
                   help="Also run full stage-1 + stage-2 diffusion to extract sparse "
                        "latent features (one diffusion pass per image – slow)")
    p.add_argument("--seed",      type=int, default=42,
                   help="RNG seed used when --extract_latents is set")
    # ── execution ─────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Annotation loaders  (same contract as extract_features.py)
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
def load_rgba(img_path: str, mask_path: str | None) -> np.ndarray:
    """
    Return an H×W×4 uint8 RGBA array.
    The alpha channel encodes the object mask (255 = object, 0 = background).
    Falls back to a full-image (all-ones) mask when mask_path is absent.
    """
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    h, w = img.shape[:2]
    if mask_path and os.path.exists(mask_path):
        mask = load_mask(mask_path)           # bool (H, W), as returned by inference.py
    else:
        mask = np.ones((h, w), dtype=bool)    # full-image mask
    alpha = mask.astype(np.uint8) * 255
    return np.concatenate([img, alpha[..., None]], axis=-1)


def preserve_channels(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (C, H, W) or (1, C, H, W) tensor to a numpy array, preserving 
    all channels and spatial dimensions (no pooling).

    Returns a float32 numpy array of shape (C, H, W).
    NaN values (present in clipped pointmaps) are zeroed before extracting.
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)                       # (1,C,H,W) → (C,H,W)
    t = torch.nan_to_num(tensor.float(), nan=0.0)
    return t.cpu().numpy().astype(np.float32)            # (C, H, W)


def make_feature_dirs(output_dir: str, keys: list[str]) -> dict[str, Path]:
    """Create one sub-folder per feature key; return a {key: Path} mapping."""
    dirs = {}
    for k in keys:
        d = Path(output_dir) / k
        d.mkdir(parents=True, exist_ok=True)
        dirs[k] = d
    return dirs


def save_feature(feat_dirs: dict, key: str, stem: str, arr: np.ndarray):
    """Save arr as  <feat_dirs[key]>/<stem>.npy."""
    path = feat_dirs[key] / f"{stem}.npy"
    np.save(path, arr)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Per-image extraction
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_one(
    rgba:            np.ndarray,
    pipeline,                        # InferencePipelinePointMap
    extract_latents: bool,
    seed:            int | None,
) -> dict:
    """
    Drive a single RGBA image through each pipeline stage and return a dict of
    numpy arrays:

      "pointmap"     – depth-model output                         shape (3, H, W)
      "ss"           – ss_preprocessor output                     shape (C_ss, H_ss, W_ss)
      "slat"         – slat_preprocessor output                   shape (C_slat, H_slat, W_slat)
      "concat"       – flatten(pointmap) ++ flatten(slat)         1D array shape (3*H*W + C_slat*H_slat*W_slat,)
      "shape_latent" – pre-threshold shape latent (if requested)  shape (4096, 8)
      "latent"       – per-voxel sparse latent matrix (if req.)   shape (N_voxels, D)
    """
    results = {}

    # ── Stage 0: depth model → pointmap ──────────────────────────────────────
    # compute_pointmap accepts an RGBA uint8 numpy array; returns a dict with
    # "pointmap": (3, H, W) on device (and other entries we don't need here).
    pointmap_dict = pipeline.compute_pointmap(rgba)
    results["pointmap"] = preserve_channels(pointmap_dict["pointmap"])  # (3, H, W)

    # ── Stage 1a: sparse-structure preprocessor ───────────────────────────────
    # preprocess_image crops/resizes image+pointmap jointly; "image" is what
    # the ss_generator conditions on.   ss_input["image"]: (1, C_ss, H_ss, W_ss)
    ss_input = pipeline.preprocess_image(
        rgba,
        pipeline.ss_preprocessor,
        pointmap=pointmap_dict["pointmap"],     # (3, H, W) on device
    )
    results["ss"] = preserve_channels(ss_input["image"])    # (C_ss, H_ss, W_ss)

    # ── Stage 1b: structured-latent preprocessor ─────────────────────────────
    # Uses a separate preprocessor; only needs RGB+mask.
    # slat_input["image"]: (1, C_slat, H_slat, W_slat)
    slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
    results["slat"] = preserve_channels(slat_input["image"])  # (C_slat, H_slat, W_slat)

    # ── Concat: pointmap ⊕ slat ──────────────────────────────────────────────
    # Flatten before joining because the two preprocessors may use different
    # spatial resolutions, making direct stacking unsafe.
    results["concat"] = np.concatenate(
        [results["pointmap"].ravel(), results["slat"].ravel()]
    )

    # ── Stage 2+3: sparse structure → sparse latent  (optional, expensive) ───
    if extract_latents:
        if seed is not None:
            torch.manual_seed(seed)

        # Stage 1 diffusion: occupied voxel coordinates + pose prediction.
        ss_return = pipeline.sample_sparse_structure(ss_input)
        coords    = ss_return["coords"]          # (N, 4) int – batch + XYZ indices

        # Shape latent: dense (B, 4096, 8) tensor produced by ss_generator
        # before decoding / thresholding.  Saved as (4096, 8) per image.
        results["shape_latent"] = (
            ss_return["shape"].squeeze(0).cpu().float().numpy()
        )

        # Stage 2 diffusion: per-voxel structured latent features.
        # slat.feats: (N_voxels, D) – kept as a full matrix, no pooling.
        slat = pipeline.sample_slat(slat_input, coords)
        results["latent"] = slat.feats.float().cpu().numpy()   # (N_voxels, D)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()

    # ── Annotations ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading annotations  (format={args.format}) …")
    load_fn = load_samples_jbcs if args.format == "jbcs" else load_samples_lplc
    samples = deduplicate(load_fn(args.ann_json, args.img_dir))
    print(f"[INFO] {len(samples)} unique images to process.")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    print("[INFO] Building SAM-3D pipeline …")
    inference = Inference(args.config, compile=args.compile)
    # Reach directly into the InferencePipelinePointMap so we can call
    # individual stages without running the full end-to-end pass.
    pipeline = inference._pipeline

    # ── Output folders (one per feature type) ────────────────────────────────
    keys = ["pointmap", "ss", "slat", "concat"]
    if args.extract_latents:
        keys.extend(["shape_latent", "latent"])

    feat_dirs = make_feature_dirs(args.output_dir, keys)
    print(f"[INFO] Feature folders created under  {args.output_dir}")
    for k, d in feat_dirs.items():
        print(f"         {k:15s}  →  {d}")

    # ── Extraction loop ───────────────────────────────────────────────────────
    n_ok = n_skip = 0

    for sample in tqdm(samples, desc="Extracting"):
        img_path = sample["img_path"]
        filename = sample["filename"]
        stem     = Path(filename).stem

        # Resolve mask path (None → full-image mask inside load_rgba)
        mask_path = None
        if args.mask_dir:
            mask_path = os.path.join(args.mask_dir, stem + ".png")

        # Load image → RGBA
        try:
            rgba = load_rgba(img_path, mask_path)
        except Exception as e:
            print(f"[WARN] Skipping {filename} – could not load image: {e}")
            n_skip += 1
            continue

        # Run pipeline stages
        try:
            feats = extract_one(
                rgba,
                pipeline,
                extract_latents = args.extract_latents,
                seed            = args.seed,
            )
        except Exception as e:
            print(f"[WARN] Skipping {filename} – extraction failed: {e}")
            n_skip += 1
            continue

        # Persist each feature immediately (no in-memory accumulation)
        for k in keys:
            save_feature(feat_dirs, k, stem, feats[k].astype(np.float32))

        n_ok += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[INFO] Done.  {n_ok} images saved, {n_skip} skipped.")
    print("[INFO] Output layout:")
    for k, d in feat_dirs.items():
        saved = list(d.glob("*.npy"))
        if saved:
            shape_example = np.load(saved[0]).shape
            print(f"         {k:15s}  {len(saved):5d} files   "
                  f"shape example: {shape_example}")


if __name__ == "__main__":
    main()