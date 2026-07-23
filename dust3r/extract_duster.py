#!/usr/bin/env python3
"""
extract_dust3r_features.py

Extract raw ViT encoder features from a DUSt3R model — the feat1/feat2
tensors that sit between the patch embedding and the cross-attention decoder.
No pooling or dimensional reduction is applied.

Output layout under --output_dir:
  encoder/   – one <stem>.npy per image               shape (S, D_enc)
  pos/        – one <stem>.npy per image               shape (S, 2)

Where:
  S     = (H_resized / patch_size) × (W_resized / patch_size)   — varies per image
  D_enc = encoder embedding dimension (1024 for ViT-Large)
  pos   = integer (row, col) grid coordinates for each token
          saved separately so downstream code can reconstruct the spatial layout

Usage
─────
python extract_dust3r_features.py \
    --ann_json   data/annotations.json \
    --img_dir    data/images \
    --weights    naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt \
    --output_dir features/dust3r_enc/
"""

import os
import sys
import json
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as tvf

# ── DUSt3R imports ────────────────────────────────────────────────────────────
# Assumes the repo is on the path, either installed or via sys.path.append.
from dust3r.model import AsymmetricCroCo3DStereo


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Extract raw ViT encoder features from DUSt3R.",
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
                   help="Root directory under which feature folders are created")

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--weights",
                   default="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                   help="HuggingFace repo id or local .pth path for the model weights. ")
    p.add_argument("--image_size", type=int, default=224,
                   help="Longest-side target for hard resizing. "
                        "Must match what the checkpoint was trained with (512 or 224).")

    # ── execution ─────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Annotation loaders
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
# 3.  Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_one(img_path: str, image_size: int, patch_size: int) -> dict:
    """
    Hard resize to a square (image_size x image_size), ignoring aspect ratio.
    """
    # Ensure the size is a multiple of patch_size (ViT requirement)
    safe_size = (image_size // patch_size) * patch_size
    
    img = Image.open(img_path).convert("RGB")
    
    transform = tvf.Compose([
        tvf.Resize((safe_size, safe_size)),
        tvf.ToTensor(),
        tvf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    
    # Apply transform and add batch dimension -> shape: (1, 3, safe_size, safe_size)
    img_tensor = transform(img).unsqueeze(0) 
    
    # DUSt3R requires true_shape for positional embeddings
    true_shape = np.array([[safe_size, safe_size]], dtype=np.int32)
    
    return {
        "img": img_tensor,
        "true_shape": true_shape
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Feature directories
# ─────────────────────────────────────────────────────────────────────────────

def make_feature_dirs(output_dir: str, keys: list) -> dict:
    dirs = {}
    for k in keys:
        d = Path(output_dir) / k
        d.mkdir(parents=True, exist_ok=True)
        dirs[k] = d
    return dirs


def save_feature(feat_dirs: dict, key: str, stem: str, arr: np.ndarray):
    path = feat_dirs[key] / f"{stem}.npy"
    np.save(path, arr)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Per-image extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_one(view: dict, model, device: str) -> dict:
    """
    Run a single preprocessed view through the ViT encoder and return the
    raw token sequence and its position grid.

    The encoder (_encode_image) is called directly, bypassing the decoder and
    heads entirely.  This is sound because:
      - The encoder is fully self-contained; it never sees the second image.
      - _encode_image is exactly the code that produces feat1/feat2 in the
        normal forward pass, so the features are identical to what the decoder
        would receive.

    Returns
    -------
    "encoder"  : float32 numpy array  (S, D_enc)
                 The normalised token sequence output by enc_norm.
                 S = (H/patch_size) * (W/patch_size), varies per image.
    "pos"      : int32 numpy array    (S, 2)
                 The (row, col) integer grid coordinates generated by
                 patch_embed's PositionGetter for each token.  Saved
                 separately so downstream code can recover spatial layout
                 or replicate RoPE positions without rerunning the model.
    """
    img = view["img"].to(device)                          # (1, 3, H, W)
    true_shape = torch.from_numpy(view["true_shape"]).to(device)  # (1, 2)

    # _encode_image returns (tokens, pos, None)
    #   tokens : (B, S, D_enc)  — full sequence, enc_norm already applied
    #   pos    : (B, S, 2)      — integer (row, col) for each token (RoPE grid)
    tokens, pos, _ = model._encode_image(img, true_shape)

    return {
        # squeeze batch dim → (S, D_enc) / (S, 2)
        "encoder": tokens.squeeze(0).cpu().float().numpy(),
        "pos":     pos.squeeze(0).cpu().numpy().astype(np.int32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = args.device

    # ── Annotations ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading annotations  (format={args.format}) …")
    load_fn = load_samples_jbcs if args.format == "jbcs" else load_samples_lplc
    samples = deduplicate(load_fn(args.ann_json, args.img_dir))
    print(f"[INFO] {len(samples)} unique images to process.")

    # ── Model ─────────────────────────────────────────────────────────────────
    # .eval() + no_grad() ensure BatchNorm/Dropout are in inference mode and
    # no gradient graph is built, keeping memory usage minimal.
    print(f"[INFO] Loading model from  {args.weights} …")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.weights).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    print(f"[INFO] patch_size={patch_size},  "
          f"enc_embed_dim={model.enc_embed_dim},  "
          f"enc_depth={model.enc_depth}")

    # ── Output folders ────────────────────────────────────────────────────────
    feat_dirs = make_feature_dirs(args.output_dir, ["encoder", "pos"])
    print(f"[INFO] Saving features to  {args.output_dir}")

    # ── Extraction loop ───────────────────────────────────────────────────────
    n_ok = n_skip = 0

    for sample in tqdm(samples, desc="Extracting"):
        img_path = sample["img_path"]
        filename = sample["filename"]
        stem     = Path(filename).stem

        # Load + preprocess using the exact same transform as DUSt3R inference
        try:
            view = load_one(img_path, args.image_size, patch_size)
        except Exception as e:
            print(f"[WARN] Skipping {filename} – could not load image: {e}")
            n_skip += 1
            continue

        # Encode
        try:
            feats = extract_one(view, model, device)
        except Exception as e:
            print(f"[WARN] Skipping {filename} – encoding failed: {e}")
            n_skip += 1
            continue

        # Save immediately — no in-memory accumulation
        # Individual files are necessary here (not a single npz) because S
        # varies across images depending on their aspect ratio after resizing.
        save_feature(feat_dirs, "encoder", stem, feats["encoder"])
        save_feature(feat_dirs, "pos",     stem, feats["pos"])
        n_ok += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[INFO] Done.  {n_ok} saved,  {n_skip} skipped.")
    print("[INFO] Output layout:")
    for k, d in feat_dirs.items():
        saved = list(d.glob("*.npy"))
        if saved:
            shape_example = np.load(saved[0]).shape
            print(f"         {k:10s}  {len(saved):5d} files   "
                  f"example shape: {shape_example}")


if __name__ == "__main__":
    main()