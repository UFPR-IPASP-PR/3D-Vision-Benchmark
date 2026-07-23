#!/usr/bin/env python3
"""
extract_encoder_features.py
────────────────────────────
Feature extraction from DINOv2 or DepthAnything (DPT-DINOv2) encoders.

Mirrors the output layout of extract_gamba_features.py: one .npy file per
image, one sub-folder per feature type.

Models and feature types
────────────────────────

  --model dinov2
    Loads a bare DINOv2 ViT backbone from torch.hub (facebookresearch/dinov2).
    Taps the last transformer block via get_intermediate_layers(x, 1).

    dino_cls/      CLS token                         → (D,)        e.g. (1024,) for vitl
    dino_patches/  Patch tokens reshaped to (D, P, P) → (D, P, P)  e.g. (1024, 37, 37)

  --model depth_anything
    Loads DepthAnything (DPT_DINOv2) from HuggingFace Hub.
    Taps the LAST transformer block of the underlying DINOv2 encoder via
    model.pretrained.get_intermediate_layers(x, n=1, return_class_token=True)
    — i.e. the same "last layer" tap used for --model dinov2, just applied
    to the DINOv2 encoder embedded inside DepthAnything.

    da_cls/      CLS token of the last layer            → (D,)
    da_patches/  Patch tokens of the last layer,
                 reshaped to a spatial grid (no pooling) → (D, P, P)

    The token dimension D and patch grid P depend on the chosen encoder size:

      vits → D=384,  patch grid ~37×37 for 518-px input
      vitb → D=768,  patch grid ~37×37
      vitl → D=1024, patch grid ~37×37

Usage examples
──────────────
# DINOv2 (vitl, default)
python extract_encoder_features.py \\
    --model       dinov2 \\
    --ann_json    data/annotations.json \\
    --img_dir     data/images \\
    --output_dir  features/dinov2/

# DepthAnything vitb encoder
python extract_encoder_features.py \\
    --model       depth_anything \\
    --encoder     vitb \\
    --ann_json    data/annotations.json \\
    --img_dir     data/images \\
    --output_dir  features/depth_anything/
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# ── DepthAnything transform (mirrors depth_anything/app.py) ───────────────────
# Only imported when --model depth_anything is used; guarded at runtime.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Extract DINOv2 or DepthAnything encoder tokens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── what to extract ────────────────────────────────────────────────────
    p.add_argument(
        "--model", required=True, choices=["dinov2", "depth_anything"],
        help="Which encoder to load and extract from.",
    )
    p.add_argument(
        "--encoder", default="vitl", choices=["vits", "vitb", "vitl"],
        help="ViT backbone size. Applies to both model types.",
    )

    # ── data ───────────────────────────────────────────────────────────────
    p.add_argument("--ann_json",   required=True,
                   help="Path to annotation JSON (jbcs or lplc format).")
    p.add_argument("--img_dir",    required=True,
                   help="Root directory containing the images.")
    p.add_argument("--format",     default="jbcs", choices=["jbcs", "lplc"])
    p.add_argument("--output_dir", required=True,
                   help="Root directory under which per-feature-type folders are created.")

    # ── depth_anything-specific ────────────────────────────────────────────
    p.add_argument(
        "--da_checkpoint", default=None,
        help=(
            "Optional path to a local DepthAnything .pth checkpoint. "
            "If omitted, the model is downloaded from HuggingFace Hub "
            "(LiheYoung/depth_anything_{encoder}14)."
        ),
    )
    p.add_argument(
        "--da_input_size", default=518, type=int,
        help=(
            "Input resolution for DepthAnything preprocessing (must be a "
            "multiple of 14). Matches the default used in the official repo."
        ),
    )
    p.add_argument(
        "--da_localhub", action="store_true",
        help=(
            "Load DINOv2 backbone from a local torchhub clone "
            "(torchhub/facebookresearch_dinov2_main) instead of the internet. "
            "Only meaningful for --model depth_anything when --da_checkpoint "
            "is also supplied and you want fully offline inference."
        ),
    )

    # ── dinov2-specific ────────────────────────────────────────────────────
    p.add_argument(
        "--dino_input_size", default=518, type=int,
        help=(
            "Input resolution (square) for the DINOv2 preprocessing. "
            "Should be a multiple of 14 (the patch size). "
            "518 = 37 patches × 14 px is the default used by DepthAnything."
        ),
    )

    # ── execution ──────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Annotation loaders  (identical to extract_gamba_features.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_samples_jbcs(ann_json: str, img_dir: str) -> List[dict]:
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


def load_samples_lplc(ann_json: str, img_dir: str) -> List[dict]:
    with open(ann_json) as f:
        data = json.load(f)
    samples = []
    for fname in data:
        samples.append({
            "img_path": os.path.join(img_dir, fname),
            "filename": os.path.basename(fname),
        })
    return samples


def deduplicate(samples: List[dict]) -> List[dict]:
    seen, out = set(), []
    for s in samples:
        if s["filename"] not in seen:
            out.append(s)
            seen.add(s["filename"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_dinov2(encoder: str, device: str) -> torch.nn.Module:
    """Load a bare DINOv2 backbone from torch.hub."""
    hub_name = f"dinov2_{encoder}14"
    print(f"[INFO] Loading DINOv2 backbone '{hub_name}' from torch.hub …")
    model = torch.hub.load("facebookresearch/dinov2", hub_name)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_depth_anything(
    encoder: str,
    checkpoint: Optional[str],
    localhub: bool,
    device: str,
) -> torch.nn.Module:
    """
    Load DepthAnything (DPT_DINOv2).

    Priority:
      1. If --da_checkpoint is given, load DPT_DINOv2 directly from the local
         .pth file (matching the pattern in depth_anything/run.py).
      2. Otherwise, use DepthAnything.from_pretrained() from HuggingFace Hub.
    """
    if checkpoint is not None:
        # Local checkpoint path: replicate run.py build pattern
        print(f"[INFO] Building DepthAnything ({encoder}) from local checkpoint …")
        from depth_anything.dpt import DPT_DINOv2
        model_configs = {
            "vits": {"features": 64,  "out_channels": [48,  96,  192,  384]},
            "vitb": {"features": 128, "out_channels": [96,  192, 384,  768]},
            "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        cfg = model_configs[encoder]
        model = DPT_DINOv2(
            encoder=encoder,
            features=cfg["features"],
            out_channels=cfg["out_channels"],
            localhub=localhub,
        )
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print(f"[INFO] Loading DepthAnything ({encoder}) from HuggingFace Hub …")
        from depth_anything.dpt import DepthAnything
        model = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{encoder}14")

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_dinov2(img_path: str, input_size: int, device: str) -> torch.Tensor:
    """
    Load an image (RGBA or RGB) and prepare it for a DINOv2 backbone.

    Matches the RGBA compositing + bilinear resize used in
    extract_gamba_features.py's preprocess_image(), then applies ImageNet
    normalisation.

    Returns: (1, 3, input_size, input_size) float32 tensor on `device`.
    """
    img = np.array(Image.open(img_path).convert("RGBA"), dtype=np.float32) / 255.0
    rgb   = img[..., :3]
    alpha = img[..., 3:4]
    image = rgb * alpha + (1.0 - alpha)                     # white-background composite

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    tensor = F.interpolate(
        tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
    )

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)
    tensor = (tensor.to(device) - mean) / std

    return tensor  # (1, 3, H, W)


def preprocess_for_depth_anything(
    img_path: str, input_size: int, device: str
) -> torch.Tensor:
    """
    Load an image and prepare it for DepthAnything.

    Mirrors the transform pipeline in depth_anything/app.py:
      - Convert BGR→RGB, scale to [0, 1]
      - Resize so the short side ≥ input_size, keeping aspect ratio,
        and make both dims a multiple of 14 (using cv2.INTER_CUBIC)
      - ImageNet normalise
      - Add batch dim

    Returns: (1, 3, H', W') float32 tensor on `device`, where H', W' are
    multiples of 14 with short side ≥ input_size.
    """
    raw = cv2.imread(img_path)
    if raw is None:
        raise FileNotFoundError(f"cv2.imread failed: {img_path}")

    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w  = image.shape[:2]

    # Scale so the shorter side reaches input_size (lower_bound), then round
    # both dimensions to the nearest multiple of 14.
    scale = input_size / min(h, w)
    new_h = math.ceil(h * scale / 14) * 14
    new_w = math.ceil(w * scale / 14) * 14
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    image = (image - mean) / std

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor  # (1, 3, H', W')


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Token extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reshape_patch_tokens(patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    Reshape flat patch sequence (L, D) → (D, P, P) when L is a perfect square,
    otherwise keep as (D, L) so the caller can still save it.
    """
    L, D = patch_tokens.shape
    P = int(round(math.sqrt(L)))
    if P * P == L:
        return patch_tokens.permute(1, 0).reshape(D, P, P)   # (D, P, P)
    return patch_tokens.permute(1, 0)                         # (D, L) fallback


@torch.no_grad()
def extract_dinov2_tokens(
    model: torch.nn.Module,
    tensor: torch.Tensor,           # (1, 3, H, W)
) -> Dict[str, np.ndarray]:
    """
    Run a bare DINOv2 model and return:
      dino_cls     – CLS token                  → (D,)
      dino_patches – spatial patch tokens        → (D, P, P)
    """
    # get_intermediate_layers with n=1 returns the last block's output.
    # return_class_token=True → list of (patch_tokens, cls_token) tuples.
    features = model.get_intermediate_layers(tensor, n=1, return_class_token=True)

    patch_tokens, cls_token = features[0]          # (1, L, D), (1, D)
    cls_token    = cls_token[0].cpu().float()       # (D,)
    patch_tokens = patch_tokens[0].cpu().float()    # (L, D)

    return {
        "dino_cls":     torch.nan_to_num(cls_token,    nan=0.0).numpy(),
        "dino_patches": torch.nan_to_num(_reshape_patch_tokens(patch_tokens), nan=0.0).numpy(),
    }


@torch.no_grad()
def extract_depth_anything_tokens(
    model: torch.nn.Module,
    tensor: torch.Tensor,           # (1, 3, H, W)
) -> Dict[str, np.ndarray]:
    """
    Run DepthAnything's pretrained DINOv2 backbone (model.pretrained) and
    return ONLY the last transformer block's output:

      da_cls     – CLS token of the last layer    → (D,)
      da_patches – patch tokens of the last layer,
                   reshaped to (D, P, P)          → (D, P, P)

    This mirrors extract_dinov2_tokens() above, but applied to the DINOv2
    ViT embedded inside DepthAnything (model.pretrained) rather than a bare
    DINOv2 checkpoint.
    """
    # model.pretrained is the bare DINOv2 ViT inside DepthAnything.
    # n=1 → only the last transformer block's output.
    features = model.pretrained.get_intermediate_layers(
        tensor, n=1, return_class_token=True
    )   # [(patch_tokens (1,L,D), cls_token (1,D))]

    patch_tokens, cls_token = features[0]
    cls_token    = cls_token[0].cpu().float()       # (D,)
    patch_tokens = patch_tokens[0].cpu().float()    # (L, D)

    return {
        "da_cls":     torch.nan_to_num(cls_token, nan=0.0).numpy(),
        "da_patches": torch.nan_to_num(_reshape_patch_tokens(patch_tokens), nan=0.0).numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  I/O helpers  (identical to extract_gamba_features.py)
# ─────────────────────────────────────────────────────────────────────────────

def make_feature_dirs(output_dir: str, keys: List[str]) -> Dict[str, Path]:
    dirs = {}
    for k in keys:
        d = Path(output_dir) / k
        d.mkdir(parents=True, exist_ok=True)
        dirs[k] = d
    return dirs


def save_feature(feat_dirs: Dict[str, Path], key: str, stem: str, arr: np.ndarray):
    path = feat_dirs[key] / f"{stem}.npy"
    np.save(path, arr)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = args.device

    # ── Annotations ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading annotations (format={args.format}) …")
    load_fn = load_samples_jbcs if args.format == "jbcs" else load_samples_lplc
    samples = deduplicate(load_fn(args.ann_json, args.img_dir))
    print(f"[INFO] {len(samples)} unique images to process.")

    # ── Model + extraction fn ─────────────────────────────────────────────────
    if args.model == "dinov2":
        model = build_dinov2(args.encoder, device)
        keys  = ["dino_cls", "dino_patches"]

        def extract_fn(img_path: str) -> Dict[str, np.ndarray]:
            tensor = preprocess_for_dinov2(img_path, args.dino_input_size, device)
            return extract_dinov2_tokens(model, tensor)

    else:  # depth_anything
        model = build_depth_anything(
            args.encoder, args.da_checkpoint, args.da_localhub, device
        )
        keys = ["da_cls", "da_patches"]

        def extract_fn(img_path: str) -> Dict[str, np.ndarray]:
            tensor = preprocess_for_depth_anything(img_path, args.da_input_size, device)
            return extract_depth_anything_tokens(model, tensor)

    # ── Output folders ────────────────────────────────────────────────────────
    feat_dirs = make_feature_dirs(args.output_dir, keys)
    print(f"[INFO] Feature folders created under {args.output_dir}")
    for k, d in feat_dirs.items():
        print(f"         {k:30s}  →  {d}")

    # ── Extraction loop ───────────────────────────────────────────────────────
    n_ok = n_skip = n_resume = 0

    for sample in tqdm(samples, desc="Extracting"):
        img_path = sample["img_path"]
        filename = sample["filename"]
        stem     = Path(filename).stem

        # Resume: skip if every output file already exists.
        if all((feat_dirs[k] / f"{stem}.npy").exists() for k in keys):
            n_resume += 1
            continue

        try:
            feats = extract_fn(img_path)
        except Exception as e:
            print(f"[WARN] Skipping {filename} – extraction failed: {e}")
            n_skip += 1
            continue

        for k in keys:
            if k in feats:
                save_feature(feat_dirs, k, stem, feats[k].astype(np.float32))
            else:
                print(f"  [WARN] Key '{k}' missing for {filename}")

        n_ok += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"\n[INFO] Done. {n_ok} extracted, "
        f"{n_resume} already existed (skipped), "
        f"{n_skip} failed."
    )
    print("[INFO] Output layout:")
    for k, d in feat_dirs.items():
        saved = list(d.glob("*.npy"))
        if saved:
            shape_example = np.load(saved[0]).shape
            print(
                f"         {k:30s}  {len(saved):5d} files   "
                f"shape example: {shape_example}"
            )


if __name__ == "__main__":
    main()
