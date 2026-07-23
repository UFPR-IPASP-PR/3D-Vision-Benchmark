#!/usr/bin/env python3
"""
extract_encoder_features.py
────────────────────────────
Feature extraction from DINOv2, DepthAnything V2, or DepthAnything 3 encoders.

Mirrors the output layout of extract_gamba_features.py: one .npy file per
image, one sub-folder per feature type.

Models and feature types
────────────────────────

  --model dinov2
    Loads a bare DINOv2 ViT backbone from torch.hub (facebookresearch/dinov2).
    Taps the last transformer block via get_intermediate_layers(x, 1).

    dino_cls/      CLS token                         → (D,)        e.g. (1024,) for vitl
    dino_patches/  Patch tokens reshaped to (D, P, P) → (D, P, P)  e.g. (1024, 37, 37)

  --model depth_anything_v2
    Loads DepthAnythingV2 from a local .pth checkpoint.
    Taps the last block of model.pretrained (DinoVisionTransformer) via
    get_intermediate_layers(x, n=1, return_class_token=True).

    da_cls/      CLS token of the last layer            → (D,)
    da_patches/  Patch tokens, reshaped to (D, P, P)    → (D, P, P)

  --model depth_anything_3
    Loads DepthAnything3 via DepthAnything3.from_pretrained() (HuggingFace Hub)
    or by model name via DepthAnything3(model_name=...) for locally-registered
    configs.

    Depth Anything 3 destroys the semantic CLS token at layer `alt_start` 
    to inject a geometric camera condition. To retain classification power, 
    we bypass DA3's wrapper and extract features from the transformer block 
    immediately preceding `alt_start` (alt_start - 1).

    da3_cls/     Semantic CLS token extracted right before alt_start  → (D,)
    da3_patches/ Patch tokens, extracted right before alt_start       → (D, Ph, Pw)

    Preprocessing uses DA3's own InputProcessor (PIL RGB, aspect-ratio
    preserving upper_bound_resize to --da3_process_res, divisible by 14).

    The token dimension D and patch grid Ph×Pw depend on the model preset:
      da3-large  (vitl) → D=1024, grid depends on input resolution
      da3-giant  (vitg) → D=1536

Usage examples
──────────────
# DINOv2 vitl
python extract_encoder_features.py \
    --model       dinov2 \
    --ann_json    data/annotations.json \
    --img_dir     data/images \
    --output_dir  features/dinov2/

# DepthAnything V2 vitb
python extract_encoder_features.py \
    --model          depth_anything_v2 \
    --encoder        vitb \
    --da_checkpoint  checkpoints/depth_anything_v2_vitb.pth \
    --ann_json       data/annotations.json \
    --img_dir        data/images \
    --output_dir     features/depth_anything_v2/

# DepthAnything 3 from HuggingFace Hub
python extract_encoder_features.py \
    --model           depth_anything_3 \
    --da3_model_name  hf-repo/da3-large \
    --da3_from_hf \
    --ann_json        data/annotations.json \
    --img_dir         data/images \
    --output_dir      features/depth_anything_3/
"""

import os
import sys

# Auto-inject the 'src' directory so the depth_anything_3 package can be found
# without requiring a full `pip install -e .` or setting PYTHONPATH manually.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Extract DINOv2, DepthAnything V2, or DepthAnything 3 encoder tokens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── what to extract ────────────────────────────────────────────────────
    p.add_argument(
        "--model", required=True,
        choices=["dinov2", "depth_anything_v2", "depth_anything_3"],
        help="Which encoder to load and extract from.",
    )
    p.add_argument(
        "--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"],
        help=(
            "ViT backbone size. Used for --model dinov2 and depth_anything_v2. "
            "'vitg' is only valid for --model depth_anything_v2. "
            "Ignored for --model depth_anything_3 (backbone is determined by the preset)."
        ),
    )

    # ── data ───────────────────────────────────────────────────────────────
    p.add_argument("--ann_json",   required=True,
                   help="Path to annotation JSON (jbcs or lplc format).")
    p.add_argument("--img_dir",    required=True,
                   help="Root directory containing the images.")
    p.add_argument("--format",     default="jbcs", choices=["jbcs", "lplc"])
    p.add_argument("--output_dir", required=True,
                   help="Root directory under which per-feature-type folders are created.")

    # ── depth_anything_v2-specific ─────────────────────────────────────────
    p.add_argument(
        "--da_checkpoint", default=None,
        help=(
            "Path to a local DepthAnything V2 .pth checkpoint file "
            "(e.g. checkpoints/depth_anything_v2_vitl.pth). "
            "Required when --model depth_anything_v2."
        ),
    )
    p.add_argument(
        "--da_input_size", default=518, type=int,
        help=(
            "Input resolution for DepthAnythingV2 preprocessing (must be a "
            "multiple of 14). Default matches the official V2 repo."
        ),
    )

    # ── depth_anything_3-specific ──────────────────────────────────────────
    p.add_argument(
        "--da3_model_name", default="da3-large",
        help=(
            "Model preset name for DepthAnything3. "
            "When --da3_from_hf is set, this is treated as a HuggingFace repo id "
            "(e.g. 'ByteDance/depth-anything-3-large'). "
            "Otherwise it is looked up in the local MODEL_REGISTRY "
            "(must match a YAML config name, e.g. 'da3-large', 'da3-giant'). "
            "Run `python -c \"from depth_anything_3.registry import MODEL_REGISTRY; "
            "print(list(MODEL_REGISTRY))'` to list available local presets."
        ),
    )
    p.add_argument(
        "--da3_from_hf", action="store_true",
        help=(
            "Load DepthAnything3 via DepthAnything3.from_pretrained() from HuggingFace Hub "
            "using --da3_model_name as the repo id. "
            "Without this flag the model is built from the local MODEL_REGISTRY."
        ),
    )
    p.add_argument(
        "--da3_process_res", default=504, type=int,
        help=(
            "Processing resolution for DepthAnything3 InputProcessor. "
            "The longest side of each image is rescaled to this value, "
            "then both dimensions are rounded to the nearest multiple of 14. "
            "Default matches the DA3 api.py default."
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
# 2.  Annotation loaders
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


# Model configs from the official V2 run.py / app.py
_DAV2_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,   96,   192,  384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,   192,  384,  768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256,  512,  1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def build_depth_anything_v2(
    encoder: str,
    checkpoint: str,
    device: str,
) -> torch.nn.Module:
    """
    Load DepthAnythingV2 from a local .pth checkpoint.

    Replicates the exact build pattern from the official V2 run.py:
      model = DepthAnythingV2(**model_configs[encoder])
      model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
      model = model.to(device).eval()
    """
    if checkpoint is None:
        raise ValueError(
            "--da_checkpoint is required for --model depth_anything_v2. "
            "DepthAnything V2 has no HuggingFace Hub pretrained loader."
        )

    from depth_anything_v2.dpt import DepthAnythingV2

    cfg = _DAV2_MODEL_CONFIGS[encoder]
    print(f"[INFO] Building DepthAnythingV2 ({encoder}) from checkpoint: {checkpoint} …")
    model = DepthAnythingV2(**cfg)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_depth_anything_3(
    model_name: str,
    from_hf: bool,
    device: str,
) -> torch.nn.Module:
    """
    Load DepthAnything3 (DepthAnything3Net wrapped in DepthAnything3 API class).
    """
    from depth_anything_3.api import DepthAnything3

    if from_hf:
        print(f"[INFO] Loading DepthAnything3 from HuggingFace Hub: {model_name} …")
        model = DepthAnything3.from_pretrained(model_name)
    else:
        print(f"[INFO] Building DepthAnything3 from local registry preset: {model_name} …")
        model = DepthAnything3(model_name=model_name)

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_dinov2(img_path: str, input_size: int, device: str) -> torch.Tensor:
    """
    Load an image (RGBA or RGB) and prepare it for a bare DINOv2 backbone.
    """
    img = np.array(Image.open(img_path).convert("RGBA"), dtype=np.float32) / 255.0
    rgb   = img[..., :3]
    alpha = img[..., 3:4]
    image = rgb * alpha + (1.0 - alpha)   # white-background composite

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    tensor = F.interpolate(
        tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)
    return (tensor.to(device) - mean) / std   # (1, 3, H, W)


def preprocess_for_depth_anything_v2(
    img_path: str,
    model: torch.nn.Module,
    input_size: int,
    device: str,
) -> torch.Tensor:
    """
    Load an image and prepare it for DepthAnythingV2.
    """
    raw = cv2.imread(img_path)
    if raw is None:
        raise FileNotFoundError(f"cv2.imread failed: {img_path}")
    tensor, _ = model.image2tensor(raw, input_size)
    return tensor.to(device)


def preprocess_for_depth_anything_3(
    img_path: str,
    input_processor,          # depth_anything_3.utils.io.input_processor.InputProcessor
    process_res: int,
    device: str,
) -> torch.Tensor:
    """
    Load an image and prepare it for DepthAnything3Net.
    Returns: (1, 1, 3, H', W') float32 tensor on `device`.
    """
    tensor, _extrinsics, _intrinsics = input_processor(
        image=[img_path],
        extrinsics=None,
        intrinsics=None,
        process_res=process_res,
        process_res_method="upper_bound_resize",
        num_workers=1,
        sequential=True,
    )
    return tensor.unsqueeze(0).to(device)   # (1, 1, 3, H', W')


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Token extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reshape_patch_tokens(patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    Reshape a flat patch sequence (L, D) → (D, Ph, Pw).
    """
    L, D = patch_tokens.shape
    P = int(round(math.sqrt(L)))
    if P * P == L:
        return patch_tokens.permute(1, 0).reshape(D, P, P)   # (D, P, P)
    return patch_tokens.permute(1, 0)                         # (D, L) fallback


def _reshape_patch_tokens_hw(
    patch_tokens: torch.Tensor,   # (L, D)
    H: int,
    W: int,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Reshape patch tokens to a (D, Ph, Pw) spatial grid given the image H, W
    and the backbone patch size. Used for DA3 where images may be non-square.
    """
    Ph = H // patch_size
    Pw = W // patch_size
    D  = patch_tokens.shape[-1]
    return patch_tokens.permute(1, 0).reshape(D, Ph, Pw)


@torch.no_grad()
def extract_dinov2_tokens(
    model: torch.nn.Module,
    tensor: torch.Tensor,       # (1, 3, H, W)
) -> Dict[str, np.ndarray]:
    features = model.get_intermediate_layers(tensor, n=1, return_class_token=True)
    patch_tokens, cls_token = features[0]
    cls_token    = cls_token[0].cpu().float()       # (D,)
    patch_tokens = patch_tokens[0].cpu().float()    # (L, D)
    return {
        "dino_cls":     torch.nan_to_num(cls_token, nan=0.0).numpy(),
        "dino_patches": torch.nan_to_num(_reshape_patch_tokens(patch_tokens), nan=0.0).numpy(),
    }


@torch.no_grad()
def extract_depth_anything_v2_tokens(
    model: torch.nn.Module,
    tensor: torch.Tensor,       # (1, 3, H, W)
) -> Dict[str, np.ndarray]:
    features = model.pretrained.get_intermediate_layers(
        tensor, n=1, return_class_token=True
    )
    patch_tokens, cls_token = features[0]
    cls_token    = cls_token[0].cpu().float()
    patch_tokens = patch_tokens[0].cpu().float()
    return {
        "da_cls":     torch.nan_to_num(cls_token, nan=0.0).numpy(),
        "da_patches": torch.nan_to_num(_reshape_patch_tokens(patch_tokens), nan=0.0).numpy(),
    }


@torch.no_grad()
def extract_depth_anything_3_tokens(
    model: torch.nn.Module,     # DepthAnything3 API wrapper
    tensor: torch.Tensor,       # (1, 1, 3, H, W)  — B=1, N=1 views
) -> Dict[str, np.ndarray]:
    """
    Run DepthAnything3Net's backbone and return tokens from the block right
    before the camera condition token overwrites the CLS token.

    Depth Anything 3 replaces the semantic CLS token with a geometric camera
    token at layer `alt_start`. To retain the rich semantic classification
    features of the DINOv2 backbone, we tap the transformer block at `alt_start - 1`.

    Returns:
      da3_cls     → (D,)        semantic CLS token from block `alt_start - 1`
      da3_patches → (D, Ph, Pw) normed global patch tokens from block `alt_start - 1`
    """
    backbone  = model.model.backbone            # DinoV2 wrapper
    vit       = backbone.pretrained             # DinoVisionTransformer
    D         = vit.embed_dim
    alt_start = getattr(backbone, "alt_start", -1)

    # Target the block immediately before the CLS token is destroyed
    if alt_start > 0:
        target_blocks = [alt_start - 1]
    else:
        target_blocks = 1  # Fallback to the last block if no alt_start is defined

    # We call vit.get_intermediate_layers directly, bypassing the hardcoded
    # out_layers in backbone.forward().
    outputs, _aux = vit.get_intermediate_layers(tensor, n=target_blocks, export_feat_layers=[])

    # outputs: tuple of (patch_tokens, cls_token) — one entry per requested out_layer.
    patch_tokens_full, cls_token_full = outputs[-1]
    # patch_tokens_full: (B, S, L, D_full)
    # cls_token_full:    (B, S, D_full)

    if patch_tokens_full.shape[-1] == 2 * D:
        # cat_token=True: take the normed global-attention half.
        patch_tokens = patch_tokens_full[0, 0, :, D:].cpu().float()   # (L, D)
        
        # The CLS token returned by get_intermediate_layers is pre-norm. We apply 
        # vit.norm manually to the global half to match the patch tokens.
        cls_unnormed = cls_token_full[0, 0, D:]
        cls_vec      = vit.norm(cls_unnormed).cpu().float()            # (D,)
    else:
        # cat_token=False: single normed stream.
        patch_tokens = patch_tokens_full[0, 0].cpu().float()           # (L, D)
        cls_unnormed = cls_token_full[0, 0]
        cls_vec      = vit.norm(cls_unnormed).cpu().float()            # (D,)

    # Recover spatial grid from input dimensions.
    H = tensor.shape[-2]
    W = tensor.shape[-1]
    spatial = _reshape_patch_tokens_hw(patch_tokens, H, W, patch_size=14)  # (D, Ph, Pw)

    return {
        "da3_cls":     torch.nan_to_num(cls_vec,  nan=0.0).numpy(),
        "da3_patches": torch.nan_to_num(spatial,  nan=0.0).numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  I/O helpers
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

    # ── Validate encoder / model combinations ─────────────────────────────────
    if args.model == "dinov2" and args.encoder == "vitg":
        print(
            "[ERROR] --encoder vitg is not supported for --model dinov2 "
            "(torch.hub facebookresearch/dinov2 does not expose a vitg14 entry point). "
            "Use --model depth_anything_v2 for vitg."
        )
        sys.exit(1)

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

    elif args.model == "depth_anything_v2":
        model = build_depth_anything_v2(args.encoder, args.da_checkpoint, device)
        keys  = ["da_cls", "da_patches"]

        def extract_fn(img_path: str) -> Dict[str, np.ndarray]:
            tensor = preprocess_for_depth_anything_v2(
                img_path, model, args.da_input_size, device
            )
            return extract_depth_anything_v2_tokens(model, tensor)

    else:  # depth_anything_3
        model = build_depth_anything_3(args.da3_model_name, args.da3_from_hf, device)
        keys  = ["da3_cls", "da3_patches"]

        # InputProcessor is instantiated once and reused across images.
        from depth_anything_3.utils.io.input_processor import InputProcessor
        input_processor = InputProcessor()

        def extract_fn(img_path: str) -> Dict[str, np.ndarray]:
            tensor = preprocess_for_depth_anything_3(
                img_path, input_processor, args.da3_process_res, device
            )
            return extract_depth_anything_3_tokens(model, tensor)

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