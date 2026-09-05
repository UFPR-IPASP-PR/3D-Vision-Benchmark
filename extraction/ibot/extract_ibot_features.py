#!/usr/bin/env python3
"""
extract_ibot_features.py
─────────────────────────
Feature extraction from an iBOT ViT encoder (https://github.com/bytedance/ibot).

Model and feature types
────────────────────────

  Loads a bare iBOT ViT backbone (vit_small / vit_base / vit_large) via the
  iBOT codebase's own `models` package, and loads weights with the codebase's
  own `utils.load_pretrained_weights()`. Taps the LAST transformer block via
  model.get_intermediate_layers(x, n=1).

  Unlike the DINOv2 hub checkpoint used in extract_da1_dinov2_features.py, iBOT's
  get_intermediate_layers() has no `return_class_token` option: it returns the
  full post-norm token sequence (CLS token at index 0, followed by the patch
  tokens) for the last block. This script splits that sequence itself.

    ibot_cls/      CLS token                         → (D,)        e.g. (768,) for vit_base
    ibot_patches/  Patch tokens reshaped to (D, P, P) → (D, P, P)  e.g. (768, 14, 14)

    The token dimension D and patch grid P depend on the chosen encoder size
    and on --crop_size / --patch_size:

      vits → D=384,  P = crop_size / patch_size (14×14 for the 224/16 default)
      vitb → D=768,  P = crop_size / patch_size
      vitl → D=1024, P = crop_size / patch_size

IMPORTANT — requires the iBOT codebase on the import path
───────────────────────────────────────────────────────────
This script does `import models` and `import utils`, which must resolve to
iBOT's own `models/` package and `utils.py` (the ones next to main_ibot.py in
the iBOT repo). It does NOT add anything to sys.path itself, so either:

  1. place this script in the root of your local iBOT clone (next to
     main_ibot.py, models/, utils.py), or
  2. add the repo root to PYTHONPATH before running, e.g.:
       PYTHONPATH=/path/to/ibot:$PYTHONPATH python extract_ibot_features.py ...

Usage example
─────────────
python extract_ibot_features.py \\
    --encoder            vitb \\
    --patch_size          16 \\
    --pretrained_weights  checkpoints/ibot_vitb16_checkpoint.pth \\
    --checkpoint_key       teacher \\
    --ann_json            data/annotations.json \\
    --img_dir             data/images \\
    --output_dir          features/ibot/
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms as pth_transforms

# ── iBOT codebase imports (must already be importable — see header note) ────
try:
    import models as ibot_models
    import utils as ibot_utils
except ImportError as e:
    sys.exit(
        f"[ERROR] Could not import iBOT's `models` / `utils` modules ({e}).\n"
        "This script must be run with the iBOT repo on the import path:\n"
        "  - place it inside the repo root (next to main_ibot.py), or\n"
        "  - export PYTHONPATH=/path/to/ibot:$PYTHONPATH before running."
    )

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --encoder short-name → iBOT arch name (models.__dict__ key)
ENCODER_TO_ARCH = {
    "vits": "vit_small",
    "vitb": "vit_base",
    "vitl": "vit_large",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Extract iBOT ViT encoder tokens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── what to extract ────────────────────────────────────────────────────
    p.add_argument(
        "--encoder", default="vitl", choices=list(ENCODER_TO_ARCH.keys()),
        help="ViT backbone size: vits→vit_small, vitb→vit_base, vitl→vit_large.",
    )
    p.add_argument(
        "--patch_size", default=16, type=int,
        help=(
            "Patch size of the ViT (must match --pretrained_weights). "
            "iBOT/DINO checkpoints are commonly patch16, some are patch8."
        ),
    )

    # ── weights ────────────────────────────────────────────────────────────
    p.add_argument(
        "--pretrained_weights", required=True,
        help=(
            "Path to a local iBOT .pth checkpoint. The special values "
            "'download' or 'supervised' are also accepted and forwarded to "
            "iBOT's own utils.load_pretrained_weights(), but note these "
            "fetch public DINO / DeiT weights (not iBOT weights), and only "
            "cover vit_small/vit_base at patch 8 or 16."
        ),
    )
    p.add_argument(
        "--checkpoint_key", default="teacher",
        help=(
            "Key to read inside the checkpoint dict (e.g. 'teacher', "
            "'student'), matching iBOT's own eval scripts' default."
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

    # ── preprocessing ──────────────────────────────────────────────────────
    p.add_argument(
        "--resize_size", default=256, type=int,
        help="Short-side resize before center-cropping (matches iBOT's eval scripts).",
    )
    p.add_argument(
        "--crop_size", default=224, type=int,
        help=(
            "Square center-crop size fed to the ViT. Should be a multiple "
            "of --patch_size for a clean square patch grid."
        ),
    )

    # ── execution ──────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Annotation loaders  (identical to extract_da1_dinov2_features.py)
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
# 3.  Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_ibot(
    encoder: str,
    patch_size: int,
    pretrained_weights: str,
    checkpoint_key: str,
    device: str,
) -> torch.nn.Module:
    """
    Build an iBOT ViT backbone and load weights via the iBOT codebase's own
    utils.load_pretrained_weights(), so checkpoint-prefix handling
    ('module.', 'backbone.') stays identical to the rest of the codebase.
    """
    arch = ENCODER_TO_ARCH[encoder]

    # Fail loudly on a bad path instead of silently falling back to random
    # weights, which is what iBOT's own load_pretrained_weights() does.
    if pretrained_weights not in ("download", "supervised") and not os.path.isfile(pretrained_weights):
        raise FileNotFoundError(
            f"--pretrained_weights '{pretrained_weights}' is not a file and is "
            "not 'download' or 'supervised'."
        )

    print(f"[INFO] Building iBOT backbone '{arch}' (patch_size={patch_size}) …")
    model = ibot_models.__dict__[arch](patch_size=patch_size, num_classes=0)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    ibot_utils.load_pretrained_weights(
        model, pretrained_weights, checkpoint_key, arch, patch_size
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def build_ibot_transform(resize_size: int, crop_size: int):
    """
    Matches iBOT's own evaluation scripts (eval_knn.py / eval_linear.py):
    Resize(short side) → CenterCrop(square) → ToTensor → ImageNet normalize.
    """
    return pth_transforms.Compose([
        pth_transforms.Resize(resize_size),
        pth_transforms.CenterCrop(crop_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocess_for_ibot(img_path: str, transform, device: str) -> torch.Tensor:
    """
    Load an image and prepare it for the iBOT ViT.

    Images are converted to RGB via PIL's convert("RGB") — matching iBOT's
    own data loader (loader.py / ImageFolder) — which DROPS any alpha
    channel rather than compositing onto a white background. This differs
    from extract_da1_dinov2_features.py's DINOv2 preprocessing, which does an
    RGBA→white-background composite.

    Returns: (1, 3, crop_size, crop_size) float32 tensor on `device`.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        tensor = transform(img).unsqueeze(0)
    return tensor.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Token extraction
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
def extract_ibot_tokens(
    model: torch.nn.Module,
    tensor: torch.Tensor,           # (1, 3, H, W)
) -> Dict[str, np.ndarray]:
    """
    Run an iBOT ViT and return:
      ibot_cls     – CLS token of the last layer            → (D,)
      ibot_patches – patch tokens of the last layer,
                     reshaped to a spatial grid (no pooling) → (D, P, P)

    iBOT's get_intermediate_layers(x, n=1) returns a 1-element list holding
    the post-norm output of the LAST block, shaped (1, 1+L, D): the CLS
    token at position 0 followed by L patch tokens. There is no
    return_class_token option (unlike the DINOv2 hub checkpoint), so the
    split is done manually here.
    """
    tokens = model.get_intermediate_layers(tensor, n=1)[0]   # (1, 1+L, D)
    tokens = tokens[0].cpu().float()                          # (1+L, D)
    cls_token, patch_tokens = tokens[0], tokens[1:]            # (D,), (L, D)

    return {
        "ibot_cls":     torch.nan_to_num(cls_token, nan=0.0).numpy(),
        "ibot_patches": torch.nan_to_num(_reshape_patch_tokens(patch_tokens), nan=0.0).numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  I/O helpers  (identical to extract_da1_dinov2_features.py)
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

    # ── Model + transform + extraction fn ──────────────────────────────────
    model = build_ibot(
        args.encoder, args.patch_size, args.pretrained_weights,
        args.checkpoint_key, device,
    )
    transform = build_ibot_transform(args.resize_size, args.crop_size)
    keys = ["ibot_cls", "ibot_patches"]

    def extract_fn(img_path: str) -> Dict[str, np.ndarray]:
        tensor = preprocess_for_ibot(img_path, transform, device)
        return extract_ibot_tokens(model, tensor)

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

