#!/usr/bin/env python3
"""
extract_hf.py

Unified feature extractor for the three models:

    dino_v1  – facebook/dino-vitb8            
    dinov3   – facebook/dinov3-vitl16-pretrain-lvd1689m     
    sam3     – facebook/sam3                   

Each model is a subcommand. 

Usage
─────
python extract_hf.py dino_v1 \
    --input_dir  data/images \
    --output_file features/dino_v1_cls_avg.npz \
    --mode cls_avg

python extract_hf.py dinov3 \
    --input_dir  data/images \
    --output_file features/dinov3_cls_avg.npz \
    --model_id facebook/dinov3-vitl16-pretrain-lvd1689m \
    --mode cls_avg

python extract_hf.py sam3 \
    --input_dir  data/images \
    --output_file features/sam3_gap.npz \
    --mode gap
"""

import os
import argparse

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from huggingface_hub import login


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_file(path):
    """Safely reads external text files (used for HF token files)."""
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_features(output_file, features, filenames):
    final_features = np.vstack(features)
    final_filenames = np.array(filenames)
    print("-" * 40)
    print("Extraction complete.")
    print(f"Features shape: {final_features.shape}")
    print(f"Saving to {output_file}...")
    np.savez_compressed(output_file, features=final_features, filenames=final_filenames)
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# dino_v1 / dinov3
# ─────────────────────────────────────────────────────────────────────────────

class ViTImageDataset(Dataset):
    """
    Handles directory traversal, image loading, AND preprocessing.
    Moving preprocessing here (instead of in the main loop) prevents GPU
    starvation from single-threaded preprocessing.
    """
    def __init__(self, directory, processor, valid_exts=('.jpg', '.jpeg', '.png', '.bmp')):
        self.processor = processor
        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(valid_exts)
        ]
        self.files.sort()  # deterministic order

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            image = Image.open(path).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            return pixel_values, os.path.basename(path)
        except Exception as e:
            print(f"Error loading/processing {path}: {e}")
            return None, None


def vit_collate(batch):
    """Filters out failed image loads and stacks the tensors."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    pixel_values, filenames = zip(*batch)
    return torch.stack(pixel_values), list(filenames)


def run_vit_autoclass(args, default_num_registers):
    """Shared extraction loop for dino_v1 and dinov3 (both plain AutoModel ViTs)."""
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = torch.float32

    token = read_file(args.token)
    if token:
        login(token=token)

    print(f"Loading model: {args.model_id} in {dtype}...")
    processor = AutoImageProcessor.from_pretrained(args.model_id, token=token)
    model = AutoModel.from_pretrained(args.model_id, token=token, torch_dtype=dtype).to(device)
    model.eval()

    num_registers = getattr(model.config, "num_register_tokens", default_num_registers)
    print(f"Model configuration detected: {num_registers} register tokens.")

    dataset = ViTImageDataset(args.input_dir, processor=processor)
    if len(dataset) == 0:
        raise ValueError(f"No valid images found in {args.input_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=vit_collate,
        pin_memory=(device.type == "cuda"),
    )

    all_features, all_filenames = [], []
    print(f"Starting extraction for {len(dataset)} images in mode: [{args.mode}]...")

    with torch.inference_mode():
        for pixel_values, filenames in tqdm(dataloader):
            if pixel_values is None:
                continue

            pixel_values = pixel_values.to(device, dtype=dtype)
            outputs = model(pixel_values=pixel_values)
            last_hidden_state = outputs.last_hidden_state  # (B, S, D)

            cls_token = last_hidden_state[:, 0, :]
            if args.mode == "cls":
                batch_feats = cls_token
            else:  # cls_avg
                patch_start_idx = 1 + num_registers
                patch_tokens = last_hidden_state[:, patch_start_idx:, :]
                avg_patch_token = patch_tokens.mean(dim=1)
                batch_feats = torch.cat((cls_token, avg_patch_token), dim=1)

            all_features.append(batch_feats.cpu().to(torch.float32).numpy())
            all_filenames.extend(filenames)

    if not all_features:
        print("No features extracted. Exiting.")
        return

    save_features(args.output_file, all_features, all_filenames)


# ─────────────────────────────────────────────────────────────────────────────
# sam3 — Sam3Model vision-encoder path (different API, different pooling)
# ─────────────────────────────────────────────────────────────────────────────

class RawImageDataset(Dataset):
    """Returns raw PIL images; SAM3's processor is called per-image in the loop."""
    def __init__(self, directory, valid_exts=(".jpg", ".jpeg", ".png", ".bmp")):
        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(valid_exts)
        ]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            img = Image.open(path).convert("RGB")
            return img, os.path.basename(path)
        except Exception as e:
            print("Error:", path, e)
            return None, None


def raw_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    imgs, names = zip(*batch)
    return list(imgs), list(names)


def pool_sam3_embedding(emb, mode):
    # SAM3 vision encoder outputs flattened patches: (batch_size, seq_len, hidden_size)
    if mode == "gap":
        return emb.mean(dim=1).squeeze(0)
    elif mode == "flatten":
        return emb.flatten(start_dim=1).squeeze(0)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_sam3(args):
    from transformers import Sam3Processor, Sam3Model

    os.environ["HF_HOME"] = args.cache_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.token:
        token_str = read_file(args.token)
        if token_str:
            login(token=token_str)

    print("Loading SAM3 from Hugging Face...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    model.eval()

    dataset = RawImageDataset(args.input_dir)
    if len(dataset) == 0:
        raise ValueError("No images found in the specified directory.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=raw_collate,
    )

    all_features, all_filenames = [], []
    print("Starting extraction...")

    with torch.no_grad():
        for images, filenames in tqdm(dataloader):
            if images is None:
                continue

            batch_feats = []
            for img in images:
                inputs = processor(images=img, return_tensors="pt").to(device)
                vision_outputs = model.vision_encoder(inputs["pixel_values"])
                emb = vision_outputs.last_hidden_state
                pooled = pool_sam3_embedding(emb, args.mode)
                batch_feats.append(pooled.cpu().numpy())

            batch_feats = np.stack(batch_feats)
            all_features.append(batch_feats)
            all_filenames.extend(filenames)

    if not all_features:
        print("No features extracted. Exiting.")
        return

    save_features(args.output_file, all_features, all_filenames)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified extractor for HuggingFace-only models (dino_v1, dinov3, sam3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="model", required=True)

    # --- dino_v1 ---
    p_dino1 = sub.add_parser("dino_v1", help="facebook/dino-vitb8")
    p_dino1.add_argument("--input_dir", required=True, help="Folder containing images")
    p_dino1.add_argument("--output_file", required=True, help="Path to save .npz file")
    p_dino1.add_argument("--model_id", default="facebook/dino-vitb8", help="Hugging Face model ID")
    p_dino1.add_argument("--token", default=None, help="Path to HF token file (optional)")
    p_dino1.add_argument("--mode", choices=["cls", "cls_avg"], default="cls_avg")
    p_dino1.add_argument("--batch_size", type=int, default=32)
    p_dino1.add_argument("--num_workers", type=int, default=4)

    # --- dinov3 ---
    p_dino3 = sub.add_parser("dinov3", help="facebook/ddinov3-vitl16-pretrain-lvd1689m")
    p_dino3.add_argument("--input_dir", required=True, help="Folder containing images")
    p_dino3.add_argument("--output_file", required=True, help="Path to save .npz file")
    p_dino3.add_argument("--model_id", default="facebook/dinov3-vitl16-pretrain-lvd1689m", help="Hugging Face model ID")
    p_dino3.add_argument("--token", default=None, help="Path to HF token file (optional)")
    p_dino3.add_argument("--mode", choices=["cls", "cls_avg"], default="cls_avg")
    p_dino3.add_argument("--batch_size", type=int, default=32)
    p_dino3.add_argument("--num_workers", type=int, default=4)

    # --- sam3 ---
    p_sam3 = sub.add_parser("sam3", help="facebook/sam3 vision encoder")
    p_sam3.add_argument("--input_dir", required=True, help="Folder containing images")
    p_sam3.add_argument("--output_file", required=True, help="Path to save .npz file")
    p_sam3.add_argument("--token", default=None, help="Path to HF token file (optional)")
    p_sam3.add_argument("--cache_dir", default="./cache/", help="HF_HOME cache directory")
    p_sam3.add_argument("--mode", choices=["gap", "flatten"], default="gap")
    p_sam3.add_argument("--batch_size", type=int, default=8)
    p_sam3.add_argument("--num_workers", type=int, default=4)

    return parser


def main():
    args = build_parser().parse_args()

    if args.model == "dino_v1":
        run_vit_autoclass(args, default_num_registers=0)
    elif args.model == "dinov3":
        run_vit_autoclass(args, default_num_registers=4)
    elif args.model == "sam3":
        run_sam3(args)


if __name__ == "__main__":
    main()
