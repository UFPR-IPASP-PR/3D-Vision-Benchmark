#!/usr/bin/env python3
"""
Short OOD Experiment Script
───────────────────────────
Loads all images from a folder, extracts features dynamically (using DINOv3 
or DepthAnythingV2), and runs them through a pretrained classifier head 
to report how many are classified as 'CAR' vs 'NOT CAR'.

Usage Example (DINOv3 with Token):
    python run_ood_experiment.py \
        --img_dir ./ood_images/ \
        --checkpoint runs/vehicle_cls/best_val.pth \
        --extractor dinov3 \
        --dinov3_model_id facebook/dinov3-vitl14-pretrain-lvd1689m \
        --token "hf_YOUR_TOKEN_HERE"

Usage Example (DepthAnything V2):
    python run_ood_experiment.py \
        --img_dir ./ood_images/ \
        --checkpoint runs/vehicle_cls/best_val.pth \
        --extractor da_v2 \
        --da_encoder vitb \
        --da_checkpoint checkpoints/depth_anything_v2_vitb.pth
"""

import os
import argparse
from collections import Counter

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login

# HF for DINOv3
from transformers import AutoImageProcessor, AutoModel

# ---------------------------------------------
# 1. Classifier Model Definition
# ---------------------------------------------
def build_head(embed_dim: int, hidden_dim: int, num_classes: int,
               num_layers: int, dropout: float) -> nn.Sequential:
    if hidden_dim == 0 or num_layers == 0:
        return nn.Sequential(nn.Linear(embed_dim, num_classes))

    num_layers = max(1, num_layers)
    layers, in_dim = [], embed_dim
    for _ in range(num_layers):
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)

class FeatureClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: dict,
                 head_layers: int = 3, head_hidden: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        self.heads = nn.ModuleDict({
            f"attr_{attr}": build_head(embed_dim, head_hidden, n, head_layers, dropout)
            for attr, n in num_classes.items()
        })

    def forward(self, x):
        return {attr[5:]: head(x) for attr, head in self.heads.items()}


# ---------------------------------------------
# 2. Extractor Configurations & Helpers
# ---------------------------------------------
_DAV2_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,   96,   192,  384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,   192,  384,  768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256,  512,  1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

@torch.inference_mode()
def extract_dinov3_feature(model, processor, img_path, device, target_dim):
    """Extracts DINOv3 features and shapes them to match the linear probe."""
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    
    num_registers = getattr(model.config, "num_register_tokens", 4)
    cls_token = last_hidden_state[:, 0, :]
    
    if target_dim == cls_token.shape[-1]:
        return cls_token
    elif target_dim == cls_token.shape[-1] * 2:
        # cls_avg mode
        patch_start_idx = 1 + num_registers
        patch_tokens = last_hidden_state[:, patch_start_idx:, :]
        avg_patch_token = patch_tokens.mean(dim=1)
        return torch.cat((cls_token, avg_patch_token), dim=1)
    else:
        raise ValueError(f"DINOv3 Dimension mismatch: model expected {target_dim}, but CLS is {cls_token.shape[-1]}")

@torch.inference_mode()
def extract_dav2_feature(model, img_path, input_size, device, target_dim):
    """Extracts DepthAnythingV2 features and shapes them to match the linear probe."""
    raw = cv2.imread(img_path)
    if raw is None:
        raise FileNotFoundError(f"cv2.imread failed: {img_path}")
    
    tensor, _ = model.image2tensor(raw, input_size)
    tensor = tensor.to(device)
    
    features = model.pretrained.get_intermediate_layers(tensor, n=1, return_class_token=True)
    patch_tokens, cls_token = features[0]
    
    cls_token = cls_token.float()
    patch_tokens = patch_tokens.float()
    
    if target_dim == cls_token.shape[-1]:
        return cls_token
    elif target_dim == cls_token.shape[-1] * 2:
        # fuse_cls_gap mode
        avg_patch = patch_tokens.mean(dim=1)
        return torch.cat((cls_token, avg_patch), dim=1)
    else:
        raise ValueError(f"DA_v2 Dimension mismatch: model expected {target_dim}, but CLS is {cls_token.shape[-1]}")


# ---------------------------------------------
# 3. Main Execution Script
# ---------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Run OOD Experiment with existing classification heads.")
    
    # Run arguments
    p.add_argument("--img_dir", required=True, help="Directory containing OOD images")
    p.add_argument("--checkpoint", required=True, help="Path to best_val.pth classification head")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Extractor arguments
    p.add_argument("--extractor", required=True, choices=["dinov3", "da_v2"], help="Which feature backbone to use")
    
    # DINOv3 specifics
    p.add_argument("--dinov3_model_id", default="facebook/dinov3-vit7b16-pretrain-lvd1689m")
    p.add_argument("--token", type=str, default=None, help="Hugging Face authentication token string")
    
    # Depth Anything V2 specifics
    p.add_argument("--da_encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    p.add_argument("--da_checkpoint", default=None, help="Path to .pth DA V2 checkpoint")
    p.add_argument("--da_input_size", default=518, type=int)

    return p.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)

    # 1. Load Linear Probe / Multi-Head Classifier
    print(f"[INFO] Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    # Extract vocabularies mapping (e.g. {'type': ['CAR', 'TRUCK', 'SUV'], ...})
    # the training scripts save vocabs as: "vocabs": {attr: [class1, class2, ...]}
    vocabs_list = ckpt["vocabs"]
    num_classes = {k: len(v) for k, v in vocabs_list.items()}
    
    # Reconstruct arguments used during training
    ckpt_args = ckpt.get("args", {})
    head_layers = ckpt_args.get("head_layers", 3)
    head_hidden = ckpt_args.get("head_hidden", 512)
    dropout     = ckpt_args.get("dropout", 0.3)
    
    # Determine input embedding dimension from the checkpoint weights
    first_weight = next(iter(ckpt["model"].values()))
    embed_dim = first_weight.shape[1]
    print(f"[INFO] Detected Classifier Embed Dim: {embed_dim} | Head Layers: {head_layers}")

    classifier = FeatureClassifier(embed_dim, num_classes, head_layers, head_hidden, dropout)
    classifier.load_state_dict(ckpt["model"])
    classifier = classifier.to(device).eval()
    
    # Ascertain target attribute
    target_attr = "type"
    if target_attr not in vocabs_list:
        target_attr = list(vocabs_list.keys())[0]
        print(f"[WARN] 'type' attribute missing. Defaulting to first available attribute: '{target_attr}'")

    # 2. Build Extractor
    print(f"[INFO] Initializing {args.extractor} extractor...")
    if args.extractor == "dinov3":
        os.environ['HF_HOME'] = './cache/'
        
        # Login if token is provided
        if args.token:
            print("[INFO] Logging into Hugging Face Hub using provided token...")
            login(token=args.token)
            
        processor = AutoImageProcessor.from_pretrained(args.dinov3_model_id, token=args.token)
        model = AutoModel.from_pretrained(args.dinov3_model_id, token=args.token).to(device).eval()
        
    elif args.extractor == "da_v2":
        if not args.da_checkpoint:
            raise ValueError("--da_checkpoint is required for --extractor da_v2")
        from depth_anything_v2.dpt import DepthAnythingV2
        
        cfg = _DAV2_MODEL_CONFIGS[args.da_encoder]
        model = DepthAnythingV2(**cfg)
        model.load_state_dict(torch.load(args.da_checkpoint, map_location="cpu"))
        model = model.to(device).eval()

    # 3. Processing Loop
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(valid_exts)]
    
    print(f"[INFO] Starting OOD Inference on {len(image_files)} images...")
    
    results = {"CAR": 0, "NOT_CAR": 0}
    detailed_counts = Counter()

    for fname in tqdm(image_files, desc="Classifying OOD Images"):
        img_path = os.path.join(args.img_dir, fname)
        
        try:
            # A) Extract Feature
            if args.extractor == "dinov3":
                feat = extract_dinov3_feature(model, processor, img_path, device, embed_dim)
            else:
                feat = extract_dav2_feature(model, img_path, args.da_input_size, device, embed_dim)
            
            # B) Classify 
            with torch.no_grad():
                logits = classifier(feat)
                pred_idx = logits[target_attr].argmax(dim=1).item()
                pred_label = vocabs_list[target_attr][pred_idx].upper()
            
            # C) Log Results
            detailed_counts[pred_label] += 1
            if pred_label == "CAR":
                results["CAR"] += 1
            else:
                results["NOT_CAR"] += 1
                
        except Exception as e:
            print(f"\n[ERROR] Failed processing {fname}: {e}")
            
    # 4. Reporting
    print("\n" + "="*50)
    print(" 📊 OOD EXPERIMENT RESULTS ")
    print("="*50)
    print(f"Total Images processed:      {sum(detailed_counts.values())}")
    print(f"Total classified as CAR:     {results['CAR']}")
    print(f"Total classified as NOT CAR: {results['NOT_CAR']}")
    print("-" * 50)
    print("Detailed Class Distribution:")
    for label, count in detailed_counts.most_common():
        print(f"  {label:<15}: {count}")
    print("="*50)

if __name__ == "__main__":
    main()