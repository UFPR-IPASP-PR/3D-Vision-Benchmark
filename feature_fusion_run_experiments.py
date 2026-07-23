#!/usr/bin/env python3
"""
Vehicle attribute classifier trained on fused multi-model features.

This script reads from the directory structures produced by the extraction
scripts. It dynamically loads, pools (for 2D spatial/token features), and
concatenates requested feature types into a single 1D vector per image.

Usage:
    # Plain feature fusion (original behaviour):
    python train_feature_classifier.py \
        --ann_json       data/annotations.json \
        --features_dir   features/ \
        --feature_types  dino_cls,da_cls,mlp_feats,geometry_tokens,shape_latent \
        --pool_strategy  mean \
        --workspace      runs/vehicle_cls \
        --epochs 100 --batch_size 256 \
        --format jbcs \
        --head_layers 3 \
        --head_hidden 512 \
        --dropout 0.3

    # CLS + global-average-pooled-patches fusion
    # (works for dino_patches / da_patches / da3_patches):
    python train_feature_classifier.py \
        --ann_json       data/annotations.json \
        --features_dir   features/ \
        --feature_types  dino_patches,da_patches,da3_patches \
        --fuse_cls_gap \
        --workspace      runs/vehicle_cls \
        --epochs 100 --batch_size 256
    # Each *_patches entry is automatically paired with its sibling CLS/cam
    # file (dino_cls, da_cls, da3_cam).  The resulting vector per model is
    # [cls (D,) | gap_patches (D,)] = (2D,), then all models are concatenated.
"""

import os
import json
import random
import argparse
import csv
from collections import defaultdict, Counter
from datetime import datetime

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ---------------------------------------------
# 1.  Argument parsing
# ---------------------------------------------

def get_args():
    p = argparse.ArgumentParser()

    # --- data ---
    p.add_argument("--dataset",   default="data",
                   help="Name of the dataset (used for folder naming)")
    p.add_argument("--ann_json",  required=True,
                   help="Path to the annotation JSON file")
    p.add_argument("--format",    default="jbcs", choices=["jbcs", "lplc"],
                   help="Annotation format: 'jbcs' (flat list) or 'lplc' (nested dict)")
    
    # --- features ---
    p.add_argument("--features_dir",  required=True,
                   help="Root directory containing the feature subfolders "
                        "(e.g. dino_cls/, mlp_feats/, etc.)")
    p.add_argument("--feature_types", 
                   default="dino_cls,da_cls,mlp_feats,geometry_tokens,shape_latent",
                   help="Comma-separated list of feature subfolders to fuse.")
    p.add_argument("--pool_strategy", default="mean", choices=["mean", "max", "flatten"],
                   help="How to handle >1D features (e.g. (16384, 128) mlp_feats). "
                        "'mean' applies global average pooling across the token dimension. "
                        "Has no effect on patch-grid features when --fuse_cls_gap is set; "
                        "those are always spatially average-pooled.")
    p.add_argument("--fuse_cls_gap", action="store_true",
                   help="When set, any patch-grid subfolder listed in --feature_types "
                        "(dino_patches, da_patches, da3_patches) is automatically paired "
                        "with its sibling CLS/cam subfolder (dino_cls, da_cls, da3_cam). "
                        "Each pair produces a [CLS (D,) | GAP(patches) (D,)] = (2D,) vector. "
                        "Do NOT also list the sibling CLS/cam key in --feature_types; "
                        "it will be loaded automatically.")

    # --- experiment management ---
    p.add_argument("--workspace",    default="runs/vehicle_cls",
                   help="Root folder for all experiments")
    p.add_argument("--exp_name",     default=None,
                   help="Optional suffix for the experiment folder")
    p.add_argument("--continue_exp", default=None,
                   help="Name of an existing experiment folder inside "
                        "--workspace to resume from")

    # --- model ---
    p.add_argument("--head_layers",  type=int,   default=3,
                   help="Number of hidden Linear->GELU->Dropout blocks in each "
                        "classifier head (>=1)")
    p.add_argument("--head_hidden",  type=int,   default=512,
                   help="Width of each hidden layer in the classifier heads")
    p.add_argument("--dropout",      type=float, default=0.3,
                   help="Dropout probability applied after each hidden layer")

    # --- training ---
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_split",    type=float, default=0.10)
    p.add_argument("--test_split",   type=float, default=0.10)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--attribute",    type=str,   default="all",
                   help="Comma-separated attributes to train, e.g. 'color,type' or 'all'")
    p.add_argument("--splits",       default=None,
                   help="Path to directory with train.txt / val.txt / test.txt split files")

    # --- training filters ---
    p.add_argument("--train_filter_rear_view", default="all", choices=["all", "yes", "no"],
                   help="Filter training set to only include images where rear_view matches the choice.")
    p.add_argument("--train_filter_infrared", default="all", choices=["all", "yes", "no"],
                   help="Filter training set to only include images where infrared matches the choice.")

    p.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of the train split to keep for training. "
             "For example, 0.1 keeps 10%% of train samples and discards the rest. "
             "Validation and test splits are unchanged."
    )

    p.add_argument(
        "--filename_map",
        default=None,
        help="CSV mapping original filenames to anonymized filenames. "
             "Expected columns: original,new"
    )

    # --- early stopping & loss ---
    p.add_argument("--early_stopping_patience", type=int, default=10,
                   help="Stop training if val loss hasn't improved for this many epochs. (0 to disable)")
    p.add_argument(
        "--loss", default="weighted_ce",
        choices=["weighted_ce", "unweighted_ce", "scaled_weighted_ce", "focal", "focal_proportion"],
    )
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=float, default=0.25)

    return p.parse_args()


def load_filename_map(csv_path):
    """
    Reads a CSV with format:

    original,new
    long_filename.jpg,img_00001.jpg

    Returns:
        dict {original_name: anonymized_name}
    """
    mapping = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        if "original" not in reader.fieldnames or "new" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must contain columns 'original' and 'new'"
            )

        for row in reader:
            original = row["original"].strip()
            new = row["new"].strip()

            if original:
                mapping[original] = new

    print(f"[INFO] Loaded {len(mapping):,} filename mappings.")
    return mapping


# ---------------------------------------------
# 2.  Experiment folder management
# ---------------------------------------------

def make_exp_dir(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.exp_name:
        suffix = args.exp_name
    else:
        attr_str    = args.attribute.replace(",", "-")
        splits_name = os.path.basename(os.path.normpath(args.splits)) if args.splits else "auto"
        # Hash or truncate feature types to avoid massive folder names
        ftypes      = args.feature_types.split(",")
        feat_str    = f"{len(ftypes)}fused_{args.pool_strategy}"
        if args.fuse_cls_gap:
            feat_str += "_clsgap"
        
        suffix = (
            f"{args.dataset}_{attr_str}_lr{args.lr}"
            f"_wd{args.weight_decay}"
            f"_split-{splits_name}_feat-{feat_str}"
            f"_head-{args.head_layers}x{args.head_hidden}"
            f"_drop{args.dropout}"
            f"_loss-{args.loss}"
        )

        if args.train_filter_rear_view != "all":
            suffix += f"_rv-{args.train_filter_rear_view}"
        if args.train_filter_infrared != "all":
            suffix += f"_ir-{args.train_filter_infrared}"

    name = f"{timestamp}_{suffix}"
    path = os.path.join(args.workspace, name)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_exp_dir(workspace: str, continue_exp: str) -> str:
    path = os.path.join(workspace, continue_exp)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Cannot find experiment folder to continue: {path}")
    return path


# ---------------------------------------------
# 3.  Parse annotations
# ---------------------------------------------

def load_samples_jbcs(ann_json: str):
    with open(ann_json) as f:
        data = json.load(f)

    def clean(v): return (v or "UNKNOWN").strip().upper()

    samples = []
    for entry in data:
        fname = entry.get("filename", "")
        if not fname: continue
        samples.append({
            "filename": os.path.basename(fname),
            "type":      clean(entry.get("type",  "")),
            "make":      clean(entry.get("make",  "")),
            "model":     clean(entry.get("model", "")),
            "rear_view": clean(entry.get("rear_view", "")),
            "infrared":  clean(entry.get("infrared",  "")),
        })
    return samples

def load_samples_lplc(ann_json: str):
    with open(ann_json) as f:
        data = json.load(f)

    samples = []
    for fname, entry in data.items():
        for ann in entry.get("anns", []):
            car = ann.get("car", {})
            if not ann.get("car_valid", False): continue
            samples.append({
                "filename": os.path.basename(fname),
                "type":      (car.get("type",  "") or "UNKNOWN").strip().upper(),
                "make":      (car.get("make",  "") or "UNKNOWN").strip().upper(),
                "model":     (car.get("model", "") or "UNKNOWN").strip().upper(),
                "color":     (car.get("color", "") or "UNKNOWN").strip().upper(),
                "rear_view": (entry.get("rear_view", "") or "UNKNOWN").strip().upper(),
                "infrared":  (entry.get("infrared",  "") or "UNKNOWN").strip().upper(),
            })
    return samples

def load_split_file(path: str) -> set:
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

def build_vocabs(samples, target_attrs):
    vocabs = {}
    for attr in target_attrs:
        unique = sorted({s[attr] for s in samples})
        vocabs[attr] = {v: i for i, v in enumerate(unique)}
    return vocabs

def encode_sample(sample, vocabs):
    return {attr: vocabs[attr].get(sample[attr], 0) for attr in vocabs}


# ---------------------------------------------
# 4.  Multi-feature loading & fusion
# ---------------------------------------------

# Mapping from a patches subfolder key to its paired CLS/cam key.
# Derived from the output layout of extract_encoder_features.py (v1 and v2).

_PATCHES_TO_CLS = {
    "dino_patches": "dino_cls",
    "da_patches":   "da_cls",
    "da3_patches":  "da3_cam",
    "ibot_patches": "ibot_cls",   # ← add this
}


def _pool_generic(arr: np.ndarray, pool_strategy: str) -> np.ndarray:
    """Pool a (L, D) token-sequence array to (D,) using the requested strategy.

    This is used for non-patch-grid features such as mlp_feats (16384, 128)
    or geometry_tokens.  Axis 0 is the token/sequence dimension.
    """
    if pool_strategy == "mean":
        return arr.mean(axis=0)
    elif pool_strategy == "max":
        return arr.max(axis=0)
    elif pool_strategy == "flatten":
        return arr.flatten()
    raise ValueError(f"Unknown pool_strategy: {pool_strategy!r}")


def _gap_patches(arr: np.ndarray) -> np.ndarray:
    """Global average pool a (D, Ph, Pw) patch-grid to (D,).

    The extractors save patch tokens as (D, P, P) (or (D, Ph, Pw) for
    depth_anything_3 with non-square inputs). If the input is non-square
    and reshape fails, it is kept as (D, L). We want to average over the
    spatial/sequence axes, keeping the channel dimension D intact.
    """
    if arr.ndim == 2:
        return arr.mean(axis=1)    # (D, L) fallback -> (D,)
        
    assert arr.ndim == 3, (
        f"Expected a 3-D patch array (D, Ph, Pw) or 2-D (D, L) but got shape {arr.shape}"
    )
    return arr.mean(axis=(1, 2))   # (D, Ph, Pw) -> (D,)


def load_multi_features(samples, features_dir, feature_types, pool_strategy,
                        fuse_cls_gap: bool = False, filename_map=None):
    """
    Loads .npy files for each sample from the respective feature type folders.
    Applies pooling to >1D tensors, concatenates them into a single 1D vector,
    and returns a RAM cache dictionary {filename: concatenated_vector}.

    When *fuse_cls_gap* is True, any patch-grid key listed in *feature_types*
    (dino_patches / da_patches / da3_patches) is paired with its sibling
    CLS/cam key (dino_cls / da_cls / da3_cam).  The two are loaded together
    and concatenated as [CLS (D,) | GAP(patches) (D,)] = (2D,) before being
    included in the final fused vector.  The sibling key must NOT also appear
    in *feature_types* (that would double-count it; the function raises an
    error if it does).

    For all other feature types the existing pooling behaviour is unchanged:
    1-D arrays are used as-is; ≥2-D arrays (token sequences with shape
    (L, D)) are collapsed via *pool_strategy* along axis 0.

    When *filename_map* is provided (a dict {original: anonymized}), it is
    used to derive the .npy stem from the annotation filename.  The map is
    inverted so that anonymized annotation filenames are mapped back to the
    original filenames that name the .npy files on disk.
    """
    feat_index = {}
    missing_counts = defaultdict(int)
    ftypes = [f.strip() for f in feature_types.split(",")]

    # ------------------------------------------------------------------
    # Validate --fuse_cls_gap usage
    # ------------------------------------------------------------------
    if fuse_cls_gap:
        auto_cls_keys = {
            patches_key: cls_key
            for patches_key, cls_key in _PATCHES_TO_CLS.items()
            if patches_key in ftypes
        }
        # Warn and abort if the user also listed the sibling CLS key explicitly
        for patches_key, cls_key in auto_cls_keys.items():
            if cls_key in ftypes:
                raise ValueError(
                    f"--fuse_cls_gap is set: '{cls_key}' will be loaded "
                    f"automatically as the CLS partner of '{patches_key}'. "
                    f"Remove '{cls_key}' from --feature_types to avoid "
                    f"double-counting it."
                )
        # Build the full list of keys that will actually be loaded/returned,
        # inserting each auto-CLS key immediately before its patches partner.
        loaded_types = []
        for ft in ftypes:
            if ft in auto_cls_keys:
                loaded_types.append(auto_cls_keys[ft])  # e.g. dino_cls
            loaded_types.append(ft)                      # e.g. dino_patches
    else:
        auto_cls_keys = {}
        loaded_types = ftypes

    print(f"[INFO] Loading features: {ftypes}")
    if fuse_cls_gap and auto_cls_keys:
        print(f"[INFO] fuse_cls_gap: auto-loading CLS partners {list(auto_cls_keys.values())}")
        print(f"[INFO] Effective load order: {loaded_types}")
    print(f"[INFO] Pooling strategy for non-patch multi-dimensional features: {pool_strategy}")

    # Build an inverted map {anonymized_name: original_name} so we can resolve
    # the .npy stem from the (possibly anonymized) annotation filename.
    inverted_map = None
    if filename_map is not None:
        inverted_map = {v: k for k, v in filename_map.items()}
        print(f"[INFO] filename_map provided: {len(inverted_map):,} reverse mappings (anonymized → original).")

    # ------------------------------------------------------------------
    # Per-sample loading
    # ------------------------------------------------------------------
    mapped_count = 0
    for s in tqdm(samples, desc="Loading features to RAM"):
        fname = s["filename"]

        # Resolve the stem used for the on-disk .npy files.
        # If a filename_map was provided the annotation filename may be
        # anonymized; use the inverted map to get the original stem.
        if inverted_map is not None and fname in inverted_map:
            npy_stem = os.path.splitext(inverted_map[fname])[0]
            mapped_count += 1
        else:
            npy_stem = os.path.splitext(fname)[0]

        feats_to_concat = []
        missing = False

        for ft in ftypes:
            # ---- fuse_cls_gap: load CLS partner first, then GAP patches ----
            if fuse_cls_gap and ft in auto_cls_keys:
                cls_key = auto_cls_keys[ft]

                cls_path     = os.path.join(features_dir, cls_key, f"{npy_stem}.npy")
                patches_path = os.path.join(features_dir, ft,      f"{npy_stem}.npy")

                for path, key in [(cls_path, cls_key), (patches_path, ft)]:
                    if not os.path.exists(path):
                        missing_counts[key] += 1
                        missing = True
                        break
                if missing:
                    break

                cls_arr     = np.load(cls_path).astype(np.float32)     # (D,)
                patches_arr = np.load(patches_path).astype(np.float32)  # (D, Ph, Pw)

                gap_arr = _gap_patches(patches_arr)                      # (D,)
                feats_to_concat.append(cls_arr)   # CLS first …
                feats_to_concat.append(gap_arr)   # … then GAP

            # ---- standard path (no fuse_cls_gap, or non-patches key) -------
            # ---- standard path (no fuse_cls_gap, or non-patches key) -------
            else:
                fpath = os.path.join(features_dir, ft, f"{npy_stem}.npy")
                if not os.path.exists(fpath):
                    missing_counts[ft] += 1
                    missing = True
                    break

                arr = np.load(fpath)

                # Collapse spatial / token dimension
                # Safety check: route (D, L) arrays to `_gap_patches` if they are known patches.
                if arr.ndim == 3 or (arr.ndim == 2 and (ft in _PATCHES_TO_CLS or ft.endswith("_patches"))):
                    # (D, Ph, Pw) patch grid or (D, L) patch sequence without fuse_cls_gap:
                    # pool over spatial/sequence axes, same as _gap_patches
                    arr = _gap_patches(arr)
                elif arr.ndim > 1:
                    # (L, D) token sequence — pool over token axis
                    arr = _pool_generic(arr, pool_strategy)

                feats_to_concat.append(arr.astype(np.float32))

        if not missing:
            feat_index[fname] = np.concatenate(feats_to_concat, axis=0)

    if inverted_map is not None:
        print(
            f"[INFO] Applied filename mapping to "
            f"{mapped_count:,} / {len(samples):,} samples."
        )

    if missing_counts:
        print("\n[WARN] Missing features (samples dropped):")
        for ft, c in missing_counts.items():
            print(f"         {ft:20s}: {c} missing files")

    print(f"[INFO] Loaded complete feature fusion for {len(feat_index):,} images.")
    return feat_index, loaded_types


# ---------------------------------------------
# 5.  Dataset
# ---------------------------------------------

class FeatureDataset(Dataset):
    def __init__(self, samples, feat_index, vocabs):
        self.samples    = [s for s in samples if s["filename"] in feat_index]
        self.feat_index = feat_index
        self.vocabs     = vocabs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        feat   = torch.from_numpy(self.feat_index[s["filename"]].copy())
        labels = encode_sample(s, self.vocabs)
        return feat, {k: torch.tensor(v, dtype=torch.long) for k, v in labels.items()}


# ---------------------------------------------
# 6.  Model: multi-head MLP classifier
# ---------------------------------------------

def build_head(embed_dim: int, hidden_dim: int, num_classes: int,
               num_layers: int, dropout: float) -> nn.Sequential:
    
    # If hidden_dim or num_layers is 0, behave as a Linear Probe
    if hidden_dim == 0 or num_layers == 0:
        return nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    # Otherwise, build the MLP
    num_layers = max(1, num_layers)
    layers, in_dim = [], embed_dim
    for _ in range(num_layers):
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        in_dim  = hidden_dim
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)

def old_build_head(embed_dim: int, hidden_dim: int, num_classes: int,
               num_layers: int, dropout: float) -> nn.Sequential:
    num_layers = max(1, num_layers)
    layers, in_dim = [], embed_dim
    for _ in range(num_layers):
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        in_dim  = hidden_dim
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)


class FeatureClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: dict,
                 head_layers: int = 3, head_hidden: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        self.heads = nn.ModuleDict({
            f"attr_{attr}": build_head(
                embed_dim, head_hidden, n, head_layers, dropout
            )
            for attr, n in num_classes.items()
        })

    def forward(self, x):
        return {attr[5:]: head(x) for attr, head in self.heads.items()}


# ---------------------------------------------
# 7.  Early stopping
# ---------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience  = patience
        self.best_loss = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        if self.patience == 0: return False
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------
# 8.  Training & Loss Helpers
# ---------------------------------------------

def focal_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
    log_p   = F.log_softmax(logits, dim=1)
    p       = log_p.exp()
    log_pt  = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt      = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_w = (1.0 - pt) ** gamma
    loss    = -focal_w * log_pt

    if alpha is not None:
        if isinstance(alpha, (int, float)):
            alpha_t = float(alpha)
        else:
            alpha_t = alpha.to(logits.device)[targets]
        loss = alpha_t * loss

    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss


def compute_loss(logits, labels, loss_cfg, attr):
    mode = loss_cfg["mode"]
    if mode == "unweighted_ce":
        return F.cross_entropy(logits, labels)
    if mode == "weighted_ce":
        return F.cross_entropy(logits, labels, weight=loss_cfg["class_weights"][attr])
    if mode == "scaled_weighted_ce":
        return loss_cfg["scale_factors"][attr] * F.cross_entropy(logits, labels, weight=loss_cfg["class_weights"][attr])
    if mode == "focal":
        return focal_loss(logits, labels, gamma=loss_cfg["focal_gamma"], alpha=loss_cfg["focal_alpha"])
    if mode == "focal_proportion":
        return focal_loss(logits, labels, gamma=loss_cfg["focal_gamma"], alpha=loss_cfg["proportion_alpha"][attr])
    raise ValueError(f"Unknown loss mode: {mode!r}")


def run_epoch(model, loader, optimizer, device, loss_cfg=None, train=True):
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    totals = defaultdict(float)
    n = 0
    acc_macro, acc_micro, f1_macro, f1_micro = {}, {}, {}, {}

    with ctx:
        for feats, labels in tqdm(loader, desc="Train" if train else "Eval", leave=False):
            feats  = feats.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            logits = model(feats)

            if not acc_macro:
                for attr in logits:
                    nc = logits[attr].shape[1]
                    acc_macro[attr] = MulticlassAccuracy(nc, average="macro").to(device)
                    acc_micro[attr] = MulticlassAccuracy(nc, average="micro").to(device)
                    f1_macro[attr]  = MulticlassF1Score(nc, average="macro").to(device)
                    f1_micro[attr]  = MulticlassF1Score(nc, average="micro").to(device)

            if loss_cfg is None:
                loss = sum(F.cross_entropy(logits[a], labels[a]) for a in logits)
            else:
                loss = sum(compute_loss(logits[a], labels[a], loss_cfg, a) for a in logits)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = feats.size(0)
            totals["loss"] += loss.item() * bs
            for attr in logits:
                preds = torch.argmax(logits[attr], dim=1)
                acc_macro[attr].update(preds, labels[attr])
                acc_micro[attr].update(preds, labels[attr])
                f1_macro[attr].update(preds, labels[attr])
                f1_micro[attr].update(preds, labels[attr])
            n += bs

    results = {"loss": totals["loss"] / max(n, 1)}
    for attr in acc_macro:
        results[f"acc_macro_{attr}"] = acc_macro[attr].compute().item()
        results[f"acc_micro_{attr}"] = acc_micro[attr].compute().item()
        results[f"f1_macro_{attr}"]  = f1_macro[attr].compute().item()
        results[f"f1_micro_{attr}"]  = f1_micro[attr].compute().item()
    return results


def run_test(model, loader, device, vocabs):
    model.eval()
    totals = defaultdict(float)
    n = 0
    attrs = list(vocabs.keys())
    all_preds, all_labels = {a: [] for a in attrs}, {a: [] for a in attrs}
    acc_macro, acc_micro, f1_macro, f1_micro = {}, {}, {}, {}

    with torch.no_grad():
        for feats, labels in tqdm(loader, desc="Test", leave=False):
            feats  = feats.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            logits = model(feats)

            if not acc_macro:
                for attr in attrs:
                    nc = logits[attr].shape[1]
                    acc_macro[attr] = MulticlassAccuracy(nc, average="macro").to(device)
                    acc_micro[attr] = MulticlassAccuracy(nc, average="micro").to(device)
                    f1_macro[attr]  = MulticlassF1Score(nc, average="macro").to(device)
                    f1_micro[attr]  = MulticlassF1Score(nc, average="micro").to(device)

            loss = sum(F.cross_entropy(logits[a], labels[a]) for a in logits)

            bs = feats.size(0)
            totals["loss"] += loss.item() * bs
            for attr in attrs:
                preds = torch.argmax(logits[attr], dim=1)
                acc_macro[attr].update(preds, labels[attr])
                acc_micro[attr].update(preds, labels[attr])
                f1_macro[attr].update(preds, labels[attr])
                f1_micro[attr].update(preds, labels[attr])
                all_preds[attr].extend(preds.cpu().numpy().tolist())
                all_labels[attr].extend(labels[attr].cpu().numpy().tolist())
            n += bs

    metrics = {"loss": totals["loss"] / max(n, 1)}
    for attr in attrs:
        metrics[f"acc_macro_{attr}"] = acc_macro[attr].compute().item()
        metrics[f"acc_micro_{attr}"] = acc_micro[attr].compute().item()
        metrics[f"f1_macro_{attr}"]  = f1_macro[attr].compute().item()
        metrics[f"f1_micro_{attr}"]  = f1_micro[attr].compute().item()

    return metrics, {a: np.array(v) for a, v in all_preds.items()}, {a: np.array(v) for a, v in all_labels.items()}


# ---------------------------------------------
# 9.  Plotting & Checkpointing
# ---------------------------------------------

def save_loss_plot(train_losses, val_losses, out_path, stopped_epoch=None):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Train loss", linewidth=1.8)
    ax.plot(epochs, val_losses,   label="Val loss",   linewidth=1.8)
    if stopped_epoch is not None:
        ax.axvline(stopped_epoch, color="red", linestyle="--", linewidth=1.2, label=f"Early stop")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def save_confusion_matrix(y_true, y_pred, class_names, attr, out_dir):
    cm_counts = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm   = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")

    def _plot(cm, title, fname, fmt):
        n = len(class_names)
        fig_side = max(8, n * 0.45)
        fig, ax = plt.subplots(figsize=(fig_side, fig_side))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, xticks_rotation="vertical", colorbar=True, values_format=fmt)
        ax.set_title(f"{attr} - {title}")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=100)
        plt.close(fig)

    _plot(cm_counts, "Counts", f"{attr}_cm_counts.png", "d")
    _plot(cm_norm, "Normalised (recall)", f"{attr}_cm_norm.png", ".2f")

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.where(cm_counts.sum(axis=1) > 0, cm_counts.diagonal() / cm_counts.sum(axis=1), 0.0)
    per_class_dict = {class_names[i]: float(f"{per_class_acc[i]:.4f}") for i in range(len(class_names))}
    
    with open(os.path.join(out_dir, f"{attr}_per_class_acc.json"), "w") as f:
        json.dump(per_class_dict, f, indent=2)
    return per_class_dict

def save_checkpoint(path, model, optimizer, scheduler, epoch, vocabs, metrics, args):
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "vocabs": {k: list(v.keys()) for k, v in vocabs.items()},
        "metrics": metrics, "args": vars(args),
    }, path)

def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt.get("metrics", {})


def compute_class_weights(train_samples, vocabs, attr, device):
    counter = Counter([s[attr] for s in train_samples])
    num_classes = len(vocabs[attr])
    counts = torch.zeros(num_classes)
    for label_str, idx in vocabs[attr].items():
        counts[idx] = counter.get(label_str, 0)
    counts  = torch.clamp(counts, min=1)
    weights = 1.0 / counts
    return (weights / weights.sum() * num_classes).to(device)

def compute_proportion_alpha(train_samples, vocabs, attr, device):
    counter = Counter([s[attr] for s in train_samples])
    num_classes = len(vocabs[attr])
    counts = torch.zeros(num_classes)
    for label_str, idx in vocabs[attr].items():
        counts[idx] = counter.get(label_str, 0)
    return (counts / counts.sum().clamp(min=1)).to(device)


# ---------------------------------------------
# 10. Main Execution
# ---------------------------------------------

def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # -- Folders ------------------
    os.makedirs(args.workspace, exist_ok=True)
    if args.continue_exp:
        exp_dir = resolve_exp_dir(args.workspace, args.continue_exp)
        continuing = True
        print(f"[INFO] Continuing experiment from: {exp_dir}")
    else:
        exp_dir = make_exp_dir(args)
        continuing = False
        print(f"[INFO] New experiment directory: {exp_dir}")
    cm_dir = os.path.join(exp_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    # -- Load Annotations ---------
    print(f"[INFO] Loading annotations ({args.format} format) ...")
    all_samples = load_samples_jbcs(args.ann_json) if args.format == "jbcs" else load_samples_lplc(args.ann_json)

    filename_map = None
    if args.filename_map is not None:
        filename_map = load_filename_map(args.filename_map)

    # -- Load Multi-Features ------
    feat_index, loaded_types = load_multi_features(
        samples=all_samples,
        features_dir=args.features_dir,
        feature_types=args.feature_types,
        pool_strategy=args.pool_strategy,
        fuse_cls_gap=args.fuse_cls_gap,
        filename_map=filename_map,
    )
    
    if not feat_index:
        raise ValueError("No images had all the requested feature types. Check your paths.")

    # Filter samples to those with fully fused features
    all_samples = [s for s in all_samples if s["filename"] in feat_index]
    embed_dim = next(iter(feat_index.values())).shape[0]
    print(f"[INFO] Final concatenated embedding dimension: {embed_dim}")

    # -- Target Attributes --------
    if args.attribute == "all":
        target_attrs = ["type", "make", "model", "color"]
    else:
        valid = {"type", "make", "model", "color"}
        target_attrs = [a.strip() for a in args.attribute.split(",")]
        for attr in target_attrs:
            if attr not in valid: raise ValueError(f"Invalid attribute '{attr}'. Allowed: {valid}")

    # -- Vocabularies -------------
    if continuing:
        with open(os.path.join(exp_dir, "vocabs.json")) as f:
            vocabs = {attr: {v: i for i, v in enumerate(lst)} for attr, lst in json.load(f).items()}
    else:
        vocabs = build_vocabs(all_samples, target_attrs)
        with open(os.path.join(exp_dir, "vocabs.json"), "w") as f:
            json.dump({attr: list(v.keys()) for attr, v in vocabs.items()}, f, indent=2)

    for attr, v in vocabs.items():
        print(f"  {attr}: {len(v)} classes")

    # -- Splits -------------------
    if args.splits is None:
        random.shuffle(all_samples)
        n = len(all_samples)
        n_test, n_val = int(n * args.test_split), int(n * args.val_split)
        test_samples  = all_samples[:n_test]
        val_samples   = all_samples[n_test: n_test + n_val]
        train_samples = all_samples[n_test + n_val:]
    else:
        train_files = load_split_file(os.path.join(args.splits, "train.txt"))
        val_files   = load_split_file(os.path.join(args.splits, "val.txt"))
        test_files  = load_split_file(os.path.join(args.splits, "test.txt"))
        
        train_samples, val_samples, test_samples = [], [], []
        for s in all_samples:
            if s["filename"] in train_files: train_samples.append(s)
            elif s["filename"] in val_files: val_samples.append(s)
            elif s["filename"] in test_files: test_samples.append(s)

    print(f"[INFO] train={len(train_samples):,}  val={len(val_samples):,}  test={len(test_samples):,}")

    # Optional train filtering by rear_view
    if args.train_filter_rear_view.lower() != "all":
        target = args.train_filter_rear_view.upper()
        train_samples = [s for s in train_samples if s.get("rear_view", "") == target]
        print(f"[INFO] Filtered train samples by rear_view={target}, remaining: {len(train_samples):,}")

    # Optional train filtering by infrared
    if args.train_filter_infrared.lower() != "all":
        target = args.train_filter_infrared.upper()
        train_samples = [s for s in train_samples if s.get("infrared", "") == target]
        print(f"[INFO] Filtered train samples by infrared={target}, remaining: {len(train_samples):,}")

    if len(train_samples) == 0:
        raise ValueError("No training samples left after applying rear_view/infrared filters!")

    # Optional train subsampling
    if args.train_fraction < 1.0:
        if not (0.0 < args.train_fraction <= 1.0):
            raise ValueError("--train_fraction must be in (0, 1].")

        rng = random.Random(args.seed)
        rng.shuffle(train_samples)

        keep_n = max(1, int(len(train_samples) * args.train_fraction))
        original_n = len(train_samples)

        train_samples = train_samples[:keep_n]

        print(
            f"[INFO] Train subsampling enabled: "
            f"keeping {keep_n:,}/{original_n:,} "
            f"({100.0 * args.train_fraction:.1f}%) train samples"
        )

    print(
        f"[INFO] train={len(train_samples):,}  "
        f"val={len(val_samples):,}  "
        f"test={len(test_samples):,}"
    )

    # -- Losses -------------------
    loss_mode = args.loss
    needs_weights = loss_mode in ("weighted_ce", "scaled_weighted_ce")
    
    loss_cfg = {
        "mode": loss_mode,
        "class_weights": {a: compute_class_weights(train_samples, vocabs, a, device) for a in vocabs} if needs_weights else {},
        "scale_factors": {a: float(len(vocabs[a])) for a in vocabs} if loss_mode == "scaled_weighted_ce" else {},
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "proportion_alpha": {a: compute_proportion_alpha(train_samples, vocabs, a, device) for a in vocabs} if loss_mode == "focal_proportion" else {},
    }

    # -- DataLoaders --------------
    train_loader = DataLoader(FeatureDataset(train_samples, feat_index, vocabs), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader   = DataLoader(FeatureDataset(val_samples,   feat_index, vocabs), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(FeatureDataset(test_samples,  feat_index, vocabs), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # -- Model --------------------
    model = FeatureClassifier(
        embed_dim=embed_dim, num_classes={a: len(v) for a, v in vocabs.items()},
        head_layers=args.head_layers, head_hidden=args.head_hidden, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    start_epoch = 1
    if continuing and os.path.isfile(os.path.join(exp_dir, "last.pth")):
        start_epoch, _ = load_checkpoint(os.path.join(exp_dir, "last.pth"), model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"[INFO] Resumed from epoch {start_epoch - 1}")

    # -- Logging ------------------
    log_path = os.path.join(exp_dir, "log.csv")
    if not continuing or not os.path.isfile(log_path):
        with open(log_path, "w") as f:
            cols = ["epoch", "split", "loss"]
            for a in vocabs: cols += [f"acc_macro_{a}", f"acc_micro_{a}", f"f1_macro_{a}", f"f1_micro_{a}"]
            f.write(",".join(cols) + "\n")

    best_val_loss, best_train_loss = float("inf"), float("inf")
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    train_losses, val_losses, stopped_epoch = [], [], None

    if continuing and os.path.isfile(log_path):
        with open(log_path) as f:
            for row in csv.DictReader(f):
                if row["split"] == "train": train_losses.append(float(row["loss"]))
                elif row["split"] == "val": val_losses.append(float(row["loss"]))

    # -- Training Loop ------------
    for epoch in range(start_epoch, start_epoch + args.epochs):
        tm = run_epoch(model, train_loader, optimizer, device, loss_cfg, train=True)
        vm = run_epoch(model, val_loader,   optimizer, device, None,     train=False)
        scheduler.step()

        train_losses.append(tm["loss"])
        val_losses.append(vm["loss"])

        acc_str = "  ".join(f"{a}: accM={vm[f'acc_macro_{a}']:.3f}" for a in vocabs)
        print(f"[{epoch:03d}]  train_loss={tm['loss']:.4f}  val_loss={vm['loss']:.4f}  {acc_str}")

        with open(log_path, "a") as f:
            for split, m in [("train", tm), ("val", vm)]:
                row = [str(epoch), split, f"{m['loss']:.6f}"]
                for a in vocabs: row += [f"{m[f'acc_macro_{a}']:.6f}", f"{m[f'acc_micro_{a}']:.6f}", f"{m[f'f1_macro_{a}']:.6f}", f"{m[f'f1_micro_{a}']:.6f}"]
                f.write(",".join(row) + "\n")

        save_checkpoint(os.path.join(exp_dir, "last.pth"), model, optimizer, scheduler, epoch, vocabs, vm, args)
        if vm["loss"] < best_val_loss:
            best_val_loss = vm["loss"]
            save_checkpoint(os.path.join(exp_dir, "best_val.pth"), model, optimizer, scheduler, epoch, vocabs, vm, args)
        if tm["loss"] < best_train_loss:
            best_train_loss = tm["loss"]
            save_checkpoint(os.path.join(exp_dir, "best_train.pth"), model, optimizer, scheduler, epoch, vocabs, tm, args)

        if early_stopping.step(vm["loss"]):
            stopped_epoch = epoch
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    save_loss_plot(train_losses, val_losses, os.path.join(exp_dir, "loss_curve.png"), stopped_epoch)

    # -- Test Evaluation ----------
    best_ckpt = torch.load(os.path.join(exp_dir, "best_val.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model"])

    if len(test_loader) > 0:
        test_metrics, all_preds, all_labels = run_test(model, test_loader, device, vocabs)
        print("\n" + "-"*70)
        print(f"  TEST RESULTS  (best_val epoch={best_ckpt['epoch']})")
        for a in vocabs:
            print(f"  {a:<10} Macro_Acc: {test_metrics[f'acc_macro_{a}']:.4f} | Micro_Acc: {test_metrics[f'acc_micro_{a}']:.4f}")
        
        per_class_summary = {}
        for a in vocabs:
            class_names = [v for v, i in sorted(vocabs[a].items(), key=lambda x: x[1])]
            per_class_summary[a] = save_confusion_matrix(all_labels[a], all_preds[a], class_names, a, cm_dir)
        
        with open(os.path.join(exp_dir, "test_results.json"), "w") as f:
            json.dump({
                "epoch": best_ckpt["epoch"],
                "feature_types": loaded_types,
                "pool_strategy": args.pool_strategy,
                "fuse_cls_gap": args.fuse_cls_gap,
                "embed_dim": embed_dim,
                "overall": test_metrics,
                "per_class_acc": per_class_summary,
            }, f, indent=2)

if __name__ == "__main__":
    main()
