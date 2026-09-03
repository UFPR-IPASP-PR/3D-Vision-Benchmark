# 3D-Vision-Benchmark

Code for reproducing the benchmark results from our SIBGRAPI 2026 paper.

<!-- TODO (stage 4): one-paragraph description of what this benchmark measures
     and a link to the paper once available. -->

## Repository structure

```
3D-Vision-Benchmark/
├── extraction/                  # Per-model feature extraction scripts
│   ├── extract_hf.py            # dino_v1, dinov3, sam3 (HuggingFace-only, no repo to clone)
│   ├── dust3r/
│   ├── depth-anything-3/
│   ├── Depth-Anything/          # (v1, + DINOv2)
│   ├── DepthAnything-V2/
│   ├── sam-3d-objects/
│   ├── Hunyuan3D-2.1/
│   ├── mast3r/
│   ├── crocov2/
│   └── ibot/
├── training/                    
│   ├── single_feature_experiments.py
│   ├── feature_fusion_run_experiments.py
│   └── run_ood_experiment.py
└── scripts/
    ├── run_experiments.sh        
    └── configs/                  
        ├── dino_single.sh
        ├── multi_attr.sh
        ├── ibot.sh
        ├── dinov2_da.sh
        ├── naver_encoders.sh
        └── various_backbones.sh
```

> **Note:** `extraction/<model>/` folders hold only *our* extraction script for
> that model — they are not the cloned upstream repos. See "Setup" below.

## Setup

1. For each model folder under `extraction/` (all except `extract_hf.py`),
   clone the official repository elsewhere on your machine and install it
   per its own instructions, e.g.:

   ```bash
   git clone <official repo URL for the model>
   cd <cloned repo>
   pip install -e .
   ```
The official repositories for each model can be found at:

| Model | Official Repository |
| --- | --- |
| dinov1 | [GitHub](https://github.com/facebookresearch/dino) |
| dinov2 | [GitHub](https://github.com/facebookresearch/dinov2) |
| dinov3 | [GitHub](https://github.com/facebookresearch/dinov3) |
| sam3 | [GitHub](https://github.com/facebookresearch/sam3) |
| dust3r | [GitHub](https://github.com/naver/dust3r) |
| mast3r | [GitHub](https://github.com/naver/mast3r) |
| crocov2 | [GitHub](https://github.com/naver/croco) |
| depth-anything v1 | [GitHub](https://github.com/globalwetlands/depth-anything-V1) |
| depth-anything v2 | [GitHub](https://github.com/DepthAnything/Depth-Anything-V2) |
| depth-anything 3 | [GitHub](https://github.com/bytedance-seed/depth-anything-3) |
| sam-3d-objects | [GitHub](https://github.com/facebookresearch/sam-3d-objects) |
| Hunyuan3D-2.1 | [GitHub](https://github.com/tencent-hunyuan/hunyuan3d-2.1) |
| ibot | [GitHub](https://github.com/bytedance/ibot) |
| Gamba | [GitHub](https://github.com/SkyworkAI/Gamba) |

2. Copy the corresponding script from `extraction/<model>/` into your local
   clone of that model (or add the clone to your `PYTHONPATH`), then run it
   to extract features. `dino_v1`, `dinov3`, and `sam3` need no clone —
   run `extraction/extract_hf.py <model>` directly.

3. Edit the relevant config file under `scripts/configs/` to point at your
   feature output folders (each has `TODO`-marked placeholders), then run:

   ```bash
   ./scripts/run_experiments.sh scripts/configs/<name>.sh
   ```

## Model weights

| Model | Checkpoint / weights used | Source |
|---|---|---|
| dinov1 | `facebook/dino-vitb8` | HF Hub |
| dinov2 | `ViT-L/14 distilled w/ Registers` | [GitHub](https://github.com/facebookresearch/dinov2) |
| dinov3 | `dinov3-vit7b16-pretrain-lvd1689m`| HF Hub |
| sam3 | `facebook/sam3` | HF Hub |
| dust3r | `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` | [GitHub](https://github.com/naver/dust3r#checkpoints) |
| mast3r | `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` | [GitHub](https://github.com/naver/mast3r#checkpoints) |
| crocov2 | `CroCo_V2_ViTLarge_BaseDecoder.pth` | [GitHub](https://github.com/naver/croco) |
| depth-anything v1 | `LiheYoung/depth_anything_vitl14` | HF Hub |
| depth-anything v2 | `depth_anything_v2_vitl` | [GitHub](https://github.com/DepthAnything/Depth-Anything-V2) |
| depth-anything 3 | `depth-anything/DA3METRIC-LARGE` | HF Hub |
| sam-3d-objects | `InferencePipelinePointMap` | HF Hub |
| Hunyuan3D-2.1 | `hunyuan3d-dit-v2-1` | HF Hub |
| ibot | `imagenet22k` | [GitHub](https://github.com/bytedance/ibot) |
| Gamba | `gamba_ep399.pth` | HF Hub |

## Environment

This repo uses **two kinds of environments**:

- **Extraction envs** (one per model, under `extraction/<model>/`): follow
  each official repo's own install instructions (step 1 above). These are
  the standard envs from each model's GitHub page.
- **Training env** (`training/*.py` and `scripts/run_experiments.sh`):

  ```bash
  pip install -r requirements.txt
  ```

  See `requirements.txt`. We recommend Python 3.10.20, as that was the version utilized.

## Usage Instructions

Clone and install the dependencies of each model, following official instructions.

Place the relevant extraction script alongside the cloned repo and run it, saving
the features each to its own folder (or file, for `dinov3` and `sam3`, via `extract_hf.py`).

Edit the launcher scripts under `scripts/` to point to your feature folders, then run.
