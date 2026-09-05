# Config for run_experiments.sh — migrated from the original run_parallel_dinov2_da.sh.
# CLS+GAP fusion runs (feature_fusion_run_experiments.py --fuse_cls_gap) for
# DINOv2 and Depth Anything v1/v2/v3.

ROOT=""   # TODO: set to your dataset root

DINO_PATCHES_PATH="$ROOT/dinov2_vitl/dino_patches"
DINO_CLS_PATH="$ROOT/dinov2_vitl/dino_cls"

DEPTH_ANYTHING_V1_PATCHES_PATH="$ROOT/depth_anything_vitl/da_patches"
DEPTH_ANYTHING_V1_CLS_PATH="$ROOT/depth_anything_vitl/da_cls"

DEPTH_ANYTHING_V2_PATCHES_PATH="$ROOT/depth_anything_v2_vitl/da_patches"
DEPTH_ANYTHING_V2_CLS_PATH="$ROOT/depth_anything_v2_vitl/da_cls"

DEPTH_ANYTHING_V3_PATCHES_PATH="$ROOT/depth_anything_v3_vitl/da3_patches"
DEPTH_ANYTHING_V3_CAM_PATH="$ROOT/depth_anything_v3_vitl/da3_cam"

check_paths "$DINO_PATCHES_PATH" "$DINO_CLS_PATH" \
            "$DEPTH_ANYTHING_V1_PATCHES_PATH" "$DEPTH_ANYTHING_V1_CLS_PATH" \
            "$DEPTH_ANYTHING_V2_PATCHES_PATH" "$DEPTH_ANYTHING_V2_CLS_PATH" \
            "$DEPTH_ANYTHING_V3_PATCHES_PATH" "$DEPTH_ANYTHING_V3_CAM_PATH"

run_fusion_clsgap "dinov2_clsgap"            "$DINO_PATCHES_PATH"            "type"
run_fusion_clsgap "dinov2_clsgap"            "$DINO_PATCHES_PATH"            "make"
run_fusion_clsgap "dinov2_clsgap"            "$DINO_PATCHES_PATH"            "model"
run_fusion_clsgap "depth_anything_v1_clsgap" "$DEPTH_ANYTHING_V1_PATCHES_PATH" "type"
run_fusion_clsgap "depth_anything_v1_clsgap" "$DEPTH_ANYTHING_V1_PATCHES_PATH" "make"
run_fusion_clsgap "depth_anything_v1_clsgap" "$DEPTH_ANYTHING_V1_PATCHES_PATH" "model"
run_fusion_clsgap "depth_anything_v2_clsgap" "$DEPTH_ANYTHING_V2_PATCHES_PATH" "type"
run_fusion_clsgap "depth_anything_v2_clsgap" "$DEPTH_ANYTHING_V2_PATCHES_PATH" "make"
run_fusion_clsgap "depth_anything_v2_clsgap" "$DEPTH_ANYTHING_V2_PATCHES_PATH" "model"
run_fusion_clsgap "depth_anything_v3_clsgap" "$DEPTH_ANYTHING_V3_PATCHES_PATH" "type"
run_fusion_clsgap "depth_anything_v3_clsgap" "$DEPTH_ANYTHING_V3_PATCHES_PATH" "make"
run_fusion_clsgap "depth_anything_v3_clsgap" "$DEPTH_ANYTHING_V3_PATCHES_PATH" "model"
