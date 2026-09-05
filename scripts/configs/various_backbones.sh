# Multi-feature-type fusion runs (CLS + patches concatenated via
# feature_fusion_run_experiments.py's --feature_types comma-list, no
# --fuse_cls_gap): uses run_fusion_multi, where feature_dir is the PARENT
# folder that directly contains all the named feature_types subfolders.
#
ROOT=""   # TODO: set to your dataset root

DINO_V2_PATH="$ROOT/dino_features"
DEPTH_ANYTHING_V1_PATH="$ROOT/depth_anything_features"
DEPTH_ANYTHING_V2_PATH="$ROOT/depth_anything_v2"
DEPTH_ANYTHING_V3_PATH="$ROOT/depth_anything_v3"
GAMBA_PATH="$ROOT/gamba_features/mlp_feats/"
HUNYUAN_PATH="$ROOT/hunyuan/geometry_tokens"
SAM3D_PATH="$ROOT/sam_features_all_channels/shape_latent"

check_paths "$DINO_V2_PATH" "$DEPTH_ANYTHING_V1_PATH" "$DEPTH_ANYTHING_V2_PATH" \
            "$DEPTH_ANYTHING_V3_PATH" "$GAMBA_PATH" "$HUNYUAN_PATH" "$SAM3D_PATH"

run_fusion "gamba_mlp_feats"    "$GAMBA_PATH"    "type"
run_fusion "gamba_mlp_feats"    "$GAMBA_PATH"    "make"
run_fusion "gamba_mlp_feats"    "$GAMBA_PATH"    "model"
run_fusion "hunyuan_geometry"   "$HUNYUAN_PATH"  "type"
run_fusion "hunyuan_geometry"   "$HUNYUAN_PATH"  "make"
run_fusion "hunyuan_geometry"   "$HUNYUAN_PATH"  "model"
run_fusion "sam3d_shape_latent" "$SAM3D_PATH"    "type"
run_fusion "sam3d_shape_latent" "$SAM3D_PATH"    "make"
run_fusion "sam3d_shape_latent" "$SAM3D_PATH"    "model"
