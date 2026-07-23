#!/bin/bash

set -m

max_jobs=4
job_count=0

root=""

MODEL="feature_fusion_run_experiments.py"

ATTRIBUTES="type,make,model"

EPOCHS=100
BATCH_SIZE=32
LR=0.0001
DROPOUT=0.2
LOSS="weighted_ce"
PATIENCE=5

#
# Paths to the pre-extracted features
#
DINO_V2_PATH="$root/dino_features"
DEPTH_ANYTHING_V1_PATH="$root/depth_anything_features"
DEPTH_ANYTHING_V2_PATH="$root/depth_anything_v2"
DEPTH_ANYTHING_V3_PATH="$root/depth_anything_v3"
GAMBA_PATH="$root/gamba_features/mlp_feats/"
HUNYUAN_PATH="$root/hunyuan/geometry_tokens"
SAM3D_PATH="$root/sam_features_all_channels/shape_latent"

cleanup() {
    echo "Stopping all running jobs..."
    jobs -p | xargs -r kill
    exit 1
}

trap cleanup SIGINT SIGTERM


for path in \
    "$DINO_V2_PATH" \
    "$DEPTH_ANYTHING_V1_PATH" \
    "$DEPTH_ANYTHING_V2_PATH" \
    "$GAMBA_PATH" \
    "$HUNYUAN_PATH" \
    "$SAM3D_PATH"
do
    echo $path
    if [[ ! -d "$path" ]]; then
        echo "ERROR: Directory does not exist: $path"
        exit 1
    fi
done

run_feature() {

    local feature_name="$1"
    local feature_path="$2"
    local attr_name="$3"

    echo "===================================================="
    echo "Running experiments for: $feature_name"
    echo "Feature path: $feature_path"
    echo "===================================================="

    for split in {0..9}; do

        python "$MODEL" \
            --dataset UFPR-VeSV \
            --ann_json "$root/annotations.json" \
            --features_dir "$feature_path" \
            --feature_types "$(basename "$feature_path")" \
            --pool_strategy mean \
            --format jbcs \
            --workspace "linear_probes/runs/${feature_name}/${attr_name}" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --dropout "$DROPOUT" \
            --head_hidden 0 \
            --lr "$LR" \
            --loss "$LOSS" \
            --attribute "$attr_name" \
            --splits "$root/splits/$split" \
            --early_stopping_patience "$PATIENCE" &

        ((job_count++))

        if ((job_count >= max_jobs)); then
            wait -n
            ((job_count--))
        fi
    done
}


run_feature "dino_v2_cls,dino_v2_patches"                      "$DINO_V2_PATH"               "type" 
run_feature "dino_v2_cls,dino_v2_patches"                      "$DINO_V2_PATH"               "make"
run_feature "dino_v2_cls,dino_v2_patches"                      "$DINO_V2_PATH"               "model"
run_feature "depth_anything_v1_cls,depth_anything_v1_patches"  "$DEPTH_ANYTHING_V1_PATH" "type"
run_feature "depth_anything_v1_cls,depth_anything_v1_patches"  "$DEPTH_ANYTHING_V1_PATH" "make"
run_feature "depth_anything_v1_cls,depth_anything_v1_patches"  "$DEPTH_ANYTHING_V1_PATH" "model"
run_feature "depth_anything_v2_cls,depth_anything_v2_patches"  "$DEPTH_ANYTHING_V2_PATH" "type"
run_feature "depth_anything_v2_cls,depth_anything_v2_patches"  "$DEPTH_ANYTHING_V2_PATH" "make"
run_feature "depth_anything_v2_cls,depth_anything_v2_patches"  "$DEPTH_ANYTHING_V2_PATH" "model"
run_feature "depth_anything_vv_cls,depth_anything_v3_patches"  "$DEPTH_ANYTHING_V3_PATH" "type"
run_feature "depth_anything_vv_cls,depth_anything_v3_patches"  "$DEPTH_ANYTHING_V3_PATH" "make"
run_feature "depth_anything_vv_cls,depth_anything_v3_patches"  "$DEPTH_ANYTHING_V3_PATH" "model"
#run_feature "gamba_mlp_feats"        "$GAMBA_PATH"    "type"
#run_feature "gamba_mlp_feats"        "$GAMBA_PATH"    "make"
#run_feature "gamba_mlp_feats"        "$GAMBA_PATH"    "model"
#run_feature "hunyuan_geometry"       "$HUNYUAN_PATH"  "type" 
#run_feature "hunyuan_geometry"       "$HUNYUAN_PATH"  "make"
#run_feature "hunyuan_geometry"       "$HUNYUAN_PATH"  "model"
#run_feature "sam3d_shape_latent"     "$SAM3D_PATH"    "type"
#run_feature "sam3d_shape_latent"     "$SAM3D_PATH"    "make"
#run_feature "sam3d_shape_latent"     "$SAM3D_PATH"    "model"

wait

echo "All experiments completed."
