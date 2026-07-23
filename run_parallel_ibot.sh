#!/bin/bash

set -m

max_jobs=4
job_count=0

root=""

MODEL="feature_fusion_run_experiments"

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
# --- CLS+GAP fusion features (used with --fuse_cls_gap) ---
# Each *_PATCHES_PATH below must point at a "*_patches" folder that has a
# sibling CLS/cam folder living in the SAME parent directory. unified_2_.py
# auto-loads that sibling and concatenates [CLS | GAP(patches)]:
#   dino_patches <-> dino_cls
#   da_patches   <-> da_cls   (used for both depth_anything v1 and v2)
#   da3_patches  <-> da3_cam
# Do not pass the CLS/cam folder in --feature_types yourself; it is loaded
# automatically as soon as --fuse_cls_gap is set.

IBOT_22k_CLS_PATH="$root/iBOT_22k_features/ibot_cls"
IBOT_22k_PATCHES_PATH="$root/iBOT_22k_features/ibot_patches"

IBOT_1k_CLS_PATH="$root/iBOT_1k_features/ibot_cls"
IBOT_1k_PATCHES_PATH="$root/iBOT_1k_features/ibot_patches"

cleanup() {
    echo "Stopping all running jobs..."
    jobs -p | xargs -r kill
    exit 1
}

trap cleanup SIGINT SIGTERM

# Plain single-folder feature run (mean-pooled, no CLS/GAP split).
# Kept for gamba/hunyuan/sam3d; currently unused below.
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
            --features_dir "$feature_path/.." \
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

# CLS+GAP fusion run: identical to run_feature() above but adds
# --fuse_cls_gap so unified_2_.py also loads the sibling CLS/cam vector and
# concatenates it with the gap-pooled patches into one [CLS|GAP] vector.
run_feature_clsgap() {

    local feature_name="$1"
    local feature_path="$2"   # path to the "*_patches" folder; its CLS/cam
                               # sibling is auto-loaded by --fuse_cls_gap
    local attr_name="$3"

    echo "===================================================="
    echo "Running experiments for: $feature_name (CLS + GAP fusion)"
    echo "Feature path: $feature_path"
    echo "===================================================="

    for split in {0..9}; do

        python "$MODEL" \
            --dataset UFPR-VeSV \
            --ann_json "$root/annotations.json" \
            --features_dir "$feature_path/.." \
            --feature_types "$(basename "$feature_path")" \
            --fuse_cls_gap \
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




#run_feature_clsgap "iBOT_imagenet22k_clsgap"  "$IBOT_22k_PATCHES_PATH" "type"
#run_feature_clsgap "iBOT_imagenet22k_clsgap"  "$IBOT_22k_PATCHES_PATH" "make"
#run_feature_clsgap "iBOT_imagenet22k_clsgap"  "$IBOT_22k_PATCHES_PATH" "model"
run_feature_clsgap "iBOT_imagenet1k_clsgap"  "$IBOT_1k_PATCHES_PATH" "type"
run_feature_clsgap "iBOT_imagenet1k_clsgap"  "$IBOT_1k_PATCHES_PATH" "make"
run_feature_clsgap "iBOT_imagenet1k_clsgap"  "$IBOT_1k_PATCHES_PATH" "model"

wait

echo "All experiments completed."
