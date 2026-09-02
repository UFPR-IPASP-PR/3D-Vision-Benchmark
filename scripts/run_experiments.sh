#!/bin/bash
#
# run_experiments.sh — generic parallel launcher for the training scripts.
#
# Usage:
#   ./run_experiments.sh <config_file> [--max_jobs N]
#
# A config file is a bash script that:
#   1. Sets ROOT (and optionally overrides EPOCHS, BATCH_SIZE, LR, DROPOUT,
#      LOSS, PATIENCE, NUM_WORKERS, MAX_JOBS, SPLITS, FORMAT, DATASET).
#   2. Calls one or more of the run_* functions below for each
#      feature/attribute combination it wants to launch.
#

set -m

CONFIG_FILE="$1"
if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
    echo "Usage: $0 <config_file>"
    echo "See scripts/configs/*.sh for examples."
    exit 1
fi

# ---- defaults; a config file may override any of these before calling run_* ----
ROOT=""
SINGLE_SCRIPT="training/single_feature_experiments.py"
FUSION_SCRIPT="training/feature_fusion_run_experiments.py"
DATASET="UFPR-VeSV"
FORMAT="jbcs"
EPOCHS=100
BATCH_SIZE=32
LR=0.0001
DROPOUT=0.2
LOSS="weighted_ce"
PATIENCE=5
NUM_WORKERS=1
MAX_JOBS=4
SPLITS="0 1 2 3 4 5 6 7 8 9"
FILENAME_MAP=""

job_count=0

cleanup() {
    echo "Stopping all running jobs..."
    jobs -p | xargs -r kill
    exit 1
}
trap cleanup SIGINT SIGTERM

wait_for_slot() {
    ((job_count++))
    if ((job_count >= MAX_JOBS)); then
        wait -n
        ((job_count--))
    fi
}

# Fails loudly if any of the given directories don't exist. Call this from a
# config file after setting its path variables, before calling run_*.
check_paths() {
    for path in "$@"; do
        if [[ ! -d "$path" ]]; then
            echo "ERROR: Directory does not exist: $path"
            exit 1
        fi
    done
}

# ---------------------------------------------------------------------------
# run_single: single_feature_experiments.py
#   One flat .npz feature file; --attribute may be a single attribute or a
#   comma-separated list trained jointly (multi-head) in one job.
# ---------------------------------------------------------------------------
run_single() {
    local features="$1" attribute="$2" workspace="$3"
    for split in $SPLITS; do
        python "$SINGLE_SCRIPT" \
            --dataset "$DATASET" \
            --ann_json "$ROOT/annotations.json" \
            --features "$features" \
            --format "$FORMAT" \
            --workspace "$workspace" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --dropout "$DROPOUT" \
            --lr "$LR" \
            --loss "$LOSS" \
            --attribute "$attribute" \
            --split "$ROOT/splits/$split" \
            --early_stopping_patience "$PATIENCE" \
            --num_workers "$NUM_WORKERS" \
            --noise_std 0 &
        wait_for_slot
    done
}

# ---------------------------------------------------------------------------
# run_single_per_attr: single_feature_experiments.py, ONE attribute per call
#   (as opposed to run_single's comma-joined multi-head "type,make,model" in
#   one job). Matches the real command used for dinov3:
#     for task in type make model; do for split in {0..9}; do
#         python unified_dino_experiments.py --dataset vesv ... \
#             --workspace dinov3_sanity/runs/$task --head_layers 0 \
#             --attribute $task --splits .../splits/$split ...
#     done; done
# ---------------------------------------------------------------------------
run_single_per_attr() {
    local features="$1" workspace_prefix="$2" attr="$3" head_layers="${4:-0}"
    local extra_args=()
    if [[ -n "$FILENAME_MAP" ]]; then
        extra_args+=(--filename_map "$FILENAME_MAP")
    fi
    for split in $SPLITS; do
        python "$SINGLE_SCRIPT" \
            --dataset "$DATASET" \
            --ann_json "$ROOT/annotations.json" \
            --features "$features" \
            --format "$FORMAT" \
            --workspace "$workspace_prefix/$attr" \
            --head_layers "$head_layers" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --dropout "$DROPOUT" \
            --lr "$LR" \
            --loss "$LOSS" \
            --attribute "$attr" \
            --splits "$ROOT/splits/$split" \
            --early_stopping_patience "$PATIENCE" \
            "${extra_args[@]}" &
        wait_for_slot
    done
}

# ---------------------------------------------------------------------------
# run_fusion: feature_fusion_run_experiments.py, ONE feature type per call.
#   feature_path points at the specific leaf feature folder; features_dir and
#   feature_types are derived from it (parent dir / folder name).
# ---------------------------------------------------------------------------
run_fusion() {
    local feature_name="$1" feature_path="$2" attr="$3"
    echo "===================================================="
    echo "Running experiments for: $feature_name / $attr"
    echo "Feature path: $feature_path"
    echo "===================================================="
    for split in $SPLITS; do
        python "$FUSION_SCRIPT" \
            --dataset "$DATASET" \
            --ann_json "$ROOT/annotations.json" \
            --features_dir "$feature_path/.." \
            --feature_types "$(basename "$feature_path")" \
            --pool_strategy mean \
            --format "$FORMAT" \
            --workspace "linear_probes/runs/${feature_name}/${attr}" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --dropout "$DROPOUT" \
            --head_hidden 0 \
            --lr "$LR" \
            --loss "$LOSS" \
            --attribute "$attr" \
            --splits "$ROOT/splits/$split" \
            --early_stopping_patience "$PATIENCE" &
        wait_for_slot
    done
}

# ---------------------------------------------------------------------------
# run_fusion_clsgap: feature_fusion_run_experiments.py with --fuse_cls_gap.
#   feature_path points at a "*_patches" folder; its CLS/cam sibling (in the
#   same parent dir) is auto-loaded by the training script and concatenated
#   as [CLS | GAP(patches)]. Do not pass the CLS/cam folder yourself.
# ---------------------------------------------------------------------------
run_fusion_clsgap() {
    local feature_name="$1" feature_path="$2" attr="$3"
    echo "===================================================="
    echo "Running experiments for: $feature_name / $attr (CLS + GAP fusion)"
    echo "Feature path: $feature_path"
    echo "===================================================="
    for split in $SPLITS; do
        python "$FUSION_SCRIPT" \
            --dataset "$DATASET" \
            --ann_json "$ROOT/annotations.json" \
            --features_dir "$feature_path/.." \
            --feature_types "$(basename "$feature_path")" \
            --fuse_cls_gap \
            --pool_strategy mean \
            --format "$FORMAT" \
            --workspace "linear_probes/runs/${feature_name}/${attr}" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --dropout "$DROPOUT" \
            --head_hidden 0 \
            --lr "$LR" \
            --loss "$LOSS" \
            --attribute "$attr" \
            --splits "$ROOT/splits/$split" \
            --early_stopping_patience "$PATIENCE" &
        wait_for_slot
    done
}

# ---------------------------------------------------------------------------
# run_fusion_multi: feature_fusion_run_experiments.py, MULTIPLE feature types
#   fused in one call. feature_dir is the parent folder that directly
#   contains all the named feature_types subfolders (e.g. dino_v2_cls/ and
#   dino_v2_patches/ both live under it) — pass it as-is, unlike run_fusion.
# ---------------------------------------------------------------------------
run_fusion_multi() {
    local feature_types="$1" feature_dir="$2" attr="$3"
    echo "===================================================="
    echo "Running experiments for: $feature_types / $attr"
    echo "Feature dir: $feature_dir"
    echo "===================================================="
    for split in $SPLITS; do
        python "$FUSION_SCRIPT" \
            --dataset "$DATASET" \
            --ann_json "$ROOT/annotations.json" \
            --features_dir "$feature_dir" \
            --feature_types "$feature_types" \
            --pool_strategy mean \
            --format "$FORMAT" \
            --workspace "linear_probes/runs/${feature_types}/${attr}" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --dropout "$DROPOUT" \
            --head_hidden 0 \
            --lr "$LR" \
            --loss "$LOSS" \
            --attribute "$attr" \
            --splits "$ROOT/splits/$split" \
            --early_stopping_patience "$PATIENCE" &
        wait_for_slot
    done
}

source "$CONFIG_FILE"

wait
echo "All experiments completed."
