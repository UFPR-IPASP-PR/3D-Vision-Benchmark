#!/bin/bash

set -m

max_jobs=4
job_count=0

root="../nobackup5/avdelazeri/UFPR-VeSV-Preview"

MODEL="unified.py"

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

CROCOV2_PATH="$root/croco_features/encoder"
MAST3R_PATH="$root/master_features/encoder"
DUST3R_PATH="$root/duster_features/encoder"

cleanup() {
    echo "Stopping all running jobs..."
    jobs -p | xargs -r kill
    exit 1
}

trap cleanup SIGINT SIGTERM


for path in \
    "$CROCOV2_PATH" \
    "$MAST3R_PATH" \
    "$DUST3R_PATH"
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



run_feature "duster_encoder"   "$DUST3R_PATH" "type"
run_feature "duster_encoder"   "$DUST3R_PATH" "make"
run_feature "duster_encoder"   "$DUST3R_PATH" "model"

run_feature "master_encoder"   "$MAST3R_PATH" "type"
run_feature "master_encoder"   "$MAST3R_PATH" "make"
run_feature "master_encoder"   "$MAST3R_PATH" "model"

run_feature "crocov2_encoder"   "$CROCOV2_PATH" "type"
run_feature "crocov2_encoder"   "$CROCOV2_PATH" "make"
run_feature "crocov2_encoder"   "$CROCOV2_PATH" "model"


wait

echo "All experiments completed."
