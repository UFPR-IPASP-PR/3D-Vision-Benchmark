#!/bin/bash

set -m 

max_jobs=4
job_count=0

root=""

features=$root""
attributes="type,make,model"
workspace=""
model="single_feature_experiments"

i=0

cleanup() {
    echo "Stopping all running jobs..."
    jobs -p | xargs -r kill
    exit 1
}

trap cleanup SIGINT SIGTERM

for dropout in 0.2; do
    for i in {0..9}; do
        python $model \
            --dataset UFPR-VeSV \
            --ann_json "$root/annotations.json" \
            --features=$features \
            --format jbcs \
            --workspace $workspace \
            --epochs 100 \
            --batch_size 32 \
            --dropout $dropout \
            --lr 0.0001 \
            --loss weighted_ce \
            --attribute $attributes \
            --split "$root/splits/$i" \
            --early_stopping_patience 5 \
            --num_workers 1 \
            --noise_std 0 &

        ((job_count++))
        if ((job_count >= max_jobs)); then
            wait -n
            ((job_count--))
        fi

    done
done

wait
