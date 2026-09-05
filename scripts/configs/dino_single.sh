# Config for run_experiments.sh — migrated from the original run_parallel_dino.sh.
# Trains a single joint multi-head classifier (type+make+model at once) on
# one flat feature .npz file, across 10 splits, using single_feature_experiments.py.

ROOT=""            # TODO: set to your dataset root (must contain annotations.json, splits/)
FEATURES="$ROOT"   # TODO: point at a specific .npz feature file, e.g. "$ROOT/features/dino_cls_avg.npz"

WORKSPACE=""       # TODO: set an output workspace directory
ATTRIBUTES="type,make,model"

DROPOUT=0.2
MAX_JOBS=4

run_single "$FEATURES" "$ATTRIBUTES" "$WORKSPACE"
