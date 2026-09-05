# Config for run_experiments.sh — new addition for DINOv3 (previously
# missing). DINOv3 features are extracted as ONE flat .npz file (via
# extraction/extract_hf.py's dinov3 subcommand) and trained one attribute
# at a time, migrated directly from your real command:

ROOT=""                                    # TODO: set to your dataset root
DATASET=""                                 # TODO: name of your dataset here
FEATURES="$ROOT/dinov3-3M-cls-avg.npz"     # TODO: point at your actual dinov3 .npz
WORKSPACE_PREFIX="dinov3/runs"             # TODO: choose a real workspace name
HEAD_LAYERS=0
DROPOUT=0.2

run_single_per_attr "$FEATURES" "$WORKSPACE_PREFIX" "type"  "$HEAD_LAYERS"
run_single_per_attr "$FEATURES" "$WORKSPACE_PREFIX" "make"  "$HEAD_LAYERS"
run_single_per_attr "$FEATURES" "$WORKSPACE_PREFIX" "model" "$HEAD_LAYERS"
