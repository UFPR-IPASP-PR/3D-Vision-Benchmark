# Config for run_experiments.sh — SAM3 (single flat .npz, extracted via
# extraction/extract_hf.py's sam3 subcommand / the standalone sam.py).
#

ROOT=""                          # TODO: set to your dataset root
DATASET=""                       
FEATURES="$ROOT/sam3.npz"        # TODO: point at your actual sam3 .npz
                                      
WORKSPACE_PREFIX="sam3/runs"     # TODO: choose a real workspace name
HEAD_LAYERS=0
DROPOUT=0.2

run_single_per_attr "$FEATURES" "$WORKSPACE_PREFIX" "type"  "$HEAD_LAYERS"
run_single_per_attr "$FEATURES" "$WORKSPACE_PREFIX" "make"  "$HEAD_LAYERS"
run_single_per_attr "$FEATURES" "$WORKSPACE_PREFIX" "model" "$HEAD_LAYERS"
