# Config for run_experiments.sh — migrated from the original run_parallel_ibot.sh.
# CLS+GAP fusion runs (feature_fusion_run_experiments.py --fuse_cls_gap) for iBOT.
#
# Each *_PATCHES_PATH must have a sibling CLS folder in the same parent dir;
# the training script auto-loads it when --fuse_cls_gap is set. Do not also
# list the CLS folder in --feature_types yourself.

ROOT=""   # TODO: set to your dataset root

IBOT_22k_CLS_PATH="$ROOT/iBOT_22k_features/ibot_cls"
IBOT_22k_PATCHES_PATH="$ROOT/iBOT_22k_features/ibot_patches"

check_paths "$IBOT_22k_CLS_PATH" "$IBOT_22k_PATCHES_PATH"

run_fusion_clsgap "iBOT_imagenet22k_clsgap" "$IBOT_22k_PATCHES_PATH" "type"
run_fusion_clsgap "iBOT_imagenet22k_clsgap" "$IBOT_22k_PATCHES_PATH" "make"
run_fusion_clsgap "iBOT_imagenet22k_clsgap" "$IBOT_22k_PATCHES_PATH" "model"
