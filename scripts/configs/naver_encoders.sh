# Config for run_experiments.sh — migrated from the original run_parallel_naver.sh.
# Single-feature-type fusion runs for the Naver Labs encoder models
# (DUSt3R, MASt3R, CroCo v2).
#
# NOTE: the original script hardcoded a personal path here
# (root="../nobackup5/avdelazeri/UFPR-VeSV-Preview"). That has been removed
# and replaced with a blank placeholder below — see the stage-3 cleanup pass
# for the full writeup of this.
# It also called a script named "unified.py", which isn't present in this
# repo; based on its arguments it corresponds to feature_fusion_run_experiments.py,
# which is what run_fusion (and FUSION_SCRIPT) now points to.

ROOT=""   # TODO: set to your dataset root

CROCOV2_PATH="$ROOT/croco_features/encoder"
MAST3R_PATH="$ROOT/master_features/encoder"
DUST3R_PATH="$ROOT/duster_features/encoder"

check_paths "$CROCOV2_PATH" "$MAST3R_PATH" "$DUST3R_PATH"

run_fusion "duster_encoder" "$DUST3R_PATH" "type"
run_fusion "duster_encoder" "$DUST3R_PATH" "make"
run_fusion "duster_encoder" "$DUST3R_PATH" "model"

run_fusion "master_encoder" "$MAST3R_PATH" "type"
run_fusion "master_encoder" "$MAST3R_PATH" "make"
run_fusion "master_encoder" "$MAST3R_PATH" "model"

run_fusion "crocov2_encoder" "$CROCOV2_PATH" "type"
run_fusion "crocov2_encoder" "$CROCOV2_PATH" "make"
run_fusion "crocov2_encoder" "$CROCOV2_PATH" "model"
