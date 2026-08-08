#!/bin/bash
# Compute empirical equivariance error for equi_gr00t_FA_equillm_c4 checkpoints,
# broken out PER CYCLIC GROUP ORDER (C4, C16, ...).
#
# Why per-group, not one shared run:
#   The model's cyclic group order (n_group) is an ARCHITECTURAL property baked
#   into the checkpoint at fine-tune time — it comes from the data config's
#   get_rotation_config() (defaults to 4 unless overridden, e.g. via
#   `"n_group": 16` — see gr00t/experiment/data_config.py) and is persisted into
#   backbone_cfg["n_group"] / action_head_cfg["n_group"]. escnn builds the
#   regular representation at that size, so a C4 checkpoint and a C16 checkpoint
#   have DIFFERENT weight shapes — they are separate checkpoints, not one
#   checkpoint you can re-measure at a different group order.
#
#   A C4-equivariant architecture is only exactly equivariant to rotations that
#   are multiples of 90°. Measuring it with --args.n_group 16 (22.5° steps)
#   would just report how it degrades off those angles, not a fair number to
#   put next to a C16 model's own epsilon_eq. So: measure each checkpoint with
#   --args.n_group set to the SAME group order it was trained with.
#
# Checkpoint requirement:
#   Yes, compute_equivariance_error.py needs a checkpoint DIRECTORY to load —
#   it builds the policy via Gr00tn15_inference(pretrained_model_path), which
#   reads config.json (embodiment/action-head/backbone shapes, incl. n_group)
#   from that path and instantiates from it. There is no from-scratch/self-init
#   path. If you only want to check ARCHITECTURAL equivariance (independent of
#   how well training went), you don't need a fully-trained checkpoint — pass
#   RANDOM_INIT=true below and any checkpoint directory with the right
#   architecture (e.g. an early/step-0 save) will do: the script loads it, then
#   re-initialises every module's weights via reset_parameters() (same seed,
#   same architecture) before measuring. Equivariance is structural, so a
#   correctly-equivariant architecture should give epsilon_eq ~= 0 even then;
#   if it doesn't, the architecture itself isn't end-to-end equivariant.
#
# Edit the checkpoint paths below before running.
# Results are written to $SAVE_DIR/log/$EXP_NAME/<task_suite>_<model_label>.{log,json}

set -e

TASK_SUITE=${TASK_SUITE:-libero_10}
NUM_SAMPLES=${NUM_SAMPLES:-500}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
SAVE_DIR=${SAVE_DIR:-/mnt/data/sftp/data/locht1/equi_eq_error}
EXP_NAME=${EXP_NAME:-c4_c16_table}
RANDOM_INIT=${RANDOM_INIT:-false}
# set RANDOM_INIT=true to measure architecture only (see header) — no trained weights needed

# ---- checkpoint paths, one per cyclic group order (fill these in) -----------
# Each must be a checkpoint that was actually FINE-TUNED with that group order
# (i.e. get_rotation_config() returned "n_group": <N> for it) — they are not
# interchangeable across group orders.
C4_CKPT=${C4_CKPT:-""}
C16_CKPT=${C16_CKPT:-""}
# -----------------------------------------------------------------------------

source /home/locht1/miniconda3/bin/activate gr00t_equi_fa_c4

run_one() {
    local label="$1"
    local ckpt="$2"
    local n_group="$3"
    if [[ -z "$ckpt" ]]; then
        echo "[skip] $label — checkpoint path is empty"
        return
    fi
    echo "===================================================================="
    echo "Running $label  (C${n_group} group, --args.n_group ${n_group})"
    echo "  ckpt = $ckpt"
    echo "===================================================================="
    # tyro renders bool fields as bare flags (--args.random_init flips it True),
    # not `--flag value` — so only append it when the user opted in, rather than
    # trying to pass "$RANDOM_INIT" as a value.
    extra_args=()
    if [[ "$RANDOM_INIT" == "true" || "$RANDOM_INIT" == "1" ]]; then
        extra_args+=(--args.random_init)
    fi
    python evaluation/compute_equivariance_error.py \
        --args.pretrained_model_path "$ckpt" \
        --args.model_label "$label" \
        --args.task_suite_name "$TASK_SUITE" \
        --args.num_samples "$NUM_SAMPLES" \
        --args.n_group "$n_group" \
        --args.num_steps_wait "$NUM_STEPS_WAIT" \
        --args.save_dir "$SAVE_DIR" \
        --args.exp_name "$EXP_NAME" \
        "${extra_args[@]}"
}

run_one "EquiVLA_C4"  "$C4_CKPT"  4
run_one "EquiVLA_C16" "$C16_CKPT" 16
