#!/bin/bash
# Compute empirical equivariance error for the three LIBERO variants.
#
# Edit the checkpoint paths below before running.
# Each run reports epsilon_eq (mean +/- std over M|G| pairs) plus a per-group breakdown.
# Results are written to $SAVE_DIR/log/$EXP_NAME/<task_suite>_<model_label>.{log,json}

set -e

TASK_SUITE=${TASK_SUITE:-libero_goal}
NUM_SAMPLES=${NUM_SAMPLES:-500}
N_GROUP=${N_GROUP:-8}
NUM_STEPS_WAIT=${NUM_STEPS_WAIT:-10}
SAVE_DIR=${SAVE_DIR:-/mnt/data/sftp/data/locht1/equi_eq_error}
EXP_NAME=${EXP_NAME:-paper_table}

# ---- checkpoint paths (fill these in) ---------------------------------------
GR00TN15_CKPT=${GR00TN15_CKPT:-""}
EQUIDIT_CKPT=${EQUIDIT_CKPT:-""}
EQUIVLA_CKPT=${EQUIVLA_CKPT:-""}
# -----------------------------------------------------------------------------

source /home/locht1/miniconda3/bin/activate gr00t_equi_fa

run_one() {
    local label="$1"
    local ckpt="$2"
    if [[ -z "$ckpt" ]]; then
        echo "[skip] $label — checkpoint path is empty"
        return
    fi
    echo "===================================================================="
    echo "Running $label"
    echo "  ckpt = $ckpt"
    echo "===================================================================="
    python evaluation/compute_equivariance_error.py \
        --pretrained_model_path "$ckpt" \
        --model_label "$label" \
        --task_suite_name "$TASK_SUITE" \
        --num_samples "$NUM_SAMPLES" \
        --n_group "$N_GROUP" \
        --num_steps_wait "$NUM_STEPS_WAIT" \
        --save_dir "$SAVE_DIR" \
        --exp_name "$EXP_NAME"
}

run_one "EquiVLA"   "$EQUIVLA_CKPT"
