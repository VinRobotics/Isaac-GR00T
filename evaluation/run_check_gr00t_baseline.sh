#!/bin/bash
# Run preprocessing capture for gr00t_baseline branch (gr00t_baseline_locht1)
# Execute this from the repo root: bash evaluation/run_check_gr00t_baseline.sh
set -e

python evaluation/check_env_preprocessing.py \
    --branch-name gr00t_baseline \
    --gr00t-path /home/locht1/gr00t_equi_fa \
    --libero-path /mnt/data/sftp/data/locht1/LIBERO_benchmark \
    --mimicgen-path /mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen \
    --robosuite-assets /mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen/robosuite/models/assets \
    --seed 7
