#!/bin/bash
# Run preprocessing capture for equi_gr00t_fa branch
# Execute this from the repo root: bash evaluation/run_check_equi_gr00t_fa.sh
set -e

python evaluation/check_env_preprocessing.py \
    --branch-name equi_gr00t_fa \
    --gr00t-path /home/locht1/gr00t_equi_fa \
    --libero-path /mnt/data/sftp/data/locht1/LIBERO_benchmark \
    --mimicgen-path /mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen \
    --robosuite-assets /mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen/robosuite/models/assets \
    --seed 7
