#!/bin/bash
# Run preprocessing capture for equi_gr00t_dit branch (equi_gr00t_FA_exp_wo_FA)
# Execute this from the repo root: bash evaluation/run_check_equi_gr00t_dit.sh
set -e

python evaluation/check_env_preprocessing.py \
    --args.branch-name=equi_gr00t_dit \
    --args.gr00t-path=/home/locht1/gr00t_equi_fa \
    --args.libero-path=/mnt/data/sftp/data/locht1/LIBERO_benchmark \
    --args.mimicgen-path=/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen \
    --args.robosuite-assets=/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen/robosuite/models/assets \
    --args.seed=7 --args.skip_libero

python evaluation/check_env_preprocessing.py \
    --args.branch-name=equi_gr00t_dit \
    --args.gr00t-path=/home/locht1/gr00t_equi_fa \
    --args.libero-path=/mnt/data/sftp/data/locht1/LIBERO_benchmark \
    --args.mimicgen-path=/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen \
    --args.robosuite-assets=/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen/robosuite/models/assets \
    --args.seed=7 --args.skip_mimicgen


