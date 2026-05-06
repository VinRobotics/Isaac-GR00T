#!/bin/bash
# Run preprocessing capture for gr00t_baseline branch (gr00t_baseline_locht1)
# Execute this from the repo root: bash evaluation/run_check_gr00t_baseline.sh
set -e

python evaluation/check_env_preprocessing.py \
    --args.branch-name=gr00t_baseline \
    --args.gr00t-path=/home/locht1/gr00t_equi_dit \
    --args.libero-path=/mnt/data/sftp/data/locht1/LIBERO_benchmark \
    --args.mimicgen-path=/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen \
    --args.robosuite-assets=/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen/robosuite/models/assets \
    --args.seed=7 --args.skip_libero

python evaluation/check_env_preprocessing.py \
    --args.branch-name=gr00t_baseline \
    --args.gr00t-path=/home/locht1/gr00t_equi_dit \
    --args.libero-path=/mnt/data/sftp/data/locht1/LIBERO_benchmark \
    --args.mimicgen-path=/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen \
    --args.robosuite-assets=/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen/robosuite/models/assets \
    --args.seed=7 --args.skip_mimicgen
