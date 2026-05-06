#!/bin/bash
# Run preprocessing capture for gr00t_baseline branch (gr00t_baseline_locht1)
# Execute this from the repo root: bash evaluation/run_check_gr00t_baseline.sh
set -e
source /home/locht1/miniconda3/bin/activate gr00t_mimicgen
python evaluation/check_env_preprocessing.py \
    --args.branch-name=gr00t_baseline \
    --args.gr00t-path=/home/locht1/gr00t_equi_dit \
    --args.libero-path=/mnt/data/sftp/data/locht1/LIBERO_benchmark \
    --args.mimicgen-path=/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen \
    --args.seed=7 --args.skip_libero

source /home/locht1/miniconda3/bin/activate gr00t
python evaluation/check_env_preprocessing.py \
    --args.branch-name=gr00t_baseline \
    --args.gr00t-path=/home/locht1/gr00t_equi_dit \
    --args.libero-path=/mnt/data/sftp/data/locht1/LIBERO_benchmark \
    --args.mimicgen-path=/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen \
    --args.seed=7 --args.skip_mimicgen
