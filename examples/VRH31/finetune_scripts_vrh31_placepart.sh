export NUM_GPUS=2
export WANDB_PROJECT=vrh3_sim_pick_place
export CUDA_VISIBLE_DEVICES=0,1
HF_HUB_OFFLINE=0

torchrun --nproc_per_node=2 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path \
    /mnt/data/sftp/data/locht1/vr_data/20260309_VR_H31_bodyshop_place_part2_baoht9_stereo_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/20260309_VR_H31_bodyshop_place_part2_linhld23_stereo_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/20260310_VR_H31_bodyshop_place_part2_stereo_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/20260312_VR_H31_bodyshop_place_part2_stereo_rvt_p1_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/20260312_VR_H31_bodyshop_place_part2_stereo_rvt_p2_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/20260313_VR_H31_bodyshop_place_part2_correction_stereo_rvt_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/20260314_VR_H31_bodyshop_place_part2_correction_stereo_rvt_trym_speedup1 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /home/locht1/gr00t/data_config/vrh3_two_hands_effort.py \
    --effort-dim 26 \
    --effort-history-len 16 \
    --num-gpus $NUM_GPUS \
    --output-dir /mnt/data/sftp/data/locht1/vr_checkpoints/gr00t_n16_ta_vrh31_two_hands_chunk50_1403 \
    --save-total-limit 5 \
    --save-steps 10000 \
    --max-steps 80000 \
    --use-wandb \
    --global-batch-size 64 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 6