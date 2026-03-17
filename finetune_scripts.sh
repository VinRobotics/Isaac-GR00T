export NUM_GPUS=2
export WANDB_PROJECT=vrh3_sim_pick_place
export CUDA_VISIBLE_DEVICES=0,1
HF_HUB_OFFLINE=0

torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path \
    /mnt/data/sftp/data/locht1/vr_data/VR_H31_bodyshop_place_part2_base_poseA_stereo_rvt_trym_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/VR_H31_bodyshop_place_part2_pose0_linhld23_stereo_rvt_trym_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/VR_H31_bodyshop_place_part2_pose3_stereo_rvt_trym_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/VR_H31_bodyshop_place_part2_pose1_stereo_rvt_trym_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/VR_H31_bodyshop_place_part2_pose2_stereo_rvt_trym_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/VR_H31_bodyshop_place_part2_pose4_stereo_rvt_trym_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/VR_H31_bodyshop_place_part2_pose5_stereo_rvt_trym_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/VR_H31_bodyshop_place_part2_pose6_stereo_rvt_trym_speedup1 \
    /mnt/data/sftp/data/locht1/vr_data/VR_H31_bodyshop_place_part2_pose7a_stereo_rvt_trym_speedup1 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /home/locht1/gr00t/data_config/vrh3_two_hands.py \
    --num-gpus $NUM_GPUS \
    --output-dir /mnt/data/sftp/data/locht1/vr_checkpoints/gr00t_n16_vrh31_two_hands_chunk50_3random_pose_1703 \
    --save-total-limit 5 \
    --save-steps 10000 \
    --max-steps 50000 \
    --use-wandb \
    --global-batch-size 64 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4