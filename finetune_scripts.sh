export NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./home/locht1/Documents/locht1/code_convert/output/20260227_VR_H31_bodyshop_place_part2_speedup1 /home/locht1/Documents/locht1/code_convert/output/20260227_VR_H31_bodyshop_place_part2_recovery_speedup1 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /home/locht1/Documents/locht1/Isaac-GR00T/data_config/vrh3_two_hands.py \
    --num-gpus $NUM_GPUS \
    --output-dir ~/vr_checkpoints/gr00t_n16_vrh3_two_hands \
    --save-total-limit 5 \
    --save-steps 5000 \
    --max-steps 50000 \
    --use-wandb \
    --global-batch-size 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 6