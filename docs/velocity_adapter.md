# GR00T Velocity Adapter (VLASH-PD)

Fine-tune GR00T to output **PD-complete action chunks** (position + velocity) for smoother execution under VLASH-style latency and quantization.

## Overview

The velocity adapter adds a dual-head output to GR00T's action head:
- **Position head**: Original flow-matching action prediction (preserved)
- **Velocity head**: Learned velocity targets via B-spline differentiation (34.6M new parameters)

### Architecture

```
DiT Output [B, T, 1024]
       |
       +--> Position Decoder (CategorySpecificMLP) --> Position Chunk [B, 16, 32]
       |
       +--> Velocity Decoder (CategorySpecificMLP) --> Velocity Chunk [B, 16, 32]
```

The velocity decoder mirrors the position decoder architecture:
- **Layer 1**: Linear(1024 -> 1024) with ReLU, category-specific weights
- **Layer 2**: Linear(1024 -> velocity_dim), category-specific weights
- **Total**: 34,636,800 parameters (1.26% of full model)

## Training

### Two-Stage Training Protocol

| Stage | What's Trained | What's Frozen | Parameters | Recommended Steps |
|-------|----------------|---------------|------------|-------------------|
| 1 | Velocity decoder only | Backbone + Position head + DiT | 34.6M (1.26%) | 5,000-10,000 |
| 2 | Both heads + projector + DiT | Backbone | 1.1B (40%) | 2,000-5,000 |

### Stage 1: Velocity Adapter Only

```bash
python scripts/gr00t_velocity_finetune.py \
    --dataset_path /path/to/dataset \
    --data_config fourier_gr1_arms_only \
    --training_stage 1 \
    --lambda_vel 1.0 \
    --lambda_consistency 0.1 \
    --bspline_smoothing 0.0 \
    --max_steps 5000 \
    --batch_size 2 \
    --output_dir ./checkpoints/velocity_stage1 \
    --video_backend decord
```

### Stage 2: Joint Fine-tuning

```bash
python scripts/gr00t_velocity_finetune.py \
    --dataset_path /path/to/dataset \
    --data_config fourier_gr1_arms_only \
    --training_stage 2 \
    --stage1_checkpoint ./checkpoints/velocity_stage1 \
    --max_steps 2000 \
    --batch_size 2 \
    --output_dir ./checkpoints/velocity_stage2 \
    --video_backend decord
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda_vel` | 1.0 | Weight for velocity flow-matching loss |
| `--lambda_consistency` | 0.1 | Weight for position-velocity consistency loss |
| `--bspline_smoothing` | 0.0 | B-spline smoothing factor (0=interpolating, >0=smoothing) |
| `--training_stage` | 1 | Stage 1: velocity decoder only, Stage 2: joint training |
| `--stage1_checkpoint` | None | Path to stage 1 checkpoint (required for stage 2) |
| `--video_backend` | decord | Video decoder: `decord`, `torchvision_av`, or `torchcodec` |
| `--tune_llm` | False | Whether to fine-tune the LLM backbone |
| `--tune_visual` | False | Whether to fine-tune the visual backbone |

## Inference Output

```python
from gr00t.model.gr00t_n1 import GR00T_N1_5

model = GR00T_N1_5.from_pretrained("./checkpoints/velocity_stage2")
model.eval()
model.cuda()

# Forward pass returns separate position and velocity tensors
output = model.action_head.get_action(backbone_output, action_input)

pos_targets = output["action_pred"]      # [B, horizon, action_dim]
vel_targets = output["velocity_pred"]    # [B, horizon, velocity_dim]
```

## PD Controller Integration

```python
# Direct PD control with predicted targets
for t in range(horizon):
    pos_target = pos_targets[:, t]  # Target position
    vel_target = vel_targets[:, t]  # Target velocity (feedforward)
    
    # PD control law with velocity feedforward
    tau = Kp * (pos_target - pos_current) + Kd * (vel_target - vel_current)
    
    # Apply torque
    robot.apply_torque(tau)
```

## Loss Components

The dual-head training uses three loss terms:

1. **L_pos**: Position flow-matching loss (original GR00T loss)
   ```
   L_pos = MSE(pred_velocity, target_velocity) * action_mask
   ```

2. **L_vel**: Velocity flow-matching loss on B-spline-derived targets
   ```
   L_vel = MSE(pred_vel, velocity_target) * velocity_mask
   ```

3. **L_consistency**: Position-velocity consistency regularization
   ```
   L_consistency = MSE(denoised_vel, finite_diff(denoised_pos)) * velocity_mask
   ```

**Total Loss:**
```
L = L_pos + lambda_vel * L_vel + lambda_consistency * L_consistency
```

## Evaluation

Run the velocity adapter evaluation script:

```bash
python scripts/eval_velocity_adapter.py \
    --model_path ./checkpoints/velocity_stage2 \
    --dataset_path demo_data/robot_sim.PickNPlace \
    --data_config fourier_gr1_arms_only \
    --embodiment_tag new_embodiment \
    --num_trajectories 5 \
    --video_backend decord
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Jerk | Trajectory smoothness (lower = smoother) |
| Position Range | Dynamic range of position targets |
| Velocity Range | Dynamic range of velocity targets |
| B-spline vs FD MSE | Quality of B-spline velocity approximation |
| Velocity Mean/Std | Statistics of velocity predictions |

## Hardware Compatibility

### RTX 5090 / Blackwell (sm_120)

The implementation includes automatic fallback to PyTorch's SDPA when flash_attn is unavailable:

```python
# Automatic detection in modeling_eagle2_5_vl.py
if flash_attn version == "0.0.0" or not available:
    attn_implementation = "sdpa"  # PyTorch native
else:
    attn_implementation = "flash_attention_2"
```

For RTX 5090 users, install PyTorch nightly:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Files

### Created
| File | Description |
|------|-------------|
| `scripts/gr00t_velocity_finetune.py` | Two-stage training script |
| `scripts/eval_velocity_adapter.py` | Evaluation script with smoothness metrics |
| `gr00t/data/transform/velocity.py` | B-spline velocity computation transforms |
| `docs/velocity_adapter.md` | This documentation |

### Modified
| File | Changes |
|------|---------|
| `gr00t/model/action_head/flow_matching_action_head.py` | Added `velocity_decoder`, dual-head loss, velocity inference |
| `gr00t/model/transforms.py` | Added `use_velocity` support in `GR00TTransform` |
| `gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py` | SDPA fallback for Blackwell GPUs |
| `gr00t/model/backbone/eagle2_hg_model/radio_model.py` | SDPA fallback for vision backbone |

