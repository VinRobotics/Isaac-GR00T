# Changes for VR-H3, A2

This folder contains scripts for fine-tuning and evaluating GR00T policies on **VR-H3** with support for **RTC** and **VLASH Action Quantization**.

---

## Real-time Action Chunking (RTC)

RTC enables smooth, real-time action generation by conditioning on previously executed actions. There are **two variants**:

### 1. Training-free RTC (`--smooth_option rtc`)

- **No training required** – works with any pretrained checkpoint
- Uses classifier-free guidance to attend to action prefix at inference time
- Configured via `prefix_attention_schedule` and `max_guidance_weight`

**Evaluation:**
```bash
python scripts/eval_policy_w_rtc.py \
    --model_path /path/to/checkpoint \
    --data_config vrh3_two_hand \
    --embodiment_tag vrh3 \
    --prefix_attention_schedule linear \
    --max_guidance_weight 5.0
```

### 2. Training-time RTC (`--use_action_conditioning`)

- **Requires fine-tuning** with action conditioning enabled
- Conditions on action prefix during training with random delays
- More robust temporal consistency

**Fine-tuning:**
```bash
python scripts/gr00t_finetune.py \
    --dataset_path /path/to/vrh3_dataset \
    --output_dir /path/to/output \
    --data_config vrh3_two_hand \
    --embodiment_tag vrh3 \
    --use_action_conditioning
```

**Evaluation:**
```bash
python scripts/eval_policy_w_ttrtc.py \
    --model_path /path/to/checkpoint \
    --data_config vrh3_two_hand \
    --embodiment_tag vrh3 \
    --action_horizon 16
```

---

## VLASH Action Quantization

VLASH-style quantization discretizes continuous actions into bins, improving learning efficiency. This implementation supports **higher-order derivatives** (velocity, acceleration).

### Available VR-H3 Quantized Configs

| Config Name | Description |
|-------------|-------------|
| `vrh3_two_hand` | Standard continuous actions |
| `vrh3_two_hand_quantized` | Position-based quantization (256 bins) |
| `vrh3_two_hand_velocity_quantized` | Velocity-based quantization |
| `vrh3_two_hand_higher_order` | Higher-order derivatives (position + velocity + acceleration) |

### Fine-tuning with VLASH

```bash
# Position quantization
python scripts/gr00t_finetune.py \
    --dataset_path /path/to/vrh3_dataset \
    --output_dir /path/to/output \
    --data_config vrh3_two_hand_quantized \
    --embodiment_tag vrh3

# Velocity quantization
python scripts/gr00t_finetune.py \
    --dataset_path /path/to/vrh3_dataset \
    --output_dir /path/to/output \
    --data_config vrh3_two_hand_velocity_quantized \
    --embodiment_tag vrh3

# Higher-order derivatives
python scripts/gr00t_finetune.py \
    --dataset_path /path/to/vrh3_dataset \
    --output_dir /path/to/output \
    --data_config vrh3_two_hand_higher_order \
    --embodiment_tag vrh3
```

### Usage with Evaluation

Both RTC variants support VLASH configs:

```bash
# Training-free RTC + VLASH quantization
python scripts/eval_policy_w_rtc.py \
    --model_path /path/to/checkpoint \
    --data_config vrh3_two_hand_quantized \
    --embodiment_tag vrh3

# Training-time RTC + VLASH velocity quantization
python scripts/eval_policy_w_ttrtc.py \
    --model_path /path/to/checkpoint \
    --data_config vrh3_two_hand_velocity_quantized
```

### Custom Quantization Configs

You can create custom configs using the quantization transforms in `gr00t/data/transform/action_quantization.py`:

```python
from gr00t.data.transform import (
    ActionQuantizationTransform,
    VelocityQuantizationTransform,
    AccelerationQuantizationTransform,
    HigherOrderDerivativeTransform,
)

# Position quantization with 512 bins
position_quantizer = ActionQuantizationTransform(
    action_keys=["action.left_arm", "action.right_arm"],
    num_bins=512,
    action_min=-1.0,
    action_max=1.0,
)

# Velocity quantization with custom dt
velocity_quantizer = VelocityQuantizationTransform(
    action_keys=["action.left_arm", "action.right_arm"],
    num_bins=256,
    dt=0.02,  # 50 Hz control frequency
)

# Full higher-order derivatives
higher_order = HigherOrderDerivativeTransform(
    action_keys=["action.left_arm", "action.right_arm"],
    orders=[1, 2],  # velocity and acceleration
    dt=0.02,
)
```

---

## Combining RTC and VLASH

Both techniques can be used together:

```bash
# Fine-tune with training-time RTC + VLASH
python scripts/gr00t_finetune.py \
    --dataset_path /path/to/vrh3_dataset \
    --output_dir /path/to/output \
    --data_config vrh3_two_hand_quantized \
    --embodiment_tag vrh3 \
    --use_action_conditioning

# Evaluate
python scripts/eval_policy_w_ttrtc.py \
    --model_path /path/to/checkpoint \
    --data_config vrh3_two_hand_quantized \
    --embodiment_tag vrh3
```
---

## Technical Details

### Training-time RTC Implementation

Located in `gr00t/model/action_head/flow_matching_action_head_action_condition.py`:

1. During training, samples a random delay `d ~ Unif[0, H//2)` where H is action horizon
2. Conditions on action prefix `a[0:d]` with `t=1.0` (fully denoised)
3. Predicts postfix `a[d:H]` with sampled `t ~ Unif[0,1]`
4. Loss computed only on postfix predictions

### VLASH Quantization Implementation

Located in `gr00t/data/transform/action_quantization.py`:

1. **ActionQuantizer**: Uniform or adaptive binning with invertible quantize/dequantize
2. **ActionQuantizationTransform**: Direct position discretization
3. **VelocityQuantizationTransform**: Computes velocity via finite differences, then quantizes
4. **AccelerationQuantizationTransform**: Computes acceleration, then quantizes
5. **HigherOrderDerivativeTransform**: Computes arbitrary order derivatives

All transforms are invertible and integrate with the existing normalization pipeline.
