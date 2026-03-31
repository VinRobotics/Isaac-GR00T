"""
Test Task Completion
====================
Evaluates the task-completion detector on a LeRobot dataset by loading video
frames through WindowFrameTaskCompletionDataset and calling
RobotInferenceClient.get_task_completion() for each sample.

Usage (server must already be running with task_completion weights):
    python scripts/test_task_completion.py
"""

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from gr00t.eval.robot import RobotInferenceClient
from gr00t.task_completion.config import WindowTaskCompletionConfig
from gr00t.task_completion.dataset import WindowFrameTaskCompletionDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH = "/home/locht1/Documents/locht1/code_convert/output/vr_h31_place_completion_test"

# Temporal window: list of offsets relative to current step (last entry must be 0).
# Examples:
#   [0]             → single frame
#   [-4, -3, -2, -1, 0] → 5 consecutive frames
#   [-10, -5, 0]    → 3 frames with stride 5
DELTA_INDICES = list(reversed([-i * 10 for i in range(10)]))

VIDEO_KEYS = ["video.cam_head", "video.cam_left", "video.cam_right"]
LANGUAGE_KEY = "annotation.human.task_description"
TASK_COMPLETION_KEY = "observation.tasks.done"

OUTPUT_DIR = os.path.basename(DATASET_PATH)
THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Dataset  (no transforms — the server applies its own)
# ---------------------------------------------------------------------------

cfg = WindowTaskCompletionConfig(
    delta_indices=DELTA_INDICES,
    video_keys=VIDEO_KEYS,
    language_key=LANGUAGE_KEY,
    task_completion_key=TASK_COMPLETION_KEY,
)

dataset = WindowFrameTaskCompletionDataset(
    dataset_path=DATASET_PATH,
    video_keys=cfg.video_keys,
    language_key=cfg.language_key,
    task_completion_key=cfg.task_completion_key,
    delta_indices=cfg.delta_indices,
    transforms=None,  # server applies transforms
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

client = RobotInferenceClient()

correct = 0
total = 0

for i, data in enumerate(tqdm(dataset)):
    # Build raw observation (drop label before sending to server)
    obs = {k: v for k, v in data.items() if TASK_COMPLETION_KEY not in k}

    result = client.get_task_completion(obs)
    pred_prob = float(result["task_completion_pred"])
    pred = int(pred_prob >= THRESHOLD)

    label_raw = data[TASK_COMPLETION_KEY]
    if isinstance(label_raw, torch.Tensor):
        label = int(label_raw.flatten()[0].item())
    else:
        label = int(np.asarray(label_raw).flatten()[0])

    is_correct = pred == label
    correct += is_correct
    total += 1

    # Save the first camera frame annotated with pred / label
    img_tensor = data[VIDEO_KEYS[0]]
    if isinstance(img_tensor, torch.Tensor):
        img_arr = img_tensor.cpu().numpy()
    else:
        img_arr = np.array(img_tensor)

    # Take last frame if window > 1: shape (T, H, W, C) or (H, W, C)
    if img_arr.ndim == 4:
        img_arr = img_arr[-1]

    # (C, H, W) → (H, W, C)
    if img_arr.ndim == 3 and img_arr.shape[0] == 3:
        img_arr = np.transpose(img_arr, (1, 2, 0))

    if img_arr.max() <= 1.0:
        img_arr = (img_arr * 255).astype(np.uint8)
    else:
        img_arr = img_arr.astype(np.uint8)

    img = Image.fromarray(img_arr)
    fname = f"img_{i:04d}_pred{pred}_{pred_prob:.2f}_label{label}.png"
    img.save(os.path.join(OUTPUT_DIR, fname))

    print(f"[{i:04d}] prob={pred_prob:.3f}  pred={pred}  label={label}  correct={is_correct}  acc={correct/total:.3f}")

print(f"\nFinal accuracy: {correct}/{total} = {correct/total:.4f}")
