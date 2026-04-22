"""
Test Task Completion
====================
Evaluates the task-completion detector on a LeRobot dataset by loading video
frames through WindowFrameTaskCompletionDataset and running inference directly
with WindowTaskCompletionModel (no server required).

Usage:
    python scripts/test_task_completion.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gr00t.model.transforms import DefaultDataCollator
from gr00t.task_completion.config import WindowTaskCompletionConfig
from gr00t.task_completion.dataset import WindowFrameTaskCompletionDataset
from gr00t.task_completion.model import WindowTaskCompletionModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH = "/mnt/data/sftp/data/locht1/lerobot_vrh31_place_success_fail_4"
GROOT_MODEL_PATH = "/mnt/data/sftp/data/locht1/GR00T-N1.5-3B"
DETECTOR_WEIGHTS = "/mnt/data/sftp/data/locht1/task_completion_detection.pt"

# Temporal window: list of offsets relative to current step (last entry must be 0).
# Examples:
#   [0]             → single frame
#   [-4, -3, -2, -1, 0] → 5 consecutive frames
#   [-10, -5, 0]    → 3 frames with stride 5
DELTA_INDICES = list(reversed([-i * 10 for i in range(5)]))

VIDEO_KEYS = ["video.cam_head", "video.cam_left"]
LANGUAGE_KEY = "annotation.human.task_description"
TASK_COMPLETION_KEY = "observation.tasks.label"

OUTPUT_DIR = "/mnt/data/sftp/data/locht1/test_task_completion"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Dataset
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
    transforms=cfg.transform(training=False),
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

collator = DefaultDataCollator()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collator,
)

model = WindowTaskCompletionModel.from_groot_pretrained(
    GROOT_MODEL_PATH,
    seq_dim=1536,
    hidden_dim=1024,
    num_frames=cfg.window_size,
    num_cameras=len(cfg.video_keys),
)
model.load_detector_weights(DETECTOR_WEIGHTS)
model.eval().to(DEVICE)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

correct = 0
total = 0

for batch in tqdm(loader):
    labels_raw = batch.pop("task_completion")  # (B, 1)
    if isinstance(labels_raw, torch.Tensor):
        labels = labels_raw.flatten().long().tolist()
    else:
        labels = np.asarray(labels_raw).flatten().tolist()

    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        out = model(batch)

    pred_probs = out["task_completion_pred"].cpu().numpy()  # (B, 3)

    for j, (pred_prob, label) in enumerate(zip(pred_probs, labels)):
        pred = int(np.argmax(pred_prob))
        label = int(label)
        is_correct = pred == label
        correct += is_correct
        total += 1

        idx = total - 1
        print(f"[{idx:04d}] prob={pred_prob}  pred={pred}  label={label}  correct={is_correct}  acc={correct/total:.3f}")

print(f"\nFinal accuracy: {correct}/{total} = {correct/total:.4f}")
