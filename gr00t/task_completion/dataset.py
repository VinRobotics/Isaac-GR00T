"""
WindowFrameTaskCompletionDataset
=================================
Loads a window of frames defined by ``delta_indices`` and passes them **all
at once** to the Eagle backbone in a single forward call.

Why a single backbone call?
----------------------------
Eagle can attend across all frames jointly, giving richer cross-frame
representations than W independent calls.  All W×V images are packed into
one conversation sequence and processed together — identical to how
``observation_indices`` works in the standard training configs.

At inference, the same window must be built by buffering the last
``len(delta_indices)`` raw observations and passing them together.

delta_indices
-------------
Controls which past timesteps are included, relative to the current step.

    delta_indices = [0]           → single frame (no window)
    delta_indices = [-4,-3,-2,-1,0] → 5 consecutive frames
    delta_indices = [-10,-5,0]    → 3 frames, stride 5

Frames outside the trajectory boundary are clamped (first frame repeated).

Dataset output
--------------
A standard GR00TTransform output dict:

    eagle_input_ids, eagle_attention_mask, eagle_pixel_values, …
    state, state_mask
    task_completion   (1,) float32 — label for the current step
    embodiment_id

Collation
---------
Use the standard ``DefaultDataCollator`` from gr00t.model.transforms.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform


def _balanced_weights(counts: np.ndarray) -> list:
    """Balanced class weights: weight[c] = n_total / (n_classes * n_c)."""
    n_total = counts.sum()
    n_classes = len(counts)
    return [
        float(n_total / (n_classes * counts[c])) if counts[c] > 0 else 1.0
        for c in range(n_classes)
    ]


class WindowFrameTaskCompletionDataset(LeRobotSingleDataset):
    """
    Thin subclass of LeRobotSingleDataset that sets up the right modality
    configs for window-frame task completion training.

    Parameters
    ----------
    dataset_path : str | Path
    video_keys : list[str]
        e.g. ["video.cam_head", "video.cam_left", "video.cam_right"]
    language_key : str
        e.g. "annotation.human.task_description"
    task_completion_key : str
        Parquet column name, e.g. "observation.tasks.done"
    delta_indices : list[int]
        Temporal offsets from the current step, e.g. [-10, -5, 0].
        All W frames are fetched and concatenated into a single video
        sequence before being passed to Eagle.
    transforms : ComposedModalityTransform
        Full transform pipeline returned by WindowTaskCompletionConfig.transform().
    embodiment_tag : str
    video_backend : str
    """

    def __init__(
        self,
        dataset_path: str | Path,
        video_keys: List[str],
        language_key: str,
        task_completion_key: str,
        delta_indices: List[int],
        transforms: Optional[ComposedModalityTransform] = None,
        embodiment_tag: str = "new_embodiment",
        video_backend: str = "torchvision_av",
    ):
        assert len(delta_indices) >= 1, "delta_indices must have at least one entry"
        self._window_delta_indices = list(delta_indices)
        self._task_completion_key = task_completion_key

        modality_configs = {
            # Video: all W frames defined by delta_indices
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=video_keys,
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=[language_key],
            ),
            # Task-completion label for the current step only
            "task_completion": ModalityConfig(
                delta_indices=[0],
                modality_keys=[task_completion_key],
            ),
        }

        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            transforms=transforms,
        )

    @property
    def window_size(self) -> int:
        return len(self._window_delta_indices)

    def count_labels(self) -> Optional[np.ndarray]:
        """Count label occurrences across all parquet files.

        Returns an int64 array of shape (3,) with counts for
        [doing=0, success=1, failure=2], or None if the column is absent.
        """
        parquet_files = sorted(self.dataset_path.glob("data/*/*.parquet"))
        if not parquet_files:
            return None

        counts = np.zeros(3, dtype=np.int64)
        found_column = False
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            if self._task_completion_key not in df.columns:
                continue
            found_column = True
            labels = np.array(df[self._task_completion_key].tolist()).ravel().astype(int)
            for c in range(3):
                counts[c] += int((labels == c).sum())

        return counts if found_column else None

    def compute_class_weight(self) -> Optional[list]:
        """Compute balanced per-class weights from all parquet files.

        Classes: 0=doing, 1=success, 2=failure.
        Uses the balanced formula: weight[c] = n_total / (n_classes * n_c).

        Returns None if the task_completion column is absent, or a list of
        three floats [w_doing, w_success, w_failure].
        """
        counts = self.count_labels()
        if counts is None or counts.sum() == 0:
            return None
        return _balanced_weights(counts)
