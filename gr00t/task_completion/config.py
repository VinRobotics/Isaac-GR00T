"""
WindowTaskCompletionConfig
===========================
Data configuration for window-frame task completion training (Option B).

All W frames defined by ``delta_indices`` are packed into a single Eagle
conversation and processed jointly in one backbone forward call.

delta_indices examples
----------------------
    [0]                   → single frame (original behaviour)
    [-4, -3, -2, -1, 0]  → 5 consecutive frames
    [-10, -5, 0]          → 3 frames, stride 5

The LAST entry must be 0 (current step) so the task-completion label
aligns with the most recent observation.
"""

from dataclasses import dataclass, field
from typing import List

from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms import GR00TTransform


@dataclass
class WindowTaskCompletionConfig:
    """
    Parameters
    ----------
    delta_indices : list[int]
        Temporal offsets from the current step (all <= 0).
        Determines which past frames are packed into one Eagle call.
    video_keys : list[str]
        Camera keys, e.g. ["video.cam_head", "video.cam_left", "video.cam_right"]
    language_key : str
    task_completion_key : str
        Parquet column name for the binary done label.
    max_state_dim / max_action_dim : int
        Must match the GR00T model's dimensions.
    """

    delta_indices: List[int] = field(default_factory=lambda: [0])

    video_keys: List[str] = field(
        default_factory=lambda: ["video.cam_head", "video.cam_left", "video.cam_right"]
    )
    language_key: str = "annotation.human.task_description"
    task_completion_key: str = "observation.tasks.done"

    max_state_dim: int = 64
    max_action_dim: int = 32

    crop_scale: float = 0.95
    image_height: int = 224
    image_width: int = 224

    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5
    hue: float = 0.08

    @property
    def window_size(self) -> int:
        return len(self.delta_indices)

    def transform(self, training: bool = True) -> ComposedModalityTransform:
        """
        Full transform pipeline for all W frames at once.

        ``training=True``  → colour jitter enabled, task_completion label included.
        ``training=False`` → deterministic crop+resize, no label.
        """
        transforms = [
            VideoToTensor(apply_to=self.video_keys[:1]),
            VideoCrop(apply_to=self.video_keys[:1], scale=0.95),
            VideoResize(apply_to=self.video_keys[:1], height=224, width=224, interpolation="linear"),
            VideoToTensor(apply_to=self.video_keys[1:]),
            VideoCrop(apply_to=self.video_keys[1:], scale=0.95),
            VideoResize(apply_to=self.video_keys[1:], height=224, width=224, interpolation="linear"),
            
        ]

        if training:
            transforms.append(
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                )
            )

        transforms += [
            VideoToNumpy(apply_to=self.video_keys),
            # Merge all camera keys → data["video"] (T×V, H, W, C)
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=None,
                action_concat_order=None,
            ),
            # Eagle tokenisation — T=len(delta_indices) frames × V cameras
            GR00TTransform(
                state_horizon=self.window_size,
                action_horizon=1,
                max_state_dim=self.max_state_dim,
                max_action_dim=self.max_action_dim,
                use_task_completion=training,   # include label only when training
                training=training,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
