# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rendering for the object-centric keypoint auxiliary head (debug/eval only).

Draws the 16-step future keypoint trajectory for each object slot onto the
current-frame thumbnail, so a human can check whether the model understands
object motion. Never used on the action path.
"""

import cv2
import numpy as np

_OBJECT_COLORS = [(220, 30, 30), (30, 110, 220)]  # per-slot RGB, red / blue


def render_keypoint_overlay(
    image: np.ndarray,
    keypoints: np.ndarray,
    weight: np.ndarray,
    radius: int = 2,
    min_brightness: float = 0.35,
) -> np.ndarray:
    """Overlay a keypoint trajectory onto a copy of `image`.

    Args:
        image: (H, W, 3) uint8 RGB thumbnail.
        keypoints: (horizon, num_objects, num_points, 2) in [-1, 1], normalized the
            same way for any image size (independent axis rescaling), so this works
            regardless of what size `image` was resized to.
        weight: (horizon, num_objects) in [0, 1] — GT active flag or predicted
            probability. Modulates point color so inactive/low-confidence steps fade
            out instead of cluttering the frame; steps with weight < 0.05 are skipped.

    Returns:
        (H, W, 3) uint8 RGB copy of `image` with the trajectory drawn on top.
    """
    out = np.ascontiguousarray(image).copy()
    h, w = out.shape[:2]
    horizon, num_objects = keypoints.shape[:2]
    for obj in range(num_objects):
        base_color = np.array(_OBJECT_COLORS[obj % len(_OBJECT_COLORS)], dtype=np.float32)
        for t in range(horizon):
            w_t = float(weight[t, obj])
            if w_t < 0.05:
                continue
            # Fade early steps in, later steps fully saturated, so the direction of
            # motion is visible at a glance.
            brightness = min_brightness + (1.0 - min_brightness) * (t / max(horizon - 1, 1))
            color = tuple(int(c) for c in (base_color * brightness * w_t).clip(0, 255))
            for x_norm, y_norm in keypoints[t, obj]:
                px = int((x_norm + 1.0) * 0.5 * w)
                py = int((y_norm + 1.0) * 0.5 * h)
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(out, (px, py), radius, color, thickness=-1)
    return out


def combine_gt_pred(gt_image: np.ndarray, pred_image: np.ndarray, gap: int = 6) -> np.ndarray:
    """Concatenate a GT overlay (left) and predicted overlay (right) into one image.

    Logged as a single wandb.Image so a single panel gives both the built-in list
    index slider (to page through sample pairs) and the run's step slider (to page
    through eval calls) — GT and pred always shown side by side for the same sample.
    """
    h = gt_image.shape[0]
    divider = np.full((h, gap, 3), 255, dtype=np.uint8)
    combined = np.concatenate([gt_image, divider, pred_image], axis=1)
    cv2.putText(
        combined, "GT", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA
    )
    pred_x = gt_image.shape[1] + gap + 4
    cv2.putText(
        combined,
        "Pred",
        (pred_x, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return combined
