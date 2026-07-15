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

"""Modality config for MotionTrans human + robot co-training (bimanual, EEF actions).

Both the human (egocentric VR) and robot (static camera) datasets are converted with
scripts/convert_motiontrans_to_gr00t.py and registered under the SAME embodiment tag, so
they share the action head — that shared representation is what enables human -> robot
transfer. The human dataset additionally carries a `camera_pose` state group
(observation.camera0_pose), which triggers reprojection of the EEF state/action into the
current camera frame at sampling time (see gr00t/data/state_action/camera_projection.py);
the robot dataset omits the group and is loaded unchanged.

For a single-arm dataset, replace the right_/left_ groups with the un-prefixed
"eef_9d" / "gripper" groups emitted by the converter.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


motiontrans_config = {
    # Video: single egocentric / front camera; key must match "video" in meta/modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["camera0"],
    ),
    # State: absolute EEF pose (pos + rot6d) and gripper per arm, robot0 = right first.
    # camera_pose_key marks the group holding the moving-camera pose (human data only);
    # it is used to reproject the EEF groups into the current camera frame and is never
    # fed to the model.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["right_eef_9d", "right_gripper", "left_eef_9d", "left_gripper"],
        camera_pose_key="camera_pose",
    ),
    # Action: 16-step chunk. EEF groups use the RELATIVE representation
    # (inv(T_state) @ T_action) — frame-invariant, which is what makes egocentric human
    # data and static-camera robot data trainable in the same action space.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["right_eef_9d", "right_gripper", "left_eef_9d", "left_gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task string resolved from task_index via meta/tasks.jsonl
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
    # Object keypoints: aux supervision targets for the object-centric keypoint head
    # (max_keypoint_objects objects x keypoints_per_object tracked points, flat/fixed
    # identity per episode — test_keypoint_tracking_simple.py). Same 16-step window as
    # the action chunk, so keypoint step t aligns with action token t. "keypoint_valid"
    # is REQUIRED here alongside "keypoint_2d": without it, the loader/processor never
    # see a group matching the "active"/"valid" flag heuristic, has_keypoint stays 0 for
    # every sample, and keypoint_loss silently trains as a constant 0.0 forever (no
    # error - just no signal). Never fed to the model as input; datasets without a
    # "keypoint" section in modality.json skip this (has_keypoint=0 for them too).
    "keypoint": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["keypoint_2d", "keypoint_valid"],
    ),
}

register_modality_config(motiontrans_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
