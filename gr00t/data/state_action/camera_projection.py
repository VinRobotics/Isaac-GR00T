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

"""
Egocentric (moving-camera) reprojection of EEF states and actions.

Human demonstration datasets recorded with a head-mounted camera store EEF poses in the
frame of the camera at *episode start*, while the camera itself moves during the episode.
The image at timestep ``t`` and the stored pose at timestep ``t`` are therefore expressed
in different frames. Given the camera pose at ``t`` (relative to episode start), this
module left-multiplies all EEF poses by ``cam_proj(t) = inv(T_cam(t))`` so that state and
action are expressed in the frame of the camera that produced the image at ``t``.

Because GR00T's relative EEF action is ``inv(T_state) @ T_action`` (frame-invariant), the
same ``cam_proj`` must be applied to both the state and the raw action chunk: the
projection then cancels exactly in the relative representation (so precomputed
``relative_stats.json`` stay valid), while the absolute state fed to the model becomes
consistent with the current image. Datasets with a static camera simply omit the camera
pose column and are left untouched.

Rotation conventions match ``gr00t.data.state_action.pose.EndEffectorPose``:
rot6d = first two ROWS of the rotation matrix, flattened.
"""

from gr00t.data.types import ActionFormat, ActionType, ModalityConfig
import numpy as np
from scipy.spatial.transform import Rotation


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert batched 6D rotations to rotation matrices.

    Vectorized equivalent of ``EndEffectorPose._rot6d_to_matrix`` (rows convention,
    Gram-Schmidt orthonormalization).

    Args:
        rot6d: (..., 6) array — first two rows of the rotation matrix, flattened.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    rows = rot6d.reshape(*rot6d.shape[:-1], 2, 3)
    row1 = rows[..., 0, :]
    row2 = rows[..., 1, :]
    row1 = row1 / np.linalg.norm(row1, axis=-1, keepdims=True)
    row2 = row2 - np.sum(row1 * row2, axis=-1, keepdims=True) * row1
    row2 = row2 / np.linalg.norm(row2, axis=-1, keepdims=True)
    row3 = np.cross(row1, row2)
    return np.stack([row1, row2, row3], axis=-2)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Convert batched rotation matrices to 6D rotations (first two rows, flattened).

    Args:
        matrix: (..., 3, 3) rotation matrices.

    Returns:
        (..., 6) rot6d array.
    """
    matrix = np.asarray(matrix)
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def pose_array_to_homogeneous(pose: np.ndarray, action_format: ActionFormat) -> np.ndarray:
    """Convert a batch of flat EEF poses to homogeneous matrices.

    Args:
        pose: (N, D) array; D = 9 for XYZ_ROT6D, 6 for XYZ_ROTVEC.
        action_format: Layout of each row.

    Returns:
        (N, 4, 4) homogeneous transformation matrices.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if action_format == ActionFormat.XYZ_ROT6D:
        rot_mats = rot6d_to_matrix(pose[..., 3:9])
    elif action_format == ActionFormat.XYZ_ROTVEC:
        rot_mats = Rotation.from_rotvec(pose[..., 3:6].reshape(-1, 3)).as_matrix()
        rot_mats = rot_mats.reshape(*pose.shape[:-1], 3, 3)
    else:
        raise ValueError(f"Unsupported action format for camera projection: {action_format}")
    mats = np.zeros((*pose.shape[:-1], 4, 4), dtype=np.float64)
    mats[..., :3, :3] = rot_mats
    mats[..., :3, 3] = pose[..., :3]
    mats[..., 3, 3] = 1.0
    return mats


def homogeneous_to_pose_array(mats: np.ndarray, action_format: ActionFormat) -> np.ndarray:
    """Inverse of :func:`pose_array_to_homogeneous`.

    Args:
        mats: (N, 4, 4) homogeneous matrices.
        action_format: Desired flat layout.

    Returns:
        (N, D) flat pose array.
    """
    mats = np.asarray(mats)
    translation = mats[..., :3, 3]
    if action_format == ActionFormat.XYZ_ROT6D:
        rot = matrix_to_rot6d(mats[..., :3, :3])
    elif action_format == ActionFormat.XYZ_ROTVEC:
        rot = Rotation.from_matrix(mats[..., :3, :3].reshape(-1, 3, 3)).as_rotvec()
        rot = rot.reshape(*mats.shape[:-2], 3)
    else:
        raise ValueError(f"Unsupported action format for camera projection: {action_format}")
    return np.concatenate([translation, rot], axis=-1)


def camera_projection_from_pose(camera_pose: np.ndarray) -> np.ndarray:
    """Build the reprojection matrix from a camera pose row.

    Args:
        camera_pose: (6,) array ``[x, y, z, rx, ry, rz]`` (pos + rotvec) — pose of the
            camera at the current timestep, relative to the camera pose at episode start
            (identity when the camera has not moved).

    Returns:
        (4, 4) matrix ``cam_proj = inv(T_cam)`` mapping episode-start-frame poses into
        the current camera frame.
    """
    camera_pose = np.asarray(camera_pose, dtype=np.float64).reshape(6)
    cam_mat = np.eye(4)
    cam_mat[:3, :3] = Rotation.from_rotvec(camera_pose[3:]).as_matrix()
    cam_mat[:3, 3] = camera_pose[:3]
    return np.linalg.inv(cam_mat)


def project_poses(
    pose: np.ndarray, cam_proj: np.ndarray, action_format: ActionFormat
) -> np.ndarray:
    """Left-multiply a batch of flat EEF poses by ``cam_proj``, preserving layout/dtype shape.

    Args:
        pose: (N, D) flat pose array.
        cam_proj: (4, 4) projection matrix.
        action_format: Layout of each pose row.

    Returns:
        (N, D) projected poses (float32).
    """
    mats = pose_array_to_homogeneous(pose, action_format)
    projected = cam_proj @ mats
    return homogeneous_to_pose_array(projected, action_format).astype(np.float32)


def apply_camera_projection(
    state_data: dict[str, np.ndarray],
    action_data: dict[str, np.ndarray],
    camera_pose: np.ndarray,
    modality_configs: dict[str, ModalityConfig],
) -> None:
    """Reproject all EEF state/action groups into the current camera frame, in place.

    Groups to transform are discovered from the action ModalityConfig: every action group
    whose ``ActionConfig.type == ActionType.EEF`` is projected, together with its
    reference state group (``state_key``, defaulting to the same name). Non-EEF groups
    (grippers, joints) are left untouched.

    Args:
        state_data: Mapping group -> (T_state, D) raw state arrays (modified in place).
        action_data: Mapping group -> (T_horizon, D) raw action arrays (modified in place).
        camera_pose: (6,) camera pose (pos + rotvec) at the sampled timestep, relative to
            episode start.
        modality_configs: The embodiment's modality configs (needs "action" with
            ``action_configs``).
    """
    action_config = modality_configs.get("action")
    if action_config is None or action_config.action_configs is None:
        return

    cam_proj = camera_projection_from_pose(camera_pose)

    projected_state_keys = set()
    for key, cfg in zip(action_config.modality_keys, action_config.action_configs):
        if cfg.type != ActionType.EEF:
            continue
        if key in action_data:
            action_data[key] = project_poses(action_data[key], cam_proj, cfg.format)
        state_key = cfg.state_key or key
        if state_key in state_data and state_key not in projected_state_keys:
            state_data[state_key] = project_poses(state_data[state_key], cam_proj, cfg.format)
            projected_state_keys.add(state_key)
