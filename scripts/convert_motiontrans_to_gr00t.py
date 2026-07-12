#!/usr/bin/env python

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
Prepare a MotionTrans-exported LeRobot dataset (human VR or robot teleop) for GR00T
human/robot co-training.

Supported input schemas (see gr00t_lerobot_dataset.md in the motiontrans-pi0 repo):
    observation.state         (n_arms * (9 + g),)  [pos3, rot6d6, gripper_g] per arm —
                                                   OR absent, in which case it is
                                                   synthesized from the per-key columns
                                                   observation.robot{i}_eef_pos +
                                                   observation.robot{i}_eef_rot6d (or
                                                   _eef_rot_axis_angle) +
                                                   observation.gripper{i}_gripper_pose
    action                    [pos3, rot6d6, gripper_g] per arm (preferred), or the
                              older [pos3, rotvec3, gripper_g] export
    observation.camera0_pose  (6,)                 camera pose relative to episode start
                                                   (optional; absent for static-camera
                                                   robot data)
    observation.images.camera0                     video

GR00T slices its state/action groups from single columns, and its relative-EEF action
processing needs one contiguous 9-dim [pos3, rot6d6] state group per arm — so a flat
``observation.state`` column matching the action layout is required. This script:

    1. Detects the layout. If ``observation.state`` is missing it is synthesized from
       the per-key columns; if ``action`` is the older rotvec export it is converted to
       rot6d. Either rewrite requires --output-path.
    2. Writes ``meta/modality.json`` slicing state/action into named groups
       (right_eef_9d / right_gripper / left_eef_9d / left_gripper, plus a camera_pose
       group when ``observation.camera0_pose`` exists).
    3. In copy mode (--output-path), updates ``meta/info.json``, copies
       episodes.jsonl / tasks.jsonl and symlinks ``videos/`` (and ``data/`` when no
       rewrite is needed); pass --copy-videos to copy instead. Without --output-path the
       dataset is annotated in place (modality.json only; no rewrite may be needed).

``meta/stats.json`` and ``meta/relative_stats.json`` are NOT generated here — the GR00T
DatasetFactory generates them automatically at training start (or run gr00t/data/stats.py).

Usage:
    # dataset already has rot6d actions + observation.state -> just add modality.json
    python scripts/convert_motiontrans_to_gr00t.py --input-path /data/human_stack_cup

    # per-key export without observation.state (or older rotvec actions)
    python scripts/convert_motiontrans_to_gr00t.py \
        --input-path /data/human_stack_cup --output-path /data/human_stack_cup_gr00t
"""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import tyro


def rotvec_to_rot6d(rotvec: np.ndarray) -> np.ndarray:
    """(N, 3) rotation vectors -> (N, 6) rot6d (first two rows of the rotation matrix)."""
    matrices = Rotation.from_rotvec(rotvec).as_matrix()  # (N, 3, 3)
    return matrices[:, :2, :].reshape(len(rotvec), 6)


@dataclass
class Layout:
    n_arms: int
    gripper_dim: int
    action_is_rot6d: bool  # False: older [pos3, rotvec3, gripper] action export
    has_state: bool  # observation.state column present
    per_key_rot_is_rot6d: bool  # per-key rotation column is _eef_rot6d vs _eef_rot_axis_angle

    @property
    def state_dim(self) -> int:
        return self.n_arms * (9 + self.gripper_dim)

    @property
    def needs_rewrite(self) -> bool:
        return not (self.action_is_rot6d and self.has_state)


def infer_layout(features: dict) -> Layout:
    """Infer the dataset layout from the info.json features."""
    action_dim = features["action"]["shape"][0]
    action_names = features["action"].get("names") or []
    if action_names:
        action_is_rot6d = any("rot6d" in name for name in action_names)
    else:
        action_is_rot6d = "observation.state" in features and (
            action_dim == features["observation.state"]["shape"][0]
        )

    has_state = "observation.state" in features
    if has_state:
        state_dim = features["observation.state"]["shape"][0]
        state_names = features["observation.state"].get("names") or []
        n_arms = 2 if any("robot1_" in name for name in state_names) else 1
        gripper_dim = state_dim // n_arms - 9
    else:
        assert "observation.robot0_eef_pos" in features, (
            "Neither observation.state nor per-key observation.robot0_* columns found — "
            "not a MotionTrans-schema dataset"
        )
        n_arms = 2 if "observation.robot1_eef_pos" in features else 1
        gripper_dim = features["observation.gripper0_gripper_pose"]["shape"][0]

    per_key_rot_is_rot6d = "observation.robot0_eef_rot6d" in features

    layout = Layout(n_arms, gripper_dim, action_is_rot6d, has_state, per_key_rot_is_rot6d)
    expected_action_dim = layout.state_dim if action_is_rot6d else layout.state_dim - 3 * n_arms
    assert n_arms in (1, 2) and gripper_dim >= 1 and action_dim == expected_action_dim, (
        f"Inconsistent layout: action_dim={action_dim}, n_arms={n_arms}, "
        f"gripper_dim={gripper_dim}, action_is_rot6d={action_is_rot6d}"
    )
    return layout


def arm_prefixes(n_arms: int) -> list[str]:
    """Group-name prefixes per arm. MotionTrans convention: robot0 = right, robot1 = left."""
    return ["right_", "left_"][:n_arms] if n_arms == 2 else [""]


def convert_action_column(actions: np.ndarray, layout: Layout) -> np.ndarray:
    """(N, n*(6+g)) rotvec actions -> (N, n*(9+g)) rot6d actions."""
    in_block = 6 + layout.gripper_dim
    parts = []
    for arm in range(layout.n_arms):
        block = actions[:, arm * in_block : (arm + 1) * in_block]
        parts.append(block[:, :3])  # position
        parts.append(rotvec_to_rot6d(block[:, 3:6]))  # rotation
        parts.append(block[:, 6:])  # gripper
    return np.concatenate(parts, axis=1)


def _column_2d(df: pd.DataFrame, column: str) -> np.ndarray:
    """Stack a per-row column to (N, D). Scalar-valued rows (e.g. a 1-DOF gripper stored
    as plain floats instead of length-1 arrays) are reshaped to (N, 1)."""
    stacked = np.stack(df[column].to_numpy())
    return stacked.reshape(len(stacked), -1)


def build_state_column(df: pd.DataFrame, layout: Layout) -> np.ndarray:
    """Synthesize the flat observation.state column from the per-key columns.

    Per-arm layout matches the (converted) action column: [pos3, rot6d6, gripper_g].
    """
    parts = []
    for arm in range(layout.n_arms):
        pos = _column_2d(df, f"observation.robot{arm}_eef_pos")
        if layout.per_key_rot_is_rot6d:
            rot = _column_2d(df, f"observation.robot{arm}_eef_rot6d")
        else:
            rotvec = _column_2d(df, f"observation.robot{arm}_eef_rot_axis_angle")
            rot = rotvec_to_rot6d(rotvec.astype(np.float64))
        gripper = _column_2d(df, f"observation.gripper{arm}_gripper_pose")
        parts += [pos, rot, gripper]
    return np.concatenate(parts, axis=1).astype(np.float32)


def per_arm_names(layout: Layout) -> list[str]:
    names = []
    for arm in range(layout.n_arms):
        names += [f"robot{arm}_eef_pos_{ax}" for ax in "xyz"]
        names += [f"robot{arm}_eef_rot6d_{i}" for i in range(6)]
        if layout.gripper_dim == 1:
            names += [f"gripper{arm}_q"]
        else:
            names += [f"gripper{arm}_q{i}" for i in range(layout.gripper_dim)]
    return names


def build_modality_json(features: dict, layout: Layout) -> dict:
    state_groups, action_groups = {}, {}
    block = 9 + layout.gripper_dim  # state and (converted) action share the rot6d layout
    for arm, prefix in enumerate(arm_prefixes(layout.n_arms)):
        start = arm * block
        state_groups[f"{prefix}eef_9d"] = {"start": start, "end": start + 9}
        state_groups[f"{prefix}gripper"] = {
            "start": start + 9,
            "end": start + 9 + layout.gripper_dim,
        }
        action_groups[f"{prefix}eef_9d"] = {"start": start, "end": start + 9}
        action_groups[f"{prefix}gripper"] = {
            "start": start + 9,
            "end": start + 9 + layout.gripper_dim,
        }

    if "observation.camera0_pose" in features:
        state_groups["camera_pose"] = {
            "start": 0,
            "end": 6,
            "original_key": "observation.camera0_pose",
        }

    video_groups = {}
    for key in features:
        if features[key].get("dtype") == "video" or key.startswith("observation.images."):
            video_groups[key.removeprefix("observation.images.")] = {"original_key": key}

    return {
        "state": state_groups,
        "action": action_groups,
        "video": video_groups,
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    if copy:
        shutil.copytree(src, dst)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=True)


def main(
    input_path: Path,
    output_path: Path | None = None,
    copy_videos: bool = False,
    overwrite: bool = False,
) -> None:
    """Prepare a MotionTrans LeRobot dataset for GR00T training.

    Args:
        input_path: MotionTrans-exported LeRobot dataset (contains meta/, data/, videos/).
        output_path: Destination directory. Omit to annotate the dataset in place
            (writes meta/modality.json only; requires rot6d actions AND an existing
            observation.state column — any data rewrite needs an output path).
        copy_videos: Copy videos/ (and data/) instead of symlinking them in copy mode.
        overwrite: Allow writing into an existing output directory / overwriting an
            existing modality.json.
    """
    input_path = Path(input_path)
    with open(input_path / "meta" / "info.json") as f:
        info = json.load(f)
    features = info["features"]
    layout = infer_layout(features)
    print(
        f"Detected layout: {layout.n_arms} arm(s), gripper dim {layout.gripper_dim}, "
        f"action={'rot6d' if layout.action_is_rot6d else 'rotvec (will convert)'}, "
        f"observation.state={'present' if layout.has_state else 'absent (will synthesize)'}, "
        f"camera_pose={'yes (egocentric)' if 'observation.camera0_pose' in features else 'no (static camera)'}"
    )
    modality = build_modality_json(features, layout)

    if output_path is None:
        # In-place: the dataset is already GR00T-ready except for modality.json.
        assert not layout.needs_rewrite, (
            "This dataset needs a data rewrite ("
            + ("action rotvec->rot6d; " if not layout.action_is_rot6d else "")
            + (
                "observation.state must be synthesized from the per-key columns; "
                if not layout.has_state
                else ""
            )
            + ") — in-place annotation is not possible. Pass --output-path."
        )
        modality_path = input_path / "meta" / "modality.json"
        if modality_path.exists() and not overwrite:
            raise FileExistsError(f"{modality_path} exists, pass --overwrite to replace it")
        with open(modality_path, "w") as f:
            json.dump(modality, f, indent=4)
        print(f"Wrote {modality_path} (dataset annotated in place)")
    else:
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"{output_path} exists, pass --overwrite to replace its contents")

        # --- data/ --------------------------------------------------------------------
        if not layout.needs_rewrite:
            output_path.mkdir(parents=True, exist_ok=True)
            _link_or_copy(input_path / "data", output_path / "data", copy=copy_videos)
        else:
            parquet_files = sorted(input_path.glob("data/*/*.parquet"))
            assert parquet_files, f"No parquet files found under {input_path}/data"
            for parquet_file in tqdm(parquet_files, desc="Rewriting parquet data"):
                df = pd.read_parquet(parquet_file)
                if not layout.action_is_rot6d:
                    actions = np.stack(df["action"].to_numpy()).astype(np.float64)
                    df["action"] = list(convert_action_column(actions, layout).astype(np.float32))
                if not layout.has_state:
                    df["observation.state"] = list(build_state_column(df, layout))
                out_file = output_path / parquet_file.relative_to(input_path)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out_file)

        # --- meta/ ----------------------------------------------------------------------
        meta_out = output_path / "meta"
        meta_out.mkdir(parents=True, exist_ok=True)

        if not layout.action_is_rot6d:
            info["features"]["action"] = {
                "dtype": "float32",
                "shape": [layout.state_dim],
                "names": per_arm_names(layout),
            }
        if not layout.has_state:
            info["features"]["observation.state"] = {
                "dtype": "float32",
                "shape": [layout.state_dim],
                "names": per_arm_names(layout),
            }
        with open(meta_out / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        for filename in ["episodes.jsonl", "tasks.jsonl"]:
            shutil.copy(input_path / "meta" / filename, meta_out / filename)
        # episodes_stats.jsonl is intentionally not copied: GR00T reads meta/stats.json
        # (auto-generated) instead, and its stats would be stale after a rewrite.

        with open(meta_out / "modality.json", "w") as f:
            json.dump(modality, f, indent=4)

        # --- videos/ ----------------------------------------------------------------------
        _link_or_copy(input_path / "videos", output_path / "videos", copy=copy_videos)

        print(f"Done. Converted dataset written to {output_path}")

    print("Groups in modality.json:")
    for modality_type in ["state", "action"]:
        for name, group in modality[modality_type].items():
            print(f"  {modality_type}.{name}: [{group['start']}:{group['end']}]")


if __name__ == "__main__":
    tyro.cli(main)
