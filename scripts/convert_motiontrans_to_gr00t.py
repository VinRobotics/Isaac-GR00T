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

Input schema (see gr00t_lerobot_dataset.md in the motiontrans-pi0 repo):
    observation.state        (n_arms * (9 + g),)  [pos3, rot6d6, gripper_g] per arm
                                                  (absolute, episode-start camera frame
                                                  for human data)
    action                   [pos3, rot6d6, gripper_g] per arm (preferred), or the older
                             [pos3, rotvec3, gripper_g] export
    observation.camera0_pose (6,)                 camera pose relative to episode start
                                                  (optional; absent for static-camera
                                                  robot data)
    observation.images.camera0                    video

GR00T requires the action EEF blocks to share the exact layout of the
``observation.state`` EEF blocks (pos + rot6d, matching the pretrained ``eef_9d``
representation). This script:

    1. Detects the action layout. If the action is already ``[pos3, rot6d6, gripper_g]``
       per arm, no data is rewritten; if it is the older rotvec export, the ``action``
       column is converted to rot6d (requires --output-path).
    2. Writes ``meta/modality.json`` slicing state/action into named groups
       (right_eef_9d / right_gripper / left_eef_9d / left_gripper, plus a camera_pose
       group when ``observation.camera0_pose`` exists).
    3. In copy mode (--output-path), updates ``meta/info.json``, copies
       episodes.jsonl / tasks.jsonl and symlinks ``videos/`` (and ``data/`` when no
       rewrite is needed); pass --copy-videos to copy instead. Without --output-path the
       dataset is annotated in place (modality.json only; rot6d layout required).

``meta/stats.json`` and ``meta/relative_stats.json`` are NOT generated here — the GR00T
DatasetFactory generates them automatically at training start (or run gr00t/data/stats.py).

Usage:
    # dataset already stores rot6d actions -> just add modality.json in place
    python scripts/convert_motiontrans_to_gr00t.py --input-path /data/human_stack_cup

    # older rotvec export -> full conversion into a new directory
    python scripts/convert_motiontrans_to_gr00t.py \
        --input-path /data/human_stack_cup --output-path /data/human_stack_cup_gr00t
"""

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


def infer_layout(features: dict) -> tuple[int, int, bool]:
    """Infer (n_arms, gripper_dim, action_is_rot6d) from the state/action features.

    Per arm the state block is 9 + g (pos3 + rot6d6 + gripper_g); the action block is
    9 + g when already rot6d, or 6 + g in the older rotvec export
    (state_dim - action_dim == 3 * n_arms).
    """
    state_dim = features["observation.state"]["shape"][0]
    action_dim = features["action"]["shape"][0]
    state_names = features["observation.state"].get("names") or []
    n_arms = 2 if any("robot1_" in name for name in state_names) else 1

    action_names = features["action"].get("names") or []
    if action_names:
        action_is_rot6d = any("rot6d" in name for name in action_names)
    else:
        action_is_rot6d = action_dim == state_dim

    expected_action_dim = state_dim if action_is_rot6d else state_dim - 3 * n_arms
    assert action_dim == expected_action_dim, (
        f"Inconsistent layout: state_dim={state_dim}, action_dim={action_dim}, "
        f"n_arms={n_arms}, action_is_rot6d={action_is_rot6d}"
    )
    gripper_dim = state_dim // n_arms - 9
    assert gripper_dim >= 1 and state_dim == n_arms * (9 + gripper_dim), (
        f"Inconsistent state layout: state_dim={state_dim}, n_arms={n_arms}"
    )
    return n_arms, gripper_dim, action_is_rot6d


def arm_prefixes(n_arms: int) -> list[str]:
    """Group-name prefixes per arm. MotionTrans convention: robot0 = right, robot1 = left."""
    return ["right_", "left_"][:n_arms] if n_arms == 2 else [""]


def convert_action_row_block(actions: np.ndarray, n_arms: int, gripper_dim: int) -> np.ndarray:
    """(N, n*(6+g)) rotvec actions -> (N, n*(9+g)) rot6d actions."""
    in_block = 6 + gripper_dim
    parts = []
    for arm in range(n_arms):
        block = actions[:, arm * in_block : (arm + 1) * in_block]
        parts.append(block[:, :3])  # position
        parts.append(rotvec_to_rot6d(block[:, 3:6]))  # rotation
        parts.append(block[:, 6:])  # gripper
    return np.concatenate(parts, axis=1)


def build_modality_json(
    features: dict, n_arms: int, gripper_dim: int, has_camera_pose: bool
) -> dict:
    state_groups, action_groups = {}, {}
    block = 9 + gripper_dim  # state and (converted) action share the same rot6d layout
    for arm, prefix in enumerate(arm_prefixes(n_arms)):
        start = arm * block
        state_groups[f"{prefix}eef_9d"] = {"start": start, "end": start + 9}
        state_groups[f"{prefix}gripper"] = {"start": start + 9, "end": start + 9 + gripper_dim}
        action_groups[f"{prefix}eef_9d"] = {"start": start, "end": start + 9}
        action_groups[f"{prefix}gripper"] = {"start": start + 9, "end": start + 9 + gripper_dim}

    if has_camera_pose:
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
            (writes meta/modality.json only; requires the action column to already be in
            rot6d layout).
        copy_videos: Copy videos/ (and data/) instead of symlinking them in copy mode.
        overwrite: Allow writing into an existing output directory / overwriting an
            existing modality.json.
    """
    input_path = Path(input_path)
    with open(input_path / "meta" / "info.json") as f:
        info = json.load(f)
    features = info["features"]
    n_arms, gripper_dim, action_is_rot6d = infer_layout(features)
    has_camera_pose = "observation.camera0_pose" in features
    print(
        f"Detected layout: {n_arms} arm(s), gripper dim {gripper_dim}, "
        f"action={'rot6d (no rewrite needed)' if action_is_rot6d else 'rotvec (will convert)'}, "
        f"camera_pose={'yes (egocentric)' if has_camera_pose else 'no (static camera)'}"
    )
    modality = build_modality_json(features, n_arms, gripper_dim, has_camera_pose)

    if output_path is None:
        # In-place: the dataset is already GR00T-ready except for modality.json.
        assert action_is_rot6d, (
            "The action column uses the older rotvec layout; in-place annotation is not "
            "possible. Pass --output-path to convert into a new directory."
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
            raise FileExistsError(
                f"{output_path} exists, pass --overwrite to replace its contents"
            )

        # --- data/ --------------------------------------------------------------------
        if action_is_rot6d:
            output_path.mkdir(parents=True, exist_ok=True)
            _link_or_copy(input_path / "data", output_path / "data", copy=copy_videos)
        else:
            parquet_files = sorted(input_path.glob("data/*/*.parquet"))
            assert parquet_files, f"No parquet files found under {input_path}/data"
            for parquet_file in tqdm(parquet_files, desc="Converting action column"):
                df = pd.read_parquet(parquet_file)
                actions = np.stack(df["action"].to_numpy()).astype(np.float64)
                new_actions = convert_action_row_block(actions, n_arms, gripper_dim)
                df["action"] = list(new_actions.astype(np.float32))
                out_file = output_path / parquet_file.relative_to(input_path)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out_file)

        # --- meta/ ----------------------------------------------------------------------
        meta_out = output_path / "meta"
        meta_out.mkdir(parents=True, exist_ok=True)

        if not action_is_rot6d:
            action_names = []
            for prefix in arm_prefixes(n_arms):
                arm_tag = prefix.rstrip("_") or "arm0"
                action_names += [f"{arm_tag}_eef_pos_{ax}" for ax in "xyz"]
                action_names += [f"{arm_tag}_eef_rot6d_{i}" for i in range(6)]
                action_names += [f"{arm_tag}_gripper_{i}" for i in range(gripper_dim)]
            info["features"]["action"] = {
                "dtype": "float32",
                "shape": [n_arms * (9 + gripper_dim)],
                "names": action_names,
            }
        with open(meta_out / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        for filename in ["episodes.jsonl", "tasks.jsonl"]:
            shutil.copy(input_path / "meta" / filename, meta_out / filename)
        # episodes_stats.jsonl is intentionally not copied: GR00T reads meta/stats.json
        # (auto-generated) instead, and its action stats would be stale after a rewrite.

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
