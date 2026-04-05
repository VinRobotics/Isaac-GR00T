# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import BasePolicy

# numpy print precision settings 3, dont use exponential notation
np.set_printoptions(precision=3, suppress=True)


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
    save_plot_path=None,
):
    pred_value_across_time = []

    for step_count in range(steps):
        data_point = None

        if step_count % action_horizon == 0:
            if data_point is None:
                data_point = dataset.get_step_data(traj_id, step_count)

            print("inferencing at step: ", step_count)
            value = policy.get_value(data_point)
            for _ in range(action_horizon):
                pred_value_across_time.append(np.atleast_1d(value["value_pred"]))

    # plot the joints
    pred_value_across_time = np.array(pred_value_across_time)[:steps]
    print(f"{pred_value_across_time=} {pred_value_across_time.shape=}")

    print("pred_value_joints vs time", pred_value_across_time.shape)

    # raise error when pred value has NaN
    if np.isnan(pred_value_across_time).any():
        raise ValueError("Pred value has NaN")

    if plot or save_plot_path is not None:
        info = {
            "pred_value_across_time": pred_value_across_time,
            "traj_id": traj_id,
            "action_horizon": action_horizon,
            "steps": steps,
        }
        plot_trajectory(info, f"{save_plot_path}.png")

    return


def plot_trajectory(
    info,
    save_plot_path=None,
):
    """Simple plot of the trajectory with state, gt action, and pred action."""

    # Use non interactive backend for matplotlib if headless
    if save_plot_path is not None:
        matplotlib.use("Agg")

    pred_value_across_time = info["pred_value_across_time"]
    traj_id = info["traj_id"]
    action_horizon = info["action_horizon"]
    steps = info["steps"]

    # Adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 4 * 1 + 2))

    # Leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    modality_string = ""
    title_text = f"Trajectory Analysis - ID: {traj_id}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)

    axes.plot(pred_value_across_time, label="pred value", linewidth=2)

    axes.legend(loc="upper right", framealpha=0.9)
    axes.grid(True, alpha=0.3)

    # Set better axis labels
    axes.set_xlabel("Time Step", fontsize=10)
    axes.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print("saving plot to", save_plot_path)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
