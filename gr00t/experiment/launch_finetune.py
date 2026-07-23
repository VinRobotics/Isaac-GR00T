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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


def dataset_frame_stats(dataset_path: str) -> tuple[int, bool]:
    """Return (total_frames, is_human) for a LeRobot dataset.

    Human data is detected from the observation.is_human column (first row of the first
    parquet); if the column is absent, falls back to "human" appearing in info.json's
    robot_type. Used by --alpha to compute MotionTrans-style co-training weights.
    """
    import numpy as np
    import pandas as pd

    dataset_path = Path(dataset_path)
    with open(dataset_path / "meta" / "info.json") as f:
        info = json.load(f)
    total_frames = int(info["total_frames"])

    if "observation.is_human" in info.get("features", {}):
        first_parquet = sorted(dataset_path.glob("data/*/*.parquet"))[0]
        column = pd.read_parquet(first_parquet, columns=["observation.is_human"])
        is_human = bool(np.asarray(column.iloc[0, 0]).reshape(-1)[0] > 0.5)
    else:
        is_human = "human" in str(info.get("robot_type", "")).lower()
    return total_frames, is_human


def motiontrans_alpha_weights(
    dataset_paths: list[str], alpha: float
) -> tuple[list[float], list[float]]:
    """Compute (mix_ratios, loss_weights) reproducing MotionTrans's alpha scheme.

    Sampling is proportional to each dataset's frame count (uniform over the
    concatenated frames, as MotionTransDataset does) and per-sample loss weights are
    alpha/n_robot_frames for robot data vs (1-alpha)/n_human_frames for human data, so
    the total gradient contribution is robot:human = alpha:(1-alpha).
    """
    stats = [dataset_frame_stats(path) for path in dataset_paths]
    mix_ratios = [float(frames) for frames, _ in stats]
    n_human = sum(frames for frames, is_human in stats if is_human)
    n_robot = sum(frames for frames, is_human in stats if not is_human)
    if n_human == 0 or n_robot == 0:
        # Only one class present: alpha weighting degenerates to uniform weights,
        # matching MotionTransDataset's n_human==0 / n_robot==0 branches.
        loss_weights = [1.0] * len(stats)
    else:
        loss_weights = [
            (1 - alpha) / n_human if is_human else alpha / n_robot for _, is_human in stats
        ]
        # Only the ratio matters (the action head applies a weighted mean); rescale to a
        # sampling-weighted mean of 1 so per-sample weights stay O(1).
        total_frames = sum(frames for frames, _ in stats)
        mean_weight = (
            sum(frames * weight for (frames, _), weight in zip(stats, loss_weights)) / total_frames
        )
        loss_weights = [weight / mean_weight for weight in loss_weights]
    for path, (frames, is_human), weight in zip(dataset_paths, stats, loss_weights):
        print(
            f"[alpha={alpha}] {path}: {'human' if is_human else 'robot'}, "
            f"{frames} frames, loss_weight={weight:.3e}"
        )
    return mix_ratios, loss_weights


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    def per_dataset_values(values: list[float] | None, default: float, flag: str) -> list[float]:
        if values is None:
            return [default] * len(ft_config.dataset_path)
        assert len(values) == len(ft_config.dataset_path), (
            f"{flag} has {len(values)} entries but --dataset-path has {len(ft_config.dataset_path)}"
        )
        return values

    if ft_config.alpha is not None:
        assert ft_config.dataset_mix_ratio is None and ft_config.dataset_loss_weight is None, (
            "--alpha computes dataset mix ratios and loss weights automatically; "
            "do not combine it with --dataset-mix-ratio / --dataset-loss-weight"
        )
        dataset_mix_ratios, dataset_loss_weights = motiontrans_alpha_weights(
            ft_config.dataset_path, ft_config.alpha
        )
    else:
        dataset_mix_ratios = per_dataset_values(
            ft_config.dataset_mix_ratio, 1.0, "--dataset-mix-ratio"
        )
        dataset_loss_weights = per_dataset_values(
            ft_config.dataset_loss_weight, 1.0, "--dataset-loss-weight"
        )

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        # "dataset_paths": [ft_config.dataset_path],
                        "dataset_paths": [dataset_path],
                        "mix_ratio": mix_ratio,
                        "loss_weight": loss_weight,
                        "embodiment_tag": embodiment_tag,
                    }
                    for dataset_path, mix_ratio, loss_weight in zip(
                        ft_config.dataset_path, dataset_mix_ratios, dataset_loss_weights
                    )
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    # VLM-backbone motion-keypoint head + human/robot OT alignment
    config.model.enable_motion_head = ft_config.enable_motion_head
    config.model.motion_horizon = ft_config.motion_horizon
    config.model.max_motion_objects = ft_config.max_motion_objects
    config.model.motion_points_per_object = ft_config.motion_points_per_object
    config.model.num_motion_tokens = ft_config.num_motion_tokens
    config.model.motion_loss_weight = ft_config.motion_loss_weight
    config.model.motion_static_weight = ft_config.motion_static_weight
    config.model.motion_relative = ft_config.motion_relative
    config.model.motion_pool = ft_config.motion_pool
    config.model.enable_ot_align = ft_config.enable_ot_align
    config.model.ot_align_weight = ft_config.ot_align_weight
    config.model.ot_warmup_steps = ft_config.ot_warmup_steps
    config.model.ot_sinkhorn_eps = ft_config.ot_sinkhorn_eps
    config.model.ot_sinkhorn_iters = ft_config.ot_sinkhorn_iters
    if ft_config.enable_ot_align:
        assert len(ft_config.dataset_path) == 2, (
            "--enable-ot-align requires exactly two dataset paths so batches can be "
            "interleaved 50/50 between the human and robot sources."
        )
        per_device_batch_size = ft_config.global_batch_size // ft_config.num_gpus
        assert (
            ft_config.global_batch_size % ft_config.num_gpus == 0 and per_device_batch_size % 2 == 0
        ), (
            "--enable-ot-align requires an even per-device batch size for an exact "
            "50/50 human/robot split."
        )
        config.data.interleave_two_datasets = True

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.shard_load_workers = ft_config.shard_load_workers
    config.data.video_decode_workers = ft_config.video_decode_workers
    config.data.num_ffmpeg_threads = ft_config.num_ffmpeg_threads
    config.data.overlap_episode_io = ft_config.overlap_episode_io

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    if ft_config.validation_path is not None:
        config.data.val_dataset_paths = ft_config.validation_path
        config.training.eval_strategy = "steps"
        config.training.eval_steps = ft_config.eval_steps
    elif ft_config.eval_set_split_ratio is not None:
        config.training.eval_set_split_ratio = ft_config.eval_set_split_ratio
        config.training.eval_strategy = "steps"
        config.training.eval_steps = ft_config.eval_steps

    config.training.motion_viz_max_images = ft_config.motion_viz_max_images
    config.training.motion_video_episodes = ft_config.motion_video_episodes
    config.training.motion_video_max_frames = ft_config.motion_video_max_frames
    config.training.motion_video_batch_size = ft_config.motion_video_batch_size
    config.training.motion_video_fps = ft_config.motion_video_fps
    config.training.domain_viz_max_points = ft_config.domain_viz_max_points

    run(config)
