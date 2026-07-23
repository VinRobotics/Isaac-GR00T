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

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import barrier


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def _ensure_stats(self, dataset_path: str, embodiment_tag: str) -> None:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                generate_stats(dataset_path)
                generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
        else:
            generate_stats(dataset_path)
            generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
        barrier()

    def _build_single_step_dataset(
        self,
        dataset_path: str,
        embodiment_tag: str,
        loss_weight: float = 1.0,
        episode_loader: LeRobotEpisodeLoader | None = None,
        episode_indices: list[int] | None = None,
        collect_viz_images: bool = False,
    ) -> ShardedSingleStepDataset:
        assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
        if episode_loader is None:
            # Skipped when reusing an already-built loader (train/val split): stats are
            # computed once from the whole physical dataset and don't depend on the split.
            self._ensure_stats(dataset_path, embodiment_tag)
        return ShardedSingleStepDataset(
            dataset_path=dataset_path,
            embodiment_tag=EmbodimentTag(embodiment_tag),
            modality_configs=self.config.data.modality_configs[embodiment_tag],
            video_backend=self.config.data.video_backend,
            shard_size=self.config.data.shard_size,
            episode_sampling_rate=self.config.data.episode_sampling_rate,
            seed=self.config.data.seed,
            allow_padding=self.config.data.allow_padding,
            shard_load_workers=self.config.data.shard_load_workers,
            video_decode_workers=self.config.data.video_decode_workers,
            num_ffmpeg_threads=self.config.data.num_ffmpeg_threads,
            overlap_episode_io=self.config.data.overlap_episode_io,
            loss_weight=loss_weight,
            episode_loader=episode_loader,
            episode_indices=episode_indices,
            collect_viz_images=collect_viz_images,
        )

    def _build_split_datasets(
        self,
        dataset_path: str,
        embodiment_tag: str,
        val_ratio: float,
        loss_weight: float = 1.0,
        collect_viz_images: bool = False,
    ) -> tuple[ShardedSingleStepDataset, ShardedSingleStepDataset | None]:
        """Hold out val_ratio of this dataset's episodes for validation.

        The train and val subsets share one LeRobotEpisodeLoader (metadata parsed
        once); only which episodes get sharded into each differs.
        """
        self._ensure_stats(dataset_path, embodiment_tag)
        modality_configs = self.config.data.modality_configs[embodiment_tag]
        episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            video_backend=self.config.data.video_backend,
            video_decode_workers=self.config.data.video_decode_workers,
            num_ffmpeg_threads=self.config.data.num_ffmpeg_threads,
            overlap_episode_io=self.config.data.overlap_episode_io,
        )
        num_episodes = len(episode_loader.episode_lengths)
        rng = np.random.default_rng(self.config.data.seed)
        perm = rng.permutation(num_episodes).tolist()
        num_val = max(1, round(num_episodes * val_ratio)) if num_episodes > 1 else 0
        val_indices, train_indices = perm[:num_val], perm[num_val:]
        assert train_indices, (
            f"eval_set_split_ratio={val_ratio} leaves no training episodes for "
            f"{dataset_path} ({num_episodes} episodes total)"
        )

        train_dataset = self._build_single_step_dataset(
            dataset_path,
            embodiment_tag,
            loss_weight=loss_weight,
            episode_loader=episode_loader,
            episode_indices=train_indices,
        )
        val_dataset = None
        if val_indices:
            val_dataset = self._build_single_step_dataset(
                dataset_path,
                embodiment_tag,
                # Unweighted: eval loss/metrics should reflect plain held-out
                # performance, not the co-training alpha re-weighting.
                loss_weight=1.0,
                episode_loader=episode_loader,
                episode_indices=val_indices,
                collect_viz_images=collect_viz_images,
            )
        return train_dataset, val_dataset

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        val_paths = getattr(self.config.data, "val_dataset_paths", [])
        eval_split_ratio = getattr(self.config.training, "eval_set_split_ratio", 0.0) or 0.0
        use_auto_split = (
            self.config.training.eval_strategy != "no" and not val_paths and eval_split_ratio > 0.0
        )
        collect_viz_images = getattr(self.config.model, "enable_keypoint_head", False)

        all_datasets = []
        all_weights = []
        split_val_datasets = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            embodiment_tag = dataset_spec.embodiment_tag
            assert embodiment_tag is not None, "Embodiment tag is required"
            for dataset_path in dataset_spec.dataset_paths:
                if use_auto_split:
                    train_ds, val_ds = self._build_split_datasets(
                        dataset_path,
                        embodiment_tag,
                        eval_split_ratio,
                        loss_weight=getattr(dataset_spec, "loss_weight", 1.0),
                        collect_viz_images=collect_viz_images,
                    )
                    datasets.append(train_ds)
                    if val_ds is not None:
                        split_val_datasets.append(val_ds)
                else:
                    datasets.append(
                        self._build_single_step_dataset(
                            dataset_path,
                            embodiment_tag,
                            loss_weight=getattr(dataset_spec, "loss_weight", 1.0),
                        )
                    )
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        train_dataset = ShardedMixtureDataset(
            datasets=all_datasets,
            weights=all_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            interleave_two_datasets=self.config.data.interleave_two_datasets,
        )

        eval_dataset = None
        val_datasets = []
        if self.config.training.eval_strategy != "no" and val_paths:
            # Explicit validation paths take precedence over the auto-split above.
            embodiment_tag = self.config.data.datasets[0].embodiment_tag
            assert embodiment_tag is not None, "Embodiment tag is required for validation"
            for val_path in tqdm(val_paths, desc="Initializing validation datasets"):
                val_datasets.append(
                    self._build_single_step_dataset(
                        val_path, embodiment_tag, collect_viz_images=collect_viz_images
                    )
                )
        elif use_auto_split:
            val_datasets = split_val_datasets

        if val_datasets:
            val_weights = [1.0 / len(val_datasets)] * len(val_datasets)
            eval_dataset = ShardedMixtureDataset(
                datasets=val_datasets,
                weights=val_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=False,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            )

        return train_dataset, eval_dataset
