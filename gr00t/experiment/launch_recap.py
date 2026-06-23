#!/usr/bin/env python3
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

# RECAP two-phase training launcher for GR00T N1.7.
#
# Phase 1 (--recap-phase value_head):
#   Policy frozen. Distributional value head Vϕ trained on empirical returns.
#   Requires reward fields in the dataset:
#       reward, reward.current_frame_idx, reward.episode_lengths
#
# Phase 2 (--recap-phase policy):
#   Value head frozen (provides advantage labels). Policy πθ trained with
#   advantage conditioning (RECAP Eq. 3 dual-term loss).
#   Requires: reward field only.
#
# References:
#   RECAP / π*0.6  arXiv:2511.14759
#   CFGRL          arXiv:2505.23458

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import tyro
import wandb
from omegaconf import OmegaConf
from transformers import TrainingArguments, set_seed

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.trainer import Gr00tTrainer
from gr00t.experiment.utils import BestMetricCheckpointCallback, CheckpointFormatCallback
from gr00t.model import MODEL_REGISTRY
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, save_initial_actions


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


@dataclass
class RECAPFinetuneConfig(FinetuneConfig):
    """
    Extends FinetuneConfig with RECAP-specific parameters.

    Two-phase RECAP training:
      Phase 1 (value_head): train Vϕ, policy frozen.
      Phase 2 (policy):     train πθ with frozen Vϕ and advantage conditioning.
    """

    # --- RECAP phase ---
    recap_phase: str = "value_head"
    """
    Which RECAP phase to run.
      "value_head" — Phase 1: train distributional value head only.
      "policy"     — Phase 2: train policy with advantage conditioning.
    """

    # --- RECAP hyperparameters ---
    recap_alpha: float = 1.0
    """
    α in RECAP Eq. 3: weight of the conditional term relative to the
    unconditional term. π*0.6 uses α=1.0.
    """

    advantage_cfg_dropout_prob: float = 0.3
    """
    Probability of replacing the advantage token with the null token during
    training. Enables both conditional and unconditional inference at test time.
    """

    cfg_guidance_weight: float = 1.5
    """
    CFGRL guidance weight w at inference.
      w=1.0: sample directly from advantage-conditioned policy.
      w>1.0: amplify optimality signal (useful range 1.5–2.5).
    """

    advantage_threshold_percentile: float = 0.30
    """
    Percentile of V predictions used as ε_l threshold.
    π*0.6 App. F: 0.30 for pre-training, 0.40 for fine-tuning.
    """

    value_head_hidden_dim: int = 2048
    """Hidden dimension for the distributional value head MLP."""

    num_bins: int = 201
    """Number of distributional value bins (B=201 as in π*0.6)."""

    value_loss_coeff: float = 1.0
    """Weight on the value head cross-entropy loss."""


def run_recap(ft_config: RECAPFinetuneConfig):
    """
    Main RECAP training function.

    Mirrors launch_finetune.py / experiment.py but calls set_phase_* on the
    action head BEFORE the HF Trainer (and its optimizer) are created.
    """
    assert ft_config.recap_phase in ("value_head", "policy"), (
        f"--recap-phase must be 'value_head' or 'policy', got '{ft_config.recap_phase}'"
    )

    # ----- distributed setup ---------------------------------------------------
    if dist.is_initialized():
        global_rank = dist.get_rank()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
    else:
        local_rank = 0
        global_rank = 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    if global_rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    from gr00t.data.embodiment_tags import EmbodimentTag
    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    set_seed(42)

    # ----- build config --------------------------------------------------------
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                    for dataset_path in ft_config.dataset_path
                ],
            }
        }
    )
    config.load_config_path = None

    # standard finetune overrides
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

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # RECAP-specific model config
    config.model.use_recap = True
    config.model.recap_alpha = ft_config.recap_alpha
    config.model.advantage_cfg_dropout_prob = ft_config.advantage_cfg_dropout_prob
    config.model.cfg_guidance_weight = ft_config.cfg_guidance_weight
    config.model.advantage_threshold_percentile = ft_config.advantage_threshold_percentile
    config.model.value_head_hidden_dim = ft_config.value_head_hidden_dim
    config.model.num_bins = ft_config.num_bins
    config.model.value_loss_coeff = ft_config.value_loss_coeff

    # training config
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

    # ----- output dir ----------------------------------------------------------
    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        experiment_name = output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / "processor"
    config.save(save_cfg_dir / "config.yaml")
    omegaconf_config = OmegaConf.create(config.__dict__)
    omegaconf_config["max_steps"] = config.training.max_steps
    omegaconf_config["save_steps"] = config.training.save_steps
    OmegaConf.save(omegaconf_config, save_cfg_dir / "conf.yaml", resolve=True)

    if config.training.use_wandb and global_rank == 0:
        wandb.init(
            project=config.training.wandb_project,
            name=experiment_name,
            config={**config.__dict__, "recap_phase": ft_config.recap_phase},
            tags=["recap", ft_config.recap_phase],
        )

    # ----- model setup ---------------------------------------------------------
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)
    pipeline.setup()
    model = pipeline.return_model()

    # ── RECAP phase management ──────────────────────────────────────────────────
    # Must happen BEFORE HF Trainer creates the optimizer so that only the
    # parameters for the active phase are included in the optimizer.
    action_head = model.action_head
    if ft_config.recap_phase == "value_head":
        logging.info("[RECAP] Activating Phase 1: VALUE_HEAD")
        action_head.set_phase_value_head()
    else:
        logging.info("[RECAP] Activating Phase 2: POLICY")
        action_head.set_phase_policy()

    train_dataset, eval_dataset = pipeline.return_dataset()
    data_collator = pipeline.return_collator()
    processor = pipeline.return_processor()
    processor.save_pretrained(processor_dir)

    # ----- training args -------------------------------------------------------
    if config.training.num_gpus > 1 and not config.training.use_ddp:
        deepspeed_config = config.get_deepspeed_config()
    else:
        deepspeed_config = None

    per_device_batch_size = config.training.global_batch_size // config.training.num_gpus

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        save_only_model=config.training.save_only_model,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to="wandb" if config.training.use_wandb else "none",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=config.training.ddp_bucket_cap_mb,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        batch_eval_metrics=True,
        remove_unused_columns=config.training.remove_unused_columns,
        ignore_data_skip=True,
    )

    # ----- trainer -------------------------------------------------------------
    trainer = Gr00tTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        multiprocessing_context=config.data.multiprocessing_context,
    )

    trainer.add_callback(
        CheckpointFormatCallback(
            run_name=experiment_name,
            exp_cfg_dir=save_cfg_dir,
            processor_dir=processor_dir,
        )
    )

    if config.training.save_best_eval_metric_name != "":
        trainer.add_callback(
            BestMetricCheckpointCallback(
                metric_name=config.training.save_best_eval_metric_name,
                greater_is_better=config.training.save_best_eval_metric_greater_is_better,
                exp_cfg_dir=save_cfg_dir,
            )
        )

    if hasattr(train_dataset, "get_initial_actions"):
        initial_actions = train_dataset.get_initial_actions()
        if initial_actions:
            initial_actions_path = save_cfg_dir / INITIAL_ACTIONS_FILENAME
            save_initial_actions(initial_actions, initial_actions_path)

    # ----- train ---------------------------------------------------------------
    logging.info(f"[RECAP] Starting Phase: {ft_config.recap_phase.upper()}")
    trainer.train(resume_from_checkpoint=True)

    logging.info("[RECAP] Training complete.")


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(RECAPFinetuneConfig)
    run_recap(ft_config)
