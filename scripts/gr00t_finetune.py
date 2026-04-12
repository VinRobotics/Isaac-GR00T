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

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model
def print_layer_weights(layer):
    # ESCNN Linear stores weights in: layer.weights
    if hasattr(layer, "weights"):
        W = layer.weights.data
        print("Weight shape:", tuple(W.shape))
        print("dtype:", W.dtype)
        print("min:", W.min())
        print("max:", W.max())
        print("mean:", W.mean())
        print("std:", W.std())
        print("has_nan:", torch.isnan(W).any())
    else:
        print("Layer has no .weights attribute")

    if hasattr(layer, "bias") and layer.bias is not None:
        b = layer.bias.data
        print("Bias shape:", tuple(b.shape))
        print("Bias dtype:", b.dtype)
        print("Bias min/max:", b.min(), b.max())
        print("Bias has_nan:", torch.isnan(b).any())
    else:
        print("No bias")

@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories, we assume all datasets have the same data config"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data configuration to use for training.
    Options:
    - Built-in configs: Use predefined config names like 'so100', 'fourier_gr1_arms_only', 'unitree_g1'.
    - External configs: Use 'module:ClassName' format to load custom configs from external files. e.g. 'my_dir.my_configs:RobotConfig'
    See gr00t/experiment/data_config.py for more details.
    """

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    tune_inv_dit: bool = True
    """Whether to fine-tune the inv_dit (FA hybrid arch). Should be True when FA encoders are used."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    equi_scale_factor: int = 1
    """Scale factor for the equivariant representation capacity.
    project_to_dim = pretrained_project_to_dim * equi_scale_factor.
    scale_factor=1 (default) keeps the pretrained dimensions unchanged.
    scale_factor=2 doubles project_to_dim (e.g. 2048 → 4096, blocks 256 → 512)."""

    rot_aug: bool = False
    """Whether to apply rotation augmentation during training."""

    adapter_warmup_steps: int = 1000
    """Number of steps to linearly ramp the EquiAdapter output from 0→1.
    0 (default) disables warm-up (adapter is fully active from step 0)."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend to use for training. [torchcodec, decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""

#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon and num_hand
    data_action_horizon = len(data_config_cls.action_indices)
    data_num_hand = getattr(data_config_cls, "num_hand", 2)
    data_rot_type = getattr(data_config_cls, "rot_type", "quaternion")
    data_rel_action = getattr(data_config_cls, "rel_action", False)
    
    # Get rotation config for frame averaging backbone
    rotation_config = data_config_cls.get_rotation_config() if hasattr(data_config_cls, "get_rotation_config") else {}
    print(rotation_config)
    # Build backbone_cfg overrides: rotation config + always persist n_group
    backbone_cfg_overrides = dict(rotation_config)

    # Scale project_to_dim if equi_scale_factor != 1
    if config.equi_scale_factor != 1:
        from gr00t.model.gr00t_n1 import GR00T_N1_5_Config
        from transformers import AutoConfig
        _base_cfg = AutoConfig.from_pretrained(config.base_model_path)
        _base_project_to_dim = _base_cfg.backbone_cfg.get("project_to_dim") or 2048
        _n_group = backbone_cfg_overrides.get("n_group", _base_cfg.backbone_cfg.get("n_group", 8))
        _new_project_to_dim = _base_project_to_dim * config.equi_scale_factor
        assert _new_project_to_dim % _n_group == 0, (
            f"project_to_dim ({_new_project_to_dim}) must be divisible by n_group ({_n_group})"
        )
        backbone_cfg_overrides["project_to_dim"] = _new_project_to_dim
        print(f"equi_scale_factor={config.equi_scale_factor}: project_to_dim {_base_project_to_dim} → {_new_project_to_dim} (blocks {_base_project_to_dim//_n_group} → {_new_project_to_dim//_n_group})")

    # Load model — backbone_cfg_overrides are merged into backbone_cfg before construction
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
        tune_inv_dit=config.tune_inv_dit,  # FA hybrid: whether inv_dit adapts to FA encoder outputs
        load_backbone_only=True,  # load backbone only, not the action head
        backbone_cfg_overrides=backbone_cfg_overrides,
        rot_aug=config.rot_aug,
    )

    # Persist rotation config + n_group in saved config so inference reconstructs correctly
    model.config.backbone_cfg.update(backbone_cfg_overrides)
    model.config.backbone_cfg["n_group"] = model.backbone.n_group

    # Update action_horizon and num_hand to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    action_horizon_changed = data_action_horizon != model.action_head.config.action_horizon
    num_hand_changed = data_num_hand != model.action_head.config.num_hand
    rot_type_changed = data_rot_type != model.action_head.config.rot_type
    rel_action_changed = data_rel_action != model.action_head.config.rel_action
    new_project_to_dim = backbone_cfg_overrides.get("project_to_dim", None)
    cross_attn_dim_changed = (
        new_project_to_dim is not None
        and new_project_to_dim != model.action_head.config.diffusion_model_cfg.get("cross_attention_dim")
    )

    if action_horizon_changed or num_hand_changed or rot_type_changed or rel_action_changed or cross_attn_dim_changed:
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {model.action_head.config.action_horizon}), "
            f"num_hand {data_num_hand} (was {model.action_head.config.num_hand}), "
            f"rot_type {data_rot_type} (was {model.action_head.config.rot_type}), "
            f"rel_action {data_rel_action} (was {model.action_head.config.rel_action}), "
        )

        # Update the action head config
        new_action_head_config = model.action_head.config
        new_action_head_config.action_horizon = data_action_horizon
        new_action_head_config.num_hand = data_num_hand
        new_action_head_config.rot_type = data_rot_type
        new_action_head_config.rel_action = data_rel_action
        if cross_attn_dim_changed:
            old_cross_attn = new_action_head_config.diffusion_model_cfg.get("cross_attention_dim")
            new_action_head_config.diffusion_model_cfg["cross_attention_dim"] = new_project_to_dim
            print(f"  cross_attention_dim {old_cross_attn} → {new_project_to_dim}")

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy weights from the old action head, skipping any size-mismatched params
        # (e.g. equi_vis_self_attn layers change shape when cross_attention_dim changes)
        old_sd = model.action_head.state_dict()
        new_sd = new_action_head.state_dict()
        compatible_sd = {
            k: v for k, v in old_sd.items()
            if k in new_sd and v.shape == new_sd[k].shape
        }
        skipped = [k for k in old_sd if k not in compatible_sd]
        if skipped:
            print(f"  Skipping {len(skipped)} mismatched weight(s) (shape changed): {skipped[:3]}{'...' if len(skipped) > 3 else ''}")
        new_action_head.load_state_dict(compatible_sd, strict=False)

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["num_hand"] = new_action_head_config.num_hand
        model.config.action_head_cfg["rot_type"] = new_action_head_config.rot_type
        model.config.action_head_cfg["rel_action"] = new_action_head_config.rel_action
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        if cross_attn_dim_changed:
            model.config.action_head_cfg["diffusion_model_cfg"]["cross_attention_dim"] = new_project_to_dim

        # Set trainable parameters for the new action head
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )
    model.action_head.set_trainable_parameters(
        tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
    )
    # print_layer_weights(model.action_head.state_encoder.layer1.layers[31])
    # print_layer_weights(model.action_head.action_encoder.W1.layers[31])
    # print_layer_weights(model.action_head.action_decoder.layer1.layers[31])
    # print_layer_weights(model.action_head.model.proj_out_1)

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )
    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=True,  # Required for equivariant models with unused parameters
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
        adapter_warmup_steps=config.adapter_warmup_steps,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            script_path = Path(__file__).absolute()

            # Use subprocess.run instead of os.system
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
