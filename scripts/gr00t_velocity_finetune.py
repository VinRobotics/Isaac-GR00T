"""
GR00T Velocity Adapter Fine-tuning Script

This script implements two-stage training for the velocity adapter:
- Stage 1: Train only the velocity decoder while freezing the position head
- Stage 2: Joint fine-tuning of both position and velocity heads

The velocity adapter enables PD-complete action chunks (position + velocity)
for smoother execution under VLASH-style latency and quantization.
"""

import copy
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
from gr00t.data.transform import BSplineVelocityTransform
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING


@dataclass
class VelocityAdapterConfig:
    """Configuration for GR00T Velocity Adapter fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories."""

    output_dir: str = "/tmp/gr00t_velocity"
    """Directory to save model checkpoints."""

    data_config: str = "fourier_gr1_arms_only"
    """Data configuration to use for training."""

    # Training stage parameters
    training_stage: Literal[1, 2] = 1
    """Training stage: 1 = velocity adapter only (position frozen), 2 = joint fine-tuning."""

    stage1_checkpoint: str = ""
    """Path to stage 1 checkpoint for stage 2 training. Required when training_stage=2."""

    # Velocity head parameters
    use_velocity_head: bool = True
    """Whether to enable the velocity head. Should be True for this script."""

    lambda_vel: float = 1.0
    """Weight for velocity flow-matching loss."""

    lambda_consistency: float = 0.1
    """Weight for position-velocity consistency loss."""

    bspline_smoothing: float = 0.0
    """Smoothing factor for B-spline velocity computation. 0.0 = interpolating, >0 = smoothing."""

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
    """Whether to fine-tune the projector (stage 2 only)."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model (stage 2 only)."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics."""

    # Data loading parameters
    embodiment_tag: str = "new_embodiment"
    """Embodiment tag to use for training."""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend to use for training."""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """If True, balance the dataset weights by multiplying the total trajectory to each dataset."""

    balance_trajectory_weights: bool = True
    """If True, sample trajectories within a dataset weighted by their length."""


def add_velocity_transform(transforms, config: VelocityAdapterConfig):
    """
    Add BSplineVelocityTransform to the transform pipeline.
    This should be inserted before the GR00TTransform.
    """
    from gr00t.model.transforms import GR00TTransform
    
    # Check if transforms is a ComposedModalityTransform
    if not hasattr(transforms, 'transforms'):
        raise ValueError("Expected ComposedModalityTransform with 'transforms' attribute")
    
    # Find the GR00TTransform and get the action keys from previous transforms
    groot_transform_idx = None
    action_keys = []
    
    for idx, transform in enumerate(transforms.transforms):
        if isinstance(transform, GR00TTransform):
            groot_transform_idx = idx
            break
        # Collect action keys from transforms that have them
        if hasattr(transform, 'apply_to'):
            for key in transform.apply_to:
                if 'action' in key.lower():
                    action_keys.append(key)
    
    if groot_transform_idx is None:
        raise ValueError("GR00TTransform not found in transform pipeline")
    
    # Create velocity transform
    # Note: The BSplineVelocityTransform expects the concatenated 'action' key
    # which is created by ConcatTransform before GR00TTransform
    velocity_transform = BSplineVelocityTransform(
        apply_to=["action"],  # Apply to concatenated action
        smoothing_factor=config.bspline_smoothing,
        spline_degree=3,  # Cubic B-spline
        dt=1.0,  # Normalized time step
        output_velocity_key="velocity",
    )
    
    # Insert velocity transform before GR00TTransform
    transforms.transforms.insert(groot_transform_idx, velocity_transform)
    
    # Update GR00TTransform to handle velocity
    groot_transform = transforms.transforms[groot_transform_idx + 1]
    if isinstance(groot_transform, GR00TTransform):
        groot_transform.use_velocity = True
        groot_transform.max_velocity_dim = groot_transform.max_action_dim
        print(f"Added BSplineVelocityTransform with smoothing_factor={config.bspline_smoothing}")
        print(f"Updated GR00TTransform: use_velocity=True, max_velocity_dim={groot_transform.max_velocity_dim}")
    
    return transforms


def main(config: VelocityAdapterConfig):
    """Main training function for velocity adapter."""
    
    print("\n" + "=" * 60)
    print(f"VELOCITY ADAPTER TRAINING - STAGE {config.training_stage}")
    print("=" * 60)
    
    if config.training_stage == 2 and not config.stage1_checkpoint:
        raise ValueError("stage1_checkpoint is required for stage 2 training")
    
    # ------------ Step 1: Load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 Modality configs and transforms
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    
    # Add velocity transform to the pipeline
    transforms = add_velocity_transform(transforms, config)  # type: ignore

    # 1.2 Data loader
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,  # type: ignore
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,  # type: ignore
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(dataset, 1.0) for dataset in single_datasets],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
        print(f"Loaded {len(single_datasets)} datasets")

    # ------------ Step 2: Load model ------------
    data_action_horizon = len(getattr(data_config_cls, 'action_indices', list(range(16))))
    
    # Get max_action_dim from GR00TTransform
    from gr00t.model.transforms import GR00TTransform
    last_transform = transforms.transforms[-1]  # type: ignore
    assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"
    data_max_action_dim = last_transform.max_action_dim

    # Load base model or stage 1 checkpoint
    if config.training_stage == 1:
        model_path = config.base_model_path
        print(f"Loading base model from: {model_path}")
    else:
        model_path = config.stage1_checkpoint
        print(f"Loading stage 1 checkpoint from: {model_path}")

    # Check if flash_attn is available (not a stub)
    try:
        import flash_attn
        # Check if it's a real flash_attn (version > 0.0.0)
        if hasattr(flash_attn, '__version__') and flash_attn.__version__ != "0.0.0":
            attn_impl = "flash_attention_2"
        else:
            print("flash_attn stub detected, using PyTorch SDPA attention")
            attn_impl = "sdpa"
    except ImportError:
        print("flash_attn not available, using PyTorch SDPA attention")
        attn_impl = "sdpa"

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector if config.training_stage == 2 else False,
        tune_diffusion_model=config.tune_diffusion_model if config.training_stage == 2 else False,
        attn_implementation=attn_impl,
    )

    # ------------ Step 3: Configure velocity head ------------
    from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
    
    # Update action head config for velocity
    new_action_head_config = copy.deepcopy(model.action_head.config)
    new_action_head_config.use_velocity_head = config.use_velocity_head
    new_action_head_config.velocity_dim = data_max_action_dim
    new_action_head_config.lambda_vel = config.lambda_vel
    new_action_head_config.lambda_consistency = config.lambda_consistency
    new_action_head_config.freeze_position_head = (config.training_stage == 1)
    
    # Handle action dimension changes
    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim
    
    if action_horizon_mismatch:
        new_action_head_config.action_horizon = data_action_horizon
        print(f"Updating action_horizon: {model.action_head.config.action_horizon} -> {data_action_horizon}")
    
    if action_dim_mismatch:
        new_action_head_config.action_dim = data_max_action_dim
        print(f"Updating action_dim: {model.action_head.config.action_dim} -> {data_max_action_dim}")
    
    # Create new action head with velocity support
    old_state_dict = model.action_head.state_dict()
    new_action_head = FlowmatchingActionHead(new_action_head_config)
    
    # Load weights (velocity decoder will be randomly initialized)
    missing_keys, unexpected_keys = new_action_head.load_state_dict(old_state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys (will be randomly initialized): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys (ignored): {unexpected_keys}")
    
    # Replace action head
    model.action_head = new_action_head

    # Update model config
    model.config.action_horizon = data_action_horizon
    model.action_horizon = data_action_horizon
    model.config.action_dim = data_max_action_dim
    model.action_dim = data_max_action_dim
    model.config.use_velocity_head = config.use_velocity_head
    model.use_velocity_head = config.use_velocity_head
    model.config.velocity_dim = data_max_action_dim
    model.velocity_dim = data_max_action_dim
    model.config.action_head_cfg["action_horizon"] = data_action_horizon
    model.config.action_head_cfg["action_dim"] = data_max_action_dim
    model.config.action_head_cfg["use_velocity_head"] = config.use_velocity_head
    model.config.action_head_cfg["velocity_dim"] = data_max_action_dim
    model.config.action_head_cfg["lambda_vel"] = config.lambda_vel
    model.config.action_head_cfg["lambda_consistency"] = config.lambda_consistency
    model.config.action_head_cfg["freeze_position_head"] = (config.training_stage == 1)

    # Set trainable parameters based on training stage
    model.action_head.set_trainable_parameters(
        tune_projector=config.tune_projector if config.training_stage == 2 else False,
        tune_diffusion_model=config.tune_diffusion_model if config.training_stage == 2 else False,
    )

    # Set compute dtype
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # Print training configuration
    print("\n" + "-" * 40)
    print("TRAINING CONFIGURATION:")
    print("-" * 40)
    print(f"Training Stage: {config.training_stage}")
    print(f"Velocity Head Enabled: {config.use_velocity_head}")
    print(f"Lambda Velocity: {config.lambda_vel}")
    print(f"Lambda Consistency: {config.lambda_consistency}")
    print(f"B-spline Smoothing: {config.bspline_smoothing}")
    print(f"Freeze Position Head: {config.training_stage == 1}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    print("-" * 40 + "\n")

    # ------------ Step 4: Training arguments ------------
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
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # ------------ Step 5: Run training ------------
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    experiment.train()
    
    print("\n" + "=" * 60)
    print(f"STAGE {config.training_stage} TRAINING COMPLETE")
    print(f"Checkpoint saved to: {config.output_dir}")
    if config.training_stage == 1:
        print("\nNext steps:")
        print(f"  1. Run stage 2 training with:")
        print(f"     --training_stage 2 --stage1_checkpoint {config.output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    config = tyro.cli(VelocityAdapterConfig)

    print("\n" + "=" * 50)
    print("VELOCITY ADAPTER FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    assert config.num_gpus <= available_gpus, \
        f"Requested {config.num_gpus} GPUs but only {available_gpus} available"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
