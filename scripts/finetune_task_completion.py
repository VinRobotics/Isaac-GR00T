"""
Finetune only the task_completion_detection head of GR00T.

This script:
  1. Loads the full GR00T model.
  2. Freezes all parameters except `action_head.task_completion_detection`.
  3. Trains the detector using CrossEntropyLoss on datasets with task-completion labels (0=doing, 1=success, 2=failure).
  4. After training, saves ONLY the task_completion_detection state dict to
     `<output_dir>/task_completion_detection.pt`.

Usage example:
    python scripts/finetune_task_completion.py \\
        --dataset_path /path/to/dataset \\
        --output_dir /tmp/task_completion \\
        --data_config vrh3_two_hand_task_completion \\
        --base_model_path nvidia/GR00T-N1.5-3B \\
        --max_steps 2000 \\
        --batch_size 16

Then at inference pass `--task_completion_detection_path /tmp/task_completion/task_completion_detection.pt`
to inference_service.py.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING


@dataclass
class ArgsConfig:
    """Configuration for task_completion_detection fine-tuning."""

    dataset_path: List[str]
    """Path(s) to the dataset directory. All datasets must share the same data config."""

    output_dir: str = "/tmp/task_completion"
    """Directory to save model checkpoints and the final task_completion_detection.pt."""

    data_config: str = "vrh3_two_hand_task_completion"
    """Data config name. Must be a config that has use_task_completion=True."""

    batch_size: int = 16
    """Batch size per GPU."""

    max_steps: int = 2000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use."""

    save_steps: int = 500
    """Steps between saving checkpoints."""

    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    learning_rate: float = 1e-4
    """Learning rate."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW."""

    warmup_ratio: float = 0.05
    """Warmup ratio."""

    dataloader_num_workers: int = 4
    """Number of dataloader workers per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps."""

    report_to: Literal["wandb", "tensorboard", "azure_ml", "none"] = "tensorboard"
    """Where to report metrics."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag."""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchvision_av"
    """Video backend."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    task_completion_loss_weight: float = 1.0
    """Scale factor applied to the task completion CrossEntropy loss."""

    class_weight: Optional[list] = None
    """Per-class weights for CrossEntropyLoss, length 3: [doing, success, failure].
    Set higher to upweight underrepresented classes.
    e.g. if doing dominates, try class_weight=[0.1, 1.0, 1.0]."""


#####################################################################################


def main(config: ArgsConfig):
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # Build dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            single_datasets.append(
                LeRobotSingleDataset(
                    dataset_path=p,
                    modality_configs=modality_configs,
                    transforms=transforms,
                    embodiment_tag=embodiment_tag,
                    video_backend=config.video_backend,
                )
            )
        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(ds, 1.0) for ds in single_datasets],
            mode="train",
            balance_dataset_weights=True,
            balance_trajectory_weights=True,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
    )

    # Skip diffusion forward pass to save compute — only task_completion loss is needed
    model.task_completion_only = True
    model.task_completion_loss_weight = config.task_completion_loss_weight

    # Replace loss with class-weighted version to handle class imbalance
    if config.class_weight is not None:
        cw = torch.tensor(config.class_weight)
        model.task_completion_detection_loss = torch.nn.CrossEntropyLoss(weight=cw)
        print(f"Using class_weight={config.class_weight} for [doing, success, failure]")

    # Freeze everything, then unfreeze only task_completion_detection
    for p in model.parameters():
        p.requires_grad = False
    for p in model.task_completion_detection.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

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
        dataloader_prefetch_factor=2 if config.dataloader_num_workers > 0 else None,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
    )

    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    experiment.train()

    # Save only task_completion_detection weights
    output_path = Path(config.output_dir) / "task_completion_detection.pt"
    torch.save(
        model.task_completion_detection.state_dict(),
        output_path,
    )
    print(f"\nSaved task_completion_detection weights to: {output_path}")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)

    print("\n" + "=" * 50)
    print("TASK COMPLETION DETECTION FINE-TUNING CONFIG:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, (
        f"Requested {config.num_gpus} GPUs but only {available_gpus} available"
    )
    assert config.num_gpus > 0

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
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
                *sys.argv[1:],
            ]
            print("Running torchrun command:", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
