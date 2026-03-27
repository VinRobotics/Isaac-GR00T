"""
Window-Frame Task Completion Fine-Tuning
=========================================
Trains the TaskCompletionDetector using frames sampled at user-specified
temporal offsets (delta_indices), each processed independently through the
frozen EagleBackbone.

This exactly matches the inference behaviour in WindowTaskCompletionModel /
TaskCompletionDetector (window buffer) — no training/inference mismatch.

Usage
-----
Single GPU — 3 frames with stride 5:

    python scripts/finetune_task_completion_window.py \\
        --dataset_path /path/to/dataset \\
        --model_path nvidia/GR00T-N1.5-3B \\
        --output_dir /tmp/tc_window \\
        --delta_indices -10 -5 0 \\
        --max_steps 3000

Single frame (original behaviour):

    python scripts/finetune_task_completion_window.py \\
        --dataset_path /path/to/dataset \\
        --delta_indices 0

Multi-GPU (torchrun):

    torchrun --standalone --nproc_per_node=4 \\
        scripts/finetune_task_completion_window.py \\
        --dataset_path /path/to/dataset \\
        --delta_indices -10 -5 0 \\
        --num_gpus 4 \\
        --max_steps 3000

After training, load the detector weights at inference:

    policy.model.set_task_completion_window_size(3)   # len(delta_indices)
    policy.model.task_completion_detection.load_state_dict(
        torch.load("/tmp/tc_window/task_completion_detection.pt")
    )
"""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import torch
import tyro
from transformers import TrainingArguments, Trainer, set_seed

from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, DefaultDataCollator
from gr00t.task_completion.config import WindowTaskCompletionConfig
from gr00t.task_completion.dataset import WindowFrameTaskCompletionDataset
from gr00t.task_completion.model import WindowTaskCompletionModel


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

@dataclass
class ArgsConfig:
    dataset_path: List[str]
    """One or more paths to LeRobot dataset directories."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """HuggingFace model ID or local path to a GR00T-N1.5 checkpoint."""

    output_dir: str = "/tmp/tc_window"
    """Directory to save checkpoints and final detector weights."""

    delta_indices: List[int] = field(default_factory=lambda: [0])
    """Temporal offsets (relative to current step) to sample per training example.
    All values should be <= 0 (past frames); the last entry should be 0.

    Examples:
      --delta_indices 0              single frame (no window)
      --delta_indices -4 -3 -2 -1 0  five consecutive frames
      --delta_indices -10 -5 0       three frames with stride 5
    """

    video_keys: Optional[List[str]] = None
    """Camera keys. If not set, defaults to VRH3 three-camera setup."""

    language_key: str = "annotation.human.task_description"
    task_completion_key: str = "observation.tasks.done"

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchvision_av"

    batch_size: int = 8
    """Per-GPU batch size. All W frames are packed into one Eagle call;
    reduce if GPU memory is tight with many delta_indices."""

    max_steps: int = 3000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    dataloader_num_workers: int = 4
    report_to: Literal["wandb", "tensorboard", "azure_ml", "none"] = "tensorboard"
    seed: int = 42
    resume: bool = False
    num_gpus: int = 1

    success_pos_weight: float = 9.0
    """Upweight success (done=1) labels. Set > 1 when failures dominate."""

    seq_dim: int = 2048
    """Backbone output projection dim (must match model_path's project_to_dim)."""

    hidden_dim: int = 1024
    """Hidden dim for the TaskCompletionDetector MLP."""

    detector_init_path: Optional[str] = None
    """Optional path to a task_completion_detection.pt to warm-start from."""


# ---------------------------------------------------------------------------
# Custom Trainer
# ---------------------------------------------------------------------------

class WindowTCTrainer(Trainer):
    """Thin wrapper so HuggingFace Trainer calls WindowTaskCompletionModel.forward."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: ArgsConfig):
    set_seed(args.seed)

    delta_indices = args.delta_indices
    video_keys = args.video_keys or [
        "video.cam_head",
        "video.cam_left",
        "video.cam_right",
    ]

    # --- Data config & transform ---
    data_cfg = WindowTaskCompletionConfig(
        delta_indices=delta_indices,
        video_keys=video_keys,
        language_key=args.language_key,
        task_completion_key=args.task_completion_key,
    )
    train_transform = data_cfg.transform(training=True)

    # --- Dataset(s) ---
    datasets = []
    for path in args.dataset_path:
        assert os.path.exists(path), f"Dataset path does not exist: {path}"
        datasets.append(
            WindowFrameTaskCompletionDataset(
                dataset_path=path,
                video_keys=video_keys,
                language_key=args.language_key,
                task_completion_key=args.task_completion_key,
                delta_indices=delta_indices,
                transforms=train_transform,
                embodiment_tag=args.embodiment_tag,
                video_backend=args.video_backend,
            )
        )

    if len(datasets) == 1:
        train_dataset = datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(datasets)

    collate_fn = DefaultDataCollator()

    # --- Model ---
    pos_weight = args.success_pos_weight if args.success_pos_weight != 1.0 else None
    model = WindowTaskCompletionModel.from_groot_pretrained(
        model_path=args.model_path,
        seq_dim=args.seq_dim,
        hidden_dim=args.hidden_dim,
        freeze_backbone=True,
        pos_weight=pos_weight,
    )

    if args.detector_init_path is not None:
        model.load_detector_weights(args.detector_init_path)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    print(f"delta_indices: {delta_indices}  (window_size={len(delta_indices)})")

    # --- HuggingFace TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=args.report_to,
        seed=args.seed,
        do_eval=False,
        ddp_find_unused_parameters=False,
    )

    trainer = WindowTCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    trainer.train(resume_from_checkpoint=args.resume or None)

    # --- Save detector weights ---
    output_path = Path(args.output_dir) / "task_completion_detection.pt"
    model.save_detector_weights(output_path)

    print("\nDone. To use at inference:")
    print(f"  policy.model.set_task_completion_window_size({len(delta_indices)})")
    print(f"  policy.model.task_completion_detection.load_state_dict(")
    print(f'      torch.load("{output_path}")')
    print(f"  )")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(ArgsConfig)

    print("\n" + "=" * 60)
    print("WINDOW TASK COMPLETION FINE-TUNING CONFIG:")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert args.num_gpus <= available_gpus, (
        f"Requested {args.num_gpus} GPUs but only {available_gpus} available"
    )
    assert args.num_gpus >= 1

    if args.num_gpus == 1:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        main(args)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(args)
        else:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={args.num_gpus}",
                "--nnodes=1",
                str(Path(__file__).absolute()),
                *sys.argv[1:],
            ]
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
