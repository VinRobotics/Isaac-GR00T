"""
WindowTaskCompletionModel
==========================
Standalone model for window-frame task completion training.

Architecture (Option B — single Eagle call)
--------------------------------------------
  frozen EagleBackbone  (all W frames packed into one conversation)
        ↓  (B, N_all_tokens, D)  — Eagle attends across all frames jointly
  TaskCompletionDetector  (trainable)
        ↓
  logits (B, 3)  ──  CrossEntropyLoss
  classes: 0=doing, 1=success, 2=failure

All W frames defined by delta_indices are concatenated into a single image
sequence and processed by Eagle in one forward pass, giving the model
cross-frame attention for free.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.model.backbone.eagle_backbone import EagleBackbone
from gr00t.model.action_head.task_completion import TaskCompletionDetector


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    For fine-grained tasks where success/failure look nearly identical
    (e.g. part in holes vs. slightly off), the model tends to make
    confident wrong predictions on failure.  Focal loss suppresses the
    gradient from easy/confident correct predictions and amplifies it
    for hard mis-classified examples.

    FL(p_t) = -w_t * (1 - p_t)^gamma * log(p_t)
    gamma=0 → weighted CrossEntropy; gamma=2 is the standard value.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        # Use register_buffer so the weight moves with .to(device)
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


class WindowTaskCompletionModel(nn.Module):
    """
    Parameters
    ----------
    backbone : EagleBackbone
        Already constructed backbone. Frozen by default.
    seq_dim : int
        Backbone output embedding dimension.
    hidden_dim : int
        Hidden dim for the TaskCompletionDetector MLP.
    freeze_backbone : bool
        If True (default) backbone parameters are frozen.
    class_weight : list[float] | None
        Optional per-class weights for CrossEntropyLoss (length 3).
        Order: [doing, success, failure].
    """

    def __init__(
        self,
        backbone: EagleBackbone,
        seq_dim: int,
        hidden_dim: int,
        freeze_backbone: bool = True,
        class_weight: Optional[list] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        dropout: float = 0.3,
        num_frames: int = 5,
        num_cameras: int = 2,
    ):
        super().__init__()

        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.task_completion_detection = TaskCompletionDetector(
            seq_dim=seq_dim,
            hidden_dim=hidden_dim,
            num_classes=3,
            dropout=dropout,
            num_frames=num_frames,
            num_cameras=num_cameras,
        )

        cw = torch.tensor(class_weight, dtype=torch.float32) if class_weight is not None else None
        if use_focal_loss:
            self.loss_fn = FocalLoss(weight=cw, gamma=focal_gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=cw)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """
        Args
        ----
        batch : dict
            Collated output from DefaultDataCollator.  All W frames are
            already packed into the eagle_* tensors by the transform
            pipeline (ConcatTransform + GR00TTransform).
            Optionally contains "task_completion" (B, 1) for training.

        Returns
        -------
        dict with keys:
            "loss"                 : scalar CrossEntropy loss (only when labels present)
            "logits"               : (B, 3)
            "task_completion_pred" : (B, 3) softmax probabilities (0=doing, 1=success, 2=failure)
        """
        backbone_frozen = all(not p.requires_grad for p in self.backbone.parameters())
        ctx = torch.no_grad()

        with ctx:
            backbone_input = self.backbone.prepare_input(batch)
            backbone_out = self.backbone(backbone_input)

        # (B, N_tokens, D) — Eagle attended over all W frames' tokens jointly
        features = backbone_out.backbone_features.float()

        logits = self.task_completion_detection(features)  # (B, 3)

        out = {
            "logits": logits,
            "task_completion_pred": F.softmax(logits, dim=-1),
        }

        if "task_completion" in batch:
            labels = batch["task_completion"].long().to(logits.device).squeeze(-1)
            out["loss"] = self.loss_fn(logits, labels)

        return out

    # ------------------------------------------------------------------
    # Constructors / IO
    # ------------------------------------------------------------------

    @classmethod
    def from_groot_pretrained(
        cls,
        model_path: str,
        seq_dim: int = 1536,
        hidden_dim: int = 1024,
        freeze_backbone: bool = True,
        class_weight: Optional[list] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        dropout: float = 0.3,
        num_frames: int = 5,
        num_cameras: int = 2,
    ) -> "WindowTaskCompletionModel":
        """
        Load only the EagleBackbone from a GR00T-N1.5 checkpoint.
        The action head and DiT are discarded to save memory and time.
        """
        from gr00t.model.gr00t_n1 import GR00T_N1_5

        print(f"Loading GR00T backbone from: {model_path}")
        groot = GR00T_N1_5.from_pretrained(
            model_path,
            tune_llm=False,
            tune_visual=False,
            tune_projector=False,
            tune_diffusion_model=False,
        )
        backbone = groot.backbone
        del groot

        return cls(
            backbone=backbone,
            seq_dim=seq_dim,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            class_weight=class_weight,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            dropout=dropout,
            num_frames=num_frames,
            num_cameras=num_cameras,
        )

    def load_detector_weights(self, path: str | Path):
        """Load previously saved TaskCompletionDetector weights."""
        state = torch.load(path, map_location="cpu")
        self.task_completion_detection.load_state_dict(state)
        print(f"Loaded task_completion_detection weights from: {path}")

    def save_detector_weights(self, path: str | Path):
        """Save only the TaskCompletionDetector weights."""
        torch.save(self.task_completion_detection.state_dict(), path)
        print(f"Saved task_completion_detection weights to: {path}")

    @staticmethod
    def extract_detector_weights_from_checkpoint(
        checkpoint_dir: str | Path,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Extract TaskCompletionDetector weights from a Trainer checkpoint.

        The HuggingFace Trainer saves the full model as ``model.safetensors``
        with keys prefixed by module name (e.g. ``task_completion_detection.fc1.weight``).
        This function filters those keys, strips the prefix, and writes a
        ``task_completion_detection.pt`` file that ``load_detector_weights`` can consume.

        Parameters
        ----------
        checkpoint_dir:
            Directory containing ``model.safetensors`` (a Trainer checkpoint or
            the final output directory).
        output_path:
            Where to write the ``.pt`` file.  Defaults to
            ``<checkpoint_dir>/task_completion_detection.pt``.

        Returns
        -------
        Path
            Absolute path of the written ``.pt`` file.
        """
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "safetensors is required. Install with: pip install safetensors"
            ) from e

        checkpoint_dir = Path(checkpoint_dir)
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in: {checkpoint_dir}")

        if output_path is None:
            output_path = checkpoint_dir / "task_completion_detection.pt"
        output_path = Path(output_path)

        prefix = "task_completion_detection."
        full_state = load_file(safetensors_path, device="cpu")
        detector_state = {
            k[len(prefix):]: v
            for k, v in full_state.items()
            if k.startswith(prefix)
        }

        if not detector_state:
            raise ValueError(
                f"No keys with prefix '{prefix}' found in {safetensors_path}. "
                "Make sure the checkpoint was saved from WindowTaskCompletionModel."
            )

        torch.save(detector_state, output_path)
        print(f"Extracted {len(detector_state)} tensors → {output_path}")
        return output_path
