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
  logit (B, 1)  ──  BCEWithLogitsLoss

All W frames defined by delta_indices are concatenated into a single image
sequence and processed by Eagle in one forward pass, giving the model
cross-frame attention for free.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.model.backbone.eagle_backbone import EagleBackbone
from gr00t.model.action_head.task_completion import TaskCompletionDetector


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
    pos_weight : float | None
        Optional positive-class weight for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        backbone: EagleBackbone,
        seq_dim: int,
        hidden_dim: int,
        freeze_backbone: bool = True,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()

        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.task_completion_detection = TaskCompletionDetector(
            seq_dim=seq_dim,
            hidden_dim=hidden_dim,
        )

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

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
            "loss"                 : scalar BCE loss (only when labels present)
            "logits"               : (B, 1)
            "task_completion_pred" : (B,) sigmoid probability
        """
        backbone_frozen = all(not p.requires_grad for p in self.backbone.parameters())
        ctx = torch.no_grad() if backbone_frozen else torch.enable_grad()

        with ctx:
            backbone_input = self.backbone.prepare_input(batch)
            backbone_out = self.backbone(backbone_input)

        # (B, N_tokens, D) — Eagle attended over all W frames' tokens jointly
        features = backbone_out.backbone_features.float()

        logits = self.task_completion_detection(features)  # (B, 1)

        out = {
            "logits": logits,
            "task_completion_pred": F.sigmoid(logits.squeeze(-1)),
        }

        if "task_completion" in batch:
            labels = batch["task_completion"].float().to(logits.device)
            out["loss"] = self.loss_fn(logits, labels.squeeze(-1))

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
        pos_weight: Optional[float] = None,
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
            pos_weight=pos_weight,
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
