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

"""VLM-backbone motion-keypoint head (see method_motion_keypoint_ot_v2.md).

Predicts future 2D object-keypoint positions from the post-backbone hidden
states of the VLM backbone's motion-query tokens (gr00t/model/modules/
qwen3_backbone.py's motion_query_tokens) — the successor to the old
Action-Head-side keypoint head's "tokens" mode, relocated to the backbone side
so the same learned embedding also serves as the space for OT alignment
between human and robot samples (gr00t/model/modules/optimal_transport.py).

A pure feedforward readout: one forward pass predicts the whole future
window directly, unlike the old Action-Head design, which needed a
flow-matching denoising rollout to read out keypoints. Reuses the t=0-anchor
point-identity mechanism that avoids the by-index-regression collapse (see
gr00t/model/gr00t_n1d7/motion_head.md's analysis) — identity is bound
through the decoder's anchor conditioning, not the query embedding itself.
"""

import torch
from torch import nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config


class MotionHead(nn.Module):
    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.motion_horizon = config.motion_horizon
        hidden = config.backbone_embedding_dim
        self.motion_coord_encoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.motion_position_decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, (config.motion_horizon - 1) * 2),
        )

    def _anchor_from_input(
        self,
        motion_input: BatchFeature,
        batch_size: int,
        num_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """t=0 positions of the selected points, [B, num_points, 2] — tracker
        data the sample already carries. Zeros when the batch has no keypoint
        fields at all, purely to keep the decoder in the DDP graph."""
        target_kp = motion_input.get("keypoint_target", None)
        if target_kp is None:
            return torch.zeros(batch_size, num_points, 2, device=device, dtype=dtype)
        target_kp = target_kp.view(batch_size, self.motion_horizon, num_points, 2)
        return target_kp[:, 0].to(device=device, dtype=dtype)

    def _decode(self, motion_token_features: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Returns [B, motion_horizon - 1, P, 2] predicted FUTURE positions
        (t=1..H-1) — absolute, or relative to the anchor when
        config.motion_relative."""
        batch_size, num_points = motion_token_features.shape[:2]
        anchor_emb = self.motion_coord_encoder(anchor.to(motion_token_features.dtype))
        decoded = self.motion_position_decoder(
            torch.cat((motion_token_features, anchor_emb), dim=-1)
        )
        decoded = decoded.view(batch_size, num_points, self.motion_horizon - 1, 2)
        return decoded.permute(0, 2, 1, 3)

    def _sample_weight(
        self, batch_size: int, motion_input: BatchFeature, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """has_keypoint (per-sample motion-data availability) x loss_weight
        (e.g. MotionTrans alpha co-training re-weighting)."""
        has_keypoint = motion_input.get("has_keypoint", None)
        if has_keypoint is None:
            sample_weight = torch.ones(batch_size, device=device, dtype=dtype)
        else:
            sample_weight = has_keypoint.to(dtype=dtype, device=device).view(batch_size)
        loss_weight = motion_input.get("loss_weight", None)
        if loss_weight is not None:
            sample_weight = sample_weight * loss_weight.to(dtype=dtype, device=device).view(
                batch_size
            )
        return sample_weight

    def _compute_position_loss(
        self, pred: torch.Tensor, motion_input: BatchFeature
    ) -> torch.Tensor:
        """Masked Huber position-regression loss against keypoint_target /
        keypoint_active_target, by-index (well posed since each point's
        target is unique given its own anchor — no exchangeable-target
        collapse, no Hungarian matching needed)."""
        target_kp = motion_input.get("keypoint_target", None)
        valid = motion_input.get("keypoint_active_target", None)
        if target_kp is None or valid is None:
            # Keep the decoder in the graph (DDP runs with
            # find_unused_parameters=False) while contributing nothing.
            return pred.sum() * 0.0

        batch_size, _, num_points = pred.shape[:3]
        full_horizon = self.motion_horizon
        target_kp = target_kp.view(batch_size, full_horizon, num_points, 2).to(pred.dtype)
        valid = valid.view(batch_size, full_horizon, -1).to(pred.dtype)
        if valid.shape[-1] != num_points:
            # Per-object mask -> per-point (num_points is a multiple of num_objects).
            valid = valid.repeat_interleave(num_points // valid.shape[-1], dim=-1)

        anchor = target_kp[:, :1]  # [B, 1, P, 2] — the decoder's input
        target_kp = target_kp[:, 1:]
        valid = valid[:, 1:]
        if self.config.motion_relative:
            target_kp = target_kp - anchor

        sample_weight = self._sample_weight(batch_size, motion_input, pred.device, pred.dtype)

        diff = pred - target_kp
        cost = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=0.1, reduction="none").sum(-1)
        # Supervise keypoints only where the tracker is confident (valid == 1);
        # motion_static_weight > 0 re-enables down-weighted supervision on
        # padding slots (zeros).
        step_weight = valid + (1.0 - valid) * self.config.motion_static_weight
        num = (cost * step_weight).sum(dim=(1, 2))
        den = step_weight.sum(dim=(1, 2))
        return (num * sample_weight).sum() / ((den * sample_weight).sum() + 1e-6)

    def pool(self, motion_token_features: torch.Tensor) -> torch.Tensor:
        """Mean-pool the motion tokens into one per-sample feature vector,
        the space the OT alignment loss operates in (config.motion_pool)."""
        if self.config.motion_pool == "mean":
            return motion_token_features.mean(dim=1)
        raise ValueError(f"Unsupported motion_pool: {self.config.motion_pool!r}")

    def forward(self, motion_token_features: torch.Tensor, motion_input: BatchFeature) -> dict:
        """
        Args:
            motion_token_features: [B, P, backbone_embedding_dim] post-backbone
                hidden states of the motion query tokens.
            motion_input: carries keypoint_target / keypoint_active_target /
                has_keypoint / loss_weight (see Gr00tN1d7Processor).

        Returns a dict with:
            motion_loss: scalar Huber position loss.
            motion_pred: [B, motion_horizon, P, 1, 2] absolute positions with
                the anchor prepended as step 0 — the same contract
                keypoint_viz.render_keypoint_overlay expects (each token
                reported as its own single-point group).
            motion_pooled_features: [B, backbone_embedding_dim], the OT
                alignment loss's input space.
        """
        batch_size, num_points = motion_token_features.shape[:2]
        device, dtype = motion_token_features.device, motion_token_features.dtype
        anchor = self._anchor_from_input(motion_input, batch_size, num_points, device, dtype)

        pred = self._decode(motion_token_features, anchor)
        loss = self._compute_position_loss(pred, motion_input)

        pred_abs = pred + anchor.unsqueeze(1) if self.config.motion_relative else pred
        pred_full = torch.cat((anchor.unsqueeze(1).to(pred_abs.dtype), pred_abs), dim=1)

        return {
            "motion_loss": loss,
            "motion_pred": pred_full.unsqueeze(3),
            "motion_pooled_features": self.pool(motion_token_features),
        }
