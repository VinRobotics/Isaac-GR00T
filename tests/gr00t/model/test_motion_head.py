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

"""Test MotionHead: the VLM-backbone motion-keypoint head (see
gr00t/model/gr00t_n1d7/motion_head.py). Instantiated directly (no backbone
required) and fed synthetic motion-token hidden states."""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.motion_head import MotionHead
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=32,
        motion_horizon=4,
        max_motion_objects=2,
        motion_points_per_object=3,
        num_motion_tokens=4,
        motion_static_weight=0.0,
        motion_relative=False,
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


def _motion_input(batch_size, horizon, num_points, has_keypoint=1.0):
    keypoint_target = torch.randn(batch_size, horizon, num_points, 2)
    keypoint_active_target = torch.ones(batch_size, horizon, num_points)
    return BatchFeature(
        data={
            "keypoint_target": keypoint_target.view(batch_size, -1),
            "keypoint_active_target": keypoint_active_target,
            "has_keypoint": torch.full((batch_size,), has_keypoint),
        }
    )


def test_motion_head_output_shapes():
    """Default motion_pool="concat": lossless flatten of all num_motion_tokens
    slots, so motion_pooled_features is num_motion_tokens times wider than a
    single token's feature dim (unlike backbone_pooled_features, which must
    reduce a variable-length sequence to backbone_embedding_dim)."""
    config = _small_config()
    head = MotionHead(config)

    batch_size, num_points = 3, config.num_motion_tokens
    motion_token_features = torch.randn(batch_size, num_points, config.backbone_embedding_dim)
    motion_input = _motion_input(batch_size, config.motion_horizon, num_points)

    out = head(motion_token_features, motion_input)

    assert out["motion_loss"].shape == ()
    assert out["motion_pred"].shape == (batch_size, config.motion_horizon, num_points, 1, 2)
    assert out["motion_pooled_features"].shape == (
        batch_size,
        num_points * config.backbone_embedding_dim,
    )
    norms = out["motion_pooled_features"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_motion_head_mean_pool_ablation():
    """motion_pool="mean" is kept for backward compatibility / ablation
    against the pre-concat behavior — still averages down to a single
    token's feature dim."""
    config = _small_config(motion_pool="mean")
    head = MotionHead(config)

    batch_size, num_points = 3, config.num_motion_tokens
    motion_token_features = torch.randn(batch_size, num_points, config.backbone_embedding_dim)

    pooled = head.pool(motion_token_features)
    assert pooled.shape == (batch_size, config.backbone_embedding_dim)
    norms = pooled.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_motion_head_anchor_matches_target_step0():
    """The decoder must reproduce the anchor exactly at step 0 (it's the
    input, not a prediction) — this is what keeps the by-index loss well
    posed without Hungarian matching."""
    config = _small_config()
    head = MotionHead(config)

    batch_size, num_points = 2, config.num_motion_tokens
    motion_token_features = torch.randn(batch_size, num_points, config.backbone_embedding_dim)
    motion_input = _motion_input(batch_size, config.motion_horizon, num_points)

    out = head(motion_token_features, motion_input)
    anchor_from_target = motion_input["keypoint_target"].view(
        batch_size, config.motion_horizon, num_points, 2
    )[:, 0]
    step0 = out["motion_pred"][:, 0, :, 0, :]
    assert torch.allclose(step0, anchor_from_target)


def test_motion_head_loss_zero_without_keypoint_data():
    """No keypoint_target/keypoint_active_target in the batch (schema
    mismatch, not just has_keypoint=0) -> zero loss, decoder still in the
    autograd graph (DDP find_unused_parameters=False safety)."""
    config = _small_config()
    head = MotionHead(config)

    batch_size, num_points = 2, config.num_motion_tokens
    motion_token_features = torch.randn(
        batch_size, num_points, config.backbone_embedding_dim, requires_grad=True
    )
    empty_input = BatchFeature(data={})

    out = head(motion_token_features, empty_input)
    assert torch.isclose(out["motion_loss"], torch.tensor(0.0))
    out["motion_loss"].backward()
    assert motion_token_features.grad is not None


def test_motion_head_static_weight_downweights_invalid_points():
    """A point with keypoint_active_target=0 contributes motion_static_weight
    (default 0) instead of full weight to the loss."""
    config = _small_config(motion_static_weight=0.0)
    head = MotionHead(config)

    batch_size, num_points = 2, config.num_motion_tokens
    motion_token_features = torch.randn(batch_size, num_points, config.backbone_embedding_dim)
    motion_input = _motion_input(batch_size, config.motion_horizon, num_points)
    # Mark the last point as invalid (padding slot) for every step.
    motion_input["keypoint_active_target"][:, :, -1] = 0.0

    pred = head._decode(
        motion_token_features,
        head._anchor_from_input(
            motion_input,
            batch_size,
            num_points,
            motion_token_features.device,
            motion_token_features.dtype,
        ),
    )
    loss_with_invalid = head._compute_position_loss(pred, motion_input)

    # Zeroing out the invalid point's prediction error entirely shouldn't
    # change the loss when motion_static_weight=0 (it's fully masked out).
    pred_perturbed = pred.clone()
    pred_perturbed[:, :, -1] += 100.0
    loss_perturbed = head._compute_position_loss(pred_perturbed, motion_input)
    assert torch.isclose(loss_with_invalid, loss_perturbed)
