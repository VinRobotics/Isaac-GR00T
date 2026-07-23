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

"""Test Gr00tTrainer's cross-feature OT alignment wiring (match on pooled
motion features, align pooled backbone features) — see
gr00t/experiment/trainer.py's _compute_ot_loss / _collect_domain_features /
_log_domain_alignment_viz. Calls the (unbound) methods directly against a
lightweight fake `self` rather than constructing a full HF Trainer."""

import types

from gr00t.experiment import trainer as trainer_module
import torch


def _fake_trainer(ot_sinkhorn_eps=0.1, ot_sinkhorn_iters=50, domain_viz_max_points=100):
    fake = types.SimpleNamespace()
    fake.model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            ot_sinkhorn_eps=ot_sinkhorn_eps, ot_sinkhorn_iters=ot_sinkhorn_iters
        )
    )
    fake.domain_viz_max_points = domain_viz_max_points
    fake.state = types.SimpleNamespace(global_step=1)
    return fake


def test_compute_ot_loss_matches_on_motion_aligns_on_backbone():
    """The transport plan comes from motion features; the loss value should
    change when backbone features change, even with motion features fixed."""
    torch.manual_seed(0)
    fake = _fake_trainer()
    is_human = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    motion_pooled = torch.randn(6, 4)
    backbone_close = torch.randn(6, 3)
    backbone_far = backbone_close.clone()
    backbone_far[3:] += 20.0  # push the human-side backbone features far away

    loss_close = trainer_module.Gr00tTrainer._compute_ot_loss(
        fake, motion_pooled, backbone_close, is_human
    )
    loss_far = trainer_module.Gr00tTrainer._compute_ot_loss(
        fake, motion_pooled, backbone_far, is_human
    )
    assert loss_close is not None and loss_far is not None
    assert loss_far.item() > loss_close.item()


def test_compute_ot_loss_none_when_one_domain_missing():
    fake = _fake_trainer()
    is_human = torch.zeros(5)  # all robot, no human samples this batch
    motion_pooled = torch.randn(5, 4)
    backbone_pooled = torch.randn(5, 3)
    result = trainer_module.Gr00tTrainer._compute_ot_loss(
        fake, motion_pooled, backbone_pooled, is_human
    )
    assert result is None


def test_collect_domain_features_keeps_motion_and_backbone_paired():
    """Reservoir sampling must keep motion_feats[i]/backbone_feats[i]/
    is_human_list[i] referring to the SAME underlying sample at every index —
    the neighbor-structure diagnostic depends on this pairing."""
    fake = _fake_trainer(domain_viz_max_points=4)
    motion_feats: list = []
    backbone_feats: list = []
    is_human_list: list = []
    num_seen = 0

    torch.manual_seed(0)
    for _ in range(10):
        batch_size = 3
        motion = torch.randn(batch_size, 4)
        backbone = motion[:, :3] * 10.0  # deterministic function of motion, for pairing check
        is_human = torch.randint(0, 2, (batch_size,)).float()
        num_seen = trainer_module.Gr00tTrainer._collect_domain_features(
            fake, motion, backbone, is_human, motion_feats, backbone_feats, is_human_list, num_seen
        )

    assert len(motion_feats) == len(backbone_feats) == len(is_human_list) == 4
    for m, b in zip(motion_feats, backbone_feats):
        assert (abs(m[:3] * 10.0 - b) < 1e-5).all()


def test_log_domain_alignment_viz_logs_expected_scalars(monkeypatch):
    torch.manual_seed(0)
    fake = _fake_trainer()

    logged = {}

    def fake_log(d, step=None):
        logged.update(d)

    monkeypatch.setattr(trainer_module.wandb, "run", object())  # truthy dummy run
    monkeypatch.setattr(trainer_module.wandb, "log", fake_log)
    monkeypatch.setattr(trainer_module.wandb, "Image", lambda fig: fig)

    n_robot, n_human = 8, 8
    motion_feats = [torch.randn(4).numpy() for _ in range(n_robot + n_human)]
    backbone_feats = [torch.randn(5).numpy() for _ in range(n_robot + n_human)]
    is_human_list = [0.0] * n_robot + [1.0] * n_human

    trainer_module.Gr00tTrainer._log_domain_alignment_viz(
        fake, motion_feats, backbone_feats, is_human_list, "eval"
    )

    assert "eval/backbone_domain_robot_variance" in logged
    assert "eval/backbone_domain_human_variance" in logged
    assert "eval/backbone_domain_knn_same_domain_frac" in logged
    assert "eval/motion_backbone_neighbor_agreement" in logged
    assert "eval/backbone_domain_tsne" in logged
    assert 0.0 <= logged["eval/motion_backbone_neighbor_agreement"] <= 1.0


def test_log_domain_alignment_viz_skips_when_one_domain_too_small(monkeypatch, caplog):
    fake = _fake_trainer()
    logged = {}
    monkeypatch.setattr(trainer_module.wandb, "run", object())
    monkeypatch.setattr(trainer_module.wandb, "log", lambda d, step=None: logged.update(d))

    motion_feats = [torch.randn(4).numpy() for _ in range(3)]
    backbone_feats = [torch.randn(5).numpy() for _ in range(3)]
    is_human_list = [0.0, 0.0, 1.0]  # only 1 human sample

    trainer_module.Gr00tTrainer._log_domain_alignment_viz(
        fake, motion_feats, backbone_feats, is_human_list, "eval"
    )
    assert logged == {}
