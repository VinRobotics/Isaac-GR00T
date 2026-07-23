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

"""Test the Sinkhorn entropic-OT alignment loss used for human/robot motion
feature alignment (gr00t/model/modules/optimal_transport.py)."""

from gr00t.model.modules.optimal_transport import sinkhorn_log_domain, sinkhorn_ot_loss
import torch


def test_sinkhorn_plan_marginals_are_uniform():
    torch.manual_seed(0)
    cost = torch.rand(6, 9)
    plan = sinkhorn_log_domain(cost, eps=0.1, n_iters=200)
    assert torch.allclose(plan.sum(dim=1), torch.full((6,), 1.0 / 6), atol=1e-3)
    assert torch.allclose(plan.sum(dim=0), torch.full((9,), 1.0 / 9), atol=1e-3)


def test_sinkhorn_ot_loss_penalizes_distance():
    torch.manual_seed(0)
    h_robot = torch.randn(8, 4) + torch.tensor([5.0, 0, 0, 0])
    h_human_far = torch.randn(8, 4) + torch.tensor([-5.0, 0, 0, 0])
    h_human_close = torch.randn(8, 4) + torch.tensor([5.0, 0, 0, 0])

    loss_far = sinkhorn_ot_loss(h_robot, h_human_far, eps=0.1, n_iters=50)
    loss_close = sinkhorn_ot_loss(h_robot, h_human_close, eps=0.1, n_iters=50)
    assert loss_far.item() > loss_close.item()


def test_sinkhorn_ot_loss_is_differentiable():
    torch.manual_seed(0)
    h_robot = torch.randn(8, 4, requires_grad=True)
    h_human = torch.randn(8, 4)

    loss = sinkhorn_ot_loss(h_robot, h_human, eps=0.1, n_iters=50)
    loss.backward()

    assert h_robot.grad is not None
    assert torch.isfinite(h_robot.grad).all()
    assert h_robot.grad.norm().item() > 0


def test_sinkhorn_ot_loss_cross_feature_matches_on_one_space_aligns_another():
    """Match on `h_*` (e.g. motion features) but align `target_*` (e.g.
    backbone features) — the loss must depend on the TARGET distance, not
    the matching-feature distance, and gradients must reach both spaces."""
    torch.manual_seed(0)
    h_robot = torch.randn(6, 3)
    h_human = torch.randn(6, 3)
    target_robot = torch.randn(6, 5, requires_grad=True)
    target_human_far = torch.randn(6, 5) + torch.tensor([10.0, 0, 0, 0, 0])
    target_human_close = target_robot.detach().clone() + 0.01 * torch.randn(6, 5)

    loss_far = sinkhorn_ot_loss(
        h_robot,
        h_human,
        eps=0.1,
        n_iters=50,
        target_robot=target_robot,
        target_human=target_human_far,
    )
    loss_close = sinkhorn_ot_loss(
        h_robot,
        h_human,
        eps=0.1,
        n_iters=50,
        target_robot=target_robot,
        target_human=target_human_close,
    )
    assert loss_far.item() > loss_close.item()

    loss_far.backward()
    assert target_robot.grad is not None
    assert torch.isfinite(target_robot.grad).all()
    assert target_robot.grad.norm().item() > 0


def test_sinkhorn_ot_loss_cross_feature_matching_ignores_target_values():
    """Two calls with identical h_robot/h_human (same matching plan) but
    swapped target tensors should NOT reduce to the same loss as matching on
    the targets directly — confirms the plan really comes from h_*, not
    target_*."""
    torch.manual_seed(1)
    h_robot = torch.randn(5, 3) + torch.tensor([5.0, 0, 0])
    h_human = torch.randn(5, 3) + torch.tensor([-5.0, 0, 0])
    target_robot = torch.randn(5, 2)
    target_human = torch.randn(5, 2)

    cross_loss = sinkhorn_ot_loss(
        h_robot, h_human, eps=0.1, n_iters=50, target_robot=target_robot, target_human=target_human
    )
    direct_loss = sinkhorn_ot_loss(target_robot, target_human, eps=0.1, n_iters=50)
    assert not torch.isclose(cross_loss, direct_loss)


def test_sinkhorn_ot_loss_stopgrad_isolates_matching_features():
    """With stopgrad_plan=True (the default), the cross-feature alignment
    loss must NOT push h_robot/h_human (the matching features, e.g. the
    motion-token encoder) — only target_robot/target_human (e.g. the backbone
    features) should receive gradient. This keeps the matching signal clean
    and driven only by its own task loss, not "bought" by how easy it makes
    the alignment loss."""
    torch.manual_seed(2)
    h_robot = torch.randn(5, 3, requires_grad=True)
    h_human = torch.randn(5, 3, requires_grad=True)
    target_robot = torch.randn(5, 2, requires_grad=True)
    target_human = torch.randn(5, 2, requires_grad=True)

    loss = sinkhorn_ot_loss(
        h_robot,
        h_human,
        eps=0.1,
        n_iters=50,
        target_robot=target_robot,
        target_human=target_human,
        stopgrad_plan=True,
    )
    loss.backward()

    assert h_robot.grad is None or torch.all(h_robot.grad == 0)
    assert h_human.grad is None or torch.all(h_human.grad == 0)
    assert target_robot.grad is not None and target_robot.grad.norm().item() > 0
    assert target_human.grad is not None and target_human.grad.norm().item() > 0


def test_sinkhorn_ot_loss_without_stopgrad_leaks_into_matching_features():
    """Sanity check for the flag itself: stopgrad_plan=False should let
    gradient reach h_robot/h_human too (the behavior stopgrad_plan=True is
    specifically guarding against)."""
    torch.manual_seed(3)
    h_robot = torch.randn(5, 3, requires_grad=True)
    h_human = torch.randn(5, 3, requires_grad=True)
    target_robot = torch.randn(5, 2)
    target_human = torch.randn(5, 2)

    loss = sinkhorn_ot_loss(
        h_robot,
        h_human,
        eps=0.1,
        n_iters=50,
        target_robot=target_robot,
        target_human=target_human,
        stopgrad_plan=False,
    )
    loss.backward()

    assert h_robot.grad is not None
    assert h_robot.grad.norm().item() > 0
