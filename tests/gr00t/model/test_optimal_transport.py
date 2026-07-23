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
