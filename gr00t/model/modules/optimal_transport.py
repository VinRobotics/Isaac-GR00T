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

"""Entropic Optimal Transport (Sinkhorn) for aligning pooled human/robot
motion features — see method_motion_keypoint_ot_v2.md Section 3.1."""

import math

import torch


def sinkhorn_log_domain(cost: torch.Tensor, eps: float, n_iters: int) -> torch.Tensor:
    """Log-domain-stabilized Sinkhorn algorithm for entropic OT with uniform
    marginals. Numerically stable at small eps (unlike the plain-exponential
    update, which over/underflows), fully differentiable via autograd —
    autodiff through the iterations is sufficient (no closed-form/implicit
    gradient needed), the same choice EgoBridge/COT make.

    Args:
        cost: [N, M] pairwise cost matrix.
        eps: entropic regularization strength.
        n_iters: number of Sinkhorn normalization iterations.

    Returns:
        [N, M] transport plan, rows summing to 1/N and columns to 1/M.
    """
    n, m = cost.shape
    log_mu = torch.full((n,), -math.log(n), device=cost.device, dtype=cost.dtype)
    log_nu = torch.full((m,), -math.log(m), device=cost.device, dtype=cost.dtype)
    f = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    g = torch.zeros(m, device=cost.device, dtype=cost.dtype)

    for _ in range(n_iters):
        f = eps * (log_mu - torch.logsumexp((-cost + g.unsqueeze(0)) / eps, dim=1))
        g = eps * (log_nu - torch.logsumexp((-cost + f.unsqueeze(1)) / eps, dim=0))

    log_plan = (-cost + f.unsqueeze(1) + g.unsqueeze(0)) / eps
    return log_plan.exp()


def sinkhorn_ot_loss(
    h_robot: torch.Tensor, h_human: torch.Tensor, eps: float, n_iters: int
) -> torch.Tensor:
    """Entropic OT alignment loss sum(P* * C) between two point clouds (pooled
    per-sample motion features), C = pairwise squared Euclidean distance.

    Args:
        h_robot: [N, D] pooled robot motion features.
        h_human: [M, D] pooled human motion features.
    """
    cost = torch.cdist(h_robot, h_human, p=2) ** 2
    plan = sinkhorn_log_domain(cost, eps=eps, n_iters=n_iters)
    return (plan * cost).sum()
