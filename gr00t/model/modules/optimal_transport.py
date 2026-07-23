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
features — see method_motion_keypoint_ot_v2.md Section 3.1.

The transport plan can be computed from one feature space (pooled
motion-token features — an embodiment-invariant "what motion is this" signal)
and applied to align a DIFFERENT feature space (pooled backbone features,
which actually feed the DiT/action head): motion features decide WHICH
robot/human samples correspond, the alignment loss then pulls their backbone
representations together according to that correspondence, rather than
pulling the motion features themselves together."""

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
    h_robot: torch.Tensor,
    h_human: torch.Tensor,
    eps: float,
    n_iters: int,
    target_robot: torch.Tensor | None = None,
    target_human: torch.Tensor | None = None,
    stopgrad_plan: bool = True,
) -> torch.Tensor:
    """Entropic OT alignment loss sum(P* * C) between two point clouds,
    C = pairwise squared Euclidean distance.

    By default (target_robot/target_human omitted), matches AND aligns the
    same features: the transport plan P* and the loss cost C are both
    computed from h_robot/h_human, pulling them directly together.

    Pass target_robot/target_human to decouple matching from alignment: P* is
    still computed from h_robot/h_human's pairwise distances, but the loss
    instead pulls target_robot/target_human together according to that plan
    — e.g. match on pooled motion-token features, but actually align the
    pooled backbone features that feed the DiT/action head.

    stopgrad_plan (default True): detach P* before multiplying by the cost,
    so gradients from this loss reach `target_robot`/`target_human` (via the
    cost term) but NOT `h_robot`/`h_human` (via how the plan was computed).
    This matters specifically in the cross-feature case: without it, the
    alignment loss would push the matching features (e.g. the motion-token
    encoder) toward "whatever makes backbone alignment easier" instead of
    staying a clean, embodiment-invariant signal driven only by its own task
    loss (motion_loss) — the same reason the matching side must not be
    "bought" by the alignment objective. Standard practice in minibatch-OT
    domain adaptation (e.g. DeepJDOT): the plan is treated as fixed for the
    current cost each step, not differentiated through.

    Args:
        h_robot: [N, D] pooled robot features used to compute the matching.
        h_human: [M, D] pooled human features used to compute the matching.
        target_robot: [N, D'] optional — robot features to actually align
            (defaults to h_robot).
        target_human: [M, D'] optional — human features to actually align
            (defaults to h_human).
    """
    cost = torch.cdist(h_robot, h_human, p=2) ** 2
    plan = sinkhorn_log_domain(cost, eps=eps, n_iters=n_iters)
    if stopgrad_plan:
        plan = plan.detach()
    if target_robot is None and target_human is None:
        return (plan * cost).sum()
    target_cost = torch.cdist(target_robot, target_human, p=2) ** 2
    return (plan * target_cost).sum()
