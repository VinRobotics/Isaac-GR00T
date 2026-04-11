# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
FA Hybrid Dual-Stream Modules for FlowmatchingActionHead.

FAEncoder: Frame-averaging wrapper around a frozen pretrained encoder.
EquiResAdapter: Lightweight equivariant residual adapter (SA + CA blocks).
"""

from typing import Optional

import torch
from torch import nn
import escnn.nn as enn

from .equivariant_cross_attention_dit import BasicTransformerBlock as EquiBasicBlock


class FAEncoder(nn.Module):
    """
    Frame Averaging wrapper around a frozen pretrained (non-equivariant) encoder.

    Computes:
        equi = FA_equi(x) = (1/N) Σ_h  ρ_out(h⁻¹) · f(h·x)   [regular repr]
        inv  = FA_inv(x)  = (1/N) Σ_h  f(h·x)                  [invariant plain mean]

    Input rotation:  geo_input.transform(g_elem)  using input field type
    Output rotation: wrap with fa_output_type, call .transform(h_inv)
    _apply_frame_averaging: identical to C8EquivariantTimmObsEncoder
    """

    def __init__(
        self,
        pretrained_encoder: nn.Module,
        in_type: enn.FieldType,
        n_group: int,
        output_dim: int,
    ):
        super().__init__()
        self.pretrained_encoder = pretrained_encoder
        self.in_type = in_type
        self.n_group = n_group
        self.output_dim = output_dim

        assert output_dim % n_group == 0, "output_dim must be divisible by n_group"
        blocks = output_dim // n_group
        gspace = in_type.gspace
        # Output field type: regular repr in embedding space (NOT getJointFieldType)
        self.fa_output_type = enn.FieldType(gspace, [gspace.regular_repr] * blocks)

        # Permutation matrices for frame averaging (identical to C8EquivariantTimmObsEncoder)
        N = n_group
        perm = torch.zeros(N, N, N)
        for r in range(N):
            for i in range(N):
                perm[r, i, (i + r) % N] = 1.0
        self.register_buffer("permutation_matrices", perm)
        self.register_buffer("selected_perm_matrices_template", perm)  # [N, N, N]

    def _apply_frame_averaging(self, features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        features: [BT, N, D_out]  (N encoder outputs stacked along dim=1)
        returns:  [BT, D_out]     equivariant regular repr (FA-averaged)

        Identical to C8EquivariantTimmObsEncoder._apply_frame_averaging.
        """
        feature_dim = features.shape[2]
        blocks = feature_dim // self.n_group
        features = features.reshape(batch_size, self.n_group, blocks, self.n_group)
        all_features_flat = features.reshape(-1, blocks, self.n_group)
        perm = self.selected_perm_matrices_template.repeat(batch_size, 1, 1).to(features.dtype)
        aligned = torch.bmm(all_features_flat, perm)                   # [BT*N, blocks, N]
        avg = aligned.reshape(batch_size, self.n_group, blocks, self.n_group).mean(dim=1)
        return avg.reshape(batch_size, blocks * self.n_group)           # [BT, D_out]

    def encode(
        self,
        geo_input: enn.GeometricTensor,
        cat_ids: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            geo_input: GeometricTensor with self.in_type, shape [(B*T), D_in]
            cat_ids:   [(B*T),] embodiment ids
            timestep:  [B,] diffusion timestep (action encoder only)
        Returns:
            equi: [(B*T), D_out]  equivariant regular repr
            inv:  [(B*T), D_out]  invariant plain mean
        """
        BT = geo_input.tensor.shape[0]
        h_list = []
        gspace = self.in_type.gspace

        for g_idx in range(self.n_group):
            g_elem = gspace.fibergroup.element(g_idx)
            # h·x: rotate input using its own field type
            h_x_geo = enn.GeometricTensor(geo_input.transform(g_elem).tensor, self.in_type)
            # f(h·x): forward through frozen pretrained encoder
            if timestep is not None:
                out = self.pretrained_encoder(h_x_geo, timestep, cat_ids).tensor
            else:
                out = self.pretrained_encoder(h_x_geo, cat_ids).tensor
            h_list.append(out)

        h_stack = torch.stack(h_list, dim=0)  # [N, BT, D_out]

        # Equivariant FA: ρ_out(h⁻¹) · f(h·x), then mean
        aligned_list = []
        for g_idx, h_out in enumerate(h_list):
            g_inv_idx = (self.n_group - g_idx) % self.n_group
            g_inv = gspace.fibergroup.element(g_inv_idx)
            geo_h_out = enn.GeometricTensor(h_out, self.fa_output_type)
            aligned_list.append(geo_h_out.transform(g_inv).tensor)

        aligned_stack = torch.stack(aligned_list, dim=1)               # [BT, N, D_out]
        equi = self._apply_frame_averaging(aligned_stack, BT)          # [BT, D_out]

        # Invariant FA: plain mean, no transformation needed (provably invariant)
        inv = h_stack.mean(dim=0)                                       # [BT, D_out]

        return equi, inv


class EquiResAdapter(nn.Module):
    """
    Lightweight equivariant residual adapter (~2 BasicTransformerBlocks).
    Architecture per block: SA (equi↔equi) + CA (equi←inv_output).

    Identity init: zero-init v_proj, o_proj, ff.fc2 → equi_delta = 0 at init
    → output starts as pure inv_lifted (pretrained baseline), geometry corrections
    are learned gradually during training.

    Equivariance proof:
      SA: regular Q,K,V → inv-QK + equi-V → equivariant output ✓
      CA: equi Q (in_type) + trivial K/V (inv_type)
          → o_proj: trivial → in_type (induction map) → equivariant ✓
    """

    def __init__(
        self,
        in_type: enn.FieldType,
        inv_dim: int,
        num_layers: int = 2,
        num_attention_heads: int = 32,
        attention_head_dim: int = 48,
        dropout: float = 0.2,
        final_dropout: bool = True,
    ):
        super().__init__()
        self.in_type = in_type
        gspace = in_type.gspace
        blocks = in_type.size // gspace.fibergroup.order()

        # Trivial type for inv_output cross-attention context
        self.inv_type = enn.FieldType(gspace, [gspace.trivial_repr] * inv_dim)
        inner_type = enn.FieldType(gspace, [gspace.regular_repr] * blocks)

        self.blocks = nn.ModuleList([
            EquiBasicBlock(
                in_type=in_type,
                cross_attention_type=self.inv_type,
                inner_type=inner_type,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                norm_type="layer_norm",
                final_dropout=final_dropout,
            )
            for _ in range(num_layers)
        ])

        # Identity init: zero-init v_proj, o_proj, ff.fc2 → equi_delta = 0 at init
        for blk in self.blocks:
            blk: EquiBasicBlock
            for p in blk.attn1.v_proj.parameters():
                nn.init.zeros_(p)
            for p in blk.attn1.o_proj.parameters():
                nn.init.zeros_(p)
            for p in blk.ff.fc2.parameters():
                nn.init.zeros_(p)

        # Warm-up scale: ramps 0→1 over warmup_steps
        self.register_buffer("adapter_scale", torch.ones(1))

    def set_warmup_scale(self, step: int, warmup_steps: int) -> None:
        if warmup_steps > 0:
            self.adapter_scale.fill_(min(1.0, step / warmup_steps))

    def forward(self, equi_sa: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            equi_sa: [B, T_sa, D]  regular repr (equi state+action embeddings)
            context: [B, T_sa, D]  inv_output from frozen DiT (plain invariant tensor)
        Returns:
            equi_delta [B, T_sa, D] regular repr
        """
        h = equi_sa
        h_in = h
        for blk in self.blocks:
            blk: EquiBasicBlock
            h = blk(h, encoder_hidden_states=context)
        if self.adapter_scale.item() != 1.0:
            h = h_in + (h - h_in) * self.adapter_scale
        return h
