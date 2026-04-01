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

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import escnn.nn as enn

from .equivariant_cross_attention_dit import (
    EquivariantLayerNorm,
    EquivariantAttention,
)
from .equivariant_activation import EquivariantGate


class GeoMFormerBlock(nn.Module):
    """
    GeoMFormer-style dual-stream attention block.

    Simultaneously refines:
      Z^E [B, T_eq,  G*blocks]  equivariant tokens  (regular repr, D = G*blocks)
      Z^I [B, T_inv, inv_dim]   invariant tokens     (trivial repr, inv_dim = blocks)

    Per layer (OLD values used for cross-attention — GeoMFormer parallel design):
      Z^E: LN -> EquSA(Z^E) + EquCA(Z^E <- Z^I) -> residual -> LN -> EquFFN(Z^E, Z^I) -> residual
      Z^I: LN -> InvSA(Z^I)  + InvCA(Z^I <- Z^E) -> residual -> LN -> InvFFN(Z^I)     -> residual

    EquFFN  : EquivariantGate — Z^I (mean-pooled over T_inv) gates each Z^E token.
    InvCA   : Q from Z^I, K/V = group-mean (trivial) projection of Z^E.
    EquCA   : Z^E queries Z^I context (trivial cross-type).
    """

    def __init__(
        self,
        equi_type: enn.FieldType,
        inv_dim: int,
        num_layers: int = 4,
        num_attention_heads_eq: int = 32,
        attention_head_dim_eq: int = 64,
        num_attention_heads_inv: int = 8,
        dropout: float = 0.0,
        final_dropout: bool = False,
    ):
        super().__init__()
        gspace = equi_type.gspace
        self.G = gspace.fibergroup.order()
        self.D = equi_type.size             # G * blocks
        self.blocks = self.D // self.G
        self.inv_dim = inv_dim
        self.equi_type = equi_type
        self.num_layers = num_layers
        self.Heq = num_attention_heads_eq
        self.Dheq = attention_head_dim_eq
        self.Hinv = num_attention_heads_inv
        assert self.inv_dim % self.Hinv == 0, (
            f"inv_dim ({self.inv_dim}) must be divisible by num_attention_heads_inv ({self.Hinv})"
        )
        self.Dhinv = self.inv_dim // self.Hinv
        self.scale_inv = self.Dhinv ** -0.5
        self.dropout_p = dropout

        # Trivial FieldType for Z^I (one trivial repr per scalar channel)
        self.z_inv_trivial_type = enn.FieldType(gspace, [gspace.trivial_repr] * self.inv_dim)

        # Trivial FieldType for K/V of InvCA (group-mean projection of Z^E)
        scalar_dim_inv = self.Hinv * self.Dhinv   # == inv_dim
        inv_kv_type = enn.FieldType(gspace, [gspace.trivial_repr] * scalar_dim_inv)

        # FFN inner type for Z^E (4× expansion)
        inner_type = enn.FieldType(gspace, [gspace.regular_repr] * (self.blocks * 4))

        # ── Z^E components ────────────────────────────────────────────────────
        self.eq_norm_sa  = nn.ModuleList([EquivariantLayerNorm(equi_type) for _ in range(num_layers)])
        self.eq_norm_ca  = nn.ModuleList([EquivariantLayerNorm(equi_type) for _ in range(num_layers)])
        self.eq_norm_ffn = nn.ModuleList([EquivariantLayerNorm(equi_type) for _ in range(num_layers)])

        # EquSA: equivariant self-attention on Z^E
        self.eq_sa = nn.ModuleList([
            EquivariantAttention(equi_type, equi_type,
                                 heads=self.Heq, dim_head=self.Dheq, dropout=dropout)
            for _ in range(num_layers)
        ])
        # EquCA: Z^E queries, Z^I trivial context
        self.eq_ca = nn.ModuleList([
            EquivariantAttention(equi_type, self.z_inv_trivial_type,
                                 heads=self.Heq, dim_head=self.Dheq, dropout=dropout)
            for _ in range(num_layers)
        ])
        # EquFFN: gate equivariant FFN — Z^I (mean-pooled) gates Z^E
        self.eq_ffn = nn.ModuleList([
            EquivariantGate(equi_type, inner_type, self.z_inv_trivial_type)
            for _ in range(num_layers)
        ])

        # ── Z^I components ────────────────────────────────────────────────────
        self.inv_norm_sa  = nn.ModuleList([nn.LayerNorm(self.inv_dim) for _ in range(num_layers)])
        self.inv_norm_ca  = nn.ModuleList([nn.LayerNorm(self.inv_dim) for _ in range(num_layers)])
        self.inv_norm_ffn = nn.ModuleList([nn.LayerNorm(self.inv_dim) for _ in range(num_layers)])

        # InvSA: standard multi-head self-attention on Z^I
        self.inv_sa = nn.ModuleList([
            nn.MultiheadAttention(self.inv_dim, self.Hinv, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # InvCA: Q from Z^I, K/V = group-mean (trivial) projection of Z^E
        self.inv_ca_q   = nn.ModuleList([nn.Linear(self.inv_dim, scalar_dim_inv)   for _ in range(num_layers)])
        self.inv_ca_k   = nn.ModuleList([enn.Linear(equi_type, inv_kv_type) for _ in range(num_layers)])
        self.inv_ca_v   = nn.ModuleList([enn.Linear(equi_type, inv_kv_type) for _ in range(num_layers)])
        self.inv_ca_out = nn.ModuleList([nn.Linear(scalar_dim_inv, self.inv_dim)   for _ in range(num_layers)])

        # InvFFN: standard 2-layer MLP on Z^I
        inv_ffn_inner = self.inv_dim * 4
        self.inv_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.inv_dim, inv_ffn_inner),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inv_ffn_inner, self.inv_dim),
                *([nn.Dropout(dropout)] if final_dropout else []),
            )
            for _ in range(num_layers)
        ])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _eq_norm(self, norm: EquivariantLayerNorm, x: torch.Tensor) -> torch.Tensor:
        """Apply EquivariantLayerNorm: [B, T, D] -> [B, T, D]."""
        B, T, D = x.shape
        geo = enn.GeometricTensor(x.reshape(B * T, D), norm.field_type)
        return norm(geo).tensor.reshape(B, T, D)

    def _inv_cross_attn(self, z_inv: torch.Tensor, h_vis: torch.Tensor, i: int) -> torch.Tensor:
        """
        InvCrossAttn: Z^I [B, T_inv, inv_dim] queries Z^E [B, T_eq, D].
        K/V = group-mean (trivial) projection of Z^E.
        Returns [B, T_inv, inv_dim].
        """
        B, T_inv, _ = z_inv.shape
        T_eq = h_vis.shape[1]

        Q = self.inv_ca_q[i](z_inv)                                       # [B, T_inv, H*Dh]
        h_geo = enn.GeometricTensor(h_vis.reshape(B * T_eq, self.D), self.equi_type)
        K = self.inv_ca_k[i](h_geo).tensor.reshape(B, T_eq, self.Hinv * self.Dhinv)
        V = self.inv_ca_v[i](h_geo).tensor.reshape(B, T_eq, self.Hinv * self.Dhinv)

        Q = Q.reshape(B, T_inv, self.Hinv, self.Dhinv).permute(0, 2, 1, 3)
        K = K.reshape(B, T_eq,  self.Hinv, self.Dhinv).permute(0, 2, 1, 3)
        V = V.reshape(B, T_eq,  self.Hinv, self.Dhinv).permute(0, 2, 1, 3)

        attn = torch.einsum('b h i d, b h j d -> b h i j', Q, K) * self.scale_inv
        attn = torch.softmax(attn, dim=-1)
        if self.dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, T_inv, self.Hinv * self.Dhinv)
        return self.inv_ca_out[i](out)                                     # [B, T_inv, inv_dim]

    def forward(self, h_vis: torch.Tensor, z_inv: torch.Tensor):
        """
        h_vis : [B, T_eq,  D]        equivariant tokens (regular repr, D = G*blocks)
        z_inv : [B, T_inv, inv_dim]  invariant tokens   (trivial repr)
        Returns: (h_vis, z_inv) updated, same shapes.
        """
        for i in range(self.num_layers):
            B, T_eq, _ = h_vis.shape

            # Snapshot OLD values for cross-attention (parallel GeoMFormer design)
            h_vis_old = h_vis
            z_inv_old = z_inv

            # ── Z^E: EquSA + EquCA (both from OLD values) ─────────────────────
            h_n_sa = self._eq_norm(self.eq_norm_sa[i], h_vis)
            eq_sa  = self.eq_sa[i](h_n_sa)

            h_n_ca = self._eq_norm(self.eq_norm_ca[i], h_vis)
            eq_ca  = self.eq_ca[i](h_n_ca, encoder_hidden_states=z_inv_old)

            h_vis = h_vis + eq_sa + eq_ca              # combined residual

            # EquFFN: gate equivariant FFN using Z^I (mean-pooled) as gate signal
            h_n_ffn = self._eq_norm(self.eq_norm_ffn[i], h_vis)
            h_geo   = enn.GeometricTensor(h_n_ffn.reshape(B * T_eq, self.D), self.equi_type)

            # Pool Z^I over T_inv -> [B, inv_dim], broadcast to each Z^E token
            z_inv_pool = z_inv_old.mean(dim=1)                             # [B, inv_dim]
            z_inv_gate = z_inv_pool.unsqueeze(1).expand(-1, T_eq, -1)     # [B, T_eq, inv_dim]
            z_inv_geo  = enn.GeometricTensor(
                z_inv_gate.reshape(B * T_eq, self.inv_dim), self.z_inv_trivial_type
            )

            h_vis = h_vis + self.eq_ffn[i](h_geo, z_inv_geo).tensor.reshape(B, T_eq, self.D)

            # ── Z^I: InvSA + InvCA (both from OLD values) ─────────────────────
            z_n_sa    = self.inv_norm_sa[i](z_inv)
            inv_sa, _ = self.inv_sa[i](z_n_sa, z_n_sa, z_n_sa)

            z_n_ca = self.inv_norm_ca[i](z_inv)
            inv_ca = self._inv_cross_attn(z_n_ca, h_vis_old, i)

            z_inv = z_inv + inv_sa + inv_ca            # combined residual

            # InvFFN
            z_inv = z_inv + self.inv_ffn[i](self.inv_norm_ffn[i](z_inv))

        return h_vis, z_inv


class EquiSplitCrossAttention(nn.Module):
    """
    Equivariant cross-attention from SA_embs (regular repr) to a mixed context
    of equivariant (regular repr) and invariant (trivial repr) vl tokens.

    Implements:
        sa_out = sa_embs + EquiCA(LN(sa_embs), Z^E) + EquiCA(LN(sa_embs), Z^I)

    Both cross-attentions preserve equivariance:
      - Regular K/V (Z^E): scores are inner products of equivariant projections → invariant ✓
      - Trivial K/V (Z^I): scalar K/V → scores invariant, output equivariant ✓

    Args:
        sa_type:       FieldType for SA_embs (regular repr).
        vl_equi_type:  FieldType for equivariant vl tokens (regular repr).
        vl_inv_type:   FieldType for invariant vl tokens (trivial repr).
        num_heads:     Number of attention heads.
        head_dim:      Dimension per head.
        dropout:       Dropout probability.
    """

    def __init__(
        self,
        sa_type: enn.FieldType,
        vl_equi_type: enn.FieldType,
        vl_inv_type: enn.FieldType,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.sa_type = sa_type

        self.norm     = EquivariantLayerNorm(sa_type)
        self.ca_equi  = EquivariantAttention(sa_type, vl_equi_type, heads=num_heads, dim_head=head_dim, dropout=dropout)
        self.ca_inv   = EquivariantAttention(sa_type, vl_inv_type,  heads=num_heads, dim_head=head_dim, dropout=dropout)

    def _eq_norm(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        geo = enn.GeometricTensor(x.reshape(B * T, D), self.norm.field_type)
        return self.norm(geo).tensor.reshape(B, T, D)

    def forward(
        self,
        sa_embs: torch.Tensor,   # [B, T_sa, D_sa]  regular repr
        vl_equi: torch.Tensor,   # [B, T_eq, D_bb]  regular repr
        vl_inv:  torch.Tensor,   # [B, T_inv, inv_dim]  trivial repr (scalar)
    ) -> torch.Tensor:
        sa_norm = self._eq_norm(sa_embs)
        ca_out  = self.ca_equi(sa_norm, encoder_hidden_states=vl_equi)
        ca_out  = ca_out + self.ca_inv(sa_norm, encoder_hidden_states=vl_inv)
        return sa_embs + ca_out
