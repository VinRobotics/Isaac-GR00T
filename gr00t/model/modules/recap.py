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

# RECAP / CFGRL components for GR00T N1.7
# References:
#   π*0.6 / RECAP  arXiv:2511.14759  §IV-A, §IV-B, App. F
#   CFGRL          arXiv:2505.23458  Alg. 1, Alg. 2

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig


class AdvantageEmbedding(nn.Module):
    """
    Encodes the binary advantage indicator I_t from RECAP (π*0.6 §IV-B).

    Three indices:
        NULL_IDX (0) – unconditional token (CFG dropout + null inference pass)
        NEG_IDX  (1) – A(o, a) ≤ ε_l  →  "Advantage: negative"
        POS_IDX  (2) – A(o, a)  > ε_l  →  "Advantage: positive"

    The embedding is appended to encoder_hidden_states so the cross-attention
    in DiT can condition the velocity field on optimality.
    """

    NULL_IDX: int = 0
    NEG_IDX: int = 1
    POS_IDX: int = 2

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(3, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: (B,) long tensor with values in {NULL_IDX, NEG_IDX, POS_IDX}
        Returns:
            (B, 1, embedding_dim)
        """
        return self.embedding(labels).unsqueeze(1)


@dataclass
class DistributionalValueHeadConfig(PretrainedConfig):
    """
    Config for the distributional value function pϕ(V | o_t, l) from RECAP §IV-A.

    Architecture:
        backbone_features  (B, S, seq_dim)
            -> CLS cross-attention over sequence   (B, seq_dim)
            -> LayerNorm
            -> MLP  (seq_dim -> hidden_dim -> num_bins)
            -> softmax  ->  pϕ(V | o_t, l)         (B, num_bins)

    Return bins normalised to (-1, 0):
        bin 0    -> value -1.0  (failed episode)
        bin B-1  -> value  0.0  (successful completion)
    """

    seq_dim: int = field(default=2048)
    num_heads: int = field(default=8)
    dropout: float = field(default=0.1)
    state_dim: int = field(default=132)
    hidden_dim: int = field(default=512)
    num_bins: int = field(default=201)
    value_min: float = field(default=-1.0)
    value_max: float = field(default=0.0)
    value_loss_coeff: float = field(default=1.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class DistributionalValueHead(nn.Module):
    """
    Distributional value function  pϕ(V | o_t, l)  from RECAP §IV-A.

    Trains by minimising cross-entropy against discretised empirical return R_t(τ):

        min_ϕ  E_{τ∈D} [ Σ_{o_t∈τ}  H( R^B_t(τ),  pϕ(V | o_t, l) ) ]   (Eq. 1)

    Scalar value at inference:
        V(o_t, l) = Σ_b  pϕ(V=b | o_t) · v(b)
    """

    def __init__(self, config: DistributionalValueHeadConfig):
        super().__init__()
        num_heads = config.num_heads
        dropout = config.dropout
        B_bins = config.num_bins

        while config.seq_dim % num_heads != 0 and num_heads > 1:
            num_heads //= 2

        self.norm_in = nn.LayerNorm(config.seq_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.seq_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.seq_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(config.seq_dim)

        # state_proj is constructed for future use; currently logits derive from attended only
        self.state_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.seq_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.seq_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Linear(config.hidden_dim // 2, B_bins),
        )

        self.register_buffer(
            "bin_centres",
            torch.linspace(config.value_min, config.value_max, B_bins),
        )

        self.config = config
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.ones_(self.norm_in.weight)
        nn.init.zeros_(self.norm_in.bias)
        nn.init.ones_(self.norm_attn.weight)
        nn.init.zeros_(self.norm_attn.bias)
        for name, p in self.cross_attn.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def _encode(
        self,
        backbone_features: torch.Tensor,  # (B, S, seq_dim)
        state: torch.Tensor,              # (B, state_dim) or (B, 1, state_dim)
    ) -> torch.Tensor:
        cls = self.cls_token.float().expand(backbone_features.shape[0], -1, -1)
        feats = self.norm_in(backbone_features.float())
        attended, _ = self.cross_attn(query=cls, key=feats, value=feats)
        attended = self.norm_attn(attended.squeeze(1))
        return self.classifier(attended)

    def predict_value(
        self,
        backbone_features: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(self._encode(backbone_features, state), dim=-1)
        return (probs * self.bin_centres).sum(dim=-1)

    def compute_loss(
        self,
        backbone_features: torch.Tensor,
        state: torch.Tensor,
        empirical_returns: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._encode(backbone_features, state)
        bin_idx = self._discretise(empirical_returns)
        return F.cross_entropy(logits, bin_idx) * self.config.value_loss_coeff

    def _discretise(self, returns: torch.Tensor) -> torch.Tensor:
        v_min, v_max = self.config.value_min, self.config.value_max
        r = returns.clamp(v_min, v_max)
        return ((r - v_min) / (v_max - v_min) * (self.config.num_bins - 1)).long()


def compute_normalised_returns(
    success: torch.Tensor,
    episode_lengths: torch.Tensor,
    t: torch.Tensor,
    max_episode_length: int = 520,
    c_fail: float = 260.0,
) -> torch.Tensor:
    """
    Computes normalised empirical return R_t(τ) for RECAP §IV-A, §V-C.

    Reward definition (RECAP Eq. 5):
        r_{t'} = 0       if t' == T and success
                 -C_fail if t' == T and failure
                 -1      otherwise

    Return from step t:
        R_t(τ) = -(T - t)           if success
                 -(T - t) - C_fail  if failure

    Normalised to (-1, 0):
        R_t_norm = R_t(τ) / (max_episode_length + C_fail)
    """
    steps_remaining = (episode_lengths - t).float()
    raw_return = -steps_remaining
    raw_return = torch.where(success, raw_return, raw_return - c_fail)
    norm_denom = float(max_episode_length + c_fail)
    return (raw_return / norm_denom).clamp(-1.0, 0.0)
