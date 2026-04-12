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

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import SinusoidalPositionalEncoding, swish

from .cross_attention_dit import DiT, SelfAttentionTransformer


def get_prefix_weights(start: int, end: int, total: int, schedule: str) -> torch.Tensor:
    """
    With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    assert schedule in ["ones", "zeros", "linear", "exp"], f"Invalid schedule: {schedule}"
    start = min(start, end)
    idx = torch.arange(total, dtype=torch.float32)
    if schedule == "ones":
        w = torch.ones(total, dtype=torch.float32)
    elif schedule == "zeros":
        w = (idx < start).float()
    elif schedule == "linear" or schedule == "exp":
        w = torch.clamp((start - 1 - idx) / (end - start + 1) + 1, min=0, max=1)
        if schedule == "exp":
            # torch.expm1(x) = exp(x) - 1, torch.e = math.e
            w = w * torch.expm1(w) / (torch.tensor(torch.e) - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    w = torch.where(idx >= end, torch.tensor(0.0, dtype=w.dtype), w)
    return w


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        if timesteps.dim() == 2 and timesteps.shape == (B, T):
            pass  # already in desired shape
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x
    

class AdvantageEmbedding(nn.Module):
    """
    Encodes the binary advantage indicator I_t from RECAP (pi*0.6, §IV-B).
 
    Three indices:
        NULL_IDX (0)  – unconditional token used during CFG dropout and the
                        unconditional forward pass at inference.
        NEG_IDX  (1)  – A(o, a) ≤ eps_l  ->  "Advantage: negative"
        POS_IDX  (2)  – A(o, a)  > eps_l  ->  "Advantage: positive"
 
    The embedding is appended to encoder_hidden_states so the cross-attention
    in DiT can condition the velocity field on optimality.
    """
 
    NULL_IDX: int = 0
    NEG_IDX:  int = 1
    POS_IDX:  int = 2
 
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(3, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
 
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: (B,) long tensor with values in {NULL_IDX, NEG_IDX, POS_IDX}
        Returns:
            (B, 1, embedding_dim) — one token per batch element
        """
        return self.embedding(labels).unsqueeze(1)


@dataclass
class DistributionalValueHeadConfig(PretrainedConfig):
    """
    Config for the distributional value function pϕ(V | o_t, l) from RECAP §IV-A.
 
    The value function is kept intentionally SEPARATE from the policy network
    (same design as pi*0.6, which uses a smaller 670M VLM backbone for the VF
    and a larger one for the policy). Here we attach it as a lightweight MLP
    head that reads the same backbone_features the policy already computes,
    so we pay zero extra backbone cost.
 
    Architecture:
        backbone_features  (B, S, backbone_dim)
            -> mean-pool over sequence          (B, backbone_dim)
            -> LayerNorm
            -> MLP  (backbone_dim -> hidden_dim -> num_bins)
            -> softmax  ->  pϕ(V | o_t, l)      (B, num_bins)
 
    Return bins:
        Normalised to (-1, 0) following pi*0.6 §V-C.
        bin 0  ->  value  -1.0   (worst / failed episode)
        bin B-1 -> value   0.0   (successful completion, 0 steps remaining)
    """
    backbone_embedding_dim: int = field(default=1536)

    seq_dim: int = field(default=8)
    num_heads: int = field(default=8)
    dropout: float = field(default=0.1)

    state_dim: int = field(default=64)
    hidden_dim: int = field(default=512)
    num_bins: int = field(default=201)          # B = 201 as in pi*0.6
    value_min: float = field(default=-1.0)      # normalised lower bound
    value_max: float = field(default=0.0)       # normalised upper bound
    value_loss_coeff: float = field(default=1.0)
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
 
 
class DistributionalValueHead(nn.Module):
    """
    Distributional value function  pϕ(V | o_t, l)  from RECAP §IV-A.
 
    Trains by minimising cross-entropy between the predicted bin distribution
    and the discretised empirical return R_t(τ):
 
        min_ϕ  E_{τ∈D} [ Σ_{o_t∈τ}  H( R^B_t(τ),  pϕ(V | o_t, l) ) ]   (Eq. 1)
 
    At inference, extracts a scalar value as:
 
        V(o_t, l) = Σ_b  pϕ(V=b | o_t) · v(b)
 
    where v(b) is the real-valued centre of bin b.
 
    Advantage estimation (n-step, App. F of pi*0.6):
 
        A(o_t, a_t) = Σ_{t'=t}^{t+N-1} r_{t'} + V(o_{t+N}) - V(o_t)
 
    Usage
    -----
    # ── training ──
    value_loss = value_head.compute_loss(backbone_output, returns)
 
    # ── advantage labelling (offline, on whole dataset) ──
    adv_labels = value_head.compute_advantage_labels(
        backbone_features_t,      # (B, S, D) at step t
        backbone_features_t_plus_N,  # (B, S, D) at step t+N
        rewards_t_to_t_plus_N,    # (B, N)  per-step rewards
        epsilon_ell,              # per-task threshold (30th–40th percentile)
    )
    """

    def __init__(self, config: DistributionalValueHeadConfig):
        super().__init__()
        # Ensure seq_dim is divisible by num_heads
        num_heads = config.num_heads
        dropout = config.dropout
        B = config.num_bins
        
        while config.seq_dim % num_heads != 0 and num_heads > 1:
            num_heads //= 2

        self.norm_in = nn.LayerNorm(config.seq_dim)

        # Learnable CLS token — dedicated "task-completion query"
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.seq_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Cross-attention: CLS attends over all sequence tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.seq_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(config.seq_dim)

        self.state_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.seq_dim),   # project to backbone dim for clean addition
        )

        # Classifier MLP with dropout
        self.classifier = nn.Sequential(
            nn.Linear(config.seq_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Linear(config.hidden_dim // 2, B),
        )

        self.register_buffer(
            "bin_centres",
            torch.linspace(config.value_min, config.value_max, B),
        )

        self.config = config

        self._init_weights()

    def _init_weights(self):
        """
        Initialize ALL parameters in this module to safe values.
        Called both from __init__ and from GR00T_N1_5.from_pretrained (since
        HuggingFace replaces every parameter with uninitialized memory for keys
        absent from the base checkpoint, discarding the __init__ values).
        """
        # CLS token
        nn.init.normal_(self.cls_token, std=0.02)
        # LayerNorms: standard init
        nn.init.ones_(self.norm_in.weight)
        nn.init.zeros_(self.norm_in.bias)
        nn.init.ones_(self.norm_attn.weight)
        nn.init.zeros_(self.norm_attn.bias)
        # Cross-attention projection weights
        for name, p in self.cross_attn.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        # Classifier MLP
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(
        self,
        backbone_features: torch.Tensor,   # (B, S_seq, D)
        state:             torch.Tensor,   # (B, state_dim)
    ) -> torch.Tensor:
        """
        Fuse vision-language and proprioceptive state, then predict bins.

        backbone_features: mean-pooled VL tokens  -> (B, D)
        state:             raw proprioception      -> projected to (B, D)
        concatenated                               -> (B, 2D)  -> MLP  -> (B, num_bins)
        """
        cls = self.cls_token.float().expand(backbone_features.shape[0], -1, -1)  # (B, 1, seq_dim)
        backbone_features = self.norm_in(backbone_features)
        attended, _ = self.cross_attn(query=cls, key=backbone_features, value=backbone_features)  # (B, 1, seq_dim)
        attended = self.norm_attn(attended.squeeze(1))              # (B, seq_dim)
        state_proj = self.state_proj(state.float().squeeze(dim=1))
        fused      = torch.cat([attended, state_proj], dim=-1)
        logits = self.classifier(attended)
        return logits

    def predict_value(
        self,
        backbone_features: torch.Tensor,   # (B, S_seq, D)
        state:             torch.Tensor,   # (B, state_dim)
    ) -> torch.Tensor:
        probs = torch.softmax(self._encode(backbone_features, state), dim=-1)
        return (probs * self.bin_centres).sum(dim=-1)      # (B,)

    def compute_loss(
        self,
        backbone_features:  torch.Tensor,   # (B, S_seq, D)
        state:              torch.Tensor,   # (B, state_dim)
        empirical_returns:  torch.Tensor,   # (B,) in [value_min, value_max]
    ) -> torch.Tensor:
        logits  = self._encode(backbone_features, state)
        bin_idx = self._discretise(empirical_returns)
        return F.cross_entropy(logits, bin_idx) * self.config.value_loss_coeff

    def _discretise(self, returns: torch.Tensor) -> torch.Tensor:
        v_min, v_max = self.config.value_min, self.config.value_max
        r = returns.clamp(v_min, v_max)
        return ((r - v_min) / (v_max - v_min) * (self.config.num_bins - 1)).long()
 
 
# ---------------------------------------------------------------------------
# Utility: compute normalised per-step returns from episode data
# ---------------------------------------------------------------------------
 
def compute_normalised_returns(
    success: torch.Tensor,
    episode_lengths: torch.Tensor,
    t: torch.Tensor,
    max_episode_length: int = 1040,
    c_fail: float = 100.0,
) -> torch.Tensor:
    """
    Computes the normalised empirical return  R_t(τ)  used to train the
    distributional value head (RECAP §IV-A, §V-C).
 
    Reward definition (RECAP Eq. 5):
        r_{t'}  =  0          if t' == T and success
                   -C_fail    if t' == T and failure
                   -1         otherwise
 
    Return from step t:
        R_t(τ)  =  Σ_{t'=t}^{T}  r_{t'}
                =  -(T - t)          if success    (negative steps remaining)
                   -(T - t) - C_fail if failure
 
    Normalised to (-1, 0) per task using max_episode_length (pi*0.6 §V-C):
        R_t_norm  =  R_t(τ) / (max_episode_length + C_fail)
 
    Args:
        success          : (B,)  bool  — was the episode successful?
        episode_lengths  : (B,)  long  — T for each episode
        t                : (B,)  long  — current timestep index
        max_episode_length: int        — normalisation constant per task
        c_fail           : float       — large penalty for failure (default 100)
 
    Returns:
        normalised_returns : (B,)  float in [-1, 0]
    """
    steps_remaining = (episode_lengths - t).float()          # T - t
    raw_return      = -steps_remaining                        # successful base
    raw_return      = torch.where(success, raw_return, raw_return - c_fail)
    norm_denom      = float(max_episode_length + c_fail)
    return (raw_return / norm_denom).clamp(-1.0, 0.0)


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )
    advantage_cfg_dropout_prob: float = field(
        default=0.3,
        metadata={
            "help": (
                "Probability of replacing the advantage token with the null token "
                "during training. "
                "Enables both conditional and unconditional inference at test time."
            )
        },
    )
    cfg_guidance_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "CFGRL guidance weight w "
                "w=1 samples directly from the advantage-conditioned policy. "
                "w>1 amplifies the optimality signal: "
                "  v_guided = v_null + w * (v_pos − v_null). "
                "Provably improves expected return for any w≥1 (CFGRL Remark 2). "
                "Typical useful range: [1.5, 2.5]"
            )
        },
    )
    hidden_dim: int = field(
        default=2048,
        metadata={}
    )
    num_bins: int = field(
        default=201,
        metadata={}
    )          # B = 201 as in pi*0.6
    value_loss_coeff: float = field(
        default=1.0,
        metadata={}
    )

    _phase: str = field(
        default="value_head" #action_head
    )

    recap_alpha: float = field(
        default=1.0,
        metadata={
            "help": (
                "alpha in RECAP Eq. 3: weight of conditional term relative to "
                "unconditional term. pi*0.6 uses alpha=1.0 (equal weight). "
                "Higher alpha -> stronger push toward advantage-conditioned behavior."
            )
        },
    )
    advantage_threshold_percentile: float = field(
        default=0.30,
        metadata={
            "help": (
                "Percentile of V predictions used as eps_l threshold. "
                "pi*0.6 App. F: 0.30 for pre-training, 0.40 for fine-tuning."
            )
        },
    )


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

        self.init_advantage_conditioning()
        self.init_value_head()

        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)    
        
    def load_state_dict(self, state_dict, strict=True, assign=False):
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # Skip old-style state_encoder weights
            if self.config._phase == "value_head":
                if key.startswith("value_head") or key.startswith("advantage_embedding"):
                    print(f"Skipping incompatible state_encoder weight: {key}")
                    continue
            filtered_state_dict[key] = value
        # print(filtered_state_dict)
        # Call parent's load_state_dict with filtered state
        return super().load_state_dict(filtered_state_dict, strict=False, assign=assign)

    def init_advantage_conditioning(self):
        self.advantage_embedding = AdvantageEmbedding(2048)
        print(
            f"[RECAP] Advantage conditioning ENABLED  "
            f"(cfg_dropout={self.config.advantage_cfg_dropout_prob}, "
            f"cfg_guidance_weight={self.config.cfg_guidance_weight})"
        )

    def init_value_head(self):
        vh_kwargs = {
            "backbone_embedding_dim": self.config.backbone_embedding_dim,
            "state_dim": self.config.max_state_dim,
            "seq_dim": self.config.backbone_embedding_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_bins": self.config.num_bins,
            "value_loss_coeff": self.config.value_loss_coeff,
        }
        vh_config = DistributionalValueHeadConfig(**vh_kwargs)
        self.value_head = DistributionalValueHead(vh_config)
        print(
            f"[RECAP] Distributional value head ENABLED  "
            f"(num_bins={vh_config.num_bins}, "
            f"value_loss_coeff={vh_config.value_loss_coeff} "
            f"hidden_dim={vh_config.hidden_dim}), "
        )

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_phase_value_head(self):
        """
        Phase 1 — RECAP Alg 1 lines 1, 4, 8:
            Train V on D using Eq. 1.
            Policy (DiT, encoders, decoders) frozen.
            Value head trainable.
        """
        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False
        # Unfreeze value head
        for p in self.value_head.parameters():
            p.requires_grad = True
        # Unfreeze advantage embedding (learns alongside value head)

        self._phase = "value_head"
        print("[RECAP] ── Phase 1: VALUE_HEAD ──")
        print(f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def set_phase_policy(self):
        """
        Phase 2 — RECAP Alg 1 lines 2, 5, 9:
            Train piθ on D using Eq. 3 and frozen V.
            Value head frozen (used only for labelling).
            Policy trainable (respects tune_projector / tune_diffusion_model).
        """
        # Restore policy trainability
        self.set_trainable_parameters(self.config.tune_projector, self.config.tune_diffusion_model)
        # Freeze value head
        for p in self.value_head.parameters():
            p.requires_grad = False
        # Advantage embedding always trainable in policy phase
        if self.advantage_embedding is not None:
            for p in self.advantage_embedding.parameters():
                p.requires_grad = True

        self._phase = "action_head"
        print("[RECAP] ── Phase 2: POLICY ──")
        print(f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output
 
    def _apply_advantage_conditioning(
        self,
        vl_embs: torch.Tensor,
        vl_attn_mask: torch.Tensor | None,
        advantage_label: torch.Tensor | None,
        *,
        force_null: bool = False,
    ):
        """
        Append a single advantage token to the cross-attention context.
 
        During training, a CFG dropout mask randomly replaces the supplied.
        """
        B, _, D = vl_embs.shape
        device = vl_embs.device
 
        if advantage_label is None or force_null:
            labels = torch.full(
                (B,), AdvantageEmbedding.NULL_IDX, dtype=torch.long, device=device
            )
        else:
            labels = advantage_label.to(device=device)
            # CFG dropout during training
            if self.training:
                drop = torch.rand(B, device=device) < self.config.advantage_cfg_dropout_prob
                labels = labels.masked_fill(drop, AdvantageEmbedding.NULL_IDX)
 
        adv_token = self.advantage_embedding(labels)          # (B, 1, D)
        vl_embs_aug = torch.cat([vl_embs, adv_token], dim=1) # (B, S+1, D)
 
        if vl_attn_mask is not None:
            extra = torch.ones(B, 1, dtype=vl_attn_mask.dtype, device=device)
            vl_attn_mask_aug = torch.cat([vl_attn_mask, extra], dim=1)
        else:
            vl_attn_mask_aug = None
 
        return vl_embs_aug, vl_attn_mask_aug

    @torch.no_grad()
    def compute_advantage_labels_from_value_head(
        self,
        backbone_feats: torch.Tensor,   # (B, S, D)  VL features (pre-advantage-token)
        state:          torch.Tensor,   # (B, state_dim)
        reward:         torch.Tensor,   # (B,)  {-1=fail, 0=success}
        percentile:     float = 0.30,   # eps_l threshold (30th pct pre-train, 40th fine-tune)
    ) -> torch.Tensor:
        """
        Derive per-step advantage indicator I_t from the frozen value head.

        RECAP App. F: eps_l is set at the 30th percentile of V values predicted
        by the value function for the current task l, so ~30% of steps get
        positive advantage (POS_IDX).

        Hard rule: failure steps (reward=-1) are ALWAYS labelled NEG regardless
        of their predicted value — failure is never positive advantage.

        Returns (B,) long tensor ∈ {NEG_IDX=1, POS_IDX=2}
        """
        V       = self.value_head.predict_value(backbone_feats, state)       # (B,)
        epsilon = torch.quantile(V, percentile)                          # scalar eps_l

        is_failure      = reward < 0                                     # (B,) bool
        above_threshold = V > epsilon                                    # (B,) bool

        labels = torch.where(
            above_threshold & ~is_failure,
            torch.full_like(V, AdvantageEmbedding.POS_IDX, dtype=torch.long),
            torch.full_like(V, AdvantageEmbedding.NEG_IDX, dtype=torch.long),
        )
        return labels                                                     # (B,)
    
    def forward_value_head(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        vl_attn_mask = backbone_output.backbone_attention_mask
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Inject advantage token
        assert "reward" in action_input.keys(), f"No reward found in {action_input.keys()=}"
        reward = torch.squeeze(action_input["reward"], dim=-1)

        t = action_input["reward.current_frame_idx"].squeeze(dim=-1)
        episode_lengths = action_input["reward.episode_lengths"].squeeze(dim=-1)

        max_episode_length = 1040 / 2

        empirical_return = compute_normalised_returns(
            success=torch.where(reward < 0, False, True),
            episode_lengths=episode_lengths,
            t=t,
            max_episode_length=max_episode_length,
            c_fail=max_episode_length / 2
        )
        if empirical_return is not None:
            # backbone_output.backbone_features is already vlln-processed
            # but has NOT had the advantage token appended — correct input.
            value_loss = self.value_head.compute_loss(
                vl_embs,
                action_input.state.float(),              # <- add state
                empirical_return,
            )
            output_dict = {
                "loss": value_loss,
            }
        return BatchFeature(data=output_dict)
    
    def forward_action_head(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        vl_attn_mask = backbone_output.backbone_attention_mask
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Inject advantage token
        assert "reward" in action_input.keys(), f"No reward found in {action_input.keys()=}"
        reward = torch.squeeze(action_input["reward"], dim=-1)

        adv_label = torch.where(reward < 0, AdvantageEmbedding.NEG_IDX, AdvantageEmbedding.POS_IDX)
        with torch.no_grad():
            adv_label = self.compute_advantage_labels_from_value_head(
            backbone_feats=vl_embs,
            state=action_input.state.float(),
            reward=reward,
            percentile=self.config.advantage_threshold_percentile
            if hasattr(self.config, "advantage_threshold_percentile")
            else 0.30,
            )

        pos_frac = (adv_label == AdvantageEmbedding.POS_IDX).float().mean()

        # --- Step 2: build noised trajectory (shared for both terms) ──────────
        # CFGRL Alg 1

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        # --- Step 3: UNCONDITIONAL term ──────────
        # Inject NULL token — no optimality signal

        vl_embs_null, vl_attn_mask_null = self._apply_advantage_conditioning(
            vl_embs, vl_attn_mask,
            advantage_label=None,    # -> all NULL_IDX
        )

        model_output_null = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs_null,
            encoder_attention_mask=vl_attn_mask_null,   # <- correct augmented mask
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred_null  = self.action_decoder(model_output_null, embodiment_id)
        pred_null  = pred_null[:, -actions.shape[1]:]

        action_mask = action_input.action_mask
        loss_uncond = F.mse_loss(pred_null, velocity, reduction="none") * action_mask
        loss_uncond = loss_uncond.sum() / action_mask.sum()

        # --- Step 4: CONDITIONAL term ──────────
        # Inject advantage token with CFG dropout

        vl_embs_cond, vl_attn_mask_cond = self._apply_advantage_conditioning(
            vl_embs, vl_attn_mask,
            advantage_label=adv_label,   # It in {NEG, POS} from value head
        )

        model_output_cond = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs_cond,
            encoder_attention_mask=vl_attn_mask_cond,   # <- correct augmented mask
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred_cond  = self.action_decoder(model_output_cond, embodiment_id)
        pred_cond  = pred_cond[:, -actions.shape[1]:]

        loss_cond = F.mse_loss(pred_cond, velocity, reduction="none") * action_mask
        loss_cond = loss_cond.sum() / action_mask.sum()

        # ── Step 5: total loss (Eq. 3) ─────────────────────────────────────────
        # L = loss_uncond + α * loss_cond
        total_loss = loss_uncond + self.config.recap_alpha * loss_cond

        output_dict = {
            "loss": total_loss,
            "action_loss_uncond": loss_uncond.detach(),
            "action_loss_cond":   loss_cond.detach(),
            "advantage_pos_frac": pos_frac.detach(),
        }
        return BatchFeature(data=output_dict)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        if self._phase == "action_head":
            output_dict = self.forward_action_head(backbone_output, action_input)
        elif self._phase == "value_head":
            output_dict = self.forward_value_head(backbone_output, action_input)
        return output_dict

    @torch.no_grad()
    def get_value(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features

        value = self.value_head.predict_value(
            backbone_features=vl_embs,
            state=action_input.state
        )

        return BatchFeature(data={"value_pred": value})

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        vl_attn_mask = backbone_output.backbone_attention_mask
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        # Prepare advantage-conditioned and unconditional encoder contexts
        w = self.config.cfg_guidance_weight
        dual_pass = (w != 1.0)
 
        # Conditional context: I_t = POS
        vl_embs_cond, vl_attn_mask_cond = self._apply_advantage_conditioning(
            vl_embs, vl_attn_mask,
            advantage_label=torch.full(
                (batch_size,), AdvantageEmbedding.POS_IDX, dtype=torch.long, device=device
            ),
        )
        if dual_pass:
            # Unconditional context: I_t = NULL
            vl_embs_null, vl_attn_mask_null = self._apply_advantage_conditioning(
                vl_embs, vl_attn_mask,
                advantage_label=None,   # -> all NULL_IDX
            )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs_cond,
                encoder_attention_mask=vl_attn_mask_cond,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            if dual_pass:
                model_output_uncond = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs_null,
                    encoder_attention_mask=vl_attn_mask_null,
                    timestep=timesteps_tensor,
                )
                pred_uncond = self.action_decoder(model_output_uncond, embodiment_id)
                pred = pred_uncond + w * (pred - pred_uncond)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})
    
    @torch.enable_grad()
    def get_realtime_action(
        self,
        action_input: BatchFeature,
        backbone_output:BatchFeature,
        prev_action_chunk: torch.Tensor,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str,
        max_guidance_weight: float,
        sigma_d_o: float,
        actual_action_dim: int,
        use_prev_action: bool = True
    )  -> BatchFeature:
        torch.set_grad_enabled(True)
        num_steps = self.num_inference_timesteps
        self.sigma_d_o = sigma_d_o
        dt = 1.0 / num_steps
        prev_action_chunk = torch.as_tensor(prev_action_chunk, device=self.device, dtype=self.dtype)

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        actions_mask = torch.zeros_like(x_t, dtype=vl_embs.dtype, device=device)
        actions_mask[:, :, :actual_action_dim] = True

        # weights: [horizon]
        weights = get_prefix_weights(
            inference_delay, prefix_attention_horizon, self.config.action_horizon, prefix_attention_schedule
        )
        weights = weights.to(device)

        for t in range(num_steps):

            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            
            def denoiser(x_t_):
                t_discretized = int(t_cont * self.num_timestep_buckets)

                # Embed noised action trajectory.
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(x_t_, timesteps_tensor, embodiment_id)
                # Maybe add position embedding.
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                # Join vision, language, state and action embedding along sequence dimension.
                future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

                # Run model forward.
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred = self.action_decoder(model_output, embodiment_id)

                pred_velocity = pred[:, -self.action_horizon :]
                return x_t_ + pred_velocity * dt, pred_velocity
            
            (outputs, vjp_func) = torch.func.vjp(denoiser, x_t)
            (x_1_i_vjp, v_t_i_vjp) = outputs
            error = (prev_action_chunk - x_1_i_vjp) * weights[:, None] * actions_mask
            # error = F.mse_loss(x_1_i_vjp, prev_action_chunk, reduction="none") / actions_mask.sum() * weights[:, None] * actions_mask
            
            pinv_correction = vjp_func((error, torch.zeros_like(x_t)))[0]
            if pinv_correction is None:
                pinv_correction = torch.zeros_like(x_1_i_vjp)
            inv_r2 = (self.sigma_d_o**2 * t_cont**2 + (1 - t_cont)**2) / (self.sigma_d_o**2 * (1 - t_cont)**2)
            # inv_r2 = (t_cont**2 + (1 - t_cont) ** 2) / ((1 - t_cont) ** 2)
            c = torch.nan_to_num(torch.tensor((1 - t_cont) / max(t_cont, 1e-12), device=self.device, dtype=self.dtype),  # Avoid division by zero
                                 nan=0.0, posinf=max_guidance_weight)
            
            guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
            guidance_weight = torch.minimum(guidance_weight, torch.tensor(max_guidance_weight, device=device))
            v_t_corr = v_t_i_vjp + guidance_weight * pinv_correction
            x_t = x_t + v_t_corr * dt

        assert x_t.shape == (batch_size, self.config.action_horizon, self.config.action_dim), x_t.shape
        x_t = x_t.clone().detach()
        print("EACH DENOISING STEP: ", (x_t[:,:inference_delay,:actual_action_dim] - prev_action_chunk[:,:inference_delay,:actual_action_dim]).abs().mean())
        if use_prev_action:
            x_t = prev_action_chunk * weights[:, None] + x_t * (1 - weights[:, None])
            print("AFTER ASSIGN: ", (x_t[:,:inference_delay,:actual_action_dim] - prev_action_chunk[:,:inference_delay,:actual_action_dim]).abs().mean())
        return BatchFeature(data={"action_pred": x_t})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
