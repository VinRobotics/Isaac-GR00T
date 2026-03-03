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
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import AutoProcessor, PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

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


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: Optional[dict] = field(
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
    action_dim: Optional[int] = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: Optional[int] = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: Optional[int] = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: Optional[dict] = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )
    
    # Torque aware (original concatenation approach)
    torque_aware: bool = field(
        default=False, metadata={"help": "Whether to use torque-aware training."}
    )
    effort_dim: Optional[int] = field(
        default=None, metadata={"help": "Effort dimension."}
    )

    # FAST effort: history as conditioning tokens + AR decoder for future effort prediction
    use_fast_effort: bool = field(
        default=False,
        metadata={"help": "Use FAST tokenizer for effort. "
                           "History tokens condition the DiT; an AR decoder predicts future effort."},
    )
    fast_tokenizer_path: str = field(
        default="physical-intelligence/fast",
        metadata={"help": "HuggingFace model ID or local path for the FAST action tokenizer."},
    )
    fast_num_tokens: int = field(
        default=80,
        metadata={"help": "Fixed FAST token sequence length per sample (pad/truncate to this). "
                           "For (T=16, D=7) ~67 tokens are produced; for (T=16, D=26) ~243."},
    )
    fast_effort_history_horizon: int = field(
        default=16, metadata={"help": "Number of past effort timesteps used as FAST history."}
    )
    fast_effort_loss_coeff: float = field(
        default=1.0, metadata={"help": "Weight of the AR effort cross-entropy loss."}
    )
    fast_ar_num_layers: int = field(
        default=2, metadata={"help": "Number of Transformer Decoder layers in the AR effort head."}
    )
    fast_ar_emb_dim: int = field(
        default=256, metadata={"help": "Hidden dimension of the AR effort decoder."}
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
        self.torque_aware = config.torque_aware
        self.action_dim = config.action_dim
        self.effort_dim = config.effort_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim if not self.torque_aware else self.action_dim + self.effort_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim if not self.torque_aware else self.action_dim + self.effort_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # ── FAST effort modules ──────────────────────────────────────────────
        self.use_fast_effort = config.use_fast_effort
        if self.use_fast_effort:
            assert config.effort_dim is not None, "effort_dim must be set when use_fast_effort=True"
            self.fast_tokenizer = AutoProcessor.from_pretrained(
                config.fast_tokenizer_path, trust_remote_code=True
            )
            self.fast_min_token: int = self.fast_tokenizer.min_token   # e.g. -354
            # shifted token range: [0, emb_size-1]
            # special tokens: PAD=emb_size (embedding pad, CE uses -100), BOS=emb_size+1
            # EOS class in cls head output: emb_size (last logit index)
            self.fast_emb_size: int = self.fast_tokenizer.vocab_size - self.fast_min_token
            self.fast_pad_token: int = self.fast_emb_size          # embedding padding_idx
            self.fast_bos_token: int = self.fast_emb_size + 1      # AR start token
            self.fast_eos_cls: int = self.fast_emb_size             # EOS class in cls head
            self.n_fast: int = config.fast_num_tokens

            ar_emb_dim = config.fast_ar_emb_dim
            vocab_with_special = self.fast_emb_size + 2  # + PAD + BOS

            # History tokens → sa_embs conditioning (input_embedding_dim space)
            self.fast_hist_token_emb = nn.Embedding(
                vocab_with_special, self.input_embedding_dim, padding_idx=self.fast_pad_token
            )
            nn.init.normal_(self.fast_hist_token_emb.weight, std=0.02)
            self.fast_hist_pos_emb = SinusoidalPositionalEncoding(self.input_embedding_dim)

            # AR decoder modules (ar_emb_dim space, kept small for efficiency)
            self.fast_ar_effort_emb = nn.Embedding(
                vocab_with_special, ar_emb_dim, padding_idx=self.fast_pad_token
            )
            nn.init.normal_(self.fast_ar_effort_emb.weight, std=0.02)
            self.fast_ar_pos_emb = SinusoidalPositionalEncoding(ar_emb_dim)

            # Project DiT future_tokens output → AR decoder memory dim
            self.fast_context_proj = nn.Linear(self.hidden_size, ar_emb_dim)

            # Causal Transformer Decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=ar_emb_dim,
                nhead=max(1, ar_emb_dim // 64),  # e.g. 4 heads for dim=256
                dim_feedforward=4 * ar_emb_dim,
                dropout=0.0,
                batch_first=True,
            )
            self.fast_ar_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=config.fast_ar_num_layers
            )
            # Classification head: ar_emb_dim → emb_size+1 (real tokens + EOS)
            self.fast_cls_head = nn.Linear(ar_emb_dim, self.fast_emb_size + 1)
            print(
                f"FAST effort: n_fast={self.n_fast}, emb_size={self.fast_emb_size}, "
                f"ar_emb_dim={ar_emb_dim}, ar_layers={config.fast_ar_num_layers}"
            )

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
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)
        print("Using torque aware", self.action_dim if not self.torque_aware else self.action_dim + self.effort_dim)

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
            if self.use_fast_effort:
                self.fast_hist_token_emb.requires_grad_(False)
                self.fast_ar_effort_emb.requires_grad_(False)
                self.fast_context_proj.requires_grad_(False)
                self.fast_ar_decoder.requires_grad_(False)
                self.fast_cls_head.requires_grad_(False)
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

    def expand_action_weights_for_torque_aware(self, pretrained_state_dict: dict, old_action_dim: int, new_action_dim: int):
        """
        Expand action encoder and decoder weights to accommodate effort dimensions.
        This preserves pretrained weights for action dimensions and randomly initializes effort dimensions.
        
        Args:
            pretrained_state_dict: State dict from pretrained model
            old_action_dim: Original action dimension (from pretrained model)
            new_action_dim: New action dimension (action_dim + effort_dim)
        """
        
        # For action_encoder.W1: shape is (num_embodiments, action_dim, hidden_size)
        # We need to expand the action_dim dimension
        if 'action_encoder.W1.W' in pretrained_state_dict:
            old_W1_W = pretrained_state_dict['action_encoder.W1.W']  # (num_emb, old_action_dim, hidden_size)
            num_emb, _, hidden_size = old_W1_W.shape
            
            # Create new tensor with expanded dimension
            new_W1_W = torch.randn(num_emb, new_action_dim, hidden_size) * 0.02
            # Copy pretrained weights for action portion
            new_W1_W[:, :old_action_dim, :] = old_W1_W
            
            pretrained_state_dict['action_encoder.W1.W'] = new_W1_W
            print(f"  Expanded action_encoder.W1.W: {old_W1_W.shape} -> {new_W1_W.shape}")
        
        # For action_decoder.layer2: output_dim needs to be expanded
        # shape is (num_embodiments, hidden_size, output_dim)
        if 'action_decoder.layer2.W' in pretrained_state_dict:
            old_layer2_W = pretrained_state_dict['action_decoder.layer2.W']  # (num_emb, hidden_size, old_action_dim)
            num_emb, hidden_size, _ = old_layer2_W.shape
            
            # Create new tensor with expanded output dimension
            new_layer2_W = torch.randn(num_emb, hidden_size, new_action_dim) * 0.02
            # Copy pretrained weights for action portion
            new_layer2_W[:, :, :old_action_dim] = old_layer2_W
            
            pretrained_state_dict['action_decoder.layer2.W'] = new_layer2_W
            print(f"  Expanded action_decoder.layer2.W: {old_layer2_W.shape} -> {new_layer2_W.shape}")
        
        if 'action_decoder.layer2.b' in pretrained_state_dict:
            old_layer2_b = pretrained_state_dict['action_decoder.layer2.b']  # (num_emb, old_action_dim)
            num_emb, _ = old_layer2_b.shape
            
            # Create new tensor with expanded dimension
            new_layer2_b = torch.zeros(num_emb, new_action_dim)
            # Copy pretrained biases for action portion
            new_layer2_b[:, :old_action_dim] = old_layer2_b
            
            pretrained_state_dict['action_decoder.layer2.b'] = new_layer2_b
            print(f"  Expanded action_decoder.layer2.b: {old_layer2_b.shape} -> {new_layer2_b.shape}")
        
        return pretrained_state_dict

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

    def _fast_tokenize(
        self, effort_tensor: torch.Tensor, n_fast: int, truncate: bool = True
    ) -> torch.Tensor:
        """Tokenize an effort tensor using the FAST tokenizer.

        Args:
            effort_tensor: (B, T, D) float tensor on any device, values in [-1, 1]
            n_fast: sequence length when truncate=True (pad/clip to this)
            truncate: if True, clip tokens to n_fast (used for history conditioning);
                      if False, keep the full token sequence and pad to the max actual
                      length in the batch (used for future effort targets).
        Returns:
            (B, n_fast) when truncate=True, or (B, max_actual_len) when truncate=False.
            Padding positions are filled with self.fast_pad_token.
        """
        effort_np = effort_tensor.detach().cpu().float().numpy()
        tokens_list = self.fast_tokenizer(effort_np)   # list[list[int]], variable length
        B = len(tokens_list)
        if truncate:
            seq_len = n_fast
            toks_slices = [toks[:n_fast] for toks in tokens_list]
        else:
            seq_len = max(len(toks) for toks in tokens_list) if tokens_list else 0
            toks_slices = tokens_list
        result = torch.full((B, seq_len), self.fast_pad_token, dtype=torch.long)
        for i, toks in enumerate(toks_slices):
            shifted = [min(t - self.fast_min_token, self.fast_emb_size - 1) for t in toks]
            result[i, : len(shifted)] = torch.tensor(shifted, dtype=torch.long)
        return result  # (B, seq_len) on CPU

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
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
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        # torque_aware concatenates effort with action; use_fast_effort handles effort separately.
        if self.torque_aware and not self.use_fast_effort:
            actions = torch.concat([action_input.action, action_input.effort], dim=-1)
        else:
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

        # Embed FAST effort history tokens for conditioning (prepended to sa_embs).
        B = vl_embs.shape[0]
        hist_embs: Optional[torch.Tensor] = None
        if self.use_fast_effort:
            hist_tokens = self._fast_tokenize(
                action_input.effort_history[..., : self.effort_dim], self.n_fast
            ).to(device)  # (B, n_fast) int64

            hist_len = hist_tokens.shape[1]
            hist_pos = torch.arange(hist_len, device=device).unsqueeze(0)  # (1, hist_len)
            hist_embs = (
                self.fast_hist_token_emb(hist_tokens)          # (B, hist_len, emb_dim)
                + self.fast_hist_pos_emb(hist_pos)             # (1, hist_len, emb_dim)
            )

        # Join vision, language, state and action embedding along sequence dimension.
        # Layout: [state(1) | hist(n_fast) | future_tokens(n_ft) | action(T)]  (fast_effort)
        # Layout: [state(1) | future_tokens(n_ft) | action(T)]                  (standard)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        if self.use_fast_effort and hist_embs is not None:
            sa_embs = torch.cat((state_features, hist_embs, future_tokens, action_features), dim=1)
        else:
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        # Keep full prediction (including effort dims when torque_aware) and then slice the tail
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target for action loss.
        action_mask = action_input.action_mask

        pred_only_actions = pred_actions[..., : self.action_dim]
        velocity_actions = velocity[..., : self.action_dim]
        action_loss = F.mse_loss(pred_only_actions, velocity_actions, reduction="none") * action_mask
        action_loss_scalar = action_loss.sum() / action_mask.sum()
        loss = action_loss_scalar

        if self.torque_aware and not self.use_fast_effort:
            # For efforts we must slice from the full pred_actions tensor (which still contains effort dims)
            effort_mask = action_input.effort_mask
            pred_efforts = pred_actions[..., self.action_dim :]
            velocity_efforts = velocity[..., self.action_dim :]
            effort_loss = F.mse_loss(pred_efforts, velocity_efforts, reduction="none") * effort_mask
            loss = loss + 0.1 * (effort_loss.sum() / effort_mask.sum())

        effort_ar_loss: Optional[torch.Tensor] = None
        if self.use_fast_effort:
            # ── AR effort decoder ────────────────────────────────────────────
            # Context: DiT output at future_tokens positions
            n_ft = self.config.num_target_vision_tokens
            ft_start = 1 + self.n_fast   # after state(1) + hist(n_fast)
            ft_end = ft_start + n_ft
            context = self.fast_context_proj(
                model_output[:, ft_start:ft_end, :]
            )  # (B, n_ft, ar_emb_dim)

            # Tokenise future effort → training targets, full sequence (no truncation)
            target_tokens = self._fast_tokenize(
                action_input.effort[..., : self.effort_dim], self.n_fast, truncate=False
            ).to(device)

            # Append EOS class to each target sequence: (B, n_real) → (B, n_real+1)
            eos_col = torch.full((B, 1), self.fast_eos_cls, dtype=torch.long, device=device)
            target_with_eos = torch.cat([target_tokens, eos_col], dim=1)
            target_len = target_with_eos.shape[1]

            # AR input (teacher forcing): [BOS, tok_0, ..., tok_{n_real-1}]
            # EOS is the prediction target of the last step, not an input token.
            bos_ids = torch.full((B, 1), self.fast_bos_token, dtype=torch.long, device=device)
            ar_input_ids = torch.cat([bos_ids, target_tokens], dim=1)  # (B, target_len)
            ar_pos = torch.arange(target_len, device=device).unsqueeze(0)
            ar_input_embs = (
                self.fast_ar_effort_emb(ar_input_ids)   # (B, target_len, ar_emb_dim)
                + self.fast_ar_pos_emb(ar_pos)          # (1, target_len, ar_emb_dim)
            )

            # Causal mask: each position can only attend to previous positions
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                target_len, device=device
            )  # (target_len, target_len)
            ar_out = self.fast_ar_decoder(
                tgt=ar_input_embs,
                memory=context,
                tgt_mask=causal_mask,
                tgt_is_causal=True,
            )  # (B, target_len, ar_emb_dim)

            effort_logits = self.fast_cls_head(ar_out)  # (B, target_len, emb_size+1)
            _ar_loss: torch.Tensor = F.cross_entropy(
                effort_logits.reshape(B * target_len, self.fast_emb_size + 1),
                target_with_eos.reshape(B * target_len),
                ignore_index=-100,  # -100 for any padding (shouldn't occur for fixed T,D)
            )
            effort_ar_loss = _ar_loss
            loss = loss + self.config.fast_effort_loss_coeff * _ar_loss

        output_dict = {
            "loss": loss,
            "action_loss": action_loss_scalar,
        }
        if effort_ar_loss is not None:
            output_dict["effort_ar_loss"] = effort_ar_loss
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        predict_effort: bool = True,
    ) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        batch_size = vl_embs.shape[0]
        device = vl_embs.device

        # Pre-compute FAST effort history embeddings (fixed across all denoising steps).
        hist_embs: Optional[torch.Tensor] = None
        if self.use_fast_effort:
            hist_tokens = self._fast_tokenize(
                action_input.effort_history[..., : self.effort_dim], self.n_fast
            ).to(device)

            hist_len = hist_tokens.shape[1]
            hist_pos = torch.arange(hist_len, device=device).unsqueeze(0)
            hist_embs = (
                self.fast_hist_token_emb(hist_tokens)
                + self.fast_hist_pos_emb(hist_pos)
            )  # (B, hist_len, emb_dim)

        # When torque_aware (legacy), noise covers both action and effort dims.
        if self.torque_aware and not self.use_fast_effort:
            action_effort_dim = self.config.action_dim + self.config.effort_dim
        else:
            action_effort_dim = self.config.action_dim

        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, action_effort_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        model_output: Optional[torch.Tensor] = None  # set inside loop; valid after num_steps >= 1

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            if self.use_fast_effort and hist_embs is not None:
                sa_embs = torch.cat((state_features, hist_embs, future_tokens, action_features), dim=1)
            else:
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity

        # ── Post-denoising effort prediction via AR decoder ──────────────────
        if self.use_fast_effort and predict_effort:
            assert model_output is not None, "model_output is None — num_inference_timesteps must be >= 1"
            n_ft = self.config.num_target_vision_tokens
            ft_start = 1 + self.n_fast
            ft_end = ft_start + n_ft
            context = self.fast_context_proj(
                model_output[:, ft_start:ft_end, :]
            )  # (B, n_ft, ar_emb_dim)

            # Greedy AR decoding: start with BOS, stop when EOS is predicted
            generated_ids = torch.full(
                (batch_size, 1), self.fast_bos_token, dtype=torch.long, device=device
            )
            active = torch.ones(batch_size, dtype=torch.bool, device=device)
            max_steps = self.n_fast + 1  # +1 to allow the EOS token itself
            for _ in range(max_steps):
                step_len = generated_ids.shape[1]
                step_pos = torch.arange(step_len, device=device).unsqueeze(0)
                embs = (
                    self.fast_ar_effort_emb(generated_ids)
                    + self.fast_ar_pos_emb(step_pos)
                )  # (B, step_len, ar_emb_dim)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    step_len, device=device
                )
                ar_out = self.fast_ar_decoder(
                    tgt=embs,
                    memory=context,
                    tgt_mask=causal_mask,
                    tgt_is_causal=True,
                )  # (B, step_len, ar_emb_dim)
                next_logit = self.fast_cls_head(ar_out[:, -1, :])    # (B, emb_size+1)
                next_token = next_logit.argmax(dim=-1, keepdim=True)  # (B, 1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                # Stop once all samples in the batch have emitted EOS
                active &= (next_token.squeeze(-1) != self.fast_eos_cls)
                if not active.any():
                    break

            # Drop BOS; for each sample strip real tokens up to (not including) EOS
            pred_ids = generated_ids[:, 1:].cpu().tolist()
            pred_unshifted = []
            for toks in pred_ids:
                real = [t for t in toks if t != self.fast_eos_cls]
                pred_unshifted.append([t + self.fast_min_token for t in real])
            effort_np = self.fast_tokenizer.decode(
                pred_unshifted,
                time_horizon=self.config.action_horizon,
                action_dim=self.effort_dim,
            )
            effort_pred = torch.tensor(
                np.asarray(effort_np), dtype=vl_embs.dtype, device=device
            )  # (B, action_horizon, effort_dim)
            return BatchFeature(data={"action_pred": actions, "effort_pred": effort_pred})

        # Legacy torque_aware: effort was denoised jointly with action
        if self.torque_aware:
            action_pred = actions[:, :, : self.config.action_dim]
            effort_pred = actions[:, :, self.config.action_dim :]
            return BatchFeature(data={"action_pred": action_pred, "effort_pred": effort_pred})

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
    )  -> BatchFeature:
        torch.set_grad_enabled(True)
        num_steps = self.num_inference_timesteps
        self.sigma_d_o = 1.0
        dt = 1.0 / num_steps
        prev_action_chunk = torch.as_tensor(prev_action_chunk, device=self.device, dtype=self.dtype)

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds      = backbone_output.backbone_features          # [B, T_vl, D]
        embodiment_id  = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        x_t = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        for t in range(num_steps):
            # weights: [horizon]
            weights = get_prefix_weights(
                inference_delay, prefix_attention_horizon, self.config.action_horizon, prefix_attention_schedule
            )
            weights = weights.to(device)

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

                vl_embs = vl_embeds

                # Join vision, language, state and action embedding along sequence dimension.
                sa_embs = torch.cat((state_features, action_features), dim=1)

                # Run model forward.
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred = self.action_decoder(model_output, embodiment_id)

                pred_velocity = pred[:, -self.action_horizon :]
                return x_t_ + pred_velocity * (1 - t_cont), pred_velocity
            
            vjp_res = torch.func.vjp(denoiser, x_t)
            # torch.func.vjp may return a tuple of length 2 or 3 depending on PyTorch version; index safely
            outputs = vjp_res[0]
            vjp_func = vjp_res[1]
            (x_1_i_vjp, v_t_i_vjp) = outputs
            error = (prev_action_chunk - x_1_i_vjp) * weights[:, None]
            
            pinv_correction = vjp_func((error, torch.zeros_like(x_t)))[0]
            if pinv_correction is None:
                pinv_correction = torch.zeros_like(x_1_i_vjp)
            inv_r2 = (self.sigma_d_o**2 * t_cont**2 + (1 - t_cont)**2) / (self.sigma_d_o**2 * (1 - t_cont)**2)
            # inv_r2 = (t_cont**2 + (1 - t_cont) ** 2) / ((1 - t_cont) ** 2)
            c = torch.nan_to_num(torch.tensor((1 - t_cont) / max(t_cont, 1e-12), device=self.device, dtype=self.dtype),  # Avoid division by zero
                                 nan=0.0, posinf=max_guidance_weight)
            
            guidance_weight = torch.minimum(c * inv_r2, torch.tensor(max_guidance_weight, device=device))
            v_t_corr = v_t_i_vjp + guidance_weight * pinv_correction

            x_t = x_t + dt * v_t_corr

        assert x_t.shape == (batch_size, self.config.action_horizon, self.config.action_dim), x_t.shape
        x_t = x_t.clone().detach()
        return BatchFeature(data={"action_pred": x_t})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
