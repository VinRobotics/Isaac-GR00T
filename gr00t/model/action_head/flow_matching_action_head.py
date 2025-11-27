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

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
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
    
    # Torque aware
    torque_aware: bool = field(
        default=False, metadata={"help": "Whether to use torque-aware training."}
    )
    effort_dim: Optional[int] = field(
        default=None, metadata={"help": "Effort dimension."}
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
        if self.torque_aware:
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

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
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

        pred_only_actions = pred_actions[..., :self.action_dim]
        velocity_actions = velocity[..., :self.action_dim]
        action_loss = F.mse_loss(pred_only_actions, velocity_actions, reduction="none") * action_mask
        loss = (action_loss.sum() / action_mask.sum())

        if self.torque_aware:
            # For efforts we must slice from the full pred_actions tensor (which still contains effort dims)
            effort_mask = action_input.effort_mask
            pred_efforts = pred_actions[..., self.action_dim:]
            velocity_efforts = velocity[..., self.action_dim:]
            # Ensure masks and tensors broadcast correctly. effort_mask expected shape matches effort dims.
            effort_loss = F.mse_loss(pred_efforts, velocity_efforts, reduction="none") * effort_mask
            loss += 0.1 * (effort_loss.sum() / effort_mask.sum())
        
        output_dict = {
            "loss": loss,
            "action_loss": action_loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        
        # When torque_aware, we predict both action and effort
        if self.torque_aware:
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
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        
        # If torque_aware, split the output into action and effort
        if self.torque_aware:
            action_pred = actions[:, :, :self.config.action_dim]
            effort_pred = actions[:, :, self.config.action_dim:]
            return BatchFeature(data={"action_pred": action_pred, "effort_pred": effort_pred})
        else:
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
        sigma_d_o = 0.5
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
            # inv_r2 = (sigma_d_o**2 * t_cont**2 + (1 - t_cont)**2) / (sigma_d_o**2 * (1 - t_cont)**2)
            inv_r2 = (t_cont**2 + (1 - t_cont) ** 2) / ((1 - t_cont) ** 2)
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
