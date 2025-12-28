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

from gr00t.model.action_head.action_encoder import SinusoidalPositionalEncoding, swish
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig, FlowmatchingActionHead
from .cross_attention_dit_action_condition import DiT, SelfAttentionTransformer


class FlowmatchingActionHeadActionCondition(FlowmatchingActionHead):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__(config=config)
        self.model = DiT(**config.diffusion_model_cfg)
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

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
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None].expand(actions.shape[0], actions.shape[1])  # shape (B,T) for broadcast

        ##################################### training-time action conditioning #####################################
        #                                                                                                           #
        
        # sample delays from some distribution of choice
        # here, we use Unif[0, max_delay), as in our real-world experiments
        delay = torch.randint(0, actions.shape[1] // 2, (actions.shape[0],), device=device)
        prefix_mask = torch.arange(actions.shape[1], device=device)[None, :] < delay[:, None]
        t = torch.where(prefix_mask, 1.0, t) # set time to 1.0 for the action prefix

        #                                                                                                           #
        ##################################### training-time action conditioning #####################################

        noisy_trajectory = (1 - t[:, :, None]) * noise + t[:, :, None] * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t * self.num_timestep_buckets).long()
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

        t_discretized_expand = t_discretized[:, -1][:, None].expand(sa_embs.shape[0], sa_embs.shape[1] - t_discretized.shape[1])
        t_discretized = torch.cat([t_discretized_expand, t_discretized], dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        postfix_mask = torch.logical_not(prefix_mask)[:, :, None] # compute the loss on the postfix only
        total_mask = action_mask & postfix_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * total_mask
        loss = loss.sum() / (total_mask.sum() + 1e-8)
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)
    
    @torch.no_grad()
    def get_action(self, backbone_output, action_input):
        prev_action_chunk = torch.zeros((
                backbone_output.backbone_features.shape[0], self.config.action_horizon, self.config.action_dim
            ), device=backbone_output.backbone_features.device
        )
        inference_delay = 0
        return self.get_realtime_action(
            action_input,
            backbone_output,
            prev_action_chunk=prev_action_chunk,
            inference_delay=inference_delay,
            prefix_attention_horizon=None,
            prefix_attention_schedule=None,
            max_guidance_weight=None,
            sigma_d_o=None,
            actual_action_dim=None,
        )
    
    @torch.no_grad()
    def get_realtime_action(
        self,
        action_input: BatchFeature,
        backbone_output: BatchFeature,
        prev_action_chunk: torch.Tensor,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: Optional[int] = None,
        prefix_attention_schedule: Optional[str] = None,
        max_guidance_weight: Optional[float] = None,
        sigma_d_o: Optional[float] = None,
        actual_action_dim: Optional[int] = None,
        use_prev_action: bool = True,
    ) -> BatchFeature:

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

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        delay = torch.full((actions.shape[0],), fill_value=inference_delay, device=device)
        prefix_mask = torch.arange(actions.shape[1], device=device)[None, :] < delay[:, None]

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_cont = torch.tensor([t_cont], device=device)
            t_cont_expand = t_cont.expand(actions.shape[0], actions.shape[1])
            t_cont_expand = torch.where(prefix_mask, 1.0, t_cont_expand).to(device) # set time to 1.0 for the action prefix
            t_discretized = (t_cont_expand * self.num_timestep_buckets).long()

            actions = torch.where(prefix_mask[:, :, None], prev_action_chunk, actions).to(device)

            action_features = self.action_encoder(actions, t_discretized, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            t_discretized_expand = t_discretized[:, -1][:, None].expand(sa_embs.shape[0], sa_embs.shape[1] - t_discretized.shape[1])
            t_discretized = torch.cat([t_discretized_expand, t_discretized], dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

        actions = torch.where(prefix_mask[:, :, None], prev_action_chunk, actions).to(device)
        return BatchFeature(data={"action_pred": actions})
