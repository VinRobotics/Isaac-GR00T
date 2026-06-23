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

import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from gr00t.model.modules.recap import (
    AdvantageEmbedding,
    DistributionalValueHead,
    DistributionalValueHeadConfig,
    compute_normalised_returns,
)


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        # RECAP components — initialised only when use_recap=True
        self._phase = "action_head"
        self.advantage_embedding: AdvantageEmbedding | None = None
        self.value_head: DistributionalValueHead | None = None
        if getattr(config, "use_recap", False):
            self._init_recap()

        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
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
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

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
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    # ------------------------------------------------------------------
    # RECAP helpers
    # ------------------------------------------------------------------

    def _init_recap(self):
        """Initialise advantage embedding and distributional value head."""
        emb_dim = self.config.backbone_embedding_dim
        self.advantage_embedding = AdvantageEmbedding(emb_dim)
        logger.info(
            f"[RECAP] Advantage embedding ENABLED  "
            f"(cfg_dropout={self.config.advantage_cfg_dropout_prob}, "
            f"cfg_guidance_weight={self.config.cfg_guidance_weight})"
        )

        state_dim = self.config.max_state_dim * self.config.state_history_length
        vh_config = DistributionalValueHeadConfig(
            seq_dim=emb_dim,
            state_dim=state_dim,
            hidden_dim=self.config.value_head_hidden_dim,
            num_bins=self.config.num_bins,
            value_loss_coeff=self.config.value_loss_coeff,
        )
        self.value_head = DistributionalValueHead(vh_config)
        logger.info(
            f"[RECAP] Distributional value head ENABLED  "
            f"(num_bins={vh_config.num_bins}, hidden_dim={vh_config.hidden_dim})"
        )

    def set_phase_value_head(self):
        """
        Phase 1 — RECAP Alg 1 lines 1, 4, 8:
            Train Vϕ on D using Eq. 1.  Policy frozen; value head trainable.
        """
        assert self.value_head is not None, "set use_recap=True in config before calling this"
        for p in self.parameters():
            p.requires_grad = False
        for p in self.value_head.parameters():
            p.requires_grad = True
        self._phase = "value_head"
        logger.info("[RECAP] ── Phase 1: VALUE_HEAD ──")
        logger.info(
            f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )

    def set_phase_policy(self):
        """
        Phase 2 — RECAP Alg 1 lines 2, 5, 9:
            Train πθ on D using Eq. 3 with frozen Vϕ.
        """
        assert self.value_head is not None, "set use_recap=True in config before calling this"
        self.set_trainable_parameters(
            self.config.tune_projector, self.config.tune_diffusion_model, self.config.tune_vlln
        )
        for p in self.value_head.parameters():
            p.requires_grad = False
        if self.advantage_embedding is not None:
            for p in self.advantage_embedding.parameters():
                p.requires_grad = True
        self._phase = "action_head"
        logger.info("[RECAP] ── Phase 2: POLICY ──")
        logger.info(
            f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )

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

        During training, CFG dropout randomly replaces supplied labels with NULL_IDX.

        Returns:
            vl_embs_aug:      (B, S+1, D)
            vl_attn_mask_aug: (B, S+1) or None
        """
        B = vl_embs.shape[0]
        device = vl_embs.device

        if advantage_label is None or force_null:
            labels = torch.full((B,), AdvantageEmbedding.NULL_IDX, dtype=torch.long, device=device)
        else:
            labels = advantage_label.to(device=device)
            if self.training:
                drop = torch.rand(B, device=device) < self.config.advantage_cfg_dropout_prob
                labels = labels.masked_fill(drop, AdvantageEmbedding.NULL_IDX)

        adv_token = self.advantage_embedding(labels).to(dtype=vl_embs.dtype)  # (B, 1, D)
        vl_embs_aug = torch.cat([vl_embs, adv_token], dim=1)                  # (B, S+1, D)

        if vl_attn_mask is not None:
            extra = torch.ones(B, 1, dtype=vl_attn_mask.dtype, device=device)
            vl_attn_mask_aug = torch.cat([vl_attn_mask, extra], dim=1)
        else:
            vl_attn_mask_aug = None

        return vl_embs_aug, vl_attn_mask_aug

    def _run_model_forward(
        self,
        sa_embs: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        t_discretized: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        """
        Single DiT forward pass dispatching to DiT or AlternateVLDiT.

        For AlternateVLDiT, extends image_mask and backbone_attention_mask to
        account for the appended advantage token (treated as non-image text).
        """
        S_aug = encoder_hidden_states.shape[1]

        if self.config.use_alternate_vl_dit:
            orig_S = backbone_output.image_mask.shape[1]
            extra = S_aug - orig_S  # number of appended tokens (0 or 1)

            image_mask = backbone_output.image_mask
            backbone_attn = backbone_output.backbone_attention_mask

            if extra > 0:
                # Advantage token: not an image token; we do attend to it
                B, device = image_mask.shape[0], image_mask.device
                image_mask = torch.cat(
                    [image_mask, torch.zeros(B, extra, dtype=image_mask.dtype, device=device)],
                    dim=1,
                )
                backbone_attn = torch.cat(
                    [
                        backbone_attn,
                        torch.ones(B, extra, dtype=backbone_attn.dtype, device=device),
                    ],
                    dim=1,
                )

            return self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=t_discretized,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attn,
            )
        else:
            return self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=t_discretized,
            )

    @torch.no_grad()
    def compute_advantage_labels_from_value_head(
        self,
        backbone_feats: torch.Tensor,
        state: torch.Tensor,
        reward: torch.Tensor,
        percentile: float = 0.30,
    ) -> torch.Tensor:
        """
        Derive per-step advantage indicator I_t from the frozen value head.

        RECAP App. F: ε_l is the 30th percentile of V for the current task,
        so ~30% of steps get positive advantage.  Failure steps (reward < 0)
        are always labelled NEG regardless of predicted value.

        Returns: (B,) long tensor ∈ {NEG_IDX=1, POS_IDX=2}
        """
        V = self.value_head.predict_value(backbone_feats, state)
        epsilon = torch.quantile(V, percentile)

        is_failure = reward < 0
        above_threshold = V > epsilon

        return torch.where(
            above_threshold & ~is_failure,
            torch.full_like(V, AdvantageEmbedding.POS_IDX, dtype=torch.long),
            torch.full_like(V, AdvantageEmbedding.NEG_IDX, dtype=torch.long),
        )

    def forward_value_head(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> dict:
        """
        Phase 1 — RECAP §IV-A Eq. 1: train distributional value head Vϕ.

        Required action_input keys:
            reward                    : (B, 1) or (B,)   {-1=fail, 0=success}
            reward.current_frame_idx  : (B, 1)   step index t
            reward.episode_lengths    : (B, 1)   episode length T
        """
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embs = backbone_output.backbone_features

        reward = torch.squeeze(action_input["reward"], dim=-1).float()
        t_idx = action_input["reward.current_frame_idx"].squeeze(dim=-1).long()
        ep_len = action_input["reward.episode_lengths"].squeeze(dim=-1).long()

        max_ep = 520
        c_fail = 260.0
        empirical_return = compute_normalised_returns(
            success=reward >= 0,
            episode_lengths=ep_len,
            t=t_idx,
            max_episode_length=max_ep,
            c_fail=c_fail,
        )

        state = action_input.state.float()
        value_loss = self.value_head.compute_loss(vl_embs, state, empirical_return)

        return {"loss": value_loss, "value_loss": value_loss.detach()}

    def forward_action_head_recap(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> dict:
        """
        Phase 2 — RECAP §IV-B Eq. 3: train policy πθ with advantage conditioning.

        L = ||vθ(a_t,t,s,∅) − v||²  +  α * ||vθ(a_t,t,s,I_t) − v||²

        I_t derived from frozen value head.  CFG dropout trains both branches.

        Required action_input keys:
            state          : (B, state_history_length, max_state_dim)
            action         : (B, action_horizon, action_dim)
            action_mask    : (B, action_horizon, action_dim)
            embodiment_id  : (B,)
            reward         : (B, 1) or (B,)
        """
        self.set_frozen_modules_to_eval_mode()
        self.value_head.eval()

        backbone_output = self.process_backbone_output(backbone_output)

        vl_embs = backbone_output.backbone_features
        vl_attn_mask = backbone_output.backbone_attention_mask
        device = vl_embs.device
        embodiment_id = action_input.embodiment_id

        reward = torch.squeeze(action_input["reward"], dim=-1).float()

        # Step 1: advantage labels from frozen value head
        with torch.no_grad():
            adv_labels = self.compute_advantage_labels_from_value_head(
                backbone_feats=vl_embs,
                state=action_input.state.float(),
                reward=reward,
                percentile=self.config.advantage_threshold_percentile,
            )
        pos_frac = (adv_labels == AdvantageEmbedding.POS_IDX).float().mean()

        # Step 2: shared noised trajectory
        # Reshape state: (B, H, max_state_dim) -> (B, 1, H*max_state_dim)
        state = action_input.state.view(action_input.state.shape[0], 1, -1)
        state_features = self.state_encoder(state, embodiment_id)

        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=device) < self.state_dropout_prob
            )
            state_features = state_features * (1 - do_dropout[:, None, None].to(state_features.dtype))

        actions = action_input.action
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], device=device, dtype=actions.dtype)
        t_bcast = t[:, None, None]

        noisy_trajectory = (1 - t_bcast) * noise + t_bcast * actions
        velocity = actions - noise

        t_discretized = (t * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((state_features, action_features), dim=1)
        action_mask = action_input.action_mask

        # Step 3: unconditional term — null advantage token
        vl_null, mask_null = self._apply_advantage_conditioning(
            vl_embs, vl_attn_mask, advantage_label=None
        )
        out_null = self._run_model_forward(sa_embs, vl_null, mask_null, t_discretized, backbone_output)
        pred_null = self.action_decoder(out_null, embodiment_id)[:, -actions.shape[1]:]
        loss_uncond = (
            F.mse_loss(pred_null, velocity, reduction="none") * action_mask
        ).sum() / (action_mask.sum() + 1e-6)

        # Step 4: conditional term — advantage-labelled token with CFG dropout
        vl_cond, mask_cond = self._apply_advantage_conditioning(
            vl_embs, vl_attn_mask, advantage_label=adv_labels
        )
        out_cond = self._run_model_forward(sa_embs, vl_cond, mask_cond, t_discretized, backbone_output)
        pred_cond = self.action_decoder(out_cond, embodiment_id)[:, -actions.shape[1]:]
        loss_cond = (
            F.mse_loss(pred_cond, velocity, reduction="none") * action_mask
        ).sum() / (action_mask.sum() + 1e-6)

        # Step 5: total loss (Eq. 3)
        total_loss = loss_uncond + self.config.recap_alpha * loss_cond

        return {
            "loss": total_loss,
            "action_loss_uncond": loss_uncond.detach(),
            "action_loss_cond": loss_cond.detach(),
            "advantage_pos_frac": pos_frac.detach(),
        }

    @torch.no_grad()
    def get_value(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """Predict scalar value V(o_t, l) from backbone features."""
        assert self.value_head is not None, "use_recap must be True"
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        value = self.value_head.predict_value(vl_embs, action_input.state.float())
        return BatchFeature(data={"value_pred": value})

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        When use_recap=True, dispatches to value-head training (phase 1) or
        advantage-conditioned policy training (phase 2) based on self._phase.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            dict containing loss and other outputs
        """
        if getattr(self.config, "use_recap", False):
            if self._phase == "value_head":
                return self.forward_value_head(backbone_output, action_input)
            else:
                return self.forward_action_head_recap(backbone_output, action_input)

        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

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
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # RECAP: pre-build advantage-conditioned and null encoder contexts when needed
        use_recap = getattr(self.config, "use_recap", False)
        w = getattr(self.config, "cfg_guidance_weight", 1.0) if use_recap else 1.0
        recap_dual_pass = use_recap and (w != 1.0)
        recap_cond_embs = recap_cond_mask = recap_null_embs = recap_null_mask = None

        if use_recap and self.advantage_embedding is not None:
            pos_labels = torch.full(
                (batch_size,), AdvantageEmbedding.POS_IDX, dtype=torch.long, device=device
            )
            recap_cond_embs, recap_cond_mask = self._apply_advantage_conditioning(
                vl_embeds, backbone_output.backbone_attention_mask, advantage_label=pos_labels
            )
            if recap_dual_pass:
                recap_null_embs, recap_null_mask = self._apply_advantage_conditioning(
                    vl_embeds, backbone_output.backbone_attention_mask, advantage_label=None
                )

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if use_recap and recap_cond_embs is not None:
                # RECAP: use advantage-conditioned encoder context
                pred = self.action_decoder(
                    self._run_model_forward(
                        sa_embs, recap_cond_embs, recap_cond_mask, timesteps_tensor, backbone_output
                    ),
                    embodiment_id,
                )
                if recap_dual_pass:
                    pred_null = self.action_decoder(
                        self._run_model_forward(
                            sa_embs,
                            recap_null_embs,
                            recap_null_mask,
                            timesteps_tensor,
                            backbone_output,
                        ),
                        embodiment_id,
                    )
                    pred = pred_null + w * (pred - pred_null)
            elif self.config.use_alternate_vl_dit:
                pred = self.action_decoder(
                    self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        image_mask=backbone_output.image_mask,
                        backbone_attention_mask=backbone_output.backbone_attention_mask,
                    ),
                    embodiment_id,
                )
            else:
                pred = self.action_decoder(
                    self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                    ),
                    embodiment_id,
                )

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
