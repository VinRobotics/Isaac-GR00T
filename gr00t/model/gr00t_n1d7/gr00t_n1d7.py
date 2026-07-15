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
from typing import Any, Optional, Tuple

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
        # real_action_dim is the actual robot/human action width; action_dim is the
        # width of the flow-matching vector actually run through the encoder/DiT/
        # decoder, which "share_dim" mode widens with extra channels (see below).
        self.real_action_dim = config.max_action_dim
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        keypoint_mode = config.keypoint_head_mode if config.enable_keypoint_head else None
        if keypoint_mode == "share_dim":
            # Fold the active-flag channels into the action vector itself (see
            # Gr00tN1d7Config.keypoint_head_mode docstring) — widens both ends of
            # the flow-matching action space, the same expand_action_dimension
            # mechanism this codebase already has for other per-embodiment
            # action-space widening (embodiment_conditioned_mlp.py).
            self.action_dim = config.max_action_dim + config.max_keypoint_objects

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

        # Object-centric keypoint auxiliary head. "default"/"tokens" are a pure
        # readout — action_pred is architecturally provably unaffected by whether
        # the head runs ("tokens" mode via a one-directional self-attention mask,
        # see _keypoint_self_attention_mask). "share_dim" is NOT a pure readout:
        # active-flag channels are extra input columns to action_encoder, whose
        # W1 is a dense matrix mixing every input channel into one embedding, so
        # active-flag values (even noised) genuinely influence the shared
        # representation that also produces the real action prediction during
        # training — exactly like this codebase's existing effort/torque-aware
        # channels, which have the same property. The guarantee "share_dim" DOES
        # keep is the one that actually matters operationally: no real keypoint
        # DATA is ever fed as input, at train or inference time (the Euler
        # rollout's active channels start from pure noise exactly like the real
        # action channels — see get_action_with_features), and action_pred's
        # returned tensor never leaks the extra channels (sliced to
        # real_action_dim). Position decoding (Chamfer) is shared across
        # embodiments (no CategorySpecificMLP) in every mode: object motion is
        # embodiment-invariant, which is what lets this head pull human and robot
        # representations into a common space.
        if config.enable_keypoint_head:
            if keypoint_mode in ("default", "share_dim"):
                assert config.keypoint_horizon <= config.action_horizon, (
                    "keypoint_horizon must not exceed action_horizon when keypoint "
                    "positions are decoded from action-token hidden states "
                    "(keypoint_head_mode='default' or 'share_dim')"
                )
            # Fully independent trunks (no shared parameters) for position
            # regression vs active classification. A single shared trunk was tried
            # first: since static_keypoint_weight=0 hard-masks the Chamfer position
            # loss to active steps only (~30% of cells) while the active BCE trains
            # on every step, and keypoint_loss_weight (1.0) dominates
            # keypoint_active_loss_weight (0.1), the shared hidden layer's gradient
            # was increasingly dominated by the frequent, high-weight position
            # signal — active-flag accuracy degraded over training even as
            # position accuracy kept improving. Separate trunks remove that
            # interference regardless of loss-weight tuning. ("share_dim" mode
            # sidesteps this differently: active isn't decoded by a trunk at all,
            # it's a flow-matching regression target folded into action_decoder.)
            position_output_dim = config.max_keypoint_objects * config.keypoints_per_object * 2
            self.keypoint_position_decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, position_output_dim),
            )
            if keypoint_mode != "share_dim":
                self.keypoint_active_decoder = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size//8, config.max_keypoint_objects),
                )
            if keypoint_mode == "tokens":
                # DETR-style learned query tokens: one per future keypoint step,
                # appended to the DiT sequence so the keypoint task gets its own
                # self-/cross-attention capacity instead of reusing the action
                # tokens' hidden states.
                self.keypoint_query_embedding = nn.Embedding(
                    config.keypoint_horizon, self.input_embedding_dim
                )
                nn.init.normal_(self.keypoint_query_embedding.weight, mean=0.0, std=0.02)

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
            if self.config.enable_keypoint_head:
                self.keypoint_position_decoder.requires_grad_(False)
                if self.config.keypoint_head_mode != "share_dim":
                    self.keypoint_active_decoder.requires_grad_(False)
                if self.config.keypoint_head_mode == "tokens":
                    self.keypoint_query_embedding.requires_grad_(False)
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
                if self.config.enable_keypoint_head:
                    self.keypoint_position_decoder.eval()
                    if self.config.keypoint_head_mode != "share_dim":
                        self.keypoint_active_decoder.eval()
                    if self.config.keypoint_head_mode == "tokens":
                        self.keypoint_query_embedding.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def _keypoint_hidden_states(self, model_output: torch.Tensor) -> torch.Tensor:
        """Slice the hidden states keypoint position/active decoding reads from.

        "default"/"share_dim": the DiT sequence is [state(1), action(action_horizon)];
        keypoints for step t are decoded from the same hidden state that decodes
        action t. "tokens": the sequence additionally has keypoint_horizon dedicated
        query tokens appended at the end
        ([state(1), action(action_horizon), keypoint_queries(keypoint_horizon)]),
        decoded from those instead.
        """
        if self.config.keypoint_head_mode == "tokens":
            return model_output[:, -self.config.keypoint_horizon :]
        return model_output[:, 1 : 1 + self.config.keypoint_horizon]

    def _decode_keypoint_positions(self, model_output: torch.Tensor) -> torch.Tensor:
        """Decode future object keypoint positions. Used by all three modes —
        even "share_dim" (which folds active flags into the action vector) keeps
        position on this same Chamfer-matched decoder, since point index has no
        fixed convention across episodes (unlike object-slot index).

        Returns pred_kp: [B, keypoint_horizon, num_objects, kp_per_object, 2] in
        [-1, 1] image-normalized coordinates.
        """
        num_objects = self.config.max_keypoint_objects
        kp_per_object = self.config.keypoints_per_object
        h = self._keypoint_hidden_states(model_output)
        batch_size, horizon = h.shape[:2]
        return self.keypoint_position_decoder(h).view(
            batch_size, horizon, num_objects, kp_per_object, 2
        )

    def _decode_keypoint_active_logits(self, model_output: torch.Tensor) -> torch.Tensor:
        """Decode future active-flag logits via the dedicated BCE trunk. Only used
        in "default"/"tokens" mode — "share_dim" mode predicts active flags as
        flow-matching regression targets instead (see forward())."""
        h = self._keypoint_hidden_states(model_output)
        return self.keypoint_active_decoder(h).view(
            h.shape[0], h.shape[1], self.config.max_keypoint_objects
        )

    def _append_keypoint_queries(self, sa_embs: torch.Tensor) -> torch.Tensor:
        """Append dedicated keypoint query tokens to the DiT sequence ("tokens"
        mode only; no-op otherwise or when the head is disabled)."""
        if not (self.config.enable_keypoint_head and self.config.keypoint_head_mode == "tokens"):
            return sa_embs
        batch_size = sa_embs.shape[0]
        queries = self.keypoint_query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat((sa_embs, queries), dim=1)

    def _keypoint_self_attention_mask(
        self, batch_size: int, protected_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """One-directional self-attention mask for "tokens" mode.

        State+action tokens (indices [0, protected_len)) may only attend to each
        other, never to the keypoint query tokens appended after them — this keeps
        action_pred a pure readout, unaffected by the keypoint head, exactly like
        "default" mode. Keypoint query tokens may attend to everything (state,
        action, and each other), so keypoint prediction stays conditioned on the
        specific action being generated — the coupling the aux loss is meant to
        exploit.

        Returns None outside "tokens" mode / when the head is disabled (no masking
        needed).
        """
        if not (self.config.enable_keypoint_head and self.config.keypoint_head_mode == "tokens"):
            return None
        total_len = protected_len + self.config.keypoint_horizon
        # True = allowed to attend (same boolean convention as image_mask /
        # backbone_attention_mask elsewhere in this codebase).
        mask = torch.ones(total_len, total_len, dtype=torch.bool, device=device)
        mask[:protected_len, protected_len:] = False
        return mask.unsqueeze(0).expand(batch_size, -1, -1)

    def _keypoint_sample_weight(
        self, batch_size: int, action_input: BatchFeature, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """has_keypoint (per-sample keypoint availability) x loss_weight (e.g.
        MotionTrans alpha co-training re-weighting) — shared by every keypoint
        loss term regardless of mode."""
        has_keypoint = action_input.get("has_keypoint", None)
        if has_keypoint is None:
            sample_weight = torch.ones(batch_size, device=device, dtype=dtype)
        else:
            sample_weight = has_keypoint.to(dtype=dtype, device=device).view(batch_size)
        loss_weight = action_input.get("loss_weight", None)
        if loss_weight is not None:
            sample_weight = sample_weight * loss_weight.to(dtype=dtype, device=device).view(
                batch_size
            )
        return sample_weight

    def _compute_keypoint_position_loss(
        self, pred_kp: torch.Tensor, action_input: BatchFeature
    ) -> torch.Tensor:
        """Chamfer position loss on future object keypoints, used by all three modes.

        Object slot identity is fixed, not permutation-matched: the data pipeline
        assigns slot 0/1 by (first-appearance frame, then left-to-right centroid x)
        and holds it fixed for the whole episode (see assign_slots in
        test_keypoint_tracking.py / convert_keypoint_tracking.py) — a stable,
        image-visible convention the model can learn directly via cross-attention to
        the current frame, rather than a truly arbitrary label the loss must be
        invariant to. Matching per-sample (as an earlier version did) would let the
        model dodge learning that convention, and the hard argmin choice flip-flops
        near-symmetric cases, giving each output slot an inconsistent training
        target across steps.

        Point identity WITHIN a slot has no such convention (farthest-point sampling
        assigns no fixed meaning to point index k), so that stays permutation-
        invariant via symmetric Chamfer distance (Huber-based).
        """
        target_kp = action_input.get("keypoint_target", None)
        target_active = action_input.get("keypoint_active_target", None)
        if target_kp is None or target_active is None:
            # Keep the decoder in the graph (DDP runs with find_unused_parameters=False)
            # while contributing nothing to the loss.
            return pred_kp.sum() * 0.0

        num_objects = self.config.max_keypoint_objects
        kp_per_object = self.config.keypoints_per_object
        batch_size, horizon = pred_kp.shape[:2]
        target_kp = target_kp.view(batch_size, horizon, num_objects, kp_per_object, 2).to(
            pred_kp.dtype
        )
        target_active = target_active.view(batch_size, horizon, num_objects).to(pred_kp.dtype)
        sample_weight = self._keypoint_sample_weight(
            batch_size, action_input, pred_kp.device, pred_kp.dtype
        )

        # Pairwise Huber cost between predicted and target point sets, per (fixed) slot.
        diff = pred_kp.unsqueeze(-2) - target_kp.unsqueeze(-3)  # [B,H,O,K,K,2]
        cost = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=0.1, reduction="none").sum(-1)
        chamfer = 0.5 * (
            cost.min(dim=-1).values.mean(dim=-1) + cost.min(dim=-2).values.mean(dim=-1)
        )  # [B,H,O]
        # Supervise keypoints only where the tracker is confident. active == 1
        # requires real motion within reach of a SAM3 mask sighting; inactive slots
        # may instead hold zeros (slot never appeared / failed episode), frozen
        # positions (before the first mask frame) or untrusted extrapolations, which
        # are indistinguishable from a genuinely static object in the data.
        # static_keypoint_weight > 0 re-enables down-weighted supervision on them.
        step_weight = target_active + (1.0 - target_active) * self.config.static_keypoint_weight
        kp_num = (chamfer * step_weight).sum(dim=(1, 2))  # [B]
        kp_den = step_weight.sum(dim=(1, 2))  # [B]

        # Weighted mean over all supervised (step, object) cells across the batch, so
        # samples whose window has no active object dilute nothing.
        return (kp_num * sample_weight).sum() / ((kp_den * sample_weight).sum() + 1e-6)

    def _compute_keypoint_active_bce_loss(
        self, pred_active_logits: torch.Tensor, action_input: BatchFeature
    ) -> torch.Tensor:
        """BCE active-flag loss via the dedicated trunk. "default"/"tokens" mode
        only — see _compute_share_dim_active_loss for "share_dim" mode.

        pos_weight (num_negative / num_positive, the standard BCEWithLogits
        class-balancing ratio) is estimated fresh from THIS batch's own active
        cell counts rather than a fixed constant: the true active/inactive split
        depends on the dataset mix (human vs robot, which objects/episodes are in
        play) and can drift as --dataset-mix-ratio or co-training alpha change, so
        a hardcoded number would silently go stale. Counted over has_keypoint
        samples only — the zero-filled target on has_keypoint=0 samples isn't a
        real "inactive" label and would bias the ratio toward negative.
        """
        target_active = action_input.get("keypoint_active_target", None)
        if target_active is None:
            return pred_active_logits.sum() * 0.0

        batch_size, horizon = pred_active_logits.shape[:2]
        target_active = target_active.view(
            batch_size, horizon, self.config.max_keypoint_objects
        ).to(pred_active_logits.dtype)
        sample_weight = self._keypoint_sample_weight(
            batch_size, action_input, pred_active_logits.device, pred_active_logits.dtype
        )

        has_keypoint = action_input.get("has_keypoint", None)
        if has_keypoint is not None:
            valid = (has_keypoint.to(device=pred_active_logits.device) >= 0.5).view(
                batch_size, 1, 1
            ).expand_as(target_active)
        else:
            valid = torch.ones_like(target_active, dtype=torch.bool)

        with torch.no_grad():
            num_pos = target_active[valid].sum()
            num_neg = valid.sum() - num_pos
            # No positive cell this batch: nothing to estimate a ratio from, so
            # leave BCE unweighted rather than dividing by zero / near-zero.
            # Capped so one unlucky low-positive batch can't spike the gradient.
            pos_weight = torch.where(
                num_pos > 0, num_neg / num_pos.clamp(min=1), torch.ones_like(num_pos)
            ).clamp(max=50.0)

        bce = F.binary_cross_entropy_with_logits(
            pred_active_logits, target_active, pos_weight=pos_weight, reduction="none"
        )
        active_cost = bce.mean(dim=(1, 2))
        return (active_cost * sample_weight).sum() / (sample_weight.sum() + 1e-6)

    def _share_dim_active_targets(
        self,
        batch_size: int,
        action_horizon: int,
        action_input: BatchFeature,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-padded active-flag target + mask, appended as extra
        action_encoder/action_decoder channels in "share_dim" mode (see
        Gr00tN1d7Config.keypoint_head_mode). Both are
        [B, action_horizon, max_keypoint_objects]: target holds
        keypoint_active_target for t < keypoint_horizon and zeros after (there's
        no active label beyond the keypoint window); mask is 1 wherever a real,
        has_keypoint sample supplies a target and 0 elsewhere, so padding steps
        and keypoint-missing samples don't enter the flow-matching loss.
        """
        num_objects = self.config.max_keypoint_objects
        kp_horizon = self.config.keypoint_horizon
        target = torch.zeros(batch_size, action_horizon, num_objects, device=device, dtype=dtype)
        mask = torch.zeros(batch_size, action_horizon, num_objects, device=device, dtype=dtype)

        active_target = action_input.get("keypoint_active_target", None)
        if active_target is not None:
            active_target = active_target.to(device=device, dtype=dtype).view(
                batch_size, kp_horizon, num_objects
            )
            target[:, :kp_horizon] = active_target
            step_mask = torch.ones(batch_size, kp_horizon, num_objects, device=device, dtype=dtype)
            has_keypoint = action_input.get("has_keypoint", None)
            if has_keypoint is not None:
                step_mask = step_mask * has_keypoint.to(device=device, dtype=dtype).view(
                    batch_size, 1, 1
                )
            mask[:, :kp_horizon] = step_mask
        return target, mask

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
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

        # Embed noised action trajectory. In "share_dim" mode, active-flag
        # channels are appended as extra columns of this same per-step vector and
        # jointly noised/denoised — see Gr00tN1d7Config.keypoint_head_mode.
        actions = action_input.action
        share_dim_active = (
            self.config.enable_keypoint_head and self.config.keypoint_head_mode == "share_dim"
        )
        share_dim_mask = None
        if share_dim_active:
            share_dim_target, share_dim_mask = self._share_dim_active_targets(
                actions.shape[0], actions.shape[1], action_input, actions.device, actions.dtype
            )
            actions = torch.cat((actions, share_dim_target), dim=-1)

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
        # In keypoint dedicated-tokens mode, dedicated query tokens are appended
        # after the action tokens: [state(1), action(action_horizon), kp_queries].
        sa_embs = torch.cat((state_features, action_features), dim=1)
        protected_len = sa_embs.shape[1]
        sa_embs = self._append_keypoint_queries(sa_embs)
        self_attention_mask = self._keypoint_self_attention_mask(
            sa_embs.shape[0], protected_len, device
        )
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
                self_attention_mask=self_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                self_attention_mask=self_attention_mask,
            )

        # Slice off any appended keypoint query tokens before decoding actions: in
        # "tokens" mode the action tokens are no longer the tail of the sequence,
        # and running action_decoder on the keypoint tokens would be wasted
        # (category-specific) compute anyway.
        action_end = 1 + actions.shape[1]
        pred = self.action_decoder(model_output[:, :action_end], embodiment_id)
        pred_actions = pred[:, 1:]

        # Slice out only the real action portion of pred/target — in "share_dim"
        # mode pred_actions/velocity carry extra trailing active-flag channels,
        # handled separately below so they never leak into action_loss/action_pred.
        pred_real_actions = pred_actions[..., : self.real_action_dim]
        velocity_real = velocity[..., : self.real_action_dim]
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_real_actions, velocity_real, reduction="none") * action_mask
        # Optional per-sample loss weight (e.g. MotionTrans-style human/robot alpha
        # co-training re-weighting, set per dataset via SingleDatasetConfig.loss_weight).
        # Weighted mean: relative gradient contribution follows the weights while uniform
        # weights reduce exactly to the unweighted loss.
        loss_weight = action_input.get("loss_weight", None)
        if loss_weight is not None:
            loss_weight_bc = loss_weight.to(action_loss.dtype)[:, None, None]
            loss = (action_loss * loss_weight_bc).sum() / (
                (action_mask * loss_weight_bc).sum() + 1e-6
            )
        else:
            loss_weight_bc = None
            loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        outputs = {
            "loss": loss,
            "action_loss": action_loss,
            # Scalar pure flow-matching loss, kept separate from "loss" so it can be
            # logged on its own once the keypoint aux terms are folded in below
            # (outputs["loss"] then becomes action + keypoint combined).
            "action_loss_scalar": loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
            # Exposed so eval-time keypoint viz can call get_action_with_features()
            # directly and reuse this forward pass's backbone output, instead of
            # re-running the full VLM backbone a second time per batch.
            "image_mask": backbone_output.get("image_mask"),
            "backbone_attention_mask": backbone_output.backbone_attention_mask,
        }

        if self.config.enable_keypoint_head:
            pred_kp = self._decode_keypoint_positions(model_output)
            keypoint_loss = self._compute_keypoint_position_loss(pred_kp, action_input)
            outputs["keypoint_loss"] = keypoint_loss
            combined_loss = loss + self.config.keypoint_loss_weight * keypoint_loss

            if share_dim_active:
                # Active flags are a flow-matching regression target folded into
                # the action vector (see _share_dim_active_targets) rather than a
                # BCE decoder output — plain masked MSE on the velocity, exactly
                # like the action loss above.
                pred_active_vel = pred_actions[..., self.real_action_dim :]
                velocity_active = velocity[..., self.real_action_dim :]
                active_diff = (
                    F.mse_loss(pred_active_vel, velocity_active, reduction="none") * share_dim_mask
                )
                if loss_weight_bc is not None:
                    active_loss = (active_diff * loss_weight_bc).sum() / (
                        (share_dim_mask * loss_weight_bc).sum() + 1e-6
                    )
                else:
                    active_loss = active_diff.sum() / (share_dim_mask.sum() + 1e-6)
            else:
                pred_active_logits = self._decode_keypoint_active_logits(model_output)
                active_loss = self._compute_keypoint_active_bce_loss(
                    pred_active_logits, action_input
                )

            outputs["keypoint_active_loss"] = active_loss
            outputs["loss"] = combined_loss + self.config.keypoint_active_loss_weight * active_loss

        return outputs

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

        # Constant across denoising steps: same protected/total sequence layout
        # every iteration, so build the (dedicated-tokens mode only) self-attention
        # mask once rather than per step.
        protected_len = state_features.shape[1] + self.config.action_horizon
        self_attention_mask = self._keypoint_self_attention_mask(batch_size, protected_len, device)

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

            # Use previous action instead of pure noise to do inpainting. Sliced to
            # real_action_dim on both sides: in "share_dim" mode `actions` also
            # carries extra active-flag channels that action_input["action"] (a
            # real action tensor) doesn't have — those channels keep denoising
            # from pure noise as usual, RTC inpainting only applies to real actions.
            actions[:, : options["rtc_overlap_steps"], : self.real_action_dim] = action_input[
                "action"
            ][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                : self.real_action_dim,
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

            # Join vision, language, state and action embedding along sequence
            # dimension. In keypoint dedicated-tokens mode, dedicated query tokens
            # are appended after the action tokens.
            sa_embs = torch.cat((state_features, action_features), dim=1)
            sa_embs = self._append_keypoint_queries(sa_embs)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    self_attention_mask=self_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    self_attention_mask=self_attention_mask,
                )
            # Slice off any appended keypoint query tokens before decoding actions
            # (see forward()'s action_decoder call for why).
            pred = self.action_decoder(
                model_output[:, : 1 + self.action_horizon], embodiment_id
            )
            pred_velocity = pred[:, 1:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        # Sliced to real_action_dim: in "share_dim" mode `actions` also carries the
        # extra active-flag channels denoised alongside the real action (see
        # forward()) — those never belong in action_pred.
        output_data = {
            "action_pred": actions[..., : self.real_action_dim],
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

        # Debug/eval readout: decode predicted future object keypoints from the final
        # denoising step. Needs no extra input and does not affect action_pred.
        if (
            self.config.enable_keypoint_head
            and options is not None
            and options.get("return_keypoints", False)
        ):
            # Coordinates are in [-1, 1] normalized by the original image (W, H).
            output_data["keypoint_pred"] = self._decode_keypoint_positions(model_output)
            if self.config.keypoint_head_mode == "share_dim":
                # The final Euler-integrated value of the extra channels IS the
                # model's direct regression estimate of the raw 0/1 target (the
                # flow-matching target was the label itself, not a logit) — no
                # sigmoid, unlike "default"/"tokens" mode's BCE decoder. Clamp
                # only for display; the raw value is the more honest signal.
                output_data["keypoint_active_pred"] = actions[..., self.real_action_dim :].clamp(
                    0.0, 1.0
                )
            else:
                pred_active_logits = self._decode_keypoint_active_logits(model_output)
                output_data["keypoint_active_pred"] = torch.sigmoid(pred_active_logits)

        return BatchFeature(data=output_data)

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
