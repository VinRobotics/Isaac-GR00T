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

from scipy.optimize import linear_sum_assignment
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
            # Fold future keypoint POSITION channels into the action vector itself
            # (see Gr00tN1d7Config.keypoint_head_mode docstring) — widens both ends
            # of the flow-matching action space, the same expand_action_dimension
            # mechanism this codebase already has for other per-embodiment
            # action-space widening (embodiment_conditioned_mlp.py). Valid only
            # because the current data pipeline (test_keypoint_tracking_simple.py)
            # gives every point a FIXED identity for the whole episode — no
            # permutation ambiguity left for a flow-matching per-channel target to
            # be invariant to (unlike the retired object-slot pipeline, where only
            # the *active* flags — not position — were cheap/well-posed to fold).
            n_total = config.max_keypoint_objects * config.keypoints_per_object
            self.action_dim = config.max_action_dim + n_total * 2

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

        # Object-centric keypoint auxiliary head: predicts future object keypoint
        # positions (no active/role classification in any mode — the current data
        # pipeline, test_keypoint_tracking_simple.py, tracks every point from a
        # single init frame with a FIXED identity and a STATIC per-object valid
        # mask for the whole episode, so there is nothing time-varying left to
        # classify; see Gr00tN1d7Config.keypoint_head_mode). "default"/"tokens" are
        # a pure readout — action_pred is architecturally provably unaffected by
        # whether the head runs ("tokens" mode via a one-directional self-attention
        # mask, see _keypoint_self_attention_mask). "share_dim" is NOT a pure
        # readout: position channels are extra input columns to action_encoder,
        # whose W1 is a dense matrix mixing every input channel into one embedding,
        # so position values (even noised) genuinely influence the shared
        # representation that also produces the real action prediction during
        # training — exactly like this codebase's existing effort/torque-aware
        # channels, which have the same property. The guarantee "share_dim" DOES
        # keep is the one that actually matters operationally: no real keypoint
        # DATA is ever fed as input, at train or inference time (the Euler
        # rollout's position channels start from pure noise exactly like the real
        # action channels — see get_action_with_features), and action_pred's
        # returned tensor never leaks the extra channels (sliced to
        # real_action_dim). Position decoding is shared across embodiments (no
        # CategorySpecificMLP) in every mode: object motion is embodiment-
        # invariant, which is what lets this head pull human and robot
        # representations into a common space.
        if config.enable_keypoint_head:
            n_total_keypoints = config.max_keypoint_objects * config.keypoints_per_object
            # "tokens"/"cvae": number of t0-anchored point tokens appended to
            # the DiT sequence — the keypoint_n_key subsample the processor draws
            # per training sample, or the full flat set when keypoint_n_key is None
            # (see Gr00tN1d7Config.keypoint_n_key / _append_keypoint_queries).
            self.num_keypoint_points = config.keypoint_n_key or n_total_keypoints
            if keypoint_mode in ("default", "share_dim"):
                assert config.keypoint_horizon <= config.action_horizon, (
                    "keypoint_horizon must not exceed action_horizon when keypoint "
                    "positions are decoded from action-token hidden states "
                    "(keypoint_head_mode='default' or 'share_dim')"
                )
            if keypoint_mode == "default":
                # One output head per action-token step: all N*K points at once.
                self.keypoint_position_decoder = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, n_total_keypoints * 2),
                )
            if keypoint_mode in ("tokens", "cvae"):
                # Anchored point-slot tokens (ATM-style): one learned slot
                # embedding per point (distinct per slot — identical tokens
                # would produce identical hidden states, hence identical
                # predictions), and the position decoder additionally receives
                # that point's CURRENT (t=0) position (keypoint_target's first
                # step) concatenated to the slot's hidden state, predicting the
                # remaining keypoint_horizon - 1 FUTURE steps — either absolute
                # positions or displacements from the anchor (see
                # Gr00tN1d7Config.keypoint_relative). The anchor binds each
                # slot's identity through its input, which makes the plain
                # by-index Huber loss well posed again (each slot's target is
                # unique given its anchor — no exchangeable-target collapse, no
                # Hungarian matching needed here; _match_keypoints_hungarian
                # remains for "default" only). The anchor is tracker data the
                # training/eval sample already carries — never an input on the
                # real-robot action path, where these tokens aren't appended at
                # all (see get_action_with_features). "cvae" reuses this same
                # mechanism as the substrate its style latent (z_style) is
                # injected into. keypoint_query_base holds one learned embedding
                # PER SLOT (num_keypoint_points of them): each slot can develop
                # its own attention pattern in the DiT — slot i loosely
                # specializes on the i-th point of the processor's motion-rank
                # order (e.g. "most-moving point" vs "static filler") — so h
                # differs per slot instead of collapsing to one shared context
                # vector. Firm per-point identity still comes from the decoder's
                # anchor conditioning: decoder(concat(h_i, coord_encoder(
                # anchor_i))), slot i paired with anchor i. Ties
                # keypoint_query_base's shape to keypoint_n_key, so that must
                # match between saving and loading a checkpoint.
                self.keypoint_query_base = nn.Parameter(
                    torch.randn(1, self.num_keypoint_points, self.input_embedding_dim) * 0.02
                )
                self.keypoint_query_coord_encoder = nn.Sequential(
                    nn.Linear(2, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                )
                self.keypoint_position_decoder = nn.Sequential(
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, (config.keypoint_horizon - 1) * 2),
                )

            if keypoint_mode == "cvae":
                self._init_keypoint_cvae_modules(config)

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

    def _init_keypoint_cvae_modules(self, config: Gr00tN1d7Config) -> None:
        """Build the CVAE recognition encoder + style-injection modules for
        keypoint_head_mode="cvae" (see Gr00tN1d7Config.keypoint_head_mode)."""
        heads = config.keypoint_cvae_encoder_heads
        assert self.input_embedding_dim % heads == 0, (
            f"keypoint_cvae_encoder_heads ({heads}) must divide input_embedding_dim "
            f"({self.input_embedding_dim})"
        )
        # Embeds one horizon-step's (num_keypoint_points, 2) future keypoints —
        # the same sampled point set the decoder predicts and the loss supervises
        # (the processor draws it once per sample; see
        # Gr00tN1d7Config.keypoint_n_key) — into one token; z_style is meant to be
        # one global "which future" code for the whole window, so sequence length
        # here is keypoint_horizon, not keypoint_horizon * num_keypoint_points.
        self.keypoint_label_step_embed = nn.Linear(
            self.num_keypoint_points * 2, self.input_embedding_dim
        )
        self.keypoint_cls_token = nn.Parameter(torch.randn(1, 1, self.input_embedding_dim) * 0.02)
        if config.keypoint_cvae_condition == "vlm":
            self.keypoint_condition_proj = nn.Linear(
                config.backbone_embedding_dim, self.input_embedding_dim
            )
        self.keypoint_style_encoder = SelfAttentionTransformer(
            num_attention_heads=heads,
            attention_head_dim=self.input_embedding_dim // heads,
            num_layers=config.keypoint_cvae_encoder_layers,
        )
        self.keypoint_style_head = nn.Linear(
            self.input_embedding_dim, 2 * config.keypoint_style_dim
        )
        # Zero-init so the encoder starts at mu=0, logvar=0 (the prior itself) —
        # z_style=0 at inference then matches what training sees early on too,
        # rather than an arbitrary random point the decoder never saw.
        nn.init.zeros_(self.keypoint_style_head.weight)
        nn.init.zeros_(self.keypoint_style_head.bias)
        self.keypoint_style_to_query = nn.Linear(
            config.keypoint_style_dim, self.input_embedding_dim
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
                if self.config.keypoint_head_mode != "share_dim":
                    self.keypoint_position_decoder.requires_grad_(False)
                if self.config.keypoint_head_mode in ("tokens", "cvae"):
                    self.keypoint_query_base.requires_grad_(False)
                    self.keypoint_query_coord_encoder.requires_grad_(False)
                if self.config.keypoint_head_mode == "cvae":
                    self.keypoint_label_step_embed.requires_grad_(False)
                    self.keypoint_cls_token.requires_grad_(False)
                    if self.config.keypoint_cvae_condition == "vlm":
                        self.keypoint_condition_proj.requires_grad_(False)
                    self.keypoint_style_encoder.requires_grad_(False)
                    self.keypoint_style_head.requires_grad_(False)
                    self.keypoint_style_to_query.requires_grad_(False)
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
                    if self.config.keypoint_head_mode != "share_dim":
                        self.keypoint_position_decoder.eval()
                    if self.config.keypoint_head_mode in ("tokens", "cvae"):
                        # keypoint_query_base is a bare Parameter (no .eval());
                        # only the coord-encoder MLP is a Module.
                        self.keypoint_query_coord_encoder.eval()
                    if self.config.keypoint_head_mode == "cvae":
                        self.keypoint_label_step_embed.eval()
                        if self.config.keypoint_cvae_condition == "vlm":
                            self.keypoint_condition_proj.eval()
                        self.keypoint_style_encoder.eval()
                        self.keypoint_style_head.eval()
                        self.keypoint_style_to_query.eval()
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
        """Slice the hidden states keypoint position decoding reads from.

        "default": the DiT sequence is [state(1), action(action_horizon)]; keypoints
        for step t are decoded from the same hidden state that decodes action t.
        "tokens"/"cvae": the sequence additionally has num_keypoint_points
        t0-anchored point tokens appended at the end
        ([state(1), action(action_horizon), point_queries(num_keypoint_points)]),
        decoded from those instead — one token per sampled point, each carrying
        its full future trajectory (see _append_keypoint_queries). Not called at
        all in "share_dim" mode — position there is folded directly into the
        action vector and read back off action_decoder's own output (see
        forward() / _share_dim_position_targets).
        """
        if self.config.keypoint_head_mode in ("tokens", "cvae"):
            return model_output[:, -self.num_keypoint_points :]
        return model_output[:, 1 : 1 + self.config.keypoint_horizon]

    def _decode_keypoint_positions(
        self, model_output: torch.Tensor, anchor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode future object keypoint positions. Used by "default"/"tokens"/
        "cvae" — not "share_dim", which folds position directly into the action
        vector instead (no separate decoder; see forward()).

        "tokens"/"cvae": each slot's hidden state is concatenated with its
        point's ENCODED t=0 anchor (keypoint_query_coord_encoder(anchor);
        `anchor` is [B, num_keypoint_points, 2], REQUIRED for these modes — see
        keypoint_position_decoder in __init__) and decoded into the
        keypoint_horizon - 1 FUTURE steps (t=1..H-1) — absolute positions, or
        displacements from the anchor when config.keypoint_relative. Slot i's
        hidden state (from its own learned slot embedding, see
        _append_keypoint_queries) is paired with point i's anchor — the anchor
        is what binds identity firmly; the per-slot h adds each point's own
        attended context. Returns [B, keypoint_horizon - 1, P, 2].

        "default": one action-token step decodes all n_total points for its
        step, no anchor. Returns [B, keypoint_horizon, n_total, 2].
        """
        h = self._keypoint_hidden_states(model_output)
        batch_size = h.shape[0]
        if self.config.keypoint_head_mode in ("tokens", "cvae"):
            assert anchor is not None, "tokens/cvae keypoint decoding requires the t=0 anchor"
            anchor_emb = self.keypoint_query_coord_encoder(anchor.to(h.dtype))
            decoded = self.keypoint_position_decoder(torch.cat((h, anchor_emb), dim=-1))
            # [B, P, (keypoint_horizon-1)*2] -> [B, keypoint_horizon-1, P, 2]
            decoded = decoded.view(
                batch_size, self.num_keypoint_points, self.config.keypoint_horizon - 1, 2
            )
            return decoded.permute(0, 2, 1, 3)
        # [B, keypoint_horizon, n_total*2] -> [B, keypoint_horizon, n_total, 2]
        decoded = self.keypoint_position_decoder(h)
        return decoded.view(batch_size, self.config.keypoint_horizon, -1, 2)

    def _append_keypoint_queries(
        self, sa_embs: torch.Tensor, z_style: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Append point tokens to the DiT sequence ("tokens"/"cvae" mode only;
        no-op otherwise or when the head is disabled).

        Token i is its own learned slot embedding (keypoint_query_base[:, i] —
        one per point, no coordinate content), so each slot gathers its own
        motion-scene context through attention (slot order follows the
        processor's motion-rank emission order); firm per-point identity enters
        at the position decoder, which conditions slot i's hidden state on
        point i's encoded t=0 anchor (see _decode_keypoint_positions). The
        real-robot action path never appends these tokens (see
        get_action_with_features).

        z_style ([B, keypoint_style_dim], "cvae" mode only): added as a bias to
        every point token — NOT concatenated into sa_embs itself. sa_embs also
        feeds action_decoder, and z_style is derived from the ground-truth future
        keypoints during training (_encode_keypoint_style); injecting it only
        into these point tokens keeps it behind the one-directional
        self-attention mask (_keypoint_self_attention_mask), so action_pred is
        provably unaffected — exactly the same guarantee "tokens" mode already
        has.
        """
        if not (
            self.config.enable_keypoint_head
            and self.config.keypoint_head_mode in ("tokens", "cvae")
        ):
            return sa_embs
        queries = self.keypoint_query_base.expand(sa_embs.shape[0], self.num_keypoint_points, -1)
        if z_style is not None:
            queries = queries + self.keypoint_style_to_query(z_style).unsqueeze(1)
        return torch.cat((sa_embs, queries), dim=1)

    def _keypoint_anchor(
        self, action_input: BatchFeature, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """t=0 positions of the selected points, [B, num_keypoint_points, 2], fed
        to keypoint_position_decoder as the per-slot anchor ("tokens"/"cvae"
        only, else None). Taken from keypoint_target's first horizon step —
        tracker data the training/eval sample already carries, zero-filled by
        the processor for has_keypoint=0 samples (whose loss is masked anyway).
        Zeros when the batch has no keypoint fields at all, purely to keep the
        decoder in the DDP graph. Never needed on the real-robot action path
        (point tokens aren't appended there — see get_action_with_features)."""
        if not (
            self.config.enable_keypoint_head
            and self.config.keypoint_head_mode in ("tokens", "cvae")
        ):
            return None
        target_kp = action_input.get("keypoint_target", None)
        if target_kp is None:
            return torch.zeros(batch_size, self.num_keypoint_points, 2, device=device, dtype=dtype)
        target_kp = target_kp.view(
            batch_size, self.config.keypoint_horizon, self.num_keypoint_points, 2
        )
        return target_kp[:, 0].to(device=device, dtype=dtype)

    def _keypoint_self_attention_mask(
        self, batch_size: int, protected_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """One-directional self-attention mask for "tokens"/"cvae" mode.

        State+action tokens (indices [0, protected_len)) may only attend to each
        other, never to the keypoint point tokens appended after them — this keeps
        action_pred a pure readout, unaffected by the keypoint head, exactly like
        "default" mode. It also means the point tokens can be dropped from the
        sequence entirely at inference with bitwise-identical action_pred (the
        protected rows' attention is computed as if those columns didn't exist) —
        which is exactly what get_action_with_features does unless
        return_keypoints is requested, so the real-robot action path never pays
        for them and never needs current keypoint positions. Point tokens may
        attend to everything (state, action, and each other), so keypoint
        prediction stays conditioned on the specific action being generated — the
        coupling the aux loss is meant to exploit. In "cvae" mode this is also
        what keeps the (label-derived) z_style code from ever leaking into
        action_pred (see _append_keypoint_queries).

        Returns None outside "tokens"/"cvae" mode / when the head is disabled (no
        masking needed).
        """
        if not (
            self.config.enable_keypoint_head
            and self.config.keypoint_head_mode in ("tokens", "cvae")
        ):
            return None
        total_len = protected_len + self.num_keypoint_points
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

    def _keypoint_condition_token(
        self,
        state_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        vl_attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Condition token for the CVAE recognition encoder ("cvae" mode only).

        See Gr00tN1d7Config.keypoint_cvae_condition for why "vlm" (masked mean-pool
        over the backbone's vision+language tokens) is the default over "state"
        (robot proprioception only): it gives the encoder access to at least as
        much context as the decoder already has via its own cross-attention to the
        same backbone features, so z_style only has to capture genuine residual
        ambiguity rather than compensating for a weaker view of the scene.

        Returns [B, 1, input_embedding_dim].
        """
        if self.config.keypoint_cvae_condition == "state":
            return state_features
        if vl_attn_mask is not None:
            mask = vl_attn_mask.to(vl_embeds.dtype).unsqueeze(-1)  # [B, S, 1]
            pooled = (vl_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = vl_embeds.mean(dim=1)
        return self.keypoint_condition_proj(pooled).unsqueeze(1)  # [B, 1, D]

    def _encode_keypoint_style(
        self, target_kp_flat: torch.Tensor, condition_token: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CVAE recognition network q(z_style | future keypoints, condition).

        Train-time only — needs the true future, so this is never called from
        get_action_with_features (inference always uses the zero/prior default
        instead, or an explicit prior sample — see options["keypoint_style_sample"]).

        Sees the same num_keypoint_points sampled set the decoder predicts and the
        reconstruction loss supervises (the processor draws it once per sample —
        see Gr00tN1d7Config.keypoint_n_key): posterior and decoder agree on which
        future they're describing.

        Args:
            target_kp_flat: [B, keypoint_horizon, num_keypoint_points, 2]
            condition_token: [B, 1, input_embedding_dim]
        Returns:
            (mu, logvar), each [B, keypoint_style_dim]
        """
        batch_size, horizon = target_kp_flat.shape[:2]
        label = target_kp_flat.reshape(batch_size, horizon, -1)  # [B, H, n_total*2]
        label_tokens = self.keypoint_label_step_embed(label)  # [B, H, D]

        cls = self.keypoint_cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        seq = torch.cat((cls, condition_token, label_tokens), dim=1)  # [B, 2+H, D]
        encoded = self.keypoint_style_encoder(seq)  # [B, 2+H, D]
        stats = self.keypoint_style_head(encoded[:, 0])  # [B, 2*z_dim]
        mu, logvar = stats.chunk(2, dim=-1)
        # keypoint_kl_weight is small by design (weak regularizer), so nothing else
        # bounds logvar's growth during training. Clamp before every downstream
        # exp(logvar) (here and in _compute_keypoint_kl_loss) so it can't overflow
        # to inf — an unclamped logvar is a standard VAE instability, and one inf
        # here poisons every shared parameter's gradient on the next backward pass.
        logvar = logvar.clamp(min=-10.0, max=10.0)
        return mu, logvar

    def _sample_keypoint_style(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterized sample z_style = mu + eps * std, eps ~ N(0,I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _compute_keypoint_kl_loss(
        self, mu: torch.Tensor, logvar: torch.Tensor, action_input: BatchFeature
    ) -> torch.Tensor:
        """KL(q(z_style|future,condition) || N(0,I)), masked by has_keypoint /
        loss_weight exactly like the other keypoint loss terms — samples with no
        real keypoint data would otherwise pull mu/logvar toward whatever the
        zero-filled label produces."""
        batch_size = mu.shape[0]
        sample_weight = self._keypoint_sample_weight(batch_size, action_input, mu.device, mu.dtype)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)  # [B]
        return (kl * sample_weight).sum() / (sample_weight.sum() + 1e-6)

    def _match_keypoints_hungarian(
        self,
        pred_kp: torch.Tensor,
        target_kp: torch.Tensor,
        valid: torch.Tensor,
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reorder target points (and their valid mask) to the optimal ONE-TO-ONE
        assignment against the predictions, per group, cost aggregated over the
        whole horizon — one fixed bijection per sample/group, never re-matched
        per step (a given predicted channel tracks one physical point across the
        window).

        A strict bijection (scipy linear_sum_assignment) is the load-bearing
        property, not a nicety: every target must be claimed by exactly one
        prediction, so collapsing all predictions onto one point pays full cost
        for every unclaimed target. Both softer alternatives demonstrably
        collapse — one-directional nearest-neighbor (Chamfer) has a degenerate
        near-zero-loss optimum with every prediction sitting on a single target
        point, and plain by-index regression against exchangeable targets (the
        motion-rank order of same-object points is per-window noise, since a
        rigid object's points all move the same amount) is minimized by every
        token predicting the same central point. The "all predictions cluster at
        one point" failure observed in training is exactly those two optima.

        Args:
            pred_kp, target_kp: [B, H, P, 2] flat.
            valid: [B, H, P] per-point weights, reordered with the same
                permutation so weights stay attached to their target points.
            group_size: matching is restricted to within groups of this many
                consecutive points — keypoints_per_object for "default" (never
                match across objects: the object axis carries real signal).
                "tokens"/"cvae" no longer use this at all: their decoder is
                anchored on each point's t=0 position, which binds slot
                identity through the input and makes by-index loss well posed
                (see _compute_keypoint_position_loss).
        Returns:
            (target_kp, valid) reordered along the point axis (same shapes).
        """
        batch_size, horizon, num_points = pred_kp.shape[:3]
        num_groups = num_points // group_size
        with torch.no_grad():
            pred = pred_kp.view(batch_size, horizon, num_groups, group_size, 2)
            target = target_kp.view(batch_size, horizon, num_groups, group_size, 2)
            diff = pred.permute(0, 2, 1, 3, 4).unsqueeze(4) - target.permute(
                0, 2, 1, 3, 4
            ).unsqueeze(3)  # [B, G, H, K(pred), K(target), 2]
            cost = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=0.1, reduction="none")
            cost = cost.sum(-1).sum(2)  # aggregate coords + horizon -> [B, G, Kp, Kt]
            cost_np = cost.float().cpu().numpy()
            perm = torch.empty(batch_size, num_groups, group_size, dtype=torch.long)
            for b in range(batch_size):
                for g in range(num_groups):
                    # row_ind is sorted 0..K-1 for a square cost matrix, so
                    # col_ind[k] is the target assigned to prediction k.
                    _, col_ind = linear_sum_assignment(cost_np[b, g])
                    perm[b, g] = torch.from_numpy(col_ind)
            perm = perm.to(target_kp.device)

        target = target_kp.view(batch_size, horizon, num_groups, group_size, 2)
        gather_idx = perm.view(batch_size, 1, num_groups, group_size, 1).expand(
            batch_size, horizon, num_groups, group_size, 2
        )
        matched_target = torch.gather(target, dim=3, index=gather_idx)
        valid_groups = valid.view(batch_size, horizon, num_groups, group_size)
        matched_valid = torch.gather(
            valid_groups,
            dim=3,
            index=perm.view(batch_size, 1, num_groups, group_size).expand(
                batch_size, horizon, num_groups, group_size
            ),
        )
        return (
            matched_target.reshape(batch_size, horizon, num_points, 2),
            matched_valid.reshape(batch_size, horizon, num_points),
        )

    def _compute_keypoint_position_loss(
        self, pred_kp: torch.Tensor, action_input: BatchFeature
    ) -> torch.Tensor:
        """Direct masked Huber position loss on future object keypoints, used by
        "default"/"tokens"/"cvae" on the decoder's flat output ("share_dim"
        computes the analogous MSE directly on flow-matching velocity in forward()
        instead, since it has no separate decoder).

        "tokens"/"cvae": plain by-index Huber on the keypoint_horizon - 1 FUTURE
        steps — pred_kp is [B, H-1, P, 2] and the full-window target is sliced
        to t=1..H-1 (t=0 is the decoder's input anchor, not a prediction
        target). Well posed without any matching: each slot's target is unique
        given its own anchor (no exchangeable-target collapse — the failure
        by-index had back when slots carried no anchor). When
        config.keypoint_relative, the target is the displacement from the
        anchor instead of the absolute position. "default": full-window
        [B, H, n_total, 2] by-index regression, or per-object Hungarian
        matching when keypoint_match="hungarian" (see
        _match_keypoints_hungarian — its enumeration order is arbitrary per
        episode).

        keypoint_active_target is a static valid mask, either per-object
        ([B, H, max_keypoint_objects], broadcast to every point of the object) or
        already per-point ([B, H, P], emitted when the processor samples
        keypoint_n_key points) — distinguished by its last dim.

        Args:
            pred_kp: [B, T, P, 2] flat predicted positions (T and P as in
                _decode_keypoint_positions).
        """
        target_kp = action_input.get("keypoint_target", None)
        valid = action_input.get("keypoint_active_target", None)
        if target_kp is None or valid is None:
            # Keep the decoder in the graph (DDP runs with find_unused_parameters=False)
            # while contributing nothing to the loss.
            return pred_kp.sum() * 0.0

        batch_size, _, num_points = pred_kp.shape[:3]
        full_horizon = self.config.keypoint_horizon
        target_kp = target_kp.view(batch_size, full_horizon, num_points, 2).to(pred_kp.dtype)
        valid = valid.view(batch_size, full_horizon, -1).to(pred_kp.dtype)
        if valid.shape[-1] != num_points:
            # Per-object mask -> per-point (num_points is a multiple of num_objects).
            valid = valid.repeat_interleave(num_points // valid.shape[-1], dim=-1)
        if self.config.keypoint_head_mode in ("tokens", "cvae"):
            anchor = target_kp[:, :1]  # [B, 1, P, 2] — the decoder's input
            target_kp = target_kp[:, 1:]
            valid = valid[:, 1:]
            if self.config.keypoint_relative:
                target_kp = target_kp - anchor
        elif (
            self.config.keypoint_head_mode == "default"
            and self.config.keypoint_match == "hungarian"
        ):
            target_kp, valid = self._match_keypoints_hungarian(
                pred_kp, target_kp, valid, group_size=self.config.keypoints_per_object
            )
        sample_weight = self._keypoint_sample_weight(
            batch_size, action_input, pred_kp.device, pred_kp.dtype
        )

        diff = pred_kp - target_kp
        cost = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=0.1, reduction="none").sum(-1)
        # Supervise keypoints only where the tracker is confident (valid == 1 —
        # slot had a real object at init frame). static_keypoint_weight > 0
        # re-enables down-weighted supervision on padding slots (zeros).
        step_weight = valid + (1.0 - valid) * self.config.static_keypoint_weight
        kp_num = (cost * step_weight).sum(dim=(1, 2))  # [B]
        kp_den = step_weight.sum(dim=(1, 2))  # [B]

        # Weighted mean over all supervised (step, point) cells across the batch, so
        # samples whose window has no valid object dilute nothing.
        return (kp_num * sample_weight).sum() / ((kp_den * sample_weight).sum() + 1e-6)

    def _share_dim_position_targets(
        self,
        batch_size: int,
        action_horizon: int,
        action_input: BatchFeature,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-padded future-keypoint-position target + weight, appended as extra
        action_encoder/action_decoder channels in "share_dim" mode (see
        Gr00tN1d7Config.keypoint_head_mode). Both are
        [B, action_horizon, n_total*2] (n_total = max_keypoint_objects *
        keypoints_per_object): target holds keypoint_target for t < keypoint_horizon
        and zeros after (no position label beyond the keypoint window); weight is
        0 wherever a padding step / keypoint-missing sample supplies no target, and
        otherwise follows the same static_keypoint_weight down-weighting convention
        as _compute_keypoint_position_loss for invalid (padding) slots.
        """
        n_total = self.config.max_keypoint_objects * self.config.keypoints_per_object
        kp_horizon = self.config.keypoint_horizon
        target = torch.zeros(batch_size, action_horizon, n_total * 2, device=device, dtype=dtype)
        weight = torch.zeros(batch_size, action_horizon, n_total * 2, device=device, dtype=dtype)

        target_kp = action_input.get("keypoint_target", None)
        valid = action_input.get("keypoint_active_target", None)
        if target_kp is not None and valid is not None:
            target_kp = target_kp.to(device=device, dtype=dtype).view(batch_size, kp_horizon, -1)
            target[:, :kp_horizon] = target_kp

            valid = valid.to(device=device, dtype=dtype).view(
                batch_size, kp_horizon, self.config.max_keypoint_objects
            )
            valid = valid.repeat_interleave(self.config.keypoints_per_object, dim=-1)
            step_weight = valid + (1.0 - valid) * self.config.static_keypoint_weight
            step_weight = step_weight.repeat_interleave(2, dim=-1)  # (x,y) pairs -> n_total*2
            has_keypoint = action_input.get("has_keypoint", None)
            if has_keypoint is not None:
                step_weight = step_weight * has_keypoint.to(device=device, dtype=dtype).view(
                    batch_size, 1, 1
                )
            weight[:, :kp_horizon] = step_weight
        return target, weight

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

        # Embed noised action trajectory. In "share_dim" mode, future keypoint
        # position channels are appended as extra columns of this same per-step
        # vector and jointly noised/denoised — see Gr00tN1d7Config.keypoint_head_mode.
        actions = action_input.action
        share_dim_mode = (
            self.config.enable_keypoint_head and self.config.keypoint_head_mode == "share_dim"
        )
        share_dim_weight = None
        if share_dim_mode:
            share_dim_target, share_dim_weight = self._share_dim_position_targets(
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

        # CVAE style code: computed from the TRUE future keypoints (label) before
        # the DiT runs, so it can be injected into the keypoint query tokens
        # below. See _encode_keypoint_style / _append_keypoint_queries for why
        # this never touches sa_embs / action_pred.
        cvae_mode = self.config.enable_keypoint_head and self.config.keypoint_head_mode == "cvae"
        z_style = None
        keypoint_mu = keypoint_logvar = None
        if cvae_mode:
            condition_token = self._keypoint_condition_token(
                state_features, vl_embeds, backbone_output.backbone_attention_mask
            )
            target_kp_flat = action_input.get("keypoint_target", None)
            if target_kp_flat is not None:
                target_kp_flat = target_kp_flat.view(
                    action_features.shape[0],
                    self.config.keypoint_horizon,
                    self.num_keypoint_points,
                    2,
                ).to(condition_token.dtype)
                keypoint_mu, keypoint_logvar = self._encode_keypoint_style(
                    target_kp_flat, condition_token
                )
                if self.training:
                    z_style = self._sample_keypoint_style(keypoint_mu, keypoint_logvar)
                    has_keypoint = action_input.get("has_keypoint", None)
                    if has_keypoint is not None:
                        z_style = z_style * has_keypoint.to(
                            device=z_style.device, dtype=z_style.dtype
                        ).view(-1, 1)
                else:
                    # Eval: z_style = 0, exactly what inference uses
                    # (get_action_with_features has no label to encode). This
                    # makes eval keypoint_loss measure DEPLOYMENT reconstruction
                    # quality rather than posterior-conditioned ELBO — the
                    # train-vs-eval keypoint_loss gap is then a direct readout of
                    # how much the decoder has come to depend on z_style (the
                    # failure mode a runaway keypoint_kl_loss signals: with a
                    # weak keypoint_kl_weight the encoder can keep buying
                    # reconstruction with information in z, and z=0 drifts out
                    # of the distribution the decoder was trained on). KL is
                    # still computed from the posterior (mu/logvar) as usual.
                    z_style = torch.zeros_like(keypoint_mu)

        # Join vision, language, state and action embedding along sequence dimension.
        # In "tokens"/"cvae" mode, shared-base point tokens are appended after
        # the action tokens: [state(1), action(action_horizon), point_queries];
        # per-point identity enters at the position decoder via the t=0 anchor.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        protected_len = sa_embs.shape[1]
        sa_embs = self._append_keypoint_queries(sa_embs, z_style=z_style)
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
        # mode pred_actions/velocity carry extra trailing keypoint-position
        # channels, handled separately below so they never leak into
        # action_loss/action_pred.
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
            if share_dim_mode:
                # Position channels are a flow-matching regression target folded
                # into the action vector (see _share_dim_position_targets) rather
                # than a separate decoder output — plain masked MSE on the
                # velocity, exactly like the action loss above.
                pred_kp_vel = pred_actions[..., self.real_action_dim :]
                velocity_kp = velocity[..., self.real_action_dim :]
                kp_diff = F.mse_loss(pred_kp_vel, velocity_kp, reduction="none") * share_dim_weight
                if loss_weight_bc is not None:
                    keypoint_loss = (kp_diff * loss_weight_bc).sum() / (
                        (share_dim_weight * loss_weight_bc).sum() + 1e-6
                    )
                else:
                    keypoint_loss = kp_diff.sum() / (share_dim_weight.sum() + 1e-6)
                outputs["keypoint_loss"] = keypoint_loss
                outputs["loss"] = loss + self.config.keypoint_loss_weight * keypoint_loss
            else:
                # "default" / "tokens" / "cvae": decoder output is flat
                # [B, T, P, 2]; "tokens"/"cvae" additionally condition each
                # slot's readout on its point's t=0 anchor (see
                # _decode_keypoint_positions).
                keypoint_anchor = self._keypoint_anchor(
                    action_input, model_output.shape[0], device, model_output.dtype
                )
                pred_kp = self._decode_keypoint_positions(model_output, anchor=keypoint_anchor)
                keypoint_loss = self._compute_keypoint_position_loss(pred_kp, action_input)
                outputs["keypoint_loss"] = keypoint_loss
                outputs["loss"] = loss + self.config.keypoint_loss_weight * keypoint_loss

                if cvae_mode:
                    if keypoint_mu is not None:
                        kl_loss = self._compute_keypoint_kl_loss(
                            keypoint_mu, keypoint_logvar, action_input
                        )
                    else:
                        # No keypoint_target in this batch's schema at all (rather
                        # than per-sample has_keypoint=0, which
                        # _compute_keypoint_kl_loss already masks): keep params in
                        # the DDP graph, contribute 0.
                        kl_loss = pred_kp.sum() * 0.0
                    outputs["keypoint_kl_loss"] = kl_loss
                    outputs["loss"] = outputs["loss"] + self.config.keypoint_kl_weight * kl_loss

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

        # Keypoint point tokens at inference: appended ONLY when the caller asks
        # for keypoint predictions (eval/viz, options["return_keypoints"]). On the
        # real-robot action path they're skipped entirely — the one-directional
        # self-attention mask means the protected tokens' attention is computed as
        # if those columns didn't exist, so dropping them yields bitwise-identical
        # action_pred while paying no extra sequence length and needing no current
        # keypoint positions (see _keypoint_self_attention_mask).
        return_keypoints = (
            self.config.enable_keypoint_head
            and options is not None
            and options.get("return_keypoints", False)
        )
        append_keypoint_tokens = return_keypoints and self.config.keypoint_head_mode in (
            "tokens",
            "cvae",
        )

        # Constant across denoising steps: same protected/total sequence layout
        # every iteration, so build the point-token self-attention mask once
        # rather than per step.
        protected_len = state_features.shape[1] + self.config.action_horizon
        self_attention_mask = None
        z_style = None
        if append_keypoint_tokens:
            self_attention_mask = self._keypoint_self_attention_mask(
                batch_size, protected_len, device
            )
            # CVAE style code at inference: no true future to encode (unlike
            # forward() / _encode_keypoint_style), so z_style defaults to zeros —
            # the prior's mean, which the KL term (keypoint_kl_weight) trains the
            # posterior to stay close to. options["keypoint_style_sample"]=True
            # instead draws z_style ~ N(0,I) for eval/debug diversity
            # visualization; either way this only reaches the keypoint point
            # tokens (see _append_keypoint_queries), never action_pred.
            if self.config.keypoint_head_mode == "cvae":
                z_style = torch.zeros(
                    batch_size,
                    self.config.keypoint_style_dim,
                    dtype=vl_embeds.dtype,
                    device=device,
                )
                if options.get("keypoint_style_sample", False):
                    z_style = torch.randn_like(z_style)

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
            # carries extra keypoint-position channels that action_input["action"]
            # (a real action tensor) doesn't have — those channels keep denoising
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
            # dimension. Keypoint point tokens are appended only when keypoint
            # predictions were requested (see append_keypoint_tokens above).
            sa_embs = torch.cat((state_features, action_features), dim=1)
            if append_keypoint_tokens:
                sa_embs = self._append_keypoint_queries(sa_embs, z_style=z_style)

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
        # extra future-keypoint-position channels denoised alongside the real
        # action (see forward()) — those never belong in action_pred.
        output_data = {
            "action_pred": actions[..., : self.real_action_dim],
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

        # Debug/eval readout: decode predicted future object keypoints from the final
        # denoising step. Does not affect action_pred (see the return_keypoints
        # gating above).
        if return_keypoints:
            # Coordinates are in [-1, 1] normalized by the original image (W, H).
            # keypoint_pred keeps the 5-D [B, T, num_groups, points_per_group, 2]
            # viz contract (keypoint_viz.render_keypoint_overlay): "default"/
            # "share_dim" group by object; "tokens"/"cvae" predict a flat sampled
            # point set, reported as num_keypoint_points single-point groups.
            if self.config.keypoint_head_mode == "share_dim":
                # No separate decoder in this mode — the final Euler-integrated
                # value of the extra channels IS the position estimate (the
                # flow-matching target was the raw position itself, not a logit).
                # Sliced to keypoint_horizon: the trailing action_horizon -
                # keypoint_horizon steps carry no supervision (zero weight in
                # _share_dim_position_targets) and would render as junk; slicing
                # also keeps every mode's keypoint_pred on the same horizon as
                # the GT window, so viz can weight both panels with the GT valid
                # mask.
                num_objects = self.config.max_keypoint_objects
                kp_per_object = self.config.keypoints_per_object
                pred_kp = actions[..., self.real_action_dim :].view(
                    actions.shape[0], self.action_horizon, num_objects, kp_per_object, 2
                )
                output_data["keypoint_pred"] = pred_kp[:, : self.config.keypoint_horizon]
            elif self.config.keypoint_head_mode in ("tokens", "cvae"):
                # Decoder reads each point's future off the shared point-token
                # hidden states, conditioned on that point's t=0 anchor (from
                # the eval batch's keypoint_target; zeros if absent, making the
                # readout meaningless but never crashing), and predicts the H-1
                # future steps. Reassemble the full-window absolute trajectory
                # for the viz contract: undo keypoint_relative displacements,
                # then prepend the anchor as step 0 — [B, H, P, 1, 2].
                keypoint_anchor = self._keypoint_anchor(
                    action_input, batch_size, device, vl_embeds.dtype
                )
                pred_kp = self._decode_keypoint_positions(model_output, anchor=keypoint_anchor)
                if self.config.keypoint_relative:
                    pred_kp = pred_kp + keypoint_anchor.unsqueeze(1)
                pred_kp = torch.cat(
                    (keypoint_anchor.unsqueeze(1).to(pred_kp.dtype), pred_kp), dim=1
                )
                output_data["keypoint_pred"] = pred_kp.unsqueeze(3)
            else:  # "default": flat [B, H, N*K, 2] -> [B, H, N, K, 2]
                pred_kp = self._decode_keypoint_positions(model_output)
                output_data["keypoint_pred"] = pred_kp.view(
                    pred_kp.shape[0],
                    pred_kp.shape[1],
                    self.config.max_keypoint_objects,
                    self.config.keypoints_per_object,
                    2,
                )

            # No keypoint_active_pred: no mode predicts a genuine "active" signal
            # anymore (see Gr00tN1d7Config.keypoint_head_mode), and viz weights
            # BOTH panels with the GT valid mask instead (see
            # trainer._collect_keypoint_viz) — which is also what blanks the
            # prediction overlay for samples/windows with no valid tracked points.

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
