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

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d7Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d7 model with backbone and action head.

    Gr00tN1d7 uses the Cosmos-Reason2-2B (Qwen3-VL architecture) VLM backbone,
    replacing the Eagle backbone used in Gr00tN1d6.
    """

    # Model identification
    model_type: str = "Gr00tN1d7"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_model_type: str = "qwen"
    model_revision: str | None = None
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim; must match Cosmos-Reason2-2B hidden size
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 12
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False  # Enable BF16 loading
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = (230, 230)
    image_target_size: tuple[int, int] | None = (256, 256)

    shortest_image_edge: int | None = None
    crop_fraction: float | None = None

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_percentiles: bool = True
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 132  # Default from state_shape
    max_action_dim: int = 132  # Default from action_shape
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # State history: number of consecutive state timesteps fed to the state encoder
    state_history_length: int = 1

    # Global parameters
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # State augmentation parameters
    state_dropout_prob: float = 0.8  # State dropout probability
    exclude_state: bool = False  # Zero out all state inputs (ablation)
    use_mean_std: bool = False  # Use mean/std normalization instead of min/max

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    # Object-centric keypoint auxiliary head (human/robot co-training). Predicts
    # future object keypoint POSITIONS. The data pipeline
    # (test_keypoint_tracking_simple.py / keypoint_tracking_simple_pipeline.md)
    # tracks every one of the N*K = max_keypoint_objects * keypoints_per_object
    # points from a single init frame with a FIXED identity for the whole episode
    # (no per-step active/role reassignment like the retired object-slot + FSM
    # pipeline had) — so there is no time-varying "active" signal left to classify
    # in ANY mode, and keypoint_active_target is reinterpreted as a static
    # per-object valid mask (same value at every horizon step: 1 if a real object
    # occupied that slot at the init frame, 0 if it's padding). Position uses
    # direct masked regression (point k always means the same physical point),
    # not Chamfer set-matching. "default"/"tokens" are a pure readout: action_pred
    # never depends on whether the head runs. "share_dim" is NOT — see its note
    # below.
    enable_keypoint_head: bool = False
    keypoint_horizon: int = 16
    max_keypoint_objects: int = 2
    keypoints_per_object: int = 8
    keypoint_loss_weight: float = 1.0
    # Loss weight for keypoints of objects whose valid mask is 0 (padding slot —
    # no real object at the init frame). Default 0 = hard mask: only valid slots
    # are supervised.
    static_keypoint_weight: float = 0.0
    # How the keypoint head attaches to the action head. One of:
    #   "default": decode keypoint position from the same action-token hidden
    #     states that decode the action itself (cheap, shared representation).
    #   "tokens": append keypoint_horizon dedicated learned query tokens to the DiT
    #     sequence (DETR-style) and decode position from those instead, giving the
    #     keypoint task its own capacity through the transformer's self-/cross-
    #     attention rather than squeezing it into the action tokens' hidden state.
    #     A one-directional self-attention mask (see
    #     Gr00tN1d7ActionHead._keypoint_self_attention_mask) keeps this a pure
    #     readout too: keypoint query tokens may attend to the state/action tokens
    #     (so keypoint prediction stays conditioned on the specific action being
    #     generated), but state/action tokens are masked from ever attending back
    #     to the keypoint queries, at every layer — so action_pred is unaffected by
    #     their presence, exactly like "default". Adds keypoint_query_embedding
    #     parameters, so it must match between saving and loading a checkpoint.
    #   "share_dim": fold future keypoint POSITIONS themselves as extra channels of
    #     the same per-step action vector, jointly noised/denoised by flow
    #     matching — the same mechanism this codebase already uses to extend
    #     action_encoder/action_decoder for other per-embodiment action-space
    #     widening (see expand_action_dimension in embodiment_conditioned_mlp.py).
    #     Well posed as plain per-channel MSE because point identity is fixed (see
    #     above) — no permutation ambiguity for the loss to be invariant to; this
    #     is what makes folding POSITION (not just a per-object flag) viable here,
    #     unlike the retired object-slot pipeline, where only the *active* flags
    #     (max_keypoint_objects dims) were cheap enough to fold and position had
    #     to stay on a separate Chamfer-matched decoder. NOT a pure readout, unlike
    #     "default"/"tokens": action_encoder's W1 is a dense matrix mixing every
    #     input channel into one embedding, so the (noised) position values
    #     genuinely influence the shared representation that also produces the
    #     real action prediction during training — same property this codebase's
    #     existing effort/torque-aware channels already have. What's preserved is
    #     the guarantee that actually matters: no real keypoint data is ever fed as
    #     input at train or inference time (the Euler rollout's position channels
    #     always start from pure noise, exactly like the real action channels),
    #     and the returned action_pred never leaks the extra channels (sliced to
    #     real_action_dim). No separate keypoint_position_decoder in this mode —
    #     position is read directly off action_decoder's own output. Widens
    #     action_encoder's input dim and action_decoder's output dim by
    #     max_keypoint_objects * keypoints_per_object * 2, so — like "tokens" — it
    #     must match between saving and loading a checkpoint.
    #   "cvae": everything above ("default"/"tokens" style Huber regression) plus a
    #     small CVAE to handle multimodal futures (which object moves, in what
    #     direction) that plain regression would blur into an average. A
    #     recognition encoder (Gr00tN1d7ActionHead._encode_keypoint_style) sees the
    #     TRUE future keypoints + a condition token (see keypoint_cvae_condition)
    #     during training and outputs (mu, logvar) for a style latent z_style;
    #     z_style is added to the "tokens"-mode dedicated keypoint query tokens
    #     (NOT concatenated into sa_embs — see below) before the DiT, and the
    #     position decoder (reading those tokens' hidden states, exactly like
    #     "tokens" mode) becomes the CVAE decoder p(keypoints | z_style, context).
    #     Trained with reconstruction (Huber, same as every other mode) + KL
    #     (q(z_style|...) || N(0,I)) weighted by keypoint_kl_weight. At inference
    #     (get_action_with_features) there is no true future to encode, so z_style
    #     defaults to zeros — the KL term's job is exactly to make that a
    #     reasonable stand-in (the prior's mean). Deliberately uses the "tokens"
    #     query-token/self-attention-mask mechanism rather than concatenating
    #     z_style into sa_embs directly: sa_embs also feeds action_decoder, and
    #     z_style is derived from the ground-truth future keypoints during
    #     training — concatenating it into sa_embs would leak that label into
    #     action_pred at train time while inference (z_style=0) sees none of it, a
    #     train/inference mismatch. Routing through the query tokens' one-
    #     directional mask keeps action_pred provably unaffected, same guarantee as
    #     "tokens"/"default".
    #
    # None of the four modes classify anything anymore (see above) —
    # get_action_with_features reports a constant all-ones keypoint_active_pred
    # regardless of mode, purely so existing keypoint_viz overlay code keeps
    # working unchanged.
    keypoint_head_mode: str = "default"  # one of {"default", "tokens", "share_dim", "cvae"}
    # CVAE style latent dimensionality (keypoint_head_mode="cvae" only). Kept small
    # (default 16) so it can only capture a coarse "which future" choice, not enough
    # to losslessly reconstruct the whole trajectory itself — that would let the
    # decoder ignore real conditioning and rely on z_style alone, which is fine at
    # train time (z_style is label-derived) but breaks at inference (z_style=0).
    keypoint_style_dim: int = 16
    # Weight of KL(q(z_style|future,condition) || N(0,I)) in the total loss. Keeps
    # the posterior close enough to the prior that z_style=0 at inference (see
    # keypoint_head_mode="cvae" above) is a reasonable stand-in for "no information
    # about which future". Too large collapses z_style to carry no information at
    # all (posterior == prior always); too small lets the decoder over-rely on
    # z_style and ignore real conditioning. Standard beta-VAE-style tuning knob.
    keypoint_kl_weight: float = 0.01
    # What conditions the CVAE recognition encoder alongside the true future
    # keypoints: "vlm" (default, recommended) pools the backbone's vision+language
    # tokens (masked mean over backbone_attention_mask) — richer than "state"
    # (robot proprioception only, no scene/language), so z_style only has to
    # capture the residual ambiguity the backbone genuinely can't resolve, rather
    # than being forced to also re-derive scene information the decoder already
    # has via its own cross-attention to the same backbone features. "state" is
    # provided for comparison/ablation.
    keypoint_cvae_condition: str = "vlm"  # one of {"vlm", "state"}
    keypoint_cvae_encoder_layers: int = 2
    keypoint_cvae_encoder_heads: int = 4
    # Number of the max_keypoint_objects * keypoints_per_object flat points
    # supervised (and shown to the CVAE encoder as the "future" label) per training
    # step, resampled every step. None (default) = use all of them — the first
    # experiment per keypoint_tracking_simple_pipeline.md. The position decoder
    # always predicts the full set regardless of this value; it only controls how
    # many predictions get gradient on a given step, so shrinking it later doesn't
    # require touching decoder shapes / any saved checkpoint.
    keypoint_n_key: int | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

        if self.keypoint_head_mode not in ("default", "tokens", "share_dim", "cvae"):
            raise ValueError(
                f"keypoint_head_mode must be one of 'default', 'tokens', 'share_dim', 'cvae', "
                f"got {self.keypoint_head_mode!r}"
            )
        if self.keypoint_cvae_condition not in ("vlm", "state"):
            raise ValueError(
                f"keypoint_cvae_condition must be one of 'vlm', 'state', "
                f"got {self.keypoint_cvae_condition!r}"
            )

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("Gr00tN1d7", Gr00tN1d7Config)
