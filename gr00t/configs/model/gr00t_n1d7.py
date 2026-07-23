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

    # VLM-backbone motion-keypoint head + human/robot OT alignment. Adds
    # num_motion_tokens learnable query tokens that pass through the VLM
    # backbone's own transformer layers (alongside image/language tokens — not
    # the DiT/action head), predicting motion_horizon - 1 future 2D keypoint
    # positions from their post-backbone hidden states, anchored on each
    # point's t=0 position (same anchoring mechanism the retired keypoint
    # head's "tokens" mode used, transplanted here — see
    # gr00t/model/gr00t_n1d7/motion_head.py). Applies to human AND robot
    # samples alike (dense two-domain supervision is the main anti-collapse
    # force). A pure readout with respect to the action path: the DiT/action
    # head never sees these tokens. enable_ot_align additionally aligns human
    # vs. robot samples via Sinkhorn OT (see
    # gr00t/model/modules/optimal_transport.py): the transport plan is
    # computed from pooled motion-token features (embodiment-invariant,
    # trained only by motion_loss), but the loss pulls pooled BACKBONE
    # features (the representation that actually feeds the DiT) together —
    # motion decides correspondence, backbone is what gets aligned. The plan
    # is detached (stopgrad_plan) so this never leaks a gradient back into the
    # motion-token encoder. Gated by the existing per-sample is_human signal
    # (never routed into the model itself — see
    # LeRobotEpisodeLoader.detect_is_human), computed in the Trainer.
    enable_motion_head: bool = False
    motion_horizon: int = 16
    max_motion_objects: int = 2
    motion_points_per_object: int = 8
    num_motion_tokens: int | None = 8
    motion_loss_weight: float = 1.0
    # Loss weight for keypoints of objects whose valid mask is 0 (padding slot —
    # no real object at the init frame). Default 0 = hard mask: only valid slots
    # are supervised.
    motion_static_weight: float = 0.0
    # Train the anchored position decoder on RELATIVE displacements from each
    # point's t=0 anchor instead of absolute [-1, 1] positions. Zero-centered
    # and mostly small (a static point's target is exactly 0) — an easier
    # regression geometry; the anchor is added back when reassembling absolute
    # trajectories for eval/viz. No shape/parameter change, but a checkpoint
    # trained one way predicts garbage interpreted the other way.
    motion_relative: bool = False
    # How the num_motion_tokens post-backbone hidden states are pooled into one
    # per-sample feature vector — the MATCHING signal for the OT alignment
    # loss, not the thing actually pulled together (see enable_ot_align).
    # "concat" (default): flatten all num_motion_tokens slots into one
    # num_motion_tokens*backbone_embedding_dim vector — lossless, since (unlike
    # backbone_features) this is always a small, fixed-size set with a stable
    # per-slot identity, so there's no variable-length sequence forcing a real
    # pool. "mean": average over tokens (loses cross-slot information), kept
    # for backward compatibility / ablation against the old behavior.
    motion_pool: str = "concat"  # one of {"concat", "mean"}

    enable_ot_align: bool = False
    ot_align_weight: float = 1.0
    ot_warmup_steps: int = 1000
    ot_sinkhorn_eps: float = 0.1
    ot_sinkhorn_iters: int = 50

    # Attention-pooling heads for backbone_pooled_features (the OT alignment
    # TARGET — see BackboneAttentionPool in gr00t_n1d7.py). Replaces naive
    # masked-mean pooling, which empirically let backbone_pooled_features
    # collapse toward a near-constant vector during OT training (measured via
    # Gr00tTrainer._log_domain_alignment_viz's per-domain variance dropping to
    # ~1e-8): averaging over the whole image/language token sequence dilutes
    # sample-specific signal behind hundreds of largely-similar tokens, and
    # the OT loss's unconstrained squared-Euclidean cost then has a trivial
    # "shrink everything to one point" minimum. A single learnable query
    # cross-attending over the sequence, combined with L2-normalizing the
    # pooled output (see BackboneAttentionPool.forward /
    # MotionHead.pool), removes that trivial minimum: normalized vectors
    # can't collapse via uniform shrinkage, only by literally aligning in
    # direction, a qualitatively different (and much harder to hit by
    # accident) degenerate solution. Must divide backbone_embedding_dim.
    backbone_pool_heads: int = 8

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

        if self.motion_pool not in ("concat", "mean"):
            raise ValueError(f"motion_pool must be 'concat' or 'mean', got {self.motion_pool!r}")
        if self.num_motion_tokens is not None:
            n_total = self.max_motion_objects * self.motion_points_per_object
            if not 1 <= self.num_motion_tokens <= n_total:
                raise ValueError(
                    f"num_motion_tokens must be in [1, max_motion_objects * "
                    f"motion_points_per_object = {n_total}], got {self.num_motion_tokens}"
                )
        if self.enable_ot_align and not self.enable_motion_head:
            raise ValueError("enable_ot_align requires enable_motion_head=True")
        if self.enable_motion_head and self.backbone_embedding_dim % self.backbone_pool_heads != 0:
            raise ValueError(
                f"backbone_pool_heads ({self.backbone_pool_heads}) must divide "
                f"backbone_embedding_dim ({self.backbone_embedding_dim})"
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
