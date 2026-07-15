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

# Finetune config used for single node post-training.
from dataclasses import dataclass


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    # dataset_path: str
    dataset_path: list[str]
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive). See EmbodimentTag for known tags."""

    dataset_mix_ratio: list[float] | None = None
    """Optional relative sampling weight per entry in dataset_path (normalized across the
    list). Used for co-training mixtures, e.g. human + robot data: with two datasets,
    `--dataset-mix-ratio 0.3 0.7` samples 30% human / 70% robot regardless of dataset
    sizes. If None, all datasets are weighted equally."""

    dataset_loss_weight: list[float] | None = None
    """Optional per-sample loss multiplier per entry in dataset_path (MotionTrans-style
    human/robot alpha re-weighting, applied in the action loss as a weighted mean). With
    equal mix ratios, `--dataset-loss-weight 0.3 0.7` (human first) reproduces
    MotionTrans alpha=0.7: total gradient contribution human:robot = 0.3:0.7. If None,
    all samples are weighted equally."""

    alpha: float | None = None
    """MotionTrans-style human/robot co-training alpha. When set, dataset_mix_ratio and
    dataset_loss_weight are computed automatically to reproduce MotionTrans exactly:
    sampling proportional to each dataset's total_frames (uniform over the concatenated
    frames) and per-sample loss weights alpha/n_robot_frames (robot data) vs
    (1-alpha)/n_human_frames (human data), so the total gradient contribution is
    robot:human = alpha:(1-alpha) regardless of dataset sizes. Human datasets are
    detected via the observation.is_human column (fallback: "human" in info.json
    robot_type). Mutually exclusive with dataset_mix_ratio / dataset_loss_weight."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    state_dropout_prob: float = 0.2
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Object-centric keypoint auxiliary head ---
    enable_keypoint_head: bool = False
    """If True, add the object-centric keypoint auxiliary head: predicts 16-step
    future object keypoint POSITIONS (Huber regression by fixed point index — see
    keypoint_head_mode). "default"/"tokens" are a pure readout — the action path
    and inference are unchanged. Requires datasets with a "keypoint" section in
    meta/modality.json (datasets without one contribute has_keypoint=0 samples)."""

    keypoint_horizon: int = 16
    """Number of future steps the keypoint head predicts. Must not exceed the
    action horizon. Fixes keypoint_position_decoder's shape, so it must match
    both the dataset's actual keypoint window and whatever checkpoint you
    resume/start from — a mismatch with the dataset fails at data-processing time
    (reshape error); a mismatch with the checkpoint fails at load time
    (size-mismatch error)."""

    max_keypoint_objects: int = 2
    """Number of tracked object slots the keypoint head predicts per step. Same
    matching requirement as keypoint_horizon (dataset's actual object-slot count
    and the checkpoint being resumed/started from)."""

    keypoints_per_object: int = 8
    """Number of tracked points per object the keypoint head predicts. Same
    matching requirement as keypoint_horizon (dataset's actual per-object point
    count in meta/modality.json, and the checkpoint being resumed/started from)."""

    keypoint_loss_weight: float = 1.0
    """Weight of the keypoint position loss (Huber) in the total loss."""

    static_keypoint_weight: float = 0.0
    """Relative loss weight for keypoints of objects whose valid mask is 0
    (padding slot — no real object at the init frame). Default 0 uses the valid
    mask as a hard loss mask: only valid slots are supervised."""

    keypoint_head_mode: str = "default"
    """How the keypoint head attaches to the action head. Every mode predicts
    POSITION only (point k always means the same physical point for the whole
    episode in the current data pipeline — test_keypoint_tracking_simple.py — so
    there is no time-varying "active" signal left to classify). One of:

    "default": decode keypoint position from the same action-token hidden states
    that decode the action itself — a pure readout, action_pred is bit-identical
    whether or not the head runs.

    "tokens": append keypoint_horizon dedicated learned query tokens (DETR-style) to
    the DiT sequence and decode position from those instead, giving the keypoint
    task its own capacity through the transformer. A one-directional
    self-attention mask keeps this a pure readout too: keypoint query tokens may
    attend to the state/action tokens (so keypoint prediction stays conditioned on the
    specific action being generated), but state/action tokens are masked from ever
    attending back to the keypoint queries — action_pred is unaffected by their
    presence, same guarantee as "default". Adds keypoint_query_embedding parameters,
    so it must match between saving and loading a checkpoint.

    "share_dim": fold future keypoint POSITIONS themselves as extra channels of the
    same per-step action vector, jointly noised/denoised by flow matching — plain
    masked MSE, well posed here because point identity is fixed so there's no
    arbitrary-index ambiguity for the loss to be invariant to. NOT a pure readout
    like "default"/"tokens": action_encoder mixes every input channel into one
    embedding via a dense matrix, so the (noised) position values genuinely
    influence the shared representation that also produces the real action
    prediction during training, same as this codebase's existing effort/
    torque-aware channels. What's preserved is the guarantee that actually
    matters: no real keypoint data is ever fed as input at train or inference
    time (the Euler rollout's position channels always start from pure noise,
    same as the real action channels), and action_pred never leaks the extra
    channels. No separate position decoder in this mode. Widens action_encoder's
    input dim and action_decoder's output dim by
    max_keypoint_objects*keypoints_per_object*2, so — like "tokens" — it must
    match between saving and loading a checkpoint (start_from_checkpoint loading
    splices the checkpoint's narrower action head into the widened tensors'
    leading slice automatically).

    "cvae": everything above ("default"/"tokens"-style Huber regression) plus a
    small CVAE to handle multimodal futures (which object moves, in what
    direction) that plain regression would blur into an average. An encoder sees
    the true future keypoints + a condition token (see keypoint_cvae_condition)
    during training and produces a style latent z_style, added to the
    "tokens"-mode dedicated query tokens (never concatenated into the shared
    action/state tokens, so action_pred stays a pure readout exactly like
    "tokens"/"default"); at inference z_style defaults to zeros. Trained with
    reconstruction (Huber, same as every other mode) + KL to N(0,I)
    (keypoint_kl_weight)."""

    keypoint_style_dim: int = 16
    """CVAE style latent dimensionality (keypoint_head_mode="cvae" only). Kept
    small so it can only capture a coarse "which future" choice, not enough to
    losslessly reconstruct the whole trajectory (which would make the decoder
    ignore real conditioning and rely on z_style alone — fine at train time since
    z_style is label-derived, but breaks at inference where z_style=0)."""

    keypoint_kl_weight: float = 0.01
    """Weight of KL(q(z_style|future,condition) || N(0,I)) in the total loss
    (keypoint_head_mode="cvae" only). Keeps the posterior close enough to the
    prior that z_style=0 at inference is a reasonable stand-in for "no
    information about which future"."""

    keypoint_cvae_condition: str = "vlm"
    """What conditions the CVAE recognition encoder alongside the true future
    keypoints (keypoint_head_mode="cvae" only). "vlm" (default) pools the
    backbone's vision+language tokens — richer than "state" (robot proprioception
    only), so z_style only has to capture residual ambiguity the backbone can't
    already resolve. One of {"vlm", "state"}."""

    keypoint_cvae_encoder_layers: int = 2
    """Self-attention layers in the CVAE recognition encoder (keypoint_head_mode
    ="cvae" only)."""

    keypoint_cvae_encoder_heads: int = 4
    """Attention heads in the CVAE recognition encoder (keypoint_head_mode="cvae"
    only)."""

    keypoint_n_key: int | None = None
    """Number of the max_keypoint_objects*keypoints_per_object flat points
    supervised (and shown to the CVAE encoder) per training step, resampled every
    step (keypoint_head_mode="cvae" only). None (default) = use all of them. The
    position decoder always predicts the full set regardless — this only controls
    how many predictions get gradient on a given step."""

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d7"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d7`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    shard_load_workers: int = 1
    video_decode_workers: int = 1
    num_ffmpeg_threads: int = 0
    overlap_episode_io: bool = False

    save_only_model: bool = False
    """If True, save only model weights (skip optimizer/scheduler/RNG states). Cannot resume training from these checkpoints."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    The processor (tokenizer/config) is still loaded from base_model_path.
    Useful for CI/testing to skip the slow checkpoint shard loading."""

    # --- Validation ---
    validation_path: list[str] | None = None
    """Optional path(s) to validation dataset(s). When set, eval loss is computed every eval_steps."""

    eval_set_split_ratio: float | None = None
    """Alternative to validation_path: automatically hold out this fraction of each
    training dataset's episodes as a validation split (e.g. 0.05 = 5%), rather than
    requiring separate validation dataset(s). Mutually exclusive with validation_path
    (validation_path takes precedence if both are set). Held-out episodes are excluded
    from training."""

    eval_steps: int = 2000
    """Number of training steps between validation loss evaluations (only used when
    validation_path or eval_set_split_ratio is set)."""

    # --- Object-centric keypoint debug/eval visualization ---
    keypoint_viz_max_images: int = 50
    """Max number of GT-vs-predicted keypoint overlay image pairs to log to W&B per
    evaluation run (only meaningful when enable_keypoint_head=True and eval is
    configured via validation_path or eval_set_split_ratio)."""

    keypoint_video_episodes: int = 1
    """Number of held-out episodes to roll the model forward over and log as a
    GT-vs-predicted keypoint overlay video per evaluation run, PER dataset_path
    (each dataset in the mix that carries a "keypoint" modality gets its own
    keypoint_video_episodes videos, drawn from that dataset's validation_path /
    eval_set_split_ratio split — not a total pooled across every dataset). 0
    disables episode-video logging (only meaningful when enable_keypoint_head=True)."""

    keypoint_video_max_frames: int = 100
    """Cap on frames rendered per episode video (uniformly strided across the
    episode if longer) — bounds the extra per-eval rollout cost."""

    keypoint_video_batch_size: int = 8
    """Batch size used when rolling the model forward over an episode's frames for
    keypoint_video_episodes."""

    keypoint_video_fps: int = 10
    """Playback fps for the logged keypoint overlay video."""
