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

    tune_top_llm_layers: int = 0
    """Number of top (last) LLM backbone layers to unfreeze even when tune_llm
    is False — a partial-unfreeze alternative to tuning the whole LLM. Useful
    when enable_motion_head=True: the motion-query tokens are appended to and
    processed entirely by the VLM backbone's own (select_layer-truncated)
    transformer layers, so if those layers are fully frozen (tune_llm=False,
    the default), the only trainable capacity for learning to predict future
    motion is the small MotionHead MLP + the query embedding itself — often not
    enough, showing up as motion_loss decreasing then plateauing early. Set
    this to unfreeze the last N layers without paying for tuning the whole
    backbone."""

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

    # --- VLM-backbone motion-keypoint head + human/robot OT alignment ---
    enable_motion_head: bool = False
    """If True, add `num_motion_tokens` learnable query tokens that pass through
    the VLM backbone's own transformer layers (alongside the image/language
    tokens — not the DiT/action head), trained to predict `motion_horizon - 1`
    future 2D object-keypoint positions from their post-backbone hidden states
    (Huber regression, anchored on each point's t=0 position — see
    num_motion_tokens). Applies to human AND robot samples alike. A pure
    readout with respect to the action path: the DiT/action head never sees
    these tokens, so action_pred is unaffected whether or not this head runs.
    Requires datasets with a "keypoint" section in meta/modality.json (datasets
    without one contribute has_keypoint=0 samples). See also enable_ot_align."""

    motion_horizon: int = 16
    """Number of future steps the motion head predicts. Must not exceed the
    action horizon. Fixes motion_position_decoder's shape, so it must match
    both the dataset's actual keypoint window and whatever checkpoint you
    resume/start from."""

    max_motion_objects: int = 2
    """Number of tracked object slots in the dataset's raw keypoint layout
    (meta/modality.json), before num_motion_tokens selects a subset. Must match
    the dataset and the checkpoint being resumed/started from."""

    motion_points_per_object: int = 8
    """Number of tracked points per object in the dataset's raw keypoint layout.
    Same matching requirement as max_motion_objects."""

    num_motion_tokens: int | None = 8
    """Number of learnable motion-query tokens (= number of tracked points the
    head predicts). The processor selects this many points per training sample
    from the valid subset of the max_motion_objects*motion_points_per_object
    flat set — the top-k MOST-MOVING points of the window (motion = total
    per-step displacement over the horizon); moving points carry the
    object-interaction signal, static ones are trivially predictable. None =
    use the full flat set. Changes the motion-query-token parameter shape, so
    it must match between saving and loading a checkpoint."""

    motion_loss_weight: float = 1.0
    """Weight of the motion-keypoint position loss (Huber) in the total loss."""

    motion_static_weight: float = 0.0
    """Relative loss weight for keypoints of objects whose valid mask is 0
    (padding slot — no real object at the init frame). Default 0 uses the valid
    mask as a hard loss mask: only valid slots are supervised."""

    motion_relative: bool = False
    """Train the anchored position decoder on RELATIVE displacements from each
    point's t=0 anchor instead of absolute positions — zero-centered, mostly
    small (a static point's target is exactly 0), an easier regression
    geometry. The anchor is added back for eval/viz. No parameter-shape change,
    but a checkpoint trained one way predicts garbage interpreted the other
    way — keep it consistent across save/load."""

    motion_pool: str = "concat"
    """How the num_motion_tokens post-backbone hidden states are pooled into
    one per-sample feature vector — the MATCHING signal for the OT alignment
    loss (enable_ot_align), not the thing actually pulled together (see
    enable_ot_align). "concat" (default): flatten all num_motion_tokens slots
    into one vector — lossless, since this is always a small, fixed-size set
    with a stable per-slot identity (unlike backbone_features' variable-length
    sequence, there's no real pooling need here). "mean": average across
    tokens (loses cross-slot information), kept for backward compatibility /
    ablation."""

    # --- Optimal Transport alignment between human and robot samples ---
    enable_ot_align: bool = False
    """If True, compute a Sinkhorn (entropic optimal transport) alignment loss
    between human and robot samples in each batch, ramped in linearly over
    ot_warmup_steps. The transport plan is computed from pooled
    motion-token features (motion_pooled_features — an embodiment-invariant
    "what motion is this" signal, trained only by motion_loss), but the loss
    actually pulls pooled BACKBONE features (backbone_pooled_features — the
    vlln/vl_self_attention-refined representation that really feeds the
    DiT/action head) together according to that plan: motion decides which
    robot/human samples correspond, but the representation that matters for
    transfer is the one the action head actually sees. The plan is detached
    before computing the alignment loss (see
    gr00t/model/modules/optimal_transport.py's stopgrad_plan, default True) so
    this doesn't leak an alignment-driven gradient back into the motion-token
    encoder — it stays trained purely by motion_loss. A separate switch from
    enable_motion_head so the motion-keypoint auxiliary loss can be trained
    alone (no alignment pressure) as an ablation; forced off when
    enable_motion_head=False (no pooled features to align). Distinguishing
    human/robot uses the existing per-sample is_human signal (see
    LeRobotEpisodeLoader.detect_is_human) — never routed into the model itself,
    computed purely in the Trainer. Eval-time diagnostics (see
    Gr00tTrainer._log_domain_alignment_viz) log a t-SNE scatter + KNN
    domain-composition + variance on backbone_pooled_features (the actual
    alignment target) plus a motion-vs-backbone neighbor-structure agreement
    check — the design assumes correspondence found in motion-space transfers
    sensibly to backbone-space, which isn't automatic and should be verified,
    not assumed."""

    ot_align_weight: float = 1.0
    """Target weight (lambda) of the OT alignment loss once ot_warmup_steps has
    elapsed."""

    ot_warmup_steps: int = 1000
    """Number of training steps over which the OT alignment loss weight ramps
    linearly from 0 to ot_align_weight — gives the motion-token embeddings time
    to become meaningful (via motion_loss_weight / action loss) before alignment
    pressure starts pulling human and robot representations together."""

    ot_sinkhorn_eps: float = 0.1
    """Entropic regularization strength for the Sinkhorn solver. Larger values
    give smoother (more uniform) transport plans and more stable gradients;
    smaller values approximate exact OT more closely but can be numerically
    harder to converge."""

    ot_sinkhorn_iters: int = 50
    """Number of Sinkhorn normalization iterations."""

    backbone_pool_heads: int = 8
    """Number of attention heads for BackboneAttentionPool, which pools
    backbone_features into backbone_pooled_features (the OT alignment
    TARGET) via a single learnable query cross-attending over the token
    sequence, L2-normalized — replaces naive mean-pool, which let this
    pooled vector collapse toward a near-constant value during OT training
    (see enable_ot_align, gr00t_n1d7.py's BackboneAttentionPool). Must
    divide backbone_embedding_dim."""

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

    # --- Motion-keypoint debug/eval visualization ---
    motion_viz_max_images: int = 50
    """Max number of GT-vs-predicted keypoint overlay image pairs to log to W&B per
    evaluation run (only meaningful when enable_motion_head=True and eval is
    configured via validation_path or eval_set_split_ratio)."""

    motion_video_episodes: int = 1
    """Number of held-out episodes to roll the model forward over and log as a
    GT-vs-predicted keypoint overlay video per evaluation run, PER dataset_path
    (each dataset in the mix that carries a "keypoint" modality gets its own
    motion_video_episodes videos, drawn from that dataset's validation_path /
    eval_set_split_ratio split — not a total pooled across every dataset). 0
    disables episode-video logging (only meaningful when enable_motion_head=True)."""

    motion_video_max_frames: int = 100
    """Cap on frames rendered per episode video (uniformly strided across the
    episode if longer) — bounds the extra per-eval rollout cost."""

    motion_video_batch_size: int = 8
    """Batch size used when rolling the model forward over an episode's frames for
    motion_video_episodes."""

    motion_video_fps: int = 10
    """Playback fps for the logged keypoint overlay video."""

    domain_viz_max_points: int | None = None
    """Optional cap on (pooled motion feature, pooled backbone feature,
    is_human) triples used for the human-vs-robot t-SNE scatter plot and KNN
    domain-composition diagnostic on backbone_pooled_features (see
    Gr00tTrainer._log_domain_alignment_viz). None (the default) uses every
    validation sample; set a positive value to reservoir-sample a large
    validation set. Active whenever enable_motion_head=True, REGARDLESS of
    enable_ot_align — with alignment off this shows the natural (unaligned)
    baseline distribution of the two domains, useful for comparing against
    once ot_align is turned on."""
