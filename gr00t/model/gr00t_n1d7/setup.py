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

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.model.registry import register_model


# Object-centric keypoint aux head submodule name fragments, across every
# keypoint_head_mode (see Gr00tN1d7Config.keypoint_head_mode): safe to be
# missing from an older checkpoint (fresh init) when resuming/starting from one
# that predates a given mode's extra modules. Single source of truth for
# _create_model's missing-key handling below, so adding a new keypoint
# submodule only requires updating this one list.
KEYPOINT_AUX_PARAM_NAMES = (
    "keypoint_position_decoder",
    "keypoint_query_base",
    "keypoint_query_coord_encoder",
    "keypoint_cls_token",
    "keypoint_condition_proj",
    "keypoint_label_step_embed",
    "keypoint_style_encoder",
    "keypoint_style_head",
    "keypoint_style_to_query",
)


def _reinit_fresh_linear_like(module: torch.nn.Module) -> None:
    """Re-run real parameter init for every Linear/Embedding inside `module`.

    from_pretrained's fast-init path monkey-patches torch.nn.init.{kaiming_uniform_,
    uniform_,normal_} to no-ops for the duration of model construction (skipping the
    RNG cost for weights about to be overwritten by the checkpoint anyway). A
    submodule with no checkpoint counterpart never gets overwritten, so it comes out
    of from_pretrained holding raw torch.empty(...) memory (occasionally NaN/Inf),
    not its class's normal default init. Only safe to call after from_pretrained has
    returned, once that patch context has exited.
    """
    for m in module.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
            m.reset_parameters()


def _reinit_missing_keypoint_params(action_head: torch.nn.Module, missing_keys: list) -> list:
    """Explicitly re-initialize keypoint aux-head submodules from_pretrained reported
    as missing (see _reinit_fresh_linear_like for why __init__'s own init calls can't
    be trusted to have taken effect). Mirrors the mask_token_missing handling above:
    an explicit post-load re-init rather than assuming __init__ already did it.

    keypoint_style_head is special-cased back to all-zeros (not reset_parameters'
    random init) since it must start at mu=0, logvar=0 - the CVAE prior itself, see
    Gr00tN1d7ActionHead._init_keypoint_cvae_modules.
    """
    missing_names = [
        name for name in KEYPOINT_AUX_PARAM_NAMES if any(name in k for k in missing_keys)
    ]
    for name in missing_names:
        submodule = getattr(action_head, name, None)
        if submodule is None:
            continue
        with torch.no_grad():
            if name in ("keypoint_cls_token", "keypoint_query_base"):
                submodule.data.normal_(mean=0.0, std=0.02)
            elif name == "keypoint_style_head":
                submodule.weight.zero_()
                submodule.bias.zero_()
            else:
                _reinit_fresh_linear_like(submodule)
    return missing_names


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d7Pipeline(ModelPipeline):
    model_class = Gr00tN1d7
    processor_class = Gr00tN1d7Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        if self.config.training.start_from_checkpoint is not None and not skip_weight_loading:
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                load_bf16=self.config.model.load_bf16,
                enable_keypoint_head=self.config.model.enable_keypoint_head,
                keypoint_horizon=self.config.model.keypoint_horizon,
                max_keypoint_objects=self.config.model.max_keypoint_objects,
                keypoints_per_object=self.config.model.keypoints_per_object,
                keypoint_loss_weight=self.config.model.keypoint_loss_weight,
                static_keypoint_weight=self.config.model.static_keypoint_weight,
                keypoint_head_mode=self.config.model.keypoint_head_mode,
                keypoint_style_dim=self.config.model.keypoint_style_dim,
                keypoint_kl_weight=self.config.model.keypoint_kl_weight,
                keypoint_cvae_condition=self.config.model.keypoint_cvae_condition,
                keypoint_cvae_encoder_layers=self.config.model.keypoint_cvae_encoder_layers,
                keypoint_cvae_encoder_heads=self.config.model.keypoint_cvae_encoder_heads,
                keypoint_n_key=self.config.model.keypoint_n_key,
                keypoint_relative=self.config.model.keypoint_relative,
                keypoint_match=self.config.model.keypoint_match,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                # "share_dim" mode widens action_encoder/action_decoder (see
                # keypoint_head_mode docstring); without this flag that shape
                # mismatch would hard-crash the load instead of surfacing in
                # loading_info.mismatched_keys, where it's handled explicitly below.
                ignore_mismatched_sizes=True,
                **self.transformers_loading_kwargs,
            )

            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)
            if mask_token_missing and model.action_head.mask_token is not None:
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            unexpected_keys = loading_info.get("unexpected_keys", [])
            mismatched_keys = loading_info.get("mismatched_keys", [])

            # mismatched_keys entries vary by transformers version: either plain key
            # name strings, or (key, checkpoint_shape, model_shape) tuples. Handle
            # both rather than assuming one, since indexing a plain string with [0]
            # silently returns its first *character* instead of raising.
            def _mismatched_key_name(entry):
                return entry[0] if isinstance(entry, (tuple, list)) else entry

            # Newly-added keypoint aux head params: fine to be missing (fresh init)
            # the first time enable_keypoint_head / keypoint_head_mode is turned on
            # for a checkpoint that predates them (or predates a specific mode's
            # extra modules, e.g. "cvae"'s recognition encoder). If the checkpoint
            # HAS them but the current config doesn't use them, that's caught below
            # as unexpected_keys instead (architecture mismatch, not a safe
            # default). Single source of truth (also used by other_missing below)
            # so a newly added keypoint submodule can't silently hard-crash loading
            # here while being fine everywhere else — exactly what happened when
            # "cvae"'s style-encoder params were added but not listed here.
            # Keypoint names in mismatched_keys are treated the same as missing:
            # ignore_mismatched_sizes drops the checkpoint tensor, leaving the
            # module's construction-time memory — which the fast-init path may
            # never have actually initialized (see _reinit_fresh_linear_like) —
            # e.g. keypoint_position_decoder's output dim changed when
            # "tokens"/"cvae" moved to query-conditioned point tokens.
            keypoint_missing = [
                k for k in missing_keys if any(name in k for name in KEYPOINT_AUX_PARAM_NAMES)
            ] + [
                k
                for k in (_mismatched_key_name(m) for m in mismatched_keys)
                if any(name in k for name in KEYPOINT_AUX_PARAM_NAMES)
            ]
            if keypoint_missing:
                # Not "keeping fresh initialization" — explicitly re-running it.
                # from_pretrained's fast-init path means these submodules' own
                # __init__-time init calls may never have taken effect (see
                # _reinit_fresh_linear_like); without this they can come out of
                # from_pretrained holding raw uninitialized memory (occasionally
                # NaN/Inf), which is exactly what caused keypoint_kl_loss to be inf
                # from the very first training step under keypoint_head_mode="cvae".
                reinitialized = _reinit_missing_keypoint_params(model.action_head, keypoint_missing)
                logging.info(
                    "keypoint aux head params not in checkpoint (or shape-changed) - "
                    f"explicitly re-initialized ({len(keypoint_missing)} tensors, "
                    f"modules: {reinitialized})"
                )

            # keypoint_decoder (single shared trunk for position + active) was
            # replaced by two independent heads, keypoint_position_decoder and
            # keypoint_active_decoder (shared trunk caused gradient interference:
            # the active-flag BCE degraded over training under the dominant,
            # active-only-masked position Chamfer loss); keypoint_active_decoder
            # was later removed entirely (no mode classifies "active" anymore — the
            # current data pipeline's keypoint_valid mask is static, not a
            # per-step prediction target; see Gr00tN1d7Config.keypoint_head_mode).
            # Resuming from a checkpoint saved with any of these retired modules
            # is expected to drop their weights, not an architecture-mismatch error.
            # keypoint_query_embedding (per-horizon-step learned query tokens) and
            # keypoint_rank_embedding (per-slot rank-identity tokens, briefly used
            # while no keypoint input was fed) were both retired when
            # "tokens"/"cvae" moved to t0-anchored point tokens
            # (keypoint_query_base + keypoint_query_coord_encoder).
            retired_keypoint_unexpected = [
                k
                for k in unexpected_keys
                if "keypoint_decoder." in k
                or "keypoint_active_decoder." in k
                or "keypoint_query_embedding." in k
                or "keypoint_rank_embedding." in k
            ]
            if retired_keypoint_unexpected:
                logging.info(
                    "Retired keypoint module params found in checkpoint - discarding "
                    f"({len(retired_keypoint_unexpected)} tensors); current keypoint "
                    "modules are fresh-initialized instead."
                )

            # keypoint_head_mode="share_dim" widens action_encoder.W1.W (input) and
            # action_decoder.layer2.{W,b} (output) by
            # max_keypoint_objects*keypoints_per_object*2 channels to carry the
            # future-keypoint-position flow-matching targets (see
            # Gr00tN1d7Config.keypoint_head_mode). ignore_mismatched_sizes=True
            # above means these show up as mismatched_keys with the *checkpoint's*
            # pretrained weights simply discarded (whole tensor fresh-initialized) -
            # instead of accepting that loss of pretrained action capacity, splice
            # the checkpoint's own (narrower) action head into the new tensors'
            # leading slice, so only the new trailing keypoint-position channels
            # are actually fresh-initialized.
            action_dim_param_names = (
                "action_encoder.W1.W",
                "action_decoder.layer2.W",
                "action_decoder.layer2.b",
            )
            action_dim_mismatched = [
                m
                for m in mismatched_keys
                if any(name in _mismatched_key_name(m) for name in action_dim_param_names)
            ]
            share_dim_mode = (
                self.config.model.enable_keypoint_head
                and self.config.model.keypoint_head_mode == "share_dim"
            )
            if action_dim_mismatched and share_dim_mode:
                logging.info(
                    "share_dim mode: action_encoder/action_decoder widened for the "
                    f"keypoint position channels ({len(action_dim_mismatched)} tensors "
                    "affected). Reloading the checkpoint's own (narrower) action head so "
                    "pretrained action weights are spliced into the new tensors' leading "
                    "slice, rather than lost to random re-init."
                )
                old_model, _ = AutoModel.from_pretrained(
                    self.config.training.start_from_checkpoint,
                    output_loading_info=True,
                    **self.transformers_loading_kwargs,
                )
                old_action_dim = old_model.action_head.action_dim
                with torch.no_grad():
                    model.action_head.action_encoder.W1.W.data[:, :old_action_dim, :] = (
                        old_model.action_head.action_encoder.W1.W.data
                    )
                    model.action_head.action_decoder.layer2.W.data[:, :, :old_action_dim] = (
                        old_model.action_head.action_decoder.layer2.W.data
                    )
                    model.action_head.action_decoder.layer2.b.data[:, :old_action_dim] = (
                        old_model.action_head.action_decoder.layer2.b.data
                    )
                del old_model

            other_missing = [
                k
                for k in missing_keys
                if "mask_token" not in k and not any(name in k for name in KEYPOINT_AUX_PARAM_NAMES)
            ]
            other_unexpected = [k for k in unexpected_keys if k not in retired_keypoint_unexpected]
            handled_mismatched = (
                action_dim_mismatched if (action_dim_mismatched and share_dim_mode) else []
            )
            # Keypoint-named mismatched tensors were explicitly re-initialized
            # above (shape changes across keypoint_head_mode revisions) — handled,
            # not an error.
            other_mismatched = [
                m
                for m in mismatched_keys
                if m not in handled_mismatched
                and not any(name in _mismatched_key_name(m) for name in KEYPOINT_AUX_PARAM_NAMES)
            ]
            errors = []
            if other_missing:
                errors.append(f"Missing keys ({len(other_missing)}): {other_missing}")
            if other_unexpected:
                errors.append(f"Unexpected keys ({len(other_unexpected)}): {other_unexpected}")
            if other_mismatched:
                errors.append(f"Mismatched keys ({len(other_mismatched)}): {other_mismatched}")
            if errors:
                raise RuntimeError(
                    "Checkpoint weight mismatch for "
                    f"{self.config.training.start_from_checkpoint}:\n" + "\n".join(errors)
                )

        else:
            model = self.model_class(
                self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        logging.debug(f"Model Config: {model.config}")
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        logging.debug(f"Model architecture: {model}")

        return model

    def _get_statistics(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                keypoint_horizon=self.model_config.keypoint_horizon,
                max_keypoint_objects=self.model_config.max_keypoint_objects,
                keypoints_per_object=self.model_config.keypoints_per_object,
                keypoint_n_key=self.model_config.keypoint_n_key,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation overrides
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                keypoint_horizon=self.model_config.keypoint_horizon,
                max_keypoint_objects=self.model_config.max_keypoint_objects,
                keypoints_per_object=self.model_config.keypoints_per_object,
                keypoint_n_key=self.model_config.keypoint_n_key,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        logging.debug(
            f"Processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}"
        )
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d7Config, Gr00tN1d7Pipeline)
