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

import contextlib
import logging

import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


def _resolve_inner_model(model: torch.nn.Module) -> torch.nn.Module:
    """Qwen3VLForConditionalGeneration exposes `.language_model`/`.visual`
    either directly, or one level down via `.model` depending on the
    installed transformers version's internal "nested Model" refactor —
    resolve whichever is actually present instead of assuming one, so this
    backbone doesn't silently break across a transformers version bump."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        return inner
    if hasattr(model, "language_model"):
        return model
    raise AttributeError(
        "Could not locate `language_model` on the loaded Qwen3VL model (checked "
        "both `model.language_model` and `model.model.language_model`) - the "
        "installed transformers version's Qwen3-VL internals may have changed; "
        "update Qwen3Backbone._resolve_inner_model."
    )


@contextlib.contextmanager
def _inject_motion_tokens(
    embedding_layer: torch.nn.Module, motion_query_tokens: torch.nn.Parameter
):
    """Temporarily hook the VLM's input-embedding layer so the LAST
    `motion_query_tokens.shape[1]` positions of its output are overwritten
    with our learnable per-slot embedding instead of whatever token id was
    used as a placeholder — the seam that lets `motion_query_tokens` pass
    through the backbone's own transformer layers (self-attending to the
    real image/language tokens) without touching any other internals (embed
    lookup, image-feature scatter, MRoPE position computation all run
    completely unmodified on the caller's actual input_ids/pixel_values).
    Scoped to a single forward call via a context manager so it can never
    leak into an unrelated embedding lookup elsewhere.
    """
    n = motion_query_tokens.shape[1]

    def hook(module, args, output):
        queries = motion_query_tokens.expand(output.shape[0], -1, -1).to(output.dtype)
        return torch.cat((output[:, :-n], queries), dim=1)

    handle = embedding_layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class Qwen3Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        num_motion_tokens: int = 0,
        transformers_loading_kwargs: dict = {},
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
            num_motion_tokens: number of learnable motion-query tokens (see
                gr00t/model/gr00t_n1d7/motion_head.py) appended to the input
                sequence and passed through this backbone's own transformer
                layers. 0 disables the feature entirely (default, fully
                backward compatible — the forward pass is byte-for-byte
                unchanged from before this was added).
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn is not installed. Falling back to sdpa attention. "
                    "Install flash-attn for better performance: pip install flash-attn"
                )
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **extra_kwargs,
            **transformers_loading_kwargs,
        ).eval()
        self._inner = _resolve_inner_model(self.model)

        # needed since we don't use these layers. Also saves compute
        while len(self._inner.language_model.layers) > select_layer:
            self._inner.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.num_motion_tokens = num_motion_tokens
        if num_motion_tokens > 0:
            embed_dim = self.model.get_input_embeddings().weight.shape[-1]
            self.motion_query_tokens = torch.nn.Parameter(
                torch.randn(1, num_motion_tokens, embed_dim) * 0.02
            )
        else:
            self.motion_query_tokens = None
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

    def _placeholder_token_id(self) -> int:
        """Vocab id used to reserve motion-query-token positions in input_ids
        before their embeddings get overwritten by _inject_motion_tokens —
        the actual id barely matters (it's never used as a real embedding),
        just needs to avoid colliding with special vision token ids so it
        doesn't trip the model's own image/video-feature scatter logic."""
        cfg = self.model.config
        image_token_id = getattr(cfg, "image_token_id", None)
        for name in ("pad_token_id", "eos_token_id"):
            val = getattr(cfg, name, None)
            if val is None:
                continue
            val = val if isinstance(val, int) else val[0]
            if val != image_token_id:
                return val
        return 0

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self._inner.language_model.requires_grad_(False)
        if not tune_visual:
            self._inner.visual.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self._inner.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self._inner.language_model and not self.tune_llm:
                self._inner.language_model.eval()
            if self._inner.visual and not self.tune_visual:
                self._inner.visual.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_input[k] for k in keys_to_use}

        if self.motion_query_tokens is not None:
            # Append num_motion_tokens placeholder positions to the REAL
            # input_ids/attention_mask and let the model's own forward run
            # completely unmodified over the extended sequence — its own
            # embedding lookup, image-feature scatter, and MRoPE position
            # computation all handle the extra trailing text-like positions
            # correctly with zero reimplementation. _inject_motion_tokens
            # then overwrites just those positions' embeddings with our
            # learnable parameter, right after the embedding lookup and before
            # any transformer layer runs. Because they're appended at the
            # very END of the sequence and the LM is causal, they attend to
            # every real image/language token that precedes them, while
            # nothing else can ever attend back to them — the same
            # "pure readout" guarantee the old Action-Head keypoint head's
            # one-directional attention mask achieved, here for free from
            # ordinary causal masking + position order.
            input_ids = vl_input["input_ids"]
            attention_mask = vl_input["attention_mask"]
            batch_size = input_ids.shape[0]
            placeholder_ids = input_ids.new_full(
                (batch_size, self.num_motion_tokens), self._placeholder_token_id()
            )
            motion_mask = attention_mask.new_ones((batch_size, self.num_motion_tokens))
            vl_input = {
                **vl_input,
                "input_ids": torch.cat((input_ids, placeholder_ids), dim=1),
                "attention_mask": torch.cat((attention_mask, motion_mask), dim=1),
            }
            with _inject_motion_tokens(self.model.get_input_embeddings(), self.motion_query_tokens):
                outputs = self.model(**vl_input, output_hidden_states=True)
        else:
            outputs = self.model(**vl_input, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]
        real_len = hidden.shape[1] - self.num_motion_tokens
        backbone_features = hidden[:, :real_len]
        image_mask = vl_input["input_ids"][:, :real_len] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"][:, :real_len] == 1

        data = {
            "backbone_features": backbone_features,
            "backbone_attention_mask": attention_mask,
            "image_mask": image_mask,
        }
        if self.motion_query_tokens is not None:
            data["motion_token_features"] = hidden[:, real_len:]
        return BatchFeature(data=data)  # [B, T2, hidden_size]
