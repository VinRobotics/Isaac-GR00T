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
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
        tune_visual_last_n_layers: int = 0,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
            tune_visual_last_n_layers: if > 0 AND tune_visual is True, only the
                last N encoder layers of the SigLIP vision tower (plus the
                post-encoder layernorm and the visual→LLM projector `mlp1`) are
                trainable.  0 = full unfreeze (legacy behaviour).
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_visual_last_n_layers)

    def set_trainable_parameters(
        self,
        tune_llm: bool,
        tune_visual: bool,
        tune_visual_last_n_layers: int = 0,
    ):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        self.tune_visual_last_n_layers = tune_visual_last_n_layers
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        elif tune_visual_last_n_layers > 0:
            # Partial unfreeze: freeze everything in vision_model first, then
            # re-enable just the last N encoder layers + post-encoder layernorm.
            # mlp1 stays trainable (it's the bridge into the LLM).
            self.eagle_model.vision_model.requires_grad_(False)
            encoder_layers = self._get_vision_encoder_layers()
            n_total = len(encoder_layers)
            n_unfreeze = min(tune_visual_last_n_layers, n_total)
            for layer in encoder_layers[-n_unfreeze:]:
                layer.requires_grad_(True)
            # post-encoder layernorm sits between the last encoder block and
            # the projector; train it so the new features can be re-normalised.
            post_ln = self._get_vision_post_layernorm()
            if post_ln is not None:
                post_ln.requires_grad_(True)
            print(
                f"Partial vision tune: last {n_unfreeze}/{n_total} encoder "
                f"layers + post-layernorm + mlp1 are trainable."
            )
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def _get_vision_encoder_layers(self):
        """Return the ModuleList of transformer blocks inside the vision tower.

        SigLIP layout: SiglipVisionModel.vision_model.encoder.layers
        RADIO layout falls back to whatever .encoder.layers exposes.
        """
        vm = self.eagle_model.vision_model
        # SiglipVisionModel wraps a SiglipVisionTransformer under `.vision_model`.
        inner = getattr(vm, "vision_model", vm)
        encoder = getattr(inner, "encoder", None)
        if encoder is None or not hasattr(encoder, "layers"):
            raise RuntimeError(
                "Cannot find vision encoder layers for partial unfreeze; "
                f"vision_model type={type(vm).__name__}"
            )
        return encoder.layers

    def _get_vision_post_layernorm(self):
        vm = self.eagle_model.vision_model
        inner = getattr(vm, "vision_model", vm)
        return getattr(inner, "post_layernorm", None)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )  # [B, T2, hidden_size]
