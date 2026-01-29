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
"""
Improved Eagle Backbone with Frame Averaging (v2)

Key design principles:
1. Frame averaging is applied to VLM features (combined vision-language tokens)
2. NO additional fusion layers in backbone - language-vision fusion happens via 
   self-attention in the downstream flow matching action head
3. Support for both regular and standard representations in output
4. Rotated language descriptions (in GR00TTransformFA) help VLM understand viewpoint transformations

The key insight from Frame Averaging (Puny et al., 2022) is:
    Ψ(x) = (1/|G|) Σ_{g∈G} ρ_y(g) Φ(ρ_x(g)^{-1} x)

Where:
- For vision: x is the image, ρ_x(g) is image rotation
- For language: language is in trivial representation (invariant under rotations)
  but we use rotation-aware descriptions to help the VLM understand the transformation
- The output ρ_y(g) determines how features transform (regular or standard representation)

This implementation keeps the backbone simple:
- No additional attention or fusion modules
- Language-vision fusion naturally happens in the action head's self-attention
- The backbone just extracts equivariant features via frame averaging
"""

import os
import math
from typing import Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup 
import einops
import numpy as np

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackboneFAv2(nn.Module):
    """
    Improved Frame Averaging Eagle Backbone.
    
    Key Design Principles:
    1. Frame averaging is applied to VLM features (combined vision-language)
    2. NO additional fusion layers in backbone - fusion happens via self-attention 
       in the downstream flow matching action head
    3. Support for both regular and standard representations in output
    4. Rotated language descriptions help VLM understand viewpoint transformations
    
    The backbone outputs equivariant features, and the action head's self-attention
    naturally fuses language context with vision features.
    
    Args:
        tune_llm: whether to tune the LLM model
        tune_visual: whether to tune the visual model  
        select_layer: which LLM layer to extract features from
        project_to_dim: dimension to project features to
        n_group: number of rotations (8 for C8)
        output_type: 'reg' for regular representation, 'std' for standard (2D) representation
    """

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
        num_images_per_sample: int = 1,
        rotate_image_indices: list[int] | None = None,
        n_group: int = 4,
        output_type: Literal['reg', 'std'] = 'reg',
    ):
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"
        
        self.num_images_per_sample = num_images_per_sample
        self.n_group = n_group
        self.output_type = output_type
        
        # Default: rotate all images if not specified
        if rotate_image_indices is None:
            self.rotate_image_indices = list(range(num_images_per_sample))
        else:
            self.rotate_image_indices = rotate_image_indices
        self.duplicate_image_indices = [i for i in range(num_images_per_sample) if i not in self.rotate_image_indices]

        # Initialize Eagle VLM
        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)
        
        # Frame Average group setup
        self.group = gspaces.no_base_space(CyclicGroup(self.n_group))
        self._init_rotation_matrices()
        self._init_frame_averaging_matrices()
        
        # Projection layers (only existing weight)
        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
            self.feature_dim = project_to_dim
        else:
            self.eagle_linear = torch.nn.Identity()
            self.feature_dim = 2048

        # Remove unused LLM layers for efficiency
        while len(self.eagle_model.language_model.model.layers) > abs(select_layer):
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        
        # NO additional fusion modules - fusion happens in action head via self-attention
            
        self.set_trainable_parameters(tune_llm, tune_visual)
        
    def _init_rotation_matrices(self):
        """Initialize rotation matrices for image rotation."""
        angles = torch.linspace(0, 2 * math.pi, self.n_group + 1)[:-1]
        rotation_matrices = torch.zeros(self.n_group, 2, 3)
        
        for i, angle in enumerate(angles):
            cos_val = math.cos(-angle.item())
            sin_val = math.sin(-angle.item())
            rotation_matrices[i, 0, 0] = cos_val
            rotation_matrices[i, 0, 1] = -sin_val
            rotation_matrices[i, 1, 0] = sin_val
            rotation_matrices[i, 1, 1] = cos_val
            
        self.register_buffer("rotation_matrices_buffer", rotation_matrices)
        
        # Store angles for standard representation
        self.register_buffer("rotation_angles", angles)
        
    def _init_frame_averaging_matrices(self):
        """Initialize matrices for frame averaging (regular representation)."""
        # Permutation matrices for regular representation
        permutation_matrices = torch.zeros(self.n_group, self.n_group, self.n_group)
        for r in range(self.n_group):
            for i in range(self.n_group):
                j = (i + r) % self.n_group
                permutation_matrices[r, i, j] = 1.0
        
        self.register_buffer("permutation_matrices", permutation_matrices)
        
        perm_matrices_flat = permutation_matrices.reshape(self.n_group, -1)
        self.register_buffer("perm_matrices_flat", perm_matrices_flat)
        
        indices_template = torch.arange(self.n_group)
        self.register_buffer("indices_template", indices_template)
        
        selected_perm_matrices_template = perm_matrices_flat[indices_template].reshape(
            self.n_group, self.n_group, self.n_group
        )
        self.register_buffer("selected_perm_matrices_template", selected_perm_matrices_template)
        
        # 2x2 rotation matrices for standard representation
        angles = torch.linspace(0, 2 * math.pi, self.n_group + 1)[:-1]
        std_rotation_matrices = torch.zeros(self.n_group, 2, 2)
        for i, angle in enumerate(angles):
            cos_val = math.cos(angle.item())  # Note: positive angle for output transformation
            sin_val = math.sin(angle.item())
            std_rotation_matrices[i, 0, 0] = cos_val
            std_rotation_matrices[i, 0, 1] = -sin_val
            std_rotation_matrices[i, 1, 0] = sin_val
            std_rotation_matrices[i, 1, 1] = cos_val
        self.register_buffer("std_rotation_matrices", std_rotation_matrices)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")

    def set_frozen_modules_to_eval_mode(self):
        """Set frozen modules to eval mode for proper behavior."""
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
    
    def forward_eagle(self, vl_input: BatchFeature) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Eagle model with frame averaging.
        
        Process rotated images + language through VLM, then apply frame averaging.
        Language-vision fusion happens downstream in the action head via self-attention.
        """
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]
        
        # Input has B*N samples (N rotations per original sample)
        B_times_N, seq_len = eagle_input["input_ids"].shape
        B = B_times_N // self.n_group
        
        # Forward through Eagle VLM
        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]
        eagle_features = self.eagle_linear(eagle_features)
        
        # Rearrange for frame averaging: [B*N, T, D] -> [B*T, N, D]
        eagle_features = einops.rearrange(
            eagle_features, "(b n) t d -> (b t) n d", b=B, n=self.n_group
        )
        
        # Apply frame averaging
        Bt = eagle_features.shape[0]
        avg_feature = self._apply_frame_averaging(eagle_features, Bt)
        avg_feature = einops.rearrange(avg_feature, "(b t) d -> b t d", b=B)
        
        # Get attention mask from first rotation
        original_attention_mask = eagle_input["attention_mask"][:B]
        
        return avg_feature, original_attention_mask

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # DDP compatibility hack
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
        )

    def _apply_frame_averaging(self, features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Apply frame averaging to features.
        
        For regular representation: features transform via permutation matrices
        For standard representation: features transform via 2x2 rotation matrices
        
        Frame Averaging equation:
            Ψ(x) = (1/|G|) Σ_{g∈G} ρ_y(g) Φ(ρ_x(g)^{-1} x)
        
        Here:
        - Φ(ρ_x(g)^{-1} x) are the features from rotated inputs (already computed)
        - ρ_y(g) is the output representation transformation
        - We average over the group G
        
        Args:
            features: [B, N, D] where N is n_group, D is feature dimension
            batch_size: B
            
        Returns:
            Averaged features with shape depending on output_type:
            - 'reg': [B, D] features in regular representation
            - 'std': [B, D] features in standard (2D) representation
        """
        if self.output_type == 'reg':
            return self._apply_frame_averaging_regular(features, batch_size)
        else:
            return self._apply_frame_averaging_standard(features, batch_size)
    
    def _apply_frame_averaging_regular(self, features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Frame averaging for regular representation.
        
        In regular representation, features are organized into blocks of size N.
        Rotation by g permutes the blocks according to the group multiplication.
        """
        feature_dim = features.shape[2]
        blocks = feature_dim // self.n_group
        
        # Reshape: [B, N, blocks, N]
        features = features.reshape(batch_size, self.n_group, blocks, self.n_group)
        
        # Apply permutation matrices to align features from different rotations
        all_features_flat = features.reshape(batch_size * self.n_group, blocks, self.n_group)
        
        selected_perm_matrices = self.selected_perm_matrices_template.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )
        selected_perm_matrices = selected_perm_matrices.reshape(
            batch_size * self.n_group, self.n_group, self.n_group
        )
        
        # Apply permutation
        aligned_features_flat = torch.bmm(all_features_flat, selected_perm_matrices)
        aligned_features = aligned_features_flat.reshape(batch_size, self.n_group, blocks, self.n_group)
        
        # Average over group
        avg_features = torch.mean(aligned_features, dim=1)  # [B, blocks, N]
        
        return avg_features.reshape(batch_size, blocks * self.n_group)
    
    def _apply_frame_averaging_standard(self, features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Frame averaging for standard (2D vector) representation.
        
        In standard representation, features are organized into 2D vectors.
        Rotation by g applies a 2x2 rotation matrix to each vector.
        
        This is useful when you want the output to transform like 2D vectors
        (e.g., for predicting 2D positions or velocities).
        """
        feature_dim = features.shape[2]
        assert feature_dim % 2 == 0, "Feature dim must be even for standard representation"
        num_vectors = feature_dim // 2
        
        # Reshape: [B, N, num_vectors, 2]
        features = features.reshape(batch_size, self.n_group, num_vectors, 2)
        
        # Apply 2x2 rotation matrices to align features
        all_features_flat = features.reshape(batch_size * self.n_group, num_vectors, 2)
        
        # For each rotation r, apply R(2πr/N) to the output
        selected_rot_matrices = self.std_rotation_matrices.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )
        selected_rot_matrices = selected_rot_matrices.reshape(
            batch_size * self.n_group, 2, 2
        )
        
        # Apply rotation: [B*N, num_vectors, 2] @ [B*N, 2, 2] -> [B*N, num_vectors, 2]
        aligned_features_flat = torch.bmm(all_features_flat, selected_rot_matrices)
        aligned_features = aligned_features_flat.reshape(batch_size, self.n_group, num_vectors, 2)
        
        # Average over group
        avg_features = torch.mean(aligned_features, dim=1)  # [B, num_vectors, 2]
        
        return avg_features.reshape(batch_size, num_vectors * 2)


class EagleBackboneFAv2WithRotatedLanguage(EagleBackboneFAv2):
    """
    Extension that explicitly handles rotated language descriptions.
    
    The key insight is that providing rotation-aware language descriptions
    (e.g., "viewed from 45 degrees") helps the VLM understand the transformation,
    leading to better feature alignment during frame averaging.
    
    This is similar to what TransformFA does but integrated into the backbone.
    """
    
    def __init__(
        self,
        *args,
        use_rotated_descriptions: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_rotated_descriptions = use_rotated_descriptions
        
    def get_rotation_descriptions(self) -> list[str]:
        """
        Get natural language descriptions for each rotation angle.
        
        These descriptions help the VLM understand the viewpoint transformation,
        which improves feature consistency across rotations.
        """
        if self.n_group == 8:
            return [
                "Current view: canonical front view.",
                "Current view: 45° clockwise rotation.",
                "Current view: 90° right side view.",
                "Current view: 135° rear-right view.",
                "Current view: 180° rear view.",
                "Current view: 225° rear-left view.",
                "Current view: 270° left side view.",
                "Current view: 315° front-left view."
            ]
        elif self.n_group == 4:
            return [
                "Current view: canonical front view.",
                "Current view: 90° right side view.",
                "Current view: 180° rear view.",
                "Current view: 270° left side view."
            ]
        else:
            return [
                f"Current view: {int(360 * i / self.n_group)}° rotated view."
                for i in range(self.n_group)
            ]
