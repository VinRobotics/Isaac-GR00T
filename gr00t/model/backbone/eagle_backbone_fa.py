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

from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup 
import einops
import numpy as np
import math

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackboneFA(nn.Module):

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
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)
        
        # Frame Average
        self.n_group = 8 # hardcode
        self.group = gspaces.no_base_space(CyclicGroup(self.n_group))
        self.init_rotation_matrices()
        
        
        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        
        self.set_trainable_parameters(tune_llm, tune_visual)
        
    def init_rotation_matrices(self):
        # Precompute rotation matrices for grid_sample
        angles = torch.linspace(0, 2 * math.pi, self.n_group+1)[:-1]  # N angles from 0 to 2π
        self.rotation_matrices = torch.zeros(self.n_group, 2, 3)

        for i, angle in enumerate(angles):
            cos_val = math.cos(-angle.item())
            sin_val = math.sin(-angle.item())
            
            self.rotation_matrices[i, 0, 0] = cos_val
            self.rotation_matrices[i, 0, 1] = -sin_val
            self.rotation_matrices[i, 1, 0] = sin_val
            self.rotation_matrices[i, 1, 1] = cos_val
            
        # Register buffer for rotation matrices
        self.register_buffer("rotation_matrices_buffer", self.rotation_matrices)
        
        # Get the group elements for transformations
        self.group_elements = list(self.group.testing_elements)
        
        # Initialize permutation matrices for frame averaging
        permutation_matrices = torch.zeros(self.n_group, self.n_group, self.n_group)
        for r in range(self.n_group):
            for i in range(self.n_group):
                j = (i + r) % self.n_group
                permutation_matrices[r, i, j] = 1.0
        
        # Register as buffer
        self.register_buffer("permutation_matrices", permutation_matrices)
        
        # Pre-compute the flattened permutation matrices for batch operations
        perm_matrices_flat = permutation_matrices.reshape(self.n_group, -1)
        self.register_buffer("perm_matrices_flat", perm_matrices_flat)
        
        # Pre-compute indices for selecting the appropriate permutation matrix for each rotation
        indices_template = torch.arange(self.n_group)
        self.register_buffer("indices_template", indices_template)
        
        # Pre-compute the selected permutation matrices for batch size 1
        selected_perm_matrices_template = perm_matrices_flat[indices_template].reshape(self.n_group, self.n_group, self.n_group)
        self.register_buffer("selected_perm_matrices_template", selected_perm_matrices_template)
        

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
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

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
    
    
    def rotate_rgb_batch(self, vl_batch):
        """
        Apply all N rotations to a batch of images efficiently in a single operation.
        
        Args:
            img_batch: [B, C, H, W] tensor of RGB images
            
        Returns:
            Batch of rotated images with shape [B*N, C, H, W], where N is the number of rotations
            The output is organized as [img0_rot0, img0_rot1, ..., img0_rotN-1, img1_rot0, ...]
        """
        for k, v in vl_batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape, v.dtype, v.device)
        input_ids_batch = vl_batch["input_ids"]
        attn_mask_batch = vl_batch["attention_mask"]
        img_batch = vl_batch["pixel_values"]
        B, C, H, W = img_batch.shape
        
        # Create an expanded batch by repeating each image N times
        # We need to ensure images and their rotations are grouped together
        # [B, C, H, W] -> [B*N, C, H, W] where each block of N images contains all rotations of a single input
        
        # First, expand each image to N copies
        # This creates [B, N, C, H, W] where each original image is repeated N times
        expanded = img_batch.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)
        
        # Reshape to [B*N, C, H, W]
        img_batch_expanded = expanded.reshape(B*self.n_group, C, H, W)

        # Now create the rotation matrices
        # For each image block of N copies, we need to apply a different rotation to each copy
        
        # Create pattern of indices for the N rotations, repeated for each image in the batch
        # For B=2, N=3 this would be [0,1,2, 0,1,2]
        rotation_indices = torch.arange(self.n_group, device=img_batch.device).repeat(B)
        
        # Use these indices to select the correct rotation matrix for each image
        # [B*N, 2, 3]
        rotation_matrices = self.rotation_matrices_buffer[rotation_indices]
        
        # Generate sampling grid for all rotations at once
        grid = torch.nn.functional.affine_grid(
            rotation_matrices, 
            size=(B*self.n_group, C, H, W),
            align_corners=True
        )
        
        # Apply all transformations in a single grid_sample operation
        rotated_imgs = torch.nn.functional.grid_sample(
            img_batch_expanded, 
            grid, 
            align_corners=True,
            padding_mode='zeros'
        )
        input_ids_batch_expanded = input_ids_batch.repeat((self.n_group, 1))
        attn_mask_batch_expanded = attn_mask_batch.repeat((self.n_group, 1))
        
        vl_batch["pixel_values"] = rotated_imgs
        vl_batch["input_ids"] = input_ids_batch_expanded
        vl_batch["attention_mask"] = attn_mask_batch_expanded
        return vl_batch

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]
        B, _ = eagle_input["input_ids"].shape
        rotated_vl_input = self.rotate_rgb_batch(eagle_input)
        
        for k, v in eagle_input.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape, v.dtype, v.device)

        eagle_output = self.eagle_model(**rotated_vl_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        eagle_features = einops.rearrange(eagle_features, "(b n) t d -> (b t) n d", b=B, n=self.n_group)

        Bt, _, _ = eagle_features.shape
        
        avg_feature = self._apply_frame_averaging(eagle_features, Bt)
        avg_feature = einops.rearrange(avg_feature, "(b t) d -> b t d", b=B,)

        return avg_feature, eagle_input["attention_mask"]

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


    def _apply_frame_averaging(self, features, batch_size):
        """
        Apply frame averaging to features using permutation matrices.
        
        Args:
            features: Tensor of shape [B, N, feature_dim]
            batch_size: Batch size
            
        Returns:
            Tensor of shape [B, feature_dim] with frame averaging applied
        """
        # Regular representation - use permutation matrices
        feature_dim = features.shape[2]
        blocks = feature_dim // self.n_group
        features = features.reshape(batch_size, self.n_group, blocks, self.n_group)

        all_features_flat = features.reshape(-1, blocks, self.n_group)
        selected_perm_matrices = self.selected_perm_matrices_template.repeat(batch_size, 1, 1)
        aligned_features_flat = torch.bmm(all_features_flat, selected_perm_matrices)
        aligned_features = aligned_features_flat.reshape(batch_size, self.n_group, blocks, self.n_group)
        avg_features = torch.mean(aligned_features, dim=1)  # [B, blocks, N]
        return avg_features.reshape(batch_size, blocks * self.n_group)