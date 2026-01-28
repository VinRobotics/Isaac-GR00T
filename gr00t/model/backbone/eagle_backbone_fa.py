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
        num_images_per_sample: int = 1,
        rotate_image_indices: list[int] | None = None,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
            num_images_per_sample: number of images per sample (e.g., 2 for two camera views)
            rotate_image_indices: list of indices (0-based) indicating which images should be 
                                  rotated. Images not in this list will be duplicated like input_ids.
                                  If None, all images will be rotated.
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"
        
        self.num_images_per_sample = num_images_per_sample
        # Default: rotate all images if not specified
        if rotate_image_indices is None:
            self.rotate_image_indices = list(range(num_images_per_sample))
        else:
            self.rotate_image_indices = rotate_image_indices
        # Indices of images to duplicate (not rotate)
        self.duplicate_image_indices = [i for i in range(num_images_per_sample) if i not in self.rotate_image_indices]

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
        Handles multiple images per sample, rotating only specified image indices.
        
        Args:
            vl_batch: dict containing:
                - pixel_values: [B*num_images_per_sample, C, H, W] tensor of RGB images
                - input_ids: [B, seq_len] tensor
                - attention_mask: [B, seq_len] tensor
            
        Returns:
            Modified vl_batch with:
                - pixel_values: [B*num_images_per_sample*N, C, H, W] for rotated images
                - input_ids: [B*N, seq_len] duplicated
                - attention_mask: [B*N, seq_len] duplicated
                
        For images at rotate_image_indices: apply N rotations
        For images at duplicate_image_indices: duplicate N times (like input_ids)
        """
        input_ids_batch = vl_batch["input_ids"]
        attn_mask_batch = vl_batch["attention_mask"]
        img_batch = vl_batch["pixel_values"]
        
        total_imgs, C, H, W = img_batch.shape
        B = total_imgs // self.num_images_per_sample
        
        # Reshape to [B, num_images_per_sample, C, H, W]
        img_batch_reshaped = img_batch.reshape(B, self.num_images_per_sample, C, H, W)
        
        # Process each image index separately
        processed_images = []
        
        for img_idx in range(self.num_images_per_sample):
            # Get all images at this index across the batch: [B, C, H, W]
            imgs_at_idx = img_batch_reshaped[:, img_idx, :, :, :]
            
            if img_idx in self.rotate_image_indices:
                # Apply rotations: [B, C, H, W] -> [B*N, C, H, W]
                rotated = self._apply_rotations(imgs_at_idx)
                processed_images.append(rotated)
            else:
                # Duplicate N times: [B, C, H, W] -> [B*N, C, H, W]
                # Expand each image to N copies, keeping them grouped together
                expanded = imgs_at_idx.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)
                duplicated = expanded.reshape(B * self.n_group, C, H, W)
                processed_images.append(duplicated)
        
        # Stack processed images: [num_images_per_sample, B*N, C, H, W]
        # Then rearrange to [B*N*num_images_per_sample, C, H, W]
        # We want the order to be: [sample0_img0_rot0, sample0_img1_rot0, ..., sample0_img0_rot1, sample0_img1_rot1, ...]
        # Which is: for each rotation, for each sample, all images
        
        # Current: each processed_images[i] has shape [B*N, C, H, W]
        # where the order is [sample0_rot0, sample0_rot1, ..., sample0_rotN-1, sample1_rot0, ...]
        
        # Reshape each to [B, N, C, H, W]
        processed_images_reshaped = [img.reshape(B, self.n_group, C, H, W) for img in processed_images]
        
        # Stack along new dimension: [num_images_per_sample, B, N, C, H, W]
        stacked = torch.stack(processed_images_reshaped, dim=0)
        
        # Rearrange to [B, N, num_images_per_sample, C, H, W]
        stacked = stacked.permute(1, 2, 0, 3, 4, 5)
        
        # Final reshape to [B*N*num_images_per_sample, C, H, W]
        rotated_imgs = stacked.reshape(B * self.n_group * self.num_images_per_sample, C, H, W)
        
        # Duplicate input_ids and attention_mask N times
        input_ids_batch_expanded = input_ids_batch.repeat((self.n_group, 1))
        attn_mask_batch_expanded = attn_mask_batch.repeat((self.n_group, 1))
        
        vl_batch["pixel_values"] = rotated_imgs
        vl_batch["input_ids"] = input_ids_batch_expanded
        vl_batch["attention_mask"] = attn_mask_batch_expanded
        return vl_batch
    
    def _apply_rotations(self, img_batch):
        """
        Apply all N rotations to a batch of images.
        
        Args:
            img_batch: [B, C, H, W] tensor
            
        Returns:
            [B*N, C, H, W] tensor with each image rotated N times
        """
        B, C, H, W = img_batch.shape
        
        # Expand each image to N copies: [B, N, C, H, W]
        expanded = img_batch.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)
        
        # Reshape to [B*N, C, H, W]
        img_batch_expanded = expanded.reshape(B * self.n_group, C, H, W)
        
        # Create rotation indices: [0,1,2,...,N-1, 0,1,2,...,N-1, ...]
        rotation_indices = torch.arange(self.n_group, device=img_batch.device).repeat(B)
        
        # Get rotation matrices: [B*N, 2, 3]
        rotation_matrices = self.rotation_matrices_buffer[rotation_indices]
        
        # Generate sampling grid
        grid = torch.nn.functional.affine_grid(
            rotation_matrices, 
            size=(B * self.n_group, C, H, W),
            align_corners=True
        )
        
        # Apply transformations
        rotated_imgs = torch.nn.functional.grid_sample(
            img_batch_expanded, 
            grid, 
            align_corners=True,
            padding_mode='zeros'
        )
        
        return rotated_imgs

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through Eagle model with frame averaging.
        
        Expects images to already be rotated by the transform (GR00TTransformFA).
        Input has B*N samples where N is n_group rotations per original sample.
        """
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]
        
        # The input now has B*N samples (N rotations per original sample)
        B_times_N, seq_len = eagle_input["input_ids"].shape
        B = B_times_N // self.n_group
        
        # Images are already rotated by GR00TTransformFA, no need to rotate here
        # Just forward through the Eagle model
        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        
        # Rearrange from [B*N, T, D] to [B*T, N, D] for frame averaging
        eagle_features = einops.rearrange(eagle_features, "(b n) t d -> (b t) n d", b=B, n=self.n_group)

        Bt, _, _ = eagle_features.shape
        
        avg_feature = self._apply_frame_averaging(eagle_features, Bt)
        avg_feature = einops.rearrange(avg_feature, "(b t) d -> b t d", b=B,)
        
        # Get the attention mask from the first rotation (they should all be the same)
        # Original mask shape is [B*N, seq_len], we need [B, seq_len]
        original_attention_mask = eagle_input["attention_mask"][:B]

        return avg_feature, original_attention_mask

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
            features: Tensor of shape [B, N, feature_dim] where N is n_group
            batch_size: Batch size (B)
            
        Returns:
            Tensor of shape [B, feature_dim] with frame averaging applied
        """
        # features shape: [B, N, feature_dim]
        feature_dim = features.shape[2]
        blocks = feature_dim // self.n_group
        
        # Reshape: [B, N, blocks, N] - each feature vector is split into blocks of size N
        features = features.reshape(batch_size, self.n_group, blocks, self.n_group)
        
        # For each rotation r, apply permutation matrix P_r to align the output
        # all_features_flat: [B*N, blocks, N]
        all_features_flat = features.reshape(batch_size * self.n_group, blocks, self.n_group)
        
        # selected_perm_matrices needs shape [B*N, N, N]
        # Repeat the template for each batch element
        selected_perm_matrices = self.selected_perm_matrices_template.unsqueeze(0).expand(batch_size, -1, -1, -1)
        selected_perm_matrices = selected_perm_matrices.reshape(batch_size * self.n_group, self.n_group, self.n_group)
        
        # Apply permutation: [B*N, blocks, N] @ [B*N, N, N] -> [B*N, blocks, N]
        aligned_features_flat = torch.bmm(all_features_flat, selected_perm_matrices)
        
        # Reshape back: [B, N, blocks, N]
        aligned_features = aligned_features_flat.reshape(batch_size, self.n_group, blocks, self.n_group)
        
        
        # Average over the group dimension: [B, blocks, N]
        avg_features = torch.mean(aligned_features, dim=1)
        
        return avg_features.reshape(batch_size, blocks * self.n_group)