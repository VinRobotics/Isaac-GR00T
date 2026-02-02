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
Eagle Backbone with Late Frame Averaging (Late FA).

This module implements equivariant vision processing by:
1. Rotating input images N times (for CN group)
2. Processing ALL rotations through the FULL Eagle pipeline (Vision + LLM)
3. Applying frame averaging AFTER full VL processing

Key advantages over early aggregation (FA-SA):
- No information loss: full spatial tokens processed by LLM
- LLM can reason about complete spatial relationships
- Vision-language alignment preserved
- Mathematically exact equivariance

Trade-off: N× compute cost (but better performance)
"""

import os
import math
from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

from escnn import gspaces
from escnn.group import CyclicGroup

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackboneLateFa(nn.Module):
    """
    Eagle Backbone with Late Frame Averaging.
    
    Applies frame averaging AFTER full Eagle VL processing to achieve
    exact CN-equivariance while preserving all spatial information.
    
    Pipeline:
    1. Input: [B, n_img, C, H, W] images + text
    2. Rotate each image N times -> [B*N, n_img, C, H, W]
    3. Full Eagle (Vision + LLM) -> [B*N, T_tokens, D]
    4. Frame Averaging -> [B, T_tokens, D] in regular representation
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
        # Late FA specific parameters
        n_group: int = 8,  # Number of rotations (C4 = 4, C8 = 8)
        num_images_per_sample: int = 1,
        rotate_image_indices: List[int] | None = None,  # Which images to rotate (None = all)
        output_type: str = 'reg',  # 'reg' for regular representation
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model
            tune_visual: whether to tune the visual model
            select_layer: which LLM layer to extract features from
            project_to_dim: project features to this dimension (must be divisible by n_group for reg repr)
            n_group: number of rotations for CN group (4 for C4, 8 for C8)
            num_images_per_sample: number of images per sample
            rotate_image_indices: which image indices to rotate (None = all)
            output_type: 'reg' for regular representation output
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here"
        assert output_type == 'reg', "Only regular representation is supported"
        
        # Store config
        self.n_group = n_group
        self.num_images_per_sample = num_images_per_sample
        self.output_type = output_type
        self.project_to_dim = project_to_dim if project_to_dim else 2048
        
        if rotate_image_indices is None:
            self.rotate_image_indices = list(range(num_images_per_sample))
        else:
            self.rotate_image_indices = rotate_image_indices
        
        # Indices of images to duplicate (not rotate)
        self.duplicate_image_indices = [
            i for i in range(num_images_per_sample) 
            if i not in self.rotate_image_indices
        ]
        
        # Ensure project_to_dim is divisible by n_group for regular representation
        assert self.project_to_dim % n_group == 0, \
            f"project_to_dim ({self.project_to_dim}) must be divisible by n_group ({n_group})"

        # Initialize Eagle model
        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        # Projection layer
        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # Remove unused LLM layers
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        
        # Initialize rotation and frame averaging components
        self._init_rotation_matrices()
        self._init_permutation_matrices()
        
        self.set_trainable_parameters(tune_llm, tune_visual)
        
        print(f"EagleBackboneLateFa initialized:")
        print(f"  n_group (CN): {self.n_group}")
        print(f"  project_to_dim: {self.project_to_dim}")
        print(f"  rotate_image_indices: {self.rotate_image_indices}")
        print(f"  output_type: {self.output_type}")

    def _init_rotation_matrices(self):
        """Initialize rotation matrices for image rotation via grid_sample."""
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
        
        # Store angles for potential use
        self.register_buffer("angles", angles)

    def _init_permutation_matrices(self):
        """Initialize permutation matrices for frame averaging."""
        # Permutation matrices for regular representation
        # P_r[i, j] = 1 if j = (i + r) mod N
        permutation_matrices = torch.zeros(self.n_group, self.n_group, self.n_group)
        for r in range(self.n_group):
            for i in range(self.n_group):
                j = (i + r) % self.n_group
                permutation_matrices[r, i, j] = 1.0
        
        self.register_buffer("permutation_matrices", permutation_matrices)
        
        # Pre-compute flattened version for batch operations
        perm_matrices_flat = permutation_matrices.reshape(self.n_group, -1)
        self.register_buffer("perm_matrices_flat", perm_matrices_flat)
        
        # Template for selecting permutation matrices
        indices_template = torch.arange(self.n_group)
        self.register_buffer("indices_template", indices_template)
        
        selected_perm_matrices_template = perm_matrices_flat[indices_template].reshape(
            self.n_group, self.n_group, self.n_group
        )
        self.register_buffer("selected_perm_matrices_template", selected_perm_matrices_template)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        """Set which parameters are trainable."""
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
        
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
                    
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """Set frozen modules to eval mode for proper dropout/batchnorm behavior."""
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch."""
        return BatchFeature(data=batch)

    def _apply_rotations_to_images(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        Apply all N rotations to a batch of images.
        
        Args:
            img_batch: [B, C, H, W] tensor
            
        Returns:
            [B*N, C, H, W] tensor with all rotations
        """
        B, C, H, W = img_batch.shape
        device = img_batch.device
        
        # Expand: [B, C, H, W] -> [B, N, C, H, W] -> [B*N, C, H, W]
        expanded = img_batch.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)
        img_batch_expanded = expanded.reshape(B * self.n_group, C, H, W)
        
        # Create rotation indices: [0,1,2,3, 0,1,2,3, ...] for each image
        rotation_indices = torch.arange(self.n_group, device=device).repeat(B)
        
        # Get rotation matrices for each image
        rotation_matrices = self.rotation_matrices_buffer[rotation_indices]
        
        # Generate sampling grid
        grid = F.affine_grid(
            rotation_matrices.to(img_batch.dtype),
            size=(B * self.n_group, C, H, W),
            align_corners=True
        )
        
        # Apply rotations
        rotated_imgs = F.grid_sample(
            img_batch_expanded,
            grid,
            align_corners=True,
            padding_mode='zeros'
        )
        
        return rotated_imgs

    def rotate_vl_batch(self, vl_input: dict) -> dict:
        """
        Rotate images and expand text inputs for N rotations.
        
        Args:
            vl_input: dict with eagle_ prefixed keys
            
        Returns:
            Modified dict with rotated images and expanded text
        """
        # Extract eagle inputs
        eagle_prefix = "eagle_"
        input_ids = vl_input.get(f"{eagle_prefix}input_ids")
        attention_mask = vl_input.get(f"{eagle_prefix}attention_mask")
        pixel_values = vl_input.get(f"{eagle_prefix}pixel_values")
        
        if pixel_values is None:
            return vl_input
            
        total_imgs, C, H, W = pixel_values.shape
        B = input_ids.shape[0]
        
        # Handle multiple images per sample
        if self.num_images_per_sample > 1:
            # Reshape to [B, num_images, C, H, W]
            img_batch_reshaped = pixel_values.reshape(B, self.num_images_per_sample, C, H, W)
            
            processed_images = []
            for img_idx in range(self.num_images_per_sample):
                imgs_at_idx = img_batch_reshaped[:, img_idx, :, :, :]  # [B, C, H, W]
                
                if img_idx in self.rotate_image_indices:
                    # Apply rotations: [B, C, H, W] -> [B*N, C, H, W]
                    rotated = self._apply_rotations_to_images(imgs_at_idx)
                    processed_images.append(rotated)
                else:
                    # Duplicate without rotation: [B, C, H, W] -> [B*N, C, H, W]
                    expanded = imgs_at_idx.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)
                    duplicated = expanded.reshape(B * self.n_group, C, H, W)
                    processed_images.append(duplicated)
            
            # Reorganize: stack and reshape to [B*N*num_images, C, H, W]
            # Each group of N consecutive samples should have the same rotation
            processed_images_reshaped = [
                img.reshape(B, self.n_group, C, H, W) for img in processed_images
            ]
            stacked = torch.stack(processed_images_reshaped, dim=0)  # [num_img, B, N, C, H, W]
            stacked = stacked.permute(1, 2, 0, 3, 4, 5)  # [B, N, num_img, C, H, W]
            rotated_pixel_values = stacked.reshape(
                B * self.n_group * self.num_images_per_sample, C, H, W
            )
        else:
            # Single image per sample
            rotated_pixel_values = self._apply_rotations_to_images(pixel_values)
        
        # Expand text inputs N times
        # [B, seq_len] -> [B*N, seq_len]
        input_ids_expanded = input_ids.repeat_interleave(self.n_group, dim=0)
        attention_mask_expanded = attention_mask.repeat_interleave(self.n_group, dim=0)
        
        # Create new input dict
        rotated_vl_input = dict(vl_input)
        rotated_vl_input[f"{eagle_prefix}input_ids"] = input_ids_expanded
        rotated_vl_input[f"{eagle_prefix}attention_mask"] = attention_mask_expanded
        rotated_vl_input[f"{eagle_prefix}pixel_values"] = rotated_pixel_values
        
        return rotated_vl_input

    def _apply_frame_averaging(
        self, 
        features: torch.Tensor, 
        batch_size: int
    ) -> torch.Tensor:
        """
        Apply frame averaging to features in regular representation.
        
        Uses the same convention as C8EquivariantTimmObsEncoder:
        - Apply P_r (forward permutation) to features from rotation r
        - This convention defines ρ(g_r^{-1}) = P_r in the fiber space
        
        Args:
            features: [B*N, T, D] features from N rotations
            batch_size: original batch size B
            
        Returns:
            [B, T, D] features in regular representation
        """
        B_N, T, D = features.shape
        N = self.n_group
        B = batch_size
        
        assert B_N == B * N, f"Expected B*N={B*N}, got {B_N}"
        assert D % N == 0, f"Feature dim {D} must be divisible by N={N}"
        
        blocks = D // N
        
        # Reshape: [B*N, T, D] -> [B, N, T, blocks, N]
        features = features.reshape(B, N, T, blocks, N)
        
        # Flatten for batch matrix multiplication
        # [B, N, T, blocks, N] -> [B*N*T, blocks, N]
        features_flat = features.reshape(B * N * T, blocks, N)
        
        # Get permutation matrices for each rotation
        # For rotation r, we apply P_r (same as C8EquivariantTimmObsEncoder)
        # Create indices: [0,1,...,N-1, 0,1,...,N-1, ...] repeated B*T times
        rotation_indices = torch.arange(N, device=features.device).repeat(B * T)
        
        # Select P_r for each element: [B*N*T, N, N]
        perm_matrices = self.permutation_matrices[rotation_indices]
        
        # Apply permutation: features @ P_r (NOT transposed, matching C8EquivariantTimmObsEncoder)
        aligned_features_flat = torch.bmm(features_flat, perm_matrices)
        
        # Reshape back: [B*N*T, blocks, N] -> [B, N, T, blocks, N]
        aligned_features = aligned_features_flat.reshape(B, N, T, blocks, N)
        
        # Average over the N rotations (dim=1)
        avg_features = aligned_features.mean(dim=1)  # [B, T, blocks, N]
        
        # Reshape to final output: [B, T, D]
        return avg_features.reshape(B, T, D)

    def forward_eagle(self, vl_input: BatchFeature) -> tuple:
        """
        Forward through Eagle model with frame averaging.
        
        Args:
            vl_input: Input batch with eagle_ prefixed keys
            
        Returns:
            (features, attention_mask) tuple
        """
        eagle_prefix = "eagle_"
        
        # Get original batch size before rotation
        original_input_ids = vl_input.get(f"{eagle_prefix}input_ids")
        B = original_input_ids.shape[0]
        
        # Rotate images and expand text
        rotated_vl_input = self.rotate_vl_batch(dict(vl_input))
        
        # Prepare eagle input
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in rotated_vl_input.items()
            if k.startswith(eagle_prefix)
        }
        
        # Remove image_sizes if present
        if "image_sizes" in eagle_input:
            del eagle_input["image_sizes"]
        
        # Forward through Eagle model
        eagle_output = self.eagle_model(
            **eagle_input, 
            output_hidden_states=True, 
            return_dict=True
        )
        
        # Get features from selected layer
        eagle_features = eagle_output.hidden_states[self.select_layer]  # [B*N, T, 2048]
        
        # Project to target dimension
        eagle_features = self.eagle_linear(eagle_features)  # [B*N, T, D]
        
        # Apply frame averaging
        avg_features = self._apply_frame_averaging(eagle_features, B)  # [B, T, D]
        
        # Get attention mask (from original, not expanded)
        attention_mask = original_input_ids != 0  # Simple mask, or use the actual mask
        attention_mask = vl_input.get(f"{eagle_prefix}attention_mask", attention_mask)
        
        return avg_features, attention_mask

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """
        Forward pass with Late Frame Averaging.
        
        Args:
            vl_input: Input batch
            
        Returns:
            BatchFeature with equivariant backbone_features
        """
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
            data={
                "backbone_features": eagle_embeds,
                "backbone_attention_mask": eagle_mask
            }
        )


class EagleBackboneLateFaMemoryEfficient(EagleBackboneLateFa):
    """
    Memory-efficient version that processes rotations sequentially.
    
    Trades compute time for memory by processing one rotation at a time
    instead of all N rotations in parallel.
    """
    
    def forward_eagle(self, vl_input: BatchFeature) -> tuple:
        """
        Forward through Eagle model, processing rotations sequentially.
        """
        eagle_prefix = "eagle_"
        
        # Get original inputs
        original_input_ids = vl_input.get(f"{eagle_prefix}input_ids")
        original_attention_mask = vl_input.get(f"{eagle_prefix}attention_mask")
        original_pixel_values = vl_input.get(f"{eagle_prefix}pixel_values")
        
        B = original_input_ids.shape[0]
        device = original_input_ids.device
        
        if original_pixel_values is None:
            # No images, just process normally
            eagle_input = {
                k.removeprefix(eagle_prefix): v
                for k, v in vl_input.items()
                if k.startswith(eagle_prefix)
            }
            if "image_sizes" in eagle_input:
                del eagle_input["image_sizes"]
                
            eagle_output = self.eagle_model(
                **eagle_input,
                output_hidden_states=True,
                return_dict=True
            )
            eagle_features = eagle_output.hidden_states[self.select_layer]
            eagle_features = self.eagle_linear(eagle_features)
            return eagle_features, original_attention_mask
        
        total_imgs, C, H, W = original_pixel_values.shape
        
        # Process each rotation sequentially
        all_features = []
        
        for rot_idx in range(self.n_group):
            # Get rotation matrix for this rotation
            rotation_matrix = self.rotation_matrices_buffer[rot_idx:rot_idx+1]  # [1, 2, 3]
            rotation_matrix = rotation_matrix.expand(total_imgs, -1, -1)  # [total_imgs, 2, 3]
            
            # Apply rotation to images
            grid = F.affine_grid(
                rotation_matrix.to(original_pixel_values.dtype),
                size=(total_imgs, C, H, W),
                align_corners=True
            )
            rotated_pixel_values = F.grid_sample(
                original_pixel_values,
                grid,
                align_corners=True,
                padding_mode='zeros'
            )
            
            # Create input for this rotation
            eagle_input = {
                "input_ids": original_input_ids,
                "attention_mask": original_attention_mask,
                "pixel_values": rotated_pixel_values,
            }
            
            # Forward through Eagle
            eagle_output = self.eagle_model(
                **eagle_input,
                output_hidden_states=True,
                return_dict=True
            )
            
            eagle_features = eagle_output.hidden_states[self.select_layer]  # [B, T, 2048]
            eagle_features = self.eagle_linear(eagle_features)  # [B, T, D]
            
            all_features.append(eagle_features)
        
        # Stack features: [N, B, T, D] -> [B, N, T, D]
        stacked_features = torch.stack(all_features, dim=0)  # [N, B, T, D]
        stacked_features = stacked_features.permute(1, 0, 2, 3)  # [B, N, T, D]
        
        # Apply frame averaging on stacked features
        B, N, T, D = stacked_features.shape
        blocks = D // N
        
        # Reshape for permutation: [B, N, T, blocks, N]
        features = stacked_features.reshape(B, N, T, blocks, N)
        
        # Apply permutations (matching C8EquivariantTimmObsEncoder convention)
        features_flat = features.reshape(B * N * T, blocks, N)
        rotation_indices = torch.arange(N, device=device).repeat(B * T)
        perm_matrices = self.permutation_matrices[rotation_indices]
        # NO transpose - use P_r directly like C8EquivariantTimmObsEncoder
        aligned_features_flat = torch.bmm(features_flat, perm_matrices)
        aligned_features = aligned_features_flat.reshape(B, N, T, blocks, N)
        
        # Average
        avg_features = aligned_features.mean(dim=1)  # [B, T, blocks, N]
        avg_features = avg_features.reshape(B, T, D)
        
        return avg_features, original_attention_mask
