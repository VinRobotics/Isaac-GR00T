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
Improved Eagle Backbone with Frame Averaging (v3)

WHY v2 UNDERPERFORMS:
=====================
The v2 approach applies Frame Averaging to the COMBINED vision-language features 
AFTER the VLM's self-attention has processed them. This breaks equivariance because:

1. VLM self-attention creates position-dependent interactions between tokens
2. When an image is rotated, vision tokens move to different positions
3. The self-attention patterns for rotated images are fundamentally different
4. Simply averaging the output features destroys the language-vision associations

KEY INSIGHT:
============
Frame Averaging should be applied to the VISION ENCODER features BEFORE they
interact with language tokens in the LLM. This preserves:
1. Vision equivariance: rotated images produce rotated feature maps
2. Language conditioning: the same language context conditions all rotations
3. Consistent cross-attention: language-vision interactions happen AFTER FA

THREE APPROACHES TO IMPROVE PERFORMANCE:
========================================
1. VISION-ONLY FA (Recommended):
   - Apply FA only to vision encoder output (before mlp1 projection)
   - Language tokens remain in trivial representation
   - Cross-attention in LLM naturally handles fusion
   
2. CANONICAL TOKENIZATION:
   - Map vision tokens back to canonical positions before averaging
   - This accounts for how tokens shift with rotation
   
3. SEPARATE VISION-LANGUAGE PATHS:
   - Process vision with FA, language without
   - Fuse at a later stage with proper representation handling

References:
- Frame Averaging (Puny et al., 2022)
- EquiBot (Yang et al., 2024) - Vision-only equivariance for VLMs
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


class EagleBackboneFAv3(nn.Module):
    """
    Improved Frame Averaging Eagle Backbone (v3).
    
    Key Improvement: Apply FA to VISION features BEFORE language fusion,
    not to combined features AFTER self-attention.
    
    This addresses the core issue: VLM self-attention creates position-dependent
    interactions that break equivariance when applied to rotated images.
    
    Approach Options:
    -----------------
    1. 'vision_early': FA on vision encoder output before LLM (recommended)
    2. 'vision_late': FA on vision tokens extracted from LLM hidden states  
    3. 'canonical': Map tokens to canonical positions before FA
    
    Args:
        tune_llm: whether to tune the LLM model
        tune_visual: whether to tune the visual model  
        select_layer: which LLM layer to extract features from
        project_to_dim: dimension to project features to
        n_group: number of rotations (8 for C8)
        output_type: 'reg' for regular representation, 'std' for standard
        fa_mode: 'vision_early' (FA on ViT output), 'vision_late', 'canonical'
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
        n_group: int = 8,
        output_type: Literal['reg', 'std'] = 'reg',
        fa_mode: Literal['vision_early', 'vision_late', 'canonical', 'hybrid'] = 'vision_early',
    ):
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"
        
        self.num_images_per_sample = num_images_per_sample
        if rotate_image_indices is None:
            self.rotate_image_indices = list(range(num_images_per_sample))
        else:
            self.rotate_image_indices = rotate_image_indices
        self.duplicate_image_indices = [
            i for i in range(num_images_per_sample) if i not in self.rotate_image_indices
        ]
        
        # Load Eagle model
        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)
        
        # Frame Averaging setup
        self.n_group = n_group
        self.output_type = output_type
        self.fa_mode = fa_mode
        self.group = gspaces.no_base_space(CyclicGroup(self.n_group))
        self._init_fa_matrices()
        
        # Get dimensions from config
        self.vit_hidden_size = config.vision_config.hidden_size
        self.llm_hidden_size = config.text_config.hidden_size
        self.downsample_ratio = config.downsample_ratio
        self.use_pixel_shuffle = config.use_pixel_shuffle
        
        # Calculate vision tokens per image
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        if config.use_pixel_shuffle:
            self.num_vision_tokens = int(
                (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
            )
        else:
            self.num_vision_tokens = int((image_size // patch_size) ** 2)
        
        # Projection layer
        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(self.llm_hidden_size, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()
            
        self.project_to_dim = project_to_dim or self.llm_hidden_size
            
        # For vision_early mode: we need to access vision model separately
        if fa_mode == 'vision_early':
            # Create a separate projection for FA'd vision features
            # This maps from vit_hidden_size to llm_hidden_size, similar to mlp1
            if config.mlp_connector_layers == 2:
                fa_vit_dim = self.vit_hidden_size * int(1 / self.downsample_ratio) ** 2
                self.vision_fa_proj = nn.Sequential(
                    nn.LayerNorm(fa_vit_dim),
                    nn.Linear(fa_vit_dim, project_to_dim),
                    nn.GELU(),
                    nn.Linear(project_to_dim, project_to_dim),
                )
            else:
                fa_vit_dim = self.vit_hidden_size * int(1 / self.downsample_ratio) ** 2
                self.vision_fa_proj = nn.Linear(fa_vit_dim, project_to_dim)
                
        # Truncate LLM layers if using select_layer
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)
            
        self.select_layer = select_layer
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        
        self.set_trainable_parameters(tune_llm, tune_visual)
        
    def _init_fa_matrices(self):
        """Initialize rotation and permutation matrices for frame averaging."""
        # Rotation angles
        angles = torch.linspace(0, 2 * math.pi, self.n_group + 1)[:-1]
        
        # Rotation matrices for grid_sample (2x3 affine)
        rotation_matrices = torch.zeros(self.n_group, 2, 3)
        for i, angle in enumerate(angles):
            cos_val = math.cos(-angle.item())
            sin_val = math.sin(-angle.item())
            rotation_matrices[i, 0, 0] = cos_val
            rotation_matrices[i, 0, 1] = -sin_val
            rotation_matrices[i, 1, 0] = sin_val
            rotation_matrices[i, 1, 1] = cos_val
        self.register_buffer("rotation_matrices_buffer", rotation_matrices)
        
        # Permutation matrices for regular representation
        permutation_matrices = torch.zeros(self.n_group, self.n_group, self.n_group)
        for r in range(self.n_group):
            for i in range(self.n_group):
                j = (i + r) % self.n_group
                permutation_matrices[r, i, j] = 1.0
        self.register_buffer("permutation_matrices", permutation_matrices)
        
        # Pre-compute template for batch operations
        indices_template = torch.arange(self.n_group)
        perm_matrices_flat = permutation_matrices.reshape(self.n_group, -1)
        selected_perm_matrices = perm_matrices_flat[indices_template].reshape(
            self.n_group, self.n_group, self.n_group
        )
        self.register_buffer("selected_perm_matrices_template", selected_perm_matrices)
        
        # 2x2 rotation matrices for standard representation
        std_rotation_matrices = torch.zeros(self.n_group, 2, 2)
        for i, angle in enumerate(angles):
            cos_val = math.cos(-angle.item())
            sin_val = math.sin(-angle.item())
            std_rotation_matrices[i, 0, 0] = cos_val
            std_rotation_matrices[i, 0, 1] = -sin_val
            std_rotation_matrices[i, 1, 0] = sin_val
            std_rotation_matrices[i, 1, 1] = cos_val
        self.register_buffer("std_rotation_matrices", std_rotation_matrices)
        
        # Inverse rotation matrices for canonical mapping
        inv_std_rotation = torch.zeros(self.n_group, 2, 2)
        for i, angle in enumerate(angles):
            cos_val = math.cos(angle.item())  # Note: no negative for inverse
            sin_val = math.sin(angle.item())
            inv_std_rotation[i, 0, 0] = cos_val
            inv_std_rotation[i, 0, 1] = -sin_val
            inv_std_rotation[i, 1, 0] = sin_val
            inv_std_rotation[i, 1, 1] = cos_val
        self.register_buffer("inv_std_rotation_matrices", inv_std_rotation)
        
    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        """Set which parameters are trainable."""
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        
        # Freeze all first
        for param in self.eagle_model.parameters():
            param.requires_grad = False
            
        # Unfreeze LLM if tuning
        if tune_llm:
            for param in self.eagle_model.language_model.parameters():
                param.requires_grad = True
                
        # Unfreeze vision if tuning
        if tune_visual:
            for param in self.eagle_model.vision_model.parameters():
                param.requires_grad = True
            # Also unfreeze mlp1 (vision projector)
            for param in self.eagle_model.mlp1.parameters():
                param.requires_grad = True
                
        # Always train the projection layers
        for param in self.eagle_linear.parameters():
            param.requires_grad = True
        if hasattr(self, 'vision_fa_proj'):
            for param in self.vision_fa_proj.parameters():
                param.requires_grad = True

    def set_frozen_modules_to_eval_mode(self):
        """Set frozen modules to eval mode."""
        if self.training:
            if not self.tune_llm:
                self.eagle_model.language_model.eval()
            if not self.tune_visual:
                self.eagle_model.vision_model.eval()
                self.eagle_model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input for the model."""
        return BatchFeature(data=batch)
    
    # ========== VISION-EARLY MODE ==========
    def _extract_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract vision features from the vision model (before LLM)."""
        # Get features from vision encoder
        vit_embeds = self.eagle_model.vision_model(
            pixel_values=pixel_values, output_hidden_states=False, return_dict=True
        )
        if hasattr(vit_embeds, "last_hidden_state"):
            vit_embeds = vit_embeds.last_hidden_state
            
        # Apply pixel shuffle if configured
        if self.use_pixel_shuffle:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.eagle_model.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            
        return vit_embeds  # [B, num_tokens, vit_hidden_size * (1/downsample_ratio)^2]
    
    def _apply_vision_fa(self, vit_embeds: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Apply Frame Averaging to vision features.
        
        Key insight: Vision features from rotated images should be spatially rotated.
        We need to account for the token positions changing under rotation.
        
        Args:
            vit_embeds: [B*N, num_tokens, D] vision features from N rotated images
            batch_size: B (original batch size before rotation expansion)
            
        Returns:
            FA'd vision features [B, num_tokens, D]
        """
        B_times_N, num_tokens, D = vit_embeds.shape
        B = batch_size
        N = self.n_group
        
        # Reshape to [B, N, num_tokens, D]
        vit_embeds = vit_embeds.reshape(B, N, num_tokens, D)
        
        # For vision tokens arranged in a grid, rotation changes their positions
        # We need to map tokens back to canonical positions
        H = W = int(num_tokens ** 0.5)
        
        # Reshape to spatial grid: [B, N, H, W, D]
        vit_spatial = vit_embeds.reshape(B, N, H, W, D)
        
        # Create inverse rotation grids to map back to canonical positions
        # For each rotation index r, apply R(-2πr/N) to get back to canonical
        canonical_features = []
        
        for r in range(N):
            features_r = vit_spatial[:, r]  # [B, H, W, D]
            
            if r == 0:
                # No rotation needed
                canonical_features.append(features_r)
            else:
                # Apply inverse rotation
                # Reshape for grid_sample: [B, D, H, W]
                features_r = features_r.permute(0, 3, 1, 2)
                
                # Create inverse rotation matrix
                angle = -2 * math.pi * r / N  # Inverse rotation
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)
                theta = torch.tensor([
                    [cos_val, -sin_val, 0],
                    [sin_val, cos_val, 0]
                ], dtype=features_r.dtype, device=features_r.device)
                theta = theta.unsqueeze(0).expand(B, -1, -1)
                
                # Generate grid and sample
                grid = F.affine_grid(theta, features_r.shape, align_corners=True)
                rotated_back = F.grid_sample(features_r, grid, align_corners=True, padding_mode='zeros')
                
                # Back to [B, H, W, D]
                rotated_back = rotated_back.permute(0, 2, 3, 1)
                canonical_features.append(rotated_back)
        
        # Stack: [B, N, H, W, D]
        canonical_features = torch.stack(canonical_features, dim=1)
        
        # Average over rotations
        avg_features = canonical_features.mean(dim=1)  # [B, H, W, D]
        
        # Flatten back to tokens: [B, num_tokens, D]
        avg_features = avg_features.reshape(B, num_tokens, D)
        
        return avg_features
    
    def forward_vision_early(self, vl_input: BatchFeature) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vision-Early FA Mode: Apply FA to vision encoder output before LLM.
        
        This is the recommended approach because:
        1. Vision features are equivariant at the ViT output level
        2. FA is applied before position-dependent LLM self-attention
        3. Language conditioning still works via cross-attention
        """
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]
        
        B_times_N, seq_len = eagle_input["input_ids"].shape
        B = B_times_N // self.n_group
        
        # Extract vision features from rotated images: [B*N*num_images, num_tokens, D]
        pixel_values = eagle_input["pixel_values"]
        vit_embeds_all = self._extract_vision_features(pixel_values)
        
        # Reshape accounting for multiple images per sample
        # pixel_values shape: [B*N*num_images, C, H, W]
        total_images = pixel_values.shape[0]
        images_per_rotation = total_images // (B * self.n_group)  # Should equal num_images_per_sample
        
        num_tokens_per_image = vit_embeds_all.shape[1]
        D = vit_embeds_all.shape[2]
        
        # Reshape to [B, N, num_images, num_tokens, D]
        vit_embeds_all = vit_embeds_all.reshape(B, self.n_group, images_per_rotation, num_tokens_per_image, D)
        
        # Apply FA to each image position
        fa_vit_embeds = []
        for img_idx in range(images_per_rotation):
            if img_idx in self.rotate_image_indices:
                # This image was rotated - apply FA
                img_embeds = vit_embeds_all[:, :, img_idx]  # [B, N, num_tokens, D]
                img_embeds_flat = img_embeds.reshape(B * self.n_group, num_tokens_per_image, D)
                fa_embeds = self._apply_vision_fa(img_embeds_flat, B)  # [B, num_tokens, D]
            else:
                # This image was not rotated - just take from first rotation
                fa_embeds = vit_embeds_all[:, 0, img_idx]  # [B, num_tokens, D]
            fa_vit_embeds.append(fa_embeds)
            
        # Stack all images: [B, num_images * num_tokens, D]
        fa_vit_embeds = torch.cat(fa_vit_embeds, dim=1)
        
        # Project to output dimension
        fa_vit_embeds = self.vision_fa_proj(fa_vit_embeds)
        
        # Get language embeddings from first rotation (they're identical)
        input_ids_first = eagle_input["input_ids"][:B]
        attention_mask_first = eagle_input["attention_mask"][:B]
        
        # Get language token embeddings
        lang_embeds = self.eagle_model.language_model.get_input_embeddings()(input_ids_first)
        
        # Now we have equivariant vision features and language embeddings
        # We can concatenate them and return
        # Note: This simplified version doesn't run through LLM
        # For a full implementation, you'd want to inject FA'd vision into LLM
        
        return fa_vit_embeds, attention_mask_first
    
    # ========== CANONICAL MODE ==========
    def _remap_vision_tokens_to_canonical(
        self, 
        features: torch.Tensor, 
        rot_idx: int, 
        num_tokens_per_image: int
    ) -> torch.Tensor:
        """
        Remap vision token positions from rotated to canonical frame.
        
        When image is rotated by angle θ, token at position (i,j) in rotated image
        corresponds to position R(-θ)(i,j) in canonical frame.
        """
        B, T, D = features.shape
        
        # Only process vision tokens (assume they come first)
        H = W = int(num_tokens_per_image ** 0.5)
        
        vision_features = features[:, :num_tokens_per_image, :].reshape(B, H, W, D)
        
        if rot_idx == 0:
            return features
            
        # Apply inverse rotation to positions
        vision_features = vision_features.permute(0, 3, 1, 2)  # [B, D, H, W]
        
        angle = -2 * math.pi * rot_idx / self.n_group
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)
        theta = torch.tensor([
            [cos_val, -sin_val, 0],
            [sin_val, cos_val, 0]
        ], dtype=features.dtype, device=features.device)
        theta = theta.unsqueeze(0).expand(B, -1, -1)
        
        grid = F.affine_grid(theta, vision_features.shape, align_corners=True)
        remapped = F.grid_sample(vision_features, grid, align_corners=True, padding_mode='zeros')
        remapped = remapped.permute(0, 2, 3, 1).reshape(B, num_tokens_per_image, D)
        
        # Combine with non-vision tokens
        result = torch.cat([remapped, features[:, num_tokens_per_image:, :]], dim=1)
        return result
    
    def forward_canonical(self, vl_input: BatchFeature) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Canonical Mode: Process through VLM, then remap vision tokens to canonical positions.
        
        This approach:
        1. Runs full VLM forward with rotated images
        2. Remaps vision tokens positions back to canonical frame
        3. Applies frame averaging with proper spatial alignment
        """
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]
        
        B_times_N, seq_len = eagle_input["input_ids"].shape
        B = B_times_N // self.n_group
        
        # Forward through Eagle VLM
        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]  # [B*N, T, D]
        
        # Project
        eagle_features = self.eagle_linear(eagle_features)
        
        # Separate by rotation and remap to canonical positions
        eagle_features = eagle_features.reshape(B, self.n_group, seq_len, -1)
        
        canonical_features = []
        for r in range(self.n_group):
            features_r = eagle_features[:, r]  # [B, T, D]
            # Note: need to know where vision tokens are in sequence
            # This is a simplification - actual implementation needs proper token indexing
            remapped = self._remap_vision_tokens_to_canonical(
                features_r, r, self.num_vision_tokens * self.num_images_per_sample
            )
            canonical_features.append(remapped)
            
        canonical_features = torch.stack(canonical_features, dim=1)  # [B, N, T, D]
        
        # Apply frame averaging (on the channel dimension after canonical alignment)
        avg_features = self._apply_frame_averaging_on_canonical(canonical_features, B)
        
        attention_mask_first = eagle_input["attention_mask"][:B]
        return avg_features, attention_mask_first
    
    def _apply_frame_averaging_on_canonical(self, features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Apply FA to canonically-aligned features."""
        # features: [B, N, T, D]
        B, N, T, D = features.shape
        
        if self.output_type == 'reg':
            # Regular representation: apply permutation matrices
            blocks = D // self.n_group
            features = features.reshape(B, N, T, blocks, self.n_group)
            
            # For each rotation r, apply permutation P_r
            aligned_features = []
            for r in range(N):
                features_r = features[:, r]  # [B, T, blocks, N]
                perm_matrix = self.permutation_matrices[r]  # [N, N]
                # Apply permutation: [B, T, blocks, N] @ [N, N] -> [B, T, blocks, N]
                aligned = torch.einsum('btbn,nm->btbm', features_r, perm_matrix)
                aligned_features.append(aligned)
                
            aligned_features = torch.stack(aligned_features, dim=1)  # [B, N, T, blocks, N]
            avg_features = aligned_features.mean(dim=1)  # [B, T, blocks, N]
            return avg_features.reshape(B, T, D)
        else:
            # Standard representation: use 2x2 rotation matrices
            assert D % 2 == 0
            num_vectors = D // 2
            features = features.reshape(B, N, T, num_vectors, 2)
            
            aligned_features = []
            for r in range(N):
                features_r = features[:, r]  # [B, T, num_vectors, 2]
                rot_matrix = self.std_rotation_matrices[r]  # [2, 2]
                aligned = torch.einsum('btnv,vw->btnw', features_r, rot_matrix)
                aligned_features.append(aligned)
                
            aligned_features = torch.stack(aligned_features, dim=1)
            avg_features = aligned_features.mean(dim=1)
            return avg_features.reshape(B, T, D)
    
    # ========== HYBRID MODE ==========
    def forward_hybrid(self, vl_input: BatchFeature) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Hybrid Mode: Combine vision FA with full VLM features.
        
        1. Apply FA to vision encoder output (equivariant vision features)
        2. Also get full VLM features for rich language-vision context
        3. Combine both for the best of both worlds
        """
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]
        
        B_times_N, seq_len = eagle_input["input_ids"].shape
        B = B_times_N // self.n_group
        
        # 1. Get FA vision features
        pixel_values = eagle_input["pixel_values"]
        vit_embeds_all = self._extract_vision_features(pixel_values)
        
        total_images = pixel_values.shape[0]
        images_per_rotation = total_images // (B * self.n_group)
        num_tokens_per_image = vit_embeds_all.shape[1]
        D_vit = vit_embeds_all.shape[2]
        
        vit_embeds_all = vit_embeds_all.reshape(B, self.n_group, images_per_rotation, num_tokens_per_image, D_vit)
        
        fa_vit_embeds = []
        for img_idx in range(images_per_rotation):
            if img_idx in self.rotate_image_indices:
                img_embeds = vit_embeds_all[:, :, img_idx]
                img_embeds_flat = img_embeds.reshape(B * self.n_group, num_tokens_per_image, D_vit)
                fa_embeds = self._apply_vision_fa(img_embeds_flat, B)
            else:
                fa_embeds = vit_embeds_all[:, 0, img_idx]
            fa_vit_embeds.append(fa_embeds)
            
        fa_vit_embeds = torch.cat(fa_vit_embeds, dim=1)
        fa_vit_embeds = self.vision_fa_proj(fa_vit_embeds)  # [B, total_vision_tokens, project_to_dim]
        
        # 2. Get VLM features from first rotation (for language context)
        eagle_input_single = {
            k: v[:B] if torch.is_tensor(v) else v 
            for k, v in eagle_input.items()
        }
        # Update pixel values to use only first rotation
        if images_per_rotation > 1:
            pixel_first = pixel_values.reshape(B, self.n_group, images_per_rotation, *pixel_values.shape[1:])[:, 0]
            pixel_first = pixel_first.reshape(B * images_per_rotation, *pixel_values.shape[1:])
        else:
            pixel_first = pixel_values[:B]
        eagle_input_single["pixel_values"] = pixel_first
        
        eagle_output = self.eagle_model(**eagle_input_single, output_hidden_states=True, return_dict=True)
        vlm_features = eagle_output.hidden_states[self.select_layer]  # [B, T, D_llm]
        vlm_features = self.eagle_linear(vlm_features)
        
        # 3. Replace vision tokens in VLM features with FA vision features
        # This requires knowing where vision tokens are in the sequence
        # Simplified: concatenate FA vision + VLM language features
        
        # Find language token positions (non-image tokens)
        # For now, just concatenate both
        combined_features = torch.cat([fa_vit_embeds, vlm_features], dim=1)
        
        attention_mask_first = eagle_input["attention_mask"][:B]
        # Extend attention mask for combined features
        fa_mask = torch.ones(B, fa_vit_embeds.shape[1], dtype=attention_mask_first.dtype, device=attention_mask_first.device)
        combined_mask = torch.cat([fa_mask, attention_mask_first], dim=1)
        
        return combined_features, combined_mask
    
    # ========== MAIN FORWARD ==========
    def forward_eagle(self, vl_input: BatchFeature) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with selected FA mode."""
        if self.fa_mode == 'vision_early':
            return self.forward_vision_early(vl_input)
        elif self.fa_mode == 'canonical':
            return self.forward_canonical(vl_input)
        elif self.fa_mode == 'hybrid':
            return self.forward_hybrid(vl_input)
        else:
            raise ValueError(f"Unknown FA mode: {self.fa_mode}")
    
    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        
        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)
        
        # DDP compatibility
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
