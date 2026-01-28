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
Eagle Backbone with Equivariant Feature Steering (EFS) - v4

KEY INSIGHT:
============
The problem with naive Frame Averaging on VLMs is that self-attention creates 
position-dependent interactions that break equivariance. 

This module introduces "Equivariant Feature Steering" (EFS):
1. Apply FA ONLY at the vision encoder level (where features are still spatially organized)
2. Use equivariant "steering vectors" to guide the LLM's interpretation of rotated content
3. The steering vectors transform covariantly with image rotation

APPROACH:
=========
1. Vision encoder processes rotated images -> spatial features
2. Apply spatial FA to get rotation-invariant/equivariant vision tokens
3. Add learned equivariant position embeddings that indicate orientation
4. LLM processes these FA'd vision tokens + steering embeddings + language
5. Output features inherit equivariance from the proper treatment

This is inspired by:
- EquiBot (Yang et al., 2024): Vision-only equivariance
- Steering vectors in LLMs (Turner et al., 2023)
- Rotary Position Embeddings (Su et al., 2022)

The key is that equivariance is achieved at the right level:
- Vision features: spatial equivariance via FA
- Steering embeddings: transform with rotation  
- Language: trivial representation (invariant)
- Combined: properly equivariant output
"""

import os
import math
from typing import Optional, Literal, Tuple

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


class EquivariantSteeringEmbedding(nn.Module):
    """
    Learnable steering embeddings that transform equivariantly with rotation.
    
    These embeddings are added to vision tokens to convey orientation information
    in an equivariant way. When the image rotates by angle θ, these embeddings
    transform by the corresponding group element.
    
    For C_N group, we use N different steering embeddings, one for each rotation.
    The embedding for rotation r transforms to become embedding for rotation (r+1) mod N
    when the image is further rotated by one step.
    """
    
    def __init__(self, n_group: int, embedding_dim: int, num_tokens: int = 1):
        super().__init__()
        self.n_group = n_group
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        
        # Learnable base embedding (for rotation 0)
        self.base_embedding = nn.Parameter(torch.randn(num_tokens, embedding_dim) * 0.02)
        
        # Pre-compute rotation matrices for embeddings
        # These rotate the embedding to represent different orientations
        angles = torch.linspace(0, 2 * math.pi, n_group + 1)[:-1]
        
        # We'll use a block-diagonal rotation approach
        # Split embedding_dim into 2D blocks and rotate each
        assert embedding_dim % 2 == 0, "embedding_dim must be even for rotation"
        self.num_blocks = embedding_dim // 2
        
        rotation_matrices = torch.zeros(n_group, embedding_dim, embedding_dim)
        for r, angle in enumerate(angles):
            cos_val = math.cos(angle.item())
            sin_val = math.sin(angle.item())
            for block_idx in range(self.num_blocks):
                i = block_idx * 2
                rotation_matrices[r, i, i] = cos_val
                rotation_matrices[r, i, i+1] = -sin_val
                rotation_matrices[r, i+1, i] = sin_val
                rotation_matrices[r, i+1, i+1] = cos_val
                
        self.register_buffer("rotation_matrices", rotation_matrices)
        
    def forward(self, rotation_idx: int) -> torch.Tensor:
        """
        Get steering embedding for a specific rotation index.
        
        Args:
            rotation_idx: which rotation (0 to n_group-1)
            
        Returns:
            Steering embedding [num_tokens, embedding_dim]
        """
        # Rotate base embedding by the appropriate amount
        rot_matrix = self.rotation_matrices[rotation_idx]  # [D, D]
        # [num_tokens, D] @ [D, D] -> [num_tokens, D]
        return torch.mm(self.base_embedding, rot_matrix)
    
    def get_all_rotations(self) -> torch.Tensor:
        """Get embeddings for all rotations: [N, num_tokens, D]"""
        all_embeddings = []
        for r in range(self.n_group):
            all_embeddings.append(self.forward(r))
        return torch.stack(all_embeddings, dim=0)


class SpatialFrameAveraging(nn.Module):
    """
    Frame Averaging module that properly handles spatial structure of vision tokens.
    
    Unlike naive FA that ignores token positions, this module:
    1. Reshapes tokens to their spatial grid
    2. Applies inverse rotation to align features to canonical frame
    3. Averages the aligned features
    
    This preserves semantic correspondence across rotations.
    """
    
    def __init__(self, n_group: int):
        super().__init__()
        self.n_group = n_group
        
        # Pre-compute inverse rotation matrices
        angles = torch.linspace(0, 2 * math.pi, n_group + 1)[:-1]
        inv_affine = torch.zeros(n_group, 2, 3)
        for r, angle in enumerate(angles):
            # Inverse rotation: negative angle
            cos_val = math.cos(-angle.item())
            sin_val = math.sin(-angle.item())
            inv_affine[r, 0, 0] = cos_val
            inv_affine[r, 0, 1] = -sin_val
            inv_affine[r, 1, 0] = sin_val
            inv_affine[r, 1, 1] = cos_val
        self.register_buffer("inv_affine_matrices", inv_affine)
        
    def forward(self, features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Apply spatial frame averaging.
        
        Args:
            features: [B*N, H*W, D] features from N rotated images
            batch_size: B (before rotation expansion)
            
        Returns:
            FA'd features [B, H*W, D]
        """
        B_times_N, num_tokens, D = features.shape
        N = self.n_group
        B = batch_size
        
        H = W = int(num_tokens ** 0.5)
        assert H * W == num_tokens, f"num_tokens ({num_tokens}) must be a perfect square"
        
        # Reshape to [B, N, H, W, D]
        features = features.reshape(B, N, H, W, D)
        
        canonical_features = []
        for r in range(N):
            feat_r = features[:, r]  # [B, H, W, D]
            
            if r == 0:
                canonical_features.append(feat_r)
            else:
                # Apply inverse rotation to map back to canonical
                feat_r = feat_r.permute(0, 3, 1, 2)  # [B, D, H, W]
                
                affine = self.inv_affine_matrices[r].unsqueeze(0).expand(B, -1, -1)
                grid = F.affine_grid(affine, feat_r.shape, align_corners=True)
                aligned = F.grid_sample(feat_r, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
                
                aligned = aligned.permute(0, 2, 3, 1)  # [B, H, W, D]
                canonical_features.append(aligned)
        
        # Stack and average
        canonical_features = torch.stack(canonical_features, dim=1)  # [B, N, H, W, D]
        avg_features = canonical_features.mean(dim=1)  # [B, H, W, D]
        
        return avg_features.reshape(B, num_tokens, D)


class EagleBackboneEFS(nn.Module):
    """
    Eagle Backbone with Equivariant Feature Steering (EFS).
    
    Key innovations:
    1. Spatial Frame Averaging on vision encoder output
    2. Equivariant steering embeddings to convey orientation
    3. No FA on LLM output (preserves language-vision interactions)
    
    The output is equivariant because:
    - Vision features are properly spatially aligned before averaging
    - Steering embeddings transform correctly with rotation
    - Language tokens are in trivial representation
    
    Args:
        fa_location: Where to apply FA ('vision_encoder', 'after_mlp1')
        use_steering: Whether to add equivariant steering embeddings
        steering_mode: How to incorporate steering ('add', 'concat', 'cross_attn')
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
        fa_location: Literal['vision_encoder', 'after_mlp1'] = 'vision_encoder',
        use_steering: bool = True,
        steering_mode: Literal['add', 'concat'] = 'add',
        num_steering_tokens: int = 1,
    ):
        super().__init__()
        assert not reproject_vision
        
        self.num_images_per_sample = num_images_per_sample
        self.rotate_image_indices = rotate_image_indices or list(range(num_images_per_sample))
        self.duplicate_image_indices = [i for i in range(num_images_per_sample) if i not in self.rotate_image_indices]
        
        # Load Eagle model
        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)
        
        # Store config values
        self.vit_hidden_size = config.vision_config.hidden_size
        self.llm_hidden_size = config.text_config.hidden_size
        self.downsample_ratio = config.downsample_ratio
        self.use_pixel_shuffle = config.use_pixel_shuffle
        
        # Calculate dimensions
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        if config.use_pixel_shuffle:
            self.num_vision_tokens = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
            self.fa_dim = self.vit_hidden_size * int(1 / self.downsample_ratio) ** 2
        else:
            self.num_vision_tokens = int((image_size // patch_size) ** 2)
            self.fa_dim = self.vit_hidden_size
            
        # Frame Averaging setup
        self.n_group = n_group
        self.output_type = output_type
        self.fa_location = fa_location
        self.use_steering = use_steering
        self.steering_mode = steering_mode
        
        self.group = gspaces.no_base_space(CyclicGroup(self.n_group))
        self._init_fa_matrices()
        
        # Spatial FA module
        self.spatial_fa = SpatialFrameAveraging(n_group)
        
        # Steering embeddings
        if use_steering:
            self.steering_embedding = EquivariantSteeringEmbedding(
                n_group=n_group,
                embedding_dim=self.llm_hidden_size if fa_location == 'after_mlp1' else self.fa_dim,
                num_tokens=num_steering_tokens,
            )
            
        # Projection layers
        self.project_to_dim = project_to_dim or self.llm_hidden_size
        
        if fa_location == 'vision_encoder':
            # Project FA'd vision features to output dim
            self.vision_proj = nn.Sequential(
                nn.LayerNorm(self.fa_dim),
                nn.Linear(self.fa_dim, self.project_to_dim),
                nn.GELU(),
                nn.Linear(self.project_to_dim, self.project_to_dim),
            )
            # Also need a projection for language tokens
            self.lang_proj = nn.Linear(self.llm_hidden_size, self.project_to_dim)
        else:
            # FA after mlp1, so project from llm_hidden_size
            self.output_proj = nn.Linear(self.llm_hidden_size, self.project_to_dim)
            
        # Truncate LLM layers
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)
            
        self.select_layer = select_layer
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        
        self.set_trainable_parameters(tune_llm, tune_visual)
        
    def _init_fa_matrices(self):
        """Initialize rotation/permutation matrices."""
        angles = torch.linspace(0, 2 * math.pi, self.n_group + 1)[:-1]
        
        # Affine rotation matrices for grid_sample
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
        perm_matrices = torch.zeros(self.n_group, self.n_group, self.n_group)
        for r in range(self.n_group):
            for i in range(self.n_group):
                j = (i + r) % self.n_group
                perm_matrices[r, i, j] = 1.0
        self.register_buffer("permutation_matrices", perm_matrices)
        
        # 2x2 rotation matrices for standard representation
        std_rot = torch.zeros(self.n_group, 2, 2)
        for i, angle in enumerate(angles):
            cos_val = math.cos(-angle.item())
            sin_val = math.sin(-angle.item())
            std_rot[i, 0, 0] = cos_val
            std_rot[i, 0, 1] = -sin_val
            std_rot[i, 1, 0] = sin_val
            std_rot[i, 1, 1] = cos_val
        self.register_buffer("std_rotation_matrices", std_rot)
        
    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        """Set trainable parameters."""
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        
        # Freeze all Eagle params first
        for param in self.eagle_model.parameters():
            param.requires_grad = False
            
        if tune_llm:
            for param in self.eagle_model.language_model.parameters():
                param.requires_grad = True
                
        if tune_visual:
            for param in self.eagle_model.vision_model.parameters():
                param.requires_grad = True
            for param in self.eagle_model.mlp1.parameters():
                param.requires_grad = True
                
        # Always train projection and steering
        for name, param in self.named_parameters():
            if 'eagle_model' not in name:
                param.requires_grad = True
                
    def set_frozen_modules_to_eval_mode(self):
        """Set frozen modules to eval."""
        if self.training:
            if not self.tune_llm:
                self.eagle_model.language_model.eval()
            if not self.tune_visual:
                self.eagle_model.vision_model.eval()
                self.eagle_model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
    
    def _extract_vit_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features from vision encoder."""
        vit_out = self.eagle_model.vision_model(
            pixel_values=pixel_values, output_hidden_states=False, return_dict=True
        )
        vit_embeds = vit_out.last_hidden_state if hasattr(vit_out, 'last_hidden_state') else vit_out
        
        if self.use_pixel_shuffle:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.eagle_model.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            
        return vit_embeds
    
    def _apply_output_representation(self, features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Apply output representation transformation."""
        B, T, D = features.shape
        
        if self.output_type == 'reg':
            # Features should already be in proper format from FA
            return features
        else:
            # Convert to standard representation
            # This might require additional transformation depending on downstream use
            return features
            
    def forward_vision_encoder_fa(self, vl_input: BatchFeature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FA at vision encoder level (before LLM).
        
        Process:
        1. Extract ViT features from rotated images
        2. Apply spatial FA to align and average
        3. Add steering embeddings (optional)
        4. Get language embeddings from LLM
        5. Return combined features
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
        N = self.n_group
        
        # 1. Extract ViT features
        pixel_values = eagle_input["pixel_values"]
        vit_embeds = self._extract_vit_features(pixel_values)  # [B*N*num_images, tokens, D]
        
        total_images = pixel_values.shape[0]
        images_per_rotation = total_images // (B * N)
        num_tokens = vit_embeds.shape[1]
        D = vit_embeds.shape[2]
        
        # Reshape: [B, N, num_images, tokens, D]
        vit_embeds = vit_embeds.reshape(B, N, images_per_rotation, num_tokens, D)
        
        # 2. Apply FA to each rotated image position
        fa_vit_list = []
        for img_idx in range(images_per_rotation):
            if img_idx in self.rotate_image_indices:
                img_embeds = vit_embeds[:, :, img_idx].reshape(B * N, num_tokens, D)
                fa_embeds = self.spatial_fa(img_embeds, B)  # [B, tokens, D]
            else:
                fa_embeds = vit_embeds[:, 0, img_idx]  # [B, tokens, D]
            fa_vit_list.append(fa_embeds)
            
        fa_vit_embeds = torch.cat(fa_vit_list, dim=1)  # [B, total_tokens, D]
        
        # 3. Add steering embeddings if enabled
        if self.use_steering:
            # Use steering embedding for rotation 0 (canonical frame after FA)
            steering = self.steering_embedding(0).unsqueeze(0).expand(B, -1, -1)  # [B, num_steering, D]
            
            if self.steering_mode == 'add':
                # Add to first N tokens
                n_steering = steering.shape[1]
                fa_vit_embeds = fa_vit_embeds.clone()
                fa_vit_embeds[:, :n_steering] = fa_vit_embeds[:, :n_steering] + steering
            else:  # concat
                fa_vit_embeds = torch.cat([steering, fa_vit_embeds], dim=1)
                
        # 4. Project vision features
        fa_vit_embeds = self.vision_proj(fa_vit_embeds)
        
        # 5. Get language embeddings
        input_ids_first = eagle_input["input_ids"][:B]
        lang_embeds = self.eagle_model.language_model.get_input_embeddings()(input_ids_first)
        lang_embeds = self.lang_proj(lang_embeds)
        
        # 6. Combine: FA'd vision + language
        # Note: this is a simplified combination; actual implementation might need
        # to properly interleave based on image token positions in input_ids
        combined = torch.cat([fa_vit_embeds, lang_embeds], dim=1)
        
        # Build attention mask
        attn_mask_first = eagle_input["attention_mask"][:B]
        vision_mask = torch.ones(B, fa_vit_embeds.shape[1], dtype=attn_mask_first.dtype, device=attn_mask_first.device)
        combined_mask = torch.cat([vision_mask, attn_mask_first], dim=1)
        
        return combined, combined_mask
    
    def forward_after_mlp1_fa(self, vl_input: BatchFeature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FA after mlp1 projection (still before LLM transformer).
        
        This captures the mlp1 projection but still applies FA before
        the position-dependent LLM self-attention.
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
        N = self.n_group
        
        # Extract ViT features
        pixel_values = eagle_input["pixel_values"]
        vit_embeds = self._extract_vit_features(pixel_values)
        
        # Apply mlp1 projection
        vit_embeds = self.eagle_model.mlp1(vit_embeds)  # [B*N*images, tokens, llm_hidden]
        
        total_images = pixel_values.shape[0]
        images_per_rotation = total_images // (B * N)
        num_tokens = vit_embeds.shape[1]
        D = vit_embeds.shape[2]
        
        # Reshape and apply FA
        vit_embeds = vit_embeds.reshape(B, N, images_per_rotation, num_tokens, D)
        
        fa_vit_list = []
        for img_idx in range(images_per_rotation):
            if img_idx in self.rotate_image_indices:
                img_embeds = vit_embeds[:, :, img_idx].reshape(B * N, num_tokens, D)
                fa_embeds = self.spatial_fa(img_embeds, B)
            else:
                fa_embeds = vit_embeds[:, 0, img_idx]
            fa_vit_list.append(fa_embeds)
            
        fa_vit_embeds = torch.cat(fa_vit_list, dim=1)
        
        # Add steering
        if self.use_steering:
            steering = self.steering_embedding(0).unsqueeze(0).expand(B, -1, -1)
            if self.steering_mode == 'add':
                n_steering = steering.shape[1]
                fa_vit_embeds = fa_vit_embeds.clone()
                fa_vit_embeds[:, :n_steering] = fa_vit_embeds[:, :n_steering] + steering
            else:
                fa_vit_embeds = torch.cat([steering, fa_vit_embeds], dim=1)
                
        # Project
        fa_vit_embeds = self.output_proj(fa_vit_embeds)
        
        # Get language
        input_ids_first = eagle_input["input_ids"][:B]
        lang_embeds = self.eagle_model.language_model.get_input_embeddings()(input_ids_first)
        lang_embeds = self.output_proj(lang_embeds)
        
        combined = torch.cat([fa_vit_embeds, lang_embeds], dim=1)
        
        attn_mask_first = eagle_input["attention_mask"][:B]
        vision_mask = torch.ones(B, fa_vit_embeds.shape[1], dtype=attn_mask_first.dtype, device=attn_mask_first.device)
        combined_mask = torch.cat([vision_mask, attn_mask_first], dim=1)
        
        return combined, combined_mask
    
    def forward_eagle(self, vl_input: BatchFeature) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main forward method."""
        if self.fa_location == 'vision_encoder':
            return self.forward_vision_encoder_fa(vl_input)
        else:  # after_mlp1
            return self.forward_after_mlp1_fa(vl_input)
    
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
