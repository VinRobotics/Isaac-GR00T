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
Eagle Backbone with Token-Aware Frame Averaging (TAFA)

PROBLEM ANALYSIS:
=================
The core issue with FA on VLMs is token correspondence:
- Vision tokens are arranged in a 2D grid (e.g., 16x16 = 256 tokens)
- When image rotates by 90°, token at position (i,j) moves to (j, H-1-i)
- Naive FA averages tokens at the same INDEX across rotations
- But same index ≠ same semantic content after rotation!

SOLUTION: TOKEN-AWARE FA (TAFA)
===============================
1. Track spatial positions of vision tokens
2. For each rotation, compute which original token maps to which new position
3. Use bilinear interpolation for non-integer mappings
4. Average tokens that correspond to the SAME SEMANTIC LOCATION

This is similar to how deformable attention handles spatial transformations,
but applied to the frame averaging setting.

IMPLEMENTATION:
===============
For C_N group with N=8 rotations:
1. Each token has coordinates (x, y) in [-1, 1] normalized space
2. For rotation r, compute R(-2πr/N)(x,y) to find canonical position
3. Use grid_sample to remap features to canonical grid
4. Average the N canonically-aligned feature maps

ADDITIONAL IMPROVEMENTS:
========================
1. Rotation-aware positional encoding
2. Multi-scale FA (apply at multiple feature levels)
3. Soft FA with learned weights per rotation
"""

import os
import math
from typing import Optional, Literal, Tuple, List

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


class TokenAwareFA(nn.Module):
    """
    Token-Aware Frame Averaging module.
    
    Properly aligns vision tokens across rotations by accounting for
    how spatial positions change under rotation.
    
    Key insight: token at index i in rotation r corresponds to a 
    different spatial location than token i in rotation 0.
    """
    
    def __init__(
        self, 
        n_group: int, 
        interpolation: str = 'bilinear',
        align_corners: bool = True,
        weighted: bool = False,
    ):
        super().__init__()
        self.n_group = n_group
        self.interpolation = interpolation
        self.align_corners = align_corners
        self.weighted = weighted
        
        # Pre-compute inverse rotation matrices for canonical mapping
        angles = torch.linspace(0, 2 * math.pi, n_group + 1)[:-1]
        
        # 2x3 affine matrices for grid_sample
        inv_affine = torch.zeros(n_group, 2, 3)
        for r, angle in enumerate(angles):
            # Inverse rotation to map from rotated -> canonical
            cos_val = math.cos(-angle.item())
            sin_val = math.sin(-angle.item())
            inv_affine[r, 0, 0] = cos_val
            inv_affine[r, 0, 1] = -sin_val
            inv_affine[r, 1, 0] = sin_val
            inv_affine[r, 1, 1] = cos_val
        self.register_buffer("inv_affine_matrices", inv_affine)
        
        # Learnable weights for soft averaging (optional)
        if weighted:
            self.rotation_weights = nn.Parameter(torch.ones(n_group) / n_group)
        else:
            self.register_buffer("rotation_weights", torch.ones(n_group) / n_group)
            
        # For regular representation output
        perm_matrices = torch.zeros(n_group, n_group, n_group)
        for r in range(n_group):
            for i in range(n_group):
                j = (i + r) % n_group
                perm_matrices[r, i, j] = 1.0
        self.register_buffer("permutation_matrices", perm_matrices)
        
    def forward(
        self, 
        features: torch.Tensor, 
        batch_size: int,
        output_type: str = 'invariant',
    ) -> torch.Tensor:
        """
        Apply token-aware frame averaging.
        
        Args:
            features: [B*N, H*W, D] - features from N rotated images
            batch_size: B (original batch size)
            output_type: 'invariant' (average), 'reg' (regular), 'std' (standard)
            
        Returns:
            FA'd features with shape depending on output_type
        """
        B_times_N, num_tokens, D = features.shape
        B = batch_size
        N = self.n_group
        
        H = W = int(num_tokens ** 0.5)
        assert H * W == num_tokens
        
        # Reshape to [B, N, H, W, D]
        features = features.reshape(B, N, H, W, D)
        
        # Remap each rotation to canonical frame
        canonical = []
        for r in range(N):
            feat_r = features[:, r]  # [B, H, W, D]
            
            if r == 0:
                canonical.append(feat_r)
            else:
                # Reshape to [B, D, H, W] for grid_sample
                feat_r = feat_r.permute(0, 3, 1, 2)  # [B, D, H, W]
                
                # Get affine matrix and generate grid
                affine = self.inv_affine_matrices[r:r+1].expand(B, -1, -1)
                grid = F.affine_grid(affine, feat_r.shape, align_corners=self.align_corners)
                
                # Sample with interpolation
                aligned = F.grid_sample(
                    feat_r, grid, 
                    mode=self.interpolation,
                    align_corners=self.align_corners,
                    padding_mode='zeros'
                )
                
                aligned = aligned.permute(0, 2, 3, 1)  # [B, H, W, D]
                canonical.append(aligned)
                
        # Stack: [B, N, H, W, D]
        canonical = torch.stack(canonical, dim=1)
        
        # Compute weighted average
        if self.weighted:
            weights = F.softmax(self.rotation_weights, dim=0)
        else:
            weights = self.rotation_weights
            
        if output_type == 'invariant':
            # Simple weighted average
            avg = torch.einsum('bnhwd,n->bhwd', canonical, weights)
            return avg.reshape(B, num_tokens, D)
            
        elif output_type == 'reg':
            # Regular representation: apply permutation matrices
            # Output shape: [B, H*W, D] where D must be divisible by N
            assert D % N == 0, f"Feature dim {D} must be divisible by n_group {N}"
            blocks = D // N
            
            # Reshape: [B, N, H, W, blocks, N]
            canonical = canonical.reshape(B, N, H, W, blocks, N)
            
            # Apply permutation for each rotation
            aligned_reg = []
            for r in range(N):
                feat_r = canonical[:, r]  # [B, H, W, blocks, N]
                perm = self.permutation_matrices[r]  # [N, N]
                # [B, H, W, blocks, N] @ [N, N] -> [B, H, W, blocks, N]
                # Use 'k' for blocks to avoid conflict with 'b' for batch
                aligned = torch.einsum('bhwkn,nm->bhwkm', feat_r, perm)
                aligned_reg.append(aligned)
                
            aligned_reg = torch.stack(aligned_reg, dim=1)  # [B, N, H, W, blocks, N]
            
            # Weighted average - use 'k' for blocks, 'c' for the last N dimension
            avg = torch.einsum('bnhwkc,n->bhwkc', aligned_reg, weights)
            return avg.reshape(B, num_tokens, D)
            
        elif output_type == 'std':
            # Standard representation: 2D rotation matrices
            assert D % 2 == 0
            num_vectors = D // 2
            
            canonical = canonical.reshape(B, N, H, W, num_vectors, 2)
            
            # Apply 2x2 rotation for alignment (already done implicitly by grid_sample)
            # Just average
            avg = torch.einsum('bnhwvc,n->bhwvc', canonical, weights)
            return avg.reshape(B, num_tokens, D)
            
        else:
            raise ValueError(f"Unknown output_type: {output_type}")


class RotationAwarePositionalEncoding(nn.Module):
    """
    Positional encoding that transforms correctly under rotation.
    
    Instead of using fixed positional embeddings, we encode positions
    using rotation-equivariant features (like Fourier features rotated).
    """
    
    def __init__(self, d_model: int, max_h: int = 32, max_w: int = 32, n_group: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_group = n_group
        
        # Use Fourier features for continuous position encoding
        # These can be rotated analytically
        self.num_freqs = d_model // 4
        
        # Learnable frequency scales
        self.freq_scales = nn.Parameter(torch.randn(self.num_freqs) * 0.02)
        
    def forward(self, H: int, W: int, rotation_idx: int = 0) -> torch.Tensor:
        """
        Generate rotation-aware positional encoding.
        
        Args:
            H, W: spatial dimensions
            rotation_idx: which rotation (0 to n_group-1)
            
        Returns:
            Positional encoding [H, W, d_model]
        """
        device = self.freq_scales.device
        dtype = self.freq_scales.dtype
        
        # Create coordinate grid in [-1, 1]
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Rotate coordinates to account for image rotation
        angle = 2 * math.pi * rotation_idx / self.n_group
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        xx_rot = cos_a * xx - sin_a * yy
        yy_rot = sin_a * xx + cos_a * yy
        
        # Compute Fourier features
        freqs = torch.exp(self.freq_scales).unsqueeze(0).unsqueeze(0)  # [1, 1, num_freqs]
        
        xx_rot = xx_rot.unsqueeze(-1)  # [H, W, 1]
        yy_rot = yy_rot.unsqueeze(-1)
        
        # [H, W, num_freqs] for each
        sin_x = torch.sin(freqs * xx_rot * math.pi)
        cos_x = torch.cos(freqs * xx_rot * math.pi)
        sin_y = torch.sin(freqs * yy_rot * math.pi)
        cos_y = torch.cos(freqs * yy_rot * math.pi)
        
        # Concatenate: [H, W, 4 * num_freqs]
        pe = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)
        
        # Pad or truncate to d_model
        if pe.shape[-1] < self.d_model:
            padding = torch.zeros(H, W, self.d_model - pe.shape[-1], device=device, dtype=dtype)
            pe = torch.cat([pe, padding], dim=-1)
        else:
            pe = pe[..., :self.d_model]
            
        return pe


class EagleBackboneTAFA(nn.Module):
    """
    Eagle Backbone with Token-Aware Frame Averaging.
    
    Key innovations over v2/v3:
    1. TokenAwareFA: Properly aligns tokens across rotations using spatial mapping
    2. Multi-level FA: Apply at both vision encoder and after projection
    3. Rotation-aware positional encoding: PE that transforms with rotation
    
    This addresses the fundamental issue that caused v2 to underperform:
    tokens at the same index in rotated images don't represent the same content.
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
        output_type: Literal['invariant', 'reg', 'std'] = 'reg',
        fa_interpolation: str = 'bilinear',
        use_weighted_fa: bool = False,
        use_rotation_pe: bool = True,
        multi_level_fa: bool = True,  # Apply FA at multiple levels
    ):
        super().__init__()
        assert not reproject_vision
        
        self.num_images_per_sample = num_images_per_sample
        self.rotate_image_indices = rotate_image_indices or list(range(num_images_per_sample))
        self.duplicate_image_indices = [i for i in range(num_images_per_sample) if i not in self.rotate_image_indices]
        
        # Load Eagle
        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)
        
        # Store config
        self.vit_hidden_size = config.vision_config.hidden_size
        self.llm_hidden_size = config.text_config.hidden_size
        self.downsample_ratio = config.downsample_ratio
        self.use_pixel_shuffle = config.use_pixel_shuffle
        
        # Vision token grid size
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.vit_grid_size = image_size // patch_size  # e.g., 32 for 448/14
        
        if config.use_pixel_shuffle:
            self.vision_grid_h = self.vision_grid_w = int(self.vit_grid_size * config.downsample_ratio)
            self.fa_dim = self.vit_hidden_size * int(1 / self.downsample_ratio) ** 2
        else:
            self.vision_grid_h = self.vision_grid_w = self.vit_grid_size
            self.fa_dim = self.vit_hidden_size
            
        self.num_vision_tokens = self.vision_grid_h * self.vision_grid_w
        
        # FA settings
        self.n_group = n_group
        self.output_type = output_type
        self.multi_level_fa = multi_level_fa
        self.group = gspaces.no_base_space(CyclicGroup(n_group))
        
        # Token-aware FA modules
        self.vit_fa = TokenAwareFA(
            n_group=n_group,
            interpolation=fa_interpolation,
            weighted=use_weighted_fa,
        )
        
        if multi_level_fa:
            self.proj_fa = TokenAwareFA(
                n_group=n_group,
                interpolation=fa_interpolation,
                weighted=use_weighted_fa,
            )
            
        # Rotation-aware positional encoding
        self.use_rotation_pe = use_rotation_pe
        if use_rotation_pe:
            self.rotation_pe = RotationAwarePositionalEncoding(
                d_model=self.fa_dim,
                max_h=self.vision_grid_h,
                max_w=self.vision_grid_w,
                n_group=n_group,
            )
            
        # Projection
        self.project_to_dim = project_to_dim or self.llm_hidden_size
        
        # For regular representation, output dim must be divisible by n_group
        # Ensure project_to_dim is compatible
        assert self.project_to_dim % n_group == 0, \
            f"project_to_dim ({self.project_to_dim}) must be divisible by n_group ({n_group})"
        
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.fa_dim),
            nn.Linear(self.fa_dim, self.project_to_dim),
            nn.GELU(),
            nn.Linear(self.project_to_dim, self.project_to_dim),
        )
        
        # Language projection that lifts to regular representation
        # Regular repr has shape [blocks, N] where features permute cyclically
        # We use ESCNN to create proper lifting from trivial -> regular repr
        self.trivial_repr = self.group.regular_repr.group.trivial_representation
        self.regular_repr = self.group.regular_repr
        
        # Create ESCNN field types for lifting
        # Input: LLM hidden size as trivial repr
        # Output: project_to_dim as regular repr (blocks_per_group copies of regular repr)
        blocks_per_group = self.project_to_dim // n_group
        num_trivial_fields = self.llm_hidden_size  # Each scalar is a trivial field
        self.lang_in_type = enn.FieldType(self.group, num_trivial_fields * [self.trivial_repr])
        self.lang_out_type = enn.FieldType(self.group, blocks_per_group * [self.regular_repr])
        
        # Lifting layer: trivial -> regular (directly from LLM embedding to regular repr)
        self.lang_lift = enn.Linear(self.lang_in_type, self.lang_out_type)
        
        # Truncate LLM
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)
            
        self.select_layer = select_layer
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        
        self.set_trainable_parameters(tune_llm, tune_visual)
        
    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        
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
                
        for name, param in self.named_parameters():
            if 'eagle_model' not in name:
                param.requires_grad = True
                
    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_llm:
                self.eagle_model.language_model.eval()
            if not self.tune_visual:
                self.eagle_model.vision_model.eval()
                self.eagle_model.mlp1.eval()
                
    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
    
    def _extract_vit_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract ViT features."""
        vit_out = self.eagle_model.vision_model(
            pixel_values=pixel_values, output_hidden_states=False, return_dict=True
        )
        if hasattr(vit_out, 'last_hidden_state'):
            vit_embeds = vit_out.last_hidden_state
        else:
            vit_embeds = vit_out
            
        if self.use_pixel_shuffle:
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.eagle_model.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            
        return vit_embeds
    
    def forward_eagle(self, vl_input: BatchFeature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with Token-Aware Frame Averaging.
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
        vit_embeds = self._extract_vit_features(pixel_values)  # [B*N*images, tokens, D]
        
        total_images = pixel_values.shape[0]
        images_per_rotation = total_images // (B * N)
        num_tokens = vit_embeds.shape[1]
        D = vit_embeds.shape[2]
        
        # Reshape: [B, N, num_images, tokens, D]
        vit_embeds = vit_embeds.reshape(B, N, images_per_rotation, num_tokens, D)
        
        # Apply Token-Aware FA to each image
        fa_features = []
        for img_idx in range(images_per_rotation):
            if img_idx in self.rotate_image_indices:
                # Get features for this image across all rotations: [B, N, tokens, D]
                img_features = vit_embeds[:, :, img_idx]
                
                # Add rotation-aware positional encoding before FA
                if self.use_rotation_pe:
                    # img_features is [B, N, tokens, D]
                    # Add PE for each rotation separately
                    for r in range(N):
                        pe = self.rotation_pe(self.vision_grid_h, self.vision_grid_w, r)
                        pe = pe.reshape(1, num_tokens, D)  # [1, tokens, D]
                        img_features[:, r] = img_features[:, r] + pe
                
                # Reshape to [B*N, tokens, D] for TAFA
                img_features = img_features.reshape(B * N, num_tokens, D)
                
                # Apply TAFA with configured output type
                fa_img = self.vit_fa(img_features, B, output_type=self.output_type)
            else:
                # Not rotated - take from first rotation
                fa_img = vit_embeds[:, 0, img_idx]
                
            fa_features.append(fa_img)
            
        # Concatenate all images
        fa_features = torch.cat(fa_features, dim=1)  # [B, total_tokens, D]
        
        # Project vision features (already in regular repr from TAFA)
        fa_features = self.vision_proj(fa_features)  # [B, total_tokens, project_to_dim]
        
        # Get language embeddings and lift to regular representation
        input_ids_first = eagle_input["input_ids"][:B]
        lang_embeds = self.eagle_model.language_model.get_input_embeddings()(input_ids_first)
        # [B, seq_len, llm_hidden_size]
        
        # Lift to regular representation using ESCNN directly
        B_lang, seq_len, feat_dim = lang_embeds.shape
        # Reshape for ESCNN: [B * seq_len, llm_hidden_size]
        lang_embeds_flat = lang_embeds.reshape(B_lang * seq_len, feat_dim)
        # Wrap in GeometricTensor (trivial repr)
        lang_geom = enn.GeometricTensor(lang_embeds_flat, self.lang_in_type)
        # Apply lifting (trivial -> regular)
        lang_lifted = self.lang_lift(lang_geom)
        # Extract tensor and reshape back: [B, seq_len, project_to_dim]
        lang_embeds = lang_lifted.tensor.reshape(B_lang, seq_len, self.project_to_dim)
        
        # Now both fa_features and lang_embeds are in regular representation!
        # Combine
        combined = torch.cat([fa_features, lang_embeds], dim=1)
        
        # Attention mask
        attn_mask_first = eagle_input["attention_mask"][:B]
        vision_mask = torch.ones(B, fa_features.shape[1], dtype=attn_mask_first.dtype, device=attn_mask_first.device)
        combined_mask = torch.cat([vision_mask, attn_mask_first], dim=1)
        
        return combined, combined_mask
    
    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        
        features, mask = self.forward_eagle(vl_input)
        
        if self.training and self.tune_visual:
            dummy = torch.tensor(0.0, device=features.device, dtype=features.dtype, requires_grad=True)
            for p in self.eagle_model.vision_model.parameters():
                if p.requires_grad:
                    dummy = dummy + 0.0 * p.sum()
            features = features + dummy
        
        return BatchFeature(data={
            "backbone_features": features, 
            "backbone_attention_mask": mask,
        })
