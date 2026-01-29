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
Eagle Backbone with Self-Attention Aggregation before Frame Averaging (FA-SA).

This module implements a variant of the Eagle backbone that:
1. Applies rotations to images
2. Extracts spatial tokens via Eagle vision encoder
3. Aggregates spatial tokens using Self-Attention (moved from action head)
4. Applies Frame Averaging on the aggregated (global) features

Key insight: By aggregating spatial tokens BEFORE frame averaging, we avoid
the token correspondence problem that requires TAFA. This is similar to
C8EquivariantTimmObsEncoder which pools features before FA.

Mathematical justification:
- After self-attention aggregation, we get a global feature per image
- Global features don't have spatial structure, so token index i represents
  the same "concept/channel" regardless of rotation
- Therefore, simple permutation-based FA is valid (no grid_sample needed)

Note: We use the original (non-equivariant) SelfAttentionTransformer because
the self-attention happens BEFORE frame averaging. The equivariance is achieved
through the FA step after aggregation, not during aggregation.
"""

import os
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

from escnn import gspaces
from escnn.group import CyclicGroup 
import einops

from gr00t.model.action_head.cross_attention_dit import SelfAttentionTransformer

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class SelfAttentionAggregator(nn.Module):
    """
    Self-attention aggregator that processes spatial tokens and produces
    a global representation suitable for frame averaging.
    
    Uses the original (non-equivariant) SelfAttentionTransformer because:
    1. Self-attention happens BEFORE frame averaging
    2. At this stage, features are just regular tensors
    3. Equivariance is achieved through FA AFTER aggregation
    
    This module:
    1. Takes [B*N, num_tokens, D] spatial features from rotated images
    2. Applies self-attention layers to aggregate information
    3. Outputs [B*N, num_equi_token, D] global features (via mean pooling, CLS token, or learnable queries)
    
    After aggregation, frame averaging can be applied without TAFA
    because the features no longer have spatial structure.
    """
    def __init__(
        self,
        feature_dim: int = 1536,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        dropout: float = 0.1,
        aggregation_mode: str = "mean",  # "mean", "cls", or "query"
        activation_fn: str = "gelu-approximate",
        max_num_positional_embeddings: int = 512,
        num_equi_token: int = 1,  # Number of output tokens to aggregate to
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.aggregation_mode = aggregation_mode
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_equi_token = num_equi_token
        
        # Project to inner_dim if different from feature_dim
        if feature_dim != self.inner_dim:
            self.input_proj = nn.Linear(feature_dim, self.inner_dim)
            self.output_proj = nn.Linear(self.inner_dim, feature_dim)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()
        
        # Learnable tokens for aggregation (CLS or query tokens)
        if aggregation_mode == "cls" or aggregation_mode == "query":
            self.agg_tokens = nn.Parameter(torch.randn(1, num_equi_token, self.inner_dim) * 0.02)
        
        # Use the original SelfAttentionTransformer (non-equivariant)
        # Set positional_embeddings=None to avoid dimension mismatch with variable token counts
        self.self_attention = SelfAttentionTransformer(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            output_dim=self.inner_dim,  # Not used in forward, but required
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=True,
            activation_fn=activation_fn,
            max_num_positional_embeddings=max_num_positional_embeddings,
            final_dropout=True,
            positional_embeddings=None,  # Disable positional embeddings to handle variable token counts
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(feature_dim)
        
        print(f"SelfAttentionAggregator (non-equivariant):")
        print(f"  feature_dim: {feature_dim}")
        print(f"  inner_dim: {self.inner_dim}")
        print(f"  num_layers: {num_layers}")
        print(f"  aggregation_mode: {aggregation_mode}")
        print(f"  num_equi_token: {num_equi_token}")
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B*N, num_tokens, D] spatial features from rotated images
            
        Returns:
            [B*N, num_equi_token, D] global features ready for frame averaging
        """
        B_N, num_tokens, D = features.shape
        
        # Project to inner_dim
        features = self.input_proj(features)  # [B*N, num_tokens, inner_dim]
        
        # Prepend learnable tokens if using cls or query mode
        if self.aggregation_mode == "cls" or self.aggregation_mode == "query":
            agg_tokens = self.agg_tokens.expand(B_N, -1, -1)
            features = torch.cat([agg_tokens, features], dim=1)
        
        # Apply self-attention layers
        features = self.self_attention(features)  # [B*N, num_equi_token + num_tokens, inner_dim]
        
        # Aggregate to global features
        if self.aggregation_mode == "cls" or self.aggregation_mode == "query":
            # Use the first num_equi_token tokens (the learnable aggregation tokens)
            global_features = features[:, :self.num_equi_token, :]  # [B*N, num_equi_token, inner_dim]
        else:
            # Mean pooling: divide tokens into num_equi_token groups and pool each
            if self.num_equi_token == 1:
                global_features = features.mean(dim=1, keepdim=True)  # [B*N, 1, inner_dim]
            else:
                # Split tokens into num_equi_token groups and mean pool each group
                tokens_per_group = num_tokens // self.num_equi_token
                global_features_list = []
                for i in range(self.num_equi_token):
                    start_idx = i * tokens_per_group
                    end_idx = start_idx + tokens_per_group if i < self.num_equi_token - 1 else num_tokens
                    group_features = features[:, start_idx:end_idx, :].mean(dim=1)  # [B*N, inner_dim]
                    global_features_list.append(group_features)
                global_features = torch.stack(global_features_list, dim=1)  # [B*N, num_equi_token, inner_dim]
        
        # Project back to feature_dim
        global_features = self.output_proj(global_features)  # [B*N, num_equi_token, D]
        
        # Apply final norm
        global_features = self.final_norm(global_features)
        
        return global_features


class EagleBackboneFASA(nn.Module):
    """
    Eagle Backbone with Frame Averaging and Self-Attention aggregation (FA-SA).
    
    This variant aggregates spatial tokens using self-attention BEFORE
    applying frame averaging. This avoids the token correspondence problem
    that requires TAFA, similar to C8EquivariantTimmObsEncoder.
    
    Uses the original (non-equivariant) SelfAttentionTransformer because
    self-attention happens BEFORE frame averaging. Equivariance is achieved
    through the FA step after aggregation.
    
    Pipeline:
    1. Rotate images N times (N = n_group)
    2. Process through Eagle vision encoder -> spatial tokens [B*N, T, D]
    3. Aggregate with self-attention -> global features [B*N, D]
    4. Apply frame averaging -> [B, D] equivariant features
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
        # Self-attention aggregation config (matches vl_self_attention_cfg)
        sa_num_layers: int = 4,
        sa_num_attention_heads: int = 32,
        sa_attention_head_dim: int = 64,
        sa_dropout: float = 0.2,
        sa_aggregation_mode: str = "query",
        sa_activation_fn: str = "gelu-approximate",
        sa_max_num_positional_embeddings: int = 1024,  # Not used when positional_embeddings=None
        sa_num_equi_token: int = 16,  # Number of output tokens to aggregate to
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model
            tune_visual: whether to tune the visual model
            num_images_per_sample: number of images per sample
            rotate_image_indices: indices of images to rotate (None = all)
            sa_num_layers: number of self-attention layers for aggregation
            sa_num_attention_heads: number of attention heads
            sa_attention_head_dim: dimension per attention head
            sa_dropout: dropout rate
            sa_aggregation_mode: "mean" for mean pooling, "cls" for CLS token, "query" for learnable queries
            sa_activation_fn: activation function for self-attention
            sa_max_num_positional_embeddings: max positional embeddings
            sa_num_equi_token: number of output tokens to aggregate to
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here"
        
        self.num_images_per_sample = num_images_per_sample
        if rotate_image_indices is None:
            self.rotate_image_indices = list(range(num_images_per_sample))
        else:
            self.rotate_image_indices = rotate_image_indices
        self.duplicate_image_indices = [
            i for i in range(num_images_per_sample) 
            if i not in self.rotate_image_indices
        ]

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)
        
        # Frame Average setup
        self.n_group = 4  # hardcode
        self.group = gspaces.no_base_space(CyclicGroup(self.n_group))
        self.init_rotation_matrices()
        
        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()
        
        self.project_to_dim = project_to_dim if project_to_dim else 2048
        self.num_equi_token = sa_num_equi_token
        
        # Self-attention aggregator (using original non-equivariant transformer)
        self.sa_aggregator = SelfAttentionAggregator(
            feature_dim=self.project_to_dim,
            num_layers=sa_num_layers,
            num_attention_heads=sa_num_attention_heads,
            attention_head_dim=sa_attention_head_dim,
            dropout=sa_dropout,
            aggregation_mode=sa_aggregation_mode,
            activation_fn=sa_activation_fn,
            max_num_positional_embeddings=sa_max_num_positional_embeddings,
            num_equi_token=sa_num_equi_token,
        )

        # Remove unused LLM layers
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)
        
    def init_rotation_matrices(self):
        """Initialize rotation and permutation matrices for frame averaging."""
        angles = torch.linspace(0, 2 * math.pi, self.n_group + 1)[:-1]
        self.rotation_matrices = torch.zeros(self.n_group, 2, 3)

        for i, angle in enumerate(angles):
            cos_val = math.cos(-angle.item())
            sin_val = math.sin(-angle.item())
            
            self.rotation_matrices[i, 0, 0] = cos_val
            self.rotation_matrices[i, 0, 1] = -sin_val
            self.rotation_matrices[i, 1, 0] = sin_val
            self.rotation_matrices[i, 1, 1] = cos_val
            
        self.register_buffer("rotation_matrices_buffer", self.rotation_matrices)
        self.group_elements = list(self.group.testing_elements)
        
        # Permutation matrices for frame averaging
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
        return BatchFeature(data=batch)
    
    def rotate_rgb_batch(self, vl_batch):
        """
        Apply all N rotations to a batch of images efficiently.
        Handles multiple images per sample, rotating only specified indices.
        """
        input_ids_batch = vl_batch["input_ids"]
        attn_mask_batch = vl_batch["attention_mask"]
        img_batch = vl_batch["pixel_values"]
        
        total_imgs, C, H, W = img_batch.shape
        B = total_imgs // self.num_images_per_sample
        
        img_batch_reshaped = img_batch.reshape(B, self.num_images_per_sample, C, H, W)
        
        processed_images = []
        
        for img_idx in range(self.num_images_per_sample):
            imgs_at_idx = img_batch_reshaped[:, img_idx, :, :, :]
            
            if img_idx in self.rotate_image_indices:
                rotated = self._apply_rotations(imgs_at_idx)
                processed_images.append(rotated)
            else:
                expanded = imgs_at_idx.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)
                duplicated = expanded.reshape(B * self.n_group, C, H, W)
                processed_images.append(duplicated)
        
        processed_images_reshaped = [
            img.reshape(B, self.n_group, C, H, W) for img in processed_images
        ]
        stacked = torch.stack(processed_images_reshaped, dim=0)
        stacked = stacked.permute(1, 2, 0, 3, 4, 5)
        rotated_imgs = stacked.reshape(
            B * self.n_group * self.num_images_per_sample, C, H, W
        )
        
        input_ids_batch_expanded = input_ids_batch.repeat((self.n_group, 1))
        attn_mask_batch_expanded = attn_mask_batch.repeat((self.n_group, 1))
        
        vl_batch["pixel_values"] = rotated_imgs
        vl_batch["input_ids"] = input_ids_batch_expanded
        vl_batch["attention_mask"] = attn_mask_batch_expanded
        return vl_batch
    
    def _apply_rotations(self, img_batch):
        """Apply all N rotations to a batch of images."""
        B, C, H, W = img_batch.shape
        
        expanded = img_batch.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1)
        img_batch_expanded = expanded.reshape(B * self.n_group, C, H, W)
        
        rotation_indices = torch.arange(self.n_group, device=img_batch.device).repeat(B)
        rotation_matrices = self.rotation_matrices_buffer[rotation_indices]
        
        grid = torch.nn.functional.affine_grid(
            rotation_matrices, 
            size=(B * self.n_group, C, H, W),
            align_corners=True
        )
        
        rotated_imgs = torch.nn.functional.grid_sample(
            img_batch_expanded, 
            grid, 
            align_corners=True,
            padding_mode='zeros'
        )
        
        return rotated_imgs

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through Eagle model with self-attention aggregation and frame averaging.
        
        Pipeline:
        1. Eagle encoder -> [B*N, T, D] spatial tokens
        2. Self-attention aggregation -> [B*N, num_equi_token, D] global features
        3. Frame averaging -> [B, num_equi_token, D] equivariant features
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
        
        # Forward through Eagle model
        eagle_output = self.eagle_model(
            **eagle_input, output_hidden_states=True, return_dict=True
        )
        eagle_features = eagle_output.hidden_states[self.select_layer]
        eagle_features = self.eagle_linear(eagle_features)  # [B*N, T, D]
        
        # Self-attention aggregation: [B*N, T, D] -> [B*N, num_equi_token, D]
        global_features = self.sa_aggregator(eagle_features)  # [B*N, num_equi_token, D]
        
        # Reshape for frame averaging: [B*N, num_equi_token, D] -> [B, N, num_equi_token * D]
        global_features = global_features.reshape(B, self.n_group, -1)
        
        # Apply frame averaging: [B, N, num_equi_token * D] -> [B, num_equi_token * D]
        avg_features = self._apply_frame_averaging(global_features, B)
        
        # Reshape back to [B, num_equi_token, D] for compatibility
        avg_features = avg_features.reshape(B, self.num_equi_token, self.project_to_dim)
        
        # Get attention mask from first rotation
        original_attention_mask = eagle_input["attention_mask"][:B]
        # Create attention mask with num_equi_token tokens
        token_mask = torch.ones(B, self.num_equi_token, device=original_attention_mask.device)

        return avg_features, token_mask

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

    def _apply_frame_averaging(self, features, batch_size):
        """
        Apply frame averaging to global features using permutation matrices.
        
        Args:
            features: [B, N, D] where N is n_group, D is feature_dim
            batch_size: B
            
        Returns:
            [B, D] tensor with frame averaging applied (regular representation)
        """
        feature_dim = features.shape[2]
        blocks = feature_dim // self.n_group
        
        # Reshape: [B, N, blocks, N]
        features = features.reshape(batch_size, self.n_group, blocks, self.n_group)
        
        # Flatten: [B*N, blocks, N]
        all_features_flat = features.reshape(batch_size * self.n_group, blocks, self.n_group)
        
        # Get permutation matrices: [B*N, N, N]
        selected_perm_matrices = self.selected_perm_matrices_template.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )
        selected_perm_matrices = selected_perm_matrices.reshape(
            batch_size * self.n_group, self.n_group, self.n_group
        )
        
        # Apply permutation: [B*N, blocks, N] @ [B*N, N, N] -> [B*N, blocks, N]
        aligned_features_flat = torch.bmm(all_features_flat, selected_perm_matrices)
        
        # Reshape: [B, N, blocks, N]
        aligned_features = aligned_features_flat.reshape(
            batch_size, self.n_group, blocks, self.n_group
        )
        
        # Average over group dimension: [B, blocks, N]
        avg_features = torch.mean(aligned_features, dim=1)
        
        return avg_features.reshape(batch_size, blocks * self.n_group)
