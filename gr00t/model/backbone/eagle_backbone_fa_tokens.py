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
Eagle Backbone with Late Frame Averaging on Full Vision Tokens.

This version preserves all vision transformer tokens instead of using pooled features,
giving vision features equal importance to state and action tokens.
Frame averaging is applied on the full token sequence with proper spatial alignment.

Key differences from EagleBackboneFA (pooled):
- Output: [B, num_imgs, T_vision, D_vision] instead of [B, num_imgs, D_pool]
- Preserves spatial/token information (256 tokens vs 1 pooled vector)
- Better balance with state/action tokens in downstream processing
- Each token is a regular representation of the CN group

Frame Averaging Formula:
    FA(x) = (1/|G|) * Σ_g ρ(g) · π(g⁻¹) · f(g·x)
    
Where:
- g is a rotation from the cyclic group CN
- f(g·x) = features from rotated image (tokens at rotated positions)
- π(g⁻¹) = inverse spatial permutation (revert tokens to original positions)  
- ρ(g) = feature-space transformation (regular representation)

This ensures proper per-token equivariance: f(g·x)[p] = ρ(g) · f(x)[π(g)·p]
After spatial alignment, tokens across all rotations represent the same spatial content.
"""

import os
import math
from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackboneFATokens(nn.Module):
    """
    Eagle Backbone with Late Frame Averaging on Full Vision Tokens.
    
    Unlike EagleBackboneFA which uses pooled vision features, this version
    preserves all vision transformer tokens to maintain richer visual information.
    Frame averaging is applied on the full sequence of vision tokens.
    
    Pipeline:
    1. Input: [B, n_img, C, H, W] images + text
    2. Rotate each image N times -> [B*N, n_img, C, H, W]
    3. Extract vision features -> [B*N*n_img, T_vision, D_vision]
    4. Apply Frame Averaging on vision tokens -> [B*n_img, T_vision, D_vision]
    5. Extract language features -> [B, T_text, D_text]
    
    Output provides full vision token sequences for richer visual representation.
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

        # Projection layer for vision features
        # Vision features from Eagle are typically 2048 dim after mlp1
        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # Remove unused LLM layers (only if layers exist and select_layer is positive)
        # select_layer=-1 means keep all layers, select_layer=N means keep first N layers
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        
        # Initialize rotation and frame averaging components
        self._init_rotation_matrices()
        self._init_permutation_matrices()
        # Token grid size: 16x16 = 256 tokens (typical for Eagle after pixel shuffle)
        self._init_token_permutation_indices(grid_size=16)
        
        self.set_trainable_parameters(tune_llm, tune_visual)
        
        print(f"EagleBackboneFATokens initialized:")
        print(f"  n_group (CN): {self.n_group}")
        print(f"  project_to_dim: {self.project_to_dim}")
        print(f"  rotate_image_indices: {self.rotate_image_indices}")
        print(f"  output_type: {self.output_type}")
        print(f"  Using FULL vision tokens (not pooled)")
        print(f"  Token grid size: {self.token_grid_size}x{self.token_grid_size}")

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
    
    def _init_token_permutation_indices(self, grid_size: int = 16):
        """
        Initialize token permutation indices for different rotations.
        
        Vision tokens are arranged in a grid (e.g., 16x16 = 256 tokens).
        When the image is rotated, the token positions must be permuted accordingly.
        
        For 90° CCW rotation: position (i,j) gets content from (j, N-1-i)
        We compute permutation for each rotation amount.
        
        perm[r][idx] = the source index in unrotated features that ends up at idx
                       after rotating by r steps
        
        Args:
            grid_size: size of the token grid (sqrt of num_tokens)
        """
        self.token_grid_size = grid_size
        num_tokens = grid_size * grid_size
        N = grid_size
        
        # Create permutation indices for each rotation
        # For C4/C8, each rotation is by 360/n_group degrees CCW
        token_perm_indices = torch.zeros(self.n_group, num_tokens, dtype=torch.long)
        
        # Helper function for single 90° CCW rotation
        # After 90° CCW: position (i,j) contains content from (j, N-1-i)
        def rotate_90_ccw_source(i, j, N):
            return j, N - 1 - i
        
        for r in range(self.n_group):
            # Number of 90° rotations (for C4: r=0,1,2,3 means 0,1,2,3 times 90°)
            # For C8: r=0,1,2,3,4,5,6,7 means 0,45,90,... degrees
            # We handle C4 specially for exact permutation
            
            if self.n_group == 4:
                # C4: each step is 90°
                num_90_rotations = r
            elif self.n_group == 8:
                # C8: only r=0,2,4,6 give exact 90° multiples
                # For r=1,3,5,7 (45° steps), we approximate
                num_90_rotations = r // 2  # 0,0,1,1,2,2,3,3
            else:
                # General case using rotation formula
                num_90_rotations = 0
            
            for i in range(N):
                for j in range(N):
                    orig_idx = i * N + j
                    
                    # Find source position after num_90_rotations
                    src_i, src_j = i, j
                    for _ in range(num_90_rotations):
                        src_i, src_j = rotate_90_ccw_source(src_i, src_j, N)
                    
                    # Handle non-90° rotations (for C8 odd indices)
                    if self.n_group == 8 and r % 2 == 1:
                        # For 45° offsets, use approximate rotation
                        # This won't be exact, but we use it for FA
                        angle = r * 2 * math.pi / self.n_group
                        ci = src_i - (N - 1) / 2
                        cj = src_j - (N - 1) / 2
                        cos_t = math.cos(-angle + num_90_rotations * math.pi / 2)
                        sin_t = math.sin(-angle + num_90_rotations * math.pi / 2)
                        new_ci = cos_t * ci - sin_t * cj
                        new_cj = sin_t * ci + cos_t * cj
                        src_i = int(round(new_ci + (N - 1) / 2))
                        src_j = int(round(new_cj + (N - 1) / 2))
                        src_i = max(0, min(N - 1, src_i))
                        src_j = max(0, min(N - 1, src_j))
                    
                    src_idx = src_i * N + src_j
                    token_perm_indices[r, orig_idx] = src_idx
        
        self.register_buffer("token_perm_indices", token_perm_indices)

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

    def rotate_vl_batch(self, vl_input: dict) -> torch.Tensor:
        """
        Rotate images and expand text inputs for N rotations.
        
        Args:
            vl_input: dict with eagle_ prefixed keys
            
        Returns:
            Rotated pixel values tensor
        """
        # Extract eagle inputs
        eagle_prefix = "eagle_"
        input_ids = vl_input.get(f"{eagle_prefix}input_ids")
        pixel_values = vl_input.get(f"{eagle_prefix}pixel_values")
        
        if pixel_values is None:
            return None
            
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
            stacked = stacked.permute(1, 0, 2, 3, 4, 5)  # [B, num_img, N, C, H, W]
            rotated_pixel_values = stacked.reshape(
                B * self.n_group * self.num_images_per_sample, C, H, W
            )
        else:
            # Single image per sample
            rotated_pixel_values = self._apply_rotations_to_images(pixel_values)
        return rotated_pixel_values

    def _apply_frame_averaging(
        self, 
        features: torch.Tensor, 
        batch_size: int
    ) -> torch.Tensor:
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
        selected_perm_matrices = self.selected_perm_matrices_template.repeat(batch_size, 1, 1).to(features.dtype)
        aligned_features_flat = torch.bmm(all_features_flat, selected_perm_matrices)
        aligned_features = aligned_features_flat.reshape(batch_size, self.n_group, blocks, self.n_group)
        avg_features = torch.mean(aligned_features, dim=1)  # [B, blocks, N]
        return avg_features.reshape(batch_size, blocks * self.n_group)

    def _apply_frame_averaging_tokens(
        self, 
        features: torch.Tensor, 
        batch_size: int,
        num_tokens: int
    ) -> torch.Tensor:
        """
        Apply frame averaging to a sequence of tokens.
        
        Each token is treated independently for frame averaging.
        
        Args:
            features: Tensor of shape [B*N, T, D] where:
                      - B is original batch size
                      - N is number of rotations
                      - T is number of tokens
                      - D is feature dimension (must be divisible by N for reg repr)
            batch_size: Original batch size (B)
            num_tokens: Number of tokens (T)
            
        Returns:
            Tensor of shape [B, T, D] with frame averaging applied per token
        """
        # Reshape to [B, N, T, D]
        D = features.shape[-1]
        features_reshaped = features.reshape(batch_size, self.n_group, num_tokens, D)
        
        # Permute to process each token independently: [B, T, N, D]
        features_permuted = features_reshaped.permute(0, 2, 1, 3)
        
        # Flatten B and T for batch processing: [B*T, N, D]
        features_flat = features_permuted.reshape(batch_size * num_tokens, self.n_group, D)
        
        # Apply frame averaging to each token
        # Regular representation - split D into blocks of size N
        blocks = D // self.n_group
        features_blocks = features_flat.reshape(batch_size * num_tokens, self.n_group, blocks, self.n_group)
        
        # Flatten for batch matrix multiplication
        all_features_flat = features_blocks.reshape(-1, blocks, self.n_group)
        
        # Get permutation matrices
        selected_perm_matrices = self.selected_perm_matrices_template.repeat(
            batch_size * num_tokens, 1, 1
        ).to(features.dtype)
        
        # Apply permutation
        aligned_features_flat = torch.bmm(all_features_flat, selected_perm_matrices)
        
        # Reshape back
        aligned_features = aligned_features_flat.reshape(
            batch_size * num_tokens, self.n_group, blocks, self.n_group
        )
        
        # Average over rotations
        avg_features = torch.mean(aligned_features, dim=1)  # [B*T, blocks, N]
        avg_features = avg_features.reshape(batch_size * num_tokens, blocks * self.n_group)
        
        # Reshape to [B, T, D]
        return avg_features.reshape(batch_size, num_tokens, D)

    def _permute_tokens_for_rotation(
        self, 
        tokens: torch.Tensor, 
        rotation_idx: int,
        inverse: bool = False
    ) -> torch.Tensor:
        """
        Permute token positions according to rotation.
        
        When image is rotated by rotation_idx, token at spatial position (i,j)
        moves to a new position. This function permutes tokens to align them.
        
        Args:
            tokens: [B, T, D] tensor of tokens
            rotation_idx: index of rotation (0 to n_group-1)
            inverse: if True, apply inverse permutation
            
        Returns:
            Permuted tokens [B, T, D]
        """
        if rotation_idx == 0:
            return tokens
            
        B, T, D = tokens.shape
        
        # Get permutation indices for this rotation
        if inverse:
            # For inverse, we need the inverse permutation
            # Find where each token came from
            perm = self.token_perm_indices[(self.n_group - rotation_idx) % self.n_group]
        else:
            perm = self.token_perm_indices[rotation_idx]
        
        # Apply permutation
        # perm[i] tells us which token should go to position i
        permuted = tokens[:, perm, :]
        
        return permuted

    def forward_eagle(self, vl_input: BatchFeature) -> tuple:
        """
        Forward through Eagle model with frame averaging on full vision tokens.
        
        Frame Averaging for COVARIANT equivariance:
        
        FA(x) = (1/|G|) * Σ_h ρ(h⁻¹) · f(h·x)
        
        Using ρ(h⁻¹) instead of ρ(h) gives the covariant property:
        FA(g·x) = ρ(g) · FA(x)
        
        Where:
        - h·x = image rotated by h
        - f(h·x) = features from rotated image  
        - ρ(h⁻¹) = inverse representation = spatial_perm(h⁻¹) ⊗ feature_perm(h⁻¹)
        
        Args:
            vl_input: Input batch with eagle_ prefixed keys
            
        Returns:
            (vision_features, language_features, attention_mask) tuple
        """
        eagle_prefix = "eagle_"
        
        # Get original batch size before rotation
        original_input_ids = vl_input.get(f"{eagle_prefix}input_ids")
        B = original_input_ids.shape[0]
        
        # Rotate images by all group elements
        rotated_pixel_values = self.rotate_vl_batch(dict(vl_input))
        
        # Prepare eagle input
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        
        # Remove image_sizes if present
        if "image_sizes" in eagle_input:
            del eagle_input["image_sizes"]
            
        ## Extract vision features (full tokens, not pooled)
        # vision_features: [B * num_imgs * N, T_vision, D_vision]
        vision_features, _ = self.eagle_model.extract_feature(rotated_pixel_values)
        
        # Get dimensions
        num_vision_tokens = vision_features.shape[1]   # T_vision (256)
        vision_dim = vision_features.shape[2]          # D_vision (2048)
        
        # Reshape to [B * num_imgs, N, T_vision, D_vision]
        vision_features_grouped = vision_features.reshape(
            B * self.num_images_per_sample, self.n_group, num_vision_tokens, vision_dim
        )
        
        # Apply ρ(h⁻¹) = spatial_perm(h⁻¹) ⊗ feature_perm(h⁻¹) to each f(h·x)
        # Using ρ(h⁻¹) instead of ρ(h) gives covariant output: FA(g·x) = ρ(g)·FA(x)
        # Then average over h
        
        transformed_features = []
        for h in range(self.n_group):
            # f(h·x): features from rotation h
            feat_h = vision_features_grouped[:, h, :, :]  # [B * num_imgs, T, D]
            
            if h == 0:
                # Identity - no transformation needed (h⁻¹ = 0)
                feat_transformed = feat_h
            else:
                # Apply ρ(h⁻¹) = π(h⁻¹) ⊗ ρ_reg(h⁻¹)
                h_inv = (self.n_group - h) % self.n_group
                
                # Step 1: Spatial permutation π(h⁻¹)
                spatial_perm = self.token_perm_indices[h_inv]
                feat_permuted = feat_h[:, spatial_perm, :]  # [B * num_imgs, T, D]
                
                # Step 2: Feature permutation ρ_reg(h⁻¹)
                # Regular representation: cyclically shift blocks by h_inv
                blocks = vision_dim // self.n_group
                feat_blocks = feat_permuted.reshape(
                    B * self.num_images_per_sample, num_vision_tokens, self.n_group, blocks
                )
                feat_shifted = torch.roll(feat_blocks, shifts=h_inv, dims=2)
                feat_transformed = feat_shifted.reshape(
                    B * self.num_images_per_sample, num_vision_tokens, vision_dim
                )
            
            transformed_features.append(feat_transformed)
        
        # Stack and average: [B * num_imgs, N, T, D] -> [B * num_imgs, T, D]
        transformed_features = torch.stack(transformed_features, dim=1)
        avg_vision_features = torch.mean(transformed_features, dim=1)
        
        # Reshape to [B, num_imgs, T_vision, D_vision]
        avg_vision_features = avg_vision_features.reshape(
            B, self.num_images_per_sample, num_vision_tokens, vision_dim
        )
        
        # Forward through Eagle language model to get language features
        language_features = self.eagle_model.extract_language_feature(
            **eagle_input, 
            output_hidden_states=True, 
            return_dict=True
        )
        
        # Get features from selected layer
        language_features = language_features.hidden_states[self.select_layer]  # [B, T_text, 2048]
        
        return avg_vision_features, language_features, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """
        Forward pass with Late Frame Averaging on full vision tokens.
        
        Args:
            vl_input: Input batch
            
        Returns:
            BatchFeature with:
                - backbone_vision_features: [B, num_imgs, T_vision, D] full vision token sequence
                - backbone_language_features: [B, T_text, D_text] language features
                - backbone_attention_mask: attention mask
        """
        self.set_frozen_modules_to_eval_mode()

        vision_embs, language_embs, eagle_mask = self.forward_eagle(vl_input)

        # DDP compatibility hack - ensure all trainable parameters participate in loss
        # This is needed because some parameters might not be used in certain forward paths
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=vision_embs.device, dtype=vision_embs.dtype, requires_grad=True
            )
            # Add all trainable parameters to the computation graph with zero contribution
            for param in self.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            vision_embs = vision_embs + dummy_term

        return BatchFeature(
            data={
                "backbone_vision_features": vision_embs,  # [B, num_imgs, T_vision, D_vision]
                "backbone_language_features": language_embs,  # [B, T_text, 2048]
                "backbone_attention_mask": eagle_mask
            }
        )
