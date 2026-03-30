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
Eagle Backbone with equiAdapt Canonicalization on Vision Tokens.

Instead of Frame Averaging (N backbone passes), equiAdapt uses a lightweight
canonicalization network to detect the canonical orientation of each equi image,
runs the Eagle backbone ONCE on the canonicalized image, then inverts the output
tokens using token_perm_indices to recover equivariant features.

EquiAdapt pipeline (only applied to equi images, i.e. rotate_image_indices):

    1. x_equi = equi image (may be in arbitrary orientation)
    2. [canonicalizer] detect rotation g: group_activations = canon_net(x_equi)
    3. canonical image: x_c = R(g^{-1}) · x_equi  (rotate back to canonical)
    4. [prediction_network = Eagle backbone] f(x_c) → vision tokens [B, T, D]
    5. [invert] apply ρ(g) to tokens using token_perm_indices + cyclic shift
       → equivariant output tokens f(x_equi) ≈ ρ(g) · f(x_c)

Non-equi images (not in rotate_image_indices) bypass canonicalization entirely.

Canonicalization Network (OptimizedGroupEquivariant style):
    - Group augment x_equi with all N rotations → [N·B, C, H', W']
    - Small CNN → feature vectors [N·B, out_vector_size]
    - Cosine similarity with learned reference_vector → group activations [B, N]
    - Straight-through argmax → differentiable rotation selection

Token Inversion (ρ(g) on token sequence):
    - Spatial permutation: token_perm_indices[g][p] gives source position
    - Feature cyclic shift: roll by g along the group dimension

Reference:
    equiAdapt: https://arxiv.org/abs/2209.15228
    token_perm_indices: EagleBackboneFATokens._init_token_permutation_indices
"""

import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


# ---------------------------------------------------------------------------
# Default canonicalization network (lightweight CNN)
# ---------------------------------------------------------------------------

class _DefaultCanonNet(nn.Module):
    """
    Lightweight CNN for group activation detection.

    Takes pre-processed images [B, C, H, W] and returns feature vectors
    [B, out_vector_size] used for cosine similarity with a reference vector.
    """

    def __init__(self, in_channels: int = 3, out_vector_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_vector_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x).flatten(1)  # [B, 128]
        return self.proj(feat)             # [B, out_vector_size]


# ---------------------------------------------------------------------------
# Main backbone
# ---------------------------------------------------------------------------

class EagleBackboneEquiAdapt(nn.Module):
    """
    Eagle Backbone with equiAdapt canonicalization.

    The Eagle VLM backbone acts as the prediction_network in the equiAdapt
    framework. A lightweight canonicalization network detects the rotation
    applied to each equi image; the image is then rotated to canonical form
    before being processed by Eagle. The output vision tokens are inverted
    using token_perm_indices (same spatial permutation table used in
    EagleBackboneFATokens) to recover equivariant token features.

    Only images whose indices are listed in rotate_image_indices receive
    canonicalization. All other images pass through the backbone unchanged.

    Args:
        tune_llm: whether to fine-tune the LLM part of Eagle.
        tune_visual: whether to fine-tune the visual encoder + mlp1.
        select_layer: LLM layer from which language features are extracted.
        project_to_dim: output dimension for vision tokens (divisible by n_group).
        n_group: size of cyclic group CN (4 for C4, 8 for C8).
        num_images_per_sample: total number of camera views per sample.
        rotate_image_indices: camera indices that are equivariant (get canonicalized).
            None means all cameras are equivariant.
        canonicalization_network: module mapping [B, C, H', W'] → [B, out_vector_size].
            If None, a default lightweight CNN is used.
        out_vector_size: size of feature vector from canonicalization_network.
        canon_image_size: resize equi images to this size before running the
            canonicalization network (for efficiency).
        canon_beta: temperature for straight-through softmax over group elements.
            Higher → closer to hard argmax.
    """

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: Optional[str] = None,
        project_to_dim: int = 1536,
        # Group structure
        n_group: int = 4,
        num_images_per_sample: int = 1,
        rotate_image_indices: Optional[List[int]] = None,
        # EquiAdapt canonicalization
        canonicalization_network: Optional[nn.Module] = None,
        out_vector_size: int = 128,
        canon_image_size: int = 64,
        canon_beta: float = 1.0,
    ):
        super().__init__()

        assert project_to_dim % n_group == 0, (
            f"project_to_dim ({project_to_dim}) must be divisible by n_group ({n_group})"
        )

        self.n_group = n_group
        self.num_images_per_sample = num_images_per_sample
        self.project_to_dim = project_to_dim
        self.canon_image_size = canon_image_size
        self.canon_beta = canon_beta

        if rotate_image_indices is None:
            self.rotate_image_indices = list(range(num_images_per_sample))
        else:
            self.rotate_image_indices = rotate_image_indices

        # ------------------------------------------------------------------ #
        # Eagle VLM (prediction_network)
        # ------------------------------------------------------------------ #
        eagle_model_path = eagle_path or DEFAULT_EAGLE_PATH
        config = AutoConfig.from_pretrained(eagle_model_path, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        self.eagle_linear = (
            nn.Linear(2048, project_to_dim) if project_to_dim else nn.Identity()
        )

        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)
        self.select_layer = select_layer

        # ------------------------------------------------------------------ #
        # Canonicalization network + reference vector
        # ------------------------------------------------------------------ #
        if canonicalization_network is None:
            canonicalization_network = _DefaultCanonNet(
                in_channels=3, out_vector_size=out_vector_size
            )
        self.canonicalization_network = canonicalization_network
        self.out_vector_size = out_vector_size

        # Learnable reference vector (OptimizedGroupEquivariant style)
        self.reference_vector = nn.Parameter(torch.randn(1, out_vector_size))

        # ------------------------------------------------------------------ #
        # Buffers for rotation and token permutation
        # ------------------------------------------------------------------ #
        self._init_rotation_matrices()
        self._init_token_permutation_indices(grid_size=16)

        self.set_trainable_parameters(tune_llm, tune_visual)

        print("EagleBackboneEquiAdapt initialized:")
        print(f"  n_group (CN): {self.n_group}")
        print(f"  project_to_dim: {self.project_to_dim}")
        print(f"  rotate_image_indices (equi cameras): {self.rotate_image_indices}")
        print(f"  canon_image_size: {self.canon_image_size}")
        print(f"  out_vector_size: {self.out_vector_size}")
        print(f"  Token grid: {self.token_grid_size}x{self.token_grid_size}")

    # ---------------------------------------------------------------------- #
    # Buffer initialization (shared with EagleBackboneFATokens)
    # ---------------------------------------------------------------------- #

    def _init_rotation_matrices(self):
        """Rotation matrices for image rotation via grid_sample."""
        angles = torch.linspace(0, 2 * math.pi, self.n_group + 1)[:-1]
        rot = torch.zeros(self.n_group, 2, 3)
        for i, angle in enumerate(angles):
            c = math.cos(-angle.item())
            s = math.sin(-angle.item())
            rot[i, 0, 0] = c
            rot[i, 0, 1] = -s
            rot[i, 1, 0] = s
            rot[i, 1, 1] = c
        self.register_buffer("rotation_matrices_buffer", rot)
        self.register_buffer("angles", angles)

    def _init_token_permutation_indices(self, grid_size: int = 16):
        """
        Spatial token permutation indices for each rotation.

        token_perm_indices[r, p] = source position in the *unrotated* token
        grid that ends up at position p after rotating the image by r steps.

        Identical to EagleBackboneFATokens._init_token_permutation_indices.
        """
        self.token_grid_size = grid_size
        N = grid_size
        num_tokens = N * N

        token_perm_indices = torch.zeros(self.n_group, num_tokens, dtype=torch.long)

        def rotate_90_ccw_source(i, j, N):
            return j, N - 1 - i

        for r in range(self.n_group):
            if self.n_group == 4:
                num_90 = r
            elif self.n_group == 8:
                num_90 = r // 2
            else:
                num_90 = 0

            for i in range(N):
                for j in range(N):
                    orig_idx = i * N + j
                    si, sj = i, j
                    for _ in range(num_90):
                        si, sj = rotate_90_ccw_source(si, sj, N)

                    if self.n_group == 8 and r % 2 == 1:
                        angle = r * 2 * math.pi / self.n_group
                        ci = si - (N - 1) / 2
                        cj = sj - (N - 1) / 2
                        cos_t = math.cos(-angle + num_90 * math.pi / 2)
                        sin_t = math.sin(-angle + num_90 * math.pi / 2)
                        new_ci = cos_t * ci - sin_t * cj
                        new_cj = sin_t * ci + cos_t * cj
                        si = int(round(new_ci + (N - 1) / 2))
                        sj = int(round(new_cj + (N - 1) / 2))
                        si = max(0, min(N - 1, si))
                        sj = max(0, min(N - 1, sj))

                    token_perm_indices[r, orig_idx] = si * N + sj

        self.register_buffer("token_perm_indices", token_perm_indices)

    # ---------------------------------------------------------------------- #
    # Training setup
    # ---------------------------------------------------------------------- #

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
                    print(f"  Trainable: {name}")

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_llm:
                self.eagle_model.language_model.eval()
            if not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    # ---------------------------------------------------------------------- #
    # Canonicalization helpers
    # ---------------------------------------------------------------------- #

    def _group_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augment images with all N rotations.

        Args:
            x: [B, C, H, W]
        Returns:
            [N*B, C, H, W]  — rotation r for sample b is at index r*B + b
        """
        B, C, H, W = x.shape
        device = x.device

        # Expand: [B, C, H, W] → [N*B, C, H, W]
        x_rep = x.unsqueeze(0).expand(self.n_group, -1, -1, -1, -1)  # [N, B, C, H, W]
        x_rep = x_rep.reshape(self.n_group * B, C, H, W)

        rot_idx = torch.arange(self.n_group, device=device).repeat_interleave(B)
        rot_mat = self.rotation_matrices_buffer[rot_idx]  # [N*B, 2, 3]

        grid = F.affine_grid(
            rot_mat.to(x.dtype), (self.n_group * B, C, H, W), align_corners=True
        )
        return F.grid_sample(x_rep, grid, align_corners=True, padding_mode="zeros")

    def _get_rotation_indices(
        self, x_equi: torch.Tensor
    ):
        """
        Run the canonicalization network on equi images to detect rotation.

        Uses the OptimizedGroupEquivariantImageCanonicalization approach:
        group augment → feature vectors → cosine similarity → group activations.

        Applies a straight-through estimator so gradients flow through the
        canonicalization network during training.

        Args:
            x_equi: [B, C, H, W] — may be at the original (large) resolution

        Returns:
            group_activations: [B, N]  (soft, for differentiable angle computation)
            rotation_indices: [B]      (hard argmax, for token permutation lookup)
        """
        B = x_equi.shape[0]
        device = x_equi.device

        # Resize for efficiency
        x_small = F.interpolate(
            x_equi, size=(self.canon_image_size, self.canon_image_size), mode="bilinear",
            align_corners=False,
        )

        # Group augment: [N*B, C, H', W']
        x_aug = self._group_augment(x_small)

        # Feature vectors from canonicalization network: [N*B, out_vector_size]
        vec_out = self.canonicalization_network(x_aug)

        # Cosine similarity with reference vector → scalar [N*B]
        ref = self.reference_vector.expand(self.n_group * B, -1)
        scalar = F.cosine_similarity(ref, vec_out, dim=-1)  # [N*B]

        # Cache for get_canonicalization_loss()
        self._last_vector_out = vec_out  # [N*B, out_vector_size]

        # Reshape to [N, B] → [B, N]
        group_activations = scalar.reshape(self.n_group, B).T.contiguous()  # [B, N]

        # Hard rotation indices (no gradient needed for the permutation lookup)
        rotation_indices = group_activations.argmax(dim=-1)  # [B]

        return group_activations, rotation_indices

    def _canonicalize_images(
        self,
        images: torch.Tensor,
        group_activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate each image to its canonical orientation.

        Uses a soft (differentiable) rotation angle computed from group_activations
        via the straight-through estimator:
            rotation_angle = Σ_r group_element_onehot[r] * (r * 360 / N)

        The image is then rotated by -rotation_angle to undo the detected rotation.

        Args:
            images: [B, C, H, W]
            group_activations: [B, N]  (raw logits from canonicalization network)

        Returns:
            canonical_images: [B, C, H, W]
        """
        B, C, H, W = images.shape
        device = images.device

        # Straight-through group element one-hot
        one_hot = F.one_hot(group_activations.argmax(dim=-1), self.n_group).float()
        soft = F.softmax(self.canon_beta * group_activations, dim=-1)
        if self.training:
            group_onehot = one_hot + soft - soft.detach()  # straight-through
        else:
            group_onehot = one_hot

        # Rotation angle per sample (in radians, CCW convention)
        angles_rad = self.angles.to(device)           # [N] in radians (0, 2π/N, ...)
        rotation_angle = (group_onehot * angles_rad).sum(-1)  # [B]

        # Canonicalize: rotate by -angle (undo detected rotation)
        canon_angle = -rotation_angle
        cos_v = torch.cos(canon_angle)   # [B]
        sin_v = torch.sin(canon_angle)   # [B]
        zeros = torch.zeros_like(cos_v)

        # Affine matrix [B, 2, 3]
        theta = torch.stack(
            [
                torch.stack([cos_v, -sin_v, zeros], dim=-1),
                torch.stack([sin_v,  cos_v, zeros], dim=-1),
            ],
            dim=1,
        )

        grid = F.affine_grid(theta, (B, C, H, W), align_corners=True)
        return F.grid_sample(images, grid, align_corners=True, padding_mode="zeros")

    # ---------------------------------------------------------------------- #
    # Token inversion  ρ(r) via token_perm_indices + cyclic shift
    # ---------------------------------------------------------------------- #

    def _invert_tokens(
        self,
        tokens: torch.Tensor,
        rotation_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply ρ(r) to vision token sequences to recover equivariant features.

        If the backbone is perfectly equivariant, then:
            f(R_r · x_c) = ρ(r) · f(x_c)
        so after running the backbone on the canonical image, applying ρ(r)
        recovers the output that would have been produced for the original image.

        ρ(r) consists of two parts (regular representation of C_N):
          1. Spatial permutation: token at output position p comes from
             token_perm_indices[r][p] in the canonical token sequence.
          2. Feature cyclic shift: the feature block dimension is shifted by r.

        Args:
            tokens: [B, T, D]  — output tokens from Eagle backbone on canonical images
            rotation_indices: [B]  — int in {0, ..., n_group-1}

        Returns:
            [B, T, D]  — equivariant token features
        """
        B, T, D = tokens.shape
        blocks = D // self.n_group
        device = tokens.device

        # 1. Spatial permutation
        #    perm[b, p] = source position in canonical tokens for output position p
        perm = self.token_perm_indices[rotation_indices]          # [B, T]
        b_idx = torch.arange(B, device=device).unsqueeze(1)       # [B, 1]
        tokens_spatial = tokens[b_idx, perm, :]                   # [B, T, D]

        # 2. Feature cyclic shift by r (per-sample)
        #    Equivalent to torch.roll(x, shifts=r, dims=-2) but per-sample
        tokens_blocks = tokens_spatial.reshape(B, T, self.n_group, blocks)  # [B, T, N, bl]

        arange = torch.arange(self.n_group, device=device).view(1, 1, self.n_group, 1)
        shifts = rotation_indices.view(B, 1, 1, 1)                 # [B, 1, 1, 1]
        # gather_idx[b,t,i,bl] = (i - r[b]) % N  implements roll by r
        gather_idx = (arange - shifts) % self.n_group              # [B, 1, N, 1]
        gather_idx = gather_idx.expand(B, T, self.n_group, blocks) # [B, T, N, bl]

        tokens_shifted = torch.gather(tokens_blocks, 2, gather_idx)  # [B, T, N, bl]

        return tokens_shifted.reshape(B, T, D)

    # ---------------------------------------------------------------------- #
    # Canonicalization auxiliary losses  (call after forward_eagle)
    # ---------------------------------------------------------------------- #

    def get_canonicalization_loss(self, artifact_err_wt: float = 0.0) -> torch.Tensor:
        """
        Auxiliary loss for training the canonicalization network.

        Combines two terms from OptimizedGroupEquivariantImageCanonicalization:

        1. **Orthogonality loss** (always active):
           Forces the feature vectors for different group elements to be
           orthogonal (cosine similarity ≈ 0 for distinct elements).
           This prevents mode collapse where all rotations map to the same vector.

        2. **Rotation artifact loss** (when artifact_err_wt > 0):
           Randomly re-rotates already-group-augmented images and checks that
           the canonicalization network gives consistent vectors.
           Penalises inconsistency via MSE between the two sets of vectors.
           This requires calling _compute_artifact_vectors() separately.

        Args:
            artifact_err_wt: weight for rotation artifact loss (0 = disabled).

        Returns:
            Scalar loss tensor. Add to the main task loss during training.
        """
        if not hasattr(self, "_last_vector_out"):
            raise RuntimeError(
                "get_canonicalization_loss() must be called after forward_eagle()."
            )

        vectors = self._last_vector_out  # [N*B, out_vector_size]

        # Infer batch size from cached vector
        NB = vectors.shape[0]
        B = NB // self.n_group

        # Orthogonality: shape [B, N, out_vector_size]
        vectors_per_sample = vectors.reshape(self.n_group, B, self.out_vector_size)
        vectors_per_sample = vectors_per_sample.permute(1, 0, 2)  # [B, N, D]

        # Gram matrix [B, N, N]
        distances = vectors_per_sample @ vectors_per_sample.permute(0, 2, 1)
        mask = 1.0 - torch.eye(self.n_group, device=vectors.device)  # [N, N]
        ortho_loss = torch.abs(distances * mask.unsqueeze(0)).mean()

        if artifact_err_wt > 0.0 and hasattr(self, "_last_vector_out_artifact"):
            artifact_loss = F.mse_loss(self._last_vector_out_artifact, vectors.detach())
            return ortho_loss + artifact_err_wt * artifact_loss

        return ortho_loss

    # ---------------------------------------------------------------------- #
    # Main forward
    # ---------------------------------------------------------------------- #

    def forward_eagle(self, vl_input: BatchFeature):
        """
        Forward pass with equiAdapt canonicalization on equi cameras.

        Pipeline:
          Step 1: Canonicalize each equi camera, store rotation indices.
                  Non-equi cameras are kept unchanged.

          Step 2: Build mixed pixel_values:
                  [canonical_equi_cam0, ..., original_non_equi_camK, ...]  [B*num_imgs, C, H, W]

          Step 3: Run Eagle model ONCE with mixed pixel_values:
                  a) extract_feature(pixel_values_mixed) → all-camera vision tokens [B*num_imgs, T, D]
                  b) eagle(**) with pixel_values_mixed → LLM hidden_states [B, T_text, D]
                  Both use the same canonicalized inputs, so the LLM context also
                  benefits from the canonical view of equi cameras.

          Step 4: Slice equi camera tokens from the all-camera batch,
                  apply _invert_tokens (ρ(r)) to recover equivariant features.

        Returns:
            (equi_vision_features [B, n_equi, T, D],
             vl_features [B, T_text, D],
             attention_mask)
        """
        eagle_prefix = "eagle_"

        pixel_values = vl_input[f"{eagle_prefix}pixel_values"]  # [B*num_imgs, C, H, W]
        B = vl_input[f"{eagle_prefix}input_ids"].shape[0]
        _, C, H, W = pixel_values.shape

        # [B*num_imgs, C, H, W] → [B, num_imgs, C, H, W]
        img_batch = pixel_values.reshape(B, self.num_images_per_sample, C, H, W)

        # ------------------------------------------------------------------ #
        # Step 1: Canonicalize equi cameras
        # ------------------------------------------------------------------ #
        rotation_indices_per_cam: dict = {}
        img_batch_mixed = img_batch.clone()

        for cam_idx in self.rotate_image_indices:
            x_cam = img_batch[:, cam_idx]  # [B, C, H, W]
            group_activations, rotation_indices = self._get_rotation_indices(x_cam)
            img_batch_mixed[:, cam_idx] = self._canonicalize_images(x_cam, group_activations)
            rotation_indices_per_cam[cam_idx] = rotation_indices

        # ------------------------------------------------------------------ #
        # Step 2: Build mixed pixel_values
        #         canonical for equi cams, original for non-equi cams
        # ------------------------------------------------------------------ #
        # [B, num_imgs, C, H, W] → [B*num_imgs, C, H, W]
        pixel_values_mixed = img_batch_mixed.reshape(B * self.num_images_per_sample, C, H, W)

        # ------------------------------------------------------------------ #
        # Step 3: ONE full Eagle forward with mixed pixel_values
        #
        # Eagle internally calls extract_feature and inserts vision tokens at
        # positions where input_ids == image_token_index before running the LLM.
        # hidden_states[0]  = the input embeddings (pure vision tokens + text)
        # hidden_states[select_layer] = LLM output for language features
        # ------------------------------------------------------------------ #
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        if "image_sizes" in eagle_input:
            del eagle_input["image_sizes"]
        eagle_input["pixel_values"] = pixel_values_mixed

        vl_out = self.eagle_model(
            **eagle_input, output_hidden_states=True, return_dict=True
        )

        # ------------------------------------------------------------------ #
        # Step 4: Project LLM output → vl_features, then apply ρ(r) to equi
        #         camera positions in-place and extract as equi_vision_features.
        #
        # image_mask[b, p] = True  iff input_ids[b, p] == image_token_index.
        # Image tokens are laid out contiguously per camera:
        #   cam 0 → positions [0 : T_vis]
        #   cam 1 → positions [T_vis : 2*T_vis]
        #   ...  (within the set of True positions for sample b)
        #
        # We:
        #   1. Project the full LLM hidden state with eagle_linear.
        #   2. For each equi camera, gather its T_vis token rows from vl_features,
        #      apply _invert_tokens, then scatter the result back into vl_features.
        #   3. Also collect the inverted tokens as equi_vision_features.
        # ------------------------------------------------------------------ #
        input_ids = eagle_input["input_ids"]                             # [B, N_seq]
        image_mask = (input_ids == self.eagle_model.image_token_index)  # [B, N_seq]

        # Project: [B, N_seq, D_llm] → [B, N_seq, project_to_dim]
        vl_features = self.eagle_linear(vl_out.hidden_states[self.select_layer])

        # Infer T_vis from mask (uniform across batch in GR00T)
        T_vis = image_mask[0].sum().item() // self.num_images_per_sample

        # Sequence indices of all image tokens per sample: [B, num_imgs * T_vis]
        _, col_idx = image_mask.nonzero(as_tuple=True)
        col_idx_2d = col_idx.reshape(B, self.num_images_per_sample * T_vis)  # [B, NI]

        b_idx = torch.arange(B, device=vl_features.device).unsqueeze(1)  # [B, 1]

        equi_vision_features_list = []
        for cam_idx in self.rotate_image_indices:
            start = cam_idx * T_vis
            end   = start + T_vis

            # Sequence positions of this camera's tokens: [B, T_vis]
            cam_col = col_idx_2d[:, start:end]

            # Gather: [B, T_vis, D]
            tokens_cam = vl_features[b_idx, cam_col]

            # Apply ρ(r) → equivariant tokens: [B, T_vis, D]
            equi_tokens = self._invert_tokens(tokens_cam, rotation_indices_per_cam[cam_idx])
            equi_vision_features_list.append(equi_tokens)

            # Write equivariant tokens back into vl_features at the same positions
            vl_features[b_idx, cam_col] = equi_tokens

        # [B, n_equi, T_vis, D]
        equi_vision_features = torch.stack(equi_vision_features_list, dim=1)

        attention_mask = eagle_input["attention_mask"]

        return equi_vision_features, vl_features, attention_mask

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """
        Full forward pass.

        Returns BatchFeature with:
            backbone_equi_vision_features: [B, n_equi, T_vision, D]  equivariant cameras
            backbone_vision_language_features: [B, T_text, D]         language features
            backbone_attention_mask: attention mask
        """
        self.set_frozen_modules_to_eval_mode()

        equi_vision_embs, vl_embs, eagle_mask = self.forward_eagle(vl_input)

        # DDP: ensure all trainable parameters participate in the loss graph
        if self.training and self.tune_visual:
            dummy = torch.tensor(0.0, device=vl_embs.device, dtype=vl_embs.dtype,
                                 requires_grad=True)
            for p in self.parameters():
                if p.requires_grad:
                    dummy = dummy + 0.0 * p.sum()
            vl_embs = vl_embs + dummy

        return BatchFeature(
            data={
                "backbone_equi_vision_features": equi_vision_embs,
                "backbone_vision_language_features": vl_embs,
                "backbone_attention_mask": eagle_mask,
            }
        )
