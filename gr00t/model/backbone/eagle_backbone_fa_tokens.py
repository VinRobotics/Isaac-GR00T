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

Pipeline (EquiLLM-faithful, Section 3.2 of arXiv:2502.11149):
    1. FA (Phase 1): Rotate images N times, extract features, apply FA:
       fa_equi_raw = equivariant FA (with ρ(h⁻¹) ⊗ π(h⁻¹))  → regular repr tokens
       fa_inv_raw  = invariant FA (plain average, no transformation) → invariant tokens
       Both bypass the LLM (skip connections, EquiLLM X̃' and H' analogs).
    2. Invariant FA + Eagle LLM (Phase 2): fa_inv_raw projected → injected into frozen VLM.
       VLM attends jointly over invariant vision + text tokens → vlm_hidden.
       Analogy: EquiLLM's Projector(H') + Prompt P → LLM → H^llm.
    3. EquiAdapter (Phase 3): faithful EquiLLM adapter (Eq. 6).
       h_cond     = lang_proj(vlm_img) + h_inv_skip  ← EquiLLM: H^r = proj(H^llm) + H'
       SA+CA loop = SA(h_equi) → CA(h_equi ← h_cond) ← EquiLLM: EGNN(H^r, X̃')
       text_ca    = standard CA: Q=[noequi|text], K=V=inv(h_vis)  ← text attends to vision,
                    stays invariant (both Q and K/V invariant → output invariant ✓)
    4. Output: [B, n_equi*T_vis + n_noequi*T_vis + T_lang, D]
       [:n_equi*T]        equivariant vision tokens (regular repr)
       [n_equi*T:-T_lang] non-equi camera tokens (trivial-in-regular, invariant)
       [-T_lang:]         language tokens (trivial-in-regular, invariant)
"""

import os
import math
from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import escnn
import escnn.nn as enn
from escnn.group import CyclicGroup
from gr00t.model.action_head.equivariant_cross_attention_dit import BasicTransformerBlock

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)



class EquiAdapter(nn.Module):
    """
    Phase 3: EquiLLM-faithful equivariant adapter (Section 3.2, Eq. 6).

    Single CA block: h_equi queries ← vlm_img keys/values (invariant context).
    No SA — vlm_img already contains rich cross-modal context from the frozen VLM,
    and h_equi has spatial structure from FA. One CA pass is sufficient to inject
    language/visual context into the equivariant tokens.

    Token-based adaptation:
      h_cond    = lang_proj(vlm_img)          # project VLM output into equi space (invariant)
      h_adapted = CA(h_equi, context=h_cond)  # equivariant queries attend to invariant context

    Equivariance: CA queries come from h_equi (regular repr); K/V come from h_cond (invariant).
    Attention weights are the same across all group slots → permutation of group slots
    commutes with attention → output remains equivariant ✓

    Language tokens (VLM output at text positions) are appended to output unchanged.

    Output: [B, n_equi*T_vis + n_noequi*T_vis + T_lang, D]
      [:n_equi*T_vis]    equivariant vision tokens (CA-modulated, regular repr)
      [n_equi*T:-T_lang] non-equi camera tokens   (trivial-in-regular, invariant)
      [-T_lang:]         language tokens           (trivial-in-regular, invariant)

    Identity init: zero-init v_proj + o_proj + ff.fc2 → CA output = 0 at init ✓
    """

    def __init__(
        self,
        d_eq: int,
        d_llm: int,
        n_group: int,
        num_heads: int = 32,
        attention_head_dim: int = 64,
        num_layers: int = 1,  # CA-only blocks (SA removed)
    ):
        super().__init__()
        self.n_group = n_group
        self.d_llm = d_llm
        self.d_eq = d_eq
        blocks = d_eq // n_group
        self.blocks = blocks
        scalar_dim = num_heads * attention_head_dim
        assert scalar_dim % n_group == 0, (
            f"num_heads*attention_head_dim ({scalar_dim}) must be divisible by n_group ({n_group})"
        )

        G = CyclicGroup(n_group)
        gs = escnn.gspaces.no_base_space(G)

        # h_prime type: equivariant vision tokens (regular repr)
        equi_type     = enn.FieldType(gs, [gs.regular_repr] * blocks)
        # h_cond type: invariant context (trivial repr)
        inv_type      = enn.FieldType(gs, [gs.trivial_repr] * d_llm)
        ff_inner_type = enn.FieldType(gs, [gs.regular_repr] * blocks)

        # CA: h_equi queries ← lang_proj(vlm_img) keys/values (invariant context).
        # No SA — downstream DiT handles token interaction; VLM already did cross-modal mixing.
        self.ca_blocks = nn.ModuleList([
            BasicTransformerBlock(
                in_type=equi_type,
                cross_attention_type=inv_type,
                inner_type=ff_inner_type,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                norm_type="layer_norm",
            )
            for _ in range(num_layers)
        ])

        # lang_proj: project VLM output at image positions into equi space (invariant).
        self.lang_proj = nn.Linear(d_llm, d_eq)

        # lang_embed / noequi_embed: reformat VLM output tokens as trivial-in-regular repr.
        # trivial-in-regular: R^blocks expanded as [v,v,...,v]×G → invariant subspace of regular repr.
        self.lang_embed   = nn.Linear(d_eq, blocks)
        self.noequi_embed = nn.Linear(d_eq, blocks)

        # Identity init: CA output = 0 at training start → h_equi passes through unchanged.
        for ca_blk in self.ca_blocks:
            for p in ca_blk.attn1.v_proj.parameters():
                nn.init.zeros_(p)
            for p in ca_blk.attn1.o_proj.parameters():
                nn.init.zeros_(p)
            for p in ca_blk.ff.fc2.parameters():
                nn.init.zeros_(p)

        # Warm-up scale: linearly ramp adapter delta from 0 → 1 over warmup_steps.
        # Initialized to 1.0 (full strength) so the default behavior is unchanged.
        # Call set_adapter_warmup_scale(step, warmup_steps) from the trainer each step.
        self.register_buffer("adapter_scale", torch.ones(1))

    def forward(
        self,
        h_equi: torch.Tensor,
        vlm_img: torch.Tensor,
        vlm_text: torch.Tensor,
        noequi_vlm: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            h_equi:     [B, n_equi, T_vis, D]   equivariant FA features (X̃' analog, skip)
            vlm_img:    [B, n_equi*T_vis, D]    VLM output at equi-image positions (H^llm analog)
            vlm_text:   [B, T_lang, D]           VLM output at text positions
            noequi_vlm: [B, n_noequi*T_vis, D]  VLM output at non-equi camera positions (optional)
        Returns:
            (h_vis, inv_tokens) tuple:
              h_vis:      [B, n_equi*T_vis, D]           equivariant vision tokens (regular repr)
              inv_tokens: [B, n_noequi*T_vis + T_lang, blocks]  invariant scalar tokens (D//n_group)

        Pipeline:
          CA: h_equi queries attend to lang_proj(vlm_img) context (invariant).
          inv_tokens: noequi/text VLM output projected to scalar dim (no group expansion).
        """
        B, n_equi, T, D = h_equi.shape
        dt = next(self.ca_blocks[0].parameters()).dtype

        # CA context: project vlm_img into equi space (invariant).
        h_cond = self.lang_proj(vlm_img.to(dt))                        # [B, n_equi*T, D]
        h_vis = h_equi.to(dt).reshape(B, n_equi * T, D)
        h_vis_in = h_vis                                                # save skip for warm-up scaling
        for ca_blk in self.ca_blocks:
            h_vis = ca_blk(h_vis, encoder_hidden_states=h_cond)
        # Scale only the adapter delta (not the skip path) by the warm-up factor.
        # At init: delta ≈ 0 (zero-init) → scale has no effect until adapter starts learning.
        # During warm-up: adapter_scale ramps 0→1 so newly learned deltas enter gradually.
        if self.adapter_scale.item() != 1.0:
            h_vis = h_vis_in + (h_vis - h_vis_in) * self.adapter_scale

        # Extract invariant scalar tokens: project VLM output to R^blocks (no group expansion).
        # to_inv_scalar: [B, T, D] → [B, T, blocks]  (invariant, no regular-repr lifting)
        def to_inv_scalar(x, embed):
            return embed(x)                                             # [B, T, blocks]

        text_out = to_inv_scalar(vlm_text.to(dt), self.lang_embed)     # [B, T_lang, blocks]
        if noequi_vlm is not None:
            noequi_out = to_inv_scalar(noequi_vlm.to(dt), self.noequi_embed)  # [B, T_noequi, blocks]
            inv_tokens = torch.cat([noequi_out, text_out], dim=1)      # [B, T_noequi+T_lang, blocks]
        else:
            inv_tokens = text_out                                       # [B, T_lang, blocks]

        return h_vis.to(h_equi.dtype), inv_tokens.to(h_equi.dtype)


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
        # Phase 2/3: equivariant adapter
        use_inv_projector_for_vlm: bool = True,  # Phase 2: inject inv-proj tokens into LLM
        equi_adapter_num_heads: int = 32,         # heads in SA/CA (reduced for data efficiency)
        equi_adapter_attention_head_dim: int = 64, # head dim; scalar_dim = num_heads * head_dim
        equi_adapter_num_layers: int = 1,        # CA-only blocks
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
            use_inv_projector_for_vlm: if True (Phase 2), replace SigLIP tokens in the
                Eagle LLM with InvariantProjector(H') tokens instead of running SigLIP again.
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here"
        assert output_type == 'reg', "Only regular representation is supported"

        # Store config
        self.n_group = n_group
        self.num_images_per_sample = num_images_per_sample
        self.output_type = output_type
        self.project_to_dim = project_to_dim if project_to_dim else 2048
        self.use_inv_projector_for_vlm = use_inv_projector_for_vlm
        self.equi_adapter_num_heads = equi_adapter_num_heads
        self.equi_adapter_attention_head_dim = equi_adapter_attention_head_dim
        self.equi_adapter_num_layers = equi_adapter_num_layers

        if rotate_image_indices is None:
            self.rotate_image_indices = list(range(num_images_per_sample))
        else:
            self.rotate_image_indices = rotate_image_indices

        self.non_equi_image_indices = [
            i for i in range(num_images_per_sample)
            if i not in self.rotate_image_indices
        ]

        # Ensure project_to_dim is divisible by n_group for regular representation
        assert self.project_to_dim % n_group == 0, \
            f"project_to_dim ({self.project_to_dim}) must be divisible by n_group ({n_group})"

        # Initialize Eagle model
        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        # Projection layers: Eagle vision/LLM dim → project_to_dim
        # Eagle mlp1 output dim matches LLM hidden size (e.g. 2048 for InternLM2-1.8B)
        d_eagle = self.eagle_model.language_model.config.hidden_size

        # Vision projection: equivariant in group-major format.
        # FA output uses group-major layout (D = n_group * blocks).
        # Applying the same linear to each group-slice commutes with the group cyclic shift.
        assert d_eagle % n_group == 0, \
            f"d_eagle ({d_eagle}) must be divisible by n_group ({n_group}) for equivariant projection"
        blocks_in  = d_eagle // n_group
        blocks_out = self.project_to_dim // n_group
        self.vision_proj = nn.Linear(blocks_in, blocks_out)

        # Language projection: language hidden states are invariant — plain linear is fine.
        self.eagle_linear = nn.Linear(d_eagle, self.project_to_dim)

        # Remove unused LLM layers (only if layers exist and select_layer is positive)
        # select_layer=-1 means keep all layers, select_layer=N means keep first N layers
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer

        # ── Phase 2: Invariant FA projector ─────────────────────────────────
        # FA_inv(x) = (1/|G|) Σ_g f(g·x)  — plain average, no transformation.
        # Proof: FA_inv(r·x)[p] = (1/|G|) Σ_g f(g·r·x)[p]
        #        = (1/|G|) Σ_h f(h·x)[p]   [h=g·r, same sum over G]  = FA_inv(x) ✓
        # Output lives in vision_dim (= d_eagle) space; project to d_eagle for VLM injection.
        self.inv_fa_proj = nn.Linear(d_eagle, d_eagle)

        # ── Phase 3: Invariant FA projector for EquiAdapter (H' skip) ───────
        # Projects h_inv_raw (invariant FA features) to project_to_dim.
        # This is the H' skip connection in EquiLLM: bypasses LLM, goes directly to adapter.
        # Separate from inv_fa_proj (which projects to d_eagle for LLM injection, Phase 2).
        self.inv_adapter_proj = nn.Linear(d_eagle, self.project_to_dim)

        # ── Phase 3: EquiAdapter (num_layers × CA blocks, no SA) ────────────────
        # CA: h_equi queries attend to lang_proj(vlm_img) context (invariant).
        # Language tokens appended to output unchanged.
        self.equi_adapter = EquiAdapter(
            d_eq=self.project_to_dim,
            d_llm=self.project_to_dim,
            n_group=n_group,
            num_heads=equi_adapter_num_heads,
            attention_head_dim=equi_adapter_attention_head_dim,
            num_layers=equi_adapter_num_layers,
        )

        # Initialize rotation and frame averaging components
        self._init_rotation_matrices()
        self._init_permutation_matrices()
        # Token grid size: 16x16 = 256 tokens (typical for Eagle after pixel shuffle)
        self._init_token_permutation_indices(grid_size=16)

        self.set_trainable_parameters(tune_llm, tune_visual)

        print(f"EagleBackboneFATokens initialized:")
        print(f"  n_group (CN): {self.n_group}")
        print(f"  d_eagle (LLM hidden): {d_eagle}")
        print(f"  project_to_dim: {self.project_to_dim}")
        print(f"  rotate_image_indices: {self.rotate_image_indices}")
        print(f"  non_equi_image_indices: {self.non_equi_image_indices}")
        print(f"  output_type: {self.output_type}")
        print(f"  use_inv_projector_for_vlm: {self.use_inv_projector_for_vlm}")
        print(f"  equi_adapter_num_layers:       {self.equi_adapter_num_layers} (CA-only blocks)")
        print(f"  equi_adapter_num_heads:        {self.equi_adapter_num_heads}")
        print(f"  equi_adapter_attention_head_dim: {self.equi_adapter_attention_head_dim}  (scalar_dim={self.equi_adapter_num_heads * self.equi_adapter_attention_head_dim})")
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
        When the image is rotated by g, token at spatial position q moves to
        position p where perm[r][p] = q  ("position p gets content from q").

        Convention matches _apply_rotations_to_images / F.affine_grid with
        matrix [[cos(a), sin(a)], [-sin(a), cos(a)]], i.e. CW rotation by
        angle a = r * 2π/N.  For dest pixel (i, j) the source pixel is:

            src_col = center + cos(a)*(j-center) + sin(a)*(i-center)
            src_row = center - sin(a)*(j-center) + cos(a)*(i-center)

        For exact 90° multiples nearest-neighbor is lossless (no two dest
        positions round to the same source).  For non-90° angles (C8 odd
        steps) naive rounding creates collisions; those are resolved with a
        greedy bijective assignment that minimises total displacement.

        perm[r][dest] = source index  (dest gets content from source)

        Args:
            grid_size: size of the token grid (sqrt of num_tokens)
        """
        self.token_grid_size = grid_size
        num_tokens = grid_size * grid_size
        N = grid_size
        center = (N - 1) / 2

        token_perm_indices = torch.zeros(self.n_group, num_tokens, dtype=torch.long)

        for r in range(self.n_group):
            # CW rotation angle matching affine_grid convention
            angle = r * 2 * math.pi / self.n_group
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Compute continuous source position for each destination
            float_src_row = []
            float_src_col = []
            for i in range(N):
                for j in range(N):
                    ci = j - center   # col offset
                    ri = i - center   # row offset
                    float_src_col.append(center + cos_a * ci + sin_a * ri)
                    float_src_row.append(center - sin_a * ci + cos_a * ri)

            # For exact 90° multiples, nearest-neighbor is a lossless bijection
            is_exact_90 = abs(math.sin(angle) ** 2 - round(math.sin(angle) ** 2)) < 1e-9

            if is_exact_90:
                for dest in range(num_tokens):
                    si = int(round(float_src_row[dest]))
                    sj = int(round(float_src_col[dest]))
                    si = max(0, min(N - 1, si))
                    sj = max(0, min(N - 1, sj))
                    token_perm_indices[r, dest] = si * N + sj
            else:
                # Greedy bijective assignment: process destinations in order of
                # increasing rounding error so best-fitting assignments go first.
                rounding_err = [
                    (float_src_row[d] - round(float_src_row[d])) ** 2
                    + (float_src_col[d] - round(float_src_col[d])) ** 2
                    for d in range(num_tokens)
                ]
                dests_sorted = sorted(range(num_tokens), key=lambda d: rounding_err[d])

                assigned_src = [0] * num_tokens
                used_srcs: set = set()

                for dest in dests_sorted:
                    sr = float_src_row[dest]
                    sc = float_src_col[dest]
                    best_idx = None
                    best_dist = float("inf")
                    # Expand search radius until an unused source is found
                    for radius in range(N):
                        for di in range(-radius, radius + 1):
                            for dj in range(-radius, radius + 1):
                                if max(abs(di), abs(dj)) != radius:
                                    continue
                                si = int(round(sr)) + di
                                sj = int(round(sc)) + dj
                                if not (0 <= si < N and 0 <= sj < N):
                                    continue
                                idx = si * N + sj
                                if idx in used_srcs:
                                    continue
                                dist = (sr - si) ** 2 + (sc - sj) ** 2
                                if dist < best_dist:
                                    best_dist = dist
                                    best_idx = idx
                        if best_idx is not None:
                            break
                    if best_idx is None:
                        # Fallback: first unused source (should not happen)
                        best_idx = next(i for i in range(num_tokens) if i not in used_srcs)
                    assigned_src[dest] = best_idx
                    used_srcs.add(best_idx)

                for dest in range(num_tokens):
                    token_perm_indices[r, dest] = assigned_src[dest]

        # Enforce inverse property for non-exact rotations:
        # _permute_tokens_for_rotation(·, r, inverse=True) uses perm[(n-r)%n].
        # For exact 90° steps the group structure already guarantees this; for
        # approximate (greedy) rotations we must set perm[(n-r)%n] = perm[r]⁻¹
        # explicitly so that forward ∘ inverse == identity.
        arange = torch.arange(num_tokens, dtype=torch.long)
        for r in range(1, self.n_group):
            angle = r * 2 * math.pi / self.n_group
            is_exact_90 = abs(math.sin(angle) ** 2 - round(math.sin(angle) ** 2)) < 1e-9
            if not is_exact_90:
                r_inv = (self.n_group - r) % self.n_group
                if r_inv > r:  # process each pair once
                    inv_perm = torch.zeros(num_tokens, dtype=torch.long)
                    inv_perm[token_perm_indices[r]] = arange
                    token_perm_indices[r_inv] = inv_perm

        self.register_buffer("token_perm_indices", token_perm_indices)

    # ── New layers added on top of the pretrained VLM ────────────────────────
    # These are the only parameters that should be trained / saved / loaded.
    _NEW_LAYER_PREFIXES = (
        "vision_proj.",
        "eagle_linear.",
        "inv_fa_proj.",
        "inv_adapter_proj.",
        "equi_adapter.",
    )

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        """
        Freeze the entire eagle_model (VLM) and train only the new equivariant layers.

        tune_llm / tune_visual are kept as arguments for API compatibility but
        the VLM is always frozen here — only vision_proj, eagle_linear,
        inv_projector and equi_adapter are trained.
        """
        self.tune_llm = False
        self.tune_visual = False

        # Freeze entire VLM
        self.eagle_model.requires_grad_(False)

        # Unfreeze new equivariant layers only
        for name, p in self.named_parameters():
            if name.startswith(self._NEW_LAYER_PREFIXES):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"Backbone trainable parameters ({len(trainable)}):")
        for n in trainable:
            print(f"  {n}")
        if not trainable:
            print("  Warning: no trainable parameters found.")

    def set_adapter_warmup_scale(self, step: int, warmup_steps: int) -> None:
        """
        Update the EquiAdapter warm-up scale based on the current training step.

        Call this once per training step (before or after the forward pass).
        The adapter delta is multiplied by min(1.0, step / warmup_steps), so the
        adapter's learned contribution ramps in linearly from 0 → 1 over the first
        `warmup_steps` steps.  After warm-up the scale is fixed at 1.0.

        Example (in training loop or Trainer callback):
            model.backbone.set_adapter_warmup_scale(global_step, warmup_steps=500)

        Args:
            step:          current global training step (0-indexed)
            warmup_steps:  number of steps over which to ramp; 0 = no warm-up (scale=1)
        """
        if warmup_steps <= 0:
            return
        scale = min(1.0, step / warmup_steps)
        self.equi_adapter.adapter_scale.fill_(scale)

    def load_pretrained_vlm(self, checkpoint_path: str) -> None:
        """
        Load eagle_model weights from a GR00T N1.5 checkpoint.

        Handles sharded safetensors (model.safetensors.index.json),
        single safetensors, and pytorch_model.bin formats.
        Keys are expected to be prefixed with "backbone." in the checkpoint.
        New layers (vision_proj, equi_adapter, etc.) that are absent from the
        checkpoint are left with their random initialisation.
        """
        import json

        def _load_shard(path: str) -> dict:
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(path)
            return torch.load(path, map_location="cpu")

        # Collect full state dict from checkpoint
        ckpt_path = checkpoint_path
        index_file = os.path.join(ckpt_path, "model.safetensors.index.json")
        single_sf  = os.path.join(ckpt_path, "model.safetensors")
        single_bin = os.path.join(ckpt_path, "pytorch_model.bin")

        raw: dict = {}
        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            shards = set(index["weight_map"].values())
            for shard in shards:
                raw.update(_load_shard(os.path.join(ckpt_path, shard)))
        elif os.path.exists(single_sf):
            raw = _load_shard(single_sf)
        elif os.path.exists(single_bin):
            raw = _load_shard(single_bin)
        else:
            raise FileNotFoundError(f"No checkpoint weights found in {ckpt_path}")

        # Strip "backbone." prefix → keys match self.state_dict()
        backbone_sd = {
            k.removeprefix("backbone."): v
            for k, v in raw.items()
            if k.startswith("backbone.")
        }

        # Load into backbone; strict=False so new layers are skipped
        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)

        # New-layer keys are expected to be missing — report only truly missing VLM keys
        new_prefixes = self._NEW_LAYER_PREFIXES
        unexpected_vlm = [k for k in missing if not k.startswith(new_prefixes)]
        print(f"Loaded backbone weights from {ckpt_path}")
        print(f"  New layers (random init): "
              f"{[k for k in missing if k.startswith(new_prefixes)][:5]}"
              f"{'...' if sum(k.startswith(new_prefixes) for k in missing) > 5 else ''}")
        if unexpected_vlm:
            print(f"  WARNING — unexpected missing VLM keys ({len(unexpected_vlm)}): "
                  f"{unexpected_vlm[:5]}")
        if unexpected:
            print(f"  Unexpected keys in checkpoint (ignored): {unexpected[:5]}")

    def new_layers_state_dict(self) -> dict:
        """
        Return state dict of only the new equivariant layers (vision_proj,
        eagle_linear, inv_projector, equi_adapter).  Use this to save a
        lightweight checkpoint after fine-tuning.
        """
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith(self._NEW_LAYER_PREFIXES)
        }

    def load_new_layers_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """
        Load back a checkpoint saved by new_layers_state_dict().
        Passes strict=False to the full load so VLM keys are not required.
        """
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        truly_missing = [k for k in missing if k.startswith(self._NEW_LAYER_PREFIXES)]
        if truly_missing and strict:
            raise RuntimeError(f"Missing new-layer keys: {truly_missing}")
        if truly_missing:
            print(f"  Warning — missing new-layer keys: {truly_missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:5]}")

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
            padding_mode='border'
        )
        
        return rotated_imgs

    def rotate_vl_batch(self, vl_input: dict) -> tuple:
        """
        Rotate only the equivariant images (rotate_image_indices) for FA.
        Non-equivariant images (non_equi_image_indices) are returned unrotated.

        Returns:
            equi_pixels  : [B * N * n_equi, C, H, W]
            noequi_pixels: [B * n_noequi, C, H, W]   or None if no non-equi cameras
            img_batch    : [B, num_images_per_sample, C, H, W]
            B            : original batch size
        """
        eagle_prefix = "eagle_"
        pixel_values = vl_input[f"{eagle_prefix}pixel_values"]
        B = vl_input[f"{eagle_prefix}input_ids"].shape[0]
        _, C, H, W = pixel_values.shape

        img_batch = pixel_values.reshape(B, self.num_images_per_sample, C, H, W)
        n_equi = len(self.rotate_image_indices)

        equi_imgs = [
            self._apply_rotations_to_images(img_batch[:, idx]).reshape(B, self.n_group, C, H, W)
            for idx in self.rotate_image_indices
        ]
        # [n_equi, B, N, C, H, W] → [B, N, n_equi, C, H, W] → [B*N*n_equi, C, H, W]
        equi_pixels = (
            torch.stack(equi_imgs, dim=0)
            .permute(1, 2, 0, 3, 4, 5)
            .reshape(B * self.n_group * n_equi, C, H, W)
        )

        # Non-equi cameras: no rotation, just stack and flatten
        if self.non_equi_image_indices:
            n_noequi = len(self.non_equi_image_indices)
            noequi_pixels = torch.stack(
                [img_batch[:, idx] for idx in self.non_equi_image_indices], dim=1
            ).reshape(B * n_noequi, C, H, W)  # [B*n_noequi, C, H, W]
        else:
            noequi_pixels = None

        return equi_pixels, noequi_pixels, img_batch, B

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

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2 helper: inject InvariantProjector tokens into Eagle LLM
    # ──────────────────────────────────────────────────────────────────────────

    def _forward_llm_with_injected_vision(
        self,
        eagle_input: dict,
        inv_vision_tokens: torch.Tensor,
        noequi_pixels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run Eagle's LLM with injected vision tokens.

          - Equi cameras: inv_projector(H') tokens fill the first n_equi*T image positions.
          - Non-equi cameras (e.g. wrist): SigLIP features fill the remaining positions.
            These are extracted here (single pass, no rotation) and also returned so
            forward_eagle can project and pass them to EquiAdapter + action head.

        Steps:
          1. Embed text tokens.
          2. If noequi_pixels provided, run extract_feature on them → noequi_raw [B*n_noequi, T, D].
          3. Fill image placeholders: equi tokens first, noequi tokens after.
          4. Run LLM layers on combined embeddings.

        Args:
            eagle_input:       dict with 'input_ids' [B, T_total] and 'attention_mask'
            inv_vision_tokens: [B, n_equi*T, d_eagle]  invariant equi tokens (inv_projector output)
            noequi_pixels:     [B*n_noequi, C, H, W]   raw non-equi camera pixels (optional)

        Returns:
            vlm_hidden: [B, T_total, d_eagle]  — LLM hidden states
            text_mask:  [B, T_total] bool      — True at text (non-image) positions
        """
        input_ids = eagle_input["input_ids"]          # [B, T_total]
        attn_mask = eagle_input["attention_mask"]     # [B, T_total]
        B = input_ids.shape[0]

        # Text embeddings
        embed_layer = self.eagle_model.get_input_embeddings()
        input_embeds = embed_layer(input_ids).clone()  # [B, T_total, d_eagle]

        # Extract non-equi camera features (SigLIP + mlp1 → d_eagle dim, no rotation)
        noequi_raw = None
        noequi_tokens_flat = None
        if noequi_pixels is not None:
            noequi_raw, _ = self.eagle_model.extract_feature(noequi_pixels)  # [B*n_noequi, T, d_eagle]
            n_noequi = len(self.non_equi_image_indices)
            T_vis = noequi_raw.shape[1]
            noequi_tokens_flat = noequi_raw.reshape(B, n_noequi * T_vis, -1)  # [B, n_noequi*T, d_eagle]

        # Inject into image-token placeholder positions
        img_tok_idx = getattr(self.eagle_model, "image_token_index", None)
        if img_tok_idx is not None:
            n_equi_tokens = inv_vision_tokens.shape[1]
            for b in range(B):
                positions = (input_ids[b] == img_tok_idx).nonzero(as_tuple=True)[0]
                # Fill equi positions first
                n_equi_fill = min(positions.shape[0], n_equi_tokens)
                if n_equi_fill > 0:
                    input_embeds[b, positions[:n_equi_fill]] = inv_vision_tokens[b, :n_equi_fill].to(input_embeds.dtype)
                # Fill non-equi positions after equi
                if noequi_tokens_flat is not None:
                    n_noequi_fill = min(positions.shape[0] - n_equi_fill, noequi_tokens_flat.shape[1])
                    if n_noequi_fill > 0:
                        input_embeds[b, positions[n_equi_fill:n_equi_fill + n_noequi_fill]] = \
                            noequi_tokens_flat[b, :n_noequi_fill].to(input_embeds.dtype)

        # Run through the LLM
        lm_output = self.eagle_model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = lm_output.hidden_states[self.select_layer]  # [B, T_total, d_eagle]

        # text_mask: True at positions that are NOT image-token placeholders
        # Used by forward_eagle to extract language-only hidden states for EquiAdapter.
        if img_tok_idx is not None:
            text_mask = (input_ids != img_tok_idx)           # [B, T_total]
        else:
            text_mask = torch.ones(B, input_ids.shape[1], dtype=torch.bool, device=input_ids.device)

        return hidden, text_mask

    # ──────────────────────────────────────────────────────────────────────────
    # Main forward through Eagle + FA + EquiAdapter
    # ──────────────────────────────────────────────────────────────────────────

    def forward_eagle(self, vl_input: BatchFeature) -> tuple:
        """
        Forward through Eagle model with frame averaging on full vision tokens,
        followed by Invariant FA projection (Phase 2) and EquiAdapter (Phase 3).

        Frame Averaging for COVARIANT equivariance:

            FA(x) = (1/|G|) * Σ_h ρ(h⁻¹) · f(h·x)

        Using ρ(h⁻¹) gives:  FA(g·x) = ρ(g) · FA(x)

        After FA:
          h_equi    = fa_equi_raw projected  [B, n_equi, T, project_to_dim]  (equivariant)
          vlm_features = eagle_linear(vlm_hidden) [B, T_total, project_to_dim]  (invariant, image+lang)
          h_adapted = equi_adapter(h_equi, vlm_features)                    (equivariant, language-conditioned)

        H_out is returned as BOTH:
          - backbone_equi_vision_features [B, n_equi, T, D] for DiT self-attention prefix
          - backbone_vision_language_features [B, n_equi*T, D] for DiT cross-attention context

        The cross-attention context is the equi_adapter output (not raw H_llm), so it
        already encodes vision equivariance AND language conditioning and can be used
        directly in DiT cross-attention without further processing.

        Args:
            vl_input: Input batch with eagle_ prefixed keys

        Returns:
            ((h_vis, inv_tokens), attention_mask) tuple
              h_vis:      [B, n_equi*T_vis, project_to_dim]          equivariant vision tokens
              inv_tokens: [B, T_rest, project_to_dim//N]             invariant scalar tokens
        """
        eagle_prefix = "eagle_"

        # Prepare eagle input dict (strip prefix, drop image_sizes)
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        eagle_input.pop("image_sizes", None)

        # ── Phase 1: FA on equivariant cameras ──────────────────────────────
        equi_pixels, noequi_pixels, img_batch, B = self.rotate_vl_batch(dict(vl_input))
        n_equi = len(self.rotate_image_indices)

        # [B*N*n_equi, T, vision_dim]
        vis_tokens_raw, _ = self.eagle_model.extract_feature(equi_pixels)
        num_vision_tokens = vis_tokens_raw.shape[1]
        vision_dim        = vis_tokens_raw.shape[2]

        # Reshape to [B*n_equi, N, T, vision_dim] — group rotations together per camera
        vis_by_rotation = (
            vis_tokens_raw
            .reshape(B, self.n_group, n_equi, num_vision_tokens, vision_dim)
            .permute(0, 2, 1, 3, 4)                           # [B, n_equi, N, T, D]
            .reshape(B * n_equi, self.n_group, num_vision_tokens, vision_dim)
        )

        # Two FA streams computed from the same rotated features in one loop:
        #
        #   Equivariant:  ρ(rot⁻¹) ⊗ π(rot⁻¹) applied before averaging → FA_equi(r·x) = ρ(r)·FA_equi(x)
        #   Invariant:    no transformation before averaging              → FA_inv(r·x)  = FA_inv(x)
        #
        # Invariance proof for plain average:
        #   FA_inv(r·x)[p] = (1/|G|) Σ_g f(g·r·x)[p]
        #                  = (1/|G|) Σ_h f(h·x)[p]   [h = g·r, same sum over G]
        #                  = FA_inv(x)[p]  ✓
        blocks = vision_dim // self.n_group
        fa_equi_accum = []
        fa_inv_accum  = []
        for rot in range(self.n_group):
            feat_rot = vis_by_rotation[:, rot]                 # [B*n_equi, T, D]
            # Invariant stream: no transformation, collect raw features
            fa_inv_accum.append(feat_rot)
            if rot == 0:
                fa_equi_accum.append(feat_rot)
            else:
                rot_inv = (self.n_group - rot) % self.n_group
                # Equivariant stream: π(rot⁻¹) + ρ(rot⁻¹)
                feat_rot_equi = torch.roll(
                    feat_rot[:, self.token_perm_indices[rot_inv]]
                    .reshape(B * n_equi, num_vision_tokens, self.n_group, blocks),
                    shifts=rot_inv, dims=2,
                ).reshape(B * n_equi, num_vision_tokens, vision_dim)
                fa_equi_accum.append(feat_rot_equi)

        # fa_equi_raw: X̃' analog — equivariant FA output [B, n_equi, T, vision_dim]
        fa_equi_raw = (
            torch.stack(fa_equi_accum, dim=1)
            .mean(dim=1)
            .reshape(B, n_equi, num_vision_tokens, vision_dim)
        )
        # fa_inv_raw: H' analog — invariant FA output [B, n_equi, T, vision_dim] (plain average)
        fa_inv_raw = (
            torch.stack(fa_inv_accum, dim=1)
            .mean(dim=1)
            .reshape(B, n_equi, num_vision_tokens, vision_dim)
        )

        # Project fa_equi_raw to project_to_dim via equivariant group-major projection.
        # Apply vision_proj to each group-slice independently (circulant equivariance).
        proj_dtype = self.vision_proj.weight.dtype
        blocks_in = vision_dim // self.n_group
        h_equi = self.vision_proj(
            fa_equi_raw.reshape(-1, self.n_group, blocks_in).to(proj_dtype)
        ).reshape(B, n_equi, num_vision_tokens, self.project_to_dim)

        # ── Phase 2: VLM pass ──────
        # fa_inv_raw is invariant → project to d_eagle and inject into frozen VLM.
        # Analogy: EquiLLM Projector(H') → LLM.
        inv_dtype = self.inv_fa_proj.weight.dtype
        h_inv_for_vlm = self.inv_fa_proj(
            fa_inv_raw.reshape(B, n_equi * num_vision_tokens, vision_dim).to(inv_dtype)
        )                                                              # [B, n_equi*T, d_eagle]
        vlm_hidden, text_mask = self._forward_llm_with_injected_vision(
            eagle_input, h_inv_for_vlm, noequi_pixels
        )

        # Project full vlm_hidden → project_to_dim
        lang_dtype = self.eagle_linear.weight.dtype
        vlm_features = self.eagle_linear(vlm_hidden.to(lang_dtype))       # [B, T_total, project_to_dim]

        # text_mask is batch-consistent (fixed template), so first sample's mask suffices.
        text_pos = text_mask[0]                                            # [T_total] bool

        # vlm_text: VLM output at text positions — formatted trivial-in-regular and appended to output.
        vlm_text = vlm_features[:, text_pos, :]                           # [B, T_lang, project_to_dim]

        # vlm_img: VLM output at equi-image positions — H^llm analog in EquiLLM.
        # Filling order in _forward_llm_with_injected_vision: equi tokens first, noequi after.
        img_pos_idx  = (~text_pos).nonzero(as_tuple=True)[0]              # all image positions
        equi_img_idx = img_pos_idx[:n_equi * num_vision_tokens]           # first n_equi*T
        vlm_img = vlm_features[:, equi_img_idx, :]                        # [B, n_equi*T, project_to_dim]

        # noequi_vlm: VLM output at non-equi image positions — invariant (not rotated).
        # Passed separately to EquiAdapter: used as invariant CA context during vision SA+CA,
        # then embedded as trivial-in-regular and joined with h_vis in the final mix SA block.
        noequi_img_idx = img_pos_idx[n_equi * num_vision_tokens:]         # remaining image positions
        noequi_vlm = (
            vlm_features[:, noequi_img_idx, :]                            # [B, n_noequi*T, project_to_dim]
            if noequi_img_idx.numel() > 0 else None
        )

        # ── Phase 3: EquiAdapter (EquiLLM-faithful) ───────────────────────────
        # SA+CA on h_equi; returns equi vision tokens and invariant scalar tokens separately.
        # h_vis:      [B, n_equi*T_vis, project_to_dim]           equivariant (regular repr)
        # inv_tokens: [B, n_noequi*T + T_lang, project_to_dim//N] invariant scalars
        h_vis, inv_tokens = self.equi_adapter(h_equi, vlm_img, vlm_text, noequi_vlm=noequi_vlm)

        return (h_vis, inv_tokens), eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """
        Forward pass with Late Frame Averaging on full vision tokens.

        Args:
            vl_input: Input batch

        Returns:
            BatchFeature with:
                - backbone_equi_vision_features: [B, n_equi*T_vision, D]
                    Equivariant vision tokens (regular repr, FA + equi CA).
                - backbone_inv_features: [B, n_noequi*T_vision + T_lang, D//N]
                    Invariant scalar tokens (non-equi cameras + language, no group expansion).
                - backbone_attention_mask: attention mask from Eagle tokeniser
        """
        self.set_frozen_modules_to_eval_mode()

        (h_vis, inv_tokens), eagle_mask = self.forward_eagle(vl_input)

        # DDP compatibility: ensure all trainable parameters participate in loss
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=h_vis.device, dtype=h_vis.dtype, requires_grad=True
            )
            for param in self.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            h_vis = h_vis + dummy_term

        return BatchFeature(data={
            "backbone_equi_vision_features": h_vis,       # [B, n_equi*T_vision, D] regular repr
            "backbone_inv_features":         inv_tokens,  # [B, T_rest, D//N] invariant scalars
            "backbone_attention_mask":       eagle_mask,
        })
