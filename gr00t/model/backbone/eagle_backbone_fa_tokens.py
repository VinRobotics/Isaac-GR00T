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

Pipeline (with equivariant adapter):
    1. FA wrapper (Phase 1): Rotate images N times, extract features, apply FA → H' (equivariant)
    2. InvariantProjector (Phase 2): Norm-pool H' over group channels → invariant scalars
       Optionally inject these into Eagle LLM instead of raw SigLIP tokens.
    3. EquiAdapter (Phase 3): H_llm gates H' via zero-init linear → equivariant refined features
    4. Output two contexts for DiT's two Equi Cross Attention blocks:
       - equi_adapter(H', H_llm)  [equivariant] → Equi Cross Attention #1
       - H_llm                    [invariant]   → Equi Cross Attention #2
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


class InvariantProjector(nn.Module):
    """
    Phase 2: Extract invariant scalars from equivariant vision features H'.

    FA uses group-major layout: D = n_group * blocks, where channels
    [g*blocks : (g+1)*blocks] all belong to group element g.

    Invariant pooling: max over the n_group dimension (group-major axis).
    max_g( (ρ(g_r) · v)[g] ) = max_g( v[(g - r) mod N] ) = max_g( v[g] ) ✓

    Result: blocks invariant scalars per token.  MLP projects blocks → out_dim.

    forward: [..., D] → [..., out_dim]
    """

    def __init__(self, in_dim: int, out_dim: int, n_group: int):
        super().__init__()
        self.n_group = n_group
        blocks = in_dim // n_group
        self.mlp = nn.Sequential(
            nn.LayerNorm(blocks),
            nn.Linear(blocks, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h_prime: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prime: [..., D]  equivariant features (group-major, D = n_group * blocks)
        Returns:
            [..., out_dim]  invariant features
        """
        *shape, D = h_prime.shape
        blocks = D // self.n_group
        # Group-major max pool: reshape → [..., n_group, blocks], max over n_group dim
        inv = h_prime.reshape(*shape, self.n_group, blocks).max(dim=-2).values  # [..., blocks]
        return self.mlp(inv)


class EquiAdapter(nn.Module):
    """
    Phase 3: Fuse H_llm (invariant language context) with H' (equivariant vision carrier).

    FA uses group-major layout: D = n_group * blocks.
    The group action ρ(g_r) cyclically shifts the n_group chunks of size `blocks`.

    Equivariant linear (group-major): apply the SAME nn.Linear to each of the n_group
    group-slices independently.  This commutes with the group-major cyclic shift:
        W_equi(ρ(g_r) · v)[g] = W_block(v[(g-r) mod N]) = (ρ(g_r) · W_equi(v))[g]  ✓

    Gate design — per-token cross-attention (invariant):
        q  = q_proj(GroupPool(h_prime))          # [B, n_equi, T, blocks]  invariant query
        k,v = k_proj(H_llm), v_proj(H_llm)      # [B, T_text, blocks]     invariant keys/values
        ctx = softmax(q @ k^T / √d) @ v          # [B, n_equi, T, blocks]  per-token language ctx
        gate = sigmoid(ctx)                       # invariant gate per vision token ✓

    Each vision token attends to the full task instruction independently, allowing
    task-relevant spatial regions to be selectively modulated.

    Zero-init of v_proj (bias = 0, weight = 0) → gate ≈ 0.5 → near-identity at init ✓
    """

    def __init__(self, d_eq: int, d_llm: int, n_group: int):
        super().__init__()
        self.n_group = n_group
        self.blocks = d_eq // n_group
        self.scale = self.blocks ** -0.5

        # Invariant query from each vision token (GroupPool → blocks scalars)
        self.q_proj = nn.Linear(self.blocks, self.blocks)
        # Language keys and values
        self.k_proj = nn.Linear(d_llm, self.blocks)
        self.v_proj = nn.Linear(d_llm, self.blocks)
        # Zero-init v_proj so gate starts at sigmoid(0) = 0.5 → near-identity residual
        nn.init.zeros_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)

        # FiLM: per-token scale (γ) and shift (β) from cross-attention context
        # Both derived from invariant ctx → equivariance preserved ✓
        # Zero-init so γ=1, β=0 at init → identity residual
        self.film_gamma = nn.Linear(self.blocks, self.blocks)
        self.film_beta  = nn.Linear(self.blocks, self.blocks)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)   # γ starts at 1
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)   # β starts at 0

        # Equivariant linear in group-major: same linear applied to each group-slice
        self.equi_proj = nn.Linear(self.blocks, self.blocks, bias=False)

    def forward(self, h_prime: torch.Tensor, h_llm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prime: [B, n_equi, T, D_eq]  equivariant vision features (group-major)
            h_llm:   [B, T_text, D_llm]    invariant LLM features (task instruction)
        Returns:
            [B, n_equi, T, D_eq]  language-conditioned equivariant features
        """
        B, n_equi, T, D = h_prime.shape
        proj_dtype = self.equi_proj.weight.dtype
        h = h_prime.to(proj_dtype)

        # ── Equivariant transform ────────────────────────────────────────────
        h_slices = h.reshape(B, n_equi, T, self.n_group, self.blocks)
        equi_out = self.equi_proj(h_slices).reshape(B, n_equi, T, D)

        # ── Invariant query: GroupPool over n_group dim → per-token scalar ───
        # max over group elements is invariant to cyclic feature shift
        inv_feat = h_slices.max(dim=-2).values           # [B, n_equi, T, blocks]
        q = self.q_proj(inv_feat)                        # [B, n_equi, T, blocks]

        # ── Cross-attention to full task instruction ─────────────────────────
        h_llm_proj = h_llm.to(proj_dtype)
        k = self.k_proj(h_llm_proj)                      # [B, T_text, blocks]
        v = self.v_proj(h_llm_proj)                      # [B, T_text, blocks]

        # attn: [B, n_equi, T, T_text] — invariant (q,k,v all invariant)
        attn = torch.softmax(
            torch.einsum('bntd, bsd -> bnts', q, k) * self.scale, dim=-1
        )
        ctx = torch.einsum('bnts, bsd -> bntd', attn, v)  # [B, n_equi, T, blocks]

        # ── FiLM + gate from per-token language context ──────────────────────
        # All derived from invariant ctx → equivariance preserved ✓
        gate  = torch.sigmoid(ctx)                        # [B, n_equi, T, blocks]
        gamma = self.film_gamma(ctx)                      # [B, n_equi, T, blocks]  scale
        beta  = self.film_beta(ctx)                       # [B, n_equi, T, blocks]  shift

        def expand_to_group_major(x):
            # [B, n_equi, T, blocks] → [B, n_equi, T, D] (group-major)
            return x[:, :, :, None, :].expand(B, n_equi, T, self.n_group, self.blocks) \
                                      .reshape(B, n_equi, T, D)

        gate_exp  = expand_to_group_major(gate)
        gamma_exp = expand_to_group_major(gamma)
        beta_exp  = expand_to_group_major(beta)

        # gamma * equi_out: equivariant (invariant scale × equivariant)  ✓
        # beta:             invariant additive shift (same value per block across all groups) ✓
        out = gamma_exp * equi_out + beta_exp
        return h_prime + (gate_exp * out).to(h_prime.dtype)  # residual


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

        # ── Phase 2: InvariantProjector ─────────────────────────────────────
        # Maps H' (equivariant, project_to_dim) → invariant tokens (d_eagle)
        # for injection into the Eagle LLM in place of SigLIP tokens.
        self.inv_projector = InvariantProjector(
            in_dim=self.project_to_dim,
            out_dim=d_eagle,
            n_group=n_group,
        )

        # ── Phase 3: EquiAdapter ─────────────────────────────────────────────
        # Fuses H_llm (invariant gate) with H' (equivariant carrier).
        # d_llm for the gate is project_to_dim because vl_features are already projected.
        # If non-equi cameras are present, also gate with their pooled features (same dim).
        self.equi_adapter = EquiAdapter(
            d_eq=self.project_to_dim,
            d_llm=self.project_to_dim,
            n_group=n_group,
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

    # ── New layers added on top of the pretrained VLM ────────────────────────
    # These are the only parameters that should be trained / saved / loaded.
    _NEW_LAYER_PREFIXES = (
        "vision_proj.",
        "eagle_linear.",
        "inv_projector.",
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
            padding_mode='zeros'
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
    ) -> torch.Tensor:
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
            hidden_states: [B, T_total, d_eagle]
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
                    input_embeds[b, positions[:n_equi_fill]] = inv_vision_tokens[b, :n_equi_fill]
                # Fill non-equi positions after equi
                if noequi_tokens_flat is not None:
                    n_noequi_fill = min(positions.shape[0] - n_equi_fill, noequi_tokens_flat.shape[1])
                    if n_noequi_fill > 0:
                        input_embeds[b, positions[n_equi_fill:n_equi_fill + n_noequi_fill]] = \
                            noequi_tokens_flat[b, :n_noequi_fill]

        # Run through the LLM
        lm_output = self.eagle_model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return lm_output.hidden_states[self.select_layer]  # [B, T_total, d_eagle]

    # ──────────────────────────────────────────────────────────────────────────
    # Main forward through Eagle + FA + EquiAdapter
    # ──────────────────────────────────────────────────────────────────────────

    def forward_eagle(self, vl_input: BatchFeature) -> tuple:
        """
        Forward through Eagle model with frame averaging on full vision tokens,
        followed by InvariantProjector (Phase 2) and EquiAdapter (Phase 3).

        Frame Averaging for COVARIANT equivariance:

            FA(x) = (1/|G|) * Σ_h ρ(h⁻¹) · f(h·x)

        Using ρ(h⁻¹) gives:  FA(g·x) = ρ(g) · FA(x)

        After FA:
          H'    = avg_vision_features  [B, n_equi, T, project_to_dim]  (equivariant)
          H_llm = vl_features          [B, T_text, project_to_dim]      (invariant)
          H_out = equi_adapter(H', H_llm)                               (equivariant, language-conditioned)

        H_out is returned as BOTH:
          - backbone_equi_vision_features [B, n_equi, T, D] for DiT self-attention prefix
          - backbone_vision_language_features [B, n_equi*T, D] for DiT cross-attention context

        The cross-attention context is the equi_adapter output (not raw H_llm), so it
        already encodes vision equivariance AND language conditioning and can be used
        directly in DiT cross-attention without further processing.

        Args:
            vl_input: Input batch with eagle_ prefixed keys

        Returns:
            (h_prime, h_prime_flat, attention_mask) tuple
              h_prime:      [B, n_equi, T, project_to_dim] — equivariant adapted features
              h_prime_flat: [B, n_equi*T, project_to_dim]  — same, flattened for cross-attention
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
        equi_raw, _ = self.eagle_model.extract_feature(equi_pixels)
        num_vision_tokens = equi_raw.shape[1]
        vision_dim        = equi_raw.shape[2]

        # Reshape to [B*n_equi, N, T, vision_dim]
        equi_grouped = (
            equi_raw
            .reshape(B, self.n_group, n_equi, num_vision_tokens, vision_dim)
            .permute(0, 2, 1, 3, 4)                           # [B, n_equi, N, T, D]
            .reshape(B * n_equi, self.n_group, num_vision_tokens, vision_dim)
        )

        # Apply ρ(h⁻¹) = spatial_perm(h⁻¹) ⊗ feature_perm(h⁻¹), then average
        blocks = vision_dim // self.n_group
        transformed_features = []
        for h in range(self.n_group):
            feat_h = equi_grouped[:, h]                        # [B*n_equi, T, D]
            if h == 0:
                transformed_features.append(feat_h)
            else:
                h_inv = (self.n_group - h) % self.n_group
                feat_h = feat_h[:, self.token_perm_indices[h_inv]]
                feat_h = torch.roll(
                    feat_h.reshape(B * n_equi, num_vision_tokens, self.n_group, blocks),
                    shifts=h_inv, dims=2,
                ).reshape(B * n_equi, num_vision_tokens, vision_dim)
                transformed_features.append(feat_h)

        # H' = [B, n_equi, T, vision_dim]
        h_prime_raw = (
            torch.stack(transformed_features, dim=1)
            .mean(dim=1)
            .reshape(B, n_equi, num_vision_tokens, vision_dim)
        )

        # Project H' to project_to_dim via equivariant group-major projection:
        # apply vision_proj to each of the n_group group-slices independently.
        proj_dtype = self.vision_proj.weight.dtype
        blocks_in = vision_dim // self.n_group
        h_prime = self.vision_proj(
            h_prime_raw.reshape(-1, self.n_group, blocks_in).to(proj_dtype)
        ).reshape(B, n_equi, num_vision_tokens, self.project_to_dim)

        # ── Phase 2: VLM pass ──────
        # Equi: inv_projector(H') tokens; non-equi pixels injected at remaining positions.
        # vl_features encodes both camera views via the LLM.
        inv_tokens = self.inv_projector(h_prime)                      # [B, n_equi, T, d_eagle]
        inv_tokens_flat = inv_tokens.reshape(B, n_equi * num_vision_tokens, -1)
        vl_hidden = self._forward_llm_with_injected_vision(
            eagle_input, inv_tokens_flat, noequi_pixels
        )
        
        # Project LLM hidden states → project_to_dim (language is invariant, plain linear is fine)
        lang_dtype = self.eagle_linear.weight.dtype
        vl_features = self.eagle_linear(vl_hidden.to(lang_dtype))         # [B, T_text, project_to_dim]

        # ── Phase 3: EquiAdapter ──────────────────────────────────────────────
        # vl_features gates h_prime; wrist info is already inside vl_features.
        h_prime = self.equi_adapter(h_prime, vl_features)

        return h_prime, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """
        Forward pass with Late Frame Averaging on full vision tokens.

        Args:
            vl_input: Input batch

        Returns:
            BatchFeature with:
                - backbone_equi_vision_features:    [B, n_equi, T_vision, D]
                    Equivariant adapted features for DiT self-attention prefix.
                - backbone_vision_language_features: [B, n_equi*T_vision, D]
                    Same features flattened — use directly as DiT cross-attention context
                    (language-conditioned equivariant, no further processing needed).
                - backbone_attention_mask: attention mask from Eagle tokeniser
        """
        self.set_frozen_modules_to_eval_mode()

        h_prime, eagle_mask = self.forward_eagle(vl_input)

        # DDP compatibility: ensure all trainable parameters participate in loss
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=h_prime.device, dtype=h_prime.dtype, requires_grad=True
            )
            for param in self.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            h_prime = h_prime + dummy_term

        return BatchFeature(data={
            "backbone_equi_vision_features": h_prime,   # [B, n_equi, T_vision, D]
            "backbone_attention_mask": eagle_mask,
        })
