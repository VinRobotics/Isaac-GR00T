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
Eagle Backbone: N-pass Frame Averaging after VLM.

Runs the full VLM (SigLIP + LLM) once per group rotation so the language model
sees each rotated view and produces rotation-conditioned hidden states.
Frame averaging is then applied AFTER the VLM on the resulting features.

Pipeline:
  1. Pre-rotate the n_equi cameras by each g ∈ C_N (N = n_group).
  2. For each rotation r: build pixel_values_r = [rotated_equi(r), non_equi_original]
     and run VLM([pixel_values_r, text]) → hidden_r [B, T_total, d_eagle].
  3. Extract three streams from hidden_r:
       equi_vis_r  [B, n_equi, T_vis, d_eagle]  — equi camera VLM features
       lang_r      [B, T_lang, d_eagle]          — language + cross-modal features
       noequi_vis  [B, n_noequi, T_vis, d_eagle] — non-equi camera features (r=0 only,
                                                    same for all r since pixels unchanged)
  4. FA_equi on {equi_vis_r}:
       H'[p] = (1/N) Σ_r ρ(r⁻¹) π(r⁻¹) equi_vis_r[p]   → [B, n_equi, T_vis, d_eagle]
       Equivariant: H'(g·x) = ρ(g) H'(x)  ✓
  5. FA_inv on {lang_r}:
       h_lang = (1/N) Σ_r lang_r                           → [B, T_lang, d_eagle]
       Invariant: plain average over same-sum relabelling   ✓
  6. Project:
       H'       → vision_proj  (group-major equivariant linear)   → [B, n_equi, T_vis, D]
       h_lang   → eagle_linear (plain linear, invariant)          → [B, T_lang, D]
       noequi   → noequi_proj  (plain linear, invariant)          → [B, n_noequi*T_vis, D]

Output (BatchFeature):
  backbone_equi_vision_features : [B, n_equi*T_vis, D]            equivariant
  backbone_invariant_context    : [B, T_lang + n_noequi*T_vis, D]  invariant
  backbone_attention_mask       : Eagle tokeniser attention mask
"""

import os
import math
from typing import List

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
    Eagle Backbone with N-pass Frame Averaging (FA applied after VLM).

    Each group rotation gets its own full VLM forward pass so language features
    are conditioned on the actual rotated visual input.  FA is then applied to
    the VLM outputs to produce equivariant vision features and invariant language
    features.
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
        # FA parameters
        n_group: int = 8,
        num_images_per_sample: int = 1,
        rotate_image_indices: List[int] | None = None,
    ):
        """
        Args:
            project_to_dim: output feature dimension (must be divisible by n_group)
            n_group: CN group order (4 → C4, 8 → C8)
            num_images_per_sample: total number of camera views per sample
            rotate_image_indices: which image indices to apply rotations to
                                  (None = all); remaining images are non-equivariant
        """
        super().__init__()
        assert not reproject_vision, "reproject_vision not implemented"

        self.n_group = n_group
        self.num_images_per_sample = num_images_per_sample
        self.project_to_dim = project_to_dim if project_to_dim else 2048

        if rotate_image_indices is None:
            self.rotate_image_indices = list(range(num_images_per_sample))
        else:
            self.rotate_image_indices = rotate_image_indices

        self.non_equi_image_indices = [
            i for i in range(num_images_per_sample)
            if i not in self.rotate_image_indices
        ]

        assert self.project_to_dim % n_group == 0, (
            f"project_to_dim ({self.project_to_dim}) must be divisible by n_group ({n_group})"
        )

        # ── Eagle VLM ────────────────────────────────────────────────────────
        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        d_eagle = self.eagle_model.language_model.config.hidden_size

        assert d_eagle % n_group == 0, (
            f"d_eagle ({d_eagle}) must be divisible by n_group ({n_group})"
        )

        # ── Projection layers ─────────────────────────────────────────────────
        # Equivariant vision projection: applied per group-slice (group-major layout).
        # D = n_group * blocks; the same linear maps each block independently
        # so it commutes with the cyclic-shift action → equivariant.
        blocks_in  = d_eagle       // n_group
        blocks_out = self.project_to_dim // n_group
        self.vision_proj = nn.Linear(blocks_in, blocks_out)

        # Invariant language projection (trivial repr → trivial repr).
        self.eagle_linear = nn.Linear(d_eagle, self.project_to_dim)

        # Invariant non-equi camera projection (plain linear).
        # Created even when non_equi_image_indices is empty so checkpoint loading
        # is consistent; unused in forward when there are no non-equi cameras.
        self.noequi_proj = nn.Linear(d_eagle, self.project_to_dim)

        # Truncate LLM to select_layer layers for efficiency
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)
        self.select_layer = select_layer

        # ── Spatial helpers ───────────────────────────────────────────────────
        self._init_rotation_matrices()
        self._init_permutation_matrices()
        self._init_token_permutation_indices(grid_size=16)

        self.set_trainable_parameters(tune_llm, tune_visual)

        n_equi   = len(self.rotate_image_indices)
        n_noequi = len(self.non_equi_image_indices)
        print(f"EagleBackboneFATokens (N-pass FA after VLM):")
        print(f"  n_group (CN): {self.n_group}")
        print(f"  d_eagle:      {d_eagle}")
        print(f"  project_to_dim: {self.project_to_dim}")
        print(f"  equi cameras:   {self.rotate_image_indices} ({n_equi} views)")
        print(f"  non-equi cameras: {self.non_equi_image_indices} ({n_noequi} views)")
        print(f"  VLM passes per sample: {self.n_group}")
        print(f"  Token grid: {self.token_grid_size}x{self.token_grid_size}")

    # ── Spatial initialisation ────────────────────────────────────────────────

    def _init_rotation_matrices(self):
        """Affine matrices for image rotation via grid_sample (CW by angle = r*2π/N)."""
        angles = torch.linspace(0, 2 * math.pi, self.n_group + 1)[:-1]
        mats = torch.zeros(self.n_group, 2, 3)
        for i, angle in enumerate(angles):
            c = math.cos(-angle.item())
            s = math.sin(-angle.item())
            mats[i, 0, 0] =  c;  mats[i, 0, 1] = -s
            mats[i, 1, 0] =  s;  mats[i, 1, 1] =  c
        self.register_buffer("rotation_matrices_buffer", mats)
        self.register_buffer("angles", angles)

    def _init_permutation_matrices(self):
        """Cyclic permutation matrices for the regular representation."""
        P = torch.zeros(self.n_group, self.n_group, self.n_group)
        for r in range(self.n_group):
            for i in range(self.n_group):
                P[r, i, (i + r) % self.n_group] = 1.0
        self.register_buffer("permutation_matrices", P)
        self.register_buffer("perm_matrices_flat", P.reshape(self.n_group, -1))
        idx = torch.arange(self.n_group)
        self.register_buffer("indices_template", idx)
        self.register_buffer(
            "selected_perm_matrices_template",
            P.reshape(self.n_group, -1)[idx].reshape(self.n_group, self.n_group, self.n_group),
        )

    def _init_token_permutation_indices(self, grid_size: int = 16):
        """
        Token permutation indices for spatial realignment after rotation.

        perm[r][dest] = source  means: after rotating image by r steps CW,
        the token at 'dest' in the rotated grid came from 'source' in the
        original grid.  Applying the inverse (perm[n-r]) un-rotates the tokens.

        Uses greedy bijective assignment for non-90° angles (C8 odd steps).
        """
        self.token_grid_size = grid_size
        N = grid_size
        num_tokens = N * N
        center = (N - 1) / 2
        perm = torch.zeros(self.n_group, num_tokens, dtype=torch.long)

        for r in range(self.n_group):
            angle = r * 2 * math.pi / self.n_group
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            float_src_row, float_src_col = [], []
            for i in range(N):
                for j in range(N):
                    ci, ri = j - center, i - center
                    float_src_col.append(center + cos_a * ci + sin_a * ri)
                    float_src_row.append(center - sin_a * ci + cos_a * ri)

            is_exact_90 = abs(math.sin(angle) ** 2 - round(math.sin(angle) ** 2)) < 1e-9

            if is_exact_90:
                for dest in range(num_tokens):
                    si = max(0, min(N - 1, int(round(float_src_row[dest]))))
                    sj = max(0, min(N - 1, int(round(float_src_col[dest]))))
                    perm[r, dest] = si * N + sj
            else:
                err = [
                    (float_src_row[d] - round(float_src_row[d])) ** 2
                    + (float_src_col[d] - round(float_src_col[d])) ** 2
                    for d in range(num_tokens)
                ]
                dests_sorted = sorted(range(num_tokens), key=lambda d: err[d])
                assigned = [0] * num_tokens
                used: set = set()
                for dest in dests_sorted:
                    sr, sc = float_src_row[dest], float_src_col[dest]
                    best_idx, best_dist = None, float("inf")
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
                                if idx in used:
                                    continue
                                d = (sr - si) ** 2 + (sc - sj) ** 2
                                if d < best_dist:
                                    best_dist, best_idx = d, idx
                        if best_idx is not None:
                            break
                    if best_idx is None:
                        best_idx = next(i for i in range(num_tokens) if i not in used)
                    assigned[dest] = best_idx
                    used.add(best_idx)
                for dest in range(num_tokens):
                    perm[r, dest] = assigned[dest]

        # Enforce exact inverse property for greedy (non-90°) steps
        arange = torch.arange(num_tokens, dtype=torch.long)
        for r in range(1, self.n_group):
            angle = r * 2 * math.pi / self.n_group
            if not (abs(math.sin(angle) ** 2 - round(math.sin(angle) ** 2)) < 1e-9):
                r_inv = (self.n_group - r) % self.n_group
                if r_inv > r:
                    inv_perm = torch.zeros(num_tokens, dtype=torch.long)
                    inv_perm[perm[r]] = arange
                    perm[r_inv] = inv_perm

        self.register_buffer("token_perm_indices", perm)

    # ── Trainable parameter management ────────────────────────────────────────

    _NEW_LAYER_PREFIXES = (
        "vision_proj.",
        "eagle_linear.",
        "noequi_proj.",
    )

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        """Freeze VLM; train only the new projection layers."""
        self.tune_llm = False
        self.tune_visual = False
        self.eagle_model.requires_grad_(False)
        for name, p in self.named_parameters():
            p.requires_grad_(any(name.startswith(pfx) for pfx in self._NEW_LAYER_PREFIXES))
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        print(f"Backbone trainable parameters ({len(trainable)}):")
        for n in trainable:
            print(f"  {n}")
        if not trainable:
            print("  Warning: no trainable parameters found.")

    def load_pretrained_vlm(self, checkpoint_path: str) -> None:
        """Load eagle_model weights from a GR00T N1.5 checkpoint (sharded or single)."""
        import json

        def _load_shard(path):
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file  # type: ignore[import-untyped]
                return load_file(path)
            return torch.load(path, map_location="cpu")

        ckpt = checkpoint_path
        index_file = os.path.join(ckpt, "model.safetensors.index.json")
        single_sf  = os.path.join(ckpt, "model.safetensors")
        single_bin = os.path.join(ckpt, "pytorch_model.bin")

        raw: dict = {}
        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            for shard in set(index["weight_map"].values()):
                raw.update(_load_shard(os.path.join(ckpt, shard)))
        elif os.path.exists(single_sf):
            raw = _load_shard(single_sf)
        elif os.path.exists(single_bin):
            raw = _load_shard(single_bin)
        else:
            raise FileNotFoundError(f"No checkpoint weights found in {ckpt}")

        backbone_sd = {k.removeprefix("backbone."): v for k, v in raw.items() if k.startswith("backbone.")}
        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)
        new_missing = [k for k in missing if k.startswith(self._NEW_LAYER_PREFIXES)]
        vlm_missing  = [k for k in missing if not k.startswith(self._NEW_LAYER_PREFIXES)]
        print(f"Loaded backbone from {ckpt}")
        print(f"  New layers (random init): {new_missing[:5]}{'...' if len(new_missing) > 5 else ''}")
        if vlm_missing:
            print(f"  WARNING — unexpected missing VLM keys ({len(vlm_missing)}): {vlm_missing[:5]}")

    def new_layers_state_dict(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if k.startswith(self._NEW_LAYER_PREFIXES)}

    def load_new_layers_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        missing, _ = self.load_state_dict(state_dict, strict=False)
        truly_missing = [k for k in missing if k.startswith(self._NEW_LAYER_PREFIXES)]
        if truly_missing and strict:
            raise RuntimeError(f"Missing new-layer keys: {truly_missing}")

    # ── Utility methods ───────────────────────────────────────────────────────

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _apply_rotations_to_images(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        Apply all N rotations to a batch of images in one grid_sample call.

        Args:
            img_batch: [B, C, H, W]
        Returns:
            [B*N, C, H, W]  — layout: (b=0,r=0), (b=0,r=1), …, (b=B-1,r=N-1)
        """
        B, C, H, W = img_batch.shape
        N = self.n_group
        flat = img_batch.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W)
        rot_idx = torch.arange(N, device=img_batch.device).repeat(B)
        mats = self.rotation_matrices_buffer[rot_idx]
        grid = F.affine_grid(mats.to(img_batch.dtype), size=(B * N, C, H, W), align_corners=True)
        return F.grid_sample(flat, grid, align_corners=True, padding_mode="zeros")

    # ── Main forward pass ─────────────────────────────────────────────────────

    def _build_image_seq_positions(
        self,
        input_ids: torch.Tensor,
        T_vis: int,
        img_tok_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Compute sequence positions for equi tokens, non-equi tokens, and text tokens.

        All positions are computed from the first sample (input_ids[0]); the template
        is assumed identical across the batch (same prompt structure).

        Returns:
            equi_seq_pos  : [n_equi * T_vis]  — LLM sequence indices of equi image tokens
            noequi_seq_pos: [n_noequi * T_vis] or None
            text_mask     : [T_total] bool — True at language/text token positions
        """
        ids0 = input_ids[0]
        is_img = ids0 == img_tok_idx                                  # [T_total] bool
        all_img_pos = is_img.nonzero(as_tuple=True)[0]                # [n_imgs * T_vis]
        text_mask = ~is_img                                           # [T_total] bool

        equi_seq_pos = torch.cat([
            all_img_pos[idx * T_vis : (idx + 1) * T_vis]
            for idx in self.rotate_image_indices
        ])

        if self.non_equi_image_indices:
            noequi_seq_pos = torch.cat([
                all_img_pos[idx * T_vis : (idx + 1) * T_vis]
                for idx in self.non_equi_image_indices
            ])
        else:
            noequi_seq_pos = None

        return equi_seq_pos, noequi_seq_pos, text_mask

    def forward_eagle(self, vl_input: BatchFeature) -> tuple:
        """
        Batched N-pass VLM forward + FA.

        All N rotations are processed in a single SigLIP call and a single LLM call
        by expanding the batch dimension: effective batch = B × N.

        Correctness:
          - equi vision: FA_equi  (ρ(r⁻¹) π(r⁻¹) before averaging) → equivariant H'
          - language:    FA_inv   (plain average)                     → invariant
          - non-equi:    FA_inv   (plain average over all N passes)   → invariant
            Note: non-equi pixels are the same for all r, but their LLM hidden states
            differ because the LLM attends globally to the changing equi tokens.
            Averaging them over N passes (FA_inv) gives the correct invariant feature.

        Returns:
            h_prime_flat : [B, n_equi*T_vis, project_to_dim]  equivariant
            h_lang       : [B, T_lang, project_to_dim]         invariant
            h_noequi     : [B, n_noequi*T_vis, project_to_dim] or None  invariant
            attn_mask    : Eagle tokeniser attention mask
        """
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        eagle_input.pop("image_sizes", None)

        pixel_values = eagle_input.pop("pixel_values")  # [B*n_imgs, C, H, W]
        input_ids    = eagle_input["input_ids"]          # [B, T_total]
        B            = input_ids.shape[0]
        N            = self.n_group
        n_imgs       = self.num_images_per_sample
        T_total      = input_ids.shape[1]
        _, C, H, W   = pixel_values.shape
        img_batch    = pixel_values.reshape(B, n_imgs, C, H, W)

        d_eagle: int = self.eagle_model.language_model.config.hidden_size
        n_equi = len(self.rotate_image_indices)
        img_tok_idx = getattr(self.eagle_model, "image_token_index", None)

        # ── Step 1: build pixel_values for all N rotations at once ────────────
        # pv_all: [B, N, n_imgs, C, H, W] — equi cameras replaced per rotation,
        # non-equi cameras are identical across the N dimension.
        pv_all = img_batch.unsqueeze(1).expand(-1, N, -1, -1, -1, -1).clone()
        for idx in self.rotate_image_indices:
            # _apply_rotations_to_images returns [B*N, C, H, W] in (b,r) layout
            all_rots = self._apply_rotations_to_images(img_batch[:, idx])     # [B*N, C, H, W]
            pv_all[:, :, idx] = all_rots.reshape(B, N, C, H, W)
        # Flatten to [B*N*n_imgs, C, H, W] for SigLIP
        pv_all_flat = pv_all.reshape(B * N * n_imgs, C, H, W)

        # ── Step 2: one SigLIP + mlp1 call for all B×N×n_imgs images ─────────
        raw_all, _ = self.eagle_model.extract_feature(pv_all_flat)  # [B*N*n_imgs, T_vis, d_eagle]
        T_vis = raw_all.shape[1]
        assert T_vis == self.token_grid_size * self.token_grid_size, (
            f"SigLIP returned {T_vis} tokens but token_perm_indices was built for "
            f"{self.token_grid_size}×{self.token_grid_size}={self.token_grid_size**2}. "
            f"Pass the correct grid_size to _init_token_permutation_indices."
        )
        raw_all = raw_all.reshape(B * N, n_imgs, T_vis, d_eagle)     # [B*N, n_imgs, T_vis, d_eagle]

        # ── Step 3: build input_embeds for all B×N LLM passes ────────────────
        # Text embeddings are identical across all N rotations — compute once.
        embed_layer  = self.eagle_model.get_input_embeddings()
        text_embeds  = embed_layer(input_ids)                         # [B, T_total, d_eagle]
        # Expand to [B*N, T_total, d_eagle] — each of the N copies gets the same text
        input_embeds = (
            text_embeds.unsqueeze(1)
            .expand(-1, N, -1, -1)
            .reshape(B * N, T_total, d_eagle)
            .clone()                                                   # clone: we'll overwrite img positions
        )

        # Inject vision tokens. Positions are the same across the batch (fixed template).
        if img_tok_idx is not None:
            positions = (input_ids[0] == img_tok_idx).nonzero(as_tuple=True)[0]  # [n_imgs * T_vis]
            for img_idx in range(n_imgs):
                pos_slice = positions[img_idx * T_vis : (img_idx + 1) * T_vis]   # [T_vis]
                # raw_all[:, img_idx]: [B*N, T_vis, d_eagle] — vectorized over whole batch
                input_embeds[:, pos_slice, :] = raw_all[:, img_idx, :, :]

        # ── Step 4: one LLM call for all B×N passes ───────────────────────────
        attn_mask_all = (
            eagle_input["attention_mask"]
            .unsqueeze(1)
            .expand(-1, N, -1)
            .reshape(B * N, T_total)
        )
        lm_out = self.eagle_model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask_all,
            output_hidden_states=False,   # only last layer — saves memory vs storing all layers
            return_dict=True,
        )
        # last_hidden_state = output of the truncated LLM's final layer = hidden_states[select_layer]
        hidden_all = lm_out.last_hidden_state.reshape(B, N, T_total, d_eagle)

        # ── Step 5: extract token streams ────────────────────────────────────
        if img_tok_idx is not None:
            equi_seq_pos, noequi_seq_pos, text_mask = self._build_image_seq_positions(
                input_ids, T_vis, img_tok_idx
            )
        else:
            text_mask    = torch.ones(T_total, dtype=torch.bool, device=input_ids.device)
            equi_seq_pos = noequi_seq_pos = None

        # equi_vis_all:  [B, N, n_equi, T_vis, d_eagle]
        if equi_seq_pos is not None:
            equi_vis_all = (
                hidden_all[:, :, equi_seq_pos, :]
                .reshape(B, N, n_equi, T_vis, d_eagle)
            )
        else:
            # Fallback: treat all tokens as equi (no image index knowledge)
            equi_vis_all = hidden_all[:, :, :, :].reshape(B, N, 1, T_total, d_eagle)

        # lang_all: [B, N, T_lang, d_eagle]
        lang_all = hidden_all[:, :, text_mask, :]

        # noequi_all: [B, N, n_noequi*T_vis, d_eagle] — average over N (FA_inv)
        # Non-equi pixels are the same for every rotation but the LLM hidden states
        # differ (global attention sees different equi tokens each pass).
        # FA_inv (plain average) gives the correct invariant representation.
        noequi_all = (
            hidden_all[:, :, noequi_seq_pos, :]
            if noequi_seq_pos is not None else None
        )

        # ── Step 6: FA_equi on equivariant VLM vision features ───────────────
        # H'[p] = (1/N) Σ_r  ρ(r⁻¹) · π(r⁻¹) · equi_vis_all[:,r,:,p,:]
        # ρ(r⁻¹): roll group-blocks by r⁻¹  (regular repr cyclic shift)
        # π(r⁻¹): inverse token spatial permutation (undoes image rotation in token grid)
        blocks = d_eagle // N
        BN_eq  = B * n_equi
        # [B, N, n_equi, T_vis, D] → process rotation axis explicitly
        feats = equi_vis_all.permute(1, 0, 2, 3, 4)  # [N, B, n_equi, T_vis, D]
        transformed = torch.empty_like(feats)
        transformed[0] = feats[0]
        for h in range(1, N):
            h_inv    = (N - h) % N
            feat_h   = feats[h].reshape(BN_eq, T_vis, d_eagle)
            # π(h⁻¹): gather tokens from inverse-permuted positions
            feat_perm  = feat_h[:, self.token_perm_indices[h_inv], :]       # [BN_eq, T, D]
            # ρ(h⁻¹): cyclic roll on the n_group block dimension
            transformed[h] = torch.roll(
                feat_perm.reshape(BN_eq, T_vis, N, blocks),
                shifts=h_inv, dims=2,
            ).reshape(B, n_equi, T_vis, d_eagle)

        h_prime_raw = transformed.mean(dim=0).reshape(B, n_equi, T_vis, d_eagle)

        # ── Step 7: FA_inv on language + non-equi (plain average) ────────────
        h_lang_raw   = lang_all.mean(dim=1)                           # [B, T_lang, d_eagle]
        h_noequi_raw = noequi_all.mean(dim=1) if noequi_all is not None else None

        # ── Step 8: project to project_to_dim ────────────────────────────────
        proj_dt   = self.vision_proj.weight.dtype
        blocks_in = d_eagle // N
        # Equivariant projection: same linear on each of the N group-slices
        h_prime = self.vision_proj(
            h_prime_raw.reshape(-1, N, blocks_in).to(proj_dt)
        ).reshape(B, n_equi, T_vis, self.project_to_dim)
        h_prime_flat = h_prime.reshape(B, n_equi * T_vis, self.project_to_dim)

        lang_dt = self.eagle_linear.weight.dtype
        h_lang  = self.eagle_linear(h_lang_raw.to(lang_dt))           # [B, T_lang, D]

        h_noequi = None
        if h_noequi_raw is not None:
            h_noequi = self.noequi_proj(h_noequi_raw.to(lang_dt))     # [B, n_noequi*T_vis, D]

        return h_prime_flat, h_lang, h_noequi, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """
        Forward pass.

        Returns BatchFeature with:
          backbone_equi_vision_features : [B, n_equi*T_vis, D]
              Equivariant vision tokens (regular representation, group-major).
              Use as equivariant prefix in DiT self-attention.

          backbone_invariant_context    : [B, T_lang + n_noequi*T_vis, D]
              Concatenation of invariant language features and non-equi camera features.
              Use as cross-attention context in DiT (language and non-equi are both trivial repr).

          backbone_attention_mask       : Eagle tokeniser attention mask.
        """
        self.set_frozen_modules_to_eval_mode()

        h_prime_flat, h_lang, h_noequi, attn_mask = self.forward_eagle(vl_input)

        # Invariant context: cat language + non-equi along the sequence dimension
        if h_noequi is not None:
            invariant_ctx = torch.cat([h_lang, h_noequi], dim=1)  # [B, T_lang + n_noequi*T, D]
        else:
            invariant_ctx = h_lang                                  # [B, T_lang, D]

        # DDP: ensure all trainable parameters participate in loss
        if self.training and self.tune_visual:
            dummy = torch.tensor(0.0, device=h_prime_flat.device, dtype=h_prime_flat.dtype,
                                 requires_grad=True)
            for p in self.parameters():
                if p.requires_grad:
                    dummy = dummy + 0.0 * p.sum()
            h_prime_flat = h_prime_flat + dummy
            invariant_ctx = invariant_ctx + dummy

        return BatchFeature(data={
            "backbone_equi_vision_features": h_prime_flat,   # [B, n_equi*T_vis, D]
            "backbone_invariant_context":    invariant_ctx,  # [B, T_lang + n_noequi*T, D]
            "backbone_attention_mask":       attn_mask,
        })
