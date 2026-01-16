# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Improved Equivariant Modules for Large-Scale Training

Key improvements:
1. Deeper encoders/decoders with residual connections
2. Layer normalization for stable training
3. Gated mechanisms for better gradient flow
4. Multi-scale feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup
from typing import Optional, List

from gr00t.model.action_head.action_encoder import SinusoidalPositionalEncoding, swish
from gr00t.model.action_head.equivariant_cross_attention_dit import EquivariantLayerNorm


class EquivariantResidualBlock(nn.Module):
    """
    Equivariant residual block with optional layer normalization.
    Improves gradient flow and training stability on large datasets.
    """
    def __init__(
        self, 
        in_type: enn.FieldType, 
        hidden_type: enn.FieldType,
        out_type: enn.FieldType,
        use_layer_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.hidden_type = hidden_type
        
        # Pre-norm architecture (more stable for deep networks)
        self.norm1 = EquivariantLayerNorm(in_type, affine=False) if use_layer_norm else None
        self.fc1 = enn.Linear(in_type, hidden_type, bias=True)
        self.act = enn.ReLU(hidden_type)
        
        self.norm2 = EquivariantLayerNorm(hidden_type, affine=False) if use_layer_norm else None
        self.fc2 = enn.Linear(hidden_type, out_type, bias=True)
        
        self.dropout = enn.PointwiseDropout(out_type, p=dropout) if dropout > 0 else None
        
        # Skip connection (identity if same type, otherwise projection)
        if in_type.size != out_type.size:
            self.skip = enn.Linear(in_type, out_type, bias=False)
        else:
            self.skip = None
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        residual = x
        
        # Pre-norm
        if self.norm1 is not None:
            x = self.norm1(x)
        
        x = self.fc1(x)
        x = self.act(x)
        
        if self.norm2 is not None:
            x = self.norm2(x)
        
        x = self.fc2(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Skip connection
        if self.skip is not None:
            residual = self.skip(residual)
        
        # Add residual
        out_tensor = x.tensor + residual.tensor
        return enn.GeometricTensor(out_tensor, self.out_type)


class EquivariantGatedMLP(nn.Module):
    """
    Gated MLP with equivariance - uses gating mechanism for better expressivity.
    Similar to SwiGLU but with full equivariance.
    """
    def __init__(
        self, 
        in_type: enn.FieldType,
        hidden_type: enn.FieldType,
        out_type: enn.FieldType,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        
        # Main path
        self.fc_main = enn.Linear(in_type, hidden_type, bias=True)
        # Gate path
        self.fc_gate = enn.Linear(in_type, hidden_type, bias=True)
        # Output projection
        self.fc_out = enn.Linear(hidden_type, out_type, bias=True)
        
        self.norm = EquivariantLayerNorm(hidden_type, affine=False)
        self.dropout = enn.PointwiseDropout(out_type, p=dropout) if dropout > 0 else None
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        main = self.fc_main(x)
        gate = self.fc_gate(x)
        
        # Gating with sigmoid (applied element-wise, preserves equivariance)
        gated = enn.GeometricTensor(
            main.tensor * torch.sigmoid(gate.tensor),
            main.type
        )
        
        gated = self.norm(gated)
        out = self.fc_out(gated)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class DeepEquiCategorySpecificMLP(nn.Module):
    """
    Deeper category-specific MLP with:
    - Multiple residual blocks
    - Layer normalization
    - Optional gating
    - Skip connections from input to output
    
    This is more suitable for large-scale training than the 2-layer version.
    """
    def __init__(
        self, 
        num_categories: int,
        in_type: enn.FieldType,
        hidden_type: enn.FieldType,
        out_type: enn.FieldType,
        num_layers: int = 3,  # Deeper than original 2-layer
        use_gating: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.num_categories = num_categories
        self.num_layers = num_layers
        
        # Build layers for each category
        self.category_layers = nn.ModuleList()
        
        for _ in range(num_categories):
            layers = nn.ModuleList()
            
            # Input projection
            layers.append(enn.Linear(in_type, hidden_type, bias=True))
            
            # Intermediate residual blocks
            for i in range(num_layers - 2):
                if use_gating:
                    layers.append(EquivariantGatedMLP(
                        hidden_type, hidden_type, hidden_type, dropout=dropout
                    ))
                else:
                    layers.append(EquivariantResidualBlock(
                        hidden_type, hidden_type, hidden_type, 
                        use_layer_norm=True, dropout=dropout
                    ))
            
            # Output projection
            layers.append(enn.Linear(hidden_type, out_type, bias=True))
            
            self.category_layers.append(layers)
        
        # Normalization
        self.input_norm = EquivariantLayerNorm(in_type, affine=False)
        self.hidden_norm = EquivariantLayerNorm(hidden_type, affine=False)
        self.output_norm = EquivariantLayerNorm(out_type, affine=False)
        
        # Input-to-output skip connection (if dimensions match)
        if in_type.size == out_type.size:
            self.input_skip = None  # Use identity
        else:
            self.input_skip = enn.Linear(in_type, out_type, bias=False)

    def _group_indices(self, cat_ids):
        unique = torch.unique(cat_ids)
        groups = {}
        for c in unique:
            idx = (cat_ids == c).nonzero(as_tuple=False).view(-1)
            groups[int(c)] = idx
        return groups

    def forward(self, x: enn.GeometricTensor, cat_ids) -> enn.GeometricTensor:
        B = x.tensor.shape[0]
        device = x.tensor.device
        
        # Normalize input
        x_normed = self.input_norm(x)
        
        all_outputs = []
        all_indices = []
        groups = self._group_indices(cat_ids)
        
        for cat, idx in groups.items():
            xb = x_normed.tensor[idx]
            x_skip = x.tensor[idx]  # For skip connection
            
            geom_xb = enn.GeometricTensor(xb, self.in_type)
            layers = self.category_layers[cat]
            
            # Forward through layers
            # Input projection
            h = layers[0](geom_xb)
            h = enn.GeometricTensor(F.relu(h.tensor), h.type)
            
            # Intermediate blocks
            for layer in layers[1:-1]:
                h = layer(h)
            
            # Normalize before output
            h = self.hidden_norm(h)
            
            # Output projection
            out = layers[-1](h)
            
            # Add skip connection
            if self.input_skip is not None:
                skip_geo = enn.GeometricTensor(x_skip, self.in_type)
                skip_out = self.input_skip(skip_geo)
                out = enn.GeometricTensor(out.tensor + skip_out.tensor * 0.1, out.type)
            elif self.in_type.size == self.out_type.size:
                out = enn.GeometricTensor(out.tensor + x_skip * 0.1, out.type)
            
            all_outputs.append(out.tensor)
            all_indices.append(idx)
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        
        inverse_perm = torch.argsort(all_indices)
        out_tensor = all_outputs[inverse_perm]
        
        # Final normalization
        out_geo = enn.GeometricTensor(out_tensor, self.out_type)
        out_geo = self.output_norm(out_geo)
        
        return out_geo


class ImprovedMultiEmbodimentActionEncoder(nn.Module):
    """
    Improved action encoder with:
    - Deeper architecture (3 layers instead of implicit 2)
    - Better timestep conditioning using FiLM (Feature-wise Linear Modulation)
    - Residual connections
    - Layer normalization
    """
    def __init__(
        self, 
        in_type: enn.FieldType,
        out_type: enn.FieldType,
        num_embodiments: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.num_embodiments = num_embodiments
        
        gspace = out_type.gspace
        
        # Timestep embedding (trivial - doesn't rotate)
        self.time_type = enn.FieldType(gspace, [gspace.trivial_repr] * out_type.size)
        
        # Deeper action embedding
        self.action_embed = DeepEquiCategorySpecificMLP(
            num_categories=num_embodiments,
            in_type=in_type,
            hidden_type=out_type,  
            out_type=out_type,
            num_layers=num_layers,
            use_gating=True,
            dropout=dropout,
        )
        
        # FiLM conditioning from timestep
        # Use standard linear because timestep is trivial/invariant
        self.pos_encoding = SinusoidalPositionalEncoding(out_type.size)
        self.film_scale = nn.Linear(out_type.size, out_type.size)
        self.film_shift = nn.Linear(out_type.size, out_type.size)
        
        # Initialize FiLM to near-identity
        nn.init.ones_(self.film_scale.weight.data.fill_diagonal_(1.0))
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)
        
        # Final projection
        self.final_proj = enn.Linear(out_type, out_type, bias=True)
        self.final_norm = EquivariantLayerNorm(out_type, affine=False)

    def forward(self, actions: enn.GeometricTensor, timesteps: torch.Tensor, cat_ids: torch.Tensor):
        """
        actions: GeometricTensor (B*T, action_dim)
        timesteps: (B,) diffusion timestep
        cat_ids: (B*T,) category IDs
        """
        B = actions.tensor.shape[0]
        
        # Expand timesteps
        timesteps = timesteps.repeat(B // timesteps.shape[0])
        
        # Embed actions (deep equivariant MLP)
        action_features = self.action_embed(actions, cat_ids)
        
        # Get timestep embedding
        tau_emb = self.pos_encoding(timesteps).to(dtype=action_features.tensor.dtype)
        
        # FiLM conditioning: scale and shift based on timestep
        # This is equivariant because scale/shift are element-wise
        scale = self.film_scale(tau_emb)
        shift = self.film_shift(tau_emb)
        
        # Apply FiLM: x * (1 + scale) + shift
        conditioned = action_features.tensor * (1 + scale) + shift
        conditioned_geo = enn.GeometricTensor(conditioned, self.out_type)
        
        # Final projection
        out = self.final_proj(conditioned_geo)
        out = self.final_norm(out)
        
        return out


class ImprovedActionDecoder(nn.Module):
    """
    Improved action decoder with:
    - Multi-scale feature fusion
    - Residual refinement
    - Layer normalization
    """
    def __init__(
        self,
        num_categories: int,
        in_type: enn.FieldType,
        hidden_type: enn.FieldType,
        out_type: enn.FieldType,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Deep decoder
        self.decoder = DeepEquiCategorySpecificMLP(
            num_categories=num_categories,
            in_type=in_type,
            hidden_type=hidden_type,
            out_type=out_type,
            num_layers=num_layers,
            use_gating=True,
            dropout=dropout,
        )
        
        # Refinement head (additional 1-layer refinement)
        self.refine = enn.Linear(out_type, out_type, bias=True)
        self.refine_norm = EquivariantLayerNorm(out_type, affine=False)
    
    def forward(self, x: enn.GeometricTensor, cat_ids) -> enn.GeometricTensor:
        # Main decoding
        decoded = self.decoder(x, cat_ids)
        
        # Refinement with residual
        refined = self.refine(decoded)
        refined = self.refine_norm(refined)
        
        # Residual connection
        out_tensor = decoded.tensor + refined.tensor * 0.1
        return enn.GeometricTensor(out_tensor, self.decoder.out_type)


class StateFeatureFusion(nn.Module):
    """
    Fuses state features with VL (vision-language) features before the DiT.
    Uses cross-attention to allow state to attend to relevant VL context.
    """
    def __init__(
        self,
        state_type: enn.FieldType,
        vl_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_type = state_type
        self.vl_dim = vl_dim
        
        gspace = state_type.gspace
        
        # VL features are treated as trivial (invariant)
        self.vl_type = enn.FieldType(gspace, [gspace.trivial_repr] * vl_dim)
        
        # Project VL to match state dimension for attention
        self.vl_proj = nn.Linear(vl_dim, state_type.size)
        
        # Cross-attention (state attends to VL)
        # Use standard attention since VL is trivial
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=state_type.size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm = EquivariantLayerNorm(state_type, affine=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        state_features: enn.GeometricTensor,  # (B*T, D)
        vl_features: torch.Tensor,  # (B, S, vl_dim)
        batch_size: int,
        seq_len: int,
    ) -> enn.GeometricTensor:
        """
        Fuse state features with VL context.
        """
        # Reshape state for attention: (B*T, D) -> (B, T, D)
        state = einops.rearrange(state_features.tensor, "(b t) d -> b t d", b=batch_size, t=seq_len)
        
        # Project VL features
        vl_proj = self.vl_proj(vl_features)  # (B, S, D)
        
        # Cross-attention: state queries VL
        attn_out, _ = self.cross_attn(
            query=state,
            key=vl_proj,
            value=vl_proj,
        )
        
        # Dropout and residual
        attn_out = self.dropout(attn_out)
        fused = state + attn_out
        
        # Reshape back and normalize
        fused_flat = einops.rearrange(fused, "b t d -> (b t) d")
        fused_geo = enn.GeometricTensor(fused_flat, self.state_type)
        fused_geo = self.norm(fused_geo)
        
        return fused_geo
