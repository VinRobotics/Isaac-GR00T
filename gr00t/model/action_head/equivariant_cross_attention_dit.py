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

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    Timesteps,
    TimestepEmbedding,
)
from torch import nn

import torch
import torch.nn as nn
import einops
import escnn.nn as enn
from escnn.group import CyclicGroup
import escnn
from typing import Optional
from .equivariant_activation import (
    EquivariantGeLU,
    EquivariantSiLU,
    ApproximateGELU,
    EquivariantGEGLU,
    EquivariantSwiGLU,
)


class EquivariantLayerNorm(nn.Module):
    """
    Equivariant Layer Normalization that rearranges (B*T, D*G) to (B*T, G, D),
    applies layer norm, then converts back to (B*T, D*G).
    """
    def __init__(self, field_type: enn.FieldType, affine=False, eps=1e-5):
        super().__init__()
        self.field_type = field_type
        self.eps = eps
        self.affine = affine
        
        # Get the group size
        self.group_size = field_type.gspace.fibergroup.order()
        
        # Get the total dimension (D * G)
        self.total_dim = field_type.size
        
        # Calculate D (dimension per group element)
        assert self.total_dim % self.group_size == 0, \
            f"Total dimension {self.total_dim} must be divisible by group size {self.group_size}"
        self.dim_per_group = self.total_dim // self.group_size
        
        # LayerNorm is applied on the last dimension (D)
        if self.affine:
            self.layer_norm = nn.LayerNorm(self.dim_per_group, eps=eps, elementwise_affine=True)
        else:
            self.layer_norm = nn.LayerNorm(self.dim_per_group, eps=eps, elementwise_affine=False)
    
    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        """
        Args:
            x: GeometricTensor with shape (B*T, D*G)
        
        Returns:
            GeometricTensor with shape (B*T, D*G)
        """
        # Get the tensor from GeometricTensor
        tensor = x.tensor  # Shape: (B*T, D*G)
        batch_size = tensor.shape[0]
        
        # Rearrange from (B*T, D*G) to (B*T, G, D)
        tensor_reshaped = tensor.view(batch_size, self.group_size, self.dim_per_group)
        
        # Apply layer norm on the last dimension (D)
        tensor_normed = self.layer_norm(tensor_reshaped)
        
        # Rearrange back from (B*T, G, D) to (B*T, D*G)
        tensor_output = tensor_normed.view(batch_size, self.total_dim)
        
        # Return as GeometricTensor
        return enn.GeometricTensor(tensor_output, self.field_type)

class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb

class EquivariantAdaLayerNorm(nn.Module):
    """
    Equivariant Adaptive Layer Normalization.
    
    Uses standard PyTorch linear layers for scale/shift projection (faster convergence)
    because temb is trivial (invariant) and scale/shift are applied element-wise.
    """
    def __init__(self, token_type: enn.FieldType, temb_type: Optional[enn.FieldType] = None, eps=1e-5):
        super().__init__()

        self.token_type = token_type
        # temb_type is trivial repr for timestep (scalar, not rotated)
        self.temb_type = temb_type if temb_type is not None else token_type
        
        self.token_dim = token_type.size
        self.temb_dim = self.temb_type.size

        # Representation-aware normalization
        self.norm = EquivariantLayerNorm(token_type, affine=False, eps=eps)

        # SiLU activation
        self.silu = nn.SiLU()

        # Use standard PyTorch linear for faster convergence
        # Projects trivial temb to scale and shift
        # This is valid because:
        # 1. temb is trivial (invariant) - same value for all group elements
        # 2. scale/shift are applied element-wise - equivariance is preserved
        self.linear = nn.Linear(self.temb_dim, 2 * self.token_dim)
        
        # Initialize scale close to 0 and shift to 0 for stable training
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: enn.GeometricTensor, temb: torch.Tensor):
        """
        x    : GeometricTensor (B*T, token_dim) - equivariant features
        temb : torch.Tensor (B*T, temb_dim) - trivial/invariant timestep embedding
        """
        # Normalize equivariantly
        gx_normed = self.norm(x)

        # Apply SiLU and project to scale + shift
        temb_activated = self.silu(temb)
        scale_shift = self.linear(temb_activated)
        scale, shift = scale_shift.chunk(2, dim=1)

        # AdaLayerNorm: x_norm * (1 + scale) + shift
        # This is equivariant because scale/shift are element-wise operations
        out = gx_normed.tensor * (1 + scale) + shift

        return enn.GeometricTensor(out, self.token_type)
    

class EquivariantAttention(nn.Module):

    def __init__(
        self,
        in_type: enn.FieldType,
        cross_attention_type: Optional[enn.FieldType] = None,
        *,
        heads=4,
        dim_head=16,
        dropout=0.0,
        bias=True,
        out_bias=True,
        use_relative_position_bias=False,
        max_relative_position=32,
    ):
        super().__init__()

        self.in_type = in_type
        self.cross_type = cross_attention_type or in_type

        gspace = in_type.gspace
        assert gspace == self.cross_type.gspace
        self.G = gspace.fibergroup.order()

        # Head dims
        self.H = heads
        self.Dh = dim_head
        self.scalar_dim = self.H * self.Dh

        # Scalar FieldType for Q, K, V
        self.scalar_type = enn.FieldType(
            gspace,
            [gspace.trivial_repr] * self.scalar_dim
        )

        # ESCNN-equivariant projections
        self.q_proj = enn.Linear(in_type, self.scalar_type, bias=bias)
        self.k_proj = enn.Linear(self.cross_type, self.scalar_type, bias=bias)
        self.v_proj = enn.Linear(self.cross_type, self.scalar_type, bias=bias)

        self.o_proj = enn.Linear(self.scalar_type, in_type, bias=out_bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = (dim_head) ** -0.5
        
        # Relative position bias (equivariant because it's based on relative distances)
        self.use_relative_position_bias = use_relative_position_bias
        if use_relative_position_bias:
            self.max_relative_position = max_relative_position
            # Learnable relative position bias per head
            # Shape: (heads, 2*max_relative_position + 1)
            self.relative_position_bias = nn.Parameter(
                torch.zeros(heads, 2 * max_relative_position + 1)
            )

    def _get_relative_position_bias(self, Tq, Tk, device):
        """
        Compute relative position bias matrix.
        This is equivariant because it only depends on relative distances.
        
        Returns: (H, Tq, Tk) bias tensor
        """
        # Create position indices
        q_pos = torch.arange(Tq, device=device).unsqueeze(1)  # (Tq, 1)
        k_pos = torch.arange(Tk, device=device).unsqueeze(0)  # (1, Tk)
        
        # Compute relative positions and clip
        relative_position = q_pos - k_pos  # (Tq, Tk)
        relative_position = torch.clamp(
            relative_position,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift to make all indices positive
        relative_position_index = relative_position + self.max_relative_position  # (Tq, Tk)
        
        # Index into learned bias
        # relative_position_bias: (H, 2*max_relative_position + 1)
        # Output: (H, Tq, Tk)
        bias = self.relative_position_bias[:, relative_position_index]  # (H, Tq, Tk)
        
        return bias
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        """
        hidden_states         : (B, Tq, Cq*G)
        encoder_hidden_states : (B, Tk, Ckv*G)
        """
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        B, Tq, _ = hidden_states.shape
        B2, Tk, _ = encoder_hidden_states.shape
        assert B == B2

        # Wrap into geometric tensors
        q_geo = enn.GeometricTensor(
            einops.rearrange(hidden_states, "b t d -> (b t) d"),
            self.in_type
        )
        k_geo = enn.GeometricTensor(
            einops.rearrange(encoder_hidden_states, "b t d -> (b t) d"),
            self.cross_type
        )
        v_geo = enn.GeometricTensor(
            einops.rearrange(encoder_hidden_states, "b t d -> (b t) d"),
            self.cross_type
        )

        # ESCNN equivariant projection → scalar fields
        Q = self.q_proj(q_geo).tensor     # (B*Tq, scalar_dim)
        K = self.k_proj(k_geo).tensor     # (B*Tk, scalar_dim)
        V = self.v_proj(v_geo).tensor     # (B*Tk, scalar_dim)

        # reshape back
        Q = einops.rearrange(Q, "(b t) d -> b t d", b=B)
        K = einops.rearrange(K, "(b t) d -> b t d", b=B)
        V = einops.rearrange(V, "(b t) d -> b t d", b=B)

        # Split into heads using einops
        # Q,K,V: (B, T, H, Dh)
        Q = einops.rearrange(Q, "b t (h d) -> b t h d", h=self.H)
        K = einops.rearrange(K, "b t (h d) -> b t h d", h=self.H)
        V = einops.rearrange(V, "b t (h d) -> b t h d", h=self.H)

        # Scaled dot-product attention (equivariant because Q,K are scalar reps)
        # attn: (B, H, Tq, Tk)
        attn = torch.einsum("b t h d, b k h d -> b h t k", Q, K) * self.scale
        
        # Add relative position bias (equivariant because it's translation-invariant)
        if self.use_relative_position_bias:
            rel_pos_bias = self._get_relative_position_bias(Tq, Tk, attn.device)  # (H, Tq, Tk)
            attn = attn + rel_pos_bias.unsqueeze(0)  # (B, H, Tq, Tk)
        
        if attention_mask is not None:
            attn = attn + attention_mask[:, None, :, :]

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to V
        # out: (B, H, Tq, Dh)
        out = torch.einsum("b h t k, b k h d -> b h t d", attn, V)

        # merge heads
        out = einops.rearrange(out, "b h t d -> (b t) (h d)")

        # Lift scalar back to group representation with ESCNN
        out_geo = enn.GeometricTensor(out, self.scalar_type)
        out = self.o_proj(out_geo).tensor          # (B*Tq, Cq*G)
        out = einops.rearrange(out, "(b t) d -> b t d", b=B)

        return out


class EquivariantFeedForward(nn.Module):
    r"""
    An equivariant feed-forward layer using ESCNN.

    Parameters:
        in_type (`enn.FieldType`): The input field type.
        inner_type (`enn.FieldType`): The inner/hidden field type.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"gelu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        in_type: enn.FieldType,
        inner_type: enn.FieldType,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        final_dropout: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.in_type = in_type
        self.inner_type = inner_type
        
        # Choose activation function based on the type
        if activation_fn == "gelu":
            self.act = EquivariantGeLU(in_type, inner_type, bias=bias)
        elif activation_fn == "gelu-approximate":
            self.act = ApproximateGELU(in_type, inner_type, bias=bias)
        elif activation_fn in ["geglu", "geglu-approximate"]:
            # GEGLU combines projection and activation, so no separate fc1 needed
            self.act = EquivariantGEGLU(in_type, inner_type, bias=bias)
        elif activation_fn == "swiglu":
            # SwiGLU also combines projection and activation
            self.act = EquivariantSwiGLU(in_type, inner_type, bias=bias)
        else:
            # Default to GELU
            self.act = EquivariantGeLU(in_type, inner_type, bias=bias)
        
        # Dropout
        self.dropout = enn.PointwiseDropout(inner_type, p=dropout) if dropout > 0.0 else None
        
        # Project back from inner_type to in_type (contraction)
        self.fc2 = enn.Linear(inner_type, in_type, bias=bias)
        
        # Final dropout
        self.final_dropout = enn.PointwiseDropout(in_type, p=dropout) if final_dropout and dropout > 0.0 else None

    def forward(self, hidden_states):
        """
        hidden_states: (B*T, C*G) where C*G is the dimension of in_type
        """
        # Forward pass through activation (which includes projection for all our activation types)
        x_geo = self.act(hidden_states)
        
        if self.dropout is not None:
            x_geo = self.dropout(x_geo)
        
        # Project back to in_type
        x_geo = self.fc2(x_geo)
        
        if self.final_dropout is not None:
            x_geo = self.final_dropout(x_geo)

        return x_geo


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        in_type: enn.FieldType,
        cross_attention_type: enn.FieldType,
        inner_type: enn.FieldType,
        temb_type: Optional[enn.FieldType] = None,  # trivial type for timestep embedding
        
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        dropout=0.0,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        norm_type: str = "ada_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_out_bias: bool = True,
        use_relative_position_bias: bool = False,
        max_relative_position: int = 32,
    ):
        super().__init__()

        ## attn
        self.in_type = in_type
        self.cross_attention_type = cross_attention_type
        ## ff
        self.inner_type = inner_type
        # temb_type for trivial timestep embedding
        self.temb_type = temb_type
        
        # Store norm_type for forward pass
        self.norm_type = norm_type
        
        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = EquivariantAdaLayerNorm(self.in_type, temb_type=self.temb_type)
        else:
            self.norm1 = EquivariantLayerNorm(self.in_type, affine=False, eps=norm_eps)

        self.attn1 = EquivariantAttention(
            in_type=self.in_type,
            cross_attention_type=self.cross_attention_type,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
            use_relative_position_bias=use_relative_position_bias,
            max_relative_position=max_relative_position,
        )

        # 3. Feed-forward
        self.norm3 = EquivariantLayerNorm(self.in_type, affine=False, eps=norm_eps)
        self.ff = EquivariantFeedForward(
            self.in_type,
            self.inner_type,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout
        )
        if final_dropout:
            self.final_dropout = enn.PointwiseDropout(self.in_type, p=dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the equivariant transformer block.
        
        Args:
            hidden_states: (B, T, C*G) tensor where C*G is in_type.size
            encoder_hidden_states: (B, S, C*G) tensor for cross-attention
            attention_mask: Optional attention mask
            encoder_attention_mask: Optional encoder attention mask
            temb: Optional time embedding for adaptive norm
            
        Returns:
            Output tensor of shape (B, T, C*G)
        """
        B, T, _ = hidden_states.shape
        
        # Flatten for ESCNN processing: (B, T, C*G) -> (B*T, C*G)
        hidden_states_flat = einops.rearrange(hidden_states, "b t d -> (b t) d")
        hidden_geo = enn.GeometricTensor(hidden_states_flat, self.in_type)

        # 1. Self-Attention with normalization
        if self.norm_type == "ada_norm":
            # AdaLayerNorm expects flattened input
            temb_flat = einops.repeat(temb, "b d -> (b t) d", t=T) if temb is not None else None
            norm_hidden_geo = self.norm1(hidden_geo, temb_flat)
        else:
            norm_hidden_geo = self.norm1(hidden_geo)
        
        # Extract normalized tensor and reshape for attention: (B*T, C*G) -> (B, T, C*G)
        norm_hidden_states = einops.rearrange(norm_hidden_geo.tensor, "(b t) d -> b t d", b=B)

        # Apply attention (EquivariantAttention handles tensor input/output)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        
        # Residual connection
        hidden_states = attn_output + hidden_states
        
        # 2. Feed-forward with normalization
        # Flatten for ESCNN processing
        hidden_states_flat = einops.rearrange(hidden_states, "b t d -> (b t) d")
        hidden_geo = enn.GeometricTensor(hidden_states_flat, self.in_type)
        
        norm_hidden_geo = self.norm3(hidden_geo)
        
        # Apply feedforward (EquivariantFeedForward handles GeometricTensor input/output)
        ff_output_geo = self.ff(norm_hidden_geo)
        
        # Extract tensor and reshape: (B*T, C*G) -> (B, T, C*G)
        ff_output = einops.rearrange(ff_output_geo.tensor, "(b t) d -> b t d", b=B)
        
        # Apply final dropout if needed
        if self.final_dropout is not None:
            ff_output_flat = einops.rearrange(ff_output, "b t d -> (b t) d")
            ff_output_geo = enn.GeometricTensor(ff_output_flat, self.in_type)
            ff_output_geo = self.final_dropout(ff_output_geo)
            ff_output = einops.rearrange(ff_output_geo.tensor, "(b t) d -> b t d", b=B)
        
        # Residual connection
        hidden_states = ff_output + hidden_states
        
        return hidden_states


class EDiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        n_group: int = 8,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        use_relative_position_bias: bool = True,
        max_relative_position: int = 32,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False
        self.n_group = n_group

        # Setup ESCNN group space
        G = CyclicGroup(n_group)
        self.gspace = escnn.gspaces.no_base_space(G)
        
        # Define field types for equivariant layers
        # Each field type contains multiple regular representations
        self.in_type = enn.FieldType(
            self.gspace, 
            [self.gspace.regular_repr] * int(self.inner_dim/self.n_group)
        )
        # Cross-attention uses TRIVIAL representation for INVARIANT VL features
        # This ensures the model is equivariant w.r.t. state/action regardless of VL content
        self.cross_attention_type = enn.FieldType(
            self.gspace,
            [self.gspace.regular_repr] * int(self.config.cross_attention_dim / self.n_group)
        )
        self.ff_inner_type = enn.FieldType(
            self.gspace,
            [self.gspace.regular_repr] * int(self.inner_dim*4/self.n_group)
        )
        self.out_type = enn.FieldType(
            self.gspace,
            [self.gspace.regular_repr] * int(self.config.output_dim/self.n_group)
        )
        
        print(f"EDiT Equivariant Configuration:")
        print(f"  Group: C{n_group}")
        print(f"  in_type size: {self.in_type.size} (regular repr - equivariant)")
        print(f"  cross_attention_type size: {self.cross_attention_type.size} (trivial repr - invariant)")
        print(f"  inner_type size: {self.ff_inner_type.size}")

        # Timestep encoder outputs TRIVIAL repr (time is scalar, not rotated)

        self.time_out_type = enn.FieldType(
            self.gspace, 
            [self.gspace.trivial_repr] * int(self.in_type.size)
        )
        
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.in_type.size, compute_dtype=self.config.compute_dtype
        )
        
        # Build equivariant transformer blocks
        all_blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            
            # For self-attention layers, use in_type for both query and key/value
            curr_cross_type = self.in_type if use_self_attn else self.cross_attention_type

            all_blocks += [
                BasicTransformerBlock(
                    in_type=self.in_type,
                    cross_attention_type=curr_cross_type,
                    inner_type=self.ff_inner_type,
                    temb_type=self.time_out_type,  # Pass trivial timestep type
                    
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    norm_type=norm_type,
                    norm_eps=self.config.norm_eps,
                    final_dropout=final_dropout,
                    use_relative_position_bias=use_relative_position_bias,
                    max_relative_position=max_relative_position,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output blocks (operating on equivariant features)
        # Use EquivariantLayerNorm instead of LayerNorm for equivariance
        self.norm_out = EquivariantLayerNorm(self.in_type, affine=False, eps=1e-6)
        
        # Project from trivial temb to scale/shift for final AdaLN
        # Use standard nn.Linear for faster convergence (same reasoning as EquivariantAdaLayerNorm)
        # temb is trivial/invariant, and scale/shift are applied element-wise
        self.proj_out_1 = nn.Linear(self.time_out_type.size, 2 * self.in_type.size)
        # Initialize to zero for stable training (starts as identity: out = norm(x) * 1 + 0)
        nn.init.zeros_(self.proj_out_1.weight)
        nn.init.zeros_(self.proj_out_1.bias)
        
        # Final projection to output dimension - KEEP EQUIVARIANT (same as in_type)
        # Output now maintains equivariance!
        self.proj_out_2 = enn.Linear(self.in_type, self.out_type)
        
        print(
            "Total number of Equivariant DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )


    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, in_type.size)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, cross_attention_type.size)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        """
        Forward pass through the equivariant DiT model.
        
        Args:
            hidden_states: (B, T, in_type.size) - equivariant input features
            encoder_hidden_states: (B, S, cross_attention_type.size) - equivariant encoder features
            timestep: timestep for conditioning
            encoder_attention_mask: Optional attention mask
            return_all_hidden_states: Whether to return intermediate hidden states
            
        Returns:
            Output tensor of shape (B, T, output_dim) or tuple with all hidden states
        """
        B, T, _ = hidden_states.shape
        
        # Encode timesteps (non-equivariant conditioning)
        temb = self.timestep_encoder(timestep)  # (B, in_type.size)

        # Process through transformer blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through equivariant transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                # Self-attention layer
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,  # Self-attention
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # Cross-attention layer
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # Output processing with equivariant layers
        # Normalize
        hidden_states_flat = einops.rearrange(hidden_states, "b t d -> (b t) d")
        hidden_geo = enn.GeometricTensor(hidden_states_flat, self.in_type)
        hidden_geo = self.norm_out(hidden_geo)
        
        # AdaLN conditioning with timestep (timestep is TRIVIAL - not rotated)
        # Use standard linear (faster convergence) - no GeometricTensor needed
        scale_shift = self.proj_out_1(temb)  # (B, 2*in_type.size)
        scale_tensor, shift_tensor = scale_shift.chunk(2, dim=1)
        
        # Apply scale and shift (expand to match sequence length)
        scale_expanded = einops.repeat(scale_tensor, "b d -> (b t) d", t=T)
        shift_expanded = einops.repeat(shift_tensor, "b d -> (b t) d", t=T)
        
        # Apply conditioning: output = norm(x) * (1 + scale) + shift
        # This is equivariant because scale/shift are element-wise operations
        conditioned_tensor = hidden_geo.tensor * (1 + scale_expanded) + shift_expanded
        conditioned_geo = enn.GeometricTensor(conditioned_tensor, self.in_type)
        
        # Final equivariant projection (in_type -> out_type)
        output_geo = self.proj_out_2(conditioned_geo)
        output = einops.rearrange(output_geo.tensor, "(b t) d -> b t d", b=B)
        
        if return_all_hidden_states:
            return output, all_hidden_states
        else:
            return output


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        n_group: int = 8,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        use_relative_position_bias: bool = True,
        max_relative_position: int = 32,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False
        self.n_group = n_group
        
        # Setup ESCNN group space
        G = CyclicGroup(n_group)
        self.gspace = escnn.gspaces.no_base_space(G)
        
        # Define field types for equivariant layers
        # Each field type contains multiple regular representations
        self.in_type = enn.FieldType(
            self.gspace, 
            [self.gspace.regular_repr] * int(self.inner_dim/self.n_group)
        )
        # Cross-attention uses TRIVIAL representation for INVARIANT VL features
        # This ensures the model is equivariant w.r.t. state/action regardless of VL content
        self.ff_inner_type = enn.FieldType(
            self.gspace,
            [self.gspace.regular_repr] * int(self.inner_dim*4/self.n_group)
        )
        self.out_type = enn.FieldType(
            self.gspace,
            [self.gspace.regular_repr] * int(self.config.output_dim/self.n_group)
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    in_type=self.in_type,
                    cross_attention_type=self.in_type,
                    inner_type=self.ff_inner_type,
                    temb_type=None,  # No timestep embedding for self-attention transformer
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    norm_type="layer_norm",  # Use layer_norm, not ada_norm (no timestep)
                    norm_eps=1e-5,
                    final_dropout=final_dropout,
                    use_relative_position_bias=use_relative_position_bias,
                    max_relative_position=max_relative_position,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states