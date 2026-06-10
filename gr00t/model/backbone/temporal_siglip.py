"""
MEM-style temporal SigLIP: sparse causal temporal attention inserted every `stride` layers
of the SigLIP ViT encoder.  When num_frames == 1 the forward is identical to the
original SiglipVisionModel so pre-trained weights are preserved exactly.

Reference: Torne et al., "Empowering Embodied Manipulation Agents with
Embodied 3D Perception" (MEM), 2025.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling


# ---------------------------------------------------------------------------
# Temporal position encoding
# ---------------------------------------------------------------------------

def _build_temporal_pe(T: int, H: int, device, dtype) -> torch.Tensor:
    """
    Sinusoidal temporal PE with e(0) = 0 for the current (most-recent) frame.

    Frame ordering within pixel_values (time-major, as produced by transforms.py):
        index 0 .. T*V-1  ->  (t=0, v=0), (t=0, v=1), ..., (t=T-1, v=V-1)

    We assign temporal position p = T-1-t, so:
        t=T-1 (current)  ->  p=0  ->  PE = 0
        t=T-2            ->  p=1
        ...
        t=0  (oldest)    ->  p=T-1

    Shape returned: [T, H]
    """
    positions = torch.arange(T - 1, -1, -1, device=device, dtype=dtype)  # [T-1,...,0]
    half = H // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)
    )
    angles = positions[:, None] * freqs[None, :]   # [T, half]
    pe = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [T, H]
    pe[-1] = 0.0   # current frame (p=0) gets exactly zero
    return pe


# ---------------------------------------------------------------------------
# Temporal causal self-attention
# ---------------------------------------------------------------------------

class TemporalCausalAttention(nn.Module):
    """
    Causal self-attention over the time axis.

    Input:  hidden_states  [B*T*V, P, H]
    Output: hidden_states  [B*T*V, P, H]  (residual added)

    Internally reshapes to [B*V*P, T, H], applies causal attention, reshapes back.
    Output projection is zero-initialized so at init this is a pure identity.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm = nn.LayerNorm(hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out = nn.Linear(hidden_size, hidden_size, bias=True)

        # Zero-init output so the residual path starts as identity
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        T: int,
        V: int,
    ) -> torch.Tensor:
        BTV, P, H = hidden_states.shape
        B = BTV // (T * V)

        # [B*T*V, P, H] -> [B, T, V, P, H] -> [B, V, P, T, H] -> [B*V*P, T, H]
        x = hidden_states.reshape(B, T, V, P, H)
        x = x.permute(0, 2, 3, 1, 4).contiguous()     # [B, V, P, T, H]
        x = x.reshape(B * V * P, T, H)

        # Sinusoidal temporal PE; current frame (last index) gets PE=0
        pe = _build_temporal_pe(T, H, x.device, x.dtype)  # [T, H]
        x_pe = x + pe.unsqueeze(0)                         # [B*V*P, T, H]

        x_norm = self.norm(x_pe)

        BVP = B * V * P
        nh, hd = self.num_heads, self.head_dim

        q = self.q(x_norm).reshape(BVP, T, nh, hd).transpose(1, 2)  # [BVP, nh, T, hd]
        k = self.k(x_norm).reshape(BVP, T, nh, hd).transpose(1, 2)
        v = self.v(x_norm).reshape(BVP, T, nh, hd).transpose(1, 2)

        # Causal mask: frame i can attend to frames 0..i (oldest to current)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(BVP, T, H)
        attn_out = self.out(attn_out)

        x = x + attn_out  # residual; [B*V*P, T, H]

        # Reshape back: [B*V*P, T, H] -> [B, V, P, T, H] -> [B, T, V, P, H] -> [B*T*V, P, H]
        x = x.reshape(B, V, P, T, H)
        x = x.permute(0, 3, 1, 2, 4).contiguous()     # [B, T, V, P, H]
        x = x.reshape(B * T * V, P, H)

        return x


# ---------------------------------------------------------------------------
# Temporal SigLIP wrapper
# ---------------------------------------------------------------------------

class TemporalSiglipVisionModel(nn.Module):
    """
    Drop-in replacement for SiglipVisionModel that inserts causal temporal
    attention after every `temporal_stride`-th encoder layer.

    The base SiglipVisionModel weights are kept intact (no modification).
    Only the new TemporalCausalAttention modules are freshly initialised.

    Usage:
        model.vision_model = TemporalSiglipVisionModel(
            base_vision_model=model.vision_model,
            num_frames=T,
            num_views=V,
            temporal_stride=4,
        )

    Forward signature mirrors SiglipVisionModel so extract_feature() needs
    no changes.
    """

    def __init__(
        self,
        base_vision_model: nn.Module,   # SiglipVisionModel
        num_frames: int = 1,
        num_views: int = 1,
        temporal_stride: int = 4,
    ):
        super().__init__()
        self.base_vision_model = base_vision_model
        self.num_frames = num_frames
        self.num_views = num_views
        self.temporal_stride = temporal_stride

        vt = base_vision_model.vision_model          # SiglipVisionTransformer
        hidden_size = vt.config.hidden_size
        num_heads = vt.config.num_attention_heads
        num_layers = len(vt.encoder.layers)

        n_temporal = sum(
            1 for i in range(num_layers) if (i + 1) % temporal_stride == 0
        )
        self.temporal_layers = nn.ModuleList(
            [TemporalCausalAttention(hidden_size, num_heads) for _ in range(n_temporal)]
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        T = self.num_frames
        V = self.num_views

        if T == 1:
            # Single frame: identical to original model (temporal layers inactive)
            return self.base_vision_model(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        vt = self.base_vision_model.vision_model   # SiglipVisionTransformer

        # --- Patch embedding ---
        hidden_states = vt.embeddings(pixel_values)  # [B*T*V, P, H]

        # --- Encoder with interleaved temporal attention ---
        all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = () if output_hidden_states else None
        temporal_idx = 0

        for i, layer in enumerate(vt.encoder.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Standard spatial self-attention (unchanged SigLIP layer)
            layer_out = layer(
                hidden_states,
                attention_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_out[0]

            # Causal temporal attention after every stride-th layer
            if (i + 1) % self.temporal_stride == 0:
                hidden_states = self.temporal_layers[temporal_idx](
                    hidden_states, T, V
                )
                temporal_idx += 1

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # --- Final layer norm ---
        hidden_states = vt.post_layernorm(hidden_states)  # [B*T*V, P, H]

        if return_dict:
            return BaseModelOutputWithPooling(
                last_hidden_state=hidden_states,
                pooler_output=None,
                hidden_states=all_hidden_states,
                attentions=None,
            )
        return (hidden_states,)
