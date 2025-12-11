"""
Test script to verify BasicTransformerBlock implementation
"""
import torch
import escnn
import escnn.nn as enn
from escnn import gspaces
import einops
# Import the BasicTransformerBlock
import sys
sys.path.insert(0, '/home/locht1/Documents/locht1/Isaac-GR00T')

from gr00t.model.action_head.equivariant_cross_attention_dit import BasicTransformerBlock

def test_basic_transformer_block():
    """Test that BasicTransformerBlock works correctly"""
    
    # Setup: Create a 0D group space for linear layers (required by enn.Linear)
    # Using cyclic group C4
    from escnn.group import CyclicGroup
    G = CyclicGroup(8)
    gspace = escnn.gspaces.no_base_space(G)
    
    # Define field types
    hidden_dim = 8  # Number of representations
    cross_dim = 8
    inner_dim = 16
    
    # Create field types with regular representations
    in_type = enn.FieldType(gspace, [gspace.regular_repr] * hidden_dim)
    cross_attention_type = enn.FieldType(gspace, [gspace.regular_repr] * cross_dim)
    inner_type = enn.FieldType(gspace, [gspace.regular_repr] * inner_dim)
    
    print(f"in_type size: {in_type.size}")
    print(f"cross_attention_type size: {cross_attention_type.size}")
    print(f"inner_type size: {inner_type.size}")
    
    # Create the transformer block
    block = BasicTransformerBlock(
        in_type=in_type,
        cross_attention_type=cross_attention_type,
        inner_type=inner_type,
        num_attention_heads=8,
        attention_head_dim=8,
        dropout=0.0,
        activation_fn="gelu",
        attention_bias=True,
        norm_type="layer_norm",
        final_dropout=False,
    )
    
    print(f"✓ BasicTransformerBlock created successfully")
    
    # Create test input
    B, T, S = 2, 10, 5
    hidden_states = torch.randn(B, T, in_type.size)
    encoder_hidden_states = torch.randn(B, S, cross_attention_type.size)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Encoder hidden states shape: {encoder_hidden_states.shape}")


    # Forward pass
    origin_output = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
    
    for element in gspace.testing_elements:
        # Rotate input
        rotated_hidden_states = einops.rearrange(hidden_states, 'b t c -> (b t) c')
        rotated_hidden_states = enn.GeometricTensor(
            rotated_hidden_states,
            in_type
        ).transform(element).tensor
        rotated_hidden_states = einops.rearrange(rotated_hidden_states, '(b t) c -> b t c', b=B)
        
        rotated_encoder_states = einops.rearrange(encoder_hidden_states, 'b s c -> (b s) c')
        rotated_encoder_states = enn.GeometricTensor(
            rotated_encoder_states,
            cross_attention_type
        ).transform(element).tensor
        rotated_encoder_states = einops.rearrange(rotated_encoder_states, '(b s) c -> b s c', b=B)
        
        # Forward pass with rotated input
        rotated_output = block(
            rotated_hidden_states,
            encoder_hidden_states=rotated_encoder_states
        )
        
        # rotate original output
        rotated_origin_output = einops.rearrange(origin_output, 'b t c -> (b t) c')
        rotated_origin_output = enn.GeometricTensor(
            rotated_origin_output,
            in_type
        ).transform(element).tensor
        rotated_origin_output = einops.rearrange(rotated_origin_output, '(b t) c -> b t c', b=B)
        # print(rotated_output)
            
        err = (rotated_origin_output - rotated_output).abs().mean()
        print(torch.allclose(rotated_output, rotated_origin_output, atol=1e-4), element, err)


if __name__ == "__main__":
    test_basic_transformer_block()
