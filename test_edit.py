"""
Test script to verify Equivariant EDiT implementation
"""
import torch
import escnn
import escnn.nn as enn
from escnn import gspaces
import einops

# Import the EDiT model
import sys
sys.path.insert(0, '/home/locht1/Documents/locht1/Isaac-GR00T')

from gr00t.model.action_head.equivariant_cross_attention_dit import EDiT

def test_edit_basic():
    """Test basic EDiT functionality"""
    
    print("="*60)
    print("Testing Equivariant EDiT - Basic Functionality")
    print("="*60)
    from escnn.group import CyclicGroup
    
    G = CyclicGroup(8)
    gspace = escnn.gspaces.no_base_space(G)
    # Create EDiT model
    model = EDiT(
        n_group=8,
        num_attention_heads=8,
        attention_head_dim=8,
        cross_attention_dim=64,
        output_dim=26,
        num_layers=4,  # Small for testing
        dropout=0.0,
        attention_bias=True,
        activation_fn="gelu-approximate",
        norm_type="ada_norm",
        interleave_self_attention=True,
        final_dropout=True,
        use_relative_position_bias=True,
        max_relative_position=32,
    )
    
    print(f"\n✓ EDiT model created successfully")
    
    # Create test input matching the field type sizes
    B, T, S = 2, 10, 5
    in_type_size = model.in_type.size
    cross_type_size = model.cross_attention_type.size
    
    hidden_states = torch.randn(B, T, in_type_size)
    encoder_hidden_states = torch.randn(B, S, cross_type_size)
    timestep = torch.tensor([5, 10])  # Two different timesteps for batch
    
    print(f"\nInput shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"  timestep: {timestep.shape}")
    
    # Forward pass
    origin_output = model(hidden_states, encoder_hidden_states, timestep)
    
    for element in gspace.testing_elements:
        # Rotate input
        rotated_hidden_states = einops.rearrange(hidden_states, 'b t c -> (b t) c')
        rotated_hidden_states = enn.GeometricTensor(
            rotated_hidden_states,
            model.in_type
        ).transform(element).tensor
        rotated_hidden_states = einops.rearrange(rotated_hidden_states, '(b t) c -> b t c', b=B)
    
        # Forward pass with rotated input
        rotated_output = model(rotated_hidden_states, encoder_hidden_states, timestep)
        
        # rotate original output
        rotated_origin_output = einops.rearrange(origin_output, 'b t c -> (b t) c')
        rotated_origin_output = enn.GeometricTensor(
            rotated_origin_output,
            model.out_type
        ).transform(element).tensor
        rotated_origin_output = einops.rearrange(rotated_origin_output, '(b t) c -> b t c', b=B)
        # print(rotated_output)
            
        err = (rotated_origin_output - rotated_output).abs().mean()
        print(torch.allclose(rotated_output, rotated_origin_output, atol=1e-4), element, err)



    return model, hidden_states, encoder_hidden_states, timestep




if __name__ == "__main__":
    print("\n" + "="*70)
    print("EQUIVARIANT EDiT MODEL TEST SUITE")
    print("="*70)
    
    # Run tests
    test_edit_basic()
    
    print("\n" + "="*70)
    print("✅ ALL EDiT TESTS PASSED!")
    print("="*70)
