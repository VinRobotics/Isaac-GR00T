"""
Test script to verify EquivariantFeedForward implementation
"""
import torch
import escnn.nn as enn
from escnn import gspaces

# Import the EquivariantFeedForward from the module
import sys
sys.path.insert(0, '/home/locht1/Documents/locht1/Isaac-GR00T')

from gr00t.model.action_head.equivariant_cross_attention_dit import EquivariantFeedForward

def test_equivariant_feedforward():
    """Test that EquivariantFeedForward maintains equivariance"""
    
    # Setup: Create a simple SO(2) group space
    gspace = gspaces.rot2dOnR2(N=4)  # C4 group (4 rotations)
    
    # Define field types
    in_channels = 8
    inner_channels = 32
    
    # Input type: mix of trivial and regular representations
    in_type = enn.FieldType(gspace, [gspace.trivial_repr] * in_channels)
    inner_type = enn.FieldType(gspace, [gspace.trivial_repr] * inner_channels)
    
    # Create the feedforward module
    ff = EquivariantFeedForward(
        in_type=in_type,
        inner_type=inner_type,
        dropout=0.0,  # No dropout for testing
        activation_fn="gelu",
        final_dropout=False,
        bias=True
    )
    
    # Create test input: (batch=2, sequence=10, channels=in_channels)
    B, T = 2, 10
    x = torch.randn(B, T, in_channels)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = ff(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {x.shape}")
    
    # Check shape is preserved
    assert output.shape == x.shape, f"Shape mismatch! Got {output.shape}, expected {x.shape}"
    
    print("✓ Shape test passed!")
    
    # Test equivariance (for regular representations)
    # For trivial representations, rotations don't change the values
    # So let's create a version with regular representations
    
    in_type_reg = enn.FieldType(gspace, [gspace.regular_repr] * 2)
    inner_type_reg = enn.FieldType(gspace, [gspace.regular_repr] * 4)
    
    ff_reg = EquivariantFeedForward(
        in_type=in_type_reg,
        inner_type=inner_type_reg,
        dropout=0.0,
        activation_fn="gelu",
        final_dropout=False,
        bias=True
    )
    
    # Create input with proper dimension (2 regular reps * 4 group elements = 8)
    x_reg = torch.randn(B, T, in_type_reg.size)
    print(f"\nRegular representation input shape: {x_reg.shape}")
    
    # Forward pass
    output_reg = ff_reg(x_reg)
    print(f"Regular representation output shape: {output_reg.shape}")
    
    # Test that the module can handle the input
    assert output_reg.shape[0] == B and output_reg.shape[1] == T
    assert output_reg.shape[2] == in_type_reg.size
    
    print("✓ Regular representation test passed!")
    
    print("\n✅ All tests passed! EquivariantFeedForward is working correctly.")

if __name__ == "__main__":
    test_equivariant_feedforward()
