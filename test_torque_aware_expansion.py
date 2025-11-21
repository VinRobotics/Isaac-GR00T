#!/usr/bin/env python3
"""
Test script to verify that torque-aware expansion works correctly
"""

import torch
from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)

def test_weight_expansion():
    """Test that weight expansion preserves pretrained weights"""
    
    # Create a config for the original model (without torque_aware)
    original_config = FlowmatchingActionHeadConfig(
        action_dim=14,
        action_horizon=16,
        hidden_size=1024,
        input_embedding_dim=1536,
        backbone_embedding_dim=1536,
        max_num_embodiments=32,
        max_state_dim=64,
        num_timestep_buckets=1000,
        num_inference_timesteps=10,
        diffusion_model_cfg={
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 4,
        },
        vl_self_attention_cfg={
            "hidden_size": 1536,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
        },
        torque_aware=False,
        effort_dim=None,
    )
    
    # Create original action head
    print("Creating original action head (no torque_aware)...")
    original_head = FlowmatchingActionHead(original_config)
    original_state_dict = original_head.state_dict()
    
    # Get some key tensors to compare later
    original_W1_W = original_state_dict['action_encoder.W1.W'].clone()
    original_layer2_W = original_state_dict['action_decoder.layer2.W'].clone()
    original_layer2_b = original_state_dict['action_decoder.layer2.b'].clone()
    
    print(f"Original action_encoder.W1.W shape: {original_W1_W.shape}")
    print(f"Original action_decoder.layer2.W shape: {original_layer2_W.shape}")
    print(f"Original action_decoder.layer2.b shape: {original_layer2_b.shape}")
    
    # Create a new config with torque_aware enabled
    torque_config = FlowmatchingActionHeadConfig(
        action_dim=14,
        action_horizon=16,
        hidden_size=1024,
        input_embedding_dim=1536,
        backbone_embedding_dim=1536,
        max_num_embodiments=32,
        max_state_dim=64,
        num_timestep_buckets=1000,
        num_inference_timesteps=10,
        diffusion_model_cfg={
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 4,
        },
        vl_self_attention_cfg={
            "hidden_size": 1536,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
        },
        torque_aware=True,
        effort_dim=7,
    )
    
    # Create new action head with torque awareness
    print("\nCreating torque-aware action head...")
    torque_head = FlowmatchingActionHead(torque_config)
    
    # Expand the weights
    print("\nExpanding pretrained weights...")
    expanded_state_dict = torque_head.expand_action_weights_for_torque_aware(
        original_state_dict,
        old_action_dim=14,
        new_action_dim=14 + 7,
    )
    
    # Load the expanded weights
    missing, unexpected = torque_head.load_state_dict(expanded_state_dict, strict=False)
    print(f"\nMissing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    # Verify that the action portion of the weights matches
    new_W1_W = torque_head.action_encoder.W1.W
    new_layer2_W = torque_head.action_decoder.layer2.W
    new_layer2_b = torque_head.action_decoder.layer2.b
    
    print(f"\nNew action_encoder.W1.W shape: {new_W1_W.shape}")
    print(f"New action_decoder.layer2.W shape: {new_layer2_W.shape}")
    print(f"New action_decoder.layer2.b shape: {new_layer2_b.shape}")
    
    # Check that the original action portion is preserved
    print("\nVerifying weight preservation...")
    
    # For W1: (num_emb, action_dim, hidden_size) -> check first action_dim dimension
    w1_match = torch.allclose(new_W1_W[:, :14, :], original_W1_W, rtol=1e-5, atol=1e-8)
    print(f"action_encoder.W1.W action portion preserved: {w1_match}")
    
    # For layer2.W: (num_emb, hidden_size, action_dim) -> check first action_dim in output
    layer2_w_match = torch.allclose(new_layer2_W[:, :, :14], original_layer2_W, rtol=1e-5, atol=1e-8)
    print(f"action_decoder.layer2.W action portion preserved: {layer2_w_match}")
    
    # For layer2.b: (num_emb, action_dim) -> check first action_dim
    layer2_b_match = torch.allclose(new_layer2_b[:, :14], original_layer2_b, rtol=1e-5, atol=1e-8)
    print(f"action_decoder.layer2.b action portion preserved: {layer2_b_match}")
    
    if w1_match and layer2_w_match and layer2_b_match:
        print("\n✅ SUCCESS: All pretrained weights preserved correctly!")
        return True
    else:
        print("\n❌ FAILURE: Some weights were not preserved correctly!")
        return False

if __name__ == "__main__":
    success = test_weight_expansion()
    exit(0 if success else 1)
