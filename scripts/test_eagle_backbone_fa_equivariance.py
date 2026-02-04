# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to verify that EagleBackboneFA is equivariant on images.

Equivariance property: f(g · x) = ρ(g) · f(x)
Where:
- g is a rotation from the cyclic group CN
- x is the input image
- f is the backbone function
- ρ(g) is the representation of g on the output space (regular representation)

Testing approach:
1. origin = f(x)           # Forward pass on original input
2. new_output = f(g*x)     # Forward pass on rotated input
3. rotate_origin = g*f(x)  # Apply group action to original output
4. Check: new_output ≈ rotate_origin (within tolerance)
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import sys
import os
from typing import List, Tuple, Any, Optional
from copy import deepcopy
from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup 
from gr00t.model.backbone.eagle_backbone_fa import EagleBackboneFA
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoModel, AutoConfig

# Model path - use HuggingFace hub path
BASE_MODEL_PATH = "nvidia/GR00T-N1.5-3B"
# Or local checkpoint path
CHECKPOINT_PATH = "checkpoints/GR00T-N1.5-3B"


def rotate_image(img: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotate image by angle (in radians) using grid_sample.
    
    Args:
        img: [B, C, H, W] or [C, H, W] tensor
        angle: rotation angle in radians
        
    Returns:
        Rotated image tensor
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    B, C, H, W = img.shape
    original_dtype = img.dtype
    
    # Convert to float32 for grid_sample (doesn't support bfloat16)
    img_float = img.float()
    
    # Create rotation matrix (negative angle for clockwise rotation)
    cos_val = math.cos(-angle)
    sin_val = math.sin(-angle)
    
    rotation_matrix = torch.tensor([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0]
    ], dtype=torch.float32, device=img.device).unsqueeze(0).expand(B, -1, -1)
    
    # Generate sampling grid
    grid = F.affine_grid(rotation_matrix, list([B, C, H, W]), align_corners=True)
    
    # Apply rotation
    rotated = F.grid_sample(img_float, grid, align_corners=True, padding_mode='zeros')
    
    # Convert back to original dtype
    rotated = rotated.to(original_dtype)
    
    if squeeze:
        rotated = rotated.squeeze(0)
    
    return rotated


def transform_regular_repr(features: torch.Tensor, group_element: int, n_group: int) -> torch.Tensor:
    """
    Apply group transformation to features in regular representation.
    
    For regular representation ρ_reg, the action of group element g_k is:
    ρ_reg(g_k)[f]_i = f_{(i-k) mod N}
    
    This is equivalent to cyclically shifting the feature blocks by k positions.
    
    Args:
        features: [B, T, D] or [B, D] tensor where D = n_group * block_size
        group_element: index k of the group element (0 to n_group-1)
        n_group: size of cyclic group
        
    Returns:
        Transformed features: g * f(x)
    """
    original_shape = features.shape
    
    if features.dim() == 2:
        B, D = features.shape
        T = 1
        features = features.unsqueeze(1)
    else:
        B, T, D = features.shape
    
    # D should be divisible by n_group
    assert D % n_group == 0, f"D ({D}) must be divisible by n_group ({n_group})"
    block_size = D // n_group
    
    # Reshape to [B, T, n_group, block_size]
    features_reshaped = features.reshape(B, T, n_group, block_size)
    
    # Apply cyclic permutation: ρ(g_k) shifts by k positions
    # For regular repr: ρ(g_k)[f]_i = f_{(i-k) mod N}
    # torch.roll with shifts=k shifts elements to the right by k positions
    features_transformed = torch.roll(features_reshaped, shifts=group_element, dims=2)
    
    # Reshape back
    features_transformed = features_transformed.reshape(B, T, D)
    
    if len(original_shape) == 2:
        features_transformed = features_transformed.squeeze(1)
    
    return features_transformed


def create_test_input(batch_size: int, num_images: int, img_size: int = 224, 
                      seq_len: int = 64, device: torch.device = None,
                      dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Create test vision-language input with structured (non-symmetric) pattern.
    
    Args:
        batch_size: batch size
        num_images: number of images per sample
        img_size: image height/width
        seq_len: sequence length for text tokens
        device: torch device
        dtype: torch dtype
        
    Returns:
        Dictionary with eagle_ prefixed keys
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create images with non-rotationally-symmetric pattern
    pixel_values = torch.zeros(batch_size * num_images, 3, img_size, img_size, 
                               device=device, dtype=dtype)
    
    # Create a distinctive pattern: gradient + asymmetric shapes
    for i in range(img_size):
        for j in range(img_size):
            # Horizontal gradient in red channel
            pixel_values[:, 0, i, j] = j / img_size
            # Vertical gradient in green channel  
            pixel_values[:, 1, i, j] = i / img_size
            # Diagonal pattern in blue channel
            if i > j:
                pixel_values[:, 2, i, j] = 0.8
            elif i < j - img_size // 4:
                pixel_values[:, 2, i, j] = 0.3
    
    # Add small random noise for numerical stability
    pixel_values = pixel_values + 0.05 * torch.randn_like(pixel_values)
    pixel_values = torch.clamp(pixel_values, 0, 1)
    
    # Create dummy text tokens (same for all rotations to isolate vision equivariance)
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    
    return {
        "eagle_pixel_values": pixel_values,
        "eagle_input_ids": input_ids,
        "eagle_attention_mask": attention_mask,
        "attention_mask": attention_mask,
    }


def check_equivariance(
    model: EagleBackboneFA,
    device: torch.device,
    batch_size: int = 1,
    num_images: int = 2,
    num_tests: int = None,  # If None, test all group elements
    atol: float = 1e-5,
    rtol: float = 1e-4,
    verbose: bool = True
) -> List[Tuple[Any, float, float, float]]:
    """
    Test equivariance of EagleBackboneFA following the ESCNN pattern.
    
    For each group element g:
        origin = f(x)           # Forward on original
        new_output = f(g*x)     # Forward on transformed input
        rotate_origin = g*f(x)  # Transform the original output
        
        Check: new_output ≈ rotate_origin
    
    Args:
        model: EagleBackboneFA model
        device: torch device
        batch_size: batch size for testing
        num_images: number of images per sample
        num_tests: number of group elements to test (None = all)
        atol: absolute tolerance
        rtol: relative tolerance
        verbose: print detailed results
        
    Returns:
        List of tuples: (group_element, max_error, mean_error, variance)
    """
    model.eval()
    n_group = model.n_group

    
    # Create base input
    base_input = create_test_input(batch_size, num_images, device=device)
    
    errors = []
    all_passed = True
    n_group = 8
    group = gspaces.no_base_space(CyclicGroup(n_group))
    with torch.no_grad():
        # origin = f(x)
        output_original = model(BatchFeature(data=base_input))
        vision_original = output_original["backbone_vision_features"].float()  # [B, num_imgs, D]
        
        vision_original =vision_original.reshape(vision_original.shape[0] * vision_original.shape[1], -1)
        vision_original = enn.GeometricTensor(
            vision_original, 
            enn.FieldType(
                group, 
                int(vision_original.shape[1] // n_group) * [group.regular_repr])
            )
        if verbose:
            print(f"\nVision features shape: {vision_original.shape}")
            print(f"N_group (CN): C{n_group}")
            print("-" * 70)
        
        for k in list(group.testing_elements):
            print(k)
            # Compute rotation angle for group element g_k
            angle = k.value * 2 * math.pi / n_group
            
            # g*x: Rotate the input images by angle
            rotated_pixel_values = rotate_image(
                base_input["eagle_pixel_values"],
                angle
            )
            
            rotated_input = {
                **base_input,
                "eagle_pixel_values": rotated_pixel_values
            }
            
            # new_output = f(g*x)
            output_rotated = model(BatchFeature(data=rotated_input))
            vision_rotated = output_rotated["backbone_vision_features"].float()
            vision_rotated = vision_rotated.reshape(vision_rotated.shape[0] * vision_rotated.shape[1], -1)
            # rotate_origin = g*f(x) = ρ(g) * origin
            vision_transformed = deepcopy(vision_original).transform(k).tensor
            
            # Compute errors: new_output - rotate_origin
            # Convert to float32 for numpy (bfloat16 not supported)
            errs = (vision_rotated - vision_transformed).float().cpu().numpy()
            errs_flat = np.abs(errs).reshape(-1)
            
            max_err = errs_flat.max()
            mean_err = errs_flat.mean()
            var_err = errs_flat.var()
            
            # Check if within tolerance
            is_close = np.allclose(
                vision_rotated.float().cpu().numpy(),
                vision_transformed.float().cpu().numpy(),
                atol=atol,
                rtol=rtol
            )
            
            if verbose:
                status = "✓ PASS" if is_close else "✗ FAIL"
                print(f"  g_{k} ({math.degrees(angle):6.1f}°): "
                      f"mean={mean_err:.2e}")
            
            if not is_close:
                all_passed = False
                if verbose:
                    print(f"    ERROR: Equivariance check failed for g_{k}!")
                    print(f"    max={max_err:.6f}, mean={mean_err:.6f}, var={var_err:.6f}")
            
            errors.append((k, max_err, mean_err, var_err))
    
    if verbose:
        print("-" * 70)
        if all_passed:
            print(f"✓ All equivariance tests PASSED!")
        else:
            print(f"✗ Some equivariance tests FAILED!")
        
        # Summary statistics
        max_errors = [e[1] for e in errors]
        mean_errors = [e[2] for e in errors]
        print(f"\nSummary:")
        print(f"  Overall max error: {max(max_errors):.2e}")
        print(f"  Overall mean error: {np.mean(mean_errors):.2e}")
    
    return errors


def check_language_invariance(
    model: EagleBackboneFA,
    device: torch.device,
    batch_size: int = 1,
    num_images: int = 1,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    verbose: bool = True
) -> List[Tuple[Any, float, float, float]]:
    """
    Test that language features are INVARIANT under image rotation.
    
    Language uses trivial representation, so: f_lang(g*x) = f_lang(x)
    
    Args:
        model: EagleBackboneFA model
        device: torch device
        batch_size: batch size
        num_images: number of images per sample
        atol: absolute tolerance
        rtol: relative tolerance
        verbose: print detailed results
        
    Returns:
        List of tuples: (group_element, max_error, mean_error, variance)
    """
    model.eval()
    n_group = model.n_group
    
    # Create base input
    base_input = create_test_input(batch_size, num_images, device=device)
    
    errors = []
    all_passed = True
    
    with torch.no_grad():
        # f_lang(x)
        output_original = model(BatchFeature(data=base_input))
        language_original = output_original["backbone_language_features"]
        
        if verbose:
            print(f"\nLanguage features shape: {language_original.shape}")
            print(f"Testing invariance for {n_group - 1} rotations...")
            print("-" * 70)
        
        for k in range(1, n_group):
            angle = k * 2 * math.pi / n_group
            
            # Rotate input
            rotated_pixel_values = rotate_image(
                base_input["eagle_pixel_values"],
                angle
            )
            
            rotated_input = {
                **base_input,
                "eagle_pixel_values": rotated_pixel_values
            }
            
            # f_lang(g*x) - should equal f_lang(x) for trivial repr
            output_rotated = model(BatchFeature(data=rotated_input))
            language_rotated = output_rotated["backbone_language_features"]
            
            # Compute errors (convert to float32 for numpy)
            errs = (language_rotated - language_original).float().cpu().numpy()
            errs_flat = np.abs(errs).reshape(-1)
            
            max_err = errs_flat.max()
            mean_err = errs_flat.mean()
            var_err = errs_flat.var()
            
            is_close = np.allclose(
                language_rotated.float().cpu().numpy(),
                language_original.float().cpu().numpy(),
                atol=atol,
                rtol=rtol
            )
            
            if verbose:
                status = "✓ PASS" if is_close else "✗ FAIL"
                print(f"  g_{k} ({math.degrees(angle):6.1f}°): "
                      f"max={max_err:.2e}, mean={mean_err:.2e}, var={var_err:.2e} [{status}]")
            
            if not is_close:
                all_passed = False
            
            errors.append((k, max_err, mean_err, var_err))
    
    if verbose:
        print("-" * 70)
        if all_passed:
            print(f"✓ Language invariance test PASSED!")
        else:
            print(f"✗ Language invariance test FAILED!")
    
    return errors


def load_backbone_from_checkpoint(checkpoint_path: str, n_group: int = 8, 
                                   num_images: int = 1, device: torch.device = None,
                                   use_hf_hub: bool = True):
    """
    Load EagleBackboneFA with weights from GR00T checkpoint.
    
    Args:
        checkpoint_path: path to GR00T checkpoint or HuggingFace hub model ID
        n_group: cyclic group order
        num_images: number of images per sample
        device: torch device
        use_hf_hub: if True, try to load from HuggingFace hub first
        
    Returns:
        EagleBackboneFA model with loaded weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading backbone from: {checkpoint_path}")
    
    # Create the model
    model = EagleBackboneFA(
        tune_llm=False,
        tune_visual=False,
        n_group=n_group,
        num_images_per_sample=num_images,
        project_to_dim=1152,  # Must be divisible by n_group
    )
    
    loaded = False
    
    # Try HuggingFace hub first if it looks like a hub path
    if use_hf_hub and "/" in checkpoint_path and not os.path.exists(checkpoint_path):
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            print(f"Attempting to load from HuggingFace hub: {checkpoint_path}")
            
            # List files in repo to find safetensors
            repo_files = list_repo_files(checkpoint_path)
            safetensor_files = [f for f in repo_files if f.endswith('.safetensors')]
            
            if safetensor_files:
                from safetensors.torch import load_file
                state_dict = {}
                
                for sf in safetensor_files:
                    print(f"  Downloading {sf}...")
                    local_path = hf_hub_download(checkpoint_path, sf)
                    shard_dict = load_file(local_path)
                    state_dict.update(shard_dict)
                
                # Filter for backbone weights
                backbone_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("backbone."):
                        new_key = key.replace("backbone.", "")
                        backbone_state_dict[new_key] = value
                
                if backbone_state_dict:
                    missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)
                    print(f"Loaded {len(backbone_state_dict)} backbone weights from HuggingFace hub")
                    if missing:
                        print(f"Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"Unexpected keys: {len(unexpected)}")
                    loaded = True
                    
        except Exception as e:
            print(f"Failed to load from HuggingFace hub: {e}")
            print("Falling back to local checkpoint...")
    
    # Try local checkpoint if not loaded from hub
    if not loaded and os.path.exists(checkpoint_path):
        # Check for sharded safetensors first (model.safetensors.index.json)
        index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
        checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
        
        if os.path.exists(index_file):
            import json
            from safetensors.torch import load_file
            
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            # Get unique shard files
            shard_files = set(index["weight_map"].values())
            
            # Check if all shard files exist
            all_shards_exist = all(
                os.path.exists(os.path.join(checkpoint_path, shard)) 
                for shard in shard_files
            )
            
            if all_shards_exist:
                print(f"Loading from {len(shard_files)} sharded safetensors files...")
                state_dict = {}
                for shard in shard_files:
                    shard_path = os.path.join(checkpoint_path, shard)
                    print(f"  Loading {shard}...")
                    shard_dict = load_file(shard_path)
                    state_dict.update(shard_dict)
                
                # Filter for backbone weights
                backbone_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("backbone."):
                        new_key = key.replace("backbone.", "")
                        backbone_state_dict[new_key] = value
                
                if backbone_state_dict:
                    missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)
                    print(f"Loaded {len(backbone_state_dict)} backbone weights")
                    if missing:
                        print(f"Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"Unexpected keys: {len(unexpected)}")
                    loaded = True
            else:
                missing_shards = [s for s in shard_files if not os.path.exists(os.path.join(checkpoint_path, s))]
                print(f"Warning: Sharded safetensors index found but shard files missing!")
                print(f"  Missing: {missing_shards[:3]}{'...' if len(missing_shards) > 3 else ''}")
                print(f"  Run 'cd {checkpoint_path} && git lfs pull' to download weights")
                
        elif os.path.exists(checkpoint_file):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_file)
            
            # Filter for backbone weights
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("backbone."):
                    new_key = key.replace("backbone.", "")
                    backbone_state_dict[new_key] = value
            
            if backbone_state_dict:
                missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)
                print(f"Loaded {len(backbone_state_dict)} backbone weights")
                loaded = True
        else:
            # Try pytorch checkpoint
            checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(checkpoint_file):
                state_dict = torch.load(checkpoint_file, map_location='cpu')
                
                backbone_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("backbone."):
                        new_key = key.replace("backbone.", "")
                        backbone_state_dict[new_key] = value
                
                if backbone_state_dict:
                    missing, unexpected = model.load_state_dict(backbone_state_dict, strict=False)
                    print(f"Loaded {len(backbone_state_dict)} backbone weights")
                    loaded = True
    
    if not loaded:
        print("Warning: Using random initialization (no weights loaded)")
    
    model = model.to(device)
    model.eval()
    
    return model


def main():
    """Main test function."""
    print("=" * 70)
    print("EagleBackboneFA Equivariance Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test parameters
    n_group = 8  # C8 group
    batch_size = 1
    num_images = 2
    
    # Tolerances for equivariance check
    atol = 1e-4
    rtol = 1e-3
    
    print(f"\nTest configuration:")
    print(f"  n_group (CN): C{n_group}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_images: {num_images}")
    print(f"  atol: {atol}")
    print(f"  rtol: {rtol}")
    
    # Initialize model
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)
    
    try:
        # Try to load from HuggingFace hub first, then local checkpoint
        model = load_backbone_from_checkpoint(
            BASE_MODEL_PATH,  # Try HuggingFace hub path first
            n_group=n_group,
            num_images=num_images,
            device=device,
            use_hf_hub=True
        )
        
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 1: Vision Equivariance
    print("\n" + "=" * 70)
    print("Test 1: Vision Features Equivariance (Regular Representation)")
    print("  f(g*x) = g*f(x) where g is rotation, f is backbone")
    print("=" * 70)
    
    try:
        vision_errors = check_equivariance(
            model, device,
            batch_size=batch_size,
            num_images=num_images,
            num_tests=None,  # Test all group elements
            atol=atol,
            rtol=rtol,
            verbose=True
        )
        
        vision_max_err = max(e[1] for e in vision_errors)
        vision_passed = vision_max_err < atol + rtol * vision_max_err
        
    except Exception as e:
        print(f"Vision equivariance test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        vision_passed = False
    
    # Test 2: Language Invariance
    print("\n" + "=" * 70)
    print("Test 2: Language Features Invariance (Trivial Representation)")
    print("  f_lang(g*x) = f_lang(x) - language should not change with image rotation")
    print("=" * 70)
    
    try:
        language_errors = check_language_invariance(
            model, device,
            batch_size=batch_size,
            num_images=num_images,
            atol=atol,
            rtol=rtol,
            verbose=True
        )
        
        language_max_err = max(e[1] for e in language_errors)
        language_passed = language_max_err < atol + rtol * language_max_err
        
    except Exception as e:
        print(f"Language invariance test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        language_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    all_passed = vision_passed and language_passed
    
    print(f"\n  Vision Equivariance:   {'✓ PASSED' if vision_passed else '✗ FAILED'}")
    print(f"  Language Invariance:   {'✓ PASSED' if language_passed else '✗ FAILED'}")
    
    if all_passed:
        print("\n✓ All equivariance tests PASSED!")
    else:
        print("\n✗ Some equivariance tests FAILED!")
        print("\nNote: Large errors may indicate:")
        print("  1. The backbone is not truly equivariant")
        print("  2. Numerical precision issues")
        print("  3. Interpolation artifacts from image rotation")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
