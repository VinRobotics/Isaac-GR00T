# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to verify that EagleBackboneFATokens is equivariant on images.

This version tests the full vision token backbone (not pooled).

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
from gr00t.model.backbone.eagle_backbone_fa_tokens import EagleBackboneFATokens
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
                      dtype: torch.dtype = torch.bfloat16,
                      save_visualization: bool = False,
                      save_path: str = "test_input_image.png") -> dict:
    """
    Create test vision-language input with structured (non-symmetric) pattern.
    Each batch sample and each image within a sample has a unique pattern.
    
    Args:
        batch_size: batch size
        num_images: number of images per sample
        img_size: image height/width
        seq_len: sequence length for text tokens
        device: torch device
        dtype: torch dtype
        save_visualization: if True, save the test image as PNG
        save_path: path to save the visualization
        
    Returns:
        Dictionary with eagle_ prefixed keys
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create images with non-rotationally-symmetric pattern
    pixel_values = torch.zeros(batch_size * num_images, 3, img_size, img_size, 
                               device=device, dtype=dtype)
    
    # Create different patterns for each batch and each image
    for img_idx in range(batch_size * num_images):
        batch_idx = img_idx // num_images  # Which batch sample
        image_idx = img_idx % num_images   # Which image within the sample
        
        center = img_size // 2
        
        if image_idx == 0:
            # First image in each batch: gradient + diagonal pattern
            for i in range(img_size):
                for j in range(img_size):
                    # Horizontal gradient in red channel (shifted by batch)
                    pixel_values[img_idx, 0, i, j] = ((j + batch_idx * 30) % img_size) / img_size
                    # Vertical gradient in green channel (shifted by batch)
                    pixel_values[img_idx, 1, i, j] = ((i + batch_idx * 50) % img_size) / img_size
                    # Diagonal pattern in blue channel (different angle per batch)
                    threshold = j + batch_idx * 20
                    if i > threshold % img_size:
                        pixel_values[img_idx, 2, i, j] = 0.8
                    elif i < (threshold - img_size // 4) % img_size:
                        pixel_values[img_idx, 2, i, j] = 0.3
        else:
            # Second+ image in each batch: circle + stripe pattern
            circle_radius = img_size // 4 + batch_idx * 10
            stripe_width = 15 + image_idx * 5 + batch_idx * 3
            
            cx = center + batch_idx * 15
            cy = center - batch_idx * 10
            
            for i in range(img_size):
                for j in range(img_size):
                    # Circle pattern in red channel
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist < circle_radius:
                        pixel_values[img_idx, 0, i, j] = 0.9 - batch_idx * 0.1
                    else:
                        pixel_values[img_idx, 0, i, j] = 0.2 + batch_idx * 0.1
                    
                    # Stripe pattern in green channel
                    if j % stripe_width < stripe_width // 2:
                        pixel_values[img_idx, 1, i, j] = 0.7
                    else:
                        pixel_values[img_idx, 1, i, j] = 0.3
                    
                    # Corner marker in blue channel
                    corner_size = img_size // 4
                    if batch_idx % 4 == 0:
                        in_corner = i < corner_size and j > img_size - corner_size
                    elif batch_idx % 4 == 1:
                        in_corner = i < corner_size and j < corner_size
                    elif batch_idx % 4 == 2:
                        in_corner = i > img_size - corner_size and j < corner_size
                    else:
                        in_corner = i > img_size - corner_size and j > img_size - corner_size
                    
                    if in_corner:
                        pixel_values[img_idx, 2, i, j] = 1.0
                    else:
                        pixel_values[img_idx, 2, i, j] = 0.1
    
    # Add small random noise
    pixel_values = pixel_values + 0.05 * torch.randn_like(pixel_values)
    pixel_values = torch.clamp(pixel_values, 0, 1)
    
    # Save visualization if requested
    if save_visualization:
        save_test_image(pixel_values, save_path)
    
    # Create dummy text tokens
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    
    return {
        "eagle_pixel_values": pixel_values,
        "eagle_input_ids": input_ids,
        "eagle_attention_mask": attention_mask,
        "attention_mask": attention_mask,
    }


def save_test_image(pixel_values: torch.Tensor, save_path: str = "test_input_image.png"):
    """Save test images as PNG for visualization."""
    from PIL import Image
    
    imgs = pixel_values.float().cpu().numpy()
    num_images = imgs.shape[0]
    
    if num_images == 1:
        img = imgs[0].transpose(1, 2, 0)  # CHW -> HWC
        img = (img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(save_path)
        print(f"Saved test image to: {save_path}")
    else:
        H, W = imgs.shape[2], imgs.shape[3]
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols
        
        grid = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
        
        for idx in range(num_images):
            row = idx // cols
            col = idx % cols
            img = imgs[idx].transpose(1, 2, 0)
            img = (img * 255).clip(0, 255).astype(np.uint8)
            grid[row*H:(row+1)*H, col*W:(col+1)*W, :] = img
        
        Image.fromarray(grid).save(save_path)
        print(f"Saved {num_images} test images as grid to: {save_path}")


def create_token_permutation_indices(grid_size: int, n_group: int) -> torch.Tensor:
    """
    Create token permutation indices for rotations.
    
    Args:
        grid_size: size of token grid (e.g., 16 for 16x16=256 tokens)
        n_group: number of rotations in cyclic group
        
    Returns:
        Tensor of shape [n_group, num_tokens] with permutation indices
    """
    num_tokens = grid_size * grid_size
    token_perm_indices = torch.zeros(n_group, num_tokens, dtype=torch.long)
    
    for r in range(n_group):
        for i in range(grid_size):
            for j in range(grid_size):
                orig_idx = i * grid_size + j
                
                # Center coordinates
                ci = i - (grid_size - 1) / 2
                cj = j - (grid_size - 1) / 2
                
                # Rotation angle
                theta = r * 2 * math.pi / n_group
                
                # Rotate (inverse to find source)
                cos_t = math.cos(-theta)
                sin_t = math.sin(-theta)
                
                new_ci = cos_t * ci - sin_t * cj
                new_cj = sin_t * ci + cos_t * cj
                
                # Convert back to grid
                new_i = int(round(new_ci + (grid_size - 1) / 2))
                new_j = int(round(new_cj + (grid_size - 1) / 2))
                
                new_i = max(0, min(grid_size - 1, new_i))
                new_j = max(0, min(grid_size - 1, new_j))
                
                new_idx = new_i * grid_size + new_j
                token_perm_indices[r, orig_idx] = new_idx
    
    return token_perm_indices


def check_equivariance_tokens(
    model: EagleBackboneFATokens,
    device: torch.device,
    batch_size: int = 1,
    num_images: int = 2,
    num_tests: int = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    verbose: bool = True
) -> List[Tuple[Any, float, float, float]]:
    """
    Test equivariance of EagleBackboneFATokens output.
    
    Equivariance property: f(g·x) = ρ(g) · f(x)
    
    Where:
    - g·x = rotated input image
    - f = FA backbone
    - ρ(g) = output representation = spatial_perm(g) ⊗ feature_perm(g)
    
    The FA construction guarantees this property:
    FA(g·x) = ρ(g) · FA(x)
    
    Args:
        model: EagleBackboneFATokens model
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
    grid_size = model.token_grid_size

    # Create base input and save visualization
    base_input = create_test_input(batch_size, num_images, device=device, 
                                   save_visualization=True, 
                                   save_path="test_tokens_input_image.png")
    
    errors = []
    all_passed = True
    group = gspaces.no_base_space(CyclicGroup(n_group))
    
    # Get token permutation indices from model
    token_perm_indices = model.token_perm_indices
    
    with torch.no_grad():
        # f(x) = FA(x)
        output_original = model(BatchFeature(data=base_input))
        vision_original = output_original["backbone_vision_features"].float()
        
        B, num_imgs, T_vision, D_vision = vision_original.shape
        
        if verbose:
            print(f"\nVision features shape: {vision_original.shape}")
            print(f"  Batch size: {B}")
            print(f"  Num images: {num_imgs}")
            print(f"  Num vision tokens: {T_vision}")
            print(f"  Vision dim: {D_vision}")
            print(f"  Token grid: {grid_size}x{grid_size}")
            print(f"N_group (CN): C{n_group}")
            print("-" * 70)
            print("Testing EQUIVARIANCE: f(g·x) ≈ ρ(g) · f(x)")
            print("Where ρ(g) = spatial_perm(g) ⊗ feature_perm(g)")
            print("-" * 70)
        
        for k in list(group.testing_elements):
            angle = k.value * 2 * math.pi / n_group
            r = k.value
            
            if verbose:
                print(f"  Testing g_{r} ({math.degrees(angle):6.1f}°)...")
            
            # g·x: Rotate the input images
            rotated_pixel_values = rotate_image(
                base_input["eagle_pixel_values"],
                angle
            )
            
            save_test_image(rotated_pixel_values, f"test_tokens_rotated_image_g{r}.png")
            
            rotated_input = {
                **base_input,
                "eagle_pixel_values": rotated_pixel_values
            }
            
            # f(g·x) = FA(g·x)
            output_rotated = model(BatchFeature(data=rotated_input))
            vision_rotated = output_rotated["backbone_vision_features"].float()
            
            # Compute ρ(g) · f(x):
            # ρ(g) = spatial_perm(g) ⊗ feature_perm(g)
            # 1. Spatial permutation: token at p moves to π(g)(p)
            # 2. Feature permutation: regular repr cyclic shift by r
            
            if r == 0:
                expected = vision_original
            else:
                # Reshape: [B, num_imgs, T, D] -> [B*num_imgs, T, D]
                vis_flat = vision_original.reshape(B * num_imgs, T_vision, D_vision)
                
                # Step 1: Spatial permutation π(g)
                # token_perm_indices[r][i] = source index that moves TO position i
                # We want: result[i] = input[π(g)(i)] which is input[perm[i]]
                spatial_perm = token_perm_indices[r]
                vis_permuted = vis_flat[:, spatial_perm, :]
                
                # Step 2: Feature permutation (regular representation)
                # ρ_reg(g_r) shifts blocks by r positions
                blocks = D_vision // n_group
                vis_blocks = vis_permuted.reshape(B * num_imgs, T_vision, n_group, blocks)
                vis_shifted = torch.roll(vis_blocks, shifts=r, dims=2)
                vis_transformed = vis_shifted.reshape(B * num_imgs, T_vision, D_vision)
                
                expected = vis_transformed.reshape(B, num_imgs, T_vision, D_vision)
            
            # Check: f(g·x) ≈ ρ(g) · f(x)
            errs = (vision_rotated - expected).float().cpu().numpy()
            errs_flat = np.abs(errs).reshape(-1)
            
            max_err = errs_flat.max()
            mean_err = errs_flat.mean()
            var_err = errs_flat.var()
            
            is_close = mean_err < atol
            
            status = "✓ PASS" if is_close else "✗ FAIL"
            if verbose:
                print(f"    g_{r} ({math.degrees(angle):6.1f}°): "
                      f"max={max_err:.2e}, mean={mean_err:.2e} [{status}]")
            
            if not is_close:
                all_passed = False
                if verbose:
                    print(f"    ERROR: f(g·x) ≠ ρ(g)·f(x) for g_{r}!")
            
            errors.append((k, max_err, mean_err, var_err))
    
    if verbose:
        print("-" * 70)
        if all_passed:
            print(f"✓ All equivariance tests PASSED!")
        else:
            print(f"✗ Some equivariance tests FAILED!")
        
        max_errors = [e[1] for e in errors]
        mean_errors = [e[2] for e in errors]
        print(f"\nSummary:")
        print(f"  Overall max error: {max(max_errors):.2e}")
        print(f"  Overall mean error: {np.mean(mean_errors):.2e}")
    
    return errors


def check_language_invariance(
    model: EagleBackboneFATokens,
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
    """
    model.eval()
    n_group = model.n_group
    
    base_input = create_test_input(batch_size, num_images, device=device)
    
    errors = []
    all_passed = True
    
    with torch.no_grad():
        output_original = model(BatchFeature(data=base_input))
        language_original = output_original["backbone_language_features"]
        
        if verbose:
            print(f"\nLanguage features shape: {language_original.shape}")
            print(f"Testing invariance for {n_group - 1} rotations...")
            print("-" * 70)
        
        for k in range(1, n_group):
            angle = k * 2 * math.pi / n_group
            
            rotated_pixel_values = rotate_image(
                base_input["eagle_pixel_values"],
                angle
            )
            
            rotated_input = {
                **base_input,
                "eagle_pixel_values": rotated_pixel_values
            }
            
            output_rotated = model(BatchFeature(data=rotated_input))
            language_rotated = output_rotated["backbone_language_features"]
            
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
    Load EagleBackboneFATokens with weights from GR00T checkpoint.
    
    Args:
        checkpoint_path: path to GR00T checkpoint or HuggingFace hub model ID
        n_group: cyclic group order
        num_images: number of images per sample
        device: torch device
        use_hf_hub: if True, try to load from HuggingFace hub first
        
    Returns:
        EagleBackboneFATokens model with loaded weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading EagleBackboneFATokens from: {checkpoint_path}")
    
    # Create the model - note: project_to_dim must be divisible by n_group
    # Vision features are 2048 dim which is divisible by 8
    model = EagleBackboneFATokens(
        tune_llm=False,
        tune_visual=False,
        n_group=n_group,
        num_images_per_sample=num_images,
        project_to_dim=2048,  # Must be divisible by n_group, 2048 / 8 = 256
    )
    
    loaded = False
    
    # Try HuggingFace hub first
    if use_hf_hub and "/" in checkpoint_path and not os.path.exists(checkpoint_path):
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            print(f"Attempting to load from HuggingFace hub: {checkpoint_path}")
            
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
    
    # Try local checkpoint
    if not loaded and os.path.exists(checkpoint_path):
        index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
        checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
        
        if os.path.exists(index_file):
            import json
            from safetensors.torch import load_file
            
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            shard_files = set(index["weight_map"].values())
            
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


def compare_output_shapes():
    """Compare output shapes between pooled and full token versions."""
    print("\n" + "=" * 70)
    print("Output Shape Comparison")
    print("=" * 70)
    
    print("\nEagleBackboneFA (pooled):")
    print("  backbone_vision_features: [B, num_imgs, D_pool]")
    print("  - D_pool is typically 1152 (pooled feature dimension)")
    print("  - Loses spatial/token information")
    
    print("\nEagleBackboneFATokens (full tokens):")
    print("  backbone_vision_features: [B, num_imgs, T_vision, D_vision]")
    print("  - T_vision is typically 256 tokens (after pixel shuffle)")
    print("  - D_vision is 2048 (vision transformer hidden dim)")
    print("  - Preserves all spatial information")
    print("  - Better balance with state/action tokens in downstream processing")
    
    print("=" * 70)


def main():
    """Main test function."""
    print("=" * 70)
    print("EagleBackboneFATokens Equivariance Test")
    print("(Full Vision Tokens - Not Pooled)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test parameters
    n_group = 8  # C8 group (45° rotations)
    batch_size = 2
    num_images = 2
    
    # Tolerances - 0.05 is acceptable for equivariance
    atol = 0.05
    rtol = 0.01
    
    print(f"\nTest configuration:")
    print(f"  n_group (CN): C{n_group}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_images: {num_images}")
    print(f"  atol: {atol}")
    print(f"  rtol: {rtol}")
    
    # Show shape comparison
    compare_output_shapes()
    
    # Initialize model
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)
    
    try:
        model = load_backbone_from_checkpoint(
            BASE_MODEL_PATH,
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
    
    # Test 1: Vision Equivariance (Full Tokens)
    print("\n" + "=" * 70)
    print("Test 1: Vision Features Equivariance (Full Tokens)")
    print("  f(g*x) = g*f(x) where g is rotation, f is backbone")
    print("  Output shape: [B, num_imgs, T_vision, D_vision]")
    print("=" * 70)
    
    try:
        vision_errors = check_equivariance_tokens(
            model, device,
            batch_size=batch_size,
            num_images=num_images,
            num_tests=None,
            atol=atol,
            rtol=rtol,
            verbose=True
        )
        
        # With spatial realignment FA, all rotations should produce similar output
        # Check if all mean errors are below tolerance
        all_mean_errs = [e[2] for e in vision_errors]
        vision_passed = all(err < atol for err in all_mean_errs)
        
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
        language_passed = language_max_err < atol
        
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
    
    print(f"\n  Vision Equivariance (Tokens):   {'✓ PASSED' if vision_passed else '✗ FAILED'}")
    print(f"  Language Invariance:            {'✓ PASSED' if language_passed else '✗ FAILED'}")
    
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
