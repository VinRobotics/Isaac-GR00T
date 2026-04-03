# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to verify equivariance of:
  - MultiEmbodimentActionEncoder
  - EquiCategorySpecificMLP  (as state_encoder, action_decoder)

Equivariance property: f(g · x) = ρ_out(g) · f(x)
Where:
  - g  is a rotation from the cyclic group C_N
  - x  is the input GeometricTensor
  - f  is the module under test
  - ρ(g) is the group representation acting on output features

Representations used:
  - trivial_repr (size 1)  : no change under g
  - irrep(1)    (size 2)  : 2×2 rotation matrix R(r·2π/N)
  - regular_repr (size N) : cyclic block-shift by r positions

Testing approach:
  1. x_transformed = ρ_in(g) · x          (transform input)
  2. out_rot       = f(x_transformed)      (forward on transformed input)
  3. out_orig      = f(x)                  (forward on original input)
  4. expected      = ρ_out(g) · out_orig   (transform output)
  5. Check: out_rot ≈ expected  (within tolerance)
"""

import sys
import math
import numpy as np
import torch
from typing import List, Tuple

from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup

from gr00t.model.action_head.flow_matching_action_head import (
    EquiCategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


# ─────────────────────────────────────────────────────────────────────────────
# Group action helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_group_action(
    x: torch.Tensor,
    field_type: enn.FieldType,
    r: int,
    n_group: int,
) -> torch.Tensor:
    """
    Apply cyclic group element g_r to a batch of feature vectors.

    Args:
        x          : [B, D] float32 tensor
        field_type : escnn FieldType describing the feature layout
        r          : group element index (0 … N-1)
        n_group    : cyclic group order N

    Returns:
        Transformed tensor [B, D]
    """
    # Build the cyclic group element once
    G = field_type.gspace.fibergroup
    g = G.element(r)

    result_blocks = []
    offset = 0
    for rep in field_type.representations:
        size = rep.size
        block = x[:, offset : offset + size]                        # [B, size]
        R = torch.tensor(rep(g), dtype=x.dtype, device=x.device)   # [size, size]
        result_blocks.append(block @ R.T)
        offset += size

    return torch.cat(result_blocks, dim=-1)   # [B, D]


# ─────────────────────────────────────────────────────────────────────────────
# Random input helpers
# ─────────────────────────────────────────────────────────────────────────────

def random_geometric_tensor(
    batch: int,
    field_type: enn.FieldType,
    seed: int = 42,
) -> enn.GeometricTensor:
    torch.manual_seed(seed)
    x = torch.randn(batch, field_type.size)
    return enn.GeometricTensor(x, field_type)


def random_cat_ids(batch: int, num_categories: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, num_categories, (batch,))


# ─────────────────────────────────────────────────────────────────────────────
# Equivariance checker
# ─────────────────────────────────────────────────────────────────────────────

def check_equivariance_mlp(
    mlp: EquiCategorySpecificMLP,
    in_type: enn.FieldType,
    out_type: enn.FieldType,
    n_group: int,
    batch: int = 4,
    num_categories: int = 3,
    atol: float = 1e-4,
    verbose: bool = True,
    label: str = "EquiCategorySpecificMLP",
) -> bool:
    """
    Test equivariance of EquiCategorySpecificMLP.

    Checks: mlp(g·x, cat_ids) ≈ g·mlp(x, cat_ids)  for all g in C_N.
    """
    mlp.eval()
    G = in_type.gspace.fibergroup

    cat_ids = random_cat_ids(batch, num_categories)
    x_geom = random_geometric_tensor(batch, in_type)

    all_passed = True
    results: List[Tuple[int, float, float]] = []

    with torch.no_grad():
        out_orig = mlp(x_geom, cat_ids)             # GeometricTensor [B, D_out]
        out_orig_t = out_orig.tensor.float()

        for r in range(n_group):
            angle_deg = r * 360.0 / n_group

            # g · x  (transform input)
            x_rotated_t = apply_group_action(x_geom.tensor.float(), in_type, r, n_group)
            x_rotated = enn.GeometricTensor(x_rotated_t.to(x_geom.tensor.dtype), in_type)

            # f(g · x)
            out_rotated = mlp(x_rotated, cat_ids).tensor.float()

            # g · f(x)  (transform output)
            expected = apply_group_action(out_orig_t, out_type, r, n_group)

            diff = (out_rotated - expected).abs()
            max_err  = diff.max().item()
            mean_err = diff.mean().item()
            passed   = mean_err < atol

            results.append((r, max_err, mean_err))
            if not passed:
                all_passed = False

            if verbose:
                status = "✓" if passed else "✗"
                print(f"  [{label}] g_{r:2d} ({angle_deg:6.1f}°): "
                      f"max={max_err:.3e}  mean={mean_err:.3e}  [{status}]")

    return all_passed, results


def check_equivariance_action_encoder(
    encoder: MultiEmbodimentActionEncoder,
    action_in_type: enn.FieldType,
    action_out_type: enn.FieldType,
    n_group: int,
    batch: int = 4,
    num_embodiments: int = 3,
    atol: float = 1e-4,
    verbose: bool = True,
) -> bool:
    """
    Test equivariance of MultiEmbodimentActionEncoder.

    The timestep embedding uses trivial (invariant) representation, so only
    the action features (regular or irrep(1)) transform under g.

    Checks: enc(g·x, t, cat_ids) ≈ g·enc(x, t, cat_ids)  for all g in C_N.
    """
    encoder.eval()

    cat_ids  = random_cat_ids(batch, num_embodiments)
    x_geom   = random_geometric_tensor(batch, action_in_type)
    timesteps = torch.randint(0, 1000, (batch,)).float()

    all_passed = True
    results: List[Tuple[int, float, float]] = []

    with torch.no_grad():
        out_orig = encoder(x_geom, timesteps, cat_ids)          # GeometricTensor [B, D]
        out_orig_t = out_orig.tensor.float()

        for r in range(n_group):
            angle_deg = r * 360.0 / n_group

            # g · x  (transform action input)
            x_rotated_t = apply_group_action(x_geom.tensor.float(), action_in_type, r, n_group)
            x_rotated = enn.GeometricTensor(x_rotated_t.to(x_geom.tensor.dtype), action_in_type)

            # f(g · x, t)   — timestep is scalar / invariant
            out_rotated = encoder(x_rotated, timesteps, cat_ids).tensor.float()

            # g · f(x, t)
            expected = apply_group_action(out_orig_t, action_out_type, r, n_group)

            diff = (out_rotated - expected).abs()
            max_err  = diff.max().item()
            mean_err = diff.mean().item()
            passed   = mean_err < atol

            results.append((r, max_err, mean_err))
            if not passed:
                all_passed = False

            if verbose:
                status = "✓" if passed else "✗"
                print(f"  [MultiEmbodimentActionEncoder] g_{r:2d} ({angle_deg:6.1f}°): "
                      f"max={max_err:.3e}  mean={mean_err:.3e}  [{status}]")

    return all_passed, results


# ─────────────────────────────────────────────────────────────────────────────
# FieldType factory — mirrors the actual usage in FlowmatchingActionHead
# ─────────────────────────────────────────────────────────────────────────────

def make_joint_field_type(
    group: gspaces.GSpace,
    n_group: int,
    num_hand: int,
    max_dim: int,
    ee_dim: int = 7,
) -> enn.FieldType:
    """
    Builds the same FieldType as FlowmatchingActionHead.getJointFieldType().

    Layout:
      num_hand * 4  irrep(1) blocks  (xy pos + 6D rot encoded as 4 × irrep(1))
      remainder     trivial blocks   (z, gripper, …)
    """
    n_trivial = max_dim - (ee_dim - 1) * num_hand
    return enn.FieldType(
        group,
        num_hand * 4 * [group.irrep(1)]
        + n_trivial * [group.trivial_repr],
    )


def make_regular_field_type(
    group: gspaces.GSpace,
    n_group: int,
    total_size: int,
) -> enn.FieldType:
    """Regular representation FieldType with total_size = n_group * num_blocks."""
    assert total_size % n_group == 0, (
        f"total_size ({total_size}) must be divisible by n_group ({n_group})"
    )
    num_blocks = total_size // n_group
    return enn.FieldType(group, num_blocks * [group.regular_repr])


# ─────────────────────────────────────────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Equivariance Tests: MultiEmbodimentActionEncoder & EquiCategorySpecificMLP")
    print("=" * 70)

    device = torch.device("cpu")   # These modules work fine on CPU
    n_group     = 4                # C4 — fast to test
    num_hand    = 1
    ee_dim      = 7                # quaternion
    max_dim     = 14               # must satisfy: max_dim - (ee_dim-1)*num_hand > 0
    #   joint irrep(1) size = num_hand * 4 * 2 = 8
    #   trivial size        = max_dim - 6     = 8
    #   total in_type size  = 16

    hidden_size         = 32       # divisible by n_group
    input_embedding_dim = 32       # divisible by n_group
    num_categories      = 3
    batch               = 6
    atol                = 1e-4

    print(f"\nConfiguration:")
    print(f"  cyclic group   : C{n_group}")
    print(f"  num_hand       : {num_hand}")
    print(f"  max_dim        : {max_dim}  (ee_dim={ee_dim})")
    print(f"  hidden_size    : {hidden_size}")
    print(f"  embed_dim      : {input_embedding_dim}")
    print(f"  num_categories : {num_categories}")
    print(f"  batch          : {batch}")
    print(f"  atol           : {atol}")

    # Build gspace & field types
    group = gspaces.no_base_space(CyclicGroup(n_group))

    joint_type  = make_joint_field_type(group, n_group, num_hand, max_dim, ee_dim)
    hidden_type = make_regular_field_type(group, n_group, hidden_size)
    out_type    = make_regular_field_type(group, n_group, input_embedding_dim)

    print(f"\nField type sizes:")
    print(f"  joint_type  (irrep(1)+trivial) : {joint_type.size}")
    print(f"  hidden_type (regular)          : {hidden_type.size}")
    print(f"  out_type    (regular)          : {out_type.size}")

    all_passed = True

    # ── Test 1: EquiCategorySpecificMLP  joint → hidden → out (state encoder) ──
    print("\n" + "─" * 70)
    print("Test 1: EquiCategorySpecificMLP  [joint → hidden → out]  (state_encoder)")
    print("  in : irrep(1) + trivial")
    print("  out: regular_repr")
    print("─" * 70)

    state_encoder = EquiCategorySpecificMLP(
        num_categories=num_categories,
        in_type=joint_type,
        hidden_type=hidden_type,
        out_type=out_type,
    )

    passed1, _ = check_equivariance_mlp(
        state_encoder, joint_type, out_type,
        n_group=n_group, batch=batch, num_categories=num_categories,
        atol=atol, verbose=True, label="state_encoder",
    )
    all_passed = all_passed and passed1
    print(f"→ {'PASSED ✓' if passed1 else 'FAILED ✗'}")

    # ── Test 2: EquiCategorySpecificMLP  hidden → hidden → joint (action decoder) ──
    print("\n" + "─" * 70)
    print("Test 2: EquiCategorySpecificMLP  [hidden → hidden → joint]  (action_decoder)")
    print("  in : regular_repr")
    print("  out: irrep(1) + trivial")
    print("─" * 70)

    action_decoder = EquiCategorySpecificMLP(
        num_categories=num_categories,
        in_type=hidden_type,
        hidden_type=hidden_type,
        out_type=joint_type,
    )

    passed2, _ = check_equivariance_mlp(
        action_decoder, hidden_type, joint_type,
        n_group=n_group, batch=batch, num_categories=num_categories,
        atol=atol, verbose=True, label="action_decoder",
    )
    all_passed = all_passed and passed2
    print(f"→ {'PASSED ✓' if passed2 else 'FAILED ✗'}")

    # ── Test 3: EquiCategorySpecificMLP  regular → regular (pure regular) ──
    print("\n" + "─" * 70)
    print("Test 3: EquiCategorySpecificMLP  [regular → regular → regular]")
    print("  in : regular_repr")
    print("  out: regular_repr")
    print("─" * 70)

    mlp_regular = EquiCategorySpecificMLP(
        num_categories=num_categories,
        in_type=out_type,
        hidden_type=hidden_type,
        out_type=out_type,
    )

    passed3, _ = check_equivariance_mlp(
        mlp_regular, out_type, out_type,
        n_group=n_group, batch=batch, num_categories=num_categories,
        atol=atol, verbose=True, label="regular→regular",
    )
    all_passed = all_passed and passed3
    print(f"→ {'PASSED ✓' if passed3 else 'FAILED ✗'}")

    # ── Test 4: MultiEmbodimentActionEncoder ──
    print("\n" + "─" * 70)
    print("Test 4: MultiEmbodimentActionEncoder  [joint → out]")
    print("  in : irrep(1) + trivial  (action with pos/rot)")
    print("  out: regular_repr")
    print("  timestep: trivial (invariant) — not transformed")
    print("─" * 70)

    action_encoder = MultiEmbodimentActionEncoder(
        in_type=joint_type,
        out_type=out_type,
        num_embodiments=num_categories,
    )

    passed4, _ = check_equivariance_action_encoder(
        action_encoder, joint_type, out_type,
        n_group=n_group, batch=batch, num_embodiments=num_categories,
        atol=atol, verbose=True,
    )
    all_passed = all_passed and passed4
    print(f"→ {'PASSED ✓' if passed4 else 'FAILED ✗'}")

    # ── Test 5: MultiEmbodimentActionEncoder with larger C8 group ──
    print("\n" + "─" * 70)
    print("Test 5: MultiEmbodimentActionEncoder  [C8 group — larger symmetry]")
    print("─" * 70)

    n_group8    = 8
    hidden8     = 64
    embed8      = 64
    max_dim8    = 14   # same layout, but group is larger

    group8      = gspaces.no_base_space(CyclicGroup(n_group8))
    joint8      = make_joint_field_type(group8, n_group8, num_hand, max_dim8, ee_dim)
    hidden_t8   = make_regular_field_type(group8, n_group8, hidden8)
    out_t8      = make_regular_field_type(group8, n_group8, embed8)

    enc8 = MultiEmbodimentActionEncoder(
        in_type=joint8, out_type=out_t8, num_embodiments=num_categories,
    )

    passed5, _ = check_equivariance_action_encoder(
        enc8, joint8, out_t8,
        n_group=n_group8, batch=batch, num_embodiments=num_categories,
        atol=atol, verbose=True,
    )
    all_passed = all_passed and passed5
    print(f"→ {'PASSED ✓' if passed5 else 'FAILED ✗'}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    tests = [
        ("EquiCategorySpecificMLP  (state_encoder,  joint→out)",   passed1),
        ("EquiCategorySpecificMLP  (action_decoder, hidden→joint)", passed2),
        ("EquiCategorySpecificMLP  (regular→regular→regular)",      passed3),
        ("MultiEmbodimentActionEncoder  (C4)",                      passed4),
        ("MultiEmbodimentActionEncoder  (C8)",                      passed5),
    ]
    for name, p in tests:
        print(f"  {'✓ PASSED' if p else '✗ FAILED'}  {name}")

    if all_passed:
        print("\n✓ All equivariance tests PASSED!")
    else:
        print("\n✗ Some equivariance tests FAILED!")
        print("\nNote: failures may indicate:")
        print("  1. The module is not equivariant (bug in the implementation)")
        print("  2. Numerical precision issues (try increasing atol)")
        print("  3. Incorrect FieldType passed to apply_group_action")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
