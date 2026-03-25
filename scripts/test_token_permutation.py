# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test token permutation correctness for EagleBackboneFATokens.

The core equivariance property we need to hold per-token:

    f(g · x)[p]  ==  f(x)[π(g)⁻¹ · p]

where:
  - g    : rotation from C_N
  - x    : input image
  - f    : vision encoder  (SigLIP)
  - p    : spatial token position
  - π(g) : token permutation induced by rotation g

Because we test WITHOUT the real vision encoder (too heavy), we simulate
equivariant features directly:

    synthetic_features[b, p, d]  =  positional_signal(p, d)

A rotation g maps token at position q to position π(g)·q, so:
    synthetic_features_rotated[b, p, d]  =  synthetic_features[b, π(g)⁻¹·p, d]

After applying _permute_tokens_for_rotation(r, inverse=False), the aligned
tokens should satisfy:

    aligned[b, p, :]  ==  original[b, p, :]   (up to numerical precision)

Tests
-----
1. test_identity          : rotation 0 → permutation is identity
2. test_c4_90deg          : C4, r=1 (90° CCW) — exact permutation
3. test_c4_composition    : two 90° rotations compose correctly
4. test_c4_full_orbit     : four 90° rotations return to identity
5. test_c8_90deg_multiples: C8 r=0,2,4,6 (exact 90° multiples)
6. test_inverse_perm      : forward then inverse permutation = identity
7. test_spatial_alignment : rotate image → extract positional tokens →
                            apply inverse perm → should equal original tokens
"""

import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Minimal stub of the permutation logic so we can test without loading Eagle
# ---------------------------------------------------------------------------

class TokenPermutationHelper:
    """
    Reproduces _init_token_permutation_indices and _permute_tokens_for_rotation
    from EagleBackboneFATokens without loading the full VLM.
    """

    def __init__(self, n_group: int, grid_size: int = 16):
        self.n_group = n_group
        self.token_grid_size = grid_size
        self._init_token_permutation_indices(grid_size)

    # Mirrors EagleBackboneFATokens._init_token_permutation_indices exactly
    def _init_token_permutation_indices(self, grid_size: int = 16):
        self.token_grid_size = grid_size
        num_tokens = grid_size * grid_size
        N = grid_size
        center = (N - 1) / 2

        token_perm_indices = torch.zeros(self.n_group, num_tokens, dtype=torch.long)

        for r in range(self.n_group):
            angle = r * 2 * math.pi / self.n_group
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            float_src_row = []
            float_src_col = []
            for i in range(N):
                for j in range(N):
                    ci = j - center
                    ri = i - center
                    float_src_col.append(center + cos_a * ci + sin_a * ri)
                    float_src_row.append(center - sin_a * ci + cos_a * ri)

            is_exact_90 = abs(math.sin(angle) ** 2 - round(math.sin(angle) ** 2)) < 1e-9

            if is_exact_90:
                for dest in range(num_tokens):
                    si = int(round(float_src_row[dest]))
                    sj = int(round(float_src_col[dest]))
                    si = max(0, min(N - 1, si))
                    sj = max(0, min(N - 1, sj))
                    token_perm_indices[r, dest] = si * N + sj
            else:
                rounding_err = [
                    (float_src_row[d] - round(float_src_row[d])) ** 2
                    + (float_src_col[d] - round(float_src_col[d])) ** 2
                    for d in range(num_tokens)
                ]
                dests_sorted = sorted(range(num_tokens), key=lambda d: rounding_err[d])

                assigned_src = [0] * num_tokens
                used_srcs: set = set()

                for dest in dests_sorted:
                    sr = float_src_row[dest]
                    sc = float_src_col[dest]
                    best_idx = None
                    best_dist = float("inf")
                    for radius in range(N):
                        for di in range(-radius, radius + 1):
                            for dj in range(-radius, radius + 1):
                                if max(abs(di), abs(dj)) != radius:
                                    continue
                                si = int(round(sr)) + di
                                sj = int(round(sc)) + dj
                                if not (0 <= si < N and 0 <= sj < N):
                                    continue
                                idx = si * N + sj
                                if idx in used_srcs:
                                    continue
                                dist = (sr - si) ** 2 + (sc - sj) ** 2
                                if dist < best_dist:
                                    best_dist = dist
                                    best_idx = idx
                        if best_idx is not None:
                            break
                    if best_idx is None:
                        best_idx = next(i for i in range(num_tokens) if i not in used_srcs)
                    assigned_src[dest] = best_idx
                    used_srcs.add(best_idx)

                for dest in range(num_tokens):
                    token_perm_indices[r, dest] = assigned_src[dest]

        arange = torch.arange(num_tokens, dtype=torch.long)
        for r in range(1, self.n_group):
            angle = r * 2 * math.pi / self.n_group
            is_exact_90 = abs(math.sin(angle) ** 2 - round(math.sin(angle) ** 2)) < 1e-9
            if not is_exact_90:
                r_inv = (self.n_group - r) % self.n_group
                if r_inv > r:
                    inv_perm = torch.zeros(num_tokens, dtype=torch.long)
                    inv_perm[token_perm_indices[r]] = arange
                    token_perm_indices[r_inv] = inv_perm

        self.token_perm_indices = token_perm_indices

    # Copied verbatim from EagleBackboneFATokens
    def _permute_tokens_for_rotation(
        self,
        tokens: torch.Tensor,
        rotation_idx: int,
        inverse: bool = False,
    ) -> torch.Tensor:
        if rotation_idx == 0:
            return tokens

        if inverse:
            perm = self.token_perm_indices[(self.n_group - rotation_idx) % self.n_group]
        else:
            perm = self.token_perm_indices[rotation_idx]

        return tokens[:, perm, :]


# ---------------------------------------------------------------------------
# Synthetic equivariant feature generator
# ---------------------------------------------------------------------------

def make_positional_features(grid_size: int, feat_dim: int = 32) -> torch.Tensor:
    """
    [1, T, D] features where value at token (i,j) encodes its grid position.
    Using sin/cos of position so features are unique and well-scaled.
    """
    N = grid_size
    T = N * N
    feats = torch.zeros(1, T, feat_dim)
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            for d in range(feat_dim):
                if d % 2 == 0:
                    feats[0, idx, d] = math.sin(math.pi * i / N * (d // 2 + 1))
                else:
                    feats[0, idx, d] = math.cos(math.pi * j / N * (d // 2 + 1))
    return feats


def rotate_features_groundtruth(
    feats: torch.Tensor,
    rotation_idx: int,
    helper: TokenPermutationHelper,
) -> torch.Tensor:
    """
    Ground-truth rotated features: feats[b, π(g)⁻¹ · p, :].

    When we rotate the image by g, token at position q ends up at position p
    where perm[p] = q (i.e. token p gets content from source q = perm[p]).
    So rotated_feats[b, p, :] = feats[b, perm[p], :] = feats[:, perm, :].

    This is exactly what _permute_tokens_for_rotation(r, inverse=False) does.
    """
    return helper._permute_tokens_for_rotation(feats, rotation_idx, inverse=False)


def rotate_image_grid(grid: torch.Tensor, rotation_idx: int, n_group: int) -> torch.Tensor:
    """
    Rotate a [1, C, H, W] image tensor by (rotation_idx * 360 / n_group) degrees CCW.
    """
    angle = rotation_idx * 2 * math.pi / n_group
    cos_t = math.cos(-angle)
    sin_t = math.sin(-angle)
    theta = torch.tensor([[cos_t, -sin_t, 0.0],
                           [sin_t,  cos_t, 0.0]], dtype=torch.float32).unsqueeze(0)
    grid_samp = F.affine_grid(theta, size=grid.shape, align_corners=True)
    return F.grid_sample(grid, grid_samp, align_corners=True, padding_mode='zeros')


# ---------------------------------------------------------------------------
# Helper: extract token positions from an image using nearest-neighbor lookup
# ---------------------------------------------------------------------------

def image_to_tokens(img: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    Simulate a vision encoder that maps [1, C, H, W] → [1, T, C] by
    splitting the image into a (grid_size × grid_size) grid of patches and
    taking the mean pixel value of each patch as the 'feature'.

    This is patch-exact for square images where H == W and H % grid_size == 0.
    """
    B, C, H, W = img.shape
    assert H == W and H % grid_size == 0
    patch_size = H // grid_size
    T = grid_size * grid_size
    tokens = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # tokens: [B, C, G, G, patch_size, patch_size]
    tokens = tokens.permute(0, 2, 3, 1, 4, 5).reshape(B, T, C * patch_size * patch_size)
    return tokens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(name: str, condition: bool, extra: str = ""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if extra:
        msg += f"  ({extra})"
    print(msg)
    return condition


def test_identity(helper: TokenPermutationHelper):
    """Rotation 0 must be identity permutation."""
    T = helper.token_grid_size ** 2
    expected = torch.arange(T)
    actual = helper.token_perm_indices[0]
    return check("identity permutation (r=0)", torch.equal(actual, expected))


def test_c4_90deg(helper: TokenPermutationHelper):
    """For C4, rotation r=1 (90° CW): position (i,j) gets content from (N-1-j, i)."""
    N = helper.token_grid_size
    passed = True
    for i in range(N):
        for j in range(N):
            dest = i * N + j
            # CW 90°: src_row = N-1-j, src_col = i
            expected_src_i = N - 1 - j
            expected_src_j = i
            expected_src = expected_src_i * N + expected_src_j
            actual_src = helper.token_perm_indices[1, dest].item()
            if actual_src != expected_src:
                print(f"    Mismatch at ({i},{j}): expected src ({expected_src_i},{expected_src_j})={expected_src}, got {actual_src}")
                passed = False
                break
        if not passed:
            break
    return check("C4 r=1 (90° CW) permutation matches closed-form", passed)


def test_c4_composition(helper: TokenPermutationHelper):
    """Applying perm[1] twice should equal perm[2]."""
    p1 = helper.token_perm_indices[1]
    p2 = helper.token_perm_indices[2]
    composed = p1[p1]  # apply perm[1] twice
    return check("C4 perm[1]∘perm[1] == perm[2]", torch.equal(composed, p2))


def test_c4_full_orbit(helper: TokenPermutationHelper):
    """Applying perm[1] four times must return to identity."""
    T = helper.token_grid_size ** 2
    p = helper.token_perm_indices[1]
    composed = p
    for _ in range(3):
        composed = p[composed]
    identity = torch.arange(T)
    return check("C4 perm[1]^4 == identity", torch.equal(composed, identity))


def test_c8_90deg_multiples(helper: TokenPermutationHelper):
    """For C8, even rotations (r=0,2,4,6) must match exact 90° CW multiples."""
    N = helper.token_grid_size
    passed = True

    def rotate_90_cw_source(i, j, N):
        # CW 90°: src_row = N-1-j, src_col = i
        return N - 1 - j, i

    for r_c8 in [0, 2, 4, 6]:
        num_90 = r_c8 // 2
        for i in range(N):
            for j in range(N):
                si, sj = i, j
                for _ in range(num_90):
                    si, sj = rotate_90_cw_source(si, sj, N)
                expected_src = si * N + sj
                actual_src = helper.token_perm_indices[r_c8, i * N + j].item()
                if actual_src != expected_src:
                    print(f"    C8 r={r_c8}: mismatch at ({i},{j}): "
                          f"expected src {expected_src}, got {actual_src}")
                    passed = False
                    break
            if not passed:
                break
        if not passed:
            break
    return check("C8 even rotations match exact 90° CW multiples", passed)


def test_inverse_perm(helper: TokenPermutationHelper):
    """forward then inverse permutation must equal identity for all r."""
    N = helper.token_grid_size
    T = N * N
    D = 8
    feats = torch.randn(1, T, D)
    all_pass = True
    for r in range(1, helper.n_group):
        fwd = helper._permute_tokens_for_rotation(feats, r, inverse=False)
        inv = helper._permute_tokens_for_rotation(fwd, r, inverse=True)
        diff = (inv - feats).abs().max().item()
        if diff > 1e-6:
            print(f"    r={r}: forward then inverse diff={diff:.2e}")
            all_pass = False
    return check("forward ∘ inverse == identity (all r)", all_pass)


def test_spatial_alignment(helper: TokenPermutationHelper, grid_size: int = 16):
    """
    Core equivariance test using a uniform-patch synthetic image.

    Each patch (gi, gj) has ALL pixels equal to the scalar value gi*G+gj.
    Because all pixels within a patch are identical, within-patch reordering
    under rotation doesn't change the extracted token vectors — only the
    between-patch permutation matters.

    Property checked:
        rotate_image(img, g) → extract_tokens → apply_inv_perm
        ==
        extract_tokens(img)
    """
    N = grid_size
    ps = 4  # pixels per patch side
    H = W = N * ps
    C = 3

    # Build image: patch (gi, gj) = uniform color gi*N+gj (normalized to [0,1])
    img = torch.zeros(1, C, H, W)
    for gi in range(N):
        for gj in range(N):
            img[:, :, gi * ps:(gi + 1) * ps, gj * ps:(gj + 1) * ps] = float(gi * N + gj) / (N * N)

    # Only test exact 90° multiples (bijective patch permutation, no interpolation ambiguity)
    step = helper.n_group // 4
    exact_r = [step * k for k in range(1, 4)]  # e.g. [1,2,3] for C4, [2,4,6] for C8

    all_pass = True
    for r in exact_r:
        rotated = rotate_image_grid(img, r, helper.n_group)
        T_rot = image_to_tokens(rotated, grid_size)
        T_aligned = helper._permute_tokens_for_rotation(T_rot, r, inverse=True)
        T_orig = image_to_tokens(img, grid_size)

        diff = (T_aligned - T_orig).abs().max().item()
        ok = diff < 1e-4
        if not ok:
            print(f"    r={r}: max diff={diff:.6f}")
        all_pass = all_pass and ok

    return check(
        "spatial alignment: rotate→encode→inv_perm == encode (uniform-patch image)",
        all_pass,
        f"tested r={exact_r}",
    )


def test_permutation_is_bijection(helper: TokenPermutationHelper):
    """Every permutation must be a bijection (no duplicates, no missing)."""
    T = helper.token_grid_size ** 2
    all_pass = True
    for r in range(helper.n_group):
        perm = helper.token_perm_indices[r]
        unique = perm.unique()
        if len(unique) != T:
            print(f"    r={r}: permutation has {len(unique)} unique values, expected {T}")
            all_pass = False
        if perm.min().item() < 0 or perm.max().item() >= T:
            print(f"    r={r}: permutation out of range [{perm.min()}, {perm.max()}]")
            all_pass = False
    return check("all permutations are bijections", all_pass)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_suite(n_group: int, grid_size: int = 16):
    print(f"\n{'='*60}")
    print(f"  C{n_group} group  |  {grid_size}×{grid_size} token grid  ({grid_size**2} tokens)")
    print(f"{'='*60}")

    helper = TokenPermutationHelper(n_group=n_group, grid_size=grid_size)
    results = []

    results.append(test_identity(helper))
    results.append(test_permutation_is_bijection(helper))
    results.append(test_inverse_perm(helper))
    results.append(test_spatial_alignment(helper, grid_size=grid_size))

    if n_group == 4:
        results.append(test_c4_90deg(helper))
        results.append(test_c4_composition(helper))
        results.append(test_c4_full_orbit(helper))
    elif n_group == 8:
        results.append(test_c8_90deg_multiples(helper))

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n  {n_pass}/{len(results)} passed", end="")
    if n_fail:
        print(f"  \033[31m({n_fail} FAILED)\033[0m")
    else:
        print("  \033[32m(all OK)\033[0m")
    return n_fail == 0


if __name__ == "__main__":
    all_ok = True
    all_ok &= run_suite(n_group=4, grid_size=16)
    all_ok &= run_suite(n_group=8, grid_size=16)
    # Smaller grid for quick sanity check
    all_ok &= run_suite(n_group=4, grid_size=8)

    print()
    if all_ok:
        print("\033[32mAll suites passed.\033[0m")
    else:
        print("\033[31mSome suites FAILED.\033[0m")
        raise SystemExit(1)
