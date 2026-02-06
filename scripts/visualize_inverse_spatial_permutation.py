#!/usr/bin/env python3
"""
Visualize Inverse Spatial Permutation in Frame Averaging.

This script demonstrates how the inverse spatial permutation π(h⁻¹) works
to revert tokens back to their original positions after image rotation.

Key Concept:
When an image is rotated by h, the token at position p moves to π(h)·p.
To align features from rotated images with the original coordinate system,
we apply the INVERSE permutation π(h⁻¹) to move tokens back.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


def create_position_grid(grid_size=4):
    """Create a grid showing position indices."""
    grid = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    return grid


def rotate_90_ccw_position(i, j, N):
    """
    Get the NEW position after 90° CCW rotation.
    Position (i,j) moves to (N-1-j, i) after 90° CCW rotation.
    """
    return N - 1 - j, i


def rotate_90_ccw_source(i, j, N):
    """
    Get the SOURCE position for 90° CCW rotation.
    After rotation, position (i,j) contains content from (j, N-1-i).
    """
    return j, N - 1 - i


def compute_forward_permutation(num_rotations, N):
    """
    Compute forward permutation π(h): where each token GOES after rotation.
    
    π(h)[src] = dst means token at src goes to dst after rotation by h.
    """
    perm = np.arange(N * N)
    
    for _ in range(num_rotations):
        new_perm = np.zeros(N * N, dtype=int)
        for src in range(N * N):
            i, j = src // N, src % N
            new_i, new_j = rotate_90_ccw_position(i, j, N)
            dst = new_i * N + new_j
            new_perm[src] = perm[dst]  # Follow the chain
        perm = new_perm
    
    # Actually compute directly
    perm = np.arange(N * N)
    for src_idx in range(N * N):
        i, j = src_idx // N, src_idx % N
        for _ in range(num_rotations):
            i, j = rotate_90_ccw_position(i, j, N)
        perm[src_idx] = i * N + j
    
    return perm


def compute_source_permutation(num_rotations, N):
    """
    Compute source permutation: where each position's content COMES FROM.
    
    perm[dst] = src means position dst contains content from src after rotation.
    This is what we use to "gather" features.
    """
    perm = np.arange(N * N)
    
    for dst_idx in range(N * N):
        i, j = dst_idx // N, dst_idx % N
        for _ in range(num_rotations):
            i, j = rotate_90_ccw_source(i, j, N)
        perm[dst_idx] = i * N + j
    
    return perm


def visualize_rotation_and_permutation():
    """Main visualization function."""
    N = 4  # 4x4 grid for clarity
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create a colorful grid to track positions
    colors = plt.cm.tab20(np.linspace(0, 1, N * N))
    
    # ===== Row 1: Original Image and Tokens =====
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.set_title("Original Image\n(with labeled regions)", fontsize=12, fontweight='bold')
    
    # Draw colored grid
    for idx in range(N * N):
        i, j = idx // N, idx % N
        rect = mpatches.Rectangle((j, N-1-i), 1, 1, 
                                    facecolor=colors[idx], 
                                    edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(j + 0.5, N - 0.5 - i, str(idx), 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlim(0, N)
    ax1.set_ylim(0, N)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # ===== Row 1: Original Token Grid =====
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.set_title("Original Token Positions\nf(x) = features at position p", fontsize=12, fontweight='bold')
    
    for idx in range(N * N):
        i, j = idx // N, idx % N
        rect = mpatches.Rectangle((j, N-1-i), 1, 1, 
                                    facecolor=colors[idx], 
                                    edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(j + 0.5, N - 0.5 - i, f"T{idx}", 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlim(0, N)
    ax2.set_ylim(0, N)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # ===== Row 1: Rotated Image (90° CCW) =====
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.set_title("After 90° CCW Rotation (h=1)\nImage content moved", fontsize=12, fontweight='bold')
    
    # After 90° CCW, content from (j, N-1-i) appears at (i,j)
    for dst_idx in range(N * N):
        dst_i, dst_j = dst_idx // N, dst_idx % N
        # Source position in original image
        src_i, src_j = rotate_90_ccw_source(dst_i, dst_j, N)
        src_idx = src_i * N + src_j
        
        rect = mpatches.Rectangle((dst_j, N-1-dst_i), 1, 1, 
                                    facecolor=colors[src_idx],  # Color shows original content
                                    edgecolor='black', linewidth=2)
        ax3.add_patch(rect)
        ax3.text(dst_j + 0.5, N - 0.5 - dst_i, str(src_idx), 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_xlim(0, N)
    ax3.set_ylim(0, N)
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    # ===== Row 1: Rotated Token Grid =====
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.set_title("Rotated Tokens f(h·x)\nTokens at new positions", fontsize=12, fontweight='bold')
    
    source_perm = compute_source_permutation(1, N)  # 1 rotation = 90° CCW
    
    for dst_idx in range(N * N):
        dst_i, dst_j = dst_idx // N, dst_idx % N
        src_idx = source_perm[dst_idx]
        
        rect = mpatches.Rectangle((dst_j, N-1-dst_i), 1, 1, 
                                    facecolor=colors[src_idx],
                                    edgecolor='black', linewidth=2)
        ax4.add_patch(rect)
        ax4.text(dst_j + 0.5, N - 0.5 - dst_i, f"T{src_idx}", 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax4.set_xlim(0, N)
    ax4.set_ylim(0, N)
    ax4.set_aspect('equal')
    ax4.axis('off')
    
    # ===== Row 2: The Problem =====
    ax5 = fig.add_subplot(3, 4, 5)
    ax5.text(0.5, 0.5, 
             "❌ PROBLEM:\n\n"
             "Token T0 in original is at position 0\n"
             "Token T0 in rotated is at position 3\n\n"
             "If we directly average:\n"
             "position 0: (T0 + T12) / 2 = WRONG!\n\n"
             "We're averaging different content!",
             ha='center', va='center', fontsize=11,
             transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='#ffcccc', edgecolor='red', linewidth=2))
    ax5.axis('off')
    
    # ===== Row 2: The Solution =====
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.text(0.5, 0.5,
             "✅ SOLUTION:\n\n"
             "Apply Inverse Permutation π(h⁻¹)\n"
             "to revert tokens to original positions\n\n"
             "For h=1 (90° CCW), h⁻¹=3 (270° CCW)\n"
             "π(h⁻¹) undoes the spatial shift\n\n"
             "Then averaging makes sense!",
             ha='center', va='center', fontsize=11,
             transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='#ccffcc', edgecolor='green', linewidth=2))
    ax6.axis('off')
    
    # ===== Row 2: Inverse Permutation Applied =====
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.set_title("After π(h⁻¹) = π(3)\n(270° CCW permutation)", fontsize=12, fontweight='bold')
    
    # Inverse permutation: h=1, so h_inv=3 (270° CCW)
    inverse_perm = compute_source_permutation(3, N)  # 3 rotations = 270° CCW
    
    # Apply inverse permutation to the rotated features
    # rotated_features[dst] has content from original[source_perm[dst]]
    # After inverse: result[final_dst] = rotated_features[inverse_perm[final_dst]]
    #              = original[source_perm[inverse_perm[final_dst]]]
    
    for final_idx in range(N * N):
        final_i, final_j = final_idx // N, final_idx % N
        # After applying inverse permutation to rotated features
        intermediate_idx = inverse_perm[final_idx]
        original_content_idx = source_perm[intermediate_idx]
        
        rect = mpatches.Rectangle((final_j, N-1-final_i), 1, 1, 
                                    facecolor=colors[original_content_idx],
                                    edgecolor='black', linewidth=2)
        ax7.add_patch(rect)
        ax7.text(final_j + 0.5, N - 0.5 - final_i, f"T{original_content_idx}", 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax7.set_xlim(0, N)
    ax7.set_ylim(0, N)
    ax7.set_aspect('equal')
    ax7.axis('off')
    
    # ===== Row 2: Result =====
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.set_title("Tokens Aligned!\nπ(h⁻¹)·f(h·x)", fontsize=12, fontweight='bold')
    
    # Show that tokens are back to original positions
    for idx in range(N * N):
        i, j = idx // N, idx % N
        rect = mpatches.Rectangle((j, N-1-i), 1, 1, 
                                    facecolor=colors[idx],
                                    edgecolor='green', linewidth=3)
        ax7.add_patch(rect)
    
    # Actually recalculate to verify
    for final_idx in range(N * N):
        final_i, final_j = final_idx // N, final_idx % N
        
        rect = mpatches.Rectangle((final_j, N-1-final_i), 1, 1, 
                                    facecolor=colors[final_idx],
                                    edgecolor='green', linewidth=3)
        ax8.add_patch(rect)
        ax8.text(final_j + 0.5, N - 0.5 - final_i, f"T{final_idx}", 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax8.set_xlim(0, N)
    ax8.set_ylim(0, N)
    ax8.set_aspect('equal')
    ax8.axis('off')
    
    # ===== Row 3: Mathematical Explanation =====
    ax9 = fig.add_subplot(3, 4, (9, 10))
    explanation = """
    MATHEMATICAL EXPLANATION
    ════════════════════════
    
    Given: f(h·x) = features from image rotated by h
           Token at position p has content from original position π(h)⁻¹·p
    
    Goal: Align all rotated features to original coordinate system
    
    Solution: Apply π(h⁻¹) which is the inverse spatial permutation
              π(h⁻¹)·f(h·x) puts each token back at its original position
    
    For C4 (90° rotations):
    • h=0: 0°   → h⁻¹=0 (identity)
    • h=1: 90°  → h⁻¹=3 (270° CCW = 90° CW)
    • h=2: 180° → h⁻¹=2 (180°)
    • h=3: 270° → h⁻¹=1 (90° CCW)
    
    h⁻¹ = (N - h) mod N  where N = |G| = 4
    """
    ax9.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=11,
             transform=ax9.transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray'))
    ax9.axis('off')
    
    # ===== Row 3: Full FA Formula =====
    ax10 = fig.add_subplot(3, 4, (11, 12))
    formula = """
    FRAME AVERAGING FORMULA (COVARIANT)
    ════════════════════════════════════
    
    FA(x) = (1/|G|) Σ_h  ρ(h⁻¹) · f(h·x)
    
    Where ρ(h⁻¹) = π(h⁻¹) ⊗ ρ_reg(h⁻¹)
    
    ┌─────────────────┬─────────────────────────────────┐
    │   Component     │         Purpose                 │
    ├─────────────────┼─────────────────────────────────┤
    │ h·x             │ Rotate image by h               │
    │ f(h·x)          │ Extract features from rotated   │
    │ π(h⁻¹)          │ Revert token POSITIONS          │
    │ ρ_reg(h⁻¹)      │ Revert feature CHANNELS         │
    │ Σ / |G|         │ Average over all rotations      │
    └─────────────────┴─────────────────────────────────┘
    
    Result: FA(g·x) = ρ(g) · FA(x)  (Equivariant!)
    """
    ax10.text(0.5, 0.5, formula, ha='center', va='center', fontsize=10,
              transform=ax10.transAxes, family='monospace',
              bbox=dict(boxstyle='round', facecolor='#ffffcc', edgecolor='orange'))
    ax10.axis('off')
    
    plt.suptitle("Inverse Spatial Permutation π(h⁻¹) in Frame Averaging", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('/home/locht1/Documents/locht1/Isaac-GR00T/inverse_spatial_permutation_visual.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved visualization to: inverse_spatial_permutation_visual.png")


def visualize_step_by_step():
    """Step-by-step visualization of the permutation process."""
    N = 4
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    colors = plt.cm.tab20(np.linspace(0, 1, N * N))
    
    def draw_grid(ax, perm_or_colors, title, show_arrows=False, arrow_perm=None):
        ax.set_title(title, fontsize=11, fontweight='bold')
        for idx in range(N * N):
            i, j = idx // N, idx % N
            color_idx = perm_or_colors[idx] if isinstance(perm_or_colors, (list, np.ndarray)) else idx
            rect = mpatches.Rectangle((j, N-1-i), 1, 1, 
                                       facecolor=colors[color_idx],
                                       edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(j + 0.5, N - 0.5 - i, str(color_idx), 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Row 1: Show original and all rotations
    source_perms = [compute_source_permutation(h, N) for h in range(4)]
    
    # Original
    draw_grid(axes[0, 0], list(range(N*N)), "Original (h=0)\nf(0·x) = f(x)")
    
    # Rotations
    for h in range(1, 4):
        # What content ends up at each position after rotation
        content_at_pos = source_perms[h]
        draw_grid(axes[0, h], content_at_pos, f"Rotated h={h} ({h*90}° CCW)\nf(h·x)")
    
    # Problem explanation
    axes[0, 4].text(0.5, 0.5,
                    "After rotation,\nsame content is at\ndifferent positions!\n\n"
                    "❌ Can't directly average\nposition-by-position",
                    ha='center', va='center', fontsize=11,
                    transform=axes[0, 4].transAxes,
                    bbox=dict(boxstyle='round', facecolor='#ffcccc'))
    axes[0, 4].axis('off')
    
    # Row 2: After inverse permutation
    draw_grid(axes[1, 0], list(range(N*N)), "Original (reference)")
    
    for h in range(1, 4):
        h_inv = (4 - h) % 4
        # After rotation h, apply inverse permutation h_inv
        # result[p] = rotated[inverse_perm[p]]
        # rotated[q] has content from original[source_perm_h[q]]
        # So result[p] = original[source_perm_h[inverse_perm_h_inv[p]]]
        
        inv_perm = compute_source_permutation(h_inv, N)
        src_perm = source_perms[h]
        
        # The final content at each position
        final_content = [src_perm[inv_perm[p]] for p in range(N*N)]
        
        draw_grid(axes[1, h], final_content, 
                  f"π(h⁻¹)·f(h·x), h⁻¹={h_inv}\nAfter inverse perm")
    
    # Solution explanation
    axes[1, 4].text(0.5, 0.5,
                    "After π(h⁻¹),\nall features are\naligned to original\ncoordinates!\n\n"
                    "✅ NOW we can average\nposition-by-position",
                    ha='center', va='center', fontsize=11,
                    transform=axes[1, 4].transAxes,
                    bbox=dict(boxstyle='round', facecolor='#ccffcc'))
    axes[1, 4].axis('off')
    
    plt.suptitle("Step-by-Step: Inverse Spatial Permutation Aligns Features", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/locht1/Documents/locht1/Isaac-GR00T/inverse_perm_step_by_step.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved step-by-step visualization to: inverse_perm_step_by_step.png")


def print_ascii_visualization():
    """Print ASCII art visualization for terminal."""
    print("""
╔════════════════════════════════════════════════════════════════════════════════════╗
║           INVERSE SPATIAL PERMUTATION π(h⁻¹) IN FRAME AVERAGING                    ║
╚════════════════════════════════════════════════════════════════════════════════════╝

ORIGINAL IMAGE (4×4 tokens):          AFTER 90° CCW ROTATION (h=1):
┌────┬────┬────┬────┐                 ┌────┬────┬────┬────┐
│ T0 │ T1 │ T2 │ T3 │                 │ T3 │ T7 │T11 │T15 │
├────┼────┼────┼────┤                 ├────┼────┼────┼────┤
│ T4 │ T5 │ T6 │ T7 │    ──────►     │ T2 │ T6 │T10 │T14 │
├────┼────┼────┼────┤    rotate      ├────┼────┼────┼────┤
│ T8 │ T9 │T10 │T11 │                 │ T1 │ T5 │ T9 │T13 │
├────┼────┼────┼────┤                 ├────┼────┼────┼────┤
│T12 │T13 │T14 │T15 │                 │ T0 │ T4 │ T8 │T12 │
└────┴────┴────┴────┘                 └────┴────┴────┴────┘

        f(x)                                f(h·x)
   (original features)               (rotated features)


❌ PROBLEM: Direct averaging mixes different spatial content!
   Position 0: avg(T0, T3, T15, T12) = WRONG (different corners!)


✅ SOLUTION: Apply inverse permutation π(h⁻¹) first!

For h=1 (90° CCW), h⁻¹ = 3 (270° CCW or 90° CW)

AFTER INVERSE PERMUTATION π(3):
┌────┬────┬────┬────┐
│ T0 │ T1 │ T2 │ T3 │    ← Same as original!
├────┼────┼────┼────┤
│ T4 │ T5 │ T6 │ T7 │    Features are now aligned
├────┼────┼────┼────┤
│ T8 │ T9 │T10 │T11 │    to original coordinate system
├────┼────┼────┼────┤
│T12 │T13 │T14 │T15 │
└────┴────┴────┴────┘

     π(h⁻¹) · f(h·x)
  (spatially aligned features)


NOW we can average position-by-position:
   Position 0: avg(T0_orig, T0_aligned_h1, T0_aligned_h2, T0_aligned_h3) = ✓ CORRECT!


═══════════════════════════════════════════════════════════════════════════════════════
                          MATHEMATICAL FORMULA
═══════════════════════════════════════════════════════════════════════════════════════

Frame Averaging for COVARIANT equivariance:

                    1
    FA(x)  =  ──────  Σ   ρ(h⁻¹) · f(h·x)
                |G|   h∈G

Where:
    ρ(h⁻¹) = π(h⁻¹) ⊗ ρ_reg(h⁻¹)
           ╰──────╯   ╰─────────╯
           spatial    feature
           permute    permute

═══════════════════════════════════════════════════════════════════════════════════════
                    WHY ρ(h⁻¹) AND NOT ρ(h)?
═══════════════════════════════════════════════════════════════════════════════════════

Using ρ(h):     FA(g·x) = ρ(g⁻¹) · FA(x)    ← Contravariant (wrong direction!)
Using ρ(h⁻¹):   FA(g·x) = ρ(g)   · FA(x)    ← Covariant (correct!)

Proof sketch for ρ(h⁻¹):
    FA(g·x) = (1/|G|) Σ_h ρ(h⁻¹) · f(h·g·x)
            = (1/|G|) Σ_k ρ((g⁻¹k)⁻¹) · f(k·x)    [substitute k = hg]
            = (1/|G|) Σ_k ρ(k⁻¹g) · f(k·x)
            = (1/|G|) Σ_k ρ(g) · ρ(k⁻¹) · f(k·x)  [representation property]
            = ρ(g) · (1/|G|) Σ_k ρ(k⁻¹) · f(k·x)
            = ρ(g) · FA(x)  ✓

═══════════════════════════════════════════════════════════════════════════════════════
                          CODE IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════════════════

# In forward_eagle():

for h in range(self.n_group):
    feat_h = vision_features_grouped[:, h, :, :]  # f(h·x)
    
    if h == 0:
        feat_transformed = feat_h  # Identity
    else:
        # Apply ρ(h⁻¹) = π(h⁻¹) ⊗ ρ_reg(h⁻¹)
        h_inv = (self.n_group - h) % self.n_group
        
        # Step 1: Spatial permutation π(h⁻¹)
        spatial_perm = self.token_perm_indices[h_inv]
        feat_permuted = feat_h[:, spatial_perm, :]  # ← INVERSE SPATIAL PERMUTATION
        
        # Step 2: Feature permutation ρ_reg(h⁻¹)
        feat_shifted = torch.roll(feat_blocks, shifts=h_inv, dims=2)
        
        feat_transformed = feat_shifted.reshape(...)
    
    transformed_features.append(feat_transformed)

# Average over all rotations
avg_vision_features = torch.mean(torch.stack(transformed_features, dim=1), dim=1)

═══════════════════════════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    print_ascii_visualization()
    
    try:
        import matplotlib
        print("\nGenerating graphical visualizations...")
        visualize_rotation_and_permutation()
        visualize_step_by_step()
    except ImportError:
        print("\nMatplotlib not available. ASCII visualization shown above.")
    except Exception as e:
        print(f"\nCould not generate graphical visualization: {e}")
        print("ASCII visualization shown above.")
