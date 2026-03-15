"""
Visualize OrbitQueryEquiPool:
  1. Orbit query structure — how n_base generators produce K queries via cyclic shift
  2. Attention maps — which 16×16 patches each query focuses on
  3. Equivariance — rotating input permutes output slots
"""

import torch
import torch.nn as nn
import escnn.nn as enn
import escnn
from escnn.group import CyclicGroup
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Setup ────────────────────────────────────────────────────────────────────
G       = 4          # C_4 for easy visualization (90° rotations)
n_base  = 2          # 2 concept generators → K = n_base * G = 8 queries
K       = n_base * G
heads   = 2
dim_head = 4
D       = 16         # in_type.size = n_blocks * G = 4 * 4

gspace  = escnn.gspaces.no_base_space(CyclicGroup(G))
in_type = enn.FieldType(gspace, [gspace.regular_repr] * (D // G))  # 4 regular blocks


# ── OrbitQueryEquiPool (self-contained) ──────────────────────────────────────
class OrbitQueryEquiPool(nn.Module):
    def __init__(self, in_type, n_base, heads, dim_head):
        super().__init__()
        gspace = in_type.gspace
        G = gspace.fibergroup.order()
        self.G, self.H, self.Dh = G, heads, dim_head
        self.K = n_base * G
        self.n_base = n_base
        self.in_dim = in_type.size
        self.in_type = in_type
        self.scale = dim_head ** -0.5
        scalar_dim = heads * dim_head
        assert scalar_dim % G == 0
        n_regular = scalar_dim // G
        self.qkv_type = enn.FieldType(gspace, [gspace.regular_repr] * n_regular)
        self.q_proj = enn.Linear(in_type, self.qkv_type, bias=True)
        self.k_proj = enn.Linear(in_type, self.qkv_type, bias=True)
        n_blocks = in_type.size // G
        # orbit-symmetric init
        base = nn.init.trunc_normal_(torch.empty(n_base, n_blocks), std=0.5)
        self.query_base = nn.Parameter(
            base.unsqueeze(-1).expand(-1, -1, G).reshape(n_base, in_type.size).contiguous()
        )

    def _orbit_queries(self):
        n_blocks = self.in_dim // self.G
        q = self.query_base.reshape(self.n_base, n_blocks, self.G)
        orbits = torch.stack([torch.roll(q, g, dims=-1) for g in range(self.G)], dim=1)
        return orbits.reshape(self.K, self.in_dim)

    def forward(self, x):
        B, N, D = x.shape
        query_tokens = self._orbit_queries()
        x_geo = enn.GeometricTensor(x.reshape(B * N, D), self.in_type)
        q_geo = enn.GeometricTensor(
            query_tokens.unsqueeze(0).expand(B, -1, -1).reshape(B * self.K, -1),
            self.in_type,
        )
        Q     = self.q_proj(q_geo).tensor.reshape(B, self.K, self.H, self.Dh)
        K_reg = self.k_proj(x_geo).tensor.reshape(B, N, self.H, self.Dh)
        attn  = torch.einsum("bkhd,bnhd->bkhn", Q, K_reg) * self.scale
        attn  = attn.softmax(dim=-1).mean(dim=2)   # (B, K, N)
        out   = torch.einsum("bkn,bnd->bkd", attn, x)
        return out, attn  # also return attn maps for visualization


pool = OrbitQueryEquiPool(in_type, n_base=n_base, heads=heads, dim_head=dim_head)
pool.eval()

# ── Helper: rotate a 16-element feature map 90° (C_4 rotation) ──────────────
def rotate_spatial_90(x_2d):
    """Rotate 4×4 spatial grid 90° CCW (permute tokens + roll features)."""
    # x_2d: (1, 16, D) → treat as (1, 4, 4, D)
    B, N, D = x_2d.shape
    h = w = int(N ** 0.5)
    x_img = x_2d.reshape(B, h, w, D)
    # 90° CCW: (r,c) → (w-1-c, r)
    x_rot = torch.rot90(x_img, k=1, dims=[1, 2])
    x_rot = x_rot.reshape(B, N, D)
    # Also roll the D-dim features by 1 (one cyclic shift = one G-step)
    n_blocks = D // G
    x_rot = torch.roll(x_rot.reshape(B, N, n_blocks, G), shifts=1, dims=-1).reshape(B, N, D)
    return x_rot


# ── Create a synthetic scene: a bright "object" in one quadrant ──────────────
torch.manual_seed(42)
h = w = 4          # 4×4 patch grid (simplified from 16×16)
N = h * w          # 16 patches

x_base = torch.randn(1, N, D) * 0.1   # background noise

# Place a strong activation in top-left quadrant (patches 0,1,4,5)
# with features pointing in "direction 0" (first G-element dominant)
object_feat = torch.zeros(D)
object_feat[0::G] = 2.0   # first element of each regular block is strong (direction 0)
for idx in [0, 1, 4, 5]:
    x_base[0, idx] = x_base[0, idx] + object_feat

# Rotate input by 90°, 180°, 270°
x_rot1 = rotate_spatial_90(x_base)          # 90°  → object in top-right
x_rot2 = rotate_spatial_90(x_rot1)          # 180° → object in bottom-right
x_rot3 = rotate_spatial_90(x_rot2)          # 270° → object in bottom-left

inputs = [x_base, x_rot1, x_rot2, x_rot3]
labels = ["0°", "90°", "180°", "270°"]

with torch.no_grad():
    results = [pool(x) for x in inputs]


# ── Figure 1: Orbit query structure ──────────────────────────────────────────
fig1, axes = plt.subplots(2, G, figsize=(12, 5))
fig1.suptitle(
    f"Orbit Query Structure  (n_base={n_base}, G={G}, K={K})\n"
    "Each row = one base concept. Columns = G cyclic shifts (rotations).",
    fontsize=12, fontweight='bold'
)

queries = pool._orbit_queries().detach()  # (K, D)
n_blocks = D // G

for b in range(n_base):
    for g in range(G):
        ax = axes[b, g]
        q = queries[b * G + g].reshape(n_blocks, G)  # (n_blocks, G)
        im = ax.imshow(q.numpy(), cmap='RdBu', vmin=-1.5, vmax=1.5, aspect='auto')
        ax.set_title(f"base {b}, shift {g}\n(query {b*G+g})", fontsize=8)
        ax.set_xlabel("G elements", fontsize=7)
        ax.set_ylabel("blocks", fontsize=7)
        ax.set_xticks(range(G))
        ax.set_yticks(range(n_blocks))
        # Highlight the relationship: each column is a cyclic shift of the previous
        if g > 0:
            ax.spines['top'].set_color('orange')
            ax.spines['right'].set_color('orange')
            ax.spines['bottom'].set_color('orange')
            ax.spines['left'].set_color('orange')
            for spine in ax.spines.values():
                spine.set_linewidth(2)

plt.colorbar(im, ax=axes, shrink=0.6, label='weight value')
plt.tight_layout()
plt.savefig('/tmp/orbit_queries_structure.png', dpi=120, bbox_inches='tight')
print("Saved: /tmp/orbit_queries_structure.png")


# ── Figure 2: Attention maps across 4 rotations ──────────────────────────────
fig2, axes = plt.subplots(K, 4, figsize=(10, 16))
fig2.suptitle(
    "Attention maps per query vs. input rotation\n"
    "Each row = one query slot.  Columns = 0°, 90°, 180°, 270° input rotation.\n"
    "Orbit equivariance: when input rotates, attention maps should ROTATE (not change).",
    fontsize=11, fontweight='bold'
)

for k in range(K):
    for r, (_, attn) in enumerate(results):
        ax = axes[k, r]
        attn_map = attn[0, k].detach().reshape(h, w).numpy()  # (4,4)
        im = ax.imshow(attn_map, cmap='hot', vmin=0, vmax=attn_map.max() + 1e-6)
        ax.set_title(f"Q{k} | {labels[r]}", fontsize=7)
        ax.axis('off')
        # Color-code by orbit membership
        color = ['#2196F3', '#4CAF50'][k // G]  # blue for base 0, green for base 1
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

# Add orbit labels on left
for b in range(n_base):
    mid_k = b * G + G // 2
    axes[mid_k, 0].set_ylabel(f"Orbit {b}", fontsize=9, fontweight='bold',
                               color=['#2196F3', '#4CAF50'][b], rotation=90, labelpad=15)

plt.tight_layout()
plt.savefig('/tmp/orbit_attn_maps.png', dpi=120, bbox_inches='tight')
print("Saved: /tmp/orbit_attn_maps.png")


# ── Figure 3: Equivariance check — output similarity ─────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(
    "Equivariance check: does rotating input permute output slots?\n"
    "Left: how similar are query attention maps across rotations (should be shifted copies)\n"
    "Right: orbit slot permutation pattern",
    fontsize=11, fontweight='bold'
)

# Left: cosine similarity matrix of attn maps between 0° and 90° rotations
attn_0   = results[0][1][0].detach()   # (K, N)  — 0° input
attn_90  = results[1][1][0].detach()   # (K, N)  — 90° input

attn_0_n  = nn.functional.normalize(attn_0,  dim=-1)
attn_90_n = nn.functional.normalize(attn_90, dim=-1)
sim = (attn_0_n @ attn_90_n.T).numpy()   # (K, K) cosine sim

im1 = axes[0].imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
axes[0].set_title("Cosine sim: attn(0°) vs attn(90°)\n"
                   "Expected: query k matches query (k+1)%G after rotation", fontsize=9)
axes[0].set_xlabel("Query index (90° input)")
axes[0].set_ylabel("Query index (0° input)")
axes[0].set_xticks(range(K))
axes[0].set_yticks(range(K))
plt.colorbar(im1, ax=axes[0])

# Annotate expected permutation
for b in range(n_base):
    for g in range(G):
        k = b * G + g
        k_expected = b * G + (g + 1) % G
        axes[0].add_patch(plt.Rectangle((k_expected - 0.5, k - 0.5), 1, 1,
                                         fill=False, edgecolor='blue', lw=2))

# Right: show output vectors for each slot across rotations (L2 norm of change)
output_norms = []
out_0 = results[0][0][0].detach()  # (K, D)
for r in range(4):
    out_r = results[r][0][0].detach()   # (K, D)
    output_norms.append(out_r.norm(dim=-1).numpy())  # (K,)

output_norms = np.array(output_norms)  # (4, K)
im2 = axes[1].imshow(output_norms, cmap='viridis', aspect='auto')
axes[1].set_title("Output slot L2 norm per rotation\n"
                   "Orbit equivariance: each row is a cyclic permutation of previous", fontsize=9)
axes[1].set_xlabel("Query slot (K)")
axes[1].set_ylabel("Input rotation")
axes[1].set_yticks(range(4))
axes[1].set_yticklabels(labels)
axes[1].set_xticks(range(K))

# Draw orbit boundaries
for b in range(1, n_base):
    axes[1].axvline(b * G - 0.5, color='white', lw=2, linestyle='--')

plt.colorbar(im2, ax=axes[1], label='L2 norm')
plt.tight_layout()
plt.savefig('/tmp/orbit_equivariance_check.png', dpi=120, bbox_inches='tight')
print("Saved: /tmp/orbit_equivariance_check.png")

plt.show()
print("\nDone. Three figures saved to /tmp/")
