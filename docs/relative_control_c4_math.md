# Why Relative Control Math is Correct for C4

## Starting Point: SO(2) Decomposition

The paper derives that for relative action $\mathbf{a}_t \in \mathbb{R}^{16}$:

$$g\mathbf{a}_t = P^{-1}\left[(\rho_0^6 \oplus \rho_1^4 \oplus \rho_2)(g)\right] P\,\mathbf{a}_t$$

Total dims: $6 \times 1 + 4 \times 2 + 1 \times 2 = 16$ ✓

---

## Restricting Each Rep from SO(2) to C4

### ρ₀ (trivial) → trivial under C4

Unaffected. $6 \times \text{irrep}(0)_{C4}$, still 6 dims. ✓

---

### ρ₁ (standard 2D) → irrep(1) of C4

Generator of C4 is $g_{90°}$:

$$\rho_1(g_{90°}) = \begin{bmatrix}0 & -1 \\ 1 & 0\end{bmatrix}$$

This is exactly **irrep(1) of C4** — a 2D rotation. No change.

$4 \times \text{irrep}(1)_{C4}$, still 8 dims. ✓

---

### ρ₂ (double-frequency 2D) → 2 × irrep(2) of C4 ← critical

$$\rho_2(g_{90°}) = \begin{bmatrix}\cos 180° & -\sin 180° \\ \sin 180° & \cos 180°\end{bmatrix} = -I_2$$

All 4 C4 elements:

| Rotation | ρ₂ matrix |
|----------|-----------|
| 0°       | $+I_2$    |
| 90°      | $-I_2$    |
| 180°     | $+I_2$    |
| 270°     | $-I_2$    |

Both columns **never mix** — they independently scale by ±1. So:

$$\rho_2\big|_{C4} = \text{irrep}(2)_{C4} \oplus \text{irrep}(2)_{C4}$$

Confirmed by the actual features $[\cos(2\theta), \sin(2\theta)]$:

$$\cos(2\theta + 180°) = -\cos(2\theta), \quad \sin(2\theta + 180°) = -\sin(2\theta)$$

Both flip independently = sign rep, twice. ✓

$2 \times \text{irrep}(2)_{C4}$, still **2 dims** ✓

---

## Final C4 Decomposition

$$g\mathbf{a}_t = P^{-1}\left[(\text{irrep}(0)^6 \oplus \text{irrep}(1)^4 \oplus \text{irrep}(2)^2)(g)\right] P\,\mathbf{a}_t$$

| Rep       | C4 size | Copies  | Total   |
|-----------|---------|---------|---------|
| irrep(0)  | 1       | 6       | 6       |
| irrep(1)  | 2       | 4       | 8       |
| irrep(2)  | 1       | **2**   | 2       |
|           |         | **Total** | **16** ✓ |

## Key Insight

SO(2)'s ρ₂ restricted to C4 is **reducible** into 2 sign reps, because:

- 90° rotation applied to the double-frequency component means $2 \times 90° = 180°$
- 180° sends any $[\cos\theta, \sin\theta]$ pair to $[-\cos\theta, -\sin\theta]$ — both components flip with **no mixing**

The code `rho2_copies = 2 // irrep(2).size` is exactly this restriction:

- **C8**: `irrep(2).size = 2` → `rho2_copies = 1` → 1 × 2D irrep = 2 dims ✓
- **C4**: `irrep(2).size = 1` → `rho2_copies = 2` → 2 × 1D sign reps = 2 dims ✓
