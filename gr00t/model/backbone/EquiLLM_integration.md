# EquiLLM Integration into EagleBackboneFATokens

Reference paper: *Large Language-Geometry Model: When LLM meets Equivariance* (arXiv:2502.11149v2)

---

## Tổng quan mapping EquiLLM → GR00T

| EquiLLM component | GR00T analog |
|---|---|
| Equivariant Encoder → **H'** (invariant) | Invariant FA: `fa_inv_raw` (plain average, no ρ(h⁻¹)) |
| Equivariant Encoder → **X̃'** (equivariant) | Equivariant FA: `fa_equi_raw` → `h_equi` (with ρ(h⁻¹) ⊗ π(h⁻¹), regular repr) |
| Projector(H') → **H^proj** → LLM | `inv_fa_proj(fa_inv_raw)` injected into frozen Eagle VLM |
| Geometric-aware Prompt **P** | Text instruction tokenized by Eagle tokenizer |
| **LLM** (frozen) → **H^llm** | Eagle-2.5 VLM (frozen) → `vlm_hidden` |
| **Skip connection** X̃' + H' → Adapter | `h_equi` (equi skip, 4D) + `h_inv_skip` (inv skip) → `equi_adapter` |
| **Equivariant Adapter** | `EquiAdapter` (SA+CA replacing EGNN) |
| *(no analog in EquiLLM)* | Non-equi cameras: `noequi_vlm` — pass through VLM only, appended as trivial-in-regular |

Frame Averaging (FA) đóng vai trò encoder equivariant: cung cấp đồng thời **H'** (invariant skip) và **X̃'** (equivariant skip) — tương ứng chính xác với output của Equivariant Encoder trong EquiLLM.

---

## Kiến trúc tổng thể (3 phases)

```
Phase 1 — FA on equi cameras:
  rotate_image_indices → [B*N*n_equi, T, D]
    ├── fa_equi_raw  [B, n_equi, T, D]   ← X̃' analog (equivariant, ρ(h⁻¹)⊗π(h⁻¹))
    └── fa_inv_raw   [B, n_equi, T, D]   ← H' analog (invariant, plain average)

  noequi cameras (non_equi_image_indices) → passed raw to VLM

Phase 2 — VLM pass:
  h_inv_for_vlm = inv_fa_proj(fa_inv_raw)     [B, n_equi*T, d_eagle]
  VLM(h_inv_for_vlm + noequi_pixels + text) → vlm_hidden
  vlm_features  = eagle_linear(vlm_hidden)    [B, T_total, project_to_dim]
    ├── vlm_text  [B, T_lang, D]              ← text positions
    ├── vlm_img   [B, n_equi*T, D]            ← equi image positions (H^llm analog)
    └── noequi_vlm [B, n_noequi*T, D]         ← non-equi image positions

Phase 3 — EquiAdapter:
  h_equi     = vision_proj(fa_equi_raw)       [B, n_equi, T, project_to_dim]
  h_inv_skip = inv_adapter_proj(fa_inv_raw)   [B, n_equi*T, project_to_dim]
  h_cond = lang_proj(vlm_img) + h_inv_skip    ← H^r = proj(H^llm) + H'  (Eq.6)
  SA+CA on h_equi, context = h_cond
  noequi/text → trivial-in-regular, appended
  Output: [B, n_equi*T + n_noequi*T + T_lang, D]
```

---

## Những thay đổi đã thực hiện

### 1. Non-equi camera support

**Khái niệm mới:** `rotate_image_indices` chọn tập con các camera sẽ được FA xử lý. Các camera không có trong danh sách này (`non_equi_image_indices`) được gọi là **non-equi cameras** — chúng đi qua VLM bình thường (không rotate, không FA), output được inject như regular vision tokens.

```python
# __init__
if rotate_image_indices is None:
    self.rotate_image_indices = list(range(num_images_per_sample))
else:
    self.rotate_image_indices = rotate_image_indices

self.non_equi_image_indices = [
    i for i in range(num_images_per_sample)
    if i not in self.rotate_image_indices
]
```

`rotate_vl_batch` tách batch thành `equi_pixels` và `noequi_pixels`. Invariant FA tokens được inject vào VLM trước, non-equi tokens sau, theo thứ tự equi → noequi trong sequence.

---

### 2. `EquiAdapter` — signature và noequi_embed

**Signature mới:**
```python
# MỚI:
def forward(
    self,
    h_equi: Tensor,         # [B, n_equi, T_vis, D]   4D, equivariant FA skip
    h_inv_skip: Tensor,     # [B, n_equi*T_vis, D]    invariant FA skip (H' analog)
    vlm_img: Tensor,        # [B, n_equi*T_vis, D]    VLM output at equi positions (H^llm)
    vlm_text: Tensor,       # [B, T_lang, D]          VLM output at text positions
    noequi_vlm: Optional[Tensor] = None,  # [B, n_noequi*T, D]  non-equi cameras
) -> Tensor  # [B, n_equi*T + n_noequi*T + T_lang, D]
```

`h_equi` là **4D tensor** `[B, n_equi, T, D]` — được reshape thành `[B, n_equi*T, D]` bên trong forward trước khi qua SA+CA.

**Pipeline forward:**
```python
# EquiLLM Eq.6: H^r = proj(H^llm) + H'
h_cond = lang_proj(vlm_img) + h_inv_skip        # [B, n_equi*T, D] invariant

# Equivariant SA+CA on h_vis = h_equi.reshape(B, n_equi*T, D)
h_vis = h_equi.reshape(B, n_equi * T, D)
for sa_blk, ca_blk in zip(sa_blocks, ca_blocks):
    h_vis = sa_blk(h_vis)
    h_vis = ca_blk(h_vis, encoder_hidden_states=h_cond)

# noequi/text → trivial-in-regular: project to R^blocks → [v,v,...,v]×G
text_out   = to_trivial_reg(vlm_text,   lang_embed)
noequi_out = to_trivial_reg(noequi_vlm, noequi_embed)  # if noequi_vlm is not None

return cat([h_vis, noequi_out, text_out], dim=1)  # or [h_vis, text_out] if no noequi
```

**Layers trong `__init__`:**
- `lang_proj` (d_llm → d_eq): tính `h_cond` cho EquiLLM conditioning (H^r)
- `lang_embed` (d_eq → blocks): format language tokens thành trivial-in-regular cho output
- `noequi_embed` (d_eq → blocks): format non-equi camera tokens thành trivial-in-regular *(mới)*

`lang_proj` và `lang_embed/noequi_embed` phục vụ **hai mục đích hoàn toàn khác nhau**:
- `lang_proj` (d_llm → d_eq): conditioning context — SA+CA mượn thông tin từ H^llm
- `lang_embed/noequi_embed` (d_eq → blocks): format output tokens — invariant tokens phải ở trivial-in-regular subspace để tương thích với DiT cross-attention

---

### 3. `EagleBackboneFATokens.__init__` — inv_adapter_proj

```python
# Phase 3 H' skip connection:
self.inv_adapter_proj = nn.Linear(d_eagle, self.project_to_dim)
```

**Lý do:** Trong EquiLLM, **H'** (invariant features từ encoder) bypass LLM và đi thẳng vào Equivariant Adapter qua skip connection. `inv_adapter_proj` project `fa_inv_raw` (invariant FA, dimension `d_eagle`) sang `project_to_dim` cho adapter — đây chính là H' skip connection.

`inv_fa_proj` (Phase 2) và `inv_adapter_proj` (Phase 3) phục vụ hai mục đích khác nhau:
- `inv_fa_proj` (d_eagle → d_eagle): inject invariant tokens vào LLM
- `inv_adapter_proj` (d_eagle → project_to_dim): H' skip vào EquiAdapter

---

### 4. `_NEW_LAYER_PREFIXES`

```python
_NEW_LAYER_PREFIXES = (
    "vision_proj.",
    "eagle_linear.",
    "inv_fa_proj.",
    "inv_adapter_proj.",
    "equi_adapter.",
)
```

Tất cả layers này được train; `eagle_model` (VLM) luôn frozen.

---

### 5. `forward_eagle` — Phase 3 call site

```python
# text_pos: True at text (non-image) positions
text_pos = text_mask[0]                                              # [T_total] bool

vlm_text     = vlm_features[:, text_pos, :]                        # [B, T_lang, D]

img_pos_idx  = (~text_pos).nonzero(as_tuple=True)[0]               # all image positions
equi_img_idx = img_pos_idx[:n_equi * num_vision_tokens]            # equi camera positions
vlm_img      = vlm_features[:, equi_img_idx, :]                    # [B, n_equi*T, D]  H^llm

noequi_img_idx = img_pos_idx[n_equi * num_vision_tokens:]          # non-equi positions
noequi_vlm  = (vlm_features[:, noequi_img_idx, :]
               if noequi_img_idx.numel() > 0 else None)            # [B, n_noequi*T, D]

inv_adap_dtype = self.inv_adapter_proj.weight.dtype
h_inv_skip = self.inv_adapter_proj(
    fa_inv_raw.reshape(B, n_equi * num_vision_tokens, vision_dim).to(inv_adap_dtype)
)                                                                   # [B, n_equi*T, project_to_dim]

# h_equi: 4D tensor [B, n_equi, T, project_to_dim]
h_adapted = self.equi_adapter(h_equi, h_inv_skip, vlm_img, vlm_text, noequi_vlm=noequi_vlm)
```

---

## Tại sao FA giữ nguyên

Frame Averaging là **lựa chọn đúng** để thay thế Equivariant Encoder của EquiLLM:

- FA cung cấp đồng thời **cả hai** output cần thiết:
  - `fa_equi_raw` → `h_equi` (equi FA): tương đương X̃' — equivariant, dùng làm skip và xử lý trong adapter
  - `fa_inv_raw` (inv FA): tương đương H' — invariant, dùng cho LLM injection (Phase 2) VÀ skip đến adapter (Phase 3)
- Invariant FA (`plain average`) thậm chí **chính xác hơn** EquiLLM khi feed vào LLM (EquiLLM feed H' equivariant vào projector → LLM, lý thuyết không hoàn toàn đúng vì LLM là invariant function)

---

## Output layout

```
[B, n_equi*T_vis + n_noequi*T_vis + T_lang, project_to_dim]
  ├── [:n_equi*T_vis]                equivariant vision tokens (SA+CA processed, regular repr)
  ├── [n_equi*T_vis:-T_lang]         non-equi camera tokens (trivial-in-regular, invariant)
  └── [-T_lang:]                     language tokens (trivial-in-regular, invariant)
```

Nếu không có non-equi cameras (tất cả cameras đều equi), segment giữa có kích thước 0 và layout rút gọn thành `[n_equi*T_vis + T_lang]`.

Downstream DiT dùng layout này cho cross-attention context và SA prefix.

---

## DiT cross-attention compatibility — trivial-in-regular

**Vấn đề:** DiT dùng `cross_attention_type = FieldType(gs, [regular_repr] * blocks)` cho cross-attention context. Trong `both_regular` mode, V_proj là `enn.Linear(regular → regular)` (circulant equivariant linear). Nó kỳ vọng mỗi context token có format **group-major** `[v_0, v_1, ..., v_{G-1}]` với `v_i = ρ(g_i)·v_0`.

**Language/noequi tokens KHÔNG có format này** — chúng là raw LLM output vectors, không phân bố theo group-major layout. Nếu V_proj nhận chúng như regular repr:
```
V_proj(vl_text) = ρ(g) · V_proj(vl_text)  ← expect equivariant, nhưng vl_text invariant
                                              → V thực sự thay đổi theo g → SAI
```

**Fix:** Format `vlm_text` và `noequi_vlm` thành **trivial-in-regular** trước khi append vào output:
```python
def to_trivial_reg(x, embed):
    scalar = embed(x)                                    # [B, T, blocks]   invariant subspace
    return scalar.unsqueeze(-2).expand(-1,-1,G,-1).reshape(B,-1,D)  # [B, T, G*blocks]=[v,v,...,v]
# Cyclic-shift([v,v,...,v]) = [v,v,...,v] → trivial-in-regular ✓
# V_proj([v,v,...,v]) = equivariant linear của invariant input → valid ✓

text_out   = to_trivial_reg(vlm_text,   self.lang_embed)    # d_eq → blocks → trivial-reg
noequi_out = to_trivial_reg(noequi_vlm, self.noequi_embed)  # d_eq → blocks → trivial-reg
```

`lang_embed` và `noequi_embed` (d_eq → blocks) chỉ dùng cho bước format output này, KHÔNG đưa các tokens này vào SA sequence.

---

## Tại sao SA+CA thay vì EGNN

EquiLLM dùng EGNN vì X̃' là **tọa độ 3D thực** (atoms) có graph edges và khoảng cách Euclidean.
Trong bài toán vision-robotics:
- Không có graph structure cố định
- "Tọa độ" đã được encode vào token positions qua `token_perm_indices` trong regular repr
- SA (global attention) tự nhiên hơn local message passing khi không có graph

`EquivariantAttention` tự động chọn mode đúng:
- SA: `in_type == cross_type == regular` → **InvQK+EquiV mode** (Q,K invariant; V equivariant) ✓
- CA: `cross_type == trivial` → **scalar mode** (Q group-mean projected, CA với invariant context) ✓

Constraint duy nhất cần đảm bảo: `scalar_dim = num_heads × head_dim` phải chia hết cho `n_group`.
