# EquiLLM Integration into EagleBackboneFATokens

Reference paper: *Large Language-Geometry Model: When LLM meets Equivariance* (arXiv:2502.11149v2)

---

## Tổng quan mapping EquiLLM → GR00T

| EquiLLM component | GR00T analog |
|---|---|
| Equivariant Encoder → **H'** (invariant) | Invariant FA: `h_inv_raw` (plain average, no ρ(h⁻¹)) |
| Equivariant Encoder → **X̃'** (equivariant) | Equivariant FA: `h_prime` (with ρ(h⁻¹) ⊗ π(h⁻¹), regular repr) |
| Projector(H') → **H^proj** → LLM | `inv_fa_proj(h_inv_raw)` injected into frozen Eagle VLM |
| Geometric-aware Prompt **P** | Text instruction tokenized by Eagle tokenizer |
| **LLM** (frozen) → **H^llm** | Eagle-2.5 VLM (frozen) → `vl_hidden` |
| **Skip connection** X̃' + H' → Adapter | `h_prime` (equi skip) + `h_inv_proj` (inv skip) → `equi_adapter` |
| **Equivariant Adapter** | `EquiAdapter` (SA+CA replacing EGNN) |

Frame Averaging (FA) đóng vai trò encoder equivariant: cung cấp đồng thời **H'** (invariant skip) và **X̃'** (equivariant skip) — tương ứng chính xác với output của Equivariant Encoder trong EquiLLM.

---

## Những thay đổi đã thực hiện

### 1. `EquiAdapter` — rewrite hoàn toàn

**Vấn đề cũ (sai so với EquiLLM):**
```python
# CŨ: concat language tokens vào SA sequence
h = cat([h_prime_flat, h_lang_reg], dim=1)   # h_lang trong equivariant SA ← SAI
h = SA(h)
h = CA(h, encoder_hidden_states=vl_features) # CA dùng full VLM output ← không đúng H^r
```

Language tokens tham gia equivariant SA cùng vision tokens — vi phạm thiết kế EquiLLM,
nơi H^llm chỉ đóng vai trò **conditioning** (qua H^r), không tham gia xử lý equivariant.

**Fix (đúng với EquiLLM Section 3.2, Eq. 6):**
```python
# MỚI: additive combination trước, SA trên h_prime only
h_r = lang_proj(vl_equi) + h_inv_flat   # ← EquiLLM: H^r = proj(H^llm) + H'
h   = SA(h_prime)                        # equivariant SA trên vision tokens only
h   = CA(h, encoder_hidden_states=h_r)  # CA ← invariant context h_r
h_out = cat([h, vl_text], dim=1)        # language tokens appended, không qua SA
```

**Thay đổi trong `__init__`:**
- Giữ: `self.lang_embed = nn.Linear(d_llm, blocks)` — nhưng mục đích thay đổi (xem issue DiT bên dưới)
- Thêm: `self.lang_proj = nn.Linear(d_llm, d_eq)` — re-project H^llm về H' space (EquiLLM Eq.6)

Hai layers này phục vụ **hai mục đích hoàn toàn khác nhau**:
- `lang_proj` (d_llm → d_eq): tính h_r cho EquiLLM conditioning
- `lang_embed` (d_llm → blocks): format language tokens thành trivial-in-regular cho DiT (xem bên dưới)

**Thay đổi signature `forward`:**
```python
# CŨ:
def forward(self, h_prime, h_lang, vl_hidden) -> [B, n_equi*T + T_lang, D]

# MỚI:
def forward(self, h_prime, h_inv_flat, vl_equi, vl_text) -> [B, n_equi*T + T_lang, D]
```

---

### 2. `EagleBackboneFATokens.__init__` — thêm `inv_adapter_proj`

```python
# THÊM MỚI (Phase 3 H' skip connection):
self.inv_adapter_proj = nn.Linear(d_eagle, self.project_to_dim)
```

**Lý do:** Trong EquiLLM, **H'** (invariant features từ encoder) bypass LLM và đi thẳng vào
Equivariant Adapter qua skip connection. `inv_adapter_proj` project `h_inv_raw` (invariant FA,
dimension `d_eagle`) sang `project_to_dim` để dùng trong adapter — đây chính là H' skip connection.

`inv_fa_proj` (đã có sẵn) vẫn giữ nguyên cho Phase 2 (inject vào LLM), hai projection này
phục vụ hai mục đích khác nhau.

---

### 3. `_NEW_LAYER_PREFIXES` — thêm `inv_adapter_proj`

```python
# CŨ:
_NEW_LAYER_PREFIXES = ("vision_proj.", "eagle_linear.", "inv_fa_proj.", "equi_adapter.")

# MỚI:
_NEW_LAYER_PREFIXES = ("vision_proj.", "eagle_linear.", "inv_fa_proj.", "inv_adapter_proj.", "equi_adapter.")
```

---

### 4. `forward_eagle` — Phase 3 call site

```python
# CŨ:
text_pos = text_mask[0]
h_lang   = vl_features[:, text_pos, :]
h_prime_llm = self.equi_adapter(h_prime, h_lang, vl_features)

# MỚI:
text_pos = text_mask[0]
vl_text  = vl_features[:, text_pos, :]                            # language tokens (append to output)

img_pos_indices  = (~text_pos).nonzero(as_tuple=True)[0]
equi_pos_indices = img_pos_indices[:n_equi * num_vision_tokens]
vl_equi  = vl_features[:, equi_pos_indices, :]                    # VLM output tại equi positions = H^llm

inv_adap_dtype = self.inv_adapter_proj.weight.dtype
h_inv_proj = self.inv_adapter_proj(
    h_inv_raw.reshape(B, n_equi * num_vision_tokens, vision_dim).to(inv_adap_dtype)
)                                                                  # H' skip: invariant FA → project_to_dim

h_prime_llm = self.equi_adapter(h_prime, h_inv_proj, vl_equi, vl_text)
```

---

## Tại sao FA giữ nguyên

Frame Averaging là **lựa chọn đúng** để thay thế Equivariant Encoder của EquiLLM:

- FA cung cấp đồng thời **cả hai** output cần thiết:
  - `h_prime` (equi FA): tương đương X̃' — equivariant, dùng làm skip và xử lý trong adapter
  - `h_inv_raw` (inv FA): tương đương H' — invariant, dùng cho LLM injection (Phase 2) VÀ skip đến adapter (Phase 3)
- Invariant FA (`plain average`) thậm chí **chính xác hơn** EquiLLM khi feed vào LLM (EquiLLM feed H' equivariant vào projector → LLM, lý thuyết không hoàn toàn đúng vì LLM là invariant function)

---

## Output layout (không đổi)

```
[B, n_equi*T_vis + T_lang, project_to_dim]
  ├── [:n_equi*T_vis]  equivariant vision tokens (processed by EquiAdapter SA+CA)
  └── [n_equi*T_vis:]  invariant language tokens (direct VLM output, appended)
```

Downstream DiT dùng layout này cho cross-attention context và SA prefix.

---

## DiT cross-attention compatibility — trivial-in-regular

**Vấn đề:** DiT dùng `cross_attention_type = FieldType(gs, [regular_repr] * blocks)` cho cross-attention context. Trong `both_regular` mode, V_proj là `enn.Linear(regular → regular)` (circulant equivariant linear). Nó kỳ vọng mỗi context token có format **group-major** `[v_0, v_1, ..., v_{G-1}]` với `v_i = ρ(g_i)·v_0`.

**Language tokens `vl_text` KHÔNG có format này** — chúng là raw LLM output vectors, không phân bố theo group-major layout. Nếu V_proj nhận vl_text như regular repr:
```
V_proj(vl_text) = ρ(g) · V_proj(vl_text)  ← expect equivariant, nhưng vl_text invariant
                                              → V thực sự thay đổi theo g → SAI
```

**Fix:** Format `vl_text` thành **trivial-in-regular** trước khi append vào output:
```python
h_lang_inv = lang_embed(vl_text)           # [B, T_lang, blocks]  — invariant subspace
h_lang_reg = repeat(h_lang_inv, G times)   # [B, T_lang, G*blocks] = [v,v,...,v]
# Cyclic-shift([v,v,...,v]) = [v,v,...,v] → trivial-in-regular ✓
# V_proj([v,v,...,v]) = equivariant linear của invariant input → valid ✓
```

`lang_embed` (d_llm → blocks) được GIỮ LẠI nhưng chỉ dùng cho bước format output này, KHÔNG đưa language tokens vào SA sequence.

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
