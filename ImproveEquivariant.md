# FA Hybrid Dual-Stream Equivariant Architecture

## Motivation

Current `FlowmatchingActionHead` builds equivariance via **native escnn layers**
(`EquiCategorySpecificMLP`, `MultiEmbodimentActionEncoder`) trained from scratch —
forfeiting all pretrained representations from the GR00T baseline.

The Eagle backbone shows a better path: **FA wraps a frozen pretrained module** to produce
equivariant and invariant features without touching its weights. This plan applies that
idea to state/action encoders, replacing the full EDiT with a **lightweight equivariant
residual adapter** (~2–4 blocks) that corrects the frozen DiT's geometry-blind output.

Baseline pretrained GR00T: branch `gr00t_baseline_locht1`

---

## Architecture Overview

```
 state (raw)  ──► getJointGeometricTensor ──► geo_state [(B*T_s), D_in]
                                                        │
                         ┌─── FAEncoder ────────────────┘
                         │  for g in G:
                         │    h_g = pretrained_state_encoder(geo_state.transform(g))
                         │  equi_state = FA_equi([h_0..h_{N-1}])  [(B*T_s), D]
                         │  inv_state  = mean([h_0..h_{N-1}])     [(B*T_s), D]
                         └──────────────────────────────────────────────────────

 noisy_action ──► getActionGT ──► GeometricTensor(action_type) [(B*T_a), D_in]
                                                        │
                         ┌─── FAEncoder ────────────────┘
                         │  for g in G:
                         │    h_g = pretrained_action_encoder(geo_action.transform(g), τ)
                         │  equi_action = FA_equi(...)  [(B*T_a), D]
                         │  inv_action  = mean(...)     [(B*T_a), D]
                         └──────────────────────────────────────────────────────

 ─── VLM SPLIT (zero extra params — EquiAdapter already separates these) ────────
 EquiAdapter.forward now returns a tuple instead of one concatenated tensor:
   vl_equi [B, n_equi*T, D]              regular repr   (equivariant vision)
   vl_inv  [B, n_noequi*T + T_lang, D/N] plain scalar   (invariant vision + language)
 No lifting noequi/text to trivial-in-regular → correct types, zero added params.

 ─── INVARIANT BRANCH (frozen) ────────────────────────────────────────────────
 [inv_state | inv_action]  →  sa_embs_inv [B, T_sa, D]
                               │
                          pretrained DiT  ← gr00t_baseline_locht1 weights, frozen
                          cross-attn ctx: vl_inv [B, T_rest, D/N]   (truly invariant)
                               │
                               ▼  inv_output [B, T_sa, D]

 ─── EQUIVARIANT BRANCH (lightweight) ─────────────────────────────────────────
 [equi_state | equi_action]  →  equi_sa [B, T_sa, D]  (regular repr)
                               │
                          EquiResAdapter  (N=2 BasicTransformerBlocks)
                          ┌──────────────────────────────────────────────────────┐
                          │ for each block:                                        │
                          │   SA:  EquivariantAttention(equi_sa ← equi_sa)       │
                          │   CA1: EquivariantAttention(equi_sa ← vl_equi)       │
                          │        equi Q + equi K/V → both_regular=True path ✓  │
                          │   CA2: EquivariantAttention(equi_sa ← inv_output)    │
                          │        equi Q + trivial K/V → induction map ✓         │
                          │   FF:  EquivariantFeedForward                          │
                          └──────────────────────────────────────────────────────┘
                               │
                               ▼  equi_delta [B, T_sa, D]  (regular repr)

 ─── RESIDUAL FUSION ───────────────────────────────────────────────────────────
 inv_lifted = inv_output.view(B,T,D//N,1).expand(-1,-1,-1,N).reshape(B,T,D)
 output = inv_lifted + equi_delta          [B, T_sa, D]  (regular repr)
               │
          EquiCategorySpecificMLP  (action decoder, fresh init)
               │
               ▼  pred_velocity [B, T_sa, action_type.size]
```

---

## Key Decisions

### FA formula and field types

```
FA(x) = (1/|G|) · Σ_h  ρ_out(h⁻¹) · f(h·x)

h·x:        geo_input.transform(h_elem)         uses input field type (getJointFieldType)
ρ_out(h⁻¹): enn.GeometricTensor(f_h, fa_output_type).transform(h_inv_elem)
             fa_output_type = FieldType([regular_repr] * blocks)  ← NOT getJointFieldType
             D_out = input_embedding_dim = n_group * blocks
```

Critical: encoder output lives in **embedding space** (regular repr), NOT in state/action
space. Using `getJointFieldType` for the output would be a dimension mismatch and
geometrically wrong.

Invariant FA is the plain mean — no transformation needed:
```
FA_inv(x) = (1/|G|) · Σ_h f(h·x)  = mean(h_stack, dim=0)   [provably invariant]
```

### FA permutation matrices (from C8EquivariantTimmObsEncoder)

The `_apply_frame_averaging` logic is **identical** to `C8EquivariantTimmObsEncoder`
and `EagleBackboneFATokens`. The only difference is the rotation step:

| | Image encoders | FAEncoder (state/action) |
|--|--|--|
| Rotate input | `F.grid_sample` + affine matrix | `geo_tensor.transform(g_elem)` |
| FA logic | `_apply_frame_averaging` | same `_apply_frame_averaging` |
| Inv FA | `h_stack.mean(dim=0)` | same |

### VLM split — modify `EquiAdapter` to return a tuple (zero extra params)

Currently `EquiAdapter.forward` lifts noequi/text tokens to trivial-in-regular and
returns a single concatenated tensor `[B, n_equi*T + n_noequi*T + T_lang, D]`.

**Change**: return `(h_vis, inv_tokens)` instead:

```python
# EquiAdapter.forward — new return
h_vis      : [B, n_equi*T, D]              regular repr   (equivariant vision)
inv_tokens : [B, n_noequi*T + T_lang, D/N] plain scalar   (invariant)
             = mean over group slots of (noequi_vlm || vlm_text)
             via: x.reshape(B, T, D//N, N).mean(-1)  — NO projection needed
```

The `to_trivial_reg` helper (currently lifts scalar → regular) is replaced by a
`to_inv_scalar` helper that extracts the invariant component:

```python
def to_inv_scalar(x: torch.Tensor) -> torch.Tensor:
    """[B, T, D] regular/trivial-in-regular → [B, T, D/N] invariant scalar."""
    B, T, D = x.shape
    return x.reshape(B, T, D // self.n_group, self.n_group).mean(-1)
```

`EagleBackboneFATokens.forward_eagle` unpacks the tuple:
```python
h_vis, inv_tokens = self.equi_adapter(h_equi, vlm_img, vlm_text, noequi_vlm=noequi_vlm)
```

`EagleBackboneFATokens.forward` puts both in BatchFeature:
```python
return BatchFeature(data={
    "backbone_equi_vision_features": h_vis,        # [B, n_equi*T, D]  regular repr
    "backbone_inv_features":         inv_tokens,   # [B, T_rest, D/N]  invariant
    "backbone_attention_mask":       eagle_mask,
})
```

Action head `process_backbone_output` reads both keys directly — no mean-pool needed.

**Why this is better than VLMSplitter:**
- Zero extra parameters (just reshape + mean, no learned layers)
- No approximation: inv_tokens are truly invariant (plain scalar), not mean-pooled from regular repr
- Correct semantic separation from the start: backbone already computes both types

### EquiResAdapter — two CA contexts per block

Each block has SA + CA1 + CA2 + FF:

| Sub-layer | Input Q | Context K/V | Type | Equivariance path |
|-----------|---------|-------------|------|-------------------|
| SA | equi_sa | equi_sa | regular ↔ regular | `both_regular=True` → InvQK + EquiV ✓ |
| CA1 | equi_sa | `vl_equi` | regular ↔ regular | `both_regular=True` → InvQK + EquiV ✓ |
| CA2 | equi_sa | `inv_output` | regular ↔ trivial | `else` path → trivial QKV, `o_proj: trivial→regular` (induction) ✓ |
| FF | equi_sa | — | regular | equivariant FF ✓ |

CA1 gives direct geometry-aware visual conditioning.
CA2 injects pretrained DiT's invariant trajectory knowledge as a skip.

**Identity init**: zero-init `v_proj`, `o_proj`, `ff.fc2` of **both CA1 and CA2** →
`equi_delta = 0` at init → `output = inv_lifted` (pure pretrained baseline). ✓

Warm-up scale `adapter_scale` ramps 0 → 1 over `warmup_steps`.

### inv_dit cross-attention context

`vl_inv` is now directly available from `backbone_inv_features [B, T_rest, D/N]` —
no mean-pool approximation. `DiT.cross_attention_dim = D/N = backbone_embedding_dim // n_group`.

---

## Detailed Implementation

### Step 1 — `FAEncoder` class (new, in `flow_matching_action_head.py`)

```python
class FAEncoder(nn.Module):
    """
    Frame Averaging wrapper around a frozen pretrained (non-equivariant) encoder.

    Computes:
        equi = FA_equi(x) = (1/N) Σ_h  ρ_out(h⁻¹) · f(h·x)   [regular repr]
        inv  = FA_inv(x)  = (1/N) Σ_h  f(h·x)                  [invariant]

    Rotation of input:  geo_input.transform(g_elem)  — uses input field type
    Rotation of output: wrap plain tensor with fa_output_type, call .transform(h_inv)
    _apply_frame_averaging: identical to C8EquivariantTimmObsEncoder
    """

    def __init__(
        self,
        pretrained_encoder: nn.Module,
        in_type: enn.FieldType,           # getJointFieldType() or action_type
        n_group: int,
        output_dim: int,                  # input_embedding_dim = n_group * blocks
    ):
        super().__init__()
        self.pretrained_encoder = pretrained_encoder
        self.in_type = in_type
        self.n_group = n_group
        self.output_dim = output_dim

        assert output_dim % n_group == 0, "output_dim must be divisible by n_group"
        blocks = output_dim // n_group
        gspace = in_type.gspace
        # Output field type: regular repr in embedding space
        self.fa_output_type = enn.FieldType(gspace, [gspace.regular_repr] * blocks)

        # Permutation matrices — identical to C8EquivariantTimmObsEncoder.__init__
        N = n_group
        perm = torch.zeros(N, N, N)
        for r in range(N):
            for i in range(N):
                perm[r, i, (i + r) % N] = 1.0
        self.register_buffer("permutation_matrices", perm)
        self.register_buffer("selected_perm_matrices_template", perm)  # [N, N, N]

    def _apply_frame_averaging(self, features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Identical to C8EquivariantTimmObsEncoder._apply_frame_averaging.

        features: [BT, N, D_out]  — N encoder outputs stacked along dim=1
        returns:  [BT, D_out]     — equivariant regular repr (FA-averaged)
        """
        feature_dim = features.shape[2]
        blocks = feature_dim // self.n_group
        features = features.reshape(batch_size, self.n_group, blocks, self.n_group)
        all_features_flat = features.reshape(-1, blocks, self.n_group)
        perm = self.selected_perm_matrices_template.repeat(batch_size, 1, 1).to(features.dtype)
        aligned = torch.bmm(all_features_flat, perm)                  # [BT*N, blocks, N]
        avg = aligned.reshape(batch_size, self.n_group, blocks, self.n_group).mean(dim=1)
        return avg.reshape(batch_size, blocks * self.n_group)          # [BT, D_out]

    def encode(
        self,
        geo_input: enn.GeometricTensor,   # [(B*T), D_in]  with self.in_type
        cat_ids: torch.Tensor,            # [(B*T),]  embodiment ids
        timestep: torch.Tensor = None,    # [B,] diffusion timestep (action encoder only)
    ):
        """
        Returns:
            equi: [(B*T), D_out]  equivariant regular repr
            inv:  [(B*T), D_out]  invariant (plain mean)
        """
        BT = geo_input.tensor.shape[0]
        h_list = []

        for g_idx in range(self.n_group):
            g_elem = self.in_type.gspace.fibergroup.element(g_idx)

            # h·x — rotate input using its own field type
            h_x_tensor = geo_input.transform(g_elem).tensor            # [(BT), D_in]
            h_x_geo = enn.GeometricTensor(h_x_tensor, self.in_type)

            # f(h·x) — forward through frozen pretrained encoder
            if timestep is not None:
                out = self.pretrained_encoder(h_x_geo, timestep, cat_ids).tensor
            else:
                out = self.pretrained_encoder(h_x_geo, cat_ids).tensor
            h_list.append(out)                                          # [(BT), D_out]

        h_stack = torch.stack(h_list, dim=0)                           # [N, BT, D_out]

        # ── Equivariant FA: ρ_out(h⁻¹) · f(h·x), then mean ──────────────────
        aligned_list = []
        for g_idx, h_out in enumerate(h_list):
            g_inv_idx = (self.n_group - g_idx) % self.n_group
            g_inv = self.in_type.gspace.fibergroup.element(g_inv_idx)
            geo_h_out = enn.GeometricTensor(h_out, self.fa_output_type)
            aligned_list.append(geo_h_out.transform(g_inv).tensor)     # [(BT), D_out]

        # Stack to [BT, N, D_out] and apply permutation-matrix FA (same as C8TimmObsEncoder)
        aligned_stack = torch.stack(aligned_list, dim=1)               # [BT, N, D_out]
        equi = self._apply_frame_averaging(aligned_stack, BT)          # [BT, D_out]

        # ── Invariant FA: plain mean, no transformation ───────────────────────
        inv = h_stack.mean(dim=0)                                       # [BT, D_out]

        return equi, inv
```

---

### Step 2 — `EquiResAdapter` class (new, in `flow_matching_action_head.py`)

```python
class EquiResAdapter(nn.Module):
    """
    Lightweight equivariant residual adapter.
    N BasicTransformerBlocks (SA equi ↔ equi, then CA equi ← trivial inv_output).
    Identity init: equi_delta = 0 at start → output = inv_output_lifted (pure baseline).

    Equivariance:
      SA: regular Q,K,V  → inv-QK + equi-V → equivariant output  ✓
      CA: equi Q (in_type), trivial K/V (inv_type from inv_output)
          → EquivariantAttention else-path: trivial Q,K,V internally
          → o_proj: trivial → in_type  (induction map, always equivariant)  ✓
    """

    def __init__(
        self,
        in_type: enn.FieldType,        # regular repr, size = D = n_group * blocks
        inv_dim: int,                  # dimension of inv_output (= D, plain scalar)
        num_layers: int = 2,
        num_attention_heads: int = 32,
        attention_head_dim: int = 48,
        dropout: float = 0.2,
        final_dropout: bool = True,
    ):
        super().__init__()
        self.in_type = in_type
        gspace = in_type.gspace
        n_group = gspace.fibergroup.order()
        blocks = in_type.size // n_group

        # Trivial type for inv_output cross-attention context
        self.inv_type = enn.FieldType(gspace, [gspace.trivial_repr] * inv_dim)

        # Inner (feedforward hidden) type
        inner_type = enn.FieldType(gspace, [gspace.regular_repr] * blocks)

        self.blocks = nn.ModuleList([
            BasicTransformerBlock(
                in_type=in_type,
                cross_attention_type=self.inv_type,
                inner_type=inner_type,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                norm_type="layer_norm",      # no ada_norm — no timestep conditioning
                final_dropout=final_dropout,
            )
            for _ in range(num_layers)
        ])

        # Identity init: zero-init v_proj, o_proj, ff.fc2 in every block
        # → equi_delta = 0 at init, output = inv_lifted (pure pretrained baseline)
        for blk in self.blocks:
            for p in blk.attn1.v_proj.parameters():
                nn.init.zeros_(p)
            for p in blk.attn1.o_proj.parameters():
                nn.init.zeros_(p)
            for p in blk.ff.fc2.parameters():
                nn.init.zeros_(p)

        # Warm-up scale: 0→1 over warmup_steps (same pattern as EquiAdapter in backbone)
        self.register_buffer("adapter_scale", torch.ones(1))

    def set_warmup_scale(self, step: int, warmup_steps: int) -> None:
        if warmup_steps > 0:
            self.adapter_scale.fill_(min(1.0, step / warmup_steps))

    def forward(
        self,
        equi_sa: torch.Tensor,             # [B, T_sa, D]  regular repr
        context: torch.Tensor,             # [B, T_sa, D]  inv_output from pretrained DiT
    ) -> torch.Tensor:
        """Returns equi_delta [B, T_sa, D] regular repr."""
        h = equi_sa
        h_in = h
        for blk in self.blocks:
            h = blk(h, encoder_hidden_states=context)

        # Scale adapter delta (warm-up)
        if self.adapter_scale.item() != 1.0:
            h = h_in + (h - h_in) * self.adapter_scale
        return h
```

---

### Step 3 — `FlowmatchingActionHeadConfig` changes

Add to the dataclass (after existing fields):

```python
# FA encoder / equi adapter
equi_adapter_num_layers: int = field(
    default=2, metadata={"help": "Number of SA+CA blocks in EquiResAdapter."})
equi_adapter_warmup_steps: int = field(
    default=500, metadata={"help": "Warm-up steps for EquiResAdapter scale 0→1."})
tune_inv_dit: bool = field(
    default=False, metadata={"help": "Whether to fine-tune the frozen pretrained DiT."})
```

The existing `diffusion_model_cfg` remains — it now configures `inv_dit` (the pretrained
DiT loaded from baseline), not an EDiT.

---

### Step 4 — `FlowmatchingActionHead.__init__` changes

**Remove:**
```python
self.model = EDiT(**diffusion_cfg)          # remove: EDiT is replaced
```

**Keep as-is** (used inside FAEncoder):
```python
self.state_in_type    # getJointFieldType(is_action=False)
self.state_out_type   # FieldType([regular_repr] * blocks)  — D = input_embedding_dim
self.action_type      # getJointFieldType(is_action=True) / getActionRelFieldType
self.action_out_type  # FieldType([regular_repr] * blocks)
```

**Add:**
```python
# ── inv_dit: frozen pretrained DiT (from baseline checkpoint) ────────────────
inv_dit_cfg = {
    **config.diffusion_model_cfg,
    "cross_attention_dim": config.backbone_embedding_dim // self.n_group,
}
self.inv_dit = DiT(**inv_dit_cfg)   # DiT from cross_attention_dit.py

# ── FA encoders ──────────────────────────────────────────────────────────────
# pretrained_encoder weights are loaded in load_state_dict via key remapping
self.fa_state_encoder = FAEncoder(
    pretrained_encoder=EquiCategorySpecificMLP(
        num_categories=config.max_num_embodiments,
        in_type=self.state_in_type,
        hidden_type=self.state_hidden_type,
        out_type=self.state_out_type,
    ),
    in_type=self.state_in_type,
    n_group=self.n_group,
    output_dim=config.input_embedding_dim,
)
self.fa_action_encoder = FAEncoder(
    pretrained_encoder=MultiEmbodimentActionEncoder(
        in_type=self.action_type,
        out_type=self.action_out_type,
        num_embodiments=config.max_num_embodiments,
    ),
    in_type=self.action_type,
    n_group=self.n_group,
    output_dim=config.input_embedding_dim,
)

# ── EquiResAdapter ────────────────────────────────────────────────────────────
_D  = config.input_embedding_dim         # must equal n_group * blocks
_blk = _D // self.n_group
_equi_in_type = enn.FieldType(self.group, [self.group.regular_repr] * _blk)
self.equi_res_adapter = EquiResAdapter(
    in_type=_equi_in_type,
    inv_dim=_D,                           # inv_output is D-dim plain tensor
    num_layers=config.equi_adapter_num_layers,
    num_attention_heads=config.diffusion_model_cfg["num_attention_heads"],
    attention_head_dim=config.diffusion_model_cfg["attention_head_dim"],
    dropout=config.diffusion_model_cfg.get("dropout", 0.2),
    final_dropout=config.diffusion_model_cfg.get("final_dropout", True),
)

# ── Equi action decoder (state_hidden_type → action_type, fresh init) ─────────
# unchanged from before
self.action_decoder = EquiCategorySpecificMLP(...)
```

---

### Step 5 — `FlowmatchingActionHead.forward` changes

Replace the single-stream DiT call with dual-stream:

```python
def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
    self.set_frozen_modules_to_eval_mode()
    backbone_output = self.process_backbone_output(backbone_output)

    vl_embs      = backbone_output.vl_features           # [B, T_vl, D]  regular repr
    encoder_mask = backbone_output.backbone_attention_mask

    embodiment_id = action_input.embodiment_id
    B, T_s, _ = action_input.state.shape

    # ── FA State ──────────────────────────────────────────────────────────────
    geo_state = self.getJointGeometricTensor(action_input.state, is_action=False)
    equi_state, inv_state = self.fa_state_encoder.encode(geo_state, embodiment_id)
    equi_state = einops.rearrange(equi_state, '(b t) c -> b t c', b=B, t=T_s)
    inv_state  = einops.rearrange(inv_state,  '(b t) c -> b t c', b=B, t=T_s)

    # ── FA Action ─────────────────────────────────────────────────────────────
    actions_gt = self.getActionGT(action_input.action)
    noise = torch.randn_like(actions_gt)
    t = self.sample_time(B, device=actions_gt.device, dtype=actions_gt.dtype)[:, None, None]
    noisy_trajectory = (1 - t) * noise + t * actions_gt
    velocity = actions_gt - noise
    t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

    T_a = actions_gt.shape[1]
    geo_action = enn.GeometricTensor(
        einops.rearrange(noisy_trajectory, 'b t c -> (b t) c'),
        self.action_type,
    )
    action_emb_id = embodiment_id.repeat(T_a)
    equi_action, inv_action = self.fa_action_encoder.encode(
        geo_action, action_emb_id, t_discretized
    )
    equi_action = einops.rearrange(equi_action, '(b t) c -> b t c', b=B, t=T_a)
    inv_action  = einops.rearrange(inv_action,  '(b t) c -> b t c', b=B, t=T_a)

    # ── Temporal pos embed (on all four streams) ──────────────────────────────
    if self.config.add_pos_embed:
        equi_state  = self._add_temporal_pos_embed(equi_state)
        inv_state   = self._add_temporal_pos_embed(inv_state)
        equi_action = self._add_temporal_pos_embed(equi_action)
        inv_action  = self._add_temporal_pos_embed(inv_action)

    # ── Invariant branch: frozen pretrained DiT ───────────────────────────────
    N = self.n_group
    D = vl_embs.shape[-1]
    T_vl = vl_embs.shape[1]
    vl_inv = vl_embs.reshape(B, T_vl, D // N, N).mean(-1)  # [B, T_vl, D/N]

    sa_embs_inv = torch.cat([inv_state, inv_action], dim=1)  # [B, T_sa, D]
    inv_output  = self.inv_dit(
        hidden_states=sa_embs_inv,
        encoder_hidden_states=vl_inv,
        encoder_attention_mask=encoder_mask,
        timestep=t_discretized,
    )                                                         # [B, T_sa, D]

    # ── Equivariant branch: lightweight EquiResAdapter ────────────────────────
    sa_embs_equi = torch.cat([equi_state, equi_action], dim=1)  # [B, T_sa, D]
    equi_delta   = self.equi_res_adapter(sa_embs_equi, context=inv_output)

    # ── Residual fusion: trivial-in-regular lift + equi delta ─────────────────
    # inv_output [B, T_sa, D] → lift to regular repr (repeat each block N times)
    T_sa = inv_output.shape[1]
    inv_lifted = (inv_output
                  .view(B, T_sa, D // N, 1)
                  .expand(-1, -1, -1, N)
                  .reshape(B, T_sa, D))
    output = inv_lifted + equi_delta                          # [B, T_sa, D]

    # ── Equi action decoder ───────────────────────────────────────────────────
    action_decoder_emb_id = embodiment_id.repeat(T_sa)
    output_flat = enn.GeometricTensor(
        einops.rearrange(output, 'b t c -> (b t) c'),
        self.state_hidden_type,
    )
    pred = self.action_decoder(output_flat, action_decoder_emb_id)
    pred = einops.rearrange(pred.tensor, '(b t) c -> b t c', b=B, t=T_sa)

    pred_actions = pred[:, -T_a:]
    with torch.no_grad():
        action_mask = self.transform_action_mask(
            action_input.action_mask.float(), velocity.shape[-1]
        ).to(velocity.device)
    loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
    loss = loss.sum() / action_mask.sum()
    return BatchFeature(data={"loss": loss})
```

---

### Step 6 — `FlowmatchingActionHead.get_action` changes

Same dual-stream logic, inside the denoising loop:

```python
@torch.no_grad()
def get_action(self, backbone_output, action_input):
    ...
    # State FA (once, outside the loop)
    geo_state = self.getJointGeometricTensor(action_input.state, is_action=False)
    equi_state, inv_state = self.fa_state_encoder.encode(geo_state, embodiment_id)
    equi_state = einops.rearrange(equi_state, '(b t) c -> b t c', b=B, t=T_s)
    inv_state  = einops.rearrange(inv_state,  '(b t) c -> b t c', b=B, t=T_s)
    if self.config.add_pos_embed:
        equi_state = self._add_temporal_pos_embed(equi_state)
        inv_state  = self._add_temporal_pos_embed(inv_state)

    vl_inv = vl_embs.reshape(B, T_vl, D // N, N).mean(-1)

    for t_step in range(num_steps):
        ...
        # Action FA
        geo_action = enn.GeometricTensor(rearrange(actions, 'b t c -> (b t) c'), self.action_type)
        action_emb_id = embodiment_id.repeat(T_a)
        equi_action, inv_action = self.fa_action_encoder.encode(
            geo_action, action_emb_id, timesteps_tensor
        )
        equi_action = rearrange(equi_action, '(b t) c -> b t c', b=B, t=T_a)
        inv_action  = rearrange(inv_action,  '(b t) c -> b t c', b=B, t=T_a)
        if self.config.add_pos_embed:
            equi_action = self._add_temporal_pos_embed(equi_action)
            inv_action  = self._add_temporal_pos_embed(inv_action)

        # Dual-stream
        sa_embs_inv  = cat([inv_state,  inv_action],  dim=1)
        sa_embs_equi = cat([equi_state, equi_action], dim=1)
        inv_output   = self.inv_dit(sa_embs_inv, vl_inv, timesteps_tensor)
        equi_delta   = self.equi_res_adapter(sa_embs_equi, context=inv_output)
        inv_lifted   = inv_output.view(B,T_sa,D//N,1).expand(-1,-1,-1,N).reshape(B,T_sa,D)
        output       = inv_lifted + equi_delta

        # Decode
        output_flat = enn.GeometricTensor(rearrange(output,'b t c->(b t) c'), self.state_hidden_type)
        pred = self.action_decoder(output_flat, embodiment_id.repeat(T_sa))
        pred = rearrange(pred.tensor, '(b t) c -> b t c', b=B, t=T_sa)
        pred_velocity = pred[:, -self.action_horizon:]
        actions = actions + dt * pred_velocity

    actions = self.getActionOutput(actions)
    return BatchFeature(data={"action_pred": actions})
```

---

### Step 7 — `load_state_dict` remapping

```python
def load_state_dict(self, state_dict, strict=True, assign=False):
    remapped = {}
    for key, value in state_dict.items():
        # Remap pretrained DiT weights → inv_dit
        if key.startswith('model.'):
            new_key = 'inv_dit.' + key[len('model.'):]
            remapped[new_key] = value
            continue
        # Remap state encoder → fa_state_encoder.pretrained_encoder
        if key.startswith('state_encoder.'):
            new_key = 'fa_state_encoder.pretrained_encoder.' + key[len('state_encoder.'):]
            remapped[new_key] = value
            continue
        # Remap action encoder → fa_action_encoder.pretrained_encoder
        if key.startswith('action_encoder.'):
            new_key = 'fa_action_encoder.pretrained_encoder.' + key[len('action_encoder.'):]
            remapped[new_key] = value
            continue
        # Skip keys that are replaced with fresh init
        skip_prefixes = (
            'action_decoder.',
            'equi_res_adapter.',
            # old keys that no longer exist:
            'future_tokens_equi_proj.', 'vl_equi_proj.', 'vlln.',
            'vl_self_attention.', 'language_proj.', 'language_lift.',
            'equi_vis_proj.',
        )
        if key.startswith(skip_prefixes):
            print(f"Skipping (fresh init): {key}")
            continue
        remapped[key] = value

    return super().load_state_dict(remapped, strict=False, assign=assign)
```

---

### Step 8 — `set_trainable_parameters` changes

```python
def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
    self.tune_projector = tune_projector
    self.tune_diffusion_model = tune_diffusion_model

    # All params trainable by default
    for p in self.parameters():
        p.requires_grad = True

    # Always freeze the pretrained encoders inside FAEncoder
    self.fa_state_encoder.pretrained_encoder.requires_grad_(False)
    self.fa_action_encoder.pretrained_encoder.requires_grad_(False)

    # Always freeze inv_dit unless explicitly requested
    if not self.config.tune_inv_dit:
        self.inv_dit.requires_grad_(False)

    if not tune_projector:
        if self.config.add_pos_embed:
            self.temporal_pos_embed.requires_grad_(False)
            self.temporal_pos_proj.requires_grad_(False)

    if not tune_diffusion_model:
        self.equi_res_adapter.requires_grad_(False)
```

---

### Step 9 — `set_frozen_modules_to_eval_mode` changes

```python
def set_frozen_modules_to_eval_mode(self):
    if self.training:
        self.fa_state_encoder.pretrained_encoder.eval()
        self.fa_action_encoder.pretrained_encoder.eval()
        if not self.config.tune_inv_dit:
            self.inv_dit.eval()
        if not self.tune_projector and self.config.add_pos_embed:
            self.temporal_pos_embed.eval()
            self.temporal_pos_proj.eval()
        if not self.tune_diffusion_model:
            self.equi_res_adapter.eval()
```

---

## Dimension Reference

| Tensor | Shape | Notes |
|--------|-------|-------|
| `geo_state` | `[(B*T_s), D_in_state]` | `D_in_state = state_in_type.size` |
| `equi_state`, `inv_state` | `[B, T_s, D]` | `D = input_embedding_dim` |
| `geo_action` | `[(B*T_a), D_in_action]` | `D_in_action = action_type.size` |
| `equi_action`, `inv_action` | `[B, T_a, D]` | `D = input_embedding_dim` |
| `sa_embs_inv` | `[B, T_sa, D]` | `T_sa = T_s + T_a` |
| `inv_output` | `[B, T_sa, D]` | output of pretrained `DiT` |
| `sa_embs_equi` | `[B, T_sa, D]` | regular repr |
| `equi_delta` | `[B, T_sa, D]` | regular repr, zero at init |
| `inv_lifted` | `[B, T_sa, D]` | regular repr (trivial-in-regular) |
| `output` | `[B, T_sa, D]` | regular repr |
| `vl_embs` | `[B, T_vl, D]` | backbone output, regular repr |
| `vl_inv` | `[B, T_vl, D/N]` | invariant, for pretrained DiT ctx |
| `fa_output_type` | size `D` | `FieldType([regular_repr] * (D//N))` |
| `inv_type` in EquiResAdapter | size `D` | `FieldType([trivial_repr] * D)` |

Note: `inv_dit.cross_attention_dim` must equal `D/N = backbone_embedding_dim // n_group`.
Set in config: `diffusion_model_cfg["cross_attention_dim"] = backbone_embedding_dim // n_group`.

---

## Checkpoint Loading Strategy

| Weight key in checkpoint | Remapped to | Frozen? |
|--------------------------|-------------|---------|
| `action_head.model.*` | `inv_dit.*` | Yes (unless `tune_inv_dit=True`) |
| `action_head.state_encoder.*` | `fa_state_encoder.pretrained_encoder.*` | Yes |
| `action_head.action_encoder.*` | `fa_action_encoder.pretrained_encoder.*` | Yes |
| `action_head.action_decoder.*` | skipped (fresh init) | — |
| `action_head.temporal_pos_embed.*` | unchanged | No |
| `action_head.temporal_pos_proj.*` | unchanged | No |
| `action_head.vl_layer_norm.*` | unchanged | No |
| `action_head.equi_res_adapter.*` | skipped (fresh init) | — |

---

## Trainable Parameter Summary

```
FROZEN (pretrained baseline, loaded from gr00t_baseline_locht1):
  fa_state_encoder.pretrained_encoder      ← baseline state_encoder
  fa_action_encoder.pretrained_encoder     ← baseline action_encoder (MultiEmbodiment)
  inv_dit                                  ← baseline DiT (16 layers, 1536-dim)

TRAINABLE (~12% overhead for N=2 blocks):
  equi_res_adapter.*     (2 × BasicTransformerBlock, equivariant SA+CA+FF)
  action_decoder.*       (EquiCategorySpecificMLP, fresh)
  temporal_pos_embed     (if tune_projector)
  temporal_pos_proj      (if tune_projector)
  vl_layer_norm          (always trainable)
```

---

## Parameter Overhead

| Config | Extra params | Overhead |
|--------|-------------|----------|
| `equi_adapter_num_layers=2` | ~2/16 of DiT | ~12% |
| `equi_adapter_num_layers=4` | ~4/16 of DiT | ~25% |

Default: **`equi_adapter_num_layers=2`**

---

## Expected Benefits

| Property | Before | After |
|----------|--------|-------|
| Equivariance | ✓ native escnn | ✓ FA + EquiResAdapter |
| Pretrained state/action repr | ✗ random init | ✓ FA wraps frozen baseline |
| Pretrained DiT trajectory model | ✗ EDiT from scratch | ✓ inv_dit from baseline |
| Geometric correction | ✓ full pipeline | ✓ EquiResAdapter delta |
| Extra parameter cost | 0% (but bad quality) | **+12%** (N=2, high quality) |
| Identity init at start | N/A | ✓ equi_delta=0 → pure baseline |

---

## Equivariance Tests

Test file: `scripts/test_fa_action_head_equivariance.py`

Pattern follows `scripts/test_action_head_equivariance.py` — same `apply_group_action`
helper, same per-element loop, same `atol` threshold.

**Core checker helper** (reuse across all tests):

```python
def check_equivariance(
    label: str,
    fn,                          # callable: x -> output (both plain tensors)
    x: torch.Tensor,             # [B, D_in]
    in_type: enn.FieldType,
    out_type: enn.FieldType,
    n_group: int,
    atol: float = 1e-4,
) -> bool:
    """
    Checks fn(ρ_in(g)·x) ≈ ρ_out(g)·fn(x)  for all g in C_N.
    fn receives and returns plain tensors (not GeometricTensors).
    """
    G = in_type.gspace.fibergroup
    with torch.no_grad():
        out_orig = fn(x)                                   # [B, D_out]
    all_passed = True
    for r in range(n_group):
        g = G.element(r)
        x_rot = apply_group_action(x, in_type, r, n_group)
        with torch.no_grad():
            out_rot = fn(x_rot)
        expected = apply_group_action(out_orig, out_type, r, n_group)
        err = (out_rot - expected).abs().mean().item()
        passed = err < atol
        all_passed = all_passed and passed
        angle = r * 360.0 / n_group
        print(f"  [{label}] g_{r} ({angle:.1f}°): mean_err={err:.3e}  {'✓' if passed else '✗'}")
    return all_passed
```

---

### Test 1 — `FAEncoder` (state): FA_equi invariance + FA_inv invariance

**What to verify:**
- `FA_equi(g·x) = ρ_out(g) · FA_equi(x)` — equivariant output (regular repr)
- `FA_inv(g·x)  = FA_inv(x)` — invariant output (plain mean)

**Setup:**

```python
n_group = 4
group   = gspaces.no_base_space(CyclicGroup(n_group))
state_in_type  = make_joint_field_type(group, ...)      # irrep(1)+trivial
output_dim     = 32                                      # n_group * blocks
fa_output_type = FieldType(group, [regular_repr] * (output_dim // n_group))

# pretrained_encoder: EquiCategorySpecificMLP (state_encoder shape)
pretrained_enc = EquiCategorySpecificMLP(
    num_categories=2, in_type=state_in_type,
    hidden_type=make_regular_type(group, 32),
    out_type=make_regular_type(group, output_dim),
)
fa_enc = FAEncoder(pretrained_enc, state_in_type, n_group, output_dim)
fa_enc.eval()

cat_ids = torch.zeros(B, dtype=torch.long)
x = torch.randn(B, state_in_type.size)

# Test equi output
def fn_equi(x_tensor):
    geo = enn.GeometricTensor(x_tensor, state_in_type)
    equi, _ = fa_enc.encode(geo, cat_ids)
    return equi

passed_equi = check_equivariance(
    "FAEncoder equi",
    fn_equi, x, state_in_type, fa_output_type, n_group
)

# Test inv output — must be fully invariant (error vs r=0 output for all g)
def fn_inv(x_tensor):
    geo = enn.GeometricTensor(x_tensor, state_in_type)
    _, inv = fa_enc.encode(geo, cat_ids)
    return inv

trivial_type = FieldType(group, [trivial_repr] * output_dim)  # all trivial
passed_inv = check_equivariance(
    "FAEncoder inv (should be invariant)",
    fn_inv, x, state_in_type, trivial_type, n_group
)
```

---

### Test 2 — `FAEncoder` (action): same as Test 1 but with `MultiEmbodimentActionEncoder`

```python
action_in_type = make_joint_field_type(group, ..., is_action=True)
pretrained_action_enc = MultiEmbodimentActionEncoder(
    in_type=action_in_type, out_type=make_regular_type(group, output_dim),
    num_embodiments=2,
)
fa_action_enc = FAEncoder(pretrained_action_enc, action_in_type, n_group, output_dim)
fa_action_enc.eval()

timestep = torch.zeros(B, dtype=torch.long)

def fn_action_equi(x_tensor):
    geo = enn.GeometricTensor(x_tensor, action_in_type)
    equi, _ = fa_action_enc.encode(geo, cat_ids, timestep=timestep)
    return equi

passed_action_equi = check_equivariance(
    "FAEncoder action equi", fn_action_equi,
    torch.randn(B, action_in_type.size),
    action_in_type, fa_output_type, n_group
)
```

---

### Test 3 — `EquiResAdapter` identity init: output = 0

Before any training, `equi_delta` must be identically zero (zero-init proof).

```python
D = output_dim                        # e.g. 32
N = n_group
blk = D // N
equi_in_type = FieldType(group, [regular_repr] * blk)
inv_dim = D

adapter = EquiResAdapter(equi_in_type, inv_dim, num_layers=2)
adapter.eval()

B_t, T_sa = 3, 6
equi_sa  = torch.randn(B_t, T_sa, D)
inv_ctx  = torch.randn(B_t, T_sa, D)

with torch.no_grad():
    delta = adapter(equi_sa, context=inv_ctx)

max_val = delta.abs().max().item()
print(f"  [EquiResAdapter identity init] max |delta| = {max_val:.3e}")
passed_identity = max_val < 1e-6
```

---

### Test 4 — `EquiResAdapter` equivariance (after random init override)

Override zero-init so the adapter has non-trivial weights, then test equivariance.

```python
# Re-init with non-zero weights
for blk in adapter.blocks:
    nn.init.normal_(blk.attn1.v_proj.weight, std=0.02)
    nn.init.normal_(blk.attn1.o_proj.weight, std=0.02)
    nn.init.normal_(blk.ff.fc2.weight, std=0.02)
adapter.eval()

# equi_sa input: regular repr [B, T_sa, D] → flatten to [B*T_sa, D]
# inv_ctx input: plain invariant [B, T_sa, D] → treated as trivial

def fn_adapter(x_flat):
    # x_flat: [B*T_sa, D]
    x_3d = x_flat.reshape(B_t, T_sa, D)
    with torch.no_grad():
        out = adapter(x_3d, context=inv_ctx)        # context is FIXED (invariant)
    return out.reshape(B_t * T_sa, D)

# equi_in_type is [regular_repr]*blk, size=D
in_type_flat  = FieldType(group, [regular_repr] * blk)
out_type_flat = FieldType(group, [regular_repr] * blk)
x_flat = torch.randn(B_t * T_sa, D)

passed_adapter_equi = check_equivariance(
    "EquiResAdapter equivariance", fn_adapter,
    x_flat, in_type_flat, out_type_flat, n_group
)
```

Note: `inv_ctx` (context) is kept FIXED during the rotation test because it is invariant
(pretrained DiT output, plain D-dim). Rotating only the equivariant input is the correct
equivariance test: `adapter(g·equi_sa, ctx) = g · adapter(equi_sa, ctx)`.

---

### Test 5 — Residual fusion equivariance

```python
# inv_lifted = inv_output.view(B,T,D//N,1).expand(...,N).reshape(B,T,D)
# output = inv_lifted + equi_delta
# Both summands are in regular repr → output is equivariant

def fn_fusion(equi_flat):
    equi_3d  = equi_flat.reshape(B_t, T_sa, D)
    inv_out  = torch.randn(B_t, T_sa, D)             # FIXED invariant context
    inv_lift = inv_out.view(B_t, T_sa, D//N, 1).expand(-1,-1,-1,N).reshape(B_t,T_sa,D)
    out = inv_lift + equi_3d                          # equi + trivial-in-regular
    return out.reshape(B_t * T_sa, D)

passed_fusion = check_equivariance(
    "Residual fusion equivariance",
    fn_fusion, x_flat, in_type_flat, out_type_flat, n_group
)
```

---

### Test 6 — Full `FlowmatchingActionHead` end-to-end equivariance

Tests the complete pipeline with a **mocked backbone output** (no VLM needed).

**Property**: rotating the state + action input rotates the predicted velocity output.

```python
# Build minimal FlowmatchingActionHead with small dims
config = FlowmatchingActionHeadConfig(
    n_group=4,
    input_embedding_dim=32,   # must be n_group * blocks
    backbone_embedding_dim=32,
    hidden_size=32,
    action_dim=26,
    action_horizon=4,
    max_state_dim=64,
    max_action_dim=26,
    num_hand=2,
    equi_adapter_num_layers=2,
    diffusion_model_cfg={
        "num_attention_heads": 4,
        "attention_head_dim": 8,    # scalar_dim = 32, divisible by n_group=4 ✓
        "num_layers": 2,
        "cross_attention_dim": 8,   # = backbone_embedding_dim // n_group = 32//4
        "output_dim": 32,
        "dropout": 0.0,
        "final_dropout": False,
        "norm_type": "ada_norm",
        "interleave_self_attention": False,
    },
)
head = FlowmatchingActionHead(config)
head.eval()

B_e, T_s_e, T_a_e = 2, 1, 4
state_dim  = config.max_state_dim     # e.g. 64 (matches getJointFieldType)
action_dim = config.action_dim        # e.g. 26

# Fixed invariant backbone output (trivial-in-regular)
D = config.backbone_embedding_dim     # 32
N = config.n_group                    # 4
T_vl = 8
vl_embs = torch.randn(B_e, T_vl, D)
# Make it trivial-in-regular: each block repeated N times
blocks_vl = D // N
vl_scalar = torch.randn(B_e, T_vl, blocks_vl)
vl_embs = vl_scalar.unsqueeze(-1).expand(-1, -1, -1, N).reshape(B_e, T_vl, D)

backbone_output = BatchFeature(data={
    "backbone_equi_vision_features": vl_embs,
    "backbone_attention_mask": torch.ones(B_e, T_vl, dtype=torch.long),
})

state  = torch.randn(B_e, T_s_e, state_dim)
action = torch.randn(B_e, T_a_e, action_dim)
action_mask = torch.ones(B_e, T_a_e, action_dim)
action_input = BatchFeature(data={
    "state": state,
    "action": action,
    "action_mask": action_mask,
    "embodiment_id": torch.zeros(B_e, dtype=torch.long),
})

# Get reference output
with torch.no_grad():
    out_orig = head(backbone_output, action_input)
    # Access internal equi output before decoder for equivariance check
    # (test at pred_actions level checks end-to-end)

# Rotate state + action via getJointGeometricTensor round-trip
for r in range(n_group):
    state_geo  = head.getJointGeometricTensor(state, is_action=False)
    action_geo_flat = enn.GeometricTensor(
        einops.rearrange(head.getActionGT(action), 'b t c -> (b t) c'),
        head.action_type
    )
    g = head.group.fibergroup.element(r)

    # Rotate in geometric tensor space, convert back to raw state
    # (simplification: test at the equi/inv feature level, not raw state)
    # Full test: rotate equi_state, equi_action; check equi output rotates accordingly
    state_rot_geo = state_geo.transform(g)
    # ... (reconstruct rotated state for action_input)

    # For a cleaner test, directly test FA outputs:
    equi_s, inv_s = head.fa_state_encoder.encode(state_geo,  emb_id)
    equi_s_from_rot, inv_s_from_rot = head.fa_state_encoder.encode(state_rot_geo_new, emb_id)

    expected_equi_s = apply_group_action(equi_s, fa_output_type, r, n_group)
    err_equi = (equi_s_from_rot - expected_equi_s).abs().mean()
    err_inv  = (inv_s_from_rot  - inv_s).abs().mean()

    print(f"  [e2e FA state] g_{r}: equi_err={err_equi:.3e}  inv_err={err_inv:.3e}")
```

---

### Test file structure

```
scripts/test_fa_action_head_equivariance.py
│
├── apply_group_action(x, field_type, r, n_group) → tensor   # from existing test
├── check_equivariance(label, fn, x, in_type, out_type, ...)  # shared helper
│
├── test_fa_encoder_equi()           → Test 1 + 2
├── test_equi_res_adapter_identity() → Test 3
├── test_equi_res_adapter_equi()     → Test 4
├── test_residual_fusion_equi()      → Test 5
├── test_action_head_e2e_equi()      → Test 6
│
└── main() — runs all, prints summary, sys.exit(0/1)
```

Run with:
```bash
cd /Users/lochathien/Documents/Code/Isaac-GR00T
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate equidiff
python scripts/test_fa_action_head_equivariance.py
```

---

### Tolerance guidelines

| Test | Expected max mean_err | Notes |
|------|----------------------|-------|
| FAEncoder equi | `< 1e-4` | float32; FA is exact up to fp rounding |
| FAEncoder inv | `< 1e-5` | plain mean, tighter tolerance |
| EquiResAdapter identity | `< 1e-6` | zero-init, should be exact |
| EquiResAdapter equi | `< 1e-4` | escnn linear layers |
| Residual fusion | `< 1e-6` | algebraic, no learned params |
| E2E FA features | `< 1e-4` | full pipeline |


# Gr00t N1.5 baseline config
```
{
  "action_dim": 32,
  "action_head_cfg": {
    "action_dim": 32,
    "action_horizon": 16,
    "add_pos_embed": true,
    "backbone_embedding_dim": 2048,
    "diffusion_model_cfg": {
      "attention_head_dim": 48,
      "cross_attention_dim": 2048,
      "dropout": 0.2,
      "final_dropout": true,
      "interleave_self_attention": true,
      "norm_type": "ada_norm",
      "num_attention_heads": 32,
      "num_layers": 16,
      "output_dim": 1024,
      "positional_embeddings": null
    },
    "hidden_size": 1024,
    "input_embedding_dim": 1536,
    "max_action_dim": 32,
    "max_state_dim": 64,
    "model_dtype": "float32",
    "noise_beta_alpha": 1.5,
    "noise_beta_beta": 1.0,
    "noise_s": 0.999,
    "num_inference_timesteps": 4,
    "num_target_vision_tokens": 32,
    "num_timestep_buckets": 1000,
    "tune_diffusion_model": true,
    "tune_projector": true,
    "use_vlln": true,
    "vl_self_attention_cfg": {
      "attention_head_dim": 64,
      "dropout": 0.2,
      "final_dropout": true,
      "num_attention_heads": 32,
      "num_layers": 4,
      "positional_embeddings": null
    }
  },
  "action_horizon": 16,
  "architectures": [
    "GR00T_N1_5"
  ],
  "attn_implementation": null,
  "backbone_cfg": {
    "eagle_path": "NVEagle/eagle_er-qwen3_1_7B-Siglip2_400M_stage1_5_128gpu_er_v7_1mlp_nops",
    "load_bf16": false,
    "project_to_dim": null,
    "reproject_vision": false,
    "select_layer": 12,
    "tune_llm": false,
    "tune_visual": true,
    "use_flash_attention": true
  },
  "hidden_size": 2048,
  "model_dtype": "float32",
  "model_type": "gr00t_n1_5",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.3"
}
```