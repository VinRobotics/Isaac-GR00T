# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# RECAP Policy Training Phase
# Implements RECAP §IV-B Eq. 3:
#   min_θ E_D [ -log πθ(at|ot,ℓ) - α log πθ(at|It,ot,ℓ) ]
#   where It = 1( A^πref(ot,at,ℓ) > ε_ℓ )
#
# Advantage labels come from the pretrained frozen value head (Phase 1 output).
# References:
#   π*0.6 / RECAP  arXiv:2511.14759  §IV-A, §IV-B, App. F
#   CFGRL          arXiv:2505.23458  Alg. 1, Alg. 2

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from .flow_matching_action_head import (   # your existing file
    AdvantageEmbedding,
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
    DistributionalValueHead,
    compute_normalised_returns,
    get_prefix_weights,
)


# ---------------------------------------------------------------------------
# Advantage labelling from frozen value head
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_advantage_labels_from_value_head(
    value_head:     DistributionalValueHead,
    backbone_feats: torch.Tensor,   # (B, S, D)  VL features (pre-advantage-token)
    state:          torch.Tensor,   # (B, state_dim)
    reward:         torch.Tensor,   # (B,)  {-1=fail, 0=success}
    percentile:     float = 0.30,   # ε_ℓ threshold (30th pct pre-train, 40th fine-tune)
) -> torch.Tensor:
    """
    Derive per-step advantage indicator I_t from the frozen value head.

    RECAP App. F: ε_ℓ is set at the 30th percentile of V values predicted
    by the value function for the current task ℓ, so ~30% of steps get
    positive advantage (POS_IDX).

    Hard rule: failure steps (reward=-1) are ALWAYS labelled NEG regardless
    of their predicted value — failure is never positive advantage.

    Returns (B,) long tensor ∈ {NEG_IDX=1, POS_IDX=2}
    """
    V       = value_head.predict_value(backbone_feats, state)       # (B,)
    epsilon = torch.quantile(V, percentile)                          # scalar ε_ℓ

    is_failure      = reward < 0                                     # (B,) bool
    above_threshold = V > epsilon                                    # (B,) bool

    labels = torch.where(
        above_threshold & ~is_failure,
        torch.full_like(V, AdvantageEmbedding.POS_IDX, dtype=torch.long),
        torch.full_like(V, AdvantageEmbedding.NEG_IDX, dtype=torch.long),
    )
    return labels                                                     # (B,)


# ---------------------------------------------------------------------------
# RECAP Eq. 3 — policy training forward
# ---------------------------------------------------------------------------

def forward_policy_recap(
    action_head:    FlowmatchingActionHead,
    backbone_output: BatchFeature,
    action_input:   BatchFeature,
    alpha:          float = 1.0,    # weight of conditional term in Eq. 3
) -> BatchFeature:
    """
    RECAP §IV-B Eq. 3 — policy training with advantage conditioning.

    Training objective (Eq. 3):
        min_θ  E_{D_πref} [ -log πθ(at|ot,ℓ) - α log πθ(at|It,ot,ℓ) ]
        where  It = 1( A^πref(ot,at,ℓ) > ε_ℓ )

    In flow-matching terms this becomes:

        L = ||vθ(at_noisy, t, s, ∅)  − velocity||²          ← unconditional term
          + α * ||vθ(at_noisy, t, s, It) − velocity||²       ← conditional term

    The two terms share the same noised trajectory and velocity target.
    CFG dropout (prob p) additionally trains the unconditional branch
    by randomly replacing It with ∅ — this is the implementation of
    the "randomly omit the indicator" from RECAP App. F.

    Required keys in action_input:
        state              : (B, state_dim)
        action             : (B, H, action_dim)
        action_mask        : (B, H, action_dim)
        embodiment_id      : (B,)
        reward             : (B, 1) or (B,)   {-1=fail, 0=success}
        reward.current_frame_idx   : (B, 1)   step index t
        reward.episode_lengths     : (B, 1)   episode length T

    Returns BatchFeature with keys:
        loss               : total loss (unconditional + α * conditional)
        action_loss_uncond : ||vθ(∅) − v||²
        action_loss_cond   : ||vθ(It) − v||²
        advantage_pos_frac : fraction of steps labelled POS (monitoring)
    """
    action_head.set_frozen_modules_to_eval_mode()
    backbone_output = action_head.process_backbone_output(backbone_output)

    # ── unpack ────────────────────────────────────────────────────────────
    vl_embs_raw   = backbone_output.backbone_features       # (B, S, D)  clean VL
    vl_attn_mask  = backbone_output.backbone_attention_mask # (B, S)
    device        = vl_embs_raw.device
    embodiment_id = action_input.embodiment_id

    reward = torch.squeeze(action_input["reward"], dim=-1).float()  # (B,)

    # ── Step 1: compute advantage labels from frozen value head ───────────
    # Use clean VL features (no advantage token yet) so the value head
    # sees the same representation it was trained on in Phase 1.
    with torch.no_grad():
        adv_labels = compute_advantage_labels_from_value_head(
            value_head=action_head.value_head,
            backbone_feats=vl_embs_raw,
            state=action_input.state.float(),
            reward=reward,
            percentile=action_head.config.advantage_threshold_percentile
            if hasattr(action_head.config, "advantage_threshold_percentile")
            else 0.30,
        )                                                            # (B,) long

    pos_frac = (adv_labels == AdvantageEmbedding.POS_IDX).float().mean()

    # ── Step 2: build noised trajectory (shared for both terms) ──────────
    # CFGRL Alg 1: a0 ~ N(0,I), t ~ U(0,1)
    state_features = action_head.state_encoder(action_input.state, embodiment_id)

    actions  = action_input.action                              # (B, H, action_dim)
    noise    = torch.randn_like(actions)                       # a0 ~ N(0,I)
    t        = torch.rand(actions.shape[0], device=device, dtype=actions.dtype)
    t_bcast  = t[:, None, None]

    noisy_trajectory = (1 - t_bcast) * noise + t_bcast * actions
    velocity         = actions - noise                          # regression target

    t_discretized   = (t * action_head.num_timestep_buckets).long()
    action_features = action_head.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

    if action_head.config.add_pos_embed:
        pos_ids         = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        action_features = action_features + action_head.position_embedding(pos_ids).unsqueeze(0)

    future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
        vl_embs_raw.shape[0], -1, -1
    )
    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

    action_mask = action_input.action_mask                     # (B, H, action_dim)

    # ── Step 3: UNCONDITIONAL term  −log πθ(at|ot,ℓ) ─────────────────────
    # Inject NULL token — no optimality signal
    vl_embs_null, vl_attn_mask_null = action_head._apply_advantage_conditioning(
        vl_embs_raw, vl_attn_mask,
        advantage_label=None,    # → all NULL_IDX = ∅
    )

    out_null   = action_head.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_embs_null,
        encoder_attention_mask=vl_attn_mask_null,   # ← correct augmented mask
        timestep=t_discretized,
        return_all_hidden_states=False,
    )
    pred_null  = action_head.action_decoder(out_null, embodiment_id)
    pred_null  = pred_null[:, -actions.shape[1]:]

    loss_uncond = F.mse_loss(pred_null, velocity, reduction="none") * action_mask
    loss_uncond = loss_uncond.sum() / action_mask.sum()

    # ── Step 4: CONDITIONAL term  −α log πθ(at|It,ot,ℓ) ─────────────────
    # Inject advantage token with CFG dropout
    # During training _apply_advantage_conditioning automatically applies
    # dropout (prob p → NULL), which trains the conditional branch on (1-p)
    # of the data and the unconditional branch on p of the data.
    vl_embs_cond, vl_attn_mask_cond = action_head._apply_advantage_conditioning(
        vl_embs_raw, vl_attn_mask,
        advantage_label=adv_labels,   # It ∈ {NEG, POS} from value head
    )

    out_cond   = action_head.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_embs_cond,
        encoder_attention_mask=vl_attn_mask_cond,   # ← correct augmented mask
        timestep=t_discretized,
        return_all_hidden_states=False,
    )
    pred_cond  = action_head.action_decoder(out_cond, embodiment_id)
    pred_cond  = pred_cond[:, -actions.shape[1]:]

    loss_cond = F.mse_loss(pred_cond, velocity, reduction="none") * action_mask
    loss_cond = loss_cond.sum() / action_mask.sum()

    # ── Step 5: total loss (Eq. 3) ─────────────────────────────────────────
    # L = loss_uncond + α * loss_cond
    total_loss = loss_uncond + alpha * loss_cond

    return BatchFeature(data={
        "loss":               total_loss,
        "action_loss_uncond": loss_uncond.detach(),
        "action_loss_cond":   loss_cond.detach(),
        "advantage_pos_frac": pos_frac.detach(),
    })


# ---------------------------------------------------------------------------
# Updated forward() — drop-in replacement for FlowmatchingActionHead.forward()
# ---------------------------------------------------------------------------

def forward_recap(
    self: FlowmatchingActionHead,
    backbone_output: BatchFeature,
    action_input:    BatchFeature,
) -> BatchFeature:
    """
    Unified forward() for FlowmatchingActionHead supporting both phases.

    Dispatches based on whether the value head is frozen:
        Phase 1 (value head training):  only value_loss returned
        Phase 2 (policy training):      Eq. 3 dual-term loss returned

    To switch phases call:
        # Phase 1
        for p in model.action_head.value_head.parameters():
            p.requires_grad = True
        for p in model.action_head.model.parameters():
            p.requires_grad = False
        ...

        # Phase 2
        for p in model.action_head.value_head.parameters():
            p.requires_grad = False
        for p in model.action_head.model.parameters():
            p.requires_grad = True
        ...
    """
    value_head_trainable = any(
        p.requires_grad for p in self.value_head.parameters()
    )

    if value_head_trainable:
        return _forward_value_head_phase(self, backbone_output, action_input)
    else:
        return _forward_policy_phase(self, backbone_output, action_input)


def _forward_value_head_phase(
    self: FlowmatchingActionHead,
    backbone_output: BatchFeature,
    action_input:    BatchFeature,
) -> BatchFeature:
    """
    Phase 1 — RECAP §IV-A Eq. 1.
    Policy frozen. Only value head updated.

    Required keys:
        reward                      : (B,1) or (B,)
        reward.current_frame_idx    : (B,1)
        reward.episode_lengths      : (B,1)
    """
    self.set_frozen_modules_to_eval_mode()
    backbone_output = self.process_backbone_output(backbone_output)

    vl_embs = backbone_output.backbone_features
    device  = vl_embs.device

    reward = torch.squeeze(action_input["reward"], dim=-1).float()
    t_idx  = action_input["reward.current_frame_idx"].squeeze(dim=-1).long()
    ep_len = action_input["reward.episode_lengths"].squeeze(dim=-1).long()

    max_ep  = 1040 // 2
    c_fail  = max_ep // 2

    empirical_return = compute_normalised_returns(
        success=reward >= 0,
        episode_lengths=ep_len,
        t=t_idx,
        max_episode_length=max_ep,
        c_fail=c_fail,
    )

    value_loss = self.value_head.compute_loss(
        vl_embs,
        action_input.state.float(),
        empirical_return,
    )

    return BatchFeature(data={
        "loss":       value_loss,
        "value_loss": value_loss.detach(),
    })


def _forward_policy_phase(
    self: FlowmatchingActionHead,
    backbone_output: BatchFeature,
    action_input:    BatchFeature,
) -> BatchFeature:
    """
    Phase 2 — RECAP §IV-B Eq. 3.

    L = ||vθ(at,t,s,∅) − v||² + α * ||vθ(at,t,s,It) − v||²

    It derived from frozen value head.
    CFG dropout (p) randomly maps It → ∅ for the conditional forward pass,
    jointly training unconditional and conditional branches (RECAP App. F).

    BUG FIXED vs original code:
        vl_attn_mask was being RESET to the pre-advantage version after
        advantage token injection. Now each forward pass uses its own
        correctly augmented mask.
    """
    self.set_frozen_modules_to_eval_mode()
    self.value_head.eval()    # value head always eval in policy phase

    backbone_output = self.process_backbone_output(backbone_output)

    vl_embs_raw  = backbone_output.backbone_features       # (B, S, D) — clean
    vl_attn_mask = backbone_output.backbone_attention_mask # (B, S)
    device       = vl_embs_raw.device
    embodiment_id = action_input.embodiment_id

    reward = torch.squeeze(action_input["reward"], dim=-1).float()

    # ── advantage labels from frozen value head ───────────────────────────
    with torch.no_grad():
        adv_labels = compute_advantage_labels_from_value_head(
            value_head=self.value_head,
            backbone_feats=vl_embs_raw,
            state=action_input.state.float(),
            reward=reward,
            percentile=getattr(self.config, "advantage_threshold_percentile", 0.30),
        )

    pos_frac = (adv_labels == AdvantageEmbedding.POS_IDX).float().mean()

    # ── shared noised trajectory (CFGRL Alg 1) ───────────────────────────
    state_features = self.state_encoder(action_input.state, embodiment_id)
    actions  = action_input.action
    noise    = torch.randn_like(actions)
    t        = torch.rand(actions.shape[0], device=device, dtype=actions.dtype)
    t_bcast  = t[:, None, None]

    noisy_trajectory = (1 - t_bcast) * noise + t_bcast * actions
    velocity         = actions - noise

    t_discretized   = (t * self.num_timestep_buckets).long()
    action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

    if self.config.add_pos_embed:
        pos_ids         = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

    future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs_raw.shape[0], -1, -1)
    sa_embs       = torch.cat((state_features, future_tokens, action_features), dim=1)
    action_mask   = action_input.action_mask

    # ── Term 1: UNCONDITIONAL  −log πθ(at|ot,ℓ) ─────────────────────────
    vl_null, mask_null = self._apply_advantage_conditioning(
        vl_embs_raw, vl_attn_mask,
        advantage_label=None,    # ∅ token
    )
    out_null  = self.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_null,
        encoder_attention_mask=mask_null,   # augmented mask, S+1 tokens
        timestep=t_discretized,
        return_all_hidden_states=False,
    )
    pred_null = self.action_decoder(out_null, embodiment_id)[:, -actions.shape[1]:]
    loss_uncond = (
        F.mse_loss(pred_null, velocity, reduction="none") * action_mask
    ).sum() / action_mask.sum()

    # ── Term 2: CONDITIONAL  −α log πθ(at|It,ot,ℓ) ──────────────────────
    # CFG dropout inside _apply_advantage_conditioning randomly maps
    # It → ∅ with prob advantage_cfg_dropout_prob, jointly training
    # both branches from one forward pass (RECAP App. F).
    vl_cond, mask_cond = self._apply_advantage_conditioning(
        vl_embs_raw, vl_attn_mask,
        advantage_label=adv_labels,   # It ∈ {NEG=1, POS=2}, dropout applied inside
    )
    out_cond  = self.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_cond,
        encoder_attention_mask=mask_cond,   # augmented mask, S+1 tokens
        timestep=t_discretized,
        return_all_hidden_states=False,
    )
    pred_cond = self.action_decoder(out_cond, embodiment_id)[:, -actions.shape[1]:]
    loss_cond = (
        F.mse_loss(pred_cond, velocity, reduction="none") * action_mask
    ).sum() / action_mask.sum()

    # ── Eq. 3 total loss ─────────────────────────────────────────────────
    alpha      = getattr(self.config, "recap_alpha", 1.0)
    total_loss = loss_uncond + alpha * loss_cond

    return BatchFeature(data={
        "loss":               total_loss,
        "action_loss_uncond": loss_uncond.detach(),
        "action_loss_cond":   loss_cond.detach(),
        "advantage_pos_frac": pos_frac.detach(),
    })


# ---------------------------------------------------------------------------
# Phase management helper — call this instead of manually toggling requires_grad
# ---------------------------------------------------------------------------

class RECAPTrainer:
    """
    Manages the two-phase RECAP training loop.

    Usage:
        trainer = RECAPTrainer(model.action_head, optimizer)

        # Phase 1: train value head
        trainer.set_phase_value_head()
        for batch in dataloader:
            loss = trainer.step(backbone_out, action_input)

        # Phase 2: train policy
        trainer.set_phase_policy()
        for batch in dataloader:
            loss = trainer.step(backbone_out, action_input)

        # Iterate: go back to Phase 1 with new data, then Phase 2 again
    """

    def __init__(self, action_head: FlowmatchingActionHead, optimizer):
        self.action_head = action_head
        self.optimizer   = optimizer
        self._phase      = None

    def set_phase_value_head(self):
        """
        Phase 1 — RECAP Alg 1 lines 1, 4, 8:
            Train Vϕ on D using Eq. 1.
            Policy (DiT, encoders, decoders) frozen.
            Value head trainable.
        """
        ah = self.action_head
        # Freeze everything
        for p in ah.parameters():
            p.requires_grad = False
        # Unfreeze value head
        for p in ah.value_head.parameters():
            p.requires_grad = True
        # Unfreeze advantage embedding (learns alongside value head)
        if ah.advantage_embedding is not None:
            for p in ah.advantage_embedding.parameters():
                p.requires_grad = True

        self._phase = "value_head"
        print("[RECAP] ── Phase 1: VALUE_HEAD ──")
        print(f"  Trainable params: {sum(p.numel() for p in ah.parameters() if p.requires_grad):,}")

    def set_phase_policy(self):
        """
        Phase 2 — RECAP Alg 1 lines 2, 5, 9:
            Train πθ on D using Eq. 3 and frozen Vϕ.
            Value head frozen (used only for labelling).
            Policy trainable (respects tune_projector / tune_diffusion_model).
        """
        ah = self.action_head
        # Restore policy trainability
        ah.set_trainable_parameters(ah.config.tune_projector, ah.config.tune_diffusion_model)
        # Freeze value head
        for p in ah.value_head.parameters():
            p.requires_grad = False
        # Advantage embedding always trainable in policy phase
        if ah.advantage_embedding is not None:
            for p in ah.advantage_embedding.parameters():
                p.requires_grad = True

        self._phase = "policy"
        print("[RECAP] ── Phase 2: POLICY ──")
        print(f"  Trainable params: {sum(p.numel() for p in ah.parameters() if p.requires_grad):,}")

    def step(
        self,
        backbone_output: BatchFeature,
        action_input:    BatchFeature,
    ) -> dict:
        self.optimizer.zero_grad()

        if self._phase == "value_head":
            out = _forward_value_head_phase(self.action_head, backbone_output, action_input)
        elif self._phase == "policy":
            out = _forward_policy_phase(self.action_head, backbone_output, action_input)
        else:
            raise RuntimeError("Call set_phase_value_head() or set_phase_policy() first.")

        out["loss"].backward()
        self.optimizer.step()

        return {k: v.item() for k, v in out.items()}


# ---------------------------------------------------------------------------
# Config addition — add recap_alpha to FlowmatchingActionHeadConfig
# ---------------------------------------------------------------------------
# Add this field to your FlowmatchingActionHeadConfig dataclass:
#
#   recap_alpha: float = field(
#       default=1.0,
#       metadata={
#           "help": (
#               "α in RECAP Eq. 3: weight of conditional term relative to "
#               "unconditional term. π*0.6 uses α=1.0 (equal weight). "
#               "Higher α → stronger push toward advantage-conditioned behavior."
#           )
#       },
#   )
#   advantage_threshold_percentile: float = field(
#       default=0.30,
#       metadata={
#           "help": (
#               "Percentile of V predictions used as ε_ℓ threshold. "
#               "π*0.6 App. F: 0.30 for pre-training, 0.40 for fine-tuning."
#           )
#       },
#   )