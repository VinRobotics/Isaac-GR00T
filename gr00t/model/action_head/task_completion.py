import torch
import torch.nn as nn


class TaskCompletionDetector(nn.Module):
    """
    Predicts task completion from a variable-length sequence of VL embeddings.

    Architecture
    ------------
    Two complementary paths are combined before classification:

    1. Global path — CLS cross-attends over ALL tokens:
       captures full scene context across all window frames.

    2. Temporal-contrast path — mean(last_frac tokens) minus mean(rest):
       captures WHAT CHANGED in the most recent portion of the window.
       For fine-grained tasks like "part in holes vs. missed holes", success
       and failure diverge only in the last 1-2 frames; this delta amplifies
       that discriminating signal instead of drowning it in the full sequence.

    The two representations are projected to the same hidden_dim and summed,
    then fed through the classifier MLP.

    Parameters
    ----------
    last_frac : float
        Fraction of the token sequence treated as the "last frame" for the
        contrast path.  Default 1/6 ≈ the last of 6 packed frames.
        For a 6-frame window of ~300 tokens/frame each camera, this is ~300-600
        tokens depending on camera count.  Set to 1/(window_size) in general.
    """

    def __init__(
        self,
        seq_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 3,
        last_frac: float = 1 / 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.last_frac = last_frac

        while seq_dim % num_heads != 0 and num_heads > 1:
            num_heads //= 2

        self.norm_in = nn.LayerNorm(seq_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 8, seq_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=seq_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(seq_dim)

        # Temporal-contrast path: project the last-vs-history delta to hidden_dim
        self.contrast_proj = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm_contrast = nn.LayerNorm(seq_dim)

        # Global path: project attended CLS to hidden_dim
        self.global_proj = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final classifier on the fused hidden_dim representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize ALL parameters to safe values.
        Called both from __init__ and from GR00T_N1_5.from_pretrained (since
        HuggingFace replaces absent keys with uninitialized memory).
        """
        nn.init.normal_(self.cls_token, std=0.02)
        for norm in (self.norm_in, self.norm_attn, self.norm_contrast):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for name, p in self.cross_attn.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        for module in (*self.contrast_proj.modules(),
                       *self.global_proj.modules(),
                       *self.classifier.modules()):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, seq_dim) — VL token sequence (all frames already packed)
        Returns:
            logits: (B, num_classes)  — 0=doing, 1=success, 2=failure
        """
        # Upcast to float32: long sequences (1000+ tokens) overflow bfloat16 softmax
        x = x.float()
        x = self.norm_in(x)

        # ── Global path ──────────────────────────────────────────────────────
        cls = self.cls_token.float().expand(x.shape[0], -1, -1)  # (B, 8, D)
        attended, _ = self.cross_attn(query=cls, key=x, value=x)  # (B, 8, D)
        global_repr = self.norm_attn(attended.mean(dim=1))         # (B, D)
        global_feat = self.global_proj(global_repr)                # (B, hidden_dim)

        # ── Temporal-contrast path ───────────────────────────────────────────
        # Split token sequence: last `last_frac` tokens are the most recent
        # frame(s); the rest is the historical context.
        T = x.shape[1]
        n_last = max(1, int(T * self.last_frac))
        last_tokens = x[:, -n_last:, :]          # (B, n_last, D)  — current frame
        hist_tokens = x[:, :-n_last, :]          # (B, T-n_last, D) — history

        last_mean = last_tokens.mean(dim=1)       # (B, D)
        hist_mean = hist_tokens.mean(dim=1)       # (B, D)

        # The delta captures the placement event / final state change
        delta = self.norm_contrast(last_mean - hist_mean)  # (B, D)
        contrast_feat = self.contrast_proj(delta)          # (B, hidden_dim)

        # ── Fuse and classify ────────────────────────────────────────────────
        fused = global_feat + contrast_feat               # (B, hidden_dim)
        return self.classifier(fused)
