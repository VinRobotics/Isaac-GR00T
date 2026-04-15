import torch
import torch.nn as nn


class TaskCompletionDetector(nn.Module):
    """
    Predicts task completion from a variable-length sequence of VL embeddings.

    Architecture
    ------------
    A learnable CLS token cross-attends over the full packed token sequence
    (all window frames) via a single MHA layer, then feeds into a small MLP.

    Parameters
    ----------
    seq_dim : int
        Backbone output embedding dimension.
    hidden_dim : int
        Hidden dim for the MLP.  Default 256 is sufficient; larger values
        tend to overfit when the backbone is frozen.
    dropout : float
        Dropout rate inside the MLP.  0.4–0.5 recommended for frozen backbone.
    num_classes : int
        Number of output classes (default 3: doing / success / failure).
    num_heads : int
        Number of attention heads for CLS pooling.
    """

    def __init__(
        self,
        seq_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
        num_classes: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, seq_dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=seq_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(seq_dim)

        # 1024 → 512 → 256 → 128 → num_classes
        self.mlp = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 8, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, seq_dim) — VL token sequence (all frames already packed)
        Returns:
            logits: (B, num_classes)  — 0=doing, 1=success, 2=failure
        """
        x = x.float()
        B = x.shape[0]

        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, D)
        cls_out, _ = self.attn(cls, x, x)             # CLS attends over full sequence
        cls_out = self.attn_norm(cls_out.squeeze(1))  # (B, D)

        return self.mlp(cls_out)
