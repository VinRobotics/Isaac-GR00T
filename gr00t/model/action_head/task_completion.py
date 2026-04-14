import torch
import torch.nn as nn


class TaskCompletionDetector(nn.Module):
    """
    Predicts task completion from a variable-length sequence of VL embeddings.

    Architecture
    ------------
    Two signals are concatenated and fed through a small MLP:

    1. last_mean  — mean of the last `last_frac` tokens (current frame state)
    2. delta      — last_mean minus the historical mean (what changed)

    The backbone (Eagle) already cross-attends across all packed frames, so
    a global CLS-style aggregation is redundant here.  The explicit temporal
    delta is the key discriminating signal: success and failure diverge from
    "doing" only in the final frames, and expressing that delta directly gives
    the MLP the right inductive bias without needing cross-attention.

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
    last_frac : float
        Fraction of the token sequence treated as the "last frame".
        Set to 1/window_size (e.g. 0.2 for a 5-frame window).
    """

    def __init__(
        self,
        seq_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        num_classes: int = 3,
        last_frac: float = 0.2,
    ):
        super().__init__()
        self.last_frac = last_frac

        self.mlp = nn.Sequential(
            nn.LayerNorm(seq_dim * 2),
            nn.Linear(seq_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
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
        T = x.shape[1]
        n_last = max(1, int(T * self.last_frac))

        last_mean = x[:, -n_last:].mean(dim=1)   # (B, D) — current frame state
        hist_mean = x[:, :-n_last].mean(dim=1)   # (B, D) — history
        delta = last_mean - hist_mean             # (B, D) — what changed

        feat = torch.cat([last_mean, delta], dim=-1)  # (B, 2*D)
        return self.mlp(feat)
