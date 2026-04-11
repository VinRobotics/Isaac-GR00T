import torch
import torch.nn as nn


class TaskCompletionDetector(nn.Module):
    """
    Predicts task completion from a variable-length sequence of VL embeddings.

    A learnable CLS token cross-attends over the full token sequence via
    nn.MultiheadAttention, then the attended representation is fed through
    a classifier MLP.

    The detector is stateless: it operates on whatever token sequence it
    receives.  When used with window-frame training (Option B), the backbone
    already packs all W frames into one sequence before calling this module,
    so no buffer is needed here.
    """

    def __init__(
        self,
        seq_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
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

        self.classifier = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
        nn.init.ones_(self.norm_in.weight)
        nn.init.zeros_(self.norm_in.bias)
        nn.init.ones_(self.norm_attn.weight)
        nn.init.zeros_(self.norm_attn.bias)
        for name, p in self.cross_attn.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        for module in self.classifier.modules():
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
        cls = self.cls_token.float().expand(x.shape[0], -1, -1)  # (B, 8, seq_dim)
        x = self.norm_in(x)
        attended, _ = self.cross_attn(query=cls, key=x, value=x)  # (B, 8, seq_dim)
        attended = self.norm_attn(attended.mean(dim=1))             # (B, seq_dim)
        return self.classifier(attended)
