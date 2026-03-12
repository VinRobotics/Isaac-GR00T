import torch
import torch.nn as nn

class TaskCompletionDetector(nn.Module):
    """
    Predicts task completion from a variable-length sequence of VL embeddings.

    A learnable CLS token (the "query") cross-attends over the full token
    sequence via nn.MultiheadAttention, then the attended CLS representation
    is fed through a classifier MLP.

    This naturally handles any sequence length and lets the model learn which
    tokens are informative for task completion (e.g. gripper-view tokens).
    """

    def __init__(self, seq_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Ensure seq_dim is divisible by num_heads
        while seq_dim % num_heads != 0 and num_heads > 1:
            num_heads //= 2

        self.norm_in = nn.LayerNorm(seq_dim)

        # Learnable CLS token — dedicated "task-completion query"
        self.cls_token = nn.Parameter(torch.zeros(1, 1, seq_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Cross-attention: CLS attends over all sequence tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=seq_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(seq_dim)

        # Classifier MLP with dropout
        self.classifier = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize ALL parameters in this module to safe values.
        Called both from __init__ and from GR00T_N1_5.from_pretrained (since
        HuggingFace replaces every parameter with uninitialized memory for keys
        absent from the base checkpoint, discarding the __init__ values).
        """
        # CLS token
        nn.init.normal_(self.cls_token, std=0.02)
        # LayerNorms: standard init
        nn.init.ones_(self.norm_in.weight)
        nn.init.zeros_(self.norm_in.bias)
        nn.init.ones_(self.norm_attn.weight)
        nn.init.zeros_(self.norm_attn.bias)
        # Cross-attention projection weights
        for name, p in self.cross_attn.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        # Classifier MLP
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, seq_dim) — variable-length VL token sequence
        Returns:
            logits: (B, 1)
        """
        # Disable bfloat16 autocast and upcast to float32: long VL sequences
        # (1000+ tokens) cause Q@K^T to overflow bfloat16 softmax → NaN

        cls = self.cls_token.float().expand(x.shape[0], -1, -1)  # (B, 1, seq_dim)
        x = self.norm_in(x)
        attended, _ = self.cross_attn(query=cls, key=x, value=x)  # (B, 1, seq_dim)
        attended = self.norm_attn(attended.squeeze(1))              # (B, seq_dim)
        logits = self.classifier(attended)
        return logits