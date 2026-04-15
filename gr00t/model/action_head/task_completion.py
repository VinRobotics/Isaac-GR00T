import torch
import torch.nn as nn


class TaskCompletionDetector(nn.Module):
    """
    Predicts task completion from a variable-length sequence of VL embeddings.

    Architecture
    ------------
    Two signals are concatenated and fed through a deep MLP:

    1. last_mean  — mean over the last-frame tokens of every camera
    2. delta      — last_mean minus the historical mean (what changed)

    Tokens are packed in camera-major order:
        [cam0_f0...cam0_fN, cam1_f0...cam1_fN, ...]
    so the last-frame of cam c starts at:
        c * num_frames * tpf + (num_frames-1) * tpf
    where tpf = T // (num_cameras * num_frames).

    Parameters
    ----------
    seq_dim : int
        Backbone output embedding dimension.
    hidden_dim : int
        First hidden dim of the MLP (halved each layer: 1024→512→256→128).
    dropout : float
        Dropout rate inside the MLP.
    num_classes : int
        Number of output classes (default 3: doing / success / failure).
    num_frames : int
        Number of temporal frames in the window (len(delta_indices)).
    num_cameras : int
        Number of camera streams (len(video_keys)).
    """

    def __init__(
        self,
        seq_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
        num_classes: int = 3,
        num_frames: int = 5,
        num_cameras: int = 2,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_cameras = num_cameras

        # 1024 → 512 → 256 → 128 → num_classes  (input is seq_dim*2)
        self.mlp = nn.Sequential(
            nn.LayerNorm(seq_dim * 2),
            nn.Linear(seq_dim * 2, hidden_dim),
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
            x: (B, T, seq_dim) — VL token sequence packed camera-major
        Returns:
            logits: (B, num_classes)  — 0=doing, 1=success, 2=failure
        """
        x = x.float()
        T = x.shape[1]
        tpf = T // (self.num_cameras * self.num_frames)  # tokens per frame per camera

        last_parts = []
        hist_parts = []
        for c in range(self.num_cameras):
            cam_start = c * self.num_frames * tpf
            lf_start = cam_start + (self.num_frames - 1) * tpf
            last_parts.append(x[:, lf_start : lf_start + tpf])  # (B, tpf, D) — last frame
            hist_parts.append(x[:, cam_start : lf_start])        # (B, (W-1)*tpf, D) — history

        last_mean = torch.cat(last_parts, dim=1).mean(dim=1)  # (B, D) — last frame, all cams
        hist_mean = torch.cat(hist_parts, dim=1).mean(dim=1)  # (B, D) — history, all cams
        delta = last_mean - hist_mean                          # (B, D) — what changed

        feat = torch.cat([last_mean, delta], dim=-1)           # (B, 2*D)
        return self.mlp(feat)
