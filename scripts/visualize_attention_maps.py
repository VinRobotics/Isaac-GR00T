"""
Interactive visualization of captured DiT cross-attention maps.

Layout:
  Left column  : stacked camera images + annotation text
  Right column : large attention heatmap (top 3/4) + per-token bar charts (bottom 1/4)
  Bottom strip : 4 compact sliders (dataset step / denoise step / layer / head)

Usage:
    python scripts/visualize_attention_maps.py --data_dir attention_maps/
    python scripts/visualize_attention_maps.py --data_dir attention_maps/ --show_labels
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.widgets import Slider


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metadata(data_dir: Path) -> dict:
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")
    with open(meta_path) as f:
        return json.load(f)


def list_step_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("step_*.npz"))
    if not files:
        raise FileNotFoundError(f"No step_*.npz files found in {data_dir}")
    return files


def load_step(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    result = {}
    for k in data.files:
        val = data[k]
        result[k] = str(val) if val.ndim == 0 else val
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_labels(token_info: dict) -> list[str]:
    ns, nf, na = (token_info["num_state_tokens"],
                  token_info["num_future_tokens"],
                  token_info["num_action_tokens"])
    return (
        [f"S{i}" for i in range(ns)] +
        [f"F{i}" for i in range(nf)] +
        [f"A{i}" for i in range(na)]
    )


def _section_boundaries(token_info: dict):
    ns = token_info["num_state_tokens"]
    nf = token_info["num_future_tokens"]
    na = token_info["num_action_tokens"]
    return ns, ns + nf, ns + nf + na   # state_end, future_end, action_end


def _style_ax(ax, fc="#1a1a2e"):
    ax.set_facecolor(fc)
    for spine in ax.spines.values():
        spine.set_color("#404060")
    ax.tick_params(colors="#aaaacc", labelsize=7)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.title.set_color("#ddddff")


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class AttentionVisualizer:

    BG       = "#0d0d1a"   # figure background
    PANEL_BG = "#1a1a2e"   # axes background
    ACCENT   = "#404060"   # spines / grid

    def __init__(self, data_dir: Path, args):
        self.data_dir  = data_dir
        self.meta      = load_metadata(data_dir)
        self.step_files = list_step_files(data_dir)
        self.num_dataset_steps  = len(self.step_files)
        self.token_info         = self.meta["token_info"]
        self.num_layers         = self.meta["num_layers"]
        self.num_heads          = self.meta["num_heads"]
        self.num_denoising_steps = self.meta["denoising_steps"]
        self.video_keys  = self.meta.get("video_keys", [])
        self.layer_is_cross = self.meta.get("layer_is_cross", [True] * self.num_layers)

        # Controls
        self.cur_step_idx     = min(args.start_step, self.num_dataset_steps - 1)
        self.cur_layer        = 0
        self.cur_denoise_step = 0
        self.cur_head         = -1   # -1 = mean
        self.show_labels      = args.show_labels
        self._figsize_override = args.figsize  # None or [W, H]
        self._cache: dict[int, dict] = {}
        self._cbar = None

        self._build_figure()
        self._update()

    # ------------------------------------------------------------------
    def _get_step_data(self, idx: int) -> dict:
        if idx not in self._cache:
            self._cache[idx] = load_step(self.step_files[idx])
        return self._cache[idx]

    def _get_attn_map(self, step_data: dict) -> np.ndarray:
        """Return (Q, K) float32 attention map for current controls."""
        layer_key = f"attn_layer_{self.cur_layer:02d}"
        if layer_key in step_data:
            layer_maps = step_data[layer_key].astype(np.float32)  # (T, H, Q, K)
            t = min(self.cur_denoise_step, layer_maps.shape[0] - 1)
            layer_map = layer_maps[t]  # (H, Q, K)
        else:
            maps = step_data.get("attention_maps")
            if maps is None:
                return np.zeros((1, 1), dtype=np.float32)
            t = min(self.cur_denoise_step, maps.shape[0] - 1)
            l = min(self.cur_layer, maps.shape[1] - 1)
            layer_map = maps[t, l].astype(np.float32)

        if self.cur_head == -1:
            return layer_map.mean(axis=0)
        return layer_map[min(self.cur_head, layer_map.shape[0] - 1)]

    # ------------------------------------------------------------------
    def _build_figure(self):
        num_cams = max(1, len(self.video_keys))

        # ── Figure ───────────────────────────────────────────────────
        # Determine figure size
        if hasattr(self, "_figsize_override") and self._figsize_override:
            fw, fh = self._figsize_override
        else:
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                dpi = 96
                fw = min(root.winfo_screenwidth()  / dpi * 0.90, 22)
                fh = min(root.winfo_screenheight() / dpi * 0.88, 12)
                root.destroy()
            except Exception:
                fw, fh = 18, 10
        fig = plt.figure(figsize=(fw, fh))
        fig.patch.set_facecolor(self.BG)
        self.fig = fig

        # ── Outer grid: [content row | slider row] ───────────────────
        outer = gridspec.GridSpec(
            2, 1, figure=fig,
            height_ratios=[11, 1],
            hspace=0.08,
            left=0.01, right=0.99, top=0.96, bottom=0.04,
        )

        # ── Content row: [left panel | right panel] ──────────────────
        content = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0],
            width_ratios=[1, 3.5],
            wspace=0.06,
        )

        # ── Left panel: cameras stacked + info ───────────────────────
        left = gridspec.GridSpecFromSubplotSpec(
            num_cams + 1, 1, subplot_spec=content[0],
            height_ratios=[1] * num_cams + [0.55],
            hspace=0.08,
        )
        self.cam_axes = []
        for i in range(num_cams):
            ax = fig.add_subplot(left[i, 0])
            ax.set_facecolor(self.PANEL_BG)
            ax.axis("off")
            self.cam_axes.append(ax)

        self.info_ax = fig.add_subplot(left[num_cams, 0])
        self.info_ax.set_facecolor(self.BG)
        self.info_ax.axis("off")
        self.info_text = self.info_ax.text(
            0.02, 0.98, "",
            transform=self.info_ax.transAxes,
            fontsize=7.5, color="#ccccee", va="top", family="monospace",
        )

        # ── Right panel: large heatmap + bar charts ───────────────────
        right = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=content[1],
            height_ratios=[3, 1],
            hspace=0.22,
        )

        self.heatmap_ax = fig.add_subplot(right[0, 0])
        _style_ax(self.heatmap_ax, self.PANEL_BG)

        bars_gs = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=right[1, 0], wspace=0.28,
        )
        self.bar_axes = {
            "state":  fig.add_subplot(bars_gs[0, 0]),
            "future": fig.add_subplot(bars_gs[0, 1]),
            "action": fig.add_subplot(bars_gs[0, 2]),
        }
        for ax in self.bar_axes.values():
            _style_ax(ax, self.PANEL_BG)

        # ── Slider row ────────────────────────────────────────────────
        slider_gs = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[1], wspace=0.06,
        )
        slider_cfg = [
            ("Dataset step",  0, self.num_dataset_steps - 1,   self.cur_step_idx,     "#4488dd"),
            ("Denoise step",  0, self.num_denoising_steps - 1, self.cur_denoise_step, "#44cc88"),
            ("DiT layer",     0, self.num_layers - 1,          self.cur_layer,        "#dd8844"),
            ("Head (−1=avg)", -1, self.num_heads - 1,          self.cur_head,         "#cc44cc"),
        ]
        self._sliders = []
        for col, (label, vmin, vmax, vinit, color) in enumerate(slider_cfg):
            ax = fig.add_subplot(slider_gs[0, col])
            ax.set_facecolor("#1a1a2e")
            sl = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=1, color=color)
            sl.label.set_color("#ccccee")
            sl.valtext.set_color("#ccccee")
            sl.label.set_fontsize(8)
            self._sliders.append(sl)

        (self.slider_step,
         self.slider_denoise,
         self.slider_layer,
         self.slider_head) = self._sliders

        for sl in self._sliders:
            sl.on_changed(self._on_slider)

        fig.suptitle(
            "GR00T DiT Attention Explorer  —  cross-attention: action/state/future → VL tokens",
            fontsize=11, color="#aaaadd", y=0.993,
        )

    # ------------------------------------------------------------------
    def _on_slider(self, _):
        self.cur_step_idx     = int(self.slider_step.val)
        self.cur_denoise_step = int(self.slider_denoise.val)
        self.cur_layer        = int(self.slider_layer.val)
        self.cur_head         = int(self.slider_head.val)
        self._update()

    def _update(self):
        step_data = self._get_step_data(self.cur_step_idx)
        attn_map  = self._get_attn_map(step_data)
        self._draw_cameras(step_data)
        self._draw_heatmap(attn_map)
        self._draw_bars(attn_map)
        self._draw_info(step_data, attn_map)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _draw_cameras(self, step_data: dict):
        clean_keys = [vk.replace(".", "_") for vk in self.video_keys]
        for i, ax in enumerate(self.cam_axes):
            ax.cla()
            ax.set_facecolor(self.PANEL_BG)
            ax.axis("off")
            if i < len(clean_keys):
                img_key = f"image_{clean_keys[i]}"
                if img_key in step_data:
                    ax.imshow(step_data[img_key])
                    label = self.video_keys[i].split(".")[-1]
                    ax.set_title(label, fontsize=8, color="#aaaadd", pad=3)
                    continue
            ax.text(0.5, 0.5, "no image", ha="center", va="center",
                    color="#555577", transform=ax.transAxes, fontsize=9)

    # ------------------------------------------------------------------
    def _draw_heatmap(self, attn_map: np.ndarray):
        ax = self.heatmap_ax
        ax.cla()
        _style_ax(ax, self.PANEL_BG)

        Q, K = attn_map.shape

        # ── image ──────────────────────────────────────────────────────
        vmax = float(attn_map.max()) if attn_map.max() > 0 else 1.0
        im = ax.imshow(
            attn_map,
            aspect="auto",
            interpolation="nearest",
            cmap="inferno",        # inferno gives better contrast than "hot"
            vmin=0.0,
            vmax=vmax,
        )

        # ── token-group dividers (horizontal) ──────────────────────────
        ti = self.token_info
        ns, nf_end, _ = _section_boundaries(ti)
        nf = ti["num_future_tokens"]
        na = ti["num_action_tokens"]

        COLORS = {"state": "#00e5ff", "future": "#69ff47", "action": "#ff9800"}
        for bnd in [ns - 0.5, nf_end - 0.5]:
            ax.axhline(bnd, color="#ffffff", linewidth=1.0, alpha=0.5, linestyle="--")

        # ── y-axis labels ───────────────────────────────────────────────
        if self.show_labels and Q <= 100:
            labels = _query_labels(ti)
            ax.set_yticks(range(Q))
            ax.set_yticklabels(labels, fontsize=5.5, color="#ccccee")
        else:
            ax.set_yticks([])
            # Section annotations as y-axis text using ax.text in data coords
            sections = [
                (ns / 2,           "STATE",  COLORS["state"]),
                (ns + nf / 2,      "FUTURE", COLORS["future"]),
                (nf_end + na / 2,  "ACTION", COLORS["action"]),
            ]
            for y_mid, name, col in sections:
                ax.text(
                    -0.01, y_mid, name,
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center",
                    fontsize=8, fontweight="bold", color=col,
                    clip_on=False,
                )

        # ── title & labels ──────────────────────────────────────────────
        is_cross = (self.cur_layer < len(self.layer_is_cross)
                    and self.layer_is_cross[self.cur_layer])
        attn_type = "cross-attn → VL tokens" if is_cross else "self-attn → query tokens"
        head_str  = "avg over heads" if self.cur_head == -1 else f"head {self.cur_head}"
        ax.set_title(
            f"Layer {self.cur_layer}  [{attn_type}]  |  "
            f"Denoise t={self.cur_denoise_step}  |  {head_str}",
            fontsize=10, color="#ddddff", pad=6,
        )
        ax.set_xlabel(
            "VL token index  (encoder output)" if is_cross else "Query token index",
            fontsize=9, color="#aaaacc",
        )
        ax.set_ylabel("Query tokens  (DiT input)", fontsize=9, color="#aaaacc")

        # ── colorbar ────────────────────────────────────────────────────
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
        self._cbar = self.fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        self._cbar.ax.tick_params(labelsize=7, colors="#aaaacc")
        self._cbar.ax.set_ylabel("Attention weight", fontsize=7, color="#aaaacc")

    # ------------------------------------------------------------------
    def _draw_bars(self, attn_map: np.ndarray):
        """
        Bar chart: for each query-token group, show mean attention across KV tokens.
        Helps see which encoder (VL) positions are most attended by each group.
        """
        ti = self.token_info
        ns = ti["num_state_tokens"]
        nf = ti["num_future_tokens"]
        na = ti["num_action_tokens"]
        Q, K = attn_map.shape
        kv_x = np.arange(K)

        COLORS = {"state": "#00e5ff", "future": "#69ff47", "action": "#ff9800"}
        groups = [
            ("state",  0,       ns,       COLORS["state"]),
            ("future", ns,      ns + nf,  COLORS["future"]),
            ("action", ns + nf, ns+nf+na, COLORS["action"]),
        ]

        for name, q0, q1, color in groups:
            ax = self.bar_axes[name]
            ax.cla()
            _style_ax(ax, self.PANEL_BG)

            if q0 >= Q or q1 > Q or q1 <= q0:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        color="#555577", transform=ax.transAxes)
                continue

            mean_attn = attn_map[q0:q1, :].mean(axis=0)  # (K,)
            # highlight top-5 positions
            top5 = np.argsort(mean_attn)[::-1][:5]
            bar_colors = [color if i not in top5 else "#ffffff" for i in range(K)]

            ax.bar(kv_x, mean_attn, color=bar_colors, alpha=0.85, width=1.0)
            ax.set_xlim(-0.5, K - 0.5)
            ax.set_ylim(0, mean_attn.max() * 1.15 + 1e-9)
            ax.set_title(
                f"{name.upper()} → encoder tokens",
                fontsize=8, color=color, pad=3,
            )
            ax.set_xlabel("VL token idx", fontsize=7, color="#aaaacc")
            ax.set_ylabel("mean attn", fontsize=7, color="#aaaacc")

            # annotate top position
            top1 = top5[0]
            ax.annotate(
                f"#{top1}\n{mean_attn[top1]:.3f}",
                xy=(top1, mean_attn[top1]),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=6.5, color="#ffffff",
            )

    # ------------------------------------------------------------------
    def _draw_info(self, step_data: dict, attn_map: np.ndarray):
        ti = self.token_info
        Q, K = attn_map.shape
        ns = ti["num_state_tokens"]
        nf = ti["num_future_tokens"]

        ann = step_data.get("annotation", "")
        step_file = self.step_files[self.cur_step_idx].name

        is_cross = (self.cur_layer < len(self.layer_is_cross)
                    and self.layer_is_cross[self.cur_layer])

        # top-3 action attention
        action_slice = attn_map[ns + nf:, :]
        if action_slice.shape[0] > 0 and K > 0:
            action_attn = action_slice.mean(axis=0)
            top3 = np.argsort(action_attn)[::-1][:3]
            top3_str = "  ".join(f"#{i}={action_attn[i]:.3f}" for i in top3)
        else:
            top3_str = "—"

        ann_wrapped = "\n  ".join(textwrap.wrap(ann[:200], width=34))

        lines = [
            f"Step : {step_file}",
            f"Attn : {'cross→VL' if is_cross else 'self→Q'}",
            f"Q    : {Q}  (S={ti['num_state_tokens']} F={ti['num_future_tokens']} A={ti['num_action_tokens']})",
            f"K    : {K}",
            f"",
            f"act→top3 KV:",
            f"  {top3_str}",
            f"",
            f"annotation:",
            f"  {ann_wrapped}",
        ]
        self.info_text.set_text("\n".join(lines))

    # ------------------------------------------------------------------
    def show(self):
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    viz = AttentionVisualizer(data_dir, args)
    viz.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize DiT cross-attention maps (interactive)"
    )
    parser.add_argument("--data_dir", type=str, default="attention_maps",
                        help="Directory with step_*.npz + metadata.json")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--show_labels", action="store_true", default=False,
                        help="Show per-token y-axis labels (good when Q<=60)")
    parser.add_argument("--figsize", type=float, nargs=2, default=None,
                        metavar=("W", "H"),
                        help="Figure size in inches, e.g. --figsize 18 10")
    args = parser.parse_args()
    main(args)
