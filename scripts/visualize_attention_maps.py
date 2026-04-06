"""
Interactive visualization of captured DiT cross-attention maps.

Loads .npz attention map files created by capture_attention_maps.py and
displays an interactive matplotlib figure with:
  - Camera image(s) for the current dataset step
  - Attention heatmap: query tokens (rows) × KV tokens (columns)
  - Row labels: state | future | action tokens
  - Sliders for: dataset step, denoising step, DiT layer, head (or mean)

Usage:
    python scripts/visualize_attention_maps.py --data_dir attention_maps/
"""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.widgets import Slider, RadioButtons, CheckButtons


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
    """Load a single step file. Returns dict of arrays + annotation string."""
    data = np.load(str(path), allow_pickle=True)
    result = {}
    for k in data.files:
        val = data[k]
        # annotation is stored as 0-d string array
        if val.ndim == 0:
            result[k] = str(val)
        else:
            result[k] = val
    return result


# ---------------------------------------------------------------------------
# Token boundary helpers
# ---------------------------------------------------------------------------

def get_query_token_labels(token_info: dict, truncate_action_at: int = 8) -> list[str]:
    """Build per-token row labels for the query axis."""
    ns = token_info["num_state_tokens"]
    nf = token_info["num_future_tokens"]
    na = token_info["num_action_tokens"]

    labels = []
    for i in range(ns):
        labels.append(f"state[{i}]")
    for i in range(nf):
        labels.append(f"fut[{i}]")
    for i in range(na):
        labels.append(f"act[{i}]")
    return labels


def get_query_section_boundaries(token_info: dict):
    """Returns (state_end, future_end, action_end) indices for heatmap vlines."""
    ns = token_info["num_state_tokens"]
    nf = token_info["num_future_tokens"]
    na = token_info["num_action_tokens"]
    return ns, ns + nf, ns + nf + na


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class AttentionVisualizer:
    def __init__(self, data_dir: Path, args):
        self.data_dir = data_dir
        self.meta = load_metadata(data_dir)
        self.step_files = list_step_files(data_dir)
        self.num_dataset_steps = len(self.step_files)
        self.token_info = self.meta["token_info"]
        self.num_layers = self.meta["num_layers"]
        self.num_heads = self.meta["num_heads"]
        self.num_denoising_steps = self.meta["denoising_steps"]
        self.video_keys = self.meta.get("video_keys", [])

        # Current state
        self.cur_step_idx = args.start_step
        self.cur_layer = 0
        self.cur_denoise_step = 0
        self.cur_head = -1        # -1 = mean over heads
        self.show_query_labels = args.show_labels
        self.vmax_mode = "per_map"  # or "global"

        # Cache loaded step data
        self._cache: dict[int, dict] = {}

        # Build figure
        self._build_figure(args)
        self._update()

    def _get_step_data(self, idx: int) -> dict:
        if idx not in self._cache:
            self._cache[idx] = load_step(self.step_files[idx])
        return self._cache[idx]

    def _get_attn_map(self, step_data: dict) -> np.ndarray:
        """
        Extract attention map for current controls.
        attention_maps shape: (T_denoise, L, H, Q, K)
        Returns: (Q, K) 2-D array
        """
        maps = step_data["attention_maps"]  # (T, L, H, Q, K)
        t = min(self.cur_denoise_step, maps.shape[0] - 1)
        l = min(self.cur_layer,        maps.shape[1] - 1)
        layer_map = maps[t, l]  # (H, Q, K)

        if self.cur_head == -1:
            return layer_map.mean(axis=0)  # (Q, K)
        else:
            h = min(self.cur_head, layer_map.shape[0] - 1)
            return layer_map[h]  # (Q, K)

    def _build_figure(self, args):
        num_cams = max(1, len(self.video_keys))

        # Layout: top row = cameras + heatmap, bottom = per-token-type bars, sliders
        fig = plt.figure(figsize=(18, 11), constrained_layout=False)
        fig.patch.set_facecolor("#1e1e1e")
        self.fig = fig

        # Main grid: rows = [cams+heatmap | per-type bars | sliders]
        outer = gridspec.GridSpec(3, 1, figure=fig,
                                  height_ratios=[5, 2, 2],
                                  hspace=0.45)

        # Top row: cameras | heatmap
        top = gridspec.GridSpecFromSubplotSpec(
            1, num_cams + 1, subplot_spec=outer[0],
            width_ratios=[1] * num_cams + [3],
            wspace=0.05,
        )

        # Camera axes
        self.cam_axes = []
        for i in range(num_cams):
            ax = fig.add_subplot(top[0, i])
            ax.set_facecolor("#2d2d2d")
            ax.axis("off")
            self.cam_axes.append(ax)

        # Heatmap axis
        self.heatmap_ax = fig.add_subplot(top[0, num_cams])
        self.heatmap_ax.set_facecolor("#2d2d2d")

        # Middle row: per-token-type attention bars (state | future | action)
        mid = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[1], wspace=0.3
        )
        self.bar_axes = {
            "state":  fig.add_subplot(mid[0, 0]),
            "future": fig.add_subplot(mid[0, 1]),
            "action": fig.add_subplot(mid[0, 2]),
        }
        for ax in self.bar_axes.values():
            ax.set_facecolor("#2d2d2d")

        # Slider area
        slider_area = outer[2]
        # Allocate sub-grid for sliders + info text
        slider_gs = gridspec.GridSpecFromSubplotSpec(
            4, 2, subplot_spec=slider_area, hspace=0.8, wspace=0.3
        )

        def make_slider_ax(row, col, label):
            ax = fig.add_subplot(slider_gs[row, col])
            ax.set_facecolor("#3a3a3a")
            return ax

        ax_step    = make_slider_ax(0, 0, "dataset step")
        ax_denoise = make_slider_ax(1, 0, "denoise step")
        ax_layer   = make_slider_ax(2, 0, "layer")
        ax_head    = make_slider_ax(3, 0, "head")

        self.slider_step = Slider(
            ax_step, "Dataset step", 0, self.num_dataset_steps - 1,
            valinit=self.cur_step_idx, valstep=1,
            color="#5588cc",
        )
        self.slider_denoise = Slider(
            ax_denoise, "Denoise step", 0, self.num_denoising_steps - 1,
            valinit=self.cur_denoise_step, valstep=1,
            color="#55cc88",
        )
        self.slider_layer = Slider(
            ax_layer, "DiT layer", 0, self.num_layers - 1,
            valinit=self.cur_layer, valstep=1,
            color="#cc8855",
        )
        self.slider_head = Slider(
            ax_head, "Head (-1=mean)", -1, self.num_heads - 1,
            valinit=self.cur_head, valstep=1,
            color="#cc55cc",
        )

        for sl in [self.slider_step, self.slider_denoise,
                   self.slider_layer, self.slider_head]:
            sl.label.set_color("white")
            sl.valtext.set_color("white")

        # Info text box (right side of sliders)
        info_gs = gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=slider_gs[:, 1]
        )
        self.info_ax = fig.add_subplot(info_gs[0])
        self.info_ax.set_facecolor("#1e1e1e")
        self.info_ax.axis("off")
        self.info_text = self.info_ax.text(
            0.05, 0.95, "", transform=self.info_ax.transAxes,
            fontsize=8, color="white", va="top", family="monospace",
            wrap=True,
        )

        # Connect sliders
        self.slider_step.on_changed(self._on_slider)
        self.slider_denoise.on_changed(self._on_slider)
        self.slider_layer.on_changed(self._on_slider)
        self.slider_head.on_changed(self._on_slider)

        # Title
        fig.suptitle(
            "GR00T DiT Cross-Attention Explorer",
            fontsize=13, color="white", y=0.99,
        )

    def _on_slider(self, _val):
        self.cur_step_idx    = int(self.slider_step.val)
        self.cur_denoise_step = int(self.slider_denoise.val)
        self.cur_layer       = int(self.slider_layer.val)
        self.cur_head        = int(self.slider_head.val)
        self._update()

    def _update(self):
        step_data = self._get_step_data(self.cur_step_idx)
        attn_map  = self._get_attn_map(step_data)   # (Q, K)

        self._draw_cameras(step_data)
        self._draw_heatmap(attn_map, step_data)
        self._draw_per_type_bars(attn_map)
        self._draw_info(step_data, attn_map)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _draw_cameras(self, step_data: dict):
        clean_keys = [vk.replace(".", "_") for vk in self.video_keys]
        for i, (ax, ck) in enumerate(zip(self.cam_axes, clean_keys)):
            ax.cla()
            ax.set_facecolor("#2d2d2d")
            ax.axis("off")
            img_key = f"image_{ck}"
            if img_key in step_data:
                img = step_data[img_key]
                ax.imshow(img)
                ax.set_title(self.video_keys[i].split(".")[-1],
                             fontsize=7, color="white", pad=2)
            else:
                ax.text(0.5, 0.5, "no image", ha="center", va="center",
                        color="gray", transform=ax.transAxes)

    # ------------------------------------------------------------------
    def _draw_heatmap(self, attn_map: np.ndarray, step_data: dict):
        ax = self.heatmap_ax
        ax.cla()
        ax.set_facecolor("#2d2d2d")

        Q, K = attn_map.shape
        vmin = 0.0
        vmax = attn_map.max() if attn_map.max() > 0 else 1.0

        im = ax.imshow(
            attn_map,
            aspect="auto",
            interpolation="nearest",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
        )

        # Query-token y-axis labels
        token_info = self.token_info
        ns = token_info["num_state_tokens"]
        nf = token_info["num_future_tokens"]
        na = token_info["num_action_tokens"]

        # Y-tick labels (query tokens)
        if self.show_query_labels and Q <= 80:
            labels = get_query_token_labels(token_info)
            ax.set_yticks(range(Q))
            ax.set_yticklabels(labels, fontsize=5, color="white")
        else:
            # Just mark boundaries with horizontal lines and labels
            ax.set_yticks([ns - 0.5, ns + nf - 0.5])
            ax.set_yticklabels([], color="white")

        # Draw horizontal dividers between token groups
        for boundary in [ns - 0.5, ns + nf - 0.5]:
            ax.axhline(boundary, color="cyan", linewidth=1.2, alpha=0.7)

        # Annotate sections on y-axis
        section_info = [
            (ns / 2,            "state",  "cyan"),
            (ns + nf / 2,       "future", "lime"),
            (ns + nf + na / 2,  "action", "orange"),
        ]
        for y, label, color in section_info:
            ax.text(-0.5, y, label, ha="right", va="center",
                    fontsize=7, color=color, transform=ax.get_yaxis_transform())

        ax.set_xlabel("VL token index (encoder)", fontsize=8, color="white")
        ax.set_ylabel("Query token (DiT input)", fontsize=8, color="white")
        ax.tick_params(colors="white", labelsize=6)
        ax.spines[:].set_color("#555")

        title_parts = [
            f"Layer {self.cur_layer}",
            f"Denoise t={self.cur_denoise_step}",
            f"Head {'mean' if self.cur_head == -1 else self.cur_head}",
        ]
        ax.set_title("  |  ".join(title_parts), fontsize=8, color="white", pad=4)

        # Colorbar
        try:
            if hasattr(self, "_cbar") and self._cbar is not None:
                self._cbar.remove()
        except Exception:
            pass
        self._cbar = self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        self._cbar.ax.tick_params(labelsize=6, colors="white")
        self._cbar.ax.yaxis.label.set_color("white")

    # ------------------------------------------------------------------
    def _draw_per_type_bars(self, attn_map: np.ndarray):
        """
        For each token group (state | future | action), show the mean attention
        over all KV tokens as a bar chart — visualises which KV positions each
        group attends to most on average.
        """
        ti = self.token_info
        ns = ti["num_state_tokens"]
        nf = ti["num_future_tokens"]
        na = ti["num_action_tokens"]

        Q, K = attn_map.shape
        groups = {
            "state":  (0,       ns,       "cyan"),
            "future": (ns,      ns + nf,  "lime"),
            "action": (ns + nf, ns+nf+na, "orange"),
        }

        kv_x = np.arange(K)
        for name, (q0, q1, color) in groups.items():
            ax = self.bar_axes[name]
            ax.cla()
            ax.set_facecolor("#2d2d2d")

            if q0 >= Q or q1 > Q:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        color="gray", transform=ax.transAxes)
                continue

            segment = attn_map[q0:q1, :]   # (group_size, K)
            mean_attn = segment.mean(axis=0)  # (K,)

            ax.bar(kv_x, mean_attn, color=color, alpha=0.8, width=1.0)
            ax.set_xlim(-0.5, K - 0.5)
            ax.set_ylim(0, mean_attn.max() * 1.1 + 1e-8)
            ax.set_title(f"{name} tokens → KV", fontsize=8, color="white", pad=2)
            ax.set_xlabel("VL token index", fontsize=6, color="white")
            ax.set_ylabel("Mean attn", fontsize=6, color="white")
            ax.tick_params(colors="white", labelsize=6)
            ax.spines[:].set_color("#555")

    # ------------------------------------------------------------------
    def _draw_info(self, step_data: dict, attn_map: np.ndarray):
        ti = self.token_info
        Q, K = attn_map.shape

        ann = step_data.get("annotation", "")
        step_file = self.step_files[self.cur_step_idx].name

        # Top-3 attended KV positions for action tokens
        ns = ti["num_state_tokens"]
        nf = ti["num_future_tokens"]
        action_attn = attn_map[ns + nf:, :].mean(axis=0)
        top3 = np.argsort(action_attn)[::-1][:3]
        top3_str = ", ".join(f"{idx}({action_attn[idx]:.3f})" for idx in top3)

        lines = [
            f"File:      {step_file}",
            f"Q tokens:  {Q}  (state={ti['num_state_tokens']} fut={ti['num_future_tokens']} act={ti['num_action_tokens']})",
            f"KV tokens: {K}  (backbone VL features)",
            f"",
            f"Action→top3 KV: {top3_str}",
            f"",
            f"Annotation:",
            f"  {ann[:120]}",
        ]
        self.info_text.set_text("\n".join(lines))

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
    parser = argparse.ArgumentParser(description="Visualize DiT cross-attention maps")
    parser.add_argument("--data_dir", type=str, default="attention_maps",
                        help="Directory containing step_*.npz files and metadata.json")
    parser.add_argument("--start_step", type=int, default=0,
                        help="Initial dataset step to display")
    parser.add_argument("--show_labels", action="store_true", default=False,
                        help="Show per-token row labels on the heatmap y-axis")
    args = parser.parse_args()
    main(args)
