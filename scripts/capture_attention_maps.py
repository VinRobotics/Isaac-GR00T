"""
Capture cross-attention maps from DiT model during action prediction.

This script loads a dataset, runs model inference for each step, and captures
attention weights from all DiT transformer blocks across all denoising steps.
Saves results as .npz files per dataset step.

Usage:
    python scripts/capture_attention_maps.py \
        --model_path nvidia/GR00T-N1.5-3B \
        --dataset_path demo_data/robot_sim.PickNPlace \
        --data_config fourier_gr1_arms_only \
        --embodiment_tag gr1 \
        --output_dir attention_maps/ \
        --num_steps 20 \
        --denoising_steps 4
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.transforms import GR00TTransform


# ---------------------------------------------------------------------------
# Custom attention processor that captures attention weights
# ---------------------------------------------------------------------------

class AttentionCaptureProcessor:
    """
    Drop-in replacement for diffusers AttnProcessor that explicitly computes
    and saves softmax attention weights.

    Accumulates one entry per forward() call (i.e., per denoising step).
    Call .clear() between dataset steps.
    """

    def __init__(self):
        # List of np.ndarray, each (num_heads, query_tokens, kv_tokens)
        # One entry per denoising step
        self.attention_history: list[np.ndarray] = []
        self.is_cross_attention_history: list[bool] = []

    def clear(self):
        self.attention_history.clear()
        self.is_cross_attention_history.clear()

    def __call__(
        self,
        attn,                           # diffusers Attention module
        hidden_states,                  # (B, Q, D) query
        encoder_hidden_states=None,     # (B, K, D) key/value, None = self-attn
        attention_mask=None,
        temb=None,
        **kwargs,
    ):
        is_cross = encoder_hidden_states is not None
        self.is_cross_attention_history.append(is_cross)

        batch_size = hidden_states.shape[0]
        kv_states = encoder_hidden_states if is_cross else hidden_states

        # Handle optional cross-norm
        if is_cross and hasattr(attn, "norm_cross") and attn.norm_cross:
            kv_states = attn.norm_encoder_hidden_states(kv_states)

        query = attn.to_q(hidden_states)
        key   = attn.to_k(kv_states)
        value = attn.to_v(kv_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        # Reshape → (B, H, seq, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size,   -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Explicit attention (float32 for stability)
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale

        if attention_mask is not None:
            kv_seq = key.shape[2]
            mask = attn.prepare_attention_mask(attention_mask, kv_seq, batch_size)
            mask = mask.view(batch_size, attn.heads, -1, mask.shape[-1])
            attn_scores = attn_scores + mask.float()

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, Q, K)

        # Save first-batch-item (B=1 during inference) averaged or per-head
        captured = attn_weights[0].detach().cpu().float().numpy()  # (H, Q, K)
        self.attention_history.append(captured)

        # Compute output
        out = torch.matmul(attn_weights.to(value.dtype), value)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = out.to(query.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def install_capture_processors(dit_model):
    """
    Replace each transformer block's attn1 processor with a capturing processor.
    Returns a list of processors (one per block) for later retrieval.
    """
    processors = []
    for block in dit_model.transformer_blocks:
        proc = AttentionCaptureProcessor()
        block.attn1.processor = proc
        processors.append(proc)
    print(f"Installed {len(processors)} attention capture processors on DiT blocks.")
    return processors


def collect_attention_maps(processors):
    """
    Collect captured attention maps from all processors.

    Cross-attention layers: K = num VL tokens
    Self-attention layers:  K = num query tokens (Q)
    These sizes differ, so we can NOT stack across layers into one array.

    Returns:
        per_layer: list[np.ndarray] of length num_layers,
                   each array has shape (num_denoising_steps, num_heads, Q, K_layer)
        is_cross:  list[list[bool]] shape (num_layers, num_denoising_steps)
    """
    num_steps = len(processors[0].attention_history)
    for i, p in enumerate(processors):
        assert len(p.attention_history) == num_steps, (
            f"Layer {i} has {len(p.attention_history)} steps, expected {num_steps}"
        )

    # One array per layer: (T_denoise, H, Q, K)
    per_layer = [np.stack(p.attention_history, axis=0) for p in processors]
    is_cross = [p.is_cross_attention_history for p in processors]
    return per_layer, is_cross


def get_token_counts(policy: Gr00tPolicy):
    """
    Infer token counts from the action head config.
    Returns dict with counts for state, future, and action tokens.
    """
    action_head = policy.model.action_head
    cfg = action_head.config
    # state_features: (B, delta_indices, state_dim) but delta=[0] → 1 token
    num_state_tokens = len(policy._modality_config.get("state", {}).delta_indices or [0])
    num_future_tokens = cfg.num_target_vision_tokens
    num_action_tokens = cfg.action_horizon
    return {
        "num_state_tokens": num_state_tokens,
        "num_future_tokens": num_future_tokens,
        "num_action_tokens": num_action_tokens,
        "total_query_tokens": num_state_tokens + num_future_tokens + num_action_tokens,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_groot_transform(transform) -> "GR00TTransform | None":
    """Recursively find the GR00TTransform inside a (composed) transform pipeline."""
    if isinstance(transform, GR00TTransform):
        return transform
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            found = _find_groot_transform(t)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data config ---
    data_config = load_data_config(args.data_config)
    modality_transform = data_config.transform()

    # Training-time config: used to initialise the policy (sets action_horizon etc.)
    train_modality_config = data_config.modality_config()

    # Inference-time config: effort modality uses only history indices (no future labels).
    # Data configs with torque/effort history expose inference_modality_config() for this.
    if hasattr(data_config, "inference_modality_config"):
        infer_modality_config = data_config.inference_modality_config()  # type: ignore[attr-defined]
        print("Using inference_modality_config() for dataset loading.")
    else:
        infer_modality_config = dict(train_modality_config)  # mutable copy
        print("No inference_modality_config() found, using modality_config().")

    # --- Policy / model ---
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=train_modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=device,
    )
    policy.model.eval()

    # --- Auto-trim effort delta_indices to match GR00TTransform.effort_history_len ---
    # When inference_modality_config() is absent, the effort modality may still include
    # future labels (history + future frames). Find effort_history_len from the transform
    # and keep only the history portion so the transform assertion doesn't fail.
    groot_t = _find_groot_transform(policy._modality_transform)
    if groot_t is not None and groot_t.effort_history_len is not None:
        for mod_key in list(infer_modality_config.keys()):
            cfg = infer_modality_config[mod_key]
            if len(cfg.delta_indices) > groot_t.effort_history_len and mod_key == "effort":
                trimmed = cfg.delta_indices[:groot_t.effort_history_len]
                infer_modality_config[mod_key] = ModalityConfig(
                    delta_indices=trimmed,
                    modality_keys=cfg.modality_keys,
                )
                print(
                    f"Auto-trimmed '{mod_key}' delta_indices: "
                    f"{len(cfg.delta_indices)} → {groot_t.effort_history_len} frames "
                    f"(effort_history_len={groot_t.effort_history_len})"
                )

    # --- Dataset: use inference modality config + no transforms (policy applies them) ---
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=infer_modality_config,
        video_backend=args.video_backend,
        transforms=None,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Trajectory lengths: {dataset.trajectory_lengths}")

    # --- Install attention capture processors ---
    dit = policy.model.action_head.model  # DiT instance
    processors = install_capture_processors(dit)

    # --- Token metadata ---
    token_info = get_token_counts(policy)
    print("Token layout in query (sa_embs):", token_info)

    # --- Determine video keys from inference modality config ---
    video_keys = infer_modality_config["video"].modality_keys \
        if "video" in infer_modality_config else []
    state_keys = infer_modality_config["state"].modality_keys \
        if "state" in infer_modality_config else []
    language_keys = infer_modality_config["language"].modality_keys \
        if "language" in infer_modality_config else []

    num_steps = min(args.num_steps, len(dataset))

    # Detect which layers are cross-attention (need one forward pass to know for sure,
    # but we can infer from the config: interleave_self_attention alternates odd layers).
    interleave = getattr(dit.config, "interleave_self_attention", False)
    layer_is_cross = [
        not (interleave and (i % 2 == 1))
        for i in range(len(dit.transformer_blocks))
    ]

    all_metadata = {
        "num_layers": len(dit.transformer_blocks),
        "num_heads": dit.config.num_attention_heads,
        "denoising_steps": args.denoising_steps,
        "token_info": token_info,
        "layer_is_cross": layer_is_cross,
        "video_keys": video_keys,
        "state_keys": state_keys,
        "language_keys": language_keys,
        "dataset_path": str(args.dataset_path),
        "embodiment_tag": args.embodiment_tag,
        "data_config": args.data_config,
        "num_dataset_steps": num_steps,
    }

    print(f"\nCapturing attention maps for {num_steps} steps → {output_dir}")

    for step_idx in range(args.start_step, args.start_step + num_steps):
        print(f"  Step {step_idx} / {args.start_step + num_steps - 1}", end="\r")

        # --- Get raw observation (transforms=None, policy handles them) ---
        raw_obs = dataset[step_idx]

        # --- Clear all processors before this step ---
        for p in processors:
            p.clear()

        # --- Run inference (this triggers the denoising loop) ---
        with torch.inference_mode():
            action_pred = policy.get_action(raw_obs)

        # --- Collect attention maps ---
        # per_layer: list[np.ndarray], each (T_denoise, H, Q, K_layer)
        # K differs between cross-attn and self-attn layers, so we save per-layer.
        per_layer_maps, is_cross = collect_attention_maps(processors)

        # --- Extract images (first frame of each camera) ---
        images = {}
        for vk in video_keys:
            if vk in raw_obs:
                frame = raw_obs[vk]
                # frame may be (T, H, W, C) or (H, W, C)
                if frame.ndim == 4:
                    frame = frame[0]  # first temporal frame
                images[vk] = frame.astype(np.uint8)

        # --- Extract state ---
        state_arrays = {}
        for sk in state_keys:
            if sk in raw_obs:
                arr = raw_obs[sk]
                if arr.ndim > 1:
                    arr = arr[0]  # first temporal frame
                state_arrays[sk] = arr.astype(np.float32)

        # --- Extract language ---
        annotation = ""
        for lk in language_keys:
            if lk in raw_obs:
                val = raw_obs[lk]
                if isinstance(val, np.ndarray):
                    annotation = str(val.flat[0])
                else:
                    annotation = str(val)
                break

        # --- Flatten action_pred to array ---
        action_array = {}
        for k, v in action_pred.items():
            if isinstance(v, np.ndarray):
                action_array[k] = v.astype(np.float32)
            elif isinstance(v, torch.Tensor):
                action_array[k] = v.cpu().numpy().astype(np.float32)

        # --- Save ---
        save_path = output_dir / f"step_{step_idx:05d}.npz"
        save_dict = {
            "annotation": np.array(annotation),
        }
        # Save per-layer attention maps: attn_layer_0, attn_layer_1, ...
        # Each array: (T_denoise, H, Q, K_layer) — K differs for self- vs cross-attn layers.
        for layer_idx, layer_map in enumerate(per_layer_maps):
            is_cross_layer = is_cross[layer_idx][0] if is_cross[layer_idx] else True
            save_dict[f"attn_layer_{layer_idx:02d}"] = layer_map.astype(np.float16)
            save_dict[f"attn_layer_{layer_idx:02d}_is_cross"] = np.array(is_cross_layer)
        # Add images
        for vk, img in images.items():
            clean_key = vk.replace(".", "_")
            save_dict[f"image_{clean_key}"] = img
        # Add states
        for sk, arr in state_arrays.items():
            clean_key = sk.replace(".", "_")
            save_dict[f"state_{clean_key}"] = arr
        # Add action predictions
        for k, arr in action_array.items():
            clean_key = k.replace(".", "_")
            save_dict[f"action_{clean_key}"] = arr

        np.savez_compressed(str(save_path), **save_dict)

    print(f"\nDone. Saved {num_steps} steps to {output_dir}")

    # Save global metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture DiT cross-attention maps")
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.5-3B",
                        help="HuggingFace hub ID or local path to model checkpoint")
    parser.add_argument("--dataset_path", type=str, default="demo_data/robot_sim.PickNPlace",
                        help="Path to the LeRobot-format dataset")
    parser.add_argument("--data_config", type=str, default="fourier_gr1_arms_only",
                        help="Data config name (see gr00t/experiment/data_config.py)")
    parser.add_argument("--embodiment_tag", type=str, default="gr1",
                        choices=list(EMBODIMENT_TAG_MAPPING.keys()))
    parser.add_argument("--output_dir", type=str, default="attention_maps",
                        help="Directory to save attention map .npz files")
    parser.add_argument("--num_steps", type=int, default=20,
                        help="Number of dataset steps to process")
    parser.add_argument("--start_step", type=int, default=0,
                        help="First dataset step index to process")
    parser.add_argument("--denoising_steps", type=int, default=4,
                        help="Number of flow-matching denoising steps")
    parser.add_argument("--video_backend", type=str, default="torchcodec",
                        choices=["torchcodec", "decord", "torchvision_av"])
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (auto-detect if not specified)")
    args = parser.parse_args()
    main(args)
