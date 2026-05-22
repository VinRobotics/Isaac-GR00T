# compute_equivariance_error.py
#
# Empirical equivariance error for LIBERO-trained GR00T variants.
#
# For M sampled observations o and all g in C_N, computes
#     epsilon_eq = (1 / (M|G|)) * sum_{g,o} || a_hat(g.o) - rho_a(g) * a_hat(o) ||
#
# Group action (rotation about world z-axis by 2 pi r / N):
#   - state / action xy             : irrep(1) — 2D rotation
#   - state / action z              : trivial — invariant
#   - state / action axis-angle ω   : NONLINEAR — ω → axis_angle(R_z(θ) · R(ω)).
#                                      All three components (ω_x, ω_y, ω_z) may
#                                      change; treating (ω_x, ω_y) as a simple
#                                      irrep(1) pair is only valid in the
#                                      first-order ω ≈ 0 limit and is wrong for
#                                      finite world rotations.
#   - gripper                       : trivial — invariant
#   - agentview image               : rotate about its centre
#   - wrist image                   : invariant (camera mounted on the gripper,
#                                      so the wrist view co-rotates with the
#                                      world about z; the equivariant vision
#                                      pathway also skips wrist — see
#                                      EquiRelLiberoConfig.get_rotation_config)
#
# Noisy init: LIBERO simulator drops objects on reset.  We step `num_steps_wait`
# dummy actions BEFORE collecting any observation so dynamics have settled and
# we are not measuring equivariance on a transient mid-air frame.

import dataclasses
import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro

# Redirect robosuite's hard-coded /tmp/robosuite.log to a user-writable path on
# shared clusters where /tmp/robosuite.log is owned by another user.
_ROBOSUITE_LOG_DIR = os.environ.get(
    "ROBOSUITE_LOG_DIR", os.path.expanduser("~/.robosuite_logs")
)
os.makedirs(_ROBOSUITE_LOG_DIR, exist_ok=True)
_OrigFileHandler = logging.FileHandler
class _PatchedFileHandler(_OrigFileHandler):
    def __init__(self, filename, *a, **kw):
        if str(filename) == "/tmp/robosuite.log":
            filename = os.path.join(_ROBOSUITE_LOG_DIR, "robosuite.log")
        super().__init__(filename, *a, **kw)
logging.FileHandler = _PatchedFileHandler

sys.path.insert(0, "/home/locht1/gr00t_rtc")
sys.path.insert(0, "/mnt/data/sftp/data/locht1/LIBERO_benchmark")

from evaluation.gr00tn15_inference import (
    Gr00tn15_inference,
    _quat2axisangle,
    _to_hwc_uint8,
)


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    pretrained_model_path: str = ""
    """Path to checkpoint directory (e.g. .../checkpoints/060000/pretrained_model)."""

    task_suite_name: str = "libero_goal"
    """LIBERO benchmark to draw observations from."""

    num_samples: int = 500
    """M in the formula — number of observations to sample."""

    n_group: int = 8
    """|G| — group order for C_N (paper uses C_8)."""

    num_steps_wait: int = 10
    """Number of dummy-action steps after env reset to skip the noisy drop-in init."""

    seed: int = 7
    """Seed for env and observation sampling."""

    noise_seed: int = 12345
    """torch seed reset before EACH policy call so the flow-matching noise is the
    same for a(o) and a(g.o).  Otherwise stochastic noise inflates epsilon_eq."""

    libero_config_path: str = ""
    """Optional path to a LIBERO config.yaml (overrides default ~/.libero/config.yaml)."""

    save_dir: str = "/tmp/equi_eq_error"
    """Root directory for the log + json result file."""

    exp_name: str = "eq_err"
    model_label: str = "model"
    """Tag written into the result filename — set per checkpoint (e.g. 'gr00tn15')."""

    infer_chunk: int = 10
    """Length of action chunk to compare (action.x, .y, ... are sliced to this)."""

    measure_flops: bool = True
    """If True, run a single fvcore FLOPs analysis on policy.model forward.
    Falls back gracefully if fvcore is unavailable or unsupported ops are hit."""

    latency_warmup: int = 5
    """Warm-up policy calls discarded before timing (CUDA kernel compile, lazy loads)."""

    bypass_normalization: bool = True
    """Option A: measure epsilon_eq in NORMALIZED space.

    When True:
      • State rotation is applied to the normalized state vector (irrep(1) pairs on
        [n_x, n_y] and [n_roll, n_pitch]) instead of to raw values — bypasses the
        non-equivariant per-key min/max normalization on the input side.
      • Image is rotated as raw pixels, then re-fed through apply_transforms
        (image preprocessing is per-channel & per-pixel, so it commutes with rotation).
      • Action error is computed on the model's normalized action_pred directly,
        skipping unapply_transforms on the output side.

    This isolates the model's architectural equivariance (the object of cor:e2e),
    free of the I/O denormalization artefacts that dominate the unnormalized score."""


# ─────────────────────────────────────────────────────────────────────────────
# Image rotation
# ─────────────────────────────────────────────────────────────────────────────

_GRID_CACHE: dict = {}


def _get_rotation_grid(H: int, W: int, angle_rad: float) -> torch.Tensor:
    """Cached F.affine_grid for a given image size and rotation angle.

    Channels are irrelevant to affine_grid output; we use C=1.
    """
    key = (H, W, round(angle_rad, 8))
    grid = _GRID_CACHE.get(key)
    if grid is None:
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        mat = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=torch.float32
        ).unsqueeze(0)
        grid = F.affine_grid(mat, [1, 1, H, W], align_corners=True).contiguous()
        _GRID_CACHE[key] = grid
    return grid


def rotate_image(img: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate uint8 HxWxC image counter-clockwise by angle_rad about its centre."""
    if angle_rad == 0.0:
        return img.copy()
    H, W, _ = img.shape
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0)
    grid = _get_rotation_grid(H, W, angle_rad)
    out = F.grid_sample(t, grid, align_corners=True, padding_mode="zeros")
    return out.squeeze(0).permute(1, 2, 0).mul_(255.0).clamp_(0, 255).to(torch.uint8).numpy()


def rotate_images(imgs: list, angle_rad: float) -> list:
    """Batch-rotate a list of identically-shaped uint8 HxWxC images by angle_rad.

    One grid_sample call for all of them; grid is cached across calls.
    """
    if angle_rad == 0.0:
        return [img.copy() for img in imgs]
    H, W, _ = imgs[0].shape
    stacked = np.stack(imgs)  # [N, H, W, C]
    t = torch.from_numpy(stacked).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    grid = _get_rotation_grid(H, W, angle_rad).expand(len(imgs), -1, -1, -1)
    out = F.grid_sample(t, grid, align_corners=True, padding_mode="zeros")
    out = out.permute(0, 2, 3, 1).mul_(255.0).clamp_(0, 255).to(torch.uint8).numpy()
    return [np.ascontiguousarray(out[i]) for i in range(len(imgs))]


# ─────────────────────────────────────────────────────────────────────────────
# Axis-angle rotation under world rotation about z
# ─────────────────────────────────────────────────────────────────────────────
#
# CRITICAL: An axis-angle vector ω = θ·n (representing orientation R(ω))
# DOES NOT rotate linearly under world rotation R_z(θ_w).  The correct
# transformation is
#       R_new = R_z(θ_w) · R(ω_old),        ω_new = axis_angle(R_new)
# which is nonlinear in ω_old.  Treating (ω_x, ω_y) as a 2D vector that
# rotates by R_z(θ_w) (i.e. simple irrep(1)) is only a first-order
# approximation around ω ≈ 0 and is wrong for finite rotations — in
# particular it leaves ω_z (yaw) invariant, whereas the correct rule
# generally produces a non-zero shift in ω_z.
#
# This helper applies the correct world-rotation transformation to a
# batch of axis-angle vectors.

def _rotate_axisangle_world_z(rpy: np.ndarray, angle_rad: float) -> np.ndarray:
    """Apply world rotation R_z(angle_rad) to axis-angle orientations.

    Args:
        rpy: [..., 3] float array of axis-angle vectors (ω_x, ω_y, ω_z).
        angle_rad: world rotation angle about z-axis (radians).

    Returns:
        [..., 3] float array of new axis-angle vectors representing
        R_z(angle_rad) ∘ R(rpy).
    """
    from scipy.spatial.transform import Rotation as _Rsc
    flat = np.asarray(rpy, dtype=np.float64).reshape(-1, 3)
    R_old = _Rsc.from_rotvec(flat).as_matrix()          # [N, 3, 3]
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    R_z = np.array(
        [[cos_a, -sin_a, 0.0],
         [sin_a,  cos_a, 0.0],
         [0.0,    0.0,   1.0]],
        dtype=np.float64,
    )
    R_new = np.einsum("ij,njk->nik", R_z, R_old)         # R_z @ R_old, batched
    new_rpy = _Rsc.from_matrix(R_new).as_rotvec()        # [N, 3]
    return new_rpy.reshape(rpy.shape).astype(np.asarray(rpy).dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Observation / action transformation
# ─────────────────────────────────────────────────────────────────────────────

def build_obs_dict(
    raw_obs: dict,
    task_description: str,
    angle_rad: float = 0.0,
    rotate_state: bool = True,
) -> dict:
    """Build the observation dict the policy expects, optionally rotated by angle_rad.

    The 180-degree image flip (LIBERO -> training orientation) is applied first,
    then we rotate the IMAGE by angle_rad about its centre.

    If rotate_state=True (default — raw-space mode):
        Raw state (x, y) is rotated linearly; the axis-angle orientation is
        rotated via R_z(θ) · R(ω) (NOT linear in (ω_x, ω_y); see
        _rotate_axisangle_world_z).

    If rotate_state=False (Option A — normalized-space mode):
        Raw state is kept un-rotated.  The caller will rotate the normalized
        state vector after policy.apply_transforms.
    """
    img = _to_hwc_uint8(raw_obs["agentview_image"])
    wrist_img = _to_hwc_uint8(raw_obs["robot0_eye_in_hand_image"])

    # LIBERO -> training orientation (matches gr00tn15_inference.get_libero_action)
    img = np.ascontiguousarray(img[::-1, ::-1])
    wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1])

    if angle_rad != 0.0:
        # Only rotate the top/agentview camera. The wrist camera is mounted on
        # the gripper and rotates with the world, so its image is invariant
        # under world rotation about z. EquiRelLiberoConfig.get_rotation_config
        # confirms this: rotate_image_indices=[0] (top only); wrist is skipped
        # from the equivariant vision pathway.
        img = rotate_image(img, angle_rad)

    pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
    rpy = _quat2axisangle(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32)).astype(np.float32)
    gripper = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)

    if angle_rad != 0.0 and rotate_state:
        # Position is a 3D vector: (x, y) rotates linearly, z invariant.
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        x_new = cos_a * pos[0] - sin_a * pos[1]
        y_new = sin_a * pos[0] + cos_a * pos[1]
        pos = np.array([x_new, y_new, pos[2]], dtype=np.float32)
        # Orientation is axis-angle: ω_new = axis_angle(R_z(θ) · R(ω_old))
        # — NOT a linear 2D rotation on (ω_x, ω_y).
        rpy = _rotate_axisangle_world_z(rpy, angle_rad).astype(np.float32)

    return {
        "video.image": img[np.newaxis, np.newaxis, ...],
        "video.wrist_image": wrist_img[np.newaxis, np.newaxis, ...],
        "state.x": np.array([[[pos[0]]]], dtype=np.float32),
        "state.y": np.array([[[pos[1]]]], dtype=np.float32),
        "state.z": np.array([[[pos[2]]]], dtype=np.float32),
        "state.roll": np.array([[[rpy[0]]]], dtype=np.float32),
        "state.pitch": np.array([[[rpy[1]]]], dtype=np.float32),
        "state.yaw": np.array([[[rpy[2]]]], dtype=np.float32),
        "state.gripper": gripper[np.newaxis, np.newaxis, ...],
        "annotation.human.action.task_description": (str(task_description),),
    }


def action_dict_to_array(action_dict: dict, infer_chunk: int) -> np.ndarray:
    """Flatten the predicted action dict to a [T, 7] numpy array.

    Columns: x, y, z, roll, pitch, yaw, gripper.
    """
    keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    out = np.zeros((infer_chunk, 7), dtype=np.float32)
    for j, key in enumerate(keys):
        val = action_dict[f"action.{key}"]
        # Strip batch dim if present
        if hasattr(val, "shape") and len(val.shape) > 0 and val.shape[0] == 1:
            val = val[0]
        val = np.asarray(val, dtype=np.float32).reshape(-1)
        out[:, j] = val[:infer_chunk]
    return out


def rotate_action_array(action: np.ndarray, angle_rad: float) -> np.ndarray:
    """Apply rho_a(g) to the predicted action chunk [T, 7].

    Layout: [x, y, z, roll, pitch, yaw, gripper] where (roll, pitch, yaw) is
    axis-angle.  Position rotates linearly; orientation rotates via rotation
    matrix composition (axis-angle is not a linear irrep under finite world
    rotation — see _rotate_axisangle_world_z).
    """
    out = action.copy()
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # Position: irrep(1) on (x, y); z invariant.
    out[:, 0] = cos_a * action[:, 0] - sin_a * action[:, 1]
    out[:, 1] = sin_a * action[:, 0] + cos_a * action[:, 1]
    # Orientation: rotation-matrix composition (nonlinear in axis-angle).
    out[:, 3:6] = _rotate_axisangle_world_z(action[:, 3:6], angle_rad)
    # gripper unchanged
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Option A: normalized-space group action
# ─────────────────────────────────────────────────────────────────────────────
#
# After ConcatTransform + GR00TTransform, the normalized state tensor has shape
# [1, max_state_dim] with the EquiRelLiberoConfig layout:
#     state[..., 0] : state.x  (normalized)
#     state[..., 1] : state.y  (normalized)
#     state[..., 2] : state.z  (normalized)
#     state[..., 3] : state.roll  (raw axis-angle ω_x — not normalized)
#     state[..., 4] : state.pitch (raw axis-angle ω_y — not normalized)
#     state[..., 5] : state.yaw   (raw axis-angle ω_z — not normalized)
#     state[..., 6:8] : state.gripper (2 dims) ── trivial
#     state[..., 8:]  : zero padding
#
# Group action of world rotation R_z(θ) on this state:
#   • (norm_x, norm_y)         : 2D rotation (irrep(1)) — min_max normalisation
#                                 is symmetric in [-1, 1] and equivariant
#                                 enough that this is the model's expected
#                                 input transformation in normalized space.
#   • norm_z                    : invariant.
#   • (ω_x, ω_y, ω_z)           : axis-angle ω → axis_angle(R_z(θ) · R(ω)).
#                                 NONLINEAR — all 3 components may change.
#   • gripper / padding         : invariant.
#
# model.get_action(normalized_input)["action_pred"] returns the predicted
# action in the same per-key layout
#     action[..., 0,1,2,3,4,5,6] = x, y, z, roll, pitch, yaw, gripper
# where (roll, pitch, yaw) are axis-angle (see getActionOutput in
# flow_matching_action_head.py — getQuaternionFromMatrix for ee_dim=6 returns
# axis-angle).  Same nonlinear rule applies to action orientation.

# Position pairs are still simple irrep(1); orientation is handled separately.
STATE_XY_PAIR  = (0, 1)
ACTION_XY_PAIR = (0, 1)
STATE_ROT_SLICE  = slice(3, 6)   # state.roll, .pitch, .yaw  (axis-angle)
ACTION_ROT_SLICE = slice(3, 6)   # action.roll, .pitch, .yaw (axis-angle)


def _rotate_xy_pair_inplace(tensor, pair, angle_rad: float):
    """In-place: rotate (tensor[..., i], tensor[..., j]) by angle_rad (irrep(1))."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    i, j = pair
    is_torch = isinstance(tensor, torch.Tensor)
    if is_torch:
        xi = tensor[..., i].clone()
        xj = tensor[..., j].clone()
    else:
        xi = tensor[..., i].copy()
        xj = tensor[..., j].copy()
    tensor[..., i] = cos_a * xi - sin_a * xj
    tensor[..., j] = sin_a * xi + cos_a * xj


def _rotate_axisangle_slice_inplace(tensor, sl: slice, angle_rad: float):
    """In-place: apply world-rotation R_z(θ) to axis-angle slice (3 dims)."""
    is_torch = isinstance(tensor, torch.Tensor)
    if is_torch:
        rpy_np = tensor[..., sl].detach().cpu().numpy()
    else:
        rpy_np = np.asarray(tensor[..., sl])
    new_rpy = _rotate_axisangle_world_z(rpy_np, angle_rad)
    if is_torch:
        tensor[..., sl] = torch.from_numpy(new_rpy).to(
            dtype=tensor.dtype, device=tensor.device
        )
    else:
        tensor[..., sl] = new_rpy.astype(tensor.dtype)


def rotate_normalized_state(state, angle_rad: float):
    """Return a copy of the normalized state with the correct group action applied.

    Position (x, y): irrep(1).
    Orientation (axis-angle ω_x, ω_y, ω_z): rotation-matrix composition.
    """
    out = state.clone() if isinstance(state, torch.Tensor) else state.copy()
    _rotate_xy_pair_inplace(out, STATE_XY_PAIR, angle_rad)
    _rotate_axisangle_slice_inplace(out, STATE_ROT_SLICE, angle_rad)
    return out


def rotate_normalized_action(action, angle_rad: float):
    """Return a copy of the normalized action with the correct group action applied.

    Same rule as state: (x, y) irrep(1), (ω_x, ω_y, ω_z) nonlinear axis-angle
    rotation, z/gripper/padding invariant.
    """
    out = action.clone() if isinstance(action, torch.Tensor) else action.copy()
    _rotate_xy_pair_inplace(out, ACTION_XY_PAIR, angle_rad)
    _rotate_axisangle_slice_inplace(out, ACTION_ROT_SLICE, angle_rad)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inference cost: parameters + FLOPs
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> dict:
    """Total / trainable parameter counts and a breakdown by top-level submodule."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_module = {
        name: sum(p.numel() for p in child.parameters())
        for name, child in model.named_children()
    }
    return {
        "total": int(total),
        "trainable": int(trainable),
        "by_module": {k: int(v) for k, v in by_module.items()},
    }


def measure_flops(policy, sample_normalized_input) -> Optional[dict]:
    """Best-effort FLOPs count for one policy.model.get_action call.

    Tries fvcore first.  Returns None (with a logged warning) if neither backend works.
    """
    # fvcore
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
    except Exception as e:
        logging.warning(f"fvcore unavailable ({e}); skipping FLOPs.")
        return None

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self._ref_input = None

        def forward(self, *_):
            return self.m.get_action(self._ref_input)["action_pred"]

    wrap = _Wrap(policy.model).eval()
    wrap._ref_input = sample_normalized_input
    try:
        with torch.inference_mode():
            # fvcore needs at least one tensor arg — pass a 1-element dummy
            dummy = torch.zeros(1, device=policy.device)
            fca = FlopCountAnalysis(wrap, (dummy,))
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            total = int(fca.total())
        return {"backend": "fvcore", "total_flops": total}
    except Exception as e:
        logging.warning(f"fvcore FLOPs analysis failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Observation collection (with noisy-init settling)
# ─────────────────────────────────────────────────────────────────────────────

def collect_observations(args: Args, M: int):
    """Reset LIBERO envs, settle past noisy init, return M raw observations."""
    if args.libero_config_path:
        os.environ["LIBERO_CONFIG_PATH"] = args.libero_config_path

    from libero.libero import benchmark
    from evaluation.eval_libero import _get_libero_env

    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Distribute samples roughly evenly across tasks
    per_task = max(1, (M + num_tasks - 1) // num_tasks)

    observations = []
    descriptions = []
    samples_remaining = M

    pbar = tqdm.tqdm(total=M, desc="collect obs (LIBERO settle)")
    for task_id in range(num_tasks):
        if samples_remaining <= 0:
            break
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        init_states = task_suite.get_task_init_states(task_id)
        if len(init_states) == 0:
            continue

        n_for_task = min(per_task, samples_remaining)
        pbar.set_postfix(task_id=task_id, task=task_description[:40])
        for ep_idx in range(n_for_task):
            env.reset()
            obs = env.set_init_state(init_states[ep_idx % len(init_states)])
            # Settle past the noisy drop-in init
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            # Cache only the fields we need (deep copy: env mutates buffers in place)
            observations.append({
                "agentview_image": np.asarray(obs["agentview_image"]).copy(),
                "robot0_eye_in_hand_image": np.asarray(obs["robot0_eye_in_hand_image"]).copy(),
                "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"]).copy(),
                "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"]).copy(),
                "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).copy(),
            })
            descriptions.append(task_description)
            samples_remaining -= 1
            pbar.update(1)
            if samples_remaining <= 0:
                break

        try:
            env.env.close()
        except Exception:
            pass
    pbar.close()

    return observations, descriptions


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def compute_equivariance_error(args: Args) -> None:
    log_dir = pathlib.Path(args.save_dir) / "log" / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{args.task_suite_name}_{args.model_label}.log"

    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    logging.info("=" * 70)
    logging.info(f"Computing equivariance error for {args.model_label}")
    logging.info(f"  task_suite_name      : {args.task_suite_name}")
    logging.info(f"  num_samples (M)      : {args.num_samples}")
    logging.info(f"  n_group   (|G|)      : {args.n_group}")
    logging.info(f"  num_steps_wait       : {args.num_steps_wait}")
    logging.info(f"  bypass_normalization : {args.bypass_normalization}  "
                 f"({'NORMALIZED-space (Option A)' if args.bypass_normalization else 'RAW-space (end-to-end)'})")
    logging.info(f"  pretrained           : {args.pretrained_model_path}")
    logging.info("=" * 70)

    # Load policy
    policy_wrapper = Gr00tn15_inference(args.pretrained_model_path, args.infer_chunk)
    policy = policy_wrapper.policy

    # ── Inference cost: parameter counts ────────────────────────────────────
    params = count_parameters(policy.model)
    logging.info(f"Params total      : {params['total']:,}")
    logging.info(f"Params trainable  : {params['trainable']:,}")
    for name, n in params["by_module"].items():
        logging.info(f"  - {name:30s} {n:>15,}")

    # Collect observations
    logging.info("Collecting observations (noisy-init settling applied)...")
    observations, descriptions = collect_observations(args, args.num_samples)
    M = len(observations)
    logging.info(f"Collected {M} observations across {args.task_suite_name}")

    angles = [2.0 * math.pi * r / args.n_group for r in range(args.n_group)]

    # per-(g) accumulators
    per_g_errors = [[] for _ in range(args.n_group)]
    # per-(g, o) flat list — what the paper formula averages
    flat_errors = []
    # Also collect ||a(o)|| to provide an empirical estimate of B for the bound check
    action_norms = []
    # Per-call wall-clock latencies (ms) — populated after warm-up
    latencies_ms: list = []
    timing_enabled = [False]  # mutable flag toggled after warm-up

    def call_policy(obs_dict):
        """RAW-space path: full policy.get_action (apply + model + unapply)."""
        torch.manual_seed(args.noise_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.noise_seed)
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = policy.get_action(obs_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if timing_enabled[0]:
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return action_dict_to_array(out, args.infer_chunk)

    # ── Normalized-space helpers (Option A) ─────────────────────────────────
    from gr00t.model.policy import unsqueeze_dict_values, COMPUTE_DTYPE

    def build_normalized_input(raw_obs, task_description, angle_rad):
        """Build raw obs (image rotated by angle_rad, STATE NOT rotated) and apply
        policy transforms to land in normalized model-input space."""
        obs_dict = build_obs_dict(raw_obs, task_description,
                                  angle_rad=angle_rad, rotate_state=False)
        obs_copy = obs_dict.copy()
        # build_obs_dict already returns batched arrays (state is [1,1,1], video is
        # [1,1,H,W,C]); mirror policy.get_action and only unsqueeze if not batched.
        if not policy._check_state_is_batched(obs_copy):
            obs_copy = unsqueeze_dict_values(obs_copy)
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)
        return policy.apply_transforms(obs_copy)

    def call_model_normalized(normalized_input):
        """NORMALIZED-space path: call policy.model.get_action directly, skip unapply.

        Returns the normalized action_pred as a numpy array of shape
        [B, action_horizon, max_action_dim].
        """
        torch.manual_seed(args.noise_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.noise_seed)
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = policy.model.get_action(normalized_input)
        action_pred = model_pred["action_pred"].float().cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if timing_enabled[0]:
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return action_pred

    # ── Latency warm-up (kernel compile, lazy module init, etc.) ────────────
    if M > 0 and args.latency_warmup > 0:
        if args.bypass_normalization:
            warm_inp = build_normalized_input(observations[0], descriptions[0], angle_rad=0.0)
            for _ in tqdm.tqdm(range(args.latency_warmup), desc="warmup", leave=False):
                call_model_normalized(warm_inp)
        else:
            warm_obs = build_obs_dict(observations[0], descriptions[0], angle_rad=0.0)
            for _ in tqdm.tqdm(range(args.latency_warmup), desc="warmup", leave=False):
                call_policy(warm_obs)
    timing_enabled[0] = True

    # ── Equivariance main loop ──────────────────────────────────────────────
    total_calls = M * args.n_group
    pbar_desc = "equivariance (normalized)" if args.bypass_normalization else "equivariance (raw)"
    pbar = tqdm.tqdm(total=total_calls, desc=pbar_desc, dynamic_ncols=True)

    for sample_idx in range(M):
        raw_obs = observations[sample_idx]
        task_description = descriptions[sample_idx]

        if args.bypass_normalization:
            # ── Option A: rotate normalized state, compare normalized action ──
            norm_input_orig = build_normalized_input(raw_obs, task_description, angle_rad=0.0)
            a_orig_full = call_model_normalized(norm_input_orig)  # [B, T, max_action_dim]
            # Real-dim slice for the norm: x, y, z, roll, pitch, yaw, gripper = first 7 dims
            a_orig_real = a_orig_full[..., :7]
            action_norms.append(float(np.linalg.norm(a_orig_real)))
            per_g_errors[0].append(0.0)
            flat_errors.append(0.0)
            pbar.update(1)

            for r in range(1, args.n_group):
                angle = angles[r]
                # Rotate image in raw pixel space, re-apply transforms → image features rotated
                norm_input_rot = build_normalized_input(raw_obs, task_description, angle_rad=angle)
                # OVERWRITE state with rho_state(g) applied to the original normalized state
                state_orig = norm_input_orig["state"]
                norm_input_rot["state"] = rotate_normalized_state(state_orig, angle)
                # state_mask is bool and depends only on real-dim count → keep as-is from norm_input_rot

                a_rot_full = call_model_normalized(norm_input_rot)
                a_expected_full = rotate_normalized_action(a_orig_full, angle)
                # Compare on real action dims only (padding stays zero → doesn't matter, but cleaner)
                err = float(np.linalg.norm(a_rot_full[..., :7] - a_expected_full[..., :7]))
                per_g_errors[r].append(err)
                flat_errors.append(err)
                pbar.update(1)
        else:
            # ── Raw-space path (kept for comparison) ──
            obs_dict_orig = build_obs_dict(raw_obs, task_description, angle_rad=0.0)
            a_orig = call_policy(obs_dict_orig)
            action_norms.append(float(np.linalg.norm(a_orig)))
            per_g_errors[0].append(0.0)
            flat_errors.append(0.0)
            pbar.update(1)

            for r in range(1, args.n_group):
                angle = angles[r]
                obs_dict_rot = build_obs_dict(raw_obs, task_description, angle_rad=angle)
                a_rot = call_policy(obs_dict_rot)
                a_expected = rotate_action_array(a_orig, angle)
                err = float(np.linalg.norm(a_rot - a_expected))
                per_g_errors[r].append(err)
                flat_errors.append(err)
                pbar.update(1)

        # Running mean (skip the g_0 zeros so the postfix reflects real error)
        if flat_errors:
            running = np.asarray(flat_errors, dtype=np.float64)
            nz = running[running > 0]
            running_mean = float(nz.mean()) if nz.size > 0 else 0.0
            pbar.set_postfix(eps_eq=f"{running_mean:.4f}", sample=f"{sample_idx + 1}/{M}")
    pbar.close()

    flat = np.asarray(flat_errors, dtype=np.float64)
    mean_eq = float(flat.mean())
    std_eq = float(flat.std())

    logging.info("=" * 70)
    logging.info(f"epsilon_eq = {mean_eq:.6f} +/- {std_eq:.6f}   (over M|G| = {len(flat)} pairs)")
    logging.info(f"empirical B (mean ||a(o)||) = {np.mean(action_norms):.6f}  "
                 f"max = {np.max(action_norms):.6f}")
    logging.info("Per-group element breakdown:")
    for r in range(args.n_group):
        arr = np.asarray(per_g_errors[r], dtype=np.float64)
        logging.info(
            f"  g_{r}  ({math.degrees(angles[r]):6.1f} deg): "
            f"mean={arr.mean():.6f}  std={arr.std():.6f}"
        )

    # ── Inference cost: wall-clock latency ──────────────────────────────────
    if latencies_ms:
        lat = np.asarray(latencies_ms, dtype=np.float64)
        ms_per_call = float(lat.mean())
        ms_per_call_std = float(lat.std())
        ms_per_step = ms_per_call / max(1, args.infer_chunk)
        logging.info("-" * 70)
        logging.info(f"Latency per get_action() call : {ms_per_call:8.2f} ms "
                     f"(+/- {ms_per_call_std:.2f}, n={len(lat)})")
        logging.info(f"Latency per action step       : {ms_per_step:8.2f} ms "
                     f"(call / infer_chunk={args.infer_chunk})")
    else:
        ms_per_call = ms_per_call_std = ms_per_step = float("nan")
        logging.info("No latency samples recorded.")

    # ── Inference cost: FLOPs (best-effort, fvcore) ─────────────────────────
    flops_info = None
    if args.measure_flops and M > 0:
        try:
            sample_obs = build_obs_dict(observations[0], descriptions[0], angle_rad=0.0)
            from gr00t.model.policy import unsqueeze_dict_values  # local import to avoid cycle
            obs_copy = sample_obs.copy()
            if not policy._check_state_is_batched(obs_copy):
                obs_copy = unsqueeze_dict_values(obs_copy)
            for k, v in obs_copy.items():
                if not isinstance(v, np.ndarray):
                    obs_copy[k] = np.array(v)
            normalized_input = policy.apply_transforms(obs_copy)
            flops_info = measure_flops(policy, normalized_input)
        except Exception as e:
            logging.warning(f"FLOPs sample prep failed: {e}")
        if flops_info is not None:
            logging.info(f"FLOPs per get_action()  : {flops_info['total_flops']:,} "
                         f"(backend={flops_info['backend']})")
        else:
            logging.info("FLOPs : skipped / unavailable")

    results = {
        "model_label": args.model_label,
        "task_suite_name": args.task_suite_name,
        "num_samples_collected": M,
        "n_group": args.n_group,
        "noise_seed": args.noise_seed,
        "num_steps_wait": args.num_steps_wait,
        "bypass_normalization": args.bypass_normalization,
        "measurement_space": "normalized" if args.bypass_normalization else "raw",
        "epsilon_eq_mean": mean_eq,
        "epsilon_eq_std": std_eq,
        "empirical_B_mean_action_norm": float(np.mean(action_norms)),
        "empirical_B_max_action_norm": float(np.max(action_norms)),
        "per_group_mean": [float(np.mean(per_g_errors[r])) for r in range(args.n_group)],
        "per_group_std":  [float(np.std(per_g_errors[r])) for r in range(args.n_group)],
        "angles_deg":     [math.degrees(a) for a in angles],
        # inference cost
        "params": params,
        "latency_ms_per_call_mean": ms_per_call,
        "latency_ms_per_call_std":  ms_per_call_std,
        "latency_ms_per_action_step": ms_per_step,
        "latency_num_samples": len(latencies_ms),
        "infer_chunk": args.infer_chunk,
        "flops": flops_info,
    }
    out_json = log_dir / f"{args.task_suite_name}_{args.model_label}.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    logging.info(f"Wrote results to {out_json}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    tyro.cli(compute_equivariance_error)


'''
Example:

python evaluation/compute_equivariance_error.py \
    --args.pretrained_model_path /path/to/checkpoints/060000/pretrained_model \
    --args.model_label EquiVLA \
    --args.task_suite_name libero_goal \
    --args.num_samples 500 \
    --args.n_group 8 \
    --args.num_steps_wait 10

Outputs (in $save_dir/log/$exp_name/<task_suite>_<model_label>.{log,json}):
  - epsilon_eq (mean ± std) and per-group breakdown
  - param counts (total / trainable / by submodule)
  - wall-clock latency: ms per get_action() call and ms per action step
  - FLOPs per get_action() call (best-effort via fvcore; may be None)
'''
