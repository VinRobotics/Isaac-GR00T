# compute_equivariance_error.py
#
# Empirical equivariance error for LIBERO-trained GR00T variants.
#
# For M sampled observations o and all g in C_N, computes
#     epsilon_eq = (1 / (M|G|)) * sum_{g,o} || a_hat(g.o) - rho_a(g) * a_hat(o) ||
#
# Group action (rotation about world z-axis by 2 pi r / N):
#   - state / action xy : rotate as a 2D vector            (irrep(1))
#   - state / action z  : invariant                        (trivial)
#   - state / action axis-angle (roll, pitch, yaw):
#         (roll, pitch) rotates as a 2D vector            (irrep(1))
#         yaw stays the same                              (trivial)
#   - gripper           : invariant                        (trivial)
#   - agentview / wrist image : rotate about its centre
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
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro

sys.path.insert(0, "/home/locht1/gr00t_equi_fa")
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


# ─────────────────────────────────────────────────────────────────────────────
# Image rotation
# ─────────────────────────────────────────────────────────────────────────────

def rotate_image(img: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate uint8 HxWxC image counter-clockwise by angle_rad about its centre."""
    if angle_rad == 0.0:
        return img.copy()
    H, W, C = img.shape
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    mat = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=torch.float32
    ).unsqueeze(0)
    grid = F.affine_grid(mat, [1, C, H, W], align_corners=True)
    out = F.grid_sample(t, grid, align_corners=True, padding_mode="zeros")
    return (out.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Observation / action transformation
# ─────────────────────────────────────────────────────────────────────────────

def build_obs_dict(raw_obs: dict, task_description: str, angle_rad: float = 0.0) -> dict:
    """Build the observation dict the policy expects, optionally rotated by angle_rad.

    The 180-degree image flip (LIBERO -> training orientation) is applied first,
    then we rotate by angle_rad about the image centre.  State (x, y) and
    (roll, pitch) axis-angle components are rotated by the same angle about z.
    """
    img = _to_hwc_uint8(raw_obs["agentview_image"])
    wrist_img = _to_hwc_uint8(raw_obs["robot0_eye_in_hand_image"])

    # LIBERO -> training orientation (matches gr00tn15_inference.get_libero_action)
    img = img[::-1, ::-1].copy()
    wrist_img = wrist_img[::-1, ::-1].copy()

    if angle_rad != 0.0:
        img = rotate_image(img, angle_rad)
        wrist_img = rotate_image(wrist_img, angle_rad)

    pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).copy()
    rpy = _quat2axisangle(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32).copy())
    rpy = rpy.astype(np.float32)
    gripper = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).copy()

    if angle_rad != 0.0:
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        x_new = cos_a * pos[0] - sin_a * pos[1]
        y_new = sin_a * pos[0] + cos_a * pos[1]
        pos = np.array([x_new, y_new, pos[2]], dtype=np.float32)
        roll_new = cos_a * rpy[0] - sin_a * rpy[1]
        pitch_new = sin_a * rpy[0] + cos_a * rpy[1]
        rpy = np.array([roll_new, pitch_new, rpy[2]], dtype=np.float32)

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
    """Apply rho_a(g) to the predicted action chunk [T, 7]."""
    out = action.copy()
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    out[:, 0] = cos_a * action[:, 0] - sin_a * action[:, 1]
    out[:, 1] = sin_a * action[:, 0] + cos_a * action[:, 1]
    out[:, 3] = cos_a * action[:, 3] - sin_a * action[:, 4]
    out[:, 4] = sin_a * action[:, 3] + cos_a * action[:, 4]
    # z, yaw, gripper unchanged
    return out


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

    for task_id in range(num_tasks):
        if samples_remaining <= 0:
            break
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        init_states = task_suite.get_task_init_states(task_id)
        if len(init_states) == 0:
            continue

        n_for_task = min(per_task, samples_remaining)
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
            if samples_remaining <= 0:
                break

        try:
            env.env.close()
        except Exception:
            pass

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
    logging.info(f"  task_suite_name : {args.task_suite_name}")
    logging.info(f"  num_samples (M) : {args.num_samples}")
    logging.info(f"  n_group   (|G|) : {args.n_group}")
    logging.info(f"  num_steps_wait  : {args.num_steps_wait}")
    logging.info(f"  pretrained      : {args.pretrained_model_path}")
    logging.info("=" * 70)

    # Load policy
    policy_wrapper = Gr00tn15_inference(args.pretrained_model_path, args.infer_chunk)
    policy = policy_wrapper.policy

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

    def call_policy(obs_dict):
        torch.manual_seed(args.noise_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.noise_seed)
        return action_dict_to_array(policy.get_action(obs_dict), args.infer_chunk)

    for sample_idx in tqdm.tqdm(range(M), desc="equivariance"):
        raw_obs = observations[sample_idx]
        task_description = descriptions[sample_idx]

        obs_dict_orig = build_obs_dict(raw_obs, task_description, angle_rad=0.0)
        a_orig = call_policy(obs_dict_orig)
        action_norms.append(float(np.linalg.norm(a_orig)))

        for r, angle in enumerate(angles):
            if r == 0:
                err = 0.0  # identity: f(e.o) = f(o), rho_a(e) = I
            else:
                obs_dict_rot = build_obs_dict(raw_obs, task_description, angle_rad=angle)
                a_rot = call_policy(obs_dict_rot)
                a_expected = rotate_action_array(a_orig, angle)
                err = float(np.linalg.norm(a_rot - a_expected))
            per_g_errors[r].append(err)
            flat_errors.append(err)

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

    results = {
        "model_label": args.model_label,
        "task_suite_name": args.task_suite_name,
        "num_samples_collected": M,
        "n_group": args.n_group,
        "noise_seed": args.noise_seed,
        "num_steps_wait": args.num_steps_wait,
        "epsilon_eq_mean": mean_eq,
        "epsilon_eq_std": std_eq,
        "empirical_B_mean_action_norm": float(np.mean(action_norms)),
        "empirical_B_max_action_norm": float(np.max(action_norms)),
        "per_group_mean": [float(np.mean(per_g_errors[r])) for r in range(args.n_group)],
        "per_group_std":  [float(np.std(per_g_errors[r])) for r in range(args.n_group)],
        "angles_deg":     [math.degrees(a) for a in angles],
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
    --pretrained_model_path /path/to/checkpoints/060000/pretrained_model \
    --model_label EquiVLA \
    --task_suite_name libero_goal \
    --num_samples 500 \
    --n_group 8 \
    --num_steps_wait 10
'''
