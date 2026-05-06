"""
check_env_preprocessing.py

Captures raw env observations and preprocessed observations (exactly as fed to the
model) for both libero and mimicgen environments, then saves them for offline comparison.

Run once per branch on the server, e.g.:
  python evaluation/check_env_preprocessing.py \
      --branch-name equi_gr00t_fa \
      --gr00t-path /home/locht1/gr00t_equi_fa \
      --libero-path /mnt/data/sftp/data/locht1/LIBERO_benchmark \
      --robosuite-assets /mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen \
      --mimicgen-path /mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen

Outputs (per branch):
  /mnt/data/sftp/data/locht1/check_env_eval/<branch-name>/
    libero/<step>/
      raw_agentview.png
      raw_wrist.png
      processed_agentview.png
      processed_wrist.png
      raw_state.npz     -> eef_pos, eef_quat, gripper_qpos
      processed_state.npz -> x, y, z, roll, pitch, yaw, gripper
      action.npy        -> 7-D action sent to env at this step
    mimicgen/<step>/
      (same structure)
"""

import builtins
import dataclasses
import os
import pathlib
import sys
from typing import Optional

import imageio
import numpy as np
import tyro

# ---------------------------------------------------------------------------
# Robosuite hardcodes /tmp/robosuite.log which is often owned by another user
# on shared servers. Redirect it to the user's home directory before any
# robosuite-touching import can run.
# ---------------------------------------------------------------------------
_rs_log = pathlib.Path.home() / ".cache" / "robosuite.log"
_rs_log.parent.mkdir(parents=True, exist_ok=True)
_orig_open = builtins.open

def _open_shim(f, *args, **kwargs):
    if str(f) == "/tmp/robosuite.log":
        f = str(_rs_log)
    return _orig_open(f, *args, **kwargs)

builtins.open = _open_shim


DUMMY_ACTION_ARM = [0.0] * 6   # zero delta for arm
FRANKA_GRIPPER_JOINT_MAX = 0.04  # meters; Franka finger joint limit


def _gripper_hold_action(obs: dict) -> float:
    """Return the gripper action value that holds the current qpos unchanged.

    Franka qpos[0] ∈ [0, 0.04] maps linearly to action ∈ [-1, +1].
    Sending this action commands the controller to stay at the current position.
    """
    q = float(obs["robot0_gripper_qpos"][0])
    return float(np.clip(2.0 * q / FRANKA_GRIPPER_JOINT_MAX - 1.0, -1.0, 1.0))
RESOLUTION = 256
WAIT_STEPS = 5
CAPTURE_STEPS = 10       # how many timesteps (after wait) to capture
LIBERO_TASK_SUITE = "libero_10"
LIBERO_TASK_ID = 0
MIMICGEN_TASK = "square"


@dataclasses.dataclass
class Args:
    branch_name: str = "equi_gr00t_fa"
    """One of: equi_gr00t_fa | equi_gr00t_dit | gr00t_baseline"""

    output_root: str = "/mnt/data/sftp/data/locht1/check_env_eval"

    gr00t_path: str = "/home/locht1/gr00t_equi_fa"
    """Path to the gr00t repo root for this branch (inserted into sys.path)."""

    libero_path: str = "/mnt/data/sftp/data/locht1/LIBERO_benchmark"
    """Path to the LIBERO package root."""

    mimicgen_path: str = "/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen"
    """Path to the mimicgen package root."""

    robosuite_assets: str = ""
    """Override robosuite asset root (leave empty for default)."""

    libero_config_path: str = ""
    """Path to LIBERO config.yaml (leave empty for default ~/.libero/config.yaml)."""

    seed: int = 7

    skip_libero: bool = False
    skip_mimicgen: bool = False


# ---------------------------------------------------------------------------
# Preprocessing helpers (mirror of Gr00tn15_inference._process_observation)
# ---------------------------------------------------------------------------

def _preprocess_obs(obs: dict, flip_mode: Optional[str]):
    """Apply the same preprocessing as Gr00tn15_inference._process_observation."""
    # Import here so sys.path is already set up when this is called
    from evaluation.gr00tn15_inference import _mimicgen_quat2axisangle, _quat2axisangle

    xyz = obs["robot0_eef_pos"]
    if flip_mode == "vertical":
        rpy = _mimicgen_quat2axisangle(obs["robot0_eef_quat"])
    else:
        rpy = _quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]

    img = obs["agentview_image"].copy()
    wrist_img = obs["robot0_eye_in_hand_image"].copy()

    if flip_mode == "horizontal":
        img = img[:, ::-1]
        wrist_img = wrist_img[:, ::-1]
    elif flip_mode == "vertical":
        img = img[::-1]
        wrist_img = wrist_img[::-1]
    elif flip_mode == "both":
        img = img[::-1, ::-1]
        wrist_img = wrist_img[::-1, ::-1]

    img = img.astype(np.uint8)
    wrist_img = wrist_img.astype(np.uint8)

    return {
        "video.image": img,                      # (H, W, 3) uint8
        "video.wrist_image": wrist_img,           # (H, W, 3) uint8
        "state.xyz": np.array(xyz, dtype=np.float32),
        "state.rpy": np.array(rpy, dtype=np.float32),
        "state.gripper": gripper.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_step(step_dir: pathlib.Path, raw_obs: dict, processed: dict, action: np.ndarray):
    step_dir.mkdir(parents=True, exist_ok=True)

    # Images
    imageio.imwrite(step_dir / "raw_agentview.png", raw_obs["agentview_image"].astype(np.uint8))
    imageio.imwrite(step_dir / "raw_wrist.png", raw_obs["robot0_eye_in_hand_image"].astype(np.uint8))
    imageio.imwrite(step_dir / "processed_agentview.png", processed["video.image"])
    imageio.imwrite(step_dir / "processed_wrist.png", processed["video.wrist_image"])

    # State arrays
    np.savez(
        step_dir / "raw_state.npz",
        eef_pos=raw_obs["robot0_eef_pos"],
        eef_quat=raw_obs["robot0_eef_quat"],
        gripper_qpos=raw_obs["robot0_gripper_qpos"],
    )
    np.savez(
        step_dir / "processed_state.npz",
        xyz=processed["state.xyz"],
        rpy=processed["state.rpy"],
        gripper=processed["state.gripper"],
    )

    # Action sent to env
    np.save(step_dir / "action.npy", action)

    print(
        f"    step {step_dir.name:<4s}  xyz={processed['state.xyz']}  "
        f"rpy={np.round(processed['state.rpy'], 4)}  "
        f"gripper={processed['state.gripper']}"
    )


# ---------------------------------------------------------------------------
# Libero capture
# ---------------------------------------------------------------------------

def capture_libero(args: Args, out_root: pathlib.Path):
    import os
    if args.libero_config_path:
        os.environ["LIBERO_CONFIG_PATH"] = args.libero_config_path

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    print(f"\n[libero] task_suite={LIBERO_TASK_SUITE}  task_id={LIBERO_TASK_ID}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[LIBERO_TASK_SUITE]()
    task = task_suite.get_task(LIBERO_TASK_ID)
    initial_states = task_suite.get_task_init_states(LIBERO_TASK_ID)
    task_description = task.language

    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
    )
    env.seed(args.seed)
    env.reset()
    obs = env.set_init_state(initial_states[0])

    print(f"  task: {task_description}")
    print(f"  flip_mode=both  (libero)")

    libero_dir = out_root / "libero"

    for t in range(WAIT_STEPS + CAPTURE_STEPS):
        action = np.array(DUMMY_ACTION, dtype=np.float32)
        obs, _, done, _ = env.step(action.tolist())

        if t >= WAIT_STEPS:
            step_idx = t - WAIT_STEPS
            processed = _preprocess_obs(obs, flip_mode="both")
            _save_step(libero_dir / f"step_{step_idx:03d}", obs, processed, action)

    env.close()
    print(f"  [libero] done → {libero_dir}")


# ---------------------------------------------------------------------------
# Mimicgen capture
# ---------------------------------------------------------------------------

def _make_mimicgen_env(robosuite_assets: str):
    import robosuite as suite
    import robosuite.models
    from robosuite.controllers import load_controller_config
    import mimicgen_envs.envs.robosuite  # noqa: F401 — registers MimicGen envs

    if robosuite_assets:
        robosuite.models.assets_root = robosuite_assets

    ctrl = load_controller_config(default_controller="OSC_POSE")
    ctrl.update({
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_limits": [0, 10],
        "position_limits": None,
        "orientation_limits": None,
        "uncouple_pos_ori": True,
        "control_delta": True,
        "interpolation": None,
        "ramp_ratio": 0.2,
    })
    env = suite.make(
        env_name="Square_D2",
        robots="Panda",
        controller_configs=ctrl,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
        control_freq=20,
        reward_shaping=False,
        ignore_done=True,
    )
    return env


def capture_mimicgen(args: Args, out_root: pathlib.Path):
    print(f"\n[mimicgen] task={MIMICGEN_TASK} (Square_D2)")

    env = _make_mimicgen_env(args.robosuite_assets)
    np.random.seed(args.seed)
    obs = env.reset()

    print(f"  flip_mode=vertical  (mimicgen)")

    mimicgen_dir = out_root / "mimicgen"

    for t in range(WAIT_STEPS + CAPTURE_STEPS):
        # Hold gripper at its current position instead of commanding close (-1)
        grip = _gripper_hold_action(obs)
        action = np.array(DUMMY_ACTION_ARM + [grip], dtype=np.float32)
        obs, _, done, _ = env.step(action.tolist())

        if t >= WAIT_STEPS:
            step_idx = t - WAIT_STEPS
            processed = _preprocess_obs(obs, flip_mode="vertical")
            _save_step(mimicgen_dir / f"step_{step_idx:03d}", obs, processed, action)

    env.close()
    print(f"  [mimicgen] done → {mimicgen_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: Args):
    # Set up sys.path for this branch
    if args.gr00t_path and args.gr00t_path not in sys.path:
        sys.path.insert(0, args.gr00t_path)
    if args.libero_path and args.libero_path not in sys.path:
        sys.path.insert(0, args.libero_path)
    if args.mimicgen_path and args.mimicgen_path not in sys.path:
        sys.path.insert(0, args.mimicgen_path)

    out_root = pathlib.Path(args.output_root) / args.branch_name
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Branch: {args.branch_name}")
    print(f"Output: {out_root}")
    print("=" * 70)

    # Write a metadata file so we know which branch/commit produced this
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        commit, branch = "unknown", "unknown"

    (out_root / "meta.txt").write_text(
        f"branch_name: {args.branch_name}\n"
        f"git_branch: {branch}\n"
        f"git_commit: {commit}\n"
        f"seed: {args.seed}\n"
        f"wait_steps: {WAIT_STEPS}\n"
        f"capture_steps: {CAPTURE_STEPS}\n"
        f"resolution: {RESOLUTION}\n"
    )

    if not args.skip_libero:
        capture_libero(args, out_root)

    if not args.skip_mimicgen:
        capture_mimicgen(args, out_root)

    print(f"\nDone. All outputs in: {out_root}")


if __name__ == "__main__":
    tyro.cli(main)
