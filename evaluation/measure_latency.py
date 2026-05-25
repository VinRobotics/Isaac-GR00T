# measure_latency.py
#
# Measure policy inference latency (ms per action step) on LIBERO observations.
#
# Collects `num_samples` observations from the LIBERO env, runs `warmup` calls
# to compile CUDA kernels / lazy modules, then times `num_samples` inference
# calls and reports mean / median / p95 / std ms per step.
#
# Usage:
#   python evaluation/measure_latency.py \
#       --pretrained_model_path /path/to/checkpoint/pretrained_model \
#       --num_samples 500

import dataclasses
import logging
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import tqdm
import tyro

# Redirect robosuite's hard-coded /tmp/robosuite.log to a user-writable path.
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


@dataclasses.dataclass
class Args:
    pretrained_model_path: str = ""
    """Path to checkpoint directory (e.g. .../checkpoints/060000/pretrained_model)."""

    task_suite_name: str = "libero_goal"
    """LIBERO benchmark to draw observations from."""

    num_samples: int = 500
    """Number of policy calls to time after warm-up."""

    warmup: int = 10
    """Number of inference calls to discard before timing starts."""

    num_steps_wait: int = 10
    """Dummy-action steps after env reset so physics settle before collecting obs."""

    seed: int = 7
    """RNG seed for env and obs sampling."""

    infer_chunk: int = 10
    """Action chunk length (passed to Gr00tn15_inference)."""

    save_dir: str = "/mnt/data/sftp/data/locht1/latency"
    """Directory to write the JSON result file."""

    model_label: str = "model"
    """Tag written into the result filename."""


# ─────────────────────────────────────────────────────────────────────────────
# Obs collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_observations(args: Args):
    if args.libero_config_path if hasattr(args, "libero_config_path") else False:
        os.environ["LIBERO_CONFIG_PATH"] = args.libero_config_path

    from libero.libero import benchmark
    from evaluation.eval_libero import _get_libero_env

    np.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    M = args.num_samples
    per_task = max(1, (M + num_tasks - 1) // num_tasks)

    observations, descriptions = [], []
    remaining = M

    pbar = tqdm.tqdm(total=M, desc="collect obs")
    for task_id in range(num_tasks):
        if remaining <= 0:
            break
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        init_states = task_suite.get_task_init_states(task_id)
        if not len(init_states):
            continue

        n_for_task = min(per_task, remaining)
        pbar.set_postfix(task_id=task_id)
        for ep_idx in range(n_for_task):
            env.reset()
            obs = env.set_init_state(init_states[ep_idx % len(init_states)])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            observations.append({
                "agentview_image": np.asarray(obs["agentview_image"]).copy(),
                "robot0_eye_in_hand_image": np.asarray(obs["robot0_eye_in_hand_image"]).copy(),
                "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"]).copy(),
                "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"]).copy(),
                "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).copy(),
            })
            descriptions.append(task_description)
            remaining -= 1
            pbar.update(1)
            if remaining <= 0:
                break

        try:
            env.env.close()
        except Exception:
            pass

    pbar.close()
    return observations, descriptions


# ─────────────────────────────────────────────────────────────────────────────
# Build obs dict in the format expected by policy.get_action
# ─────────────────────────────────────────────────────────────────────────────

def build_obs_dict(raw_obs: dict, task_description: str) -> dict:
    xyz = raw_obs["robot0_eef_pos"]
    rpy = _quat2axisangle(raw_obs["robot0_eef_quat"])
    gripper = raw_obs["robot0_gripper_qpos"]

    img = _to_hwc_uint8(raw_obs["agentview_image"])
    wrist_img = _to_hwc_uint8(raw_obs["robot0_eye_in_hand_image"])
    # LIBERO images from suite.make() are upside-down — flip both axes.
    img = img[::-1, ::-1]
    wrist_img = wrist_img[::-1, ::-1]

    return {
        "video.image": img[np.newaxis, np.newaxis, ...],
        "video.wrist_image": wrist_img[np.newaxis, np.newaxis, ...],
        "state.x": np.array([[[xyz[0]]]], dtype=np.float32),
        "state.y": np.array([[[xyz[1]]]], dtype=np.float32),
        "state.z": np.array([[[xyz[2]]]], dtype=np.float32),
        "state.roll": np.array([[[rpy[0]]]], dtype=np.float32),
        "state.pitch": np.array([[[rpy[1]]]], dtype=np.float32),
        "state.yaw": np.array([[[rpy[2]]]], dtype=np.float32),
        "state.gripper": gripper.astype(np.float32)[np.newaxis, np.newaxis, ...],
        "annotation.human.action.task_description": (str(task_description),),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def measure_latency(args: Args) -> None:
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    logging.info("=" * 60)
    logging.info(f"Latency benchmark  —  {args.model_label}")
    logging.info(f"  task_suite_name : {args.task_suite_name}")
    logging.info(f"  num_samples     : {args.num_samples}")
    logging.info(f"  warmup          : {args.warmup}")
    logging.info(f"  checkpoint      : {args.pretrained_model_path}")
    logging.info("=" * 60)

    # Load policy
    policy_wrapper = Gr00tn15_inference(args.pretrained_model_path, args.infer_chunk)
    policy = policy_wrapper.policy

    # Collect observations
    logging.info("Collecting observations …")
    observations, descriptions = collect_observations(args)
    M = len(observations)
    logging.info(f"Collected {M} observations.")

    # Pre-build all obs dicts (exclude I/O from timing)
    obs_dicts = [
        build_obs_dict(observations[i], descriptions[i]) for i in range(M)
    ]

    def timed_call(obs_dict: dict) -> float:
        """Run one policy.get_action call and return wall-clock ms."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy.get_action(obs_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    # Warm-up
    logging.info(f"Running {args.warmup} warm-up calls …")
    for i in tqdm.tqdm(range(args.warmup), desc="warmup", leave=False):
        timed_call(obs_dicts[i % M])

    # Timed calls
    latencies_ms = []
    logging.info(f"Timing {M} inference calls …")
    for i in tqdm.tqdm(range(M), desc="latency"):
        latencies_ms.append(timed_call(obs_dicts[i]))

    arr = np.array(latencies_ms)
    results = {
        "model_label": args.model_label,
        "task_suite_name": args.task_suite_name,
        "num_samples": M,
        "warmup": args.warmup,
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "latency_ms_per_action_step": float(np.mean(arr)),
        "ms_per_step": float(np.mean(arr)),
    }

    logging.info("=" * 60)
    logging.info(f"Results ({M} calls, {args.warmup} warm-up discarded):")
    logging.info(f"  mean   : {results['mean_ms']:.2f} ms")
    logging.info(f"  median : {results['median_ms']:.2f} ms")
    logging.info(f"  std    : {results['std_ms']:.2f} ms")
    logging.info(f"  p95    : {results['p95_ms']:.2f} ms")
    logging.info(f"  p99    : {results['p99_ms']:.2f} ms")
    logging.info(f"  min    : {results['min_ms']:.2f} ms")
    logging.info(f"  max    : {results['max_ms']:.2f} ms")
    logging.info("=" * 60)
    logging.info(f"  latency_ms_per_action_step = {results['latency_ms_per_action_step']:.2f} ms")
    logging.info(f"  ms_per_step                = {results['ms_per_step']:.2f} ms")
    logging.info("=" * 60)

    import json
    out_path = save_dir / f"latency_{args.task_suite_name}_{args.model_label}.json"
    out_path.write_text(json.dumps(results, indent=2))
    logging.info(f"Saved → {out_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    measure_latency(args)
