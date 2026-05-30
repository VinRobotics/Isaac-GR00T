# eval_libero.py

import os 
import sys 

sys.path.insert(0, "/home/locht1/gr00t_equi_dit")
sys.path.insert(0, "/mnt/data/sftp/data/locht1/LIBERO_benchmark")

import builtins
import pathlib

_rs_log = pathlib.Path.home() / ".cache" / "robosuite.log"
_rs_log.parent.mkdir(parents=True, exist_ok=True)
_orig_open = builtins.open

def _open_shim(f, *args, **kwargs):
    if str(f) == "/tmp/robosuite.log":
        f = str(_rs_log)
    return _orig_open(f, *args, **kwargs)

builtins.open = _open_shim

import collections
import dataclasses
import logging
import math
import pathlib
import multiprocessing as mp
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import imageio
import numpy as np
import tqdm
import tyro
import torch

from evaluation.gr00tn15_inference import Gr00tn15_inference
from gr00t.eval.scene_rotation import (
    reset_env_rotated,
    check_ee_yaw_feasibility,
    discover_robot_mount_body,
    resolve_table_center,
)


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _get_libero_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def to_video_frame(arr):
    arr = np.asarray(arr)
    if any(s < 0 for s in arr.strides) or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] > 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        if arr.shape[-1] == 2:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unexpected channel count: {arr.shape}")
    if arr.dtype in (np.float32, np.float64):
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if 0.0 <= vmin and vmax <= 1.0:
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


@dataclasses.dataclass
class Args:
    pretrained_model_path: str = "/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/binh/VLA-Humanoid/outputs/train/2025-11-26/11-34-23_libero_100%_base_12layers_only/checkpoints/060000/pretrained_model"
    resize_size: int = 256
    infer_chunk: int = 10
    task_suite_name: str="libero_goal"
    save_videos_root: str = "/mnt/data/sftp/data/locht1/libero_eval_results"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10 #50  # Number of rollouts per task

    seed: int = 7  # Random Seed (for reproducibility)
    exp_name: str = "test"
    model_type: str = "gr00tn15"
    libero_config_path: str = ""
    num_workers: int = 10

    eval_rotation_angles_deg: tuple[float, ...] = (0.0,)
    eval_rotation_step_deg: Optional[float] = None
    eval_rotation_range_deg: tuple[float, float] = (0.0, 360.0)
    rotation_mode: str = "full"
    pivot_xy: Optional[tuple[float, float]] = None
    pilot_task_ids: Optional[tuple[int, ...]] = None


def _resolve_angles(args: Args) -> tuple[float, ...]:
    if args.eval_rotation_step_deg is not None:
        start, stop = args.eval_rotation_range_deg
        return tuple(float(v) for v in np.arange(start, stop, args.eval_rotation_step_deg))
    return tuple(args.eval_rotation_angles_deg)


def eval_libero(args: Args, task_suite_name: Optional[str] = None, task_ids: Optional[list] = None) -> None:
    np.random.seed(args.seed)

    if args.libero_config_path:
        os.environ["LIBERO_CONFIG_PATH"] = args.libero_config_path

    from libero.libero import benchmark

    if task_suite_name is not None:
        args.task_suite_name = task_suite_name

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    if task_ids is None:
        task_ids = list(range(task_suite.n_tasks))

    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280*2,
        "libero_goal": 600,
        "libero_10": 1000,
        "libero_90": 400,
    }
    if args.task_suite_name not in max_steps_map:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = max_steps_map[args.task_suite_name]

    log_dir = pathlib.Path(f"{args.save_videos_root}/log/eval_results/{args.exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    task_range_tag = f"tasks_{task_ids[0]}-{task_ids[-1]}" if task_ids else "all"
    log_file = log_dir / f"{args.task_suite_name}_{task_range_tag}.log"

    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    save_video_dir = pathlib.Path(f"{args.save_videos_root}/{args.exp_name}/videos/{args.task_suite_name}")
    save_video_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.model_type == "gr00tn15":
            mypolicy = Gr00tn15_inference(args.pretrained_model_path, args.infer_chunk)
        else:
            raise ValueError(f"{args.model_type} is not supported yet")
        logging.info(f"Task {args.task_suite_name} | Loaded {args.model_type} policy")
    except Exception as e:
        logging.info(f"Task {args.task_suite_name} | Failed to load policy: {e}")
        return

    angles_deg = _resolve_angles(args)

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(task_ids):
        logging.info(f"Task_id: {task_id}")

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        env.reset()
        base_xml = env.sim.model.get_xml()
        mount_body_name = discover_robot_mount_body(env.sim)
        pivot = (np.asarray(args.pivot_xy, dtype=float) if args.pivot_xy is not None
                 else resolve_table_center(env)[0])

        task_angles = list(angles_deg)
        if args.rotation_mode == "ee_yaw" and len(task_angles) > 1:
            feasible, skipped = check_ee_yaw_feasibility(
                env, initial_states[0], task_angles, pivot_xy=pivot,
            )
            if skipped:
                logging.info(f"task={task_id} [ee_yaw] skip infeasible angles: {skipped}")
            task_angles = feasible

        task_episodes, task_successes = 0, 0

        for angle_deg in task_angles:
            theta_rad = math.radians(angle_deg)
            angle_tag = f"a{angle_deg:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "_")

            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task),
                                         desc=f"task={task_id} angle={angle_deg:+.2f}deg"):
                if hasattr(mypolicy, "reset"):
                    mypolicy.reset()

                obs = reset_env_rotated(
                    env, base_xml, initial_states[episode_idx],
                    theta_rad=theta_rad,
                    mode=args.rotation_mode,
                    mount_body_name=mount_body_name,
                    pivot_xy=pivot,
                    camera_names=()
                )

                init_png = (save_video_dir
                            / f"init_seed_{args.seed}_trial_{episode_idx}_{angle_tag}_static_{task_description}.png")
                imageio.imwrite(str(init_png), to_video_frame(obs["agentview_image"][::-1, ::-1]))

                t = 0
                done = False
                replay_images, replay_images_wrist = [], []

                if task_episodes % 10 == 0:
                    logging.info(
                        f"Task_id: {task_id} | ep={episode_idx} angle={angle_deg:+.2f}deg "
                        f"| {task_description}"
                    )

                while t < max_steps + args.num_steps_wait:
                    try:
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        action_chunk = mypolicy.get_libero_action(obs, task_description)

                        for act in action_chunk:
                            obs, reward, done, info = env.step(act.tolist())
                            t += 1
                            replay_images.append(to_video_frame(obs["agentview_image"][::-1, ::-1]))
                            replay_images_wrist.append(to_video_frame(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
                            if done:
                                task_successes += 1
                                total_successes += 1
                                break
                        if done:
                            break
                    except Exception as e:
                        logging.info(f"Caught exception: {e}")
                        break

                task_episodes += 1
                total_episodes += 1
                suffix = "success" if done else "failure"
                logging.info(f"Task_id: {task_id} | ep={episode_idx} angle={angle_deg:+.2f} -> {suffix}")

                imageio.mimwrite(
                    save_video_dir / f"rollout_seed_{args.seed}_trial_{episode_idx}_{angle_tag}_wrist_{task_description}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images_wrist],
                    fps=20, codec="libx264",
                )
                imageio.mimwrite(
                    save_video_dir / f"rollout_seed_{args.seed}_trial_{episode_idx}_{angle_tag}_static_{task_description}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=20, codec="libx264",
                )

        logging.info(f"# episodes so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        logging.info(f"Task {task_id} success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Final total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def eval_libero_all(args: Args) -> None:
    print("=" * 80)
    print("LIBERO Simulation Evaluation")
    print("=" * 80)

    if args.libero_config_path:
        os.environ["LIBERO_CONFIG_PATH"] = args.libero_config_path

    if args.pilot_task_ids is not None:
        all_task_ids = list(args.pilot_task_ids)
    else:
        from libero.libero import benchmark
        task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
        all_task_ids = list(range(task_suite.n_tasks))

    num_workers = max(1, min(args.num_workers, len(all_task_ids)))
    if num_workers == 1:
        eval_libero(args, task_ids=all_task_ids)
        return

    chunk_size = math.ceil(len(all_task_ids) / num_workers)
    task_splits = [all_task_ids[i:i + chunk_size] for i in range(0, len(all_task_ids), chunk_size)]

    ctx = mp.get_context("spawn")
    results = {}

    with ProcessPoolExecutor(max_workers=len(task_splits), mp_context=ctx) as pool:
        futures = {
            pool.submit(eval_libero, args, args.task_suite_name, task_ids): task_ids
            for task_ids in task_splits
        }
        for fut in as_completed(futures):
            task_ids = futures[fut]
            label = f"tasks_{task_ids[0]}-{task_ids[-1]}"
            try:
                results[label] = fut.result()
                print(f"[DONE] {label}")
            except Exception as e:
                print(f"[ERROR] {label} failed: {e}")

    print("All done. Results:", results)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    tyro.cli(eval_libero_all)

'''
python evaluation/eval_libero.py \
    --args.exp_name=test \
    --args.pretrained_model_path=/mnt/data/sftp/data/vla_intern/workspace/binh/2026/prunner/VLA_layer_prunner/output/2026-02-25/19-16-33_bz16_maxstep60000_libero_pi0_base/checkpoints/060000/pretrained_model

'''