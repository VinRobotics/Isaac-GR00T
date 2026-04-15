# eval_rot_libero.py
# Extends eval_libero with rotated/diverse initial states.

import os
import sys

sys.path.insert(0, "/home/locht1/gr00t_equi_dit")
sys.path.insert(0, "/mnt/data/sftp/data/locht1/LIBERO_benchmark")

import dataclasses
import logging
import multiprocessing as mp
import pathlib
from typing import Optional

import imageio
import numpy as np
import tqdm
import tyro
from concurrent.futures import ProcessPoolExecutor, as_completed
from libero.libero import benchmark

from evaluation.eval_libero import (
    Args,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
    to_video_frame,
)
from examples.Libero_rot.eval.utils import load_rotated_states
from evaluation.gr00tn15_inference import Gr00tn15_inference


@dataclasses.dataclass
class RotArgs(Args):
    rotated_states_dir: str = ""  # path to pre-baked rotated states root dir


def eval_rot_libero(args: RotArgs, task_suite_name: Optional[str] = None, task_ids: Optional[list] = None) -> None:
    np.random.seed(args.seed)

    if task_suite_name is not None:
        args.task_suite_name = task_suite_name

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    if task_ids is None:
        task_ids = list(range(num_tasks_in_suite))

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    log_dir = pathlib.Path(f"{args.save_videos_root}/log/eval_results/{args.exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    task_range_tag = f"tasks_{task_ids[0]}-{task_ids[-1]}"
    log_file = log_dir / f"{args.task_suite_name}_rot_{task_range_tag}.log"

    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    save_video_dir = pathlib.Path(f"{args.save_videos_root}/{args.exp_name}/videos/{args.task_suite_name}_rot")
    save_video_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.model_type == "gr00tn15":
            mypolicy = Gr00tn15_inference(args.pretrained_model_path, args.infer_chunk)
            logging.info(f"Task {args.task_suite_name} | Successfully loaded {args.model_type} policy")
        else:
            raise ValueError(f"{args.model_type} is not supported yet")
    except Exception as e:
        logging.info(f"Task {args.task_suite_name} | Failed to load policy: {e}")
        return

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        logging.info(f"Task_id: {task_id}")

        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Load pre-baked rotated states instead of default initial states
        rotated_states_dir = args.rotated_states_dir or None
        rotated_states, _ = load_rotated_states(
            args.task_suite_name,
            task_id,
            rotated_states_dir=f"{args.rotated_states_dir}/{args.task_suite_name}" if rotated_states_dir else None,
        )

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            env.reset()
            # Cycle through rotated states if fewer than num_trials_per_task
            state = rotated_states[episode_idx % len(rotated_states)]
            obs = env.set_init_state(state)

            t = 0
            replay_images = []
            replay_images_wrist = []

            if task_episodes % 10 == 0:
                logging.info(f"Task_id: {task_id} | Starting episode {task_episodes+1}... | {task_description}")

            done = False
            while t < max_steps + args.num_steps_wait:
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

            task_episodes += 1
            total_episodes += 1
            suffix = "success" if done else "failure"

            imageio.mimwrite(
                save_video_dir / f"rollout_seed_{args.seed}_trial_{episode_idx}_wrist_{task_description}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images_wrist],
                fps=10,
                codec="libx264",
            )
            imageio.mimwrite(
                save_video_dir / f"rollout_seed_{args.seed}_trial_{episode_idx}_static_{task_description}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
                codec="libx264",
            )

        logging.info(f"Success: {done}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def eval_rot_libero_all(args: RotArgs) -> None:
    print("=" * 80)
    print("LIBERO Rotated Evaluation")
    print("=" * 80)

    task_splits = [
        list(range(0, 5)),   # pool 0: tasks [0..4]
        list(range(5, 10)),  # pool 1: tasks [5..9]
    ]

    ctx = mp.get_context("spawn")
    results = dict()

    with ProcessPoolExecutor(max_workers=len(task_splits), mp_context=ctx) as pool:
        futures = {
            pool.submit(eval_rot_libero, args, args.task_suite_name, task_ids): task_ids
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
        import torch
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    tyro.cli(eval_rot_libero_all)

'''
python evaluation/eval_rot_libero.py \
    --args.exp_name=test_rot \
    --args.task_suite_name=libero_10 \
    --args.pretrained_model_path=<checkpoint_path> \
    --args.rotated_states_dir=examples/Libero_rot/eval/rotated_states
'''
