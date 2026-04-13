"""
Pre-bake rotated initial states for LIBERO task suites.

For each task, takes the first `num_trials_per_task` benchmark initial states
and rotates each one by a randomly sampled angle from ANGLES_DEG.
The resulting MuJoCo qpos vectors are saved to disk so eval can load them
directly with env.set_init_state() — no runtime physics settling needed.

Usage:
    python generate_rotated_states.py --task_suite_name libero_spatial
    python generate_rotated_states.py --task_suite_name libero_spatial --num_trials_per_task 3
    python generate_rotated_states.py --task_suite_name libero_10 --output_dir /my/path
"""

import os
import argparse
import numpy as np
import tqdm
from libero.libero import benchmark

from utils import get_libero_env, rotate_scene_init_state


ANGLES_DEG = list(range(-40, 41, 5)) # -45..45 step 5° + symmetric flip


def generate(task_suite_name: str, output_dir: str, num_trials_per_task: int = 5) -> None:
    os.makedirs(output_dir, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(
        f"Suite: {task_suite_name}  |  tasks: {num_tasks}"
        f"  |  trials/task: {num_trials_per_task}  |  angles: {ANGLES_DEG}"
    )

    for task_id in tqdm.tqdm(range(num_tasks), desc="tasks"):
        task = task_suite.get_task(task_id)
        all_initial_states = task_suite.get_task_init_states(task_id)
        initial_states = all_initial_states[:num_trials_per_task]
        env, task_description = get_libero_env(task, resolution=256)

        rotated_states = []   # will be shape (n_trials, qpos_dim)
        meta = []             # (angle_deg, orig_state_idx) for each row

        skipped = 0
        for state_idx, state in tqdm.tqdm(enumerate(initial_states), desc="  states"):
            angle_deg = np.random.choice(ANGLES_DEG)
            angle_rad = angle_deg * np.pi / 180.0
            env.reset()
            _, any_fell = rotate_scene_init_state(env, state, angle=angle_rad)
            # if any_fell:
            #     skipped += 1
            #     continue
            # Capture the full flattened MuJoCo state after rotation + settling
            flat = env.env.sim.get_state().flatten()
            rotated_states.append(flat)
            meta.append((angle_deg, state_idx))

        rotated_states = np.array(rotated_states, dtype=np.float64)
        meta = np.array(meta, dtype=np.int32)

        # Save per-task: states + metadata
        out_prefix = os.path.join(output_dir, f"task_{task_id:03d}")
        np.save(f"{out_prefix}_states.npy", rotated_states)
        np.save(f"{out_prefix}_meta.npy", meta)

        print(
            f"  task {task_id:2d}: '{task_description[:60]}'"
            f"  → {len(rotated_states)} states saved, {skipped} skipped (objects fell)"
        )

        env.close()

    print(f"\nDone. All rotated states written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write .npy files. Defaults to ./rotated_states/<suite_name>/",
    )
    parser.add_argument(
        "--num_trials_per_task",
        type=int,
        default=5,
        help="How many benchmark initial states per task to rotate (default: 5).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join("rotated_states", args.task_suite_name)
    generate(args.task_suite_name, output_dir, num_trials_per_task=args.num_trials_per_task)


if __name__ == "__main__":
    main()
