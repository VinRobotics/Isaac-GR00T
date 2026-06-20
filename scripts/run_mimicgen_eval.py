"""
Standalone MimicGen eval from a saved checkpoint.

Usage:
    python run_mimicgen_eval.py \
        --checkpoint_dir /path/to/checkpoint-50000 \
        --task stack \
        --use_abs_action
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/locht1/gr00t_equi_dit")
sys.path.insert(0, "/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen")

import numpy as np

from gr00t.data.schema import EmbodimentTag
from gr00t.eval.mimicgen_eval import (
    TASK_TO_ENV,
    TASK_TO_INSTRUCTION,
    TASK_MAX_STEPS,
    TASK_TO_ROBOT,
    MimicgenParallelEnvs,
    run_rollout,
    _normalize_task_name,
)
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True,
                   help="Path to checkpoint dir (must contain experiment_cfg/)")
    p.add_argument("--task", required=True,
                   help="MimicGen task name, e.g. 'stack', 'stack_three', 'square'")
    p.add_argument("--num_trials", type=int, default=50)
    p.add_argument("--n_envs", type=int, default=14)
    p.add_argument("--n_action_steps", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=8)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--n_test_vis", type=int, default=4,
                   help="Number of episodes to record as MP4 (0 to disable)")
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--output_dir", default=None,
                   help="Where to save videos (defaults to <checkpoint_dir>/eval_results)")
    p.add_argument("--embodiment_tag", default="new_embodiment")
    p.add_argument("--data_config", default="equi_mimicgen",
                   help="Data config name matching the one used during training")
    p.add_argument("--use_abs_action", action="store_true")
    p.add_argument("--num_steps_wait", type=int, default=5,
                   help="Warm-up delta steps before switching to abs control")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    task_name = _normalize_task_name(args.task)
    env_name = TASK_TO_ENV[task_name]
    robot = TASK_TO_ROBOT[task_name]
    task_instruction = TASK_TO_INSTRUCTION[task_name]
    max_steps = TASK_MAX_STEPS[task_name]

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load modality config & transform from data config ─────────────────────
    data_config_cls = load_data_config(args.data_config)
    modality_config = data_config_cls.modality_config()
    modality_transform = data_config_cls.transform()
    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    # ── Load policy from checkpoint ───────────────────────────────────────────
    print(f"Loading policy from {checkpoint_dir} ...")
    policy = Gr00tPolicy(
        model_path=str(checkpoint_dir),
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
    )
    policy.model.eval()

    n_obs_steps = len(modality_config["video"].delta_indices)
    n_envs = min(args.n_envs, args.num_trials)
    state_keys = list(modality_config["state"].modality_keys)
    action_keys = list(modality_config["action"].modality_keys)

    # ── Spin up parallel envs ─────────────────────────────────────────────────
    print(f"Spawning {n_envs} envs for task '{task_name}' ({env_name}, robot={robot}) ...")
    envs = MimicgenParallelEnvs(
        env_name=env_name,
        n_envs=n_envs,
        resolution=args.resolution,
        robot=robot,
        use_abs_action=args.use_abs_action,
        num_steps_wait=args.num_steps_wait,
    )

    # ── Set up video recording ────────────────────────────────────────────────
    imageio = None
    video_dir = None
    if args.n_test_vis > 0:
        try:
            import imageio as _imageio
            imageio = _imageio
            task_slug = task_name.replace(" ", "_")
            video_dir = output_dir / "videos" / task_slug
            video_dir.mkdir(parents=True, exist_ok=True)
        except ImportError:
            print("imageio not installed; skipping video recording")

    # ── Run eval in chunks ────────────────────────────────────────────────────
    n_chunks = math.ceil(args.num_trials / n_envs)
    all_successes = []
    seed_cursor = args.seed
    episode_cursor = 0

    try:
        for chunk_idx in range(n_chunks):
            remaining = args.num_trials - len(all_successes)
            take = min(n_envs, remaining)
            seeds = [seed_cursor + i for i in range(take)]
            seed_cursor += take

            record_video_mask = [
                (episode_cursor + i) < args.n_test_vis and imageio is not None
                for i in range(take)
            ]

            print(f"Chunk {chunk_idx + 1}/{n_chunks}: seeds {seeds[0]}..{seeds[-1]}")
            chunk_successes, chunk_videos = run_rollout(
                policy=policy,
                envs=envs,
                task_instruction=task_instruction,
                max_steps=max_steps,
                n_obs_steps=n_obs_steps,
                n_action_steps=args.n_action_steps,
                seeds=seeds,
                state_keys=state_keys,
                action_keys=action_keys,
                record_video_mask=record_video_mask,
                use_abs_action=args.use_abs_action,
            )
            all_successes.extend(chunk_successes)

            if imageio is not None and video_dir is not None:
                for i, frames in enumerate(chunk_videos):
                    if frames is None or len(frames) == 0:
                        continue
                    abs_ep = episode_cursor + i
                    ep_seed = seeds[i]
                    ep_success = chunk_successes[i]
                    suffix = "success" if ep_success else "failure"
                    mp4_path = video_dir / f"ep{abs_ep:02d}_seed{ep_seed}_{suffix}.mp4"
                    try:
                        imageio.mimwrite(
                            mp4_path,
                            [np.ascontiguousarray(f) for f in frames],
                            fps=args.video_fps,
                            codec="libx264",
                        )
                        print(f"  Saved {mp4_path}")
                    except Exception as e:
                        print(f"  Failed to write {mp4_path}: {e}")

            episode_cursor += take
    finally:
        envs.close()

    succ = int(sum(all_successes))
    rate = succ / max(1, args.num_trials)
    print(f"\n{'='*60}")
    print(f"  Task        : {task_name}")
    print(f"  Checkpoint  : {checkpoint_dir}")
    print(f"  Success     : {succ}/{args.num_trials} ({rate * 100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
