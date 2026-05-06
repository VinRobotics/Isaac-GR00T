"""
compare_preprocessing.py

Run on MacBook after syncing /mnt/data/sftp/data/locht1/check_env_eval/ locally.

Usage:
  python evaluation/compare_preprocessing.py \
      --data-root /path/to/local/check_env_eval \
      --branches equi_gr00t_fa equi_gr00t_dit gr00t_baseline
"""

import dataclasses
import pathlib
from typing import Optional

import imageio
import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    data_root: str = "/tmp/check_env_eval"
    branches: list[str] = dataclasses.field(
        default_factory=lambda: ["equi_gr00t_fa", "equi_gr00t_dit", "gr00t_baseline"]
    )
    env: str = "both"
    """Which env to compare: libero | mimicgen | both"""

    step: int = 0
    """Which captured step to compare (0-indexed after wait period)."""


def _load_step(branch_dir: pathlib.Path, env_name: str, step: int):
    step_dir = branch_dir / env_name / f"step_{step:03d}"
    if not step_dir.exists():
        return None, None, None, None, None, None

    raw_state = np.load(step_dir / "raw_state.npz")
    proc_state = np.load(step_dir / "processed_state.npz")
    raw_img = imageio.imread(step_dir / "raw_agentview.png")
    proc_img = imageio.imread(step_dir / "processed_agentview.png")
    raw_wrist = imageio.imread(step_dir / "raw_wrist.png")
    proc_wrist = imageio.imread(step_dir / "processed_wrist.png")
    return raw_state, proc_state, raw_img, proc_img, raw_wrist, proc_wrist


def _compare_state(branches, data_root, env_name, step):
    print(f"\n{'─'*60}")
    print(f"  {env_name.upper()} — step {step}")
    print(f"{'─'*60}")
    results = {}
    for b in branches:
        raw, proc, *_ = _load_step(pathlib.Path(data_root) / b, env_name, step)
        if raw is None:
            print(f"  {b}: NOT FOUND")
            continue
        results[b] = (raw, proc)
        print(f"\n  [{b}]")
        print(f"    raw eef_pos:   {raw['eef_pos']}")
        print(f"    raw eef_quat:  {raw['eef_quat']}")
        print(f"    raw gripper:   {raw['gripper_qpos']}")
        print(f"    proc xyz:      {proc['xyz']}")
        print(f"    proc rpy:      {proc['rpy']}")
        print(f"    proc gripper:  {proc['gripper']}")

    # Cross-branch diffs
    branch_list = list(results.keys())
    if len(branch_list) >= 2:
        print(f"\n  Diffs:")
        for i, b1 in enumerate(branch_list):
            for b2 in branch_list[i+1:]:
                r1, p1 = results[b1]
                r2, p2 = results[b2]
                xyz_diff = np.max(np.abs(p1["xyz"] - p2["xyz"]))
                rpy_diff = np.max(np.abs(p1["rpy"] - p2["rpy"]))
                gripper_diff = np.max(np.abs(p1["gripper"] - p2["gripper"]))
                print(f"    {b1} vs {b2}:  max|Δxyz|={xyz_diff:.6f}  max|Δrpy|={rpy_diff:.6f}  max|Δgripper|={gripper_diff:.6f}")


def _compare_images(branches, data_root, env_name, step, save_dir: pathlib.Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    for b in branches:
        raw_st, proc_st, raw_img, proc_img, raw_wrist, proc_wrist = _load_step(
            pathlib.Path(data_root) / b, env_name, step
        )
        if raw_img is None:
            continue
        out = save_dir / f"{env_name}_{b}_step{step:03d}.png"
        # Side-by-side: raw_agentview | processed_agentview | raw_wrist | processed_wrist
        panel = np.concatenate([raw_img, proc_img, raw_wrist, proc_wrist], axis=1)
        imageio.imwrite(out, panel)
        print(f"  Saved image panel → {out}")

    # Also save a cross-branch comparison (processed agentview only)
    processed_imgs = []
    labels = []
    for b in branches:
        raw_st, proc_st, raw_img, proc_img, *_ = _load_step(
            pathlib.Path(data_root) / b, env_name, step
        )
        if proc_img is not None:
            processed_imgs.append(proc_img)
            labels.append(b)
    if len(processed_imgs) > 1:
        panel = np.concatenate(processed_imgs, axis=1)
        out = save_dir / f"{env_name}_all_branches_processed_step{step:03d}.png"
        imageio.imwrite(out, panel)
        print(f"  Cross-branch panel ({' | '.join(labels)}) → {out}")


def main(args: Args):
    envs = ["libero", "mimicgen"] if args.env == "both" else [args.env]
    save_dir = pathlib.Path(args.data_root) / "_comparison_output"

    print("=" * 60)
    print("Preprocessing Comparison")
    print(f"  branches: {args.branches}")
    print(f"  envs:     {envs}")
    print(f"  step:     {args.step}")
    print("=" * 60)

    # Print meta.txt for each branch
    for b in args.branches:
        meta = pathlib.Path(args.data_root) / b / "meta.txt"
        if meta.exists():
            print(f"\n[{b}] {meta.read_text().strip()}")

    for env_name in envs:
        _compare_state(args.branches, args.data_root, env_name, args.step)
        _compare_images(args.branches, args.data_root, env_name, args.step, save_dir)

    print(f"\nImage panels saved to: {save_dir}")


if __name__ == "__main__":
    tyro.cli(main)
