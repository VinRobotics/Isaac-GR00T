# eval_mimicgen.py

import sys
sys.path.insert(0, "/home/locht1/gr00t_equi_fa")

import dataclasses
import logging
import pathlib
import random
from typing import Optional

import imageio
import numpy as np
import tqdm
import tyro

from evaluation.gr00tn15_inference import Gr00tn15_inference, _quat2axisangle, invert_gripper_action


TASK_TO_ENV: dict[str, str] = {
    "coffee":                "Coffee_D2",
    "coffee preparation":    "CoffeePreparation_D1",
    "hammer cleanup":        "HammerCleanup_D1",
    "kitchen":               "Kitchen_D1",
    "mug cleanup":           "MugCleanup_D1",
    "nut assembly":          "NutAssembly_D0",
    "pick place":            "PickPlace_D0",
    "square":                "Square_D2",
    "stack":                 "Stack_D1",
    "stack three":           "StackThree_D1",
    "threading":             "Threading_D2",
    "three piece assembly":  "ThreePieceAssembly_D2",
}

TASK_TO_INSTRUCTION: dict[str, str] = {
    "coffee":               "Pick up the white coffee pod and place it into the red holder of the coffee machine.",
    "coffee preparation":   "Place the mug under the black coffee machine, pull open the red drawer, then take the white coffee pod from the drawer and insert it into the red holder of the coffee machine.",
    "hammer cleanup":       "Pull the drawer open, pick up the hammer and place it inside the drawer, and then push the drawer closed.",
    "kitchen":              "Turn the stove on, pick up the grey container and place it on the stove, then pick up the brown food item and place it inside the grey container, then push the container onto the red trivet.",
    "mug cleanup":          "Pull the drawer open, pick up the mug and place it inside the drawer, then push the drawer closed.",
    "nut assembly":         "Pick up the square-shaped brown nut and slide it onto the square brown peg, then pick up the round silver nut and slide it onto the round silver peg.",
    "pick place":           "Pick up the white milk box and place it in the top-left bin, pick up the red cereal box and place it in the top-right of the dark tray, pick up the red can and place it in the bottom-right of the dark tray, and finally pick up the brown square box and place it in the bottom-left bin of the dark tray.",
    "square":               "Pick up the square-shaped brown nut and slide it onto the square brown peg.",
    "stack":                "Pick up the red block and stack it directly on top of the green block.",
    "stack three":          "Stack the red block on top of the green block, then stack the blue block on top of the red block.",
    "threading":            "Pick up the black handle of the threading tool and carefully insert the thin needle tip through the small metal loop on the wooden stand.",
    "three piece assembly": "Place the small T-shaped red block into the center of the hollow square base, then pick up the large notched red block and stack it on top of the T-shaped block to complete the assembly.",
}

# Per-task step budget: ~1.5× observed dataset max, rounded to nearest 50.
TASK_MAX_STEPS: dict[str, int] = {
    "coffee":               380,
    "coffee preparation":   1150,
    "hammer cleanup":       490,
    "kitchen":              1000,
    "mug cleanup":          570,
    "nut assembly":         650,
    "pick place":           1200,
    "square":               260,
    "stack":                200,
    "stack three":          450,
    "threading":            400,
    "three piece assembly": 560,
}

MIMICGEN_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MIMICGEN_ENV_RESOLUTION = 84


def _make_env(env_name: str, resolution: int, robosuite_assets_path: str = ""):
    import robosuite as suite
    import robosuite.models
    from robosuite.controllers import load_controller_config
    import mimicgen.envs.robosuite  # noqa: F401 — registers MimicGen envs

    if robosuite_assets_path:
        robosuite.models.assets_root = robosuite_assets_path

    ctrl = load_controller_config(default_controller="OSC_POSE")
    return suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=ctrl,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=20,
        reward_shaping=False,
        ignore_done=False,
    )


def to_video_frame(arr):
    arr = np.asarray(arr)
    if any(s < 0 for s in arr.strides) or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
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
    pretrained_model_path: str = ""
    resize_size: int = 84
    infer_chunk: int = 10
    save_videos_root: str = "/tmp/mimicgen_eval_results"
    num_steps_wait: int = 5
    num_trials_per_task: int = 10
    seed: int = 7
    exp_name: str = "test"
    model_type: str = "gr00tn15"
    tasks: Optional[list[str]] = None  # subset of TASK_TO_ENV keys; None = all
    # Override robosuite's asset root (MuJoCo XMLs, textures, etc.).
    # Equivalent to pointing robosuite.models.assets_root at a custom folder.
    # Leave empty to use the default installed package assets.
    robosuite_assets_path: str = ""


def eval_mimicgen(args: Args) -> None:
    np.random.seed(args.seed)

    log_dir = pathlib.Path(f"{args.save_videos_root}/log/eval_results/{args.exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "mimicgen.log"

    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    try:
        if args.model_type == "gr00tn15":
            mypolicy = Gr00tn15_inference(args.pretrained_model_path, args.infer_chunk)
        else:
            raise ValueError(f"Unsupported model_type: {args.model_type}")
        logging.info(f"Loaded {args.model_type} policy from {args.pretrained_model_path}")
    except Exception as e:
        logging.error(f"Failed to load policy: {e}")
        return

    tasks_to_run = args.tasks if args.tasks else list(TASK_TO_ENV.keys())

    total_episodes, total_successes = 0, 0
    summary_rows = []

    for task_name in tasks_to_run:
        env_name = TASK_TO_ENV.get(task_name)
        if env_name is None:
            logging.warning(f"No env mapping for task '{task_name}', skipping.")
            continue

        logging.info(f"\n{'='*60}\nTask: {task_name}  Env: {env_name}")
        print(f"\n{'='*60}\nTask: {task_name}  |  Env: {env_name}")

        save_video_dir = pathlib.Path(
            f"{args.save_videos_root}/{args.exp_name}/videos/{task_name.replace(' ', '_')}"
        )
        save_video_dir.mkdir(parents=True, exist_ok=True)

        task_instruction = TASK_TO_INSTRUCTION[task_name]
        max_steps = TASK_MAX_STEPS[task_name]
        env = _make_env(env_name, args.resize_size, args.robosuite_assets_path)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=task_name):
            random.seed(args.seed + episode_idx)
            np.random.seed(args.seed + episode_idx)
            obs = env.reset()
            t = 0
            done = False
            info = {}
            replay_images = []
            replay_images_wrist = []

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, done, info = env.step(MIMICGEN_DUMMY_ACTION)
                    t += 1
                    continue

                action_chunk = mypolicy.get_mimicgen_action(obs, task_instruction)

                for act in action_chunk:
                    obs, _, done, info = env.step(act.tolist())
                    t += 1

                    replay_images.append(to_video_frame(obs["agentview_image"][::-1, ::-1]))
                    replay_images_wrist.append(to_video_frame(obs["robot0_eye_in_hand_image"][::-1, ::-1]))

                    if done or info.get("success", False):
                        done = True
                        break

                if done:
                    break

            success = done or bool(info.get("success", False))
            if success:
                task_successes += 1
                total_successes += 1
            task_episodes += 1
            total_episodes += 1

            suffix = "success" if success else "failure"
            tag = f"{task_name.replace(' ', '_')}_seed{args.seed}_ep{episode_idx}"

            imageio.mimwrite(
                save_video_dir / f"{tag}_static_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
                codec="libx264",
            )
            imageio.mimwrite(
                save_video_dir / f"{tag}_wrist_{suffix}.mp4",
                [np.asarray(x) for x in replay_images_wrist],
                fps=10,
                codec="libx264",
            )

            logging.info(f"  ep {episode_idx}: {'success' if success else 'failure'}  steps={t}")

        env.close()

        sr = float(task_successes) / float(max(task_episodes, 1))
        logging.info(f"Task '{task_name}' success rate: {task_successes}/{task_episodes} ({sr*100:.1f}%)")
        print(f"  -> {task_name}: {task_successes}/{task_episodes} ({sr*100:.1f}%)")
        summary_rows.append((task_name, task_successes, task_episodes))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for task_name, ok, total in summary_rows:
        pct = 100 * ok / max(total, 1)
        print(f"  {task_name:<30s}  {ok:3d}/{total:3d}  ({pct:.1f}%)")
    overall_pct = 100 * total_successes / max(total_episodes, 1)
    print(f"  {'TOTAL':<30s}  {total_successes:3d}/{total_episodes:3d}  ({overall_pct:.1f}%)")
    logging.info(f"Total success rate: {total_successes}/{total_episodes} ({overall_pct:.1f}%)")


if __name__ == "__main__":
    tyro.cli(eval_mimicgen)
