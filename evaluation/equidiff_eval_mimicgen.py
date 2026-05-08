# equidiff_eval_mimicgen.py
#
# Evaluates GR00T with relative control on MimicGen environments using
# equidiff's exact environment initialization pattern:
#   - env_meta is loaded from the task's HDF5 dataset (FileUtils.get_env_metadata_from_dataset)
#   - env is created via EnvUtils.create_env_from_metadata (robomimic's EnvRobosuite)
#   - ObsUtils modality mapping is initialized before env creation
#
# Observation format from robomimic EnvRobosuite:
#   - Images: HWC uint8, already vertically flipped by robomimic (OpenGL correction).
#     No manual flip is needed – use get_mimicgen_action_equidiff() in gr00tn15_inference.
#   - Other keys (eef_pos, eef_quat, gripper_qpos): same as raw robosuite.
#
# Inference loop differences from eval_mimicgen.py:
#   1. No warm-up dummy steps (equidiff starts inference immediately after reset).
#   2. No _SuccessDoneWrapper; episode runs to max_steps (ignore_done in env_meta).
#   3. Success = max reward > 0 across full episode, matching equidiff's
#      np.max(all_rewards[i]) criterion.

import sys
sys.path.insert(0, "/home/locht1/gr00t_equi_dit")
sys.path.insert(0, "/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen")

import collections
import dataclasses
import logging
import multiprocessing as mp
import pathlib
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import imageio
import numpy as np
import tqdm
import tyro

from evaluation.gr00tn15_inference import Gr00tn15_inference


# ── Task metadata ──────────────────────────────────────────────────────────────

TASK_TO_ENV: dict[str, str] = {
    "square":                "Square_D2",
    "stack":                 "Stack_D1",
    "stack three":           "StackThree_D1",
    "hammer cleanup":        "HammerCleanup_D1",
    "kitchen":               "Kitchen_D1",
    "coffee":                "Coffee_D2",
    "coffee preparation":    "CoffeePreparation_D1",
    "mug cleanup":           "MugCleanup_D1",
    "nut assembly":          "NutAssembly_D0",
    "pick place":            "PickPlace_D0",
    "threading":             "Threading_D2",
    "three piece assembly":  "ThreePieceAssembly_D2",
}

# equidiff dataset names follow mimicgen convention: lowercase snake_case
# e.g. data/robomimic/datasets/square_d2/square_d2.hdf5
TASK_TO_DATASET_NAME: dict[str, str] = {
    "square":                "square_d2",
    "stack":                 "stack_d1",
    "stack three":           "stack_three_d1",
    "hammer cleanup":        "hammer_cleanup_d1",
    "kitchen":               "kitchen_d1",
    "coffee":                "coffee_d2",
    "coffee preparation":    "coffee_preparation_d1",
    "mug cleanup":           "mug_cleanup_d1",
    "nut assembly":          "nut_assembly_d0",
    "pick place":            "pick_place_d0",
    "threading":             "threading_d2",
    "three piece assembly":  "three_piece_assembly_d2",
}

TASK_TO_INSTRUCTION: dict[str, str] = {
    "stack three": "Stack red on green, then blue on red.",
    "coffee preparation": "Place mug under coffee machine, open red drawer, take white pod and insert into red holder.",
    "hammer cleanup": "Open drawer, place hammer inside, close drawer.",
    "coffee": "Place white coffee pod into red holder.",
    "nut assembly": "Slide square brown nut onto square peg, then round silver nut onto round peg.",
    "pick place": "Place milk box in top-left bin, cereal box in top-right tray, red can in bottom-right tray, brown box in bottom-left tray.",
    "square": "Slide square brown nut onto square peg.",
    "threading": "Pick up threading tool and insert needle tip through the metal loop on the stand.",
    "three piece assembly": "Insert T-shaped red block into square base, then stack large notched red block on top.",
    "kitchen": "Turn on stove, place grey container on stove, add brown food inside, push container onto red trivet.",
    "mug cleanup": "Open drawer, place mug inside, close drawer.",
    "stack": "Stack red block on top of green block."
}

TASK_MAX_STEPS: dict[str, int] = {
    "coffee":               400,
    "coffee preparation":   1200,
    "hammer cleanup":       540,
    "kitchen":              1050,
    "mug cleanup":          620,
    "nut assembly":         700,
    "pick place":           1250,
    "square":               400,
    "stack":                400,
    "stack three":          500,
    "threading":            500,
    "three piece assembly": 600,
}

TASK_TO_ROBOT: dict[str, str] = {
    "coffee":               "Panda",
    "coffee preparation":   "Panda",
    "hammer cleanup":       "Panda",
    "kitchen":              "Panda",
    "mug cleanup":          "Panda",
    "nut assembly":         "Sawyer",
    "pick place":           "Sawyer",
    "square":               "Panda",
    "stack":                "Panda",
    "stack three":          "Panda",
    "threading":            "Panda",
    "three piece assembly": "Panda",
}

# Obs keys that equidiff's mimicgen_rel.yaml shape_meta declares
_IMAGE_OBS_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
_LOWDIM_OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


# ── Environment creation (equidiff pattern) ────────────────────────────────────

def _init_obs_utils():
    """Initialize robomimic ObsUtils modality mapping (must run before env creation)."""
    import robomimic.utils.obs_utils as ObsUtils
    modality_mapping = collections.defaultdict(list)
    modality_mapping["rgb"].extend(_IMAGE_OBS_KEYS)
    modality_mapping["low_dim"].extend(_LOWDIM_OBS_KEYS)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)


def _make_env(dataset_path: str, resolution: int):
    """
    Create a MimicGen / robomimic env from a dataset's stored env_meta.

    This mirrors equidiff's robomimic_image_runner.py:
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env = EnvUtils.create_env_from_metadata(env_meta, ...)

    The returned env is a robomimic EnvRobosuite object.  Its step/reset API is
    identical to robosuite's, but images are already vertically flipped by
    robomimic (correcting OpenGL's upside-down rendering), so no manual flip is
    needed when building policy inputs.

    control_delta stays True (relative control) as stored in the dataset's
    env_meta – equidiff's abs_action=False path leaves this unchanged.
    """
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import mimicgen_envs.envs.robosuite  # noqa: F401 – registers MimicGen envs

    _init_obs_utils()

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    # Override camera resolution to match our model's training resolution.
    env_meta["env_kwargs"]["camera_heights"] = resolution
    env_meta["env_kwargs"]["camera_widths"] = resolution

    # Ensure image obs are enabled.
    env_meta["env_kwargs"]["use_camera_obs"] = True
    env_meta["env_kwargs"]["use_object_obs"] = False

    # hard_reset wastes memory; disable as equidiff does.
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    env.env.hard_reset = False

    return env  # robomimic EnvRobosuite – no _SuccessDoneWrapper


# ── Video helpers ──────────────────────────────────────────────────────────────

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


# ── CLI args ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    pretrained_model_path: str = ""
    # Root directory that contains per-task dataset folders, following equidiff's
    # convention:  {dataset_dir}/{task_dataset_name}/{task_dataset_name}.hdf5
    # e.g.  /data/robomimic/datasets/square_d2/square_d2.hdf5
    dataset_dir: str = ""
    resize_size: int = 84
    infer_chunk: int = 10
    save_videos_root: str = "/tmp/mimicgen_eval_results"
    num_trials_per_task: int = 10
    seed: int = 7
    exp_name: str = "test"
    model_type: str = "gr00tn15"
    tasks: Optional[list[str]] = None


# ── Per-process evaluation ─────────────────────────────────────────────────────

def eval_mimicgen(args: Args, tasks: Optional[list[str]] = None) -> None:
    np.random.seed(args.seed)

    if tasks is not None:
        args.tasks = tasks

    tasks_to_run = args.tasks if args.tasks else list(TASK_TO_ENV.keys())
    task_range_tag = (
        f"tasks_{tasks_to_run[0].replace(' ', '_')}-{tasks_to_run[-1].replace(' ', '_')}"
        if tasks_to_run else "all"
    )

    log_dir = pathlib.Path(f"{args.save_videos_root}/log/eval_results/{args.exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"mimicgen_{task_range_tag}.log"

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

    total_episodes, total_successes = 0, 0
    summary_rows = []

    for task_name in tasks_to_run:
        env_name = TASK_TO_ENV.get(task_name)
        if env_name is None:
            logging.warning(f"No env mapping for task '{task_name}', skipping.")
            continue

        dataset_name = TASK_TO_DATASET_NAME[task_name]
        dataset_path = str(
            pathlib.Path(args.dataset_dir) / dataset_name / f"{dataset_name}.hdf5"
        )

        logging.info(f"\n{'='*60}\nTask: {task_name}  Env: {env_name}")
        logging.info(f"Dataset: {dataset_path}")
        print(f"\n{'='*60}\nTask: {task_name}  |  Env: {env_name}")
        print(f"  Dataset: {dataset_path}")

        save_video_dir = pathlib.Path(
            f"{args.save_videos_root}/{args.exp_name}/videos/{task_name.replace(' ', '_')}"
        )
        save_video_dir.mkdir(parents=True, exist_ok=True)

        task_instruction = TASK_TO_INSTRUCTION[task_name]
        max_steps = TASK_MAX_STEPS[task_name]

        # equidiff env init: robomimic EnvRobosuite from dataset env_meta
        env = _make_env(dataset_path, args.resize_size)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=task_name):
            random.seed(args.seed + episode_idx)
            np.random.seed(args.seed + episode_idx)

            obs = env.reset()

            # ── equidiff: no warm-up dummy steps ──────────────────────────────
            # robomimic_image_runner starts inference immediately after env.reset().

            t = 0
            episode_max_reward = 0.0
            replay_images = []
            replay_images_wrist = []

            while t < max_steps:
                # get_mimicgen_action_equidiff processes obs with flip_mode=None because
                # robomimic's EnvRobosuite already flips images ([::-1]) internally.
                # eval_mimicgen.py uses flip_mode="vertical" on top of suite.make() for
                # the same net orientation.
                action_chunk = mypolicy.get_mimicgen_action_equidiff(obs, task_instruction)

                # Execute full chunk, breaking only if env signals done.
                # Mirrors MultiStepWrapper.step(): loops over actions and breaks on done.
                chunk_done = False
                for act in action_chunk:
                    if t >= max_steps:
                        break

                    obs, reward, done, info = env.step(act.tolist())
                    t += 1

                    # equidiff success criterion: max reward across episode > 0
                    episode_max_reward = max(episode_max_reward, float(reward))

                    # Images from robomimic EnvRobosuite are already correctly oriented
                    # (robomimic applies [::-1] flip internally).  No extra flip here.
                    replay_images.append(to_video_frame(obs["agentview_image"]))
                    replay_images_wrist.append(to_video_frame(obs["robot0_eye_in_hand_image"]))

                    if done:
                        chunk_done = True
                        break

                if chunk_done:
                    break

            # equidiff success: np.max(all_rewards[i]) > 0
            success = episode_max_reward > 0.0 or bool(env._check_success())

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
                fps=20,
                codec="libx264",
            )
            imageio.mimwrite(
                save_video_dir / f"{tag}_wrist_{suffix}.mp4",
                [np.asarray(x) for x in replay_images_wrist],
                fps=20,
                codec="libx264",
            )

            logging.info(
                f"  ep {episode_idx}: {'success' if success else 'failure'}"
                f"  steps={t}  max_reward={episode_max_reward:.3f}"
            )

        env.close()

        sr = float(task_successes) / float(max(task_episodes, 1))
        logging.info(f"Task '{task_name}' SR: {task_successes}/{task_episodes} ({sr*100:.1f}%)")
        print(f"  -> {task_name}: {task_successes}/{task_episodes} ({sr*100:.1f}%)")
        summary_rows.append((task_name, task_successes, task_episodes))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for task_name, ok, total in summary_rows:
        pct = 100 * ok / max(total, 1)
        print(f"  {task_name:<30s}  {ok:3d}/{total:3d}  ({pct:.1f}%)")
    overall_pct = 100 * total_successes / max(total_episodes, 1)
    print(f"  {'TOTAL':<30s}  {total_successes:3d}/{total_episodes:3d}  ({overall_pct:.1f}%)")
    logging.info(f"Total SR: {total_successes}/{total_episodes} ({overall_pct:.1f}%)")


def eval_mimicgen_all(args: Args) -> None:
    print("=" * 80)
    print("MimicGen Simulation Evaluation (equidiff env init)")
    print("=" * 80)

    all_tasks = args.tasks if args.tasks else list(TASK_TO_ENV.keys())
    mid = len(all_tasks) // 2
    task_splits = [all_tasks[:mid], all_tasks[mid:]]

    ctx = mp.get_context("spawn")
    results = dict()

    with ProcessPoolExecutor(max_workers=len(task_splits), mp_context=ctx) as pool:
        futures = {
            pool.submit(eval_mimicgen, args, task_ids): task_ids
            for task_ids in task_splits
        }
        for fut in as_completed(futures):
            task_ids = futures[fut]
            label = f"tasks_{task_ids[0].replace(' ', '_')}-{task_ids[-1].replace(' ', '_')}"
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

    tyro.cli(eval_mimicgen_all)
