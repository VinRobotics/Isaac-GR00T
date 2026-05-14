# eval_mimicgen_2step.py
# Like eval_mimicgen.py but uses EquiRelMimicgen2StepConfig (observation_indices=[-1, 0]).
# Both video and state are fed as 2-frame windows: shape (1, 2, H, W, C) and (1, 2, D).
# At the very first step of each episode the previous frame is duplicated (zero-order hold).

import sys
sys.path.insert(0, "/home/locht1/gr00t_equi_dit")
sys.path.insert(0, "/mnt/data/sftp/data/locht1/mimicgen_evaluation/mimicgen")

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

from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import EquiRelMimicgen2StepConfig
from gr00t.data.schema import EmbodimentTag
from evaluation.gr00tn15_inference import (
    _to_hwc_uint8,
    _mimicgen_quat2axisangle,
    invert_gripper_action,
)


# ── task tables (identical to eval_mimicgen.py) ───────────────────────────────

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

TASK_TO_INSTRUCTION: dict[str, str] = {
    "stack three":           "Stack red on green, then blue on red.",
    "coffee preparation":    "Place mug under coffee machine, open red drawer, take white pod and insert into red holder.",
    "hammer cleanup":        "Open drawer, place hammer inside, close drawer.",
    "coffee":                "Place white coffee pod into red holder.",
    "nut assembly":          "Slide square brown nut onto square peg, then round silver nut onto round peg.",
    "pick place":            "Place milk box in top-left bin, cereal box in top-right tray, red can in bottom-right tray, brown box in bottom-left tray.",
    "square":                "Slide square brown nut onto square peg.",
    "threading":             "Pick up threading tool and insert needle tip through the metal loop on the stand.",
    "three piece assembly":  "Insert T-shaped red block into square base, then stack large notched red block on top.",
    "kitchen":               "Turn on stove, place grey container on stove, add brown food inside, push container onto red trivet.",
    "mug cleanup":           "Open drawer, place mug inside, close drawer.",
    "stack":                 "Stack red block on top of green block.",
}

TASK_MAX_STEPS: dict[str, int] = {
    "coffee":                400,
    "coffee preparation":    1200,
    "hammer cleanup":        540,
    "kitchen":               1050,
    "mug cleanup":           620,
    "nut assembly":          700,
    "pick place":            1250,
    "square":                400,
    "stack":                 400,
    "stack three":           500,
    "threading":             500,
    "three piece assembly":  600,
}

TASK_TO_ROBOT: dict[str, str] = {
    "coffee":                "Panda",
    "coffee preparation":    "Panda",
    "hammer cleanup":        "Panda",
    "kitchen":               "Panda",
    "mug cleanup":           "Panda",
    "nut assembly":          "Sawyer",
    "pick place":            "Sawyer",
    "square":                "Panda",
    "stack":                 "Panda",
    "stack three":           "Panda",
    "threading":             "Panda",
    "three piece assembly":  "Panda",
}

_GRIPPER_INIT_QPOS = 0.020833
_FRANKA_GRIPPER_MAX = 0.04
_GRIPPER_WAIT_ACTION = 2.0 * _GRIPPER_INIT_QPOS / _FRANKA_GRIPPER_MAX - 1.0
MIMICGEN_DUMMY_ACTION = [0.0] * 6 + [_GRIPPER_WAIT_ACTION]
MIMICGEN_ENV_RESOLUTION = 84


# ── env wrapper ───────────────────────────────────────────────────────────────

class _SuccessDoneWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["is_success"] = bool(self.env._check_success())
        done = done or bool(info["is_success"])
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


def _make_env(env_name: str, resolution: int, robosuite_assets_path: str = "", robot: str = "Panda"):
    import robosuite as suite
    import robosuite.models
    from robosuite.controllers import load_controller_config
    import mimicgen_envs.envs.robosuite  # noqa: F401

    if robosuite_assets_path:
        robosuite.models.assets_root = robosuite_assets_path

    ctrl = load_controller_config(default_controller="OSC_POSE")
    ctrl.update({"control_delta": True})
    env = suite.make(
        env_name=env_name,
        robots=robot,
        controller_configs=ctrl,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=20,
        reward_shaping=False,
        ignore_done=True,
    )
    return _SuccessDoneWrapper(env)


# ── 2-step inference wrapper ──────────────────────────────────────────────────

class Gr00tn15_2step_inference:
    """
    Policy wrapper for EquiRelMimicgen2StepConfig (observation_indices=[-1, 0]).

    Maintains a 2-frame rolling buffer for video and state.  On the first step
    of each episode the buffer is seeded with two copies of the current frame
    (zero-order hold), matching how the dataset pads episode boundaries.

    Video input shapes:
        video.image       : (1, 2, H, W, C)   [t-1, t]
        video.wrist_image : (1, 2, H, W, C)   [t-1, t]

    State input shapes:   (1, 2, D)            [t-1, t]
    """

    def __init__(self, model_dir: str = "", infer_chunk: int = 10):
        self.model_dir = model_dir
        self.infer_chunk = infer_chunk
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

        data_config = EquiRelMimicgen2StepConfig()
        self.policy = Gr00tPolicy(
            model_path=model_dir,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            denoising_steps=4,
            device="cuda",
        )
        print(f"Loaded 2-step N1.5 policy from: {model_dir}")

        # rolling buffers — reset at the start of each episode
        self._prev_img: Optional[np.ndarray] = None        # (H, W, C) uint8
        self._prev_wrist: Optional[np.ndarray] = None      # (H, W, C) uint8
        self._prev_state: Optional[dict] = None            # {key: np.ndarray scalar-or-1d}

    def reset(self):
        """Call at the start of every episode to clear the frame buffer."""
        self._prev_img = None
        self._prev_wrist = None
        self._prev_state = None

    # ── observation → model dict ──────────────────────────────────────────────

    def _extract_state(self, obs) -> dict:
        """Pull raw state values from a robosuite obs dict."""
        xyz = obs["robot0_eef_pos"]                                    # (3,)
        rpy = _mimicgen_quat2axisangle(obs["robot0_eef_quat"])         # (3,)
        gripper = obs["robot0_gripper_qpos"].astype(np.float32)        # (2,)
        return {
            "state.x":     np.array([xyz[0]], dtype=np.float32),
            "state.y":     np.array([xyz[1]], dtype=np.float32),
            "state.z":     np.array([xyz[2]], dtype=np.float32),
            "state.roll":  np.array([rpy[0]], dtype=np.float32),
            "state.pitch": np.array([rpy[1]], dtype=np.float32),
            "state.yaw":   np.array([rpy[2]], dtype=np.float32),
            "state.gripper": gripper,                                  # (2,)
        }

    def _process_observation(self, obs, task_description: str) -> dict:
        """
        Build the 2-frame model input dict from the current robosuite obs.

        Video: robosuite renders upside-down via OpenGL; [::-1] corrects it.
        Shape: (H, W, C) uint8 → stored in buffer → stacked to (1, 2, H, W, C).
        State: scalar/1-d arrays → stacked to (1, 2, D).
        """
        # ── current frame ──
        curr_img   = _to_hwc_uint8(obs["agentview_image"])[::-1]       # (H, W, C)
        curr_wrist = _to_hwc_uint8(obs["robot0_eye_in_hand_image"])[::-1]
        curr_state = self._extract_state(obs)

        # ── seed buffer on first step ──
        if self._prev_img is None:
            self._prev_img   = curr_img.copy()
            self._prev_wrist = curr_wrist.copy()
            self._prev_state = {k: v.copy() for k, v in curr_state.items()}

        # ── stack [t-1, t] along the temporal dimension ──
        # video: (H, W, C) → (1, 2, H, W, C)
        def stack_video(prev, curr):
            return np.stack([prev, curr], axis=0)[np.newaxis]          # (1, 2, H, W, C)

        # state: (D,) → (1, 2, D)
        def stack_state(key):
            p = self._prev_state[key]
            c = curr_state[key]
            return np.stack([p, c], axis=0)[np.newaxis]                # (1, 2, D)

        model_input = {
            "video.image":       stack_video(self._prev_img, curr_img),
            "video.wrist_image": stack_video(self._prev_wrist, curr_wrist),
            "state.x":           stack_state("state.x"),
            "state.y":           stack_state("state.y"),
            "state.z":           stack_state("state.z"),
            "state.roll":        stack_state("state.roll"),
            "state.pitch":       stack_state("state.pitch"),
            "state.yaw":         stack_state("state.yaw"),
            "state.gripper":     stack_state("state.gripper"),
            "annotation.human.action.task_description": (str(task_description),),
        }

        # ── advance buffer ──
        self._prev_img   = curr_img.copy()
        self._prev_wrist = curr_wrist.copy()
        self._prev_state = {k: v.copy() for k, v in curr_state.items()}

        return model_input

    # ── action query ─────────────────────────────────────────────────────────

    def get_action(self, obs, task_description: str) -> np.ndarray:
        data = self._process_observation(obs, task_description)
        try:
            action_chunk = self.policy.get_action(data)
        except Exception as e:
            print(f"Error querying policy: {e}")
            return np.array([MIMICGEN_DUMMY_ACTION], dtype=np.float32)
        return self._convert_action_chunk(action_chunk)

    def _convert_action_chunk(self, action_chunk: dict) -> np.ndarray:
        actions = []
        for t in range(self.infer_chunk):
            components = []
            for key in self.action_keys:
                val = action_chunk[f"action.{key}"][0]
                val_t = np.array(val[t] if hasattr(val, "shape") and val.ndim > 0 else val,
                                 dtype=np.float32).reshape(-1)
                components.extend(val_t.tolist())
            action_array = np.array(components, dtype=np.float32)
            action_array = self._normalize_gripper(action_array)
            action_array = invert_gripper_action(action_array)
            actions.append(action_array)
        return np.stack(actions, axis=0)   # (infer_chunk, D)

    @staticmethod
    def _normalize_gripper(action: np.ndarray, binarize: bool = True) -> np.ndarray:
        action[..., -1] = 2 * action[..., -1] - 1
        if binarize:
            action[..., -1] = np.sign(action[..., -1])
        return action


# ── video helpers ─────────────────────────────────────────────────────────────

def to_video_frame(arr):
    arr = np.asarray(arr)
    if any(s < 0 for s in arr.strides) or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype in (np.float32, np.float64):
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        arr = (arr * 255.0).round().astype(np.uint8) if 0.0 <= vmin and vmax <= 1.0 \
              else np.clip(arr, 0, 255).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


# ── eval loop ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    pretrained_model_path: str = ""
    resize_size: int = 84
    infer_chunk: int = 10
    save_videos_root: str = "/tmp/mimicgen_eval_2step_results"
    num_trials_per_task: int = 10
    seed: int = 7
    exp_name: str = "test_2step"
    tasks: Optional[list[str]] = None   # None = all tasks
    robosuite_assets_path: str = ""


def eval_mimicgen_2step(args: Args, tasks: Optional[list[str]] = None) -> None:
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
    log_file = log_dir / f"mimicgen_2step_{task_range_tag}.log"

    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    try:
        policy = Gr00tn15_2step_inference(args.pretrained_model_path, args.infer_chunk)
        logging.info(f"Loaded 2-step policy from {args.pretrained_model_path}")
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

        logging.info(f"\n{'='*60}\nTask: {task_name}  Env: {env_name}")
        print(f"\n{'='*60}\nTask: {task_name}  |  Env: {env_name}")

        save_video_dir = pathlib.Path(
            f"{args.save_videos_root}/{args.exp_name}/videos/{task_name.replace(' ', '_')}"
        )
        save_video_dir.mkdir(parents=True, exist_ok=True)

        task_instruction = TASK_TO_INSTRUCTION[task_name]
        max_steps = TASK_MAX_STEPS[task_name]
        robot = TASK_TO_ROBOT.get(task_name, "Panda")
        env = _make_env(env_name, args.resize_size, args.robosuite_assets_path, robot)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=task_name):
            random.seed(args.seed + episode_idx)
            np.random.seed(args.seed + episode_idx)

            obs = env.reset()
            policy.reset()   # clear the 2-frame buffer at episode start

            t = 0
            done = False
            info = {}
            replay_images = []
            replay_images_wrist = []

            while t < max_steps:
                action_chunk = policy.get_action(obs, task_instruction)

                for act in action_chunk:
                    obs, _, done, info = env.step(act.tolist())
                    t += 1

                    replay_images.append(to_video_frame(obs["agentview_image"][::-1]))
                    replay_images_wrist.append(to_video_frame(obs["robot0_eye_in_hand_image"][::-1]))

                    if done or info.get("is_success", False):
                        done = True
                        break

                if done:
                    break

            success = done or bool(info.get("is_success", False))
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
                fps=20, codec="libx264",
            )
            imageio.mimwrite(
                save_video_dir / f"{tag}_wrist_{suffix}.mp4",
                [np.asarray(x) for x in replay_images_wrist],
                fps=20, codec="libx264",
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


def eval_mimicgen_2step_all(args: Args) -> None:
    print("=" * 80)
    print("MimicGen 2-Step Simulation Evaluation")
    print("=" * 80)

    all_tasks = args.tasks if args.tasks else list(TASK_TO_ENV.keys())
    all_tasks = [t.replace("_", " ") for t in all_tasks]
    if len(all_tasks) == 1:
        task_splits = [all_tasks]
    else:
        mid = len(all_tasks) // 2
        task_splits = [all_tasks[:mid], all_tasks[mid:]]

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(task_splits), mp_context=ctx) as pool:
        futures = {
            pool.submit(eval_mimicgen_2step, args, task_ids): task_ids
            for task_ids in task_splits
        }
        for fut in as_completed(futures):
            task_ids = futures[fut]
            label = f"tasks_{task_ids[0].replace(' ', '_')}-{task_ids[-1].replace(' ', '_')}"
            try:
                fut.result()
                print(f"[DONE] {label}")
            except Exception as e:
                print(f"[ERROR] {label} failed: {e}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        import torch
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    tyro.cli(eval_mimicgen_2step_all)
