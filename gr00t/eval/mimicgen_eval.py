# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# In-training MimicGen evaluation.  Mirrors equidiff's `rollout_every` pattern:
# at every `save_steps` training steps, spin up N parallel MimicGen envs and
# run `num_trials` episodes with the in-memory GR00T policy.  Logs the success
# rate to the active reporter (wandb / tensorboard) via the Trainer state.
#
# Design choices (locked in with the user):
#   - One task per training run (`--mimicgen_task_eval <task_name>`).
#   - n_envs_parallel = min(num_trials, batch_size) — caps GPU memory.
#   - AsyncVectorEnv-style: each env runs in its own subprocess and
#     communicates via a Pipe.  We batch obs from N envs into one model call.
#   - Skip Gr00tPolicy disk reload: re-use the training model in place.

import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from transformers import TrainerCallback

from gr00t.data.schema import EmbodimentTag
from gr00t.model.policy import Gr00tPolicy


# ── Task metadata (mirrors evaluation/eval_mimicgen.py) ───────────────────────

TASK_TO_ENV: Dict[str, str] = {
    "square": "Square_D2",
    "stack": "Stack_D1",
    "stack three": "StackThree_D1",
    "hammer cleanup": "HammerCleanup_D1",
    "kitchen": "Kitchen_D1",
    "coffee": "Coffee_D2",
    "coffee preparation": "CoffeePreparation_D1",
    "mug cleanup": "MugCleanup_D1",
    "nut assembly": "NutAssembly_D0",
    "pick place": "PickPlace_D0",
    "threading": "Threading_D2",
    "three piece assembly": "ThreePieceAssembly_D2",
}

TASK_TO_INSTRUCTION: Dict[str, str] = {
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
    "stack": "Stack red block on top of green block.",
}

TASK_MAX_STEPS: Dict[str, int] = {
    "coffee": 400,
    "coffee preparation": 1200,
    "hammer cleanup": 540,
    "kitchen": 1050,
    "mug cleanup": 620,
    "nut assembly": 700,
    "pick place": 1250,
    "square": 400,
    "stack": 400,
    "stack three": 500,
    "threading": 500,
    "three piece assembly": 600,
}

TASK_TO_ROBOT: Dict[str, str] = {
    "coffee": "Panda",
    "coffee preparation": "Panda",
    "hammer cleanup": "Panda",
    "kitchen": "Panda",
    "mug cleanup": "Panda",
    "nut assembly": "Sawyer",
    "pick place": "Sawyer",
    "square": "Panda",
    "stack": "Panda",
    "stack three": "Panda",
    "threading": "Panda",
    "three piece assembly": "Panda",
}


def _normalize_task_name(task: str) -> str:
    """Accept both `square` and `square_d2`-style names; normalize to canonical form."""
    t = task.replace("_", " ").strip().lower()
    if t in TASK_TO_ENV:
        return t
    raise ValueError(
        f"Unknown MimicGen task '{task}'. Known: {sorted(TASK_TO_ENV.keys())}"
    )


# ── Robosuite env worker (one subprocess per env) ─────────────────────────────


def _make_env(env_name: str, resolution: int, robot: str):
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    import mimicgen_envs.envs.robosuite  # noqa: F401  – registers MimicGen envs

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
    return env


_DUMMY_DELTA_ACTION = [0.0] * 6 + [-1.0]  # zero-pose delta, gripper open


def _env_worker(
    pipe,
    env_name: str,
    resolution: int,
    robot: str,
    sys_path_extra: List[str],
    use_abs_action: bool = False,
    num_steps_wait: int = 5,
):
    """One subprocess = one robosuite env.  Commands: reset / step / close.

    `sys_path_extra` propagates the parent's sys.path entries (the spawn
    context starts a fresh interpreter and would otherwise drop user-added
    paths like the mimicgen_envs install dir).

    When `use_abs_action=True`, each reset runs `num_steps_wait` dummy delta
    steps to let the robot settle, then switches the controller to absolute
    mode (use_delta=False) before returning the obs.
    """
    for p in sys_path_extra:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    # Each worker independently configures MUJOCO_GL for offscreen render.
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = _make_env(env_name, resolution, robot)
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                seed = data
                if seed is not None:
                    np.random.seed(seed)
                obs = env.reset()
                if use_abs_action:
                    # Ensure delta mode is active for the wait phase.
                    for robot_obj in env.robots:
                        robot_obj.controller.use_delta = True
                    for _ in range(num_steps_wait):
                        obs, _, _, _ = env.step(_DUMMY_DELTA_ACTION)
                    # Switch to absolute mode for the policy phase.
                    for robot_obj in env.robots:
                        robot_obj.controller.use_delta = False
                pipe.send(("ok", obs))
            elif cmd == "step":
                obs, reward, done, info = env.step(data)
                success = bool(env._check_success())
                pipe.send(("ok", (obs, float(reward), bool(done), success)))
            elif cmd == "close":
                pipe.send(("ok", None))
                break
            else:
                pipe.send(("err", f"unknown cmd {cmd}"))
    except Exception as e:
        try:
            pipe.send(("err", f"{type(e).__name__}: {e}"))
        except BrokenPipeError:
            pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        pipe.close()


class MimicgenParallelEnvs:
    """
    Pool of N robosuite envs, one per subprocess.  Reset/step batched across
    all N envs synchronously.  Episodes are fully managed by the caller — no
    auto-reset (matches equidiff's modified AsyncVectorEnv).
    """

    def __init__(
        self,
        env_name: str,
        n_envs: int,
        resolution: int,
        robot: str,
        use_abs_action: bool = False,
        num_steps_wait: int = 5,
    ):
        self.env_name = env_name
        self.n_envs = n_envs
        self.resolution = resolution
        self.robot = robot

        # Capture sys.path so workers can import robosuite / mimicgen_envs even
        # when those packages live in user-added paths (eval_mimicgen.py style
        # `sys.path.insert(0, "/path/to/mimicgen")`).  Spawn starts a fresh
        # interpreter that drops these.
        sys_path_extra = list(sys.path)

        ctx = mp.get_context("spawn")
        self.parent_pipes: List = []
        self.processes: List = []
        for _ in range(n_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_env_worker,
                args=(child, env_name, resolution, robot, sys_path_extra),
                kwargs={"use_abs_action": use_abs_action, "num_steps_wait": num_steps_wait},
                daemon=True,
            )
            p.start()
            child.close()
            self.parent_pipes.append(parent)
            self.processes.append(p)

    def reset(self, seeds: List[Optional[int]]) -> List[dict]:
        """Reset the first len(seeds) envs.  Remaining workers stay idle.

        Lets the caller shrink the active batch on the last chunk so we don't
        pay for unused envs (e.g. with n_envs=14 and num_trials=50, chunks are
        14, 14, 14, 8 — the last chunk only steps 8 envs).
        """
        assert 1 <= len(seeds) <= self.n_envs
        active = len(seeds)
        for pipe, seed in zip(self.parent_pipes[:active], seeds):
            pipe.send(("reset", seed))
        results = [pipe.recv() for pipe in self.parent_pipes[:active]]
        for status, payload in results:
            if status != "ok":
                raise RuntimeError(f"env worker error during reset: {payload}")
        return [r[1] for r in results]

    def step(self, actions: List[List[float]]):
        """Step the first len(actions) envs.  Symmetric with reset()."""
        assert 1 <= len(actions) <= self.n_envs
        active = len(actions)
        for pipe, a in zip(self.parent_pipes[:active], actions):
            pipe.send(("step", a))
        results = [pipe.recv() for pipe in self.parent_pipes[:active]]
        out = []
        for status, payload in results:
            if status != "ok":
                raise RuntimeError(f"env worker error during step: {payload}")
            out.append(payload)
        return out  # list of (obs, reward, done, success)

    def close(self):
        for pipe in self.parent_pipes:
            try:
                pipe.send(("close", None))
                pipe.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# ── In-memory policy (skips disk re-load of the training model) ───────────────


class _InMemoryGr00tPolicy(Gr00tPolicy):
    """
    Reuses Gr00tPolicy's transforms / get_action pipeline but skips
    `_load_model` — the model is already loaded by the trainer.
    """

    def __init__(
        self,
        model,
        exp_cfg_dir: Path,
        modality_config,
        modality_transform,
        embodiment_tag,
        device: str = "cuda",
        denoising_steps: Optional[int] = 4,
    ):
        # Mirror Gr00tPolicy.__init__ minus snapshot_download + _load_model.
        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()
        self.model_path = Path(exp_cfg_dir).parent
        self.device = device

        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        self.model = model
        self.model.eval()

        self._load_metadata(Path(exp_cfg_dir))
        self._load_horizons()

        if denoising_steps is not None and hasattr(self.model, "action_head") and hasattr(
            self.model.action_head, "num_inference_timesteps"
        ):
            self.model.action_head.num_inference_timesteps = denoising_steps

        self.smooth_option = ""  # eval path — no temporal-ensemble / rtc smoothing


# ── Obs → model-input conversion (robosuite obs ⇒ EquiRelMimicgenConfig) ──────


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """scipy as_rotvec — matches `_mimicgen_quat2axisangle` in gr00tn15_inference.py."""
    return Rotation.from_quat(quat).as_rotvec()


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.transpose(1, 2, 0)
    return img


def _extract_state(obs: dict, state_keys: List[str]) -> dict:
    """
    Build the state dict from a robosuite obs, populating only the keys the
    data config declares. Rotation representation is inferred from state_keys:
      - quaternion if state.qx/qy/qz/qw are present (uses robot0_eef_quat directly, xyzw)
      - axis-angle (rotvec) if state.roll/pitch/yaw are present
    """
    xyz = obs["robot0_eef_pos"]
    quat = obs["robot0_eef_quat"]  # xyzw
    gripper = obs["robot0_gripper_qpos"].astype(np.float32)

    out = {
        "state.x": np.array([xyz[0]], dtype=np.float32),
        "state.y": np.array([xyz[1]], dtype=np.float32),
        "state.z": np.array([xyz[2]], dtype=np.float32),
        "state.gripper": gripper,
    }
    if "state.qx" in state_keys or "state.qw" in state_keys:
        out["state.qx"] = np.array([quat[0]], dtype=np.float32)
        out["state.qy"] = np.array([quat[1]], dtype=np.float32)
        out["state.qz"] = np.array([quat[2]], dtype=np.float32)
        out["state.qw"] = np.array([quat[3]], dtype=np.float32)
    else:
        rpy = _quat2axisangle(quat)
        out["state.roll"] = np.array([rpy[0]], dtype=np.float32)
        out["state.pitch"] = np.array([rpy[1]], dtype=np.float32)
        out["state.yaw"] = np.array([rpy[2]], dtype=np.float32)
    return out


def _extract_images(obs: dict) -> Dict[str, np.ndarray]:
    """robosuite renders upside-down via OpenGL; [::-1] corrects it."""
    img = _to_hwc_uint8(obs["agentview_image"])[::-1]
    wrist = _to_hwc_uint8(obs["robot0_eye_in_hand_image"])[::-1]
    return {"video.image": img, "video.wrist_image": wrist}


def _build_batched_input(
    env_obs_list: List[dict],
    frame_buffers: List[Dict[str, list]],
    n_obs_steps: int,
    task_instruction: str,
    state_keys: List[str],
) -> Dict[str, Any]:
    """
    Build the GR00T model input dict for B envs.

    Maintains per-env rolling buffers of (state, image, wrist) for n_obs_steps.
    On the first call of an episode the buffers are seeded with copies of the
    current frame (zero-order hold), matching how the dataset pads boundaries.

    Returns a dict with keys:
        video.image:        (B, T, H, W, C)  uint8
        video.wrist_image:  (B, T, H, W, C)  uint8
        state.x..gripper:   (B, T, D)        float32
        annotation.human.action.task_description: tuple of B strings
    """
    B = len(env_obs_list)

    for i, obs in enumerate(env_obs_list):
        imgs = _extract_images(obs)
        st = _extract_state(obs, state_keys)
        buf = frame_buffers[i]
        if "video.image" not in buf:
            # seed buffer with n_obs_steps copies of the current frame
            for k in ("video.image", "video.wrist_image"):
                buf[k] = [imgs[k].copy() for _ in range(n_obs_steps)]
            for k in state_keys:
                buf[k] = [st[k].copy() for _ in range(n_obs_steps)]
        else:
            for k in ("video.image", "video.wrist_image"):
                buf[k].append(imgs[k])
                if len(buf[k]) > n_obs_steps:
                    buf[k] = buf[k][-n_obs_steps:]
            for k in state_keys:
                buf[k].append(st[k])
                if len(buf[k]) > n_obs_steps:
                    buf[k] = buf[k][-n_obs_steps:]

    def _stack_video(key):
        # Per-env: list of T (H,W,C) → (T,H,W,C).  Across B: (B,T,H,W,C).
        return np.stack(
            [np.stack(frame_buffers[i][key], axis=0) for i in range(B)], axis=0
        )

    def _stack_state(key):
        # Per-env: list of T (D,) → (T,D).  Across B: (B,T,D).
        return np.stack(
            [np.stack(frame_buffers[i][key], axis=0) for i in range(B)], axis=0
        ).astype(np.float32)

    model_input: Dict[str, Any] = {
        "video.image": _stack_video("video.image"),
        "video.wrist_image": _stack_video("video.wrist_image"),
        "annotation.human.action.task_description": tuple([str(task_instruction)] * B),
    }
    for k in state_keys:
        model_input[k] = _stack_state(k)
    return model_input


# ── Action conversion (model output ⇒ env action) ────────────────────────────


def _convert_action_chunk(
    action_chunk: dict,
    batch_size: int,
    n_action_steps: int,
    action_keys: List[str],
    use_abs_action: bool = False,
) -> np.ndarray:
    """
    action_chunk[k] (for k in action_keys) has shape (B, action_horizon) or
    (B, action_horizon, 1). Returns (B, n_action_steps, len(action_keys)).

    For relative actions the last channel (gripper) is converted from [0,1]
    to signed {-1, +1} and inverted.  For absolute actions the policy output
    is passed through unchanged (the model is trained to output the raw
    robosuite gripper signal directly).
    """
    n_dim = len(action_keys)
    out = np.zeros((batch_size, n_action_steps, n_dim), dtype=np.float32)
    for ci, k in enumerate(action_keys):
        v = np.asarray(action_chunk[k])  # (B, H) or (B, H, 1)
        if v.ndim == 3 and v.shape[-1] == 1:
            v = v[..., 0]
        out[:, :, ci] = v[:, :n_action_steps]
    if not use_abs_action:
        # gripper: [0,1] → [-1,+1] → binarize → invert (matches gr00tn15_inference)
        out[..., -1] = 2.0 * out[..., -1] - 1.0
        out[..., -1] = np.sign(out[..., -1])
        out[..., -1] *= -1.0
    return out


# ── Rollout loop ──────────────────────────────────────────────────────────────


def run_rollout(
    policy: _InMemoryGr00tPolicy,
    envs: MimicgenParallelEnvs,
    task_instruction: str,
    max_steps: int,
    n_obs_steps: int,
    n_action_steps: int,
    seeds: List[int],
    state_keys: List[str],
    action_keys: List[str],
    record_video_mask: Optional[List[bool]] = None,
    use_abs_action: bool = False,
) -> Tuple[List[bool], List[Optional[List[np.ndarray]]]]:
    """
    Run one synchronous chunk of `len(seeds)` episodes in parallel.  The chunk
    size can be smaller than `envs.n_envs` — the remaining workers stay idle.
    Inference batch size = chunk size, so the last (short) chunk also gets a
    smaller GPU forward pass.

    Each env runs until success or max_steps; an episode that finishes early
    is marked done but other envs continue stepping with their own actions.

    If `record_video_mask[i]` is True, agentview frames for env i are appended
    to the returned per-env frame list (recording stops once that env is done).
    Mirrors equidiff's `n_test_vis` first-N-episodes recording pattern.

    Returns:
        (successes, frames_per_env):
            successes        — list[bool] of length n_active
            frames_per_env   — list[Optional[list[HWC uint8 ndarray]]]; None
                               where record_video_mask was False.
    """
    n_active = len(seeds)
    assert 1 <= n_active <= envs.n_envs
    if record_video_mask is None:
        record_video_mask = [False] * n_active
    assert len(record_video_mask) == n_active

    env_obs_list = envs.reset(seeds)
    frame_buffers: List[Dict[str, list]] = [dict() for _ in range(n_active)]
    successes = [False] * n_active
    done_flags = [False] * n_active
    video_frames: List[Optional[List[np.ndarray]]] = [
        [] if record_video_mask[i] else None for i in range(n_active)
    ]

    # Capture the reset frame for any env we're recording (so video starts at
    # the initial scene, not after the first policy step).
    for i, obs in enumerate(env_obs_list):
        buf = video_frames[i]
        if buf is not None:
            buf.append(_to_hwc_uint8(obs["agentview_image"])[::-1])

    t = 0
    while t < max_steps and not all(done_flags):
        model_input = _build_batched_input(
            env_obs_list, frame_buffers, n_obs_steps, task_instruction, state_keys
        )
        try:
            action_chunk = policy.get_action(model_input)
        except Exception as e:
            print(f"[mimicgen_eval] policy.get_action failed: {e}")
            break

        actions = _convert_action_chunk(action_chunk, n_active, n_action_steps, action_keys, use_abs_action)

        for step_idx in range(n_action_steps):
            if t >= max_steps:
                break
            step_actions = [actions[i, step_idx].tolist() for i in range(n_active)]
            results = envs.step(step_actions)
            t += 1
            new_obs_list = []
            for i, (obs, _reward, done, success) in enumerate(results):
                was_done = done_flags[i]
                if not was_done:
                    if success:
                        successes[i] = True
                        done_flags[i] = True
                    elif done:
                        done_flags[i] = True
                    # Record frame as long as the episode was still running at
                    # the start of this step (so the success frame is included).
                    buf = video_frames[i]
                    if buf is not None:
                        buf.append(_to_hwc_uint8(obs["agentview_image"])[::-1])
                new_obs_list.append(obs)
            env_obs_list = new_obs_list
            if all(done_flags):
                break

    return successes, video_frames


# ── Trainer callback ──────────────────────────────────────────────────────────


class MimicgenEvalCallback(TrainerCallback):
    """
    Hooks transformers.Trainer:  on every checkpoint save, run a MimicGen
    eval on `task_name`.  Total trials = `num_trials` (default 50), run in
    chunks of `min(num_trials, batch_size)` parallel envs.  Logs
    `mimicgen/eval/<task>/success_rate` to the active reporter.
    """

    def __init__(
        self,
        task_name: str,
        modality_config,
        modality_transform,
        embodiment_tag,
        exp_cfg_dir: Path,
        batch_size: int,
        num_trials: int = 50,
        n_envs_override: int = 0,
        n_action_steps: int = 8,
        denoising_steps: int = 4,
        resolution: int = 84,
        seed: int = 100000,
        n_test_vis: int = 4,
        video_fps: int = 20,
        output_dir: Optional[str] = None,
        device: str = "cuda",
        use_abs_action: bool = False,
        num_steps_wait: int = 5,
    ):
        super().__init__()
        self.task_name = _normalize_task_name(task_name)
        self.env_name = TASK_TO_ENV[self.task_name]
        self.robot = TASK_TO_ROBOT[self.task_name]
        self.task_instruction = TASK_TO_INSTRUCTION[self.task_name]
        self.max_steps = TASK_MAX_STEPS[self.task_name]

        self.modality_config = modality_config
        self.modality_transform = modality_transform
        self.embodiment_tag = embodiment_tag
        self.exp_cfg_dir = Path(exp_cfg_dir)
        self.batch_size = batch_size
        self.num_trials = num_trials
        # n_envs precedence: explicit override > auto (min of num_trials, batch_size)
        if n_envs_override > 0:
            self.n_envs = min(n_envs_override, num_trials)
        else:
            self.n_envs = min(num_trials, batch_size)
        self.n_action_steps = n_action_steps
        self.denoising_steps = denoising_steps
        self.resolution = resolution
        self.seed = seed
        self.n_test_vis = max(0, min(n_test_vis, num_trials))
        self.video_fps = video_fps
        self.output_dir = Path(output_dir) if output_dir else None
        self.device = device
        self.use_abs_action = use_abs_action
        self.num_steps_wait = num_steps_wait

        # n_obs_steps derived from modality_config (observation_indices)
        self.n_obs_steps = len(modality_config["video"].delta_indices)

        # Lazy: envs are spawned on first use (subprocess fork after CUDA init
        # is dangerous — robosuite + EGL handle this via MUJOCO_GL=egl).
        self._envs: Optional[MimicgenParallelEnvs] = None

        # Only rank 0 runs eval (DDP).
        self._rank = int(os.environ.get("RANK", 0))

        self._wandb_metrics_defined = False

    def _ensure_envs(self):
        if self._envs is None:
            print(
                f"[mimicgen_eval] spawning {self.n_envs} envs for task "
                f"'{self.task_name}' ({self.env_name}, robot={self.robot})"
            )
            self._envs = MimicgenParallelEnvs(
                env_name=self.env_name,
                n_envs=self.n_envs,
                resolution=self.resolution,
                robot=self.robot,
                use_abs_action=self.use_abs_action,
                num_steps_wait=self.num_steps_wait,
            )

    def on_save(self, args, state, control, **kwargs):
        if self._rank != 0:
            return control
        model = kwargs.get("model", None)
        if model is None:
            print("[mimicgen_eval] model not in callback kwargs; skipping eval")
            return control

        # Unwrap DDP / FSDP / PEFT wrappers so we hit the GR00T_N1_5 forward.
        inner_model = model
        for attr in ("module", "_orig_mod"):
            if hasattr(inner_model, attr):
                inner_model = getattr(inner_model, attr)

        if not self.exp_cfg_dir.exists():
            print(
                f"[mimicgen_eval] exp_cfg_dir not found at {self.exp_cfg_dir}; "
                "skipping eval (metadata not yet written)"
            )
            return control

        # Resolve where to save MP4s: explicit > trainer's output_dir.
        out_root = self.output_dir if self.output_dir is not None else Path(args.output_dir)

        global_step = int(state.global_step)
        t0 = time.time()
        try:
            success_rate, succ_count, video_paths = self._run_eval(
                inner_model, out_root, global_step
            )
        except Exception as e:
            print(f"[mimicgen_eval] eval failed at step {global_step}: {e}")
            import traceback
            traceback.print_exc()
            return control
        elapsed = time.time() - t0

        msg = (
            f"[mimicgen_eval] step={global_step} task={self.task_name} "
            f"SR={succ_count}/{self.num_trials} ({success_rate*100:.1f}%) "
            f"elapsed={elapsed:.1f}s"
        )
        print(msg)

        # Log to the active reporter(s).  Callbacks do not receive a `trainer`
        # reference, so we call the reporter SDKs directly when available.
        task_slug = self.task_name.replace(" ", "_")
        metric_key = f"mimicgen/eval/{task_slug}/success_rate"
        metrics = {metric_key: success_rate}
        try:
            import wandb  # type: ignore
            if getattr(wandb, "run", None) is not None:
                # Bind mimicgen/eval/* metrics to the trainer's global_step axis.
                # Passing step= to wandb.log() here would desync wandb's internal
                # step counter from the HF Trainer's WandbCallback (which also
                # logs with step=state.global_step), making train/* show up at
                # wrong x positions.
                if not self._wandb_metrics_defined:
                    wandb.define_metric("train/global_step")
                    wandb.define_metric(
                        "mimicgen/eval/*", step_metric="train/global_step"
                    )
                    self._wandb_metrics_defined = True
                payload: Dict[str, Any] = dict(metrics)
                payload["train/global_step"] = global_step
                for ep_idx, vid_path, ep_success in video_paths:
                    suffix = "success" if ep_success else "failure"
                    key = f"mimicgen/eval/{task_slug}/video_ep{ep_idx:02d}_{suffix}"
                    try:
                        payload[key] = wandb.Video(
                            str(vid_path), fps=self.video_fps, format="mp4"
                        )
                    except Exception as ve:
                        print(f"[mimicgen_eval] wandb.Video failed for {vid_path}: {ve}")
                wandb.log(payload)
        except Exception:
            pass
        # Also push the scalar into log_history so it shows up in trainer_state.json.
        state.log_history.append({**metrics, "step": global_step})

        return control

    def _run_eval(
        self, model, out_root: Path, global_step: int
    ) -> Tuple[float, int, List[Tuple[int, Path, bool]]]:
        """
        Run the full eval: chunks → rollouts → optional MP4 save.

        Returns:
            success_rate          — float in [0, 1]
            succ_count            — int (numerator)
            video_paths           — list of (episode_idx, mp4_path, success) for
                                    the first `n_test_vis` episodes (or fewer
                                    if imageio is unavailable / save fails).
        """
        self._ensure_envs()
        assert self._envs is not None

        policy = _InMemoryGr00tPolicy(
            model=model,
            exp_cfg_dir=self.exp_cfg_dir,
            modality_config=self.modality_config,
            modality_transform=self.modality_transform,
            embodiment_tag=self.embodiment_tag,
            device=self.device,
            denoising_steps=self.denoising_steps,
        )

        # Lazy imageio import — only needed when recording.
        imageio = None
        video_dir: Optional[Path] = None
        if self.n_test_vis > 0:
            try:
                import imageio as _imageio  # type: ignore
                imageio = _imageio
                task_slug = self.task_name.replace(" ", "_")
                video_dir = out_root / "mimicgen_eval_videos" / f"step_{global_step:08d}" / task_slug
                video_dir.mkdir(parents=True, exist_ok=True)
            except ImportError:
                print("[mimicgen_eval] imageio not installed; skipping video recording")

        was_training = model.training
        model.eval()
        try:
            n_chunks = math.ceil(self.num_trials / self.n_envs)
            all_successes: List[bool] = []
            video_paths: List[Tuple[int, Path, bool]] = []
            seed_cursor = self.seed
            episode_cursor = 0  # absolute index 0..num_trials-1
            for chunk_idx in range(n_chunks):
                # Shrink the last chunk to remaining trials (e.g. n_envs=14
                # and num_trials=50 → chunks of 14, 14, 14, 8).  The unused
                # worker subprocesses sit idle.
                remaining = self.num_trials - len(all_successes)
                take = min(self.n_envs, remaining)
                seeds = [seed_cursor + i for i in range(take)]
                seed_cursor += take

                # Record the first n_test_vis episodes (absolute index).
                record_video_mask = [
                    (episode_cursor + i) < self.n_test_vis and imageio is not None
                    for i in range(take)
                ]
                print(
                    f"[mimicgen_eval] chunk {chunk_idx + 1}/{n_chunks}: "
                    f"running {take} envs (seeds {seeds[0]}..{seeds[-1]}; "
                    f"recording {sum(record_video_mask)} videos)"
                )
                chunk_successes, chunk_videos = run_rollout(
                    policy=policy,
                    envs=self._envs,
                    task_instruction=self.task_instruction,
                    max_steps=self.max_steps,
                    n_obs_steps=self.n_obs_steps,
                    n_action_steps=self.n_action_steps,
                    seeds=seeds,
                    state_keys=list(self.modality_config["state"].modality_keys),
                    action_keys=list(self.modality_config["action"].modality_keys),
                    record_video_mask=record_video_mask,
                    use_abs_action=self.use_abs_action,
                )
                all_successes.extend(chunk_successes)

                # Save recorded videos to disk.
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
                                fps=self.video_fps,
                                codec="libx264",
                            )
                            video_paths.append((abs_ep, mp4_path, ep_success))
                        except Exception as e:
                            print(f"[mimicgen_eval] failed to write {mp4_path}: {e}")
                episode_cursor += take
        finally:
            if was_training:
                model.train()

        succ = int(sum(all_successes))
        rate = succ / max(1, self.num_trials)
        return rate, succ, video_paths

    def on_train_end(self, args, state, control, **kwargs):
        if self._envs is not None:
            self._envs.close()
            self._envs = None
        return control
