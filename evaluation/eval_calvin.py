# eval_calvin.py
#
# CALVIN evaluation (Long-Horizon Multi-Task Language Control / LH-MTLC).
# Mirrors the structure of eval_libero.py / eval_robocasa.py and uses
# the GR00T N1.5 policy with EquiRelCalvinConfig (relative cartesian control).
#
# CALVIN ABCD-D protocol:
#   - 1000 evaluation sequences, each containing 5 sequential subtasks.
#   - Robot is reset to a neutral pose between sequences.
#   - A task oracle decides success/failure for each subtask using info dicts.
#   - Action space (relative): [dx, dy, dz, drx, dry, drz, gripper] in [-1, 1];
#     gripper must be exactly -1 (close) or +1 (open).

import os
import sys

# Match the eval_libero.py pattern so the cluster install of GR00T / CALVIN
# is on sys.path even when this script is run from arbitrary cwd.
sys.path.insert(0, "/mnt/data/sftp/data/locht1/workspace/gr00t_equi_fa_simpler_fuse")  # for gr00t.model.policy
sys.path.insert(0, "/mnt/data/sftp/data/locht1/calvin/calvin_env")
sys.path.insert(0, "/mnt/data/sftp/data/locht1/calvin/calvin_models")

import dataclasses
import logging
import math
import pathlib
import multiprocessing as mp
from collections import defaultdict
from typing import Optional

import imageio
import numpy as np
import torch
import tqdm
import tyro
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- GR00T policy -----------------------------------------------------------
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import EquiRelCalvinConfig
from gr00t.data.schema import EmbodimentTag


CALVIN_DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
EP_LEN = 360               # per CALVIN paper / official evaluator
NUM_SEQUENCES = 1000       # standard LH-MTLC eval length


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
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
    if arr.dtype in (np.float32, np.float64):
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if 0.0 <= vmin and vmax <= 1.0:
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.transpose(1, 2, 0)
    return img


# ---------------------------------------------------------------------------
# Policy wrapper (GR00T N1.5 + EquiRelCalvinConfig, relative actions)
# ---------------------------------------------------------------------------
class Gr00tCalvinInference:
    """GR00T N1.5 policy adapter for CALVIN (relative control)."""

    def __init__(self, model_dir: str = "", infer_chunk: int = 10):
        self.model_dir = model_dir
        self.infer_chunk = infer_chunk
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.policy = self._create_policy()

    def _create_policy(self):
        data_config = EquiRelCalvinConfig()
        policy = Gr00tPolicy(
            model_path=self.model_dir,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            denoising_steps=4,
            device="cuda",
        )
        print("Number of parameters:", sum(p.numel() for p in policy.model.parameters()))
        print(f"Loaded N1.5 CALVIN policy from: {self.model_dir}")
        return policy

    def _process_observation(self, obs, task_description: str):
        """
        CALVIN base env observation layout:
          obs["rgb_obs"]["rgb_static"]   : (H, W, 3) uint8, default 200x200
          obs["rgb_obs"]["rgb_gripper"]  : (H, W, 3) uint8, default 84x84
          obs["robot_obs"] (euler, len 15):
              [0:3]  tcp_pos (x, y, z)
              [3:6]  tcp_orn (roll, pitch, yaw)
              [6:7]  gripper_opening_width
              [7:14] arm_joint_states
              [14:]  last gripper_action (-1 / +1)
        """
        static = _to_hwc_uint8(obs["rgb_obs"]["rgb_static"])
        gripper = _to_hwc_uint8(obs["rgb_obs"]["rgb_gripper"])

        robot = np.asarray(obs["robot_obs"], dtype=np.float32)
        xyz = robot[0:3]
        rpy = robot[3:6]
        grip_width = robot[6:7]

        return {
            "video.image":        static[np.newaxis, np.newaxis, ...],
            "video.wrist_image":  gripper[np.newaxis, np.newaxis, ...],
            "state.x":       np.array([[[xyz[0]]]], dtype=np.float32),
            "state.y":       np.array([[[xyz[1]]]], dtype=np.float32),
            "state.z":       np.array([[[xyz[2]]]], dtype=np.float32),
            "state.roll":    np.array([[[rpy[0]]]], dtype=np.float32),
            "state.pitch":   np.array([[[rpy[1]]]], dtype=np.float32),
            "state.yaw":     np.array([[[rpy[2]]]], dtype=np.float32),
            "state.gripper": np.array([[[grip_width[0]]]], dtype=np.float32),
            "annotation.human.action.task_description": (str(task_description),),
        }

    def _to_action_chunk(self, action_chunk: dict, start_idx: int = 0) -> np.ndarray:
        """Stack the per-key dict from policy into a (chunk, 7) array."""
        out = []
        for t in range(start_idx, start_idx + self.infer_chunk):
            comps = []
            for key in self.action_keys:
                val = action_chunk.get(f"action.{key}")
                if val is None:
                    raise ValueError(f"Missing action.{key} in policy output")
                # batch-first: val has shape (1, T, D) or (T, D); take inner-most index
                v = val[0] if (hasattr(val, "shape") and val.shape[0] == 1 and val.ndim > 1) else val
                if hasattr(v, "shape") and v.ndim >= 1:
                    if t >= v.shape[0]:
                        raise IndexError(f"chunk idx {t} out of range for {key} ({v.shape[0]})")
                    v_t = v[t]
                else:
                    v_t = v
                comps.extend(np.asarray(v_t, dtype=np.float32).reshape(-1).tolist())

            act = np.asarray(comps, dtype=np.float32)
            # CALVIN: gripper must be exactly -1 or +1. Train data uses {-1, +1};
            # after min_max un-normalization the value sits near those extremes,
            # so a sign() is the safest binarization. Treat 0 as open (+1).
            g = act[-1]
            act[-1] = 1.0 if g >= 0 else -1.0
            # Clip cartesian/orientation deltas to the canonical [-1, 1] range
            # expected by the relative-action robot interface.
            act[:6] = np.clip(act[:6], -1.0, 1.0)
            out.append(act)
        return np.stack(out, axis=0)

    def get_calvin_action(self, obs, task_description: str) -> np.ndarray:
        data = self._process_observation(obs, task_description)
        try:
            action_chunk = self.policy.get_action(data)
        except Exception as e:
            print(f"[Gr00tCalvinInference] policy error: {e}")
            return np.tile(CALVIN_DUMMY_ACTION, (self.infer_chunk, 1))
        return self._to_action_chunk(action_chunk)

    # --- Batched inference for parallel envs ---------------------------------
    def _build_batched_data(self, obs_list, lang_list) -> dict:
        n = len(obs_list)
        assert n == len(lang_list), "obs / lang length mismatch"
        statics = np.stack([_to_hwc_uint8(o["rgb_obs"]["rgb_static"])  for o in obs_list], axis=0)
        grippers = np.stack([_to_hwc_uint8(o["rgb_obs"]["rgb_gripper"]) for o in obs_list], axis=0)
        robots = np.stack([np.asarray(o["robot_obs"], dtype=np.float32) for o in obs_list], axis=0)
        xyz, rpy, grip = robots[:, 0:3], robots[:, 3:6], robots[:, 6:7]
        return {
            "video.image":       statics[:, None, ...],          # (N, 1, H, W, C)
            "video.wrist_image": grippers[:, None, ...],
            "state.x":       xyz[:, None, 0:1].astype(np.float32),  # (N, 1, 1)
            "state.y":       xyz[:, None, 1:2].astype(np.float32),
            "state.z":       xyz[:, None, 2:3].astype(np.float32),
            "state.roll":    rpy[:, None, 0:1].astype(np.float32),
            "state.pitch":   rpy[:, None, 1:2].astype(np.float32),
            "state.yaw":     rpy[:, None, 2:3].astype(np.float32),
            "state.gripper": grip[:, None, :].astype(np.float32),
            "annotation.human.action.task_description": tuple(str(l) for l in lang_list),
        }

    def _extract_action_chunk(self, batched_action_chunk: dict, env_idx: int) -> np.ndarray:
        """Pull env_idx's (chunk, 7) actions out of a (B, T, D) policy output."""
        out = []
        for t in range(self.infer_chunk):
            comps = []
            for key in self.action_keys:
                val = batched_action_chunk[f"action.{key}"]
                v = val[env_idx]
                if t >= v.shape[0]:
                    raise IndexError(f"chunk idx {t} out of range for {key} ({v.shape[0]})")
                comps.extend(np.asarray(v[t], dtype=np.float32).reshape(-1).tolist())
            act = np.asarray(comps, dtype=np.float32)
            g = act[-1]
            act[-1] = 1.0 if g >= 0 else -1.0
            act[:6] = np.clip(act[:6], -1.0, 1.0)
            out.append(act)
        return np.stack(out, axis=0)

    def get_calvin_actions_batched(self, obs_list, lang_list) -> list:
        """Batched inference. Returns list of N (chunk, 7) np arrays in input order."""
        n = len(obs_list)
        if n == 0:
            return []
        data = self._build_batched_data(obs_list, lang_list)
        try:
            batched_chunk = self.policy.get_action(data)
        except Exception as e:
            print(f"[Gr00tCalvinInference] batched policy error: {e}")
            return [np.tile(CALVIN_DUMMY_ACTION, (self.infer_chunk, 1)) for _ in range(n)]
        return [self._extract_action_chunk(batched_chunk, i) for i in range(n)]


# ---------------------------------------------------------------------------
# CALVIN env / oracle helpers
# ---------------------------------------------------------------------------
def _load_calvin_env(dataset_path: Optional[str] = None, scene: str = "calvin_scene_D"):
    """
    Build the CALVIN env.

    If `dataset_path` is given and contains `validation/.hydra/merged_config.yaml`,
    we use the canonical config shipped with the dataset (most reproducible).

    Otherwise we compose the env directly from `calvin_env`'s own hydra configs —
    no dataset download required. For ABCD-D eval, use scene='calvin_scene_D'
    (the held-out env that ABCD-D is evaluated on).
    """
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    import calvin_env
    from calvin_env.envs.play_table_env import get_env

    # Path A: canonical config from a downloaded dataset folder
    if dataset_path is not None:
        cfg_path = pathlib.Path(dataset_path) / "validation" / ".hydra" / "merged_config.yaml"
        if cfg_path.is_file():
            return get_env(pathlib.Path(dataset_path) / "validation", show_gui=False)

    # Path B: no dataset — compose from calvin_env's installed configs
    conf_dir = (pathlib.Path(calvin_env.__file__).parents[1] / "conf").resolve()
    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(config_dir=str(conf_dir), version_base=None)

    cfg = compose(
        config_name="config_data_collection",
        overrides=[
            f"scene={scene}",
            "cameras=static_and_gripper",
            "robot=panda_longer_finger",
        ],
    )
    env = hydra.utils.instantiate(
        cfg.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
    )
    return env


def _load_task_oracle_and_annotations(calvin_models_root: str):
    """
    Build the task oracle and load val language annotations. The yaml paths
    follow the layout shipped in github.com/mees/calvin.
    """
    import hydra
    from omegaconf import OmegaConf

    conf_dir = pathlib.Path(calvin_models_root) / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return task_oracle, val_annotations


def _get_eval_sequences(num_sequences: int):
    """Use local copy to avoid pulling in calvin_agent.evaluation.utils, which
    transitively imports MCIL → pytorch_lightning. See calvin_sequences.py."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from calvin_sequences import get_sequences
    return get_sequences(num_sequences)


def _get_env_state_for_initial_condition(initial_condition):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from calvin_sequences import get_env_state_for_initial_condition
    return get_env_state_for_initial_condition(initial_condition)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    # Model
    pretrained_model_path: str = ""
    infer_chunk: int = 10

    # CALVIN dataset / repo paths.
    # `dataset_path` is OPTIONAL — if empty, env is built from calvin_env's
    # own hydra configs (no download needed). For ABCD-D, scene D is used.
    dataset_path: str = ""
    scene: str = "calvin_scene_D"  # ABCD-D evaluates on env D
    calvin_models_root: str = "/mnt/data/sftp/data/locht1/calvin/calvin_models"

    # Eval config
    task_suite_name: str = "calvin_abcd_d"   # used only as a label / log tag
    num_sequences: int = NUM_SEQUENCES
    ep_len: int = EP_LEN
    n_workers: int = 1   # parallel eval workers; 1 = in-process (lowest RAM)
    n_envs: int = 1      # parallel envs in a single worker (batched policy inference)

    # Output
    save_videos_root: str = "/mnt/data/sftp/data/locht1/calvin_eval_results"
    exp_name: str = "test"
    save_video_every: int = 25  # only save every Nth sequence to save disk

    seed: int = 7
    model_type: str = "gr00tn15"

    # Smoke-test mode: steps the env with CALVIN_DUMMY_ACTION (no policy load,
    # no calvin_models_root, no oracle). Verifies env build + render + video save.
    debug: bool = False
    debug_steps: int = 200


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def _count_success(results):
    """Return [SR(>=1 task), SR(>=2 tasks), ..., SR(>=5 tasks)] over result counts."""
    from collections import Counter
    cnt = Counter(results)
    out = []
    for i in range(1, 6):
        n = sum(cnt[j] for j in range(i, 6))
        out.append(n / max(len(results), 1))
    return out


def _rollout_subtask(env, policy, task_oracle, subtask, lang, ep_len, infer_chunk,
                     static_frames, gripper_frames):
    start_info = env.get_info()
    obs = env.get_obs()

    t = 0
    while t < ep_len:
        try:
            action_chunk = policy.get_calvin_action(obs, lang)
        except Exception as e:
            logging.info(f"Subtask '{subtask}' caught policy exception: {e}")
            return False

        for act in action_chunk:
            obs, _, _, info = env.step(act)
            t += 1
            static_frames.append(to_video_frame(obs["rgb_obs"]["rgb_static"]))
            gripper_frames.append(to_video_frame(obs["rgb_obs"]["rgb_gripper"]))

            completed = task_oracle.get_task_info_for_set(start_info, info, {subtask})
            if len(completed) > 0:
                return True
            if t >= ep_len:
                break
    return False


def eval_calvin(args: Args, sequence_ids: Optional[list] = None) -> list:
    np.random.seed(args.seed)

    log_dir = pathlib.Path(f"{args.save_videos_root}/log/eval_results/{args.exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"seq_{sequence_ids[0]}-{sequence_ids[-1]}"
           if sequence_ids else "all")
    log_file = log_dir / f"{args.task_suite_name}_{tag}.log"

    # Per-worker logging (clear handlers first so re-runs don't double-log)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    video_root = pathlib.Path(f"{args.save_videos_root}/{args.exp_name}/videos/{args.task_suite_name}")
    video_root.mkdir(parents=True, exist_ok=True)

    # Load policy FIRST — HF from_pretrained peaks at ~2x model size in CPU RAM
    # during load, so we want the env (pybullet + Mesa) RAM footprint absent here.
    try:
        if args.model_type == "gr00tn15":
            policy = Gr00tCalvinInference(args.pretrained_model_path, args.infer_chunk)
        else:
            raise ValueError(f"Unsupported model_type: {args.model_type}")
        logging.info(f"Loaded {args.model_type} policy from {args.pretrained_model_path}")
    except Exception as e:
        logging.error(f"Failed to load policy: {e}")
        return []

    # Load env + oracle after the policy is on the GPU
    env = _load_calvin_env(args.dataset_path or None, scene=args.scene)
    task_oracle, val_annotations = _load_task_oracle_and_annotations(args.calvin_models_root)
    eval_sequences = _get_eval_sequences(args.num_sequences)
    if sequence_ids is not None:
        eval_sequences = [eval_sequences[i] for i in sequence_ids]

    results = []
    for seq_idx, (initial_state, eval_sequence) in enumerate(tqdm.tqdm(eval_sequences)):
        robot_obs, scene_obs = _get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        success_counter = 0
        save_this = (seq_idx % max(args.save_video_every, 1) == 0)
        static_frames, gripper_frames = [], []

        for subtask in eval_sequence:
            lang = val_annotations[subtask][0]
            ok = _rollout_subtask(
                env, policy, task_oracle, subtask, lang,
                args.ep_len, args.infer_chunk,
                static_frames, gripper_frames,
            )
            if ok:
                success_counter += 1
            else:
                break

        results.append(success_counter)

        running = _count_success(results)
        logging.info(
            f"seq {seq_idx}: solved {success_counter}/5 | "
            + " | ".join(f"{i+1}+:{v*100:.1f}%" for i, v in enumerate(running))
        )

        if save_this and static_frames:
            tag = f"seq{seq_idx:04d}_solved{success_counter}"
            imageio.mimwrite(
                video_root / f"{tag}_static.mp4",
                [np.asarray(x) for x in static_frames],
                fps=30, codec="libx264",
            )
            imageio.mimwrite(
                video_root / f"{tag}_gripper.mp4",
                [np.asarray(x) for x in gripper_frames],
                fps=30, codec="libx264",
            )

    # Final per-worker summary (aggregation across workers is done in eval_calvin_all)
    avg = float(np.mean(results)) if results else 0.0
    chain = _count_success(results)
    logging.info(f"Average solved length: {avg:.3f}")
    for i, sr in enumerate(chain):
        logging.info(f"SR (>= {i+1} subtasks): {sr*100:.1f}%")
    print(f"[{tag}] avg={avg:.3f} | " + " | ".join(f"{i+1}+:{v*100:.1f}%" for i, v in enumerate(chain)))

    # Return the per-sequence counts so the parent can aggregate across workers.
    return results


def eval_calvin_batched(args: Args, sequence_ids: Optional[list] = None) -> list:
    """
    Single-process, N parallel envs with batched policy inference.
    Each env independently runs its sequence; slots are refilled from the
    pool as they finish, so the batch size stays at N until the tail.
    """
    np.random.seed(args.seed)

    log_dir = pathlib.Path(f"{args.save_videos_root}/log/eval_results/{args.exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    range_tag = (f"seq_{sequence_ids[0]}-{sequence_ids[-1]}"
                 if sequence_ids else "all")
    log_file = log_dir / f"{args.task_suite_name}_{range_tag}.log"

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    video_root = pathlib.Path(f"{args.save_videos_root}/{args.exp_name}/videos/{args.task_suite_name}")
    video_root.mkdir(parents=True, exist_ok=True)

    # Policy first (HF from_pretrained peaks in CPU RAM before envs alloc).
    try:
        if args.model_type == "gr00tn15":
            policy = Gr00tCalvinInference(args.pretrained_model_path, args.infer_chunk)
        else:
            raise ValueError(f"Unsupported model_type: {args.model_type}")
        logging.info(f"Loaded {args.model_type} policy from {args.pretrained_model_path}")
    except Exception as e:
        logging.error(f"Failed to load policy: {e}")
        return []

    n_envs = max(1, args.n_envs)
    logging.info(f"Building {n_envs} parallel envs (scene={args.scene})")
    envs = []
    for i in range(n_envs):
        envs.append(_load_calvin_env(args.dataset_path or None, scene=args.scene))
        logging.info(f"  Env {i+1}/{n_envs} ready")

    task_oracle, val_annotations = _load_task_oracle_and_annotations(args.calvin_models_root)
    eval_sequences = _get_eval_sequences(args.num_sequences)
    if sequence_ids is not None:
        eval_sequences = [eval_sequences[i] for i in sequence_ids]
    n_total = len(eval_sequences)

    # Per-slot rollout state
    slot_state = [None] * n_envs
    pending_results = {}     # local_seq_idx -> success_count
    next_seq_idx = [0]       # boxed so the closure can mutate

    def assign_next_to_slot(slot_idx: int) -> bool:
        if next_seq_idx[0] >= n_total:
            return False
        seq_idx = next_seq_idx[0]
        next_seq_idx[0] += 1
        initial_state, eval_sequence = eval_sequences[seq_idx]
        robot_obs, scene_obs = _get_env_state_for_initial_condition(initial_state)
        envs[slot_idx].reset(robot_obs=robot_obs, scene_obs=scene_obs)
        slot_state[slot_idx] = {
            "seq_idx": seq_idx,
            "eval_sequence": eval_sequence,
            "subtask_idx": 0,
            "start_info": envs[slot_idx].get_info(),
            "success_count": 0,
            "subtask_step": 0,
            "static_frames": [],
            "gripper_frames": [],
            "action_chunk": None,
            "chunk_idx": args.infer_chunk,   # forces fresh policy call on first tick
            "save_this": (seq_idx % max(args.save_video_every, 1) == 0),
        }
        return True

    for i in range(n_envs):
        assign_next_to_slot(i)

    pbar = tqdm.tqdm(total=n_total, desc=f"seqs (B={n_envs})")

    while any(s is not None for s in slot_state):
        # Phase 1: refill action chunks for slots that need a new one (batched).
        need_chunk = [i for i in range(n_envs)
                      if slot_state[i] is not None
                      and slot_state[i]["chunk_idx"] >= args.infer_chunk]
        if need_chunk:
            obs_list = [envs[i].get_obs() for i in need_chunk]
            lang_list = [
                val_annotations[
                    slot_state[i]["eval_sequence"][slot_state[i]["subtask_idx"]]
                ][0]
                for i in need_chunk
            ]
            action_chunks = policy.get_calvin_actions_batched(obs_list, lang_list)
            for k, i in enumerate(need_chunk):
                slot_state[i]["action_chunk"] = action_chunks[k]
                slot_state[i]["chunk_idx"] = 0

        # Phase 2: step every active slot once.
        for i in range(n_envs):
            s = slot_state[i]
            if s is None:
                continue
            act = s["action_chunk"][s["chunk_idx"]]
            s["chunk_idx"] += 1
            obs, _, _, info = envs[i].step(act)
            s["subtask_step"] += 1
            s["static_frames"].append(to_video_frame(obs["rgb_obs"]["rgb_static"]))
            s["gripper_frames"].append(to_video_frame(obs["rgb_obs"]["rgb_gripper"]))

            subtask = s["eval_sequence"][s["subtask_idx"]]
            completed = task_oracle.get_task_info_for_set(s["start_info"], info, {subtask})

            slot_done = False
            if len(completed) > 0:
                s["success_count"] += 1
                s["subtask_idx"] += 1
                if s["subtask_idx"] >= 5:
                    slot_done = True
                else:
                    # Move on to next subtask in the same sequence
                    s["subtask_step"] = 0
                    s["start_info"] = info
                    s["chunk_idx"] = args.infer_chunk  # fresh policy call with new lang
            elif s["subtask_step"] >= args.ep_len:
                slot_done = True   # subtask timed out → whole sequence ends here

            if slot_done:
                pending_results[s["seq_idx"]] = s["success_count"]
                pbar.update(1)
                running_avg = float(np.mean(list(pending_results.values())))
                pbar.set_postfix(avg=f"{running_avg:.2f}")

                if s["save_this"] and s["static_frames"]:
                    vtag = f"seq{s['seq_idx']:04d}_solved{s['success_count']}"
                    imageio.mimwrite(
                        video_root / f"{vtag}_static.mp4",
                        [np.asarray(x) for x in s["static_frames"]],
                        fps=30, codec="libx264",
                    )
                    imageio.mimwrite(
                        video_root / f"{vtag}_gripper.mp4",
                        [np.asarray(x) for x in s["gripper_frames"]],
                        fps=30, codec="libx264",
                    )

                if not assign_next_to_slot(i):
                    slot_state[i] = None

    pbar.close()

    # Return in submission order
    results = [pending_results[i] for i in range(n_total)]

    avg = float(np.mean(results)) if results else 0.0
    chain = _count_success(results)
    logging.info(f"Average solved length: {avg:.3f}")
    for i, sr in enumerate(chain):
        logging.info(f"SR (>= {i+1} subtasks): {sr*100:.1f}%")
    print(f"[{range_tag}] avg={avg:.3f} | "
          + " | ".join(f"{i+1}+:{v*100:.1f}%" for i, v in enumerate(chain)))

    return results


def _run_debug_smoke_test(args: Args) -> None:
    """
    Step the CALVIN env with dummy actions for N steps. No policy, no oracle,
    no calvin_models_root required. Useful to confirm the env builds, the
    camera renders, and the relative-action interface is wired up.
    """
    print("=" * 80)
    print(f"CALVIN debug smoke test (scene={args.scene}, {args.debug_steps} steps)")
    print("=" * 80)

    env = _load_calvin_env(args.dataset_path or None, scene=args.scene)
    obs = env.reset()
    static_frames, gripper_frames = [], []

    for _ in tqdm.tqdm(range(args.debug_steps)):
        obs, _, _, _ = env.step(CALVIN_DUMMY_ACTION)
        static_frames.append(to_video_frame(obs["rgb_obs"]["rgb_static"]))
        gripper_frames.append(to_video_frame(obs["rgb_obs"]["rgb_gripper"]))

    video_root = pathlib.Path(f"{args.save_videos_root}/{args.exp_name}/debug")
    video_root.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(video_root / "debug_static.mp4",
                     [np.asarray(x) for x in static_frames], fps=30, codec="libx264")
    imageio.mimwrite(video_root / "debug_gripper.mp4",
                     [np.asarray(x) for x in gripper_frames], fps=30, codec="libx264")

    robot = np.asarray(obs["robot_obs"], dtype=np.float32)
    scene = np.asarray(obs["scene_obs"], dtype=np.float32)
    print(f"\nFinal robot_obs (15): {robot}")
    print(f"Final scene_obs (24): {scene}")
    print(f"Static frame shape: {static_frames[-1].shape} dtype={static_frames[-1].dtype}")
    print(f"Gripper frame shape: {gripper_frames[-1].shape} dtype={gripper_frames[-1].dtype}")
    print(f"\nVideos saved to: {video_root}/")


def eval_calvin_all(args: Args):
    if args.debug:
        _run_debug_smoke_test(args)
        return

    print("=" * 80)
    print("CALVIN LH-MTLC Evaluation (ABCD-D)")
    print("=" * 80)

    # Split the 1000 eval sequences across parallel workers. Tune via
    # --args.n_workers; each worker spawns its own policy + env.
    n_workers = max(1, args.n_workers)
    total = args.num_sequences
    chunks = np.array_split(np.arange(total), n_workers)
    sequence_splits = [list(map(int, c)) for c in chunks if len(c) > 0]

    per_worker = dict()
    # Pick the per-worker function once. Batched path uses N parallel envs
    # in a single process with batched policy inference; otherwise the
    # legacy sequential path.
    worker_fn = eval_calvin_batched if max(1, args.n_envs) > 1 else eval_calvin
    if n_workers == 1:
        # Single-worker: run in-process so tracebacks surface and we don't
        # pay the cost of re-importing GR00T/TF in a spawned child.
        ids = sequence_splits[0]
        label = f"seq_{ids[0]}-{ids[-1]}"
        per_worker[label] = worker_fn(args, ids) or []
        print(f"[DONE] {label} ({len(per_worker[label])} sequences)")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(sequence_splits), mp_context=ctx) as pool:
            futures = {
                pool.submit(worker_fn, args, ids): ids for ids in sequence_splits
            }
            for fut in as_completed(futures):
                ids = futures[fut]
                label = f"seq_{ids[0]}-{ids[-1]}"
                try:
                    per_worker[label] = fut.result() or []
                    print(f"[DONE] {label} ({len(per_worker[label])} sequences)")
                except Exception as e:
                    print(f"[ERROR] {label} failed: {e}")
                    per_worker[label] = []

    # Aggregate across all workers
    all_results = [r for lst in per_worker.values() for r in lst]
    if not all_results:
        print("No results collected.")
        return

    avg = float(np.mean(all_results))
    chain = _count_success(all_results)
    n = len(all_results)

    summary = (
        "\n" + "=" * 60 + "\n"
        f"CALVIN LH-MTLC results over {n} sequences\n"
        + "-" * 60 + "\n"
        + f"{'task 1':>8} {'task 2':>8} {'task 3':>8} {'task 4':>8} {'task 5':>8} {'avg len':>10}\n"
        + f"{chain[0]*100:>7.1f}% {chain[1]*100:>7.1f}% {chain[2]*100:>7.1f}% "
          f"{chain[3]*100:>7.1f}% {chain[4]*100:>7.1f}% {avg:>10.3f}\n"
        + "=" * 60
    )
    print(summary)

    # Persist combined summary
    out_dir = pathlib.Path(f"{args.save_videos_root}/log/eval_results/{args.exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.task_suite_name}_SUMMARY.txt", "w") as f:
        f.write(summary + "\n")
    import json
    with open(out_dir / f"{args.task_suite_name}_SUMMARY.json", "w") as f:
        json.dump({
            "num_sequences": n,
            "avg_seq_len": avg,
            "task_1": chain[0], "task_2": chain[1], "task_3": chain[2],
            "task_4": chain[3], "task_5": chain[4],
        }, f, indent=2)
    print(f"Summary saved to: {out_dir}/{args.task_suite_name}_SUMMARY.{{txt,json}}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    tyro.cli(eval_calvin_all)


'''
Example usage (single process):

python evaluation/eval_calvin.py \
    --args.exp_name calvin_abcd_d_test \
    --args.pretrained_model_path /path/to/checkpoint/pretrained_model \
    --args.dataset_path /mnt/data/sftp/data/locht1/calvin/dataset/task_ABCD_D \
    --args.calvin_models_root /mnt/data/sftp/data/locht1/calvin/calvin_models \
    --args.num_sequences 100
'''
