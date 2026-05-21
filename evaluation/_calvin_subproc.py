# Subprocess-side worker + pool for CALVIN parallel envs.
#
# Kept in its own module so that spawning a child process does NOT re-import
# gr00t / TensorFlow / torch — only calvin_env + hydra (which are imported
# lazily inside `_load_calvin_env`). This keeps per-worker RAM around the
# size of one PyBullet env (~300-500 MB) instead of ~3 GB.

from __future__ import annotations

import multiprocessing as mp
import pathlib
import sys
from typing import List, Optional


def _load_calvin_env(dataset_path: Optional[str] = None,
                     scene: str = "calvin_scene_D"):
    """Build a CALVIN env. Imports calvin_env / hydra lazily on first call."""
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import calvin_env
    from calvin_env.envs.play_table_env import get_env

    if dataset_path is not None:
        cfg_path = (pathlib.Path(dataset_path) / "validation" / ".hydra"
                    / "merged_config.yaml")
        if cfg_path.is_file():
            return get_env(pathlib.Path(dataset_path) / "validation",
                           show_gui=False)

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


def _calvin_env_worker(child_pipe, scene: str,
                       dataset_path: Optional[str],
                       sys_path_extra: List[str]) -> None:
    """One CALVIN env in a subprocess. Pipe protocol: (cmd, payload) → (status, payload)."""
    for p in sys_path_extra:
        if p and p not in sys.path:
            sys.path.insert(0, p)

    try:
        env = _load_calvin_env(dataset_path or None, scene=scene)
    except Exception as e:
        try:
            child_pipe.send(("error", f"env init failed: {e!r}"))
        finally:
            child_pipe.close()
        return

    child_pipe.send(("ok", None))

    while True:
        try:
            cmd, payload = child_pipe.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            break
        try:
            if cmd == "reset":
                robot_obs, scene_obs = payload
                obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                info = env.get_info()
                child_pipe.send(("ok", (obs, info)))
            elif cmd == "step":
                obs, _, _, info = env.step(payload)
                child_pipe.send(("ok", (obs, info)))
            elif cmd == "close":
                child_pipe.send(("ok", None))
                break
            else:
                child_pipe.send(("error", f"unknown cmd: {cmd}"))
        except Exception as e:
            child_pipe.send(("error", repr(e)))

    try:
        child_pipe.close()
    except Exception:
        pass


class CalvinParallelEnvs:
    """
    Pool of N CALVIN envs, each in its own subprocess. Lockstep reset/step,
    one Pipe per worker. Caller manages episodes — no auto-reset.
    """

    def __init__(self, n_envs: int, scene: str,
                 dataset_path: Optional[str] = None):
        self.n_envs = n_envs
        sys_path_extra = list(sys.path)
        ctx = mp.get_context("spawn")
        self.parent_pipes = []
        self.processes = []
        for _ in range(n_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_calvin_env_worker,
                args=(child, scene, dataset_path, sys_path_extra),
                daemon=True,
            )
            p.start()
            child.close()
            self.parent_pipes.append(parent)
            self.processes.append(p)

        for i, pipe in enumerate(self.parent_pipes):
            try:
                status, payload = pipe.recv()
            except (EOFError, ConnectionResetError, OSError) as e:
                self.close()
                raise RuntimeError(
                    f"env worker {i} died during init "
                    f"(pipe {type(e).__name__}: {e}); "
                    f"check stderr above for the worker's traceback"
                ) from e
            if status != "ok":
                self.close()
                raise RuntimeError(f"env worker {i} failed to init: {payload}")

    def reset(self, indices, robot_obss, scene_obss):
        assert len(indices) == len(robot_obss) == len(scene_obss)
        for i, r, s in zip(indices, robot_obss, scene_obss):
            self.parent_pipes[i].send(("reset", (r, s)))
        out = []
        for i in indices:
            status, payload = self.parent_pipes[i].recv()
            if status != "ok":
                raise RuntimeError(f"env {i} reset failed: {payload}")
            out.append(payload)
        return out

    def step(self, indices, actions):
        assert len(indices) == len(actions)
        for i, a in zip(indices, actions):
            self.parent_pipes[i].send(("step", a))
        out = []
        for i in indices:
            status, payload = self.parent_pipes[i].recv()
            if status != "ok":
                raise RuntimeError(f"env {i} step failed: {payload}")
            out.append(payload)
        return out

    def close(self):
        for pipe in self.parent_pipes:
            try:
                pipe.send(("close", None))
                pipe.recv()
            except (BrokenPipeError, EOFError, ConnectionResetError, OSError):
                pass
            try:
                pipe.close()
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
