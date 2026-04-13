"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import time

import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, TASK_MAPPING

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        0
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def make_diverse_libero_env(
    task,
    resolution=256,
    x_range=(-0.25, 0.25),
    y_range=(-0.25, 0.25),
):
    """
    Creates a LIBERO environment with uniformly random movable-object placement.

    Fixtures (counters, shelves, containers) are still placed per the BDDL file.
    Movable objects each get an independent UniformRandomSampler over (x_range, y_range)
    with full SO(2) rotation.  Call env.reset() each episode — do NOT call
    set_init_state(), which bypasses the placement initializer entirely.

    Args:
        task: LIBERO task object (from benchmark.get_task).
        resolution: Camera image resolution in pixels.
        x_range: (low, high) metres, relative to table centre, for object x placement.
        y_range: (low, high) metres, relative to table centre, for object y placement.

    Returns:
        (env, task_description)
    """
    import libero.libero.envs.bddl_utils as BDDLUtils
    from copy import copy as _copy
    from robosuite.utils.errors import RandomizationError
    from robosuite.utils.transform_utils import quat_multiply
    from robosuite.utils.placement_samplers import (
        SequentialCompositeSampler,
        UniformRandomSampler,
    )
    from libero.libero.envs.regions import SiteSequentialCompositeSampler

    # ------------------------------------------------------------------
    # Custom sampler: adds a clearance margin to the non-overlap check so
    # objects with small horizontal_radius values don't visually intersect.
    # ------------------------------------------------------------------
    class _MarginSampler(UniformRandomSampler):
        def __init__(self, *args, margin=0.03, **kwargs):
            self._margin = margin
            super().__init__(*args, **kwargs)

        def sample(self, fixtures=None, reference=None, on_top=True):
            placed_objects = {} if fixtures is None else _copy(fixtures)

            if reference is None:
                base_offset = self.reference_pos
            elif isinstance(reference, str):
                assert reference in placed_objects, (
                    f"Reference '{reference}' not found in placed objects."
                )
                ref_pos, _, ref_obj = placed_objects[reference]
                base_offset = np.array(ref_pos)
                if on_top:
                    base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
            else:
                base_offset = np.array(reference)

            for obj in self.mujoco_objects:
                assert obj.name not in placed_objects, (
                    f"Object '{obj.name}' has already been sampled!"
                )
                r = obj.horizontal_radius
                bottom_offset = obj.bottom_offset
                success = False

                for _ in range(5000):
                    object_x = self._sample_x(r) + base_offset[0]
                    object_y = self._sample_y(r) + base_offset[1]
                    object_z = self.z_offset + base_offset[2]
                    if on_top:
                        object_z -= bottom_offset[-1]

                    location_valid = True
                    if self.ensure_valid_placement:
                        for (x, y, z), _, other_obj in placed_objects.values():
                            # Add margin as extra clearance between surfaces
                            min_dist = other_obj.horizontal_radius + r + self._margin
                            z_ok = (
                                object_z - z
                                <= other_obj.top_offset[-1] - bottom_offset[-1]
                            )
                            if (
                                np.linalg.norm((object_x - x, object_y - y)) <= min_dist
                                and z_ok
                            ):
                                location_valid = False
                                break

                    if location_valid:
                        quat = self._sample_quat()
                        if hasattr(obj, "init_quat"):
                            quat = quat_multiply(quat, obj.init_quat)
                        placed_objects[obj.name] = (
                            (object_x, object_y, object_z),
                            quat,
                            obj,
                        )
                        success = True
                        break

                if not success:
                    raise RandomizationError("Cannot place all objects ):")

            return placed_objects

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    problem_info = BDDLUtils.get_problem_info(task_bddl_file)
    problem_name = problem_info["problem_name"]
    base_class = TASK_MAPPING[problem_name]

    _x_range = list(x_range)
    _y_range = list(y_range)

    def _setup_diverse_placement_initializer(self, mujoco_arena):
        # Run the original BDDL-driven setup so fixtures and conditional samplers
        # (objects-on-sites, objects-on-objects) are handled correctly.
        base_class._setup_placement_initializer(self, mujoco_arena)

        # Rebuild the main placement_initializer:
        #   • keep any sampler whose objects are all fixtures (identity check)
        #   • replace movable-object samplers with wide UniformRandomSamplers
        fixture_obj_set = set(self.fixtures_dict.values())

        # Collect objects already claimed by conditional samplers so we don't
        # double-sample them in the main placement_initializer.
        conditional_obj_names = set()
        for cond_init in (
            self.conditional_placement_initializer,
            self.conditional_placement_on_objects_initializer,
        ):
            for sampler in cond_init.samplers.values():
                for obj in sampler.mujoco_objects:
                    conditional_obj_names.add(obj.name)

        new_main = SequentialCompositeSampler(name="ObjectSampler")
        # Keep fixture-only samplers from the BDDL setup.
        for sampler in self.placement_initializer.samplers.values():
            if all(obj in fixture_obj_set for obj in sampler.mujoco_objects):
                new_main.append_sampler(sampler)

        # Add a random sampler for every movable object NOT handled conditionally.
        for obj_name, obj in self.objects_dict.items():
            if obj.name in conditional_obj_names:
                continue
            # Use the object's own rotation_axis so objects that need a specific
            # axis to stand upright (e.g. cans whose XML default is lying down)
            # are kept correctly oriented.  Always use full (-π, π) range so
            # yaw is fully randomised regardless of what the BDDL specifies.
            rot_axis = getattr(obj, "rotation_axis", None) or "z"
            new_main.append_sampler(
                _MarginSampler(
                    name=f"{obj_name}_sampler",
                    mujoco_objects=obj,
                    x_range=_x_range,
                    y_range=_y_range,
                    rotation=(-np.pi, np.pi),
                    rotation_axis=rot_axis,
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.workspace_offset,
                    margin=0.03,  # 3 cm clearance between object surfaces
                )
            )

        self.placement_initializer = new_main

    RandomInitClass = type(
        f"RandomInit{base_class.__name__}",
        (base_class,),
        {
            "_setup_placement_initializer": _setup_diverse_placement_initializer,
            # The BDDL check compares self.__class__.__name__.lower() to the problem
            # name, which fails for our dynamically renamed subclass.  Override to
            # delegate to the base class name instead.
            "_assert_problem_name": lambda self: None,
        },
    )

    # Temporarily patch TASK_MAPPING so OffScreenRenderEnv picks up the subclass.
    TASK_MAPPING[problem_name] = RandomInitClass
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
        )
    finally:
        TASK_MAPPING[problem_name] = base_class  # always restore

    return env, task.language


def rotate_scene_init_state(env, initial_state, angle=None, n_settle=30):
    """
    Load a benchmark initial state and rotate the entire scene by one random
    angle around the table centre.

    All movable objects are treated as a single rigid body that spins in place:
      • XY positions are rotated around (workspace_offset.x, workspace_offset.y)
      • The same yaw is applied to every object's quaternion

    After placement, `n_settle` dummy-action physics steps are run internally so
    any small penetrations caused by rotated objects intersecting fixed fixtures
    are resolved quietly before the observation is returned.

    Args:
        env:           OffScreenRenderEnv returned by get_libero_env.
        initial_state: one saved state from task_suite.get_task_init_states().
        angle:         rotation in radians.  None → uniform sample from (−π, π).
        n_settle:      number of dummy-action steps to run for physics settling.

    Returns:
        obs dict after rotation and settling.
    """
    env.set_init_state(initial_state)

    if angle is None:
        # Safe rotations -30..30 step 5°
        angle = np.random.choice(list(range(-30, 31, 5))) * np.pi / 180

    sim = env.env.sim
    objects_dict = env.env.objects_dict

    # Table centre XY — workspace_offset is (cx, cy, table_z)
    cx = float(env.env.workspace_offset[0])
    cy = float(env.env.workspace_offset[1])

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rw    = np.cos(angle / 2)   # Z-rotation quaternion r = (rw, 0, 0, rz)
    rz    = np.sin(angle / 2)

    import mujoco as _mj

    # Objects whose centre Z is more than this above the table surface are
    # "elevated" (on a shelf / inside a fixture). Rotating their XY would move
    # them off their support, so only yaw is changed for them.
    table_z       = float(env.env.workspace_offset[2])
    Z_ON_TABLE    = 0.12  # metres above table_z → still "on table"
    Z_FALL_THRESH = 0.05  # metres of Z drop during settling → "fell over"

    # Table XY bounds — clamp rotated positions so no object leaves the workspace.
    # table_full_size = (x_len, y_len, thickness); workspace centre = (cx, cy).
    table_full_size = getattr(env.env, "table_full_size", (1.0, 1.2, 0.05))
    x_min = cx - table_full_size[0] / 2
    x_max = cx + table_full_size[0] / 2
    y_min = cy - table_full_size[1] / 2
    y_max = cy + table_full_size[1] / 2

    # Save benchmark qpos for every object BEFORE any rotation.
    # Used to restore fallen objects with guaranteed-zero velocity.
    benchmark_qpos = {
        name: sim.data.get_joint_qpos(obj.joints[-1]).copy()
        for name, obj in objects_dict.items()
    }

    # Apply rotation: XY+yaw for table-level objects, yaw-only for elevated ones.
    for obj_name, obj in objects_dict.items():
        q = sim.data.get_joint_qpos(obj.joints[-1]).copy()

        if (q[2] - table_z) <= Z_ON_TABLE:
            dx, dy = q[0] - cx, q[1] - cy
            new_x = cx + cos_a * dx - sin_a * dy
            new_y = cy + sin_a * dx + cos_a * dy
            # Clamp to keep the object's body fully inside the table surface.
            r = getattr(obj, "horizontal_radius", 0.0)
            q[0] = float(np.clip(new_x, x_min + r, x_max - r))
            q[1] = float(np.clip(new_y, y_min + r, y_max - r))

        ow, ox, oy, oz = q[3], q[4], q[5], q[6]
        q[3] = rw * ow - rz * oz
        q[4] = rw * ox - rz * oy
        q[5] = rw * oy + rz * ox
        q[6] = rw * oz + rz * ow

        sim.data.set_joint_qpos(obj.joints[-1], q)

    sim.forward()

    # Record XY and Z right after rotation (before any physics), then settle.
    xy_pre_settle = {
        name: sim.data.get_joint_qpos(obj.joints[-1])[:2].copy()
        for name, obj in objects_dict.items()
    }
    z_pre_settle = {
        name: sim.data.get_joint_qpos(obj.joints[-1])[2]
        for name, obj in objects_dict.items()
    }

    _dummy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    for _ in range(n_settle):
        env.env.step(_dummy)

    # Flag any object that:
    #   • fell (Z dropped more than Z_FALL_THRESH), OR
    #   • was pushed sideways > XY_PUSH_THRESH by the robot/fixture (table-level only).
    # Elevated objects (on stove, shelf, cabinet) are excluded from the XY check because
    # they naturally drift slightly on their support surface during settling.
    XY_PUSH_THRESH = 0.04   # 4 cm — robot body or fixture collision moved the object

    any_restored = False
    for obj_name, obj in objects_dict.items():
        post_q      = sim.data.get_joint_qpos(obj.joints[-1])
        is_elevated = (z_pre_settle[obj_name] - table_z) > Z_ON_TABLE
        fell        = (z_pre_settle[obj_name] - post_q[2]) > Z_FALL_THRESH
        pushed      = (
            not is_elevated
            and np.linalg.norm(post_q[:2] - xy_pre_settle[obj_name]) > XY_PUSH_THRESH
        )

        if fell or pushed:
            sim.data.set_joint_qpos(obj.joints[-1], benchmark_qpos[obj_name])
            jid = _mj.mj_name2id(
                sim.model._model,
                _mj.mjtObj.mjOBJ_JOINT,
                obj.joints[-1],
            )
            adr = sim.model._model.jnt_dofadr[jid]
            sim.data._data.qvel[adr : adr + 6] = 0.0
            any_restored = True

    if any_restored:
        sim.forward()

    env.env._post_process()
    env.env._update_observables(force=True)
    return env.env._get_observations(), any_restored


def perturb_init_state(env, initial_state, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1), margin=0.04):
    """
    Load a benchmark initial state then randomly shift each movable object's XY position.

    Orientation (quaternion) and Z are preserved exactly from the saved state, so objects
    remain upright and physically stable — no physics settling required.

    Non-overlap is enforced: each object is checked against all already-perturbed objects
    with a minimum surface-to-surface clearance of `margin` metres.  If no valid position
    is found within 5000 tries the object keeps its original benchmark position.

    Args:
        env: OffScreenRenderEnv returned by get_libero_env.
        initial_state: one saved state from task_suite.get_task_init_states().
        x_range: (low, high) random XY offset in metres relative to original position.
        y_range: (low, high) random XY offset in metres relative to original position.
        margin: minimum gap between object bounding circles (metres).

    Returns:
        obs dict after perturbation.
    """
    # Restore the benchmark state so orientations/Z are exactly correct.
    env.set_init_state(initial_state)

    sim = env.env.sim
    objects_dict = env.env.objects_dict
    fixtures_dict = env.env.fixtures_dict

    # --- Build obstacle map ------------------------------------------------
    # Start with fixtures: their XY positions are fixed and must never be
    # penetrated.  Read from body_xpos so we use the actual sim state.
    placed = {}
    for fname, fixture in fixtures_dict.items():
        try:
            body_id = sim.model.body_name2id(fixture.root_body)
            pos = sim.data.body_xpos[body_id]
            r = getattr(fixture, "horizontal_radius", 0.10)
            placed[fname] = (float(pos[0]), float(pos[1]), float(r))
        except Exception:
            pass  # skip fixtures we can't locate

    # Add each movable object at its current (benchmark) position.
    orig_qpos = {}
    for obj_name, obj in objects_dict.items():
        q = sim.data.get_joint_qpos(obj.joints[-1]).copy()
        orig_qpos[obj_name] = q
        placed[obj_name] = (float(q[0]), float(q[1]), obj.horizontal_radius)
    # -----------------------------------------------------------------------

    new_qpos = {name: q.copy() for name, q in orig_qpos.items()}

    for obj_name, obj in objects_dict.items():
        base = orig_qpos[obj_name]
        r = obj.horizontal_radius
        # Check against everything except this object itself.
        others = {k: v for k, v in placed.items() if k != obj_name}

        success = False
        for _ in range(5000):
            nx = base[0] + np.random.uniform(*x_range)
            ny = base[1] + np.random.uniform(*y_range)

            if all(
                np.linalg.norm([nx - ox, ny - oy]) >= r + or_ + margin
                for _, (ox, oy, or_) in others.items()
            ):
                new_qpos[obj_name][0] = nx
                new_qpos[obj_name][1] = ny
                placed[obj_name] = (nx, ny, r)
                success = True
                break

        if not success:
            # Keep original position; update placed so later objects check against it.
            placed[obj_name] = (float(base[0]), float(base[1]), r)

    # Apply a random yaw (Z-axis only) to every object.
    # r = (rw, 0, 0, rz)  →  result = r ⊗ orig (standard quat product, w-first)
    #   c0 = rw·ow − rz·oz
    #   c1 = rw·ox − rz·oy   ← note: −rz (not +rz)
    #   c2 = rw·oy + rz·ox   ← note: +rz (not −rz)
    #   c3 = rw·oz + rz·ow
    # With rx=ry=0, only yaw changes; pitch/roll from the benchmark are kept.
    for obj_name in objects_dict:
        q = new_qpos[obj_name]          # [x, y, z, qw, qx, qy, qz]
        theta = np.random.uniform(-np.pi, np.pi)
        rw = np.cos(theta / 2)
        rz = np.sin(theta / 2)
        ow, ox, oy, oz = q[3], q[4], q[5], q[6]
        q[3] = rw * ow - rz * oz
        q[4] = rw * ox - rz * oy   # corrected sign
        q[5] = rw * oy + rz * ox   # corrected sign
        q[6] = rw * oz + rz * ow

    # Write all new positions back in one pass.
    for obj_name, obj in objects_dict.items():
        sim.data.set_joint_qpos(obj.joints[-1], new_qpos[obj_name])

    sim.forward()
    env.env._post_process()
    env.env._update_observables(force=True)
    return env.env._get_observations()


def load_rotated_states(task_suite_name: str, task_id: int, rotated_states_dir: str = None):
    """
    Load pre-baked rotated initial states generated by generate_rotated_states.py.

    Args:
        task_suite_name: e.g. "libero_spatial"
        task_id: integer task index within the suite
        rotated_states_dir: root directory containing the suite subfolder.
                            Defaults to ./rotated_states/<task_suite_name>/

    Returns:
        states: np.ndarray of shape (N, qpos_dim) — each row is a flat MuJoCo state
        meta:   np.ndarray of shape (N, 2) — columns are (angle_deg, orig_state_idx)

    Raises:
        FileNotFoundError if the .npy files do not exist (run generate_rotated_states.py first).
    """
    if rotated_states_dir is None:
        rotated_states_dir = os.path.join("rotated_states", task_suite_name)

    prefix = os.path.join(rotated_states_dir, f"task_{task_id:03d}")
    states_path = f"{prefix}_states.npy"
    meta_path   = f"{prefix}_meta.npy"

    if not os.path.exists(states_path):
        raise FileNotFoundError(
            f"Pre-baked rotated states not found at '{states_path}'.\n"
            f"Run: python generate_rotated_states.py --task_suite_name {task_suite_name}"
        )

    states = np.load(states_path)
    meta   = np.load(meta_path)
    return states, meta


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    wrist_img = obs["robot0_eye_in_hand_image"]
    wrist_img = wrist_img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing

    return img, wrist_img


def save_rollout_video(top_view, wrist_view, idx, success, task_description, prefix_name, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts_{prefix_name}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img1, img2 in zip(top_view, wrist_view):
        combined = np.hstack((img1, img2))
        video_writer.append_data(combined)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [+1,-1].

    Normalization formula: y = 1 - 2 * (x - orig_low) / (orig_high - orig_low)
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action
