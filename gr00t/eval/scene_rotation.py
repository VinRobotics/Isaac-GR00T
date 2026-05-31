"""Free-object scene rotation around the world Z-axis for LIBERO environments.

Pure quaternion math (numpy only) + a thin LIBERO/robosuite glue layer.
Imported by `scripts/verify_rotation.py` for visual verification and (later)
by `evaluation/eval_libero.py` for controlled orientation-shift experiments.

Quaternion convention here is [w, x, y, z] to match MuJoCo's qpos layout.
This is intentionally different from `gr00t/model/common/rot_randomizer.py`
which uses [x, y, z, w] for action tensors. Do not confuse the two: this
module touches sim qpos directly, the other touches action vectors.
"""

import math
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------------
# Pure quaternion math (wxyz convention, MuJoCo native)
# ----------------------------------------------------------------------------

def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions. Returns shape (4,)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_axis_angle_z(theta: float) -> np.ndarray:
    """Unit quaternion [w, x, y, z] for a rotation of theta radians around +Z."""
    return np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion. Asserts the norm is already close to 1 first
    (guards against gross math bugs before silently absorbing them)."""
    n = float(np.linalg.norm(q))
    assert n > 1e-9, f"degenerate quaternion, norm={n}"
    assert abs(n - 1.0) < 1e-3, f"quaternion norm far from 1 before normalize: {n}"
    return q / n


def rotate_xy_about_z(xy: np.ndarray, theta: float, center: np.ndarray) -> np.ndarray:
    """Rotate a 2D point around `center` by `theta` radians (counter-clockwise
    when viewed from +Z, i.e. top-down)."""
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return center + R @ (xy - center)


def rotated_free_qpos7(qpos7: np.ndarray, theta: float, center_xy: np.ndarray) -> np.ndarray:
    """Rotate one free-joint qpos slot (7-vec [x, y, z, qw, qx, qy, qz]) by
    theta radians around the world Z-axis, about pivot `center_xy`.

    World-frame rotation -> LEFT-multiply the orientation quaternion.
    """
    assert qpos7.shape == (7,), f"expected shape (7,), got {qpos7.shape}"
    pos = qpos7[:3].copy()
    quat = qpos7[3:7].copy()
    pos[:2] = rotate_xy_about_z(pos[:2], theta, np.asarray(center_xy, dtype=pos.dtype))
    quat = quat_normalize(quat_mul_wxyz(quat_from_axis_angle_z(theta), quat))
    return np.concatenate([pos, quat])


# ----------------------------------------------------------------------------
# LIBERO / robosuite glue
# ----------------------------------------------------------------------------

_FREE_JOINT_TYPE = 0  # mjJNT_FREE in both mujoco-py and new mujoco bindings


def _joint_name(sim, j: int) -> str:
    """Resolve joint id -> name across mujoco-py / new mujoco binding."""
    m = sim.model
    if hasattr(m, "joint_id2name"):
        return m.joint_id2name(j)
    import mujoco  # new binding
    raw = getattr(m, "_model", m)
    return mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_JOINT, j)


def _body_name(sim, b: int) -> str:
    m = sim.model
    if hasattr(m, "body_id2name"):
        return m.body_id2name(b)
    import mujoco
    raw = getattr(m, "_model", m)
    return mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_BODY, b)


def _body_name_to_id(sim, name: str) -> int:
    m = sim.model
    if hasattr(m, "body_name2id"):
        try:
            return m.body_name2id(name)
        except (ValueError, KeyError):
            return -1
    import mujoco
    raw = getattr(m, "_model", m)
    bid = mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_BODY, name)
    return int(bid)


def list_free_joints(
    sim,
    exclude_substrings: tuple = ("robot", "gripper", "mount", "hand"),
) -> list:
    """Return [(jnt_id, joint_name, qpos_addr, body_name), ...] for every free
    joint in the sim whose associated body name does NOT contain any of
    `exclude_substrings`. Filtering is defensive: in LIBERO scenes only
    objects on the table are expected, but mjcf compositions can occasionally
    introduce a free joint elsewhere.
    """
    m = sim.model
    out = []
    for j in range(int(m.njnt)):
        if int(m.jnt_type[j]) != _FREE_JOINT_TYPE:
            continue
        jname = _joint_name(sim, j)
        bid = int(m.jnt_bodyid[j])
        bname = _body_name(sim, bid)
        lower = (bname or "").lower() + " " + (jname or "").lower()
        if any(sub in lower for sub in exclude_substrings):
            continue
        adr = int(m.jnt_qposadr[j])
        out.append((j, jname, adr, bname))
    return out


def resolve_table_center(env) -> tuple:
    """Return (center_xy: np.ndarray(2,), source_tag: str).

    Three-tier fallback:
      1. inner robosuite env's `workspace_offset` (the LIBERO BDDLBaseDomain default)
      2. world position of a body literally named "table"
      3. centroid of free-object xy positions (with WARNING)
    """
    inner = getattr(env, "env", env)
    wso = getattr(inner, "workspace_offset", None)
    if wso is not None and len(wso) >= 2:
        return np.array([float(wso[0]), float(wso[1])]), "workspace_offset"

    sim = env.sim
    bid = _body_name_to_id(sim, "table")
    if bid >= 0:
        return np.array(sim.data.body_xpos[bid][:2], dtype=float), "body_table"

    free = list_free_joints(sim)
    if free:
        pts = np.stack([sim.data.qpos[adr:adr + 2] for _, _, adr, _ in free])
        c = pts.mean(axis=0).astype(float)
        return c, "centroid_FALLBACK"
    return np.zeros(2), "origin_FALLBACK"


def get_agentview_rgb(obs: dict) -> np.ndarray:
    """Return the agentview RGB frame with the [::-1, ::-1] flip applied,
    matching the convention used by `evaluation/eval_libero.py`."""
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def rotate_scene_z(
    env,
    theta_rad: float,
    pivot_xy: Optional[np.ndarray] = None,
    exclude_substrings: tuple = ("robot", "gripper", "mount", "hand"),
    return_obs: bool = True,
) -> Optional[dict]:
    """Rotate every free object in the scene around world Z by `theta_rad`,
    about pivot `pivot_xy` (default: resolved table center).

    After rewriting qpos, calls `sim.forward()` (kinematics only, NO physics
    step) and regenerates the observation dict so the next render reflects
    the new pose.

    Returns the fresh obs dict if `return_obs`, else None.
    """
    sim = env.sim
    if pivot_xy is None:
        pivot_xy, _ = resolve_table_center(env)
    pivot_xy = np.asarray(pivot_xy, dtype=float)

    qpos = sim.data.qpos
    free = list_free_joints(sim, exclude_substrings=exclude_substrings)
    for _, _, adr, _ in free:
        slot = np.asarray(qpos[adr:adr + 7], dtype=float).copy()
        new = rotated_free_qpos7(slot, theta_rad, pivot_xy)
        qpos[adr:adr + 7] = new

    sim.forward()

    if not return_obs:
        return None
    inner = getattr(env, "env", env)
    if hasattr(inner, "_update_observables"):
        inner._update_observables(force=True)
    return inner._get_observations()


# ----------------------------------------------------------------------------
# v2: rotate ROBOT + objects together via XML edit + init_state rewrite
# ----------------------------------------------------------------------------

def _parse_xml_vec(s: Optional[str], default) -> np.ndarray:
    if s is None or s.strip() == "":
        return np.asarray(default, dtype=float)
    return np.array([float(t) for t in s.split()], dtype=float)


def _fmt_xml_vec(v: np.ndarray) -> str:
    return " ".join(f"{x:.9g}" for x in v)


def discover_robot_mount_body(sim) -> str:
    """Return the name of the direct-worldbody-child body that is the robot's
    top-level subtree root. LIBERO + robosuite Panda mounts that body at
    'robot0_base' by default; we accept any direct child whose name starts
    with 'robot0' / 'mount' / 'Panda' / 'Rethink' for robustness.
    """
    m = sim.model
    cands = []
    for b in range(int(m.nbody)):
        parent = int(m.body_parentid[b])
        if parent != 0:  # only direct children of the worldbody (id 0)
            continue
        name = _body_name(sim, b) or ""
        low = name.lower()
        if low.startswith("robot") or "mount" in low or "panda" in low or "rethink" in low:
            cands.append(name)
    if not cands:
        raise RuntimeError("could not find robot mount body under worldbody; "
                           "direct worldbody children inspected")
    # Prefer the canonical 'robot0_base' if present, else first candidate.
    for preferred in ("robot0_base",):
        if preferred in cands:
            return preferred
    return cands[0]


def rotate_body_in_xml(
    xml_str: str,
    body_name: str,
    theta_rad: float,
    pivot_xy=(0.0, 0.0),
) -> str:
    """Return XML with body `body_name`'s pose rotated by theta_rad around
    world +Z about pivot_xy. Operates on a direct child of <worldbody>.
    Quaternion is MuJoCo wxyz.
    """
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_str)
    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError("XML has no <worldbody>")

    target = None
    for child in wb:
        if child.tag == "body" and child.get("name") == body_name:
            target = child
            break
    if target is None:
        raise RuntimeError(
            f"body name={body_name!r} not found as direct child of <worldbody>"
        )

    if target.get("axisangle") is not None or target.get("euler") is not None:
        raise RuntimeError(
            f"body {body_name} uses axisangle/euler; only pos+quat supported"
        )

    pivot = np.asarray(pivot_xy, dtype=float)
    pos = _parse_xml_vec(target.get("pos"), [0.0, 0.0, 0.0])
    quat = _parse_xml_vec(target.get("quat"), [1.0, 0.0, 0.0, 0.0])

    # Rotate xy of position around pivot; preserve z.
    new_xy = rotate_xy_about_z(pos[:2], theta_rad, pivot)
    new_pos = np.array([new_xy[0], new_xy[1], pos[2]])
    # World-frame rotation -> LEFT-multiply orientation quat.
    new_quat = quat_normalize(quat_mul_wxyz(quat_from_axis_angle_z(theta_rad), quat))

    target.set("pos", _fmt_xml_vec(new_pos))
    target.set("quat", _fmt_xml_vec(new_quat))

    return ET.tostring(root, encoding="unicode")


def rotate_camera_in_xml(
    xml_str: str,
    camera_name: str,
    theta_rad: float,
    pivot_xy=(0.0, 0.0),
) -> str:
    """Return XML with camera `camera_name` position and orientation rotated by
    theta_rad around world +Z about pivot_xy. Searches the full XML tree (not
    restricted to worldbody direct children). Handles quat, euler, axisangle,
    zaxis, xyaxes, and no-rotation-attr cases.
    """
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_str)
    pivot = np.asarray(pivot_xy, dtype=float)

    target = None
    for cam in root.iter("camera"):
        if cam.get("name") == camera_name:
            target = cam
            break
    if target is None:
        raise RuntimeError(f"camera {camera_name!r} not found in XML")

    # Rotate position xy around pivot; preserve z.
    pos = _parse_xml_vec(target.get("pos"), [0.0, 0.0, 0.0])
    new_xy = rotate_xy_about_z(pos[:2], theta_rad, pivot)
    target.set("pos", _fmt_xml_vec(np.array([new_xy[0], new_xy[1], pos[2]])))

    # Rotate orientation: left-multiply by Z-rotation.
    dq = quat_from_axis_angle_z(theta_rad)  # [w,x,y,z]
    Rz = np.array([
        [math.cos(theta_rad), -math.sin(theta_rad), 0.0],
        [math.sin(theta_rad),  math.cos(theta_rad), 0.0],
        [0.0,                  0.0,                 1.0],
    ])

    if target.get("quat") is not None:
        q = _parse_xml_vec(target.get("quat"), [1.0, 0.0, 0.0, 0.0])
        target.set("quat", _fmt_xml_vec(quat_normalize(quat_mul_wxyz(dq, q))))

    elif target.get("euler") is not None:
        # MuJoCo default euler sequence is XYZ (intrinsic).
        from scipy.spatial.transform import Rotation
        euler = _parse_xml_vec(target.get("euler"), [0.0, 0.0, 0.0])
        R_new = Rz @ Rotation.from_euler("xyz", euler).as_matrix()
        target.set("euler", _fmt_xml_vec(Rotation.from_matrix(R_new).as_euler("xyz")))

    elif target.get("axisangle") is not None:
        # MuJoCo axisangle = "ax ay az angle" (4 values).
        from scipy.spatial.transform import Rotation
        v = _parse_xml_vec(target.get("axisangle"), [0.0, 0.0, 1.0, 0.0])
        R_new = Rz @ Rotation.from_rotvec(v[:3] * v[3]).as_matrix()
        rv = Rotation.from_matrix(R_new).as_rotvec()
        angle = float(np.linalg.norm(rv))
        axis = rv / angle if angle > 1e-12 else np.array([0.0, 0.0, 1.0])
        target.set("axisangle", _fmt_xml_vec(np.append(axis, angle)))

    elif target.get("zaxis") is not None:
        z = _parse_xml_vec(target.get("zaxis"), [0.0, 0.0, -1.0])
        target.set("zaxis", _fmt_xml_vec(Rz @ z))
        if target.get("xyaxes") is not None:
            xy = _parse_xml_vec(target.get("xyaxes"), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            x_new = Rz @ xy[:3]
            y_new = Rz @ xy[3:]
            target.set("xyaxes", _fmt_xml_vec(np.concatenate([x_new, y_new])))

    elif target.get("xyaxes") is not None:
        xy = _parse_xml_vec(target.get("xyaxes"), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        target.set("xyaxes", _fmt_xml_vec(
            np.concatenate([Rz @ xy[:3], Rz @ xy[3:]])
        ))

    else:
        # No rotation attribute — camera default orientation = identity. Add quat.
        target.set("quat", _fmt_xml_vec(dq))

    return ET.tostring(root, encoding="unicode")


def rotate_init_state_qpos(
    init_state: np.ndarray,
    free_joint_qpos_addrs,
    theta_rad: float,
    pivot_xy,
    qpos_offset_in_state: int = 1,
) -> np.ndarray:
    """Return a copy of the flat init_state with every free-joint qpos slot
    (7-vec) rotated by theta_rad around world +Z about pivot_xy.

    The flat state layout produced by MjSim.get_state().flatten() is
    [time, qpos, qvel, act, ...]. Free-joint addresses come from
    `sim.model.jnt_qposadr` and index into the qpos sub-array, so we offset
    by `qpos_offset_in_state` (=1 by default for the leading time element).
    qvel is left untouched: rotating a static pose means new velocity = 0,
    and the init_states LIBERO ships have qvel=0 already.
    """
    pivot = np.asarray(pivot_xy, dtype=float)
    out = np.asarray(init_state, dtype=float).copy()
    for adr in free_joint_qpos_addrs:
        a = adr + qpos_offset_in_state
        slot = out[a:a + 7].copy()
        out[a:a + 7] = rotated_free_qpos7(slot, theta_rad, pivot)
    return out


def find_robot_base_yaw_joint(
    sim,
    mount_body_name: Optional[str] = None,
) -> tuple:
    """Locate the robot's base-yaw joint — the hinge joint that, when rotated,
    swings the entire arm around the world +Z axis without moving the mount.

    For Panda this is `robot0_joint1` (qpos_addr 0). We discover it by walking
    the robot subtree (rooted at mount_body_name) and returning the first
    hinge joint we encounter; on the Panda chain that is joint 1 by
    construction. Returns (joint_id, qpos_addr, joint_name).
    """
    if mount_body_name is None:
        mount_body_name = discover_robot_mount_body(sim)
    m = sim.model
    mount_id = _body_name_to_id(sim, mount_body_name)
    if mount_id < 0:
        raise RuntimeError(f"mount body {mount_body_name!r} not found in sim")

    def is_descendant(bid: int) -> bool:
        while bid > 0:
            if bid == mount_id:
                return True
            bid = int(m.body_parentid[bid])
        return False

    HINGE = 3  # mjtJoint.mjJNT_HINGE
    for j in range(int(m.njnt)):
        if int(m.jnt_type[j]) != HINGE:
            continue
        if not is_descendant(int(m.jnt_bodyid[j])):
            continue
        return j, int(m.jnt_qposadr[j]), _joint_name(sim, j)
    raise RuntimeError(
        f"no hinge joint found under robot subtree {mount_body_name!r}"
    )


def find_arm_qpos_addrs(
    sim,
    mount_body_name: Optional[str] = None,
    exclude_joint_substrings: tuple = ("finger", "gripper"),
) -> list:
    """All hinge-joint qpos addresses in the robot arm subtree, excluding
    gripper/finger joints. For Panda this returns [0..6] (the 7 arm hinges)."""
    if mount_body_name is None:
        mount_body_name = discover_robot_mount_body(sim)
    m = sim.model
    mount_id = _body_name_to_id(sim, mount_body_name)

    def is_descendant(bid: int) -> bool:
        while bid > 0:
            if bid == mount_id:
                return True
            bid = int(m.body_parentid[bid])
        return False

    HINGE = 3
    out = []
    for j in range(int(m.njnt)):
        if int(m.jnt_type[j]) != HINGE:
            continue
        if not is_descendant(int(m.jnt_bodyid[j])):
            continue
        jname = (_joint_name(sim, j) or "").lower()
        if any(sub in jname for sub in exclude_joint_substrings):
            continue
        out.append(int(m.jnt_qposadr[j]))
    return out


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def compute_ee_pose_match_init_state(
    env,
    init_state: np.ndarray,
    theta_rad: float,
    pivot_xy,
    ee_body_name: str = "gripper0_eef",
    mount_body_name: Optional[str] = None,
    max_nfev: int = 200,
) -> tuple:
    """Solve IK so the EE world pose matches what FULL mode would produce at the
    same theta — but with mount + scene XML unchanged. Modifies the arm joint
    angles in init_state and returns (rotated_state, info).

    The target EE pose is the baseline (theta=0) EE world pose rigidly rotated
    by theta around world +Z about pivot_xy. Arm joints (joints 1..7 for Panda)
    are solved via Levenberg-Marquardt least-squares against the 6-dim
    pos+orn residual. Joints are clamped to their MuJoCo `jnt_range` after the
    solve; if clamping fires for any joint, an `[IK-WARN]` line is printed.

    Returns
    -------
    new_state : np.ndarray
        init_state with arm-joint qpos slots overwritten by the IK solution.
    info : dict
        Keys: pos_err (m), orn_err (axis-angle rad), nfev, clamped (list of
        joint addrs that hit their range).
    """
    from scipy.optimize import least_squares

    sim = env.sim
    m = sim.model
    pivot = np.asarray(pivot_xy, dtype=float)

    ee_id = _body_name_to_id(sim, ee_body_name)
    if ee_id < 0:
        raise RuntimeError(f"EE body {ee_body_name!r} not found in sim")

    # Establish baseline EE pose by loading the un-rotated init_state.
    env.set_init_state(init_state)
    baseline_pos = sim.data.body_xpos[ee_id].copy()
    baseline_quat = sim.data.body_xquat[ee_id].copy()  # wxyz

    # Target = rigid rotation of baseline EE around world +Z about pivot.
    target_xy = rotate_xy_about_z(baseline_pos[:2], theta_rad, pivot)
    target_pos = np.array([target_xy[0], target_xy[1], baseline_pos[2]])
    target_quat = quat_normalize(
        quat_mul_wxyz(quat_from_axis_angle_z(theta_rad), baseline_quat)
    )

    arm_addrs = find_arm_qpos_addrs(sim, mount_body_name)
    if not arm_addrs:
        raise RuntimeError("no arm hinge joints found")
    q0 = np.array([float(init_state[a + 1]) for a in arm_addrs])

    # Joint ranges for clamping (look up joint id by qpos address).
    addr_to_jid = {}
    for j in range(int(m.njnt)):
        if int(m.jnt_qposadr[j]) in arm_addrs:
            addr_to_jid[int(m.jnt_qposadr[j])] = j
    jnt_ranges = np.array([m.jnt_range[addr_to_jid[a]] for a in arm_addrs])

    def residual(q):
        for qi, addr in zip(q, arm_addrs):
            sim.data.qpos[addr] = qi
        sim.forward()
        cur_pos = sim.data.body_xpos[ee_id]
        cur_quat = sim.data.body_xquat[ee_id]
        err_pos = target_pos - cur_pos
        d = quat_mul_wxyz(target_quat, _quat_conj_wxyz(cur_quat))
        if d[0] < 0:
            d = -d
        err_orn = 2.0 * d[1:4]
        return np.concatenate([err_pos, err_orn])

    # `trf` (trust region reflective) supports underdetermined systems
    # (6 residuals < 7 vars) and joint-range bounds natively. We still clamp
    # post-hoc for safety in case scipy returns a value slightly outside.
    lo = np.array([float(r[0]) for r in jnt_ranges])
    hi = np.array([float(r[1]) for r in jnt_ranges])
    # Clip the initial guess into bounds (least_squares requires it).
    q0_clipped = np.clip(q0, lo, hi)
    res = least_squares(
        residual, q0_clipped, method="trf", max_nfev=max_nfev,
        bounds=(lo, hi),
    )
    q_new = res.x
    final_err = residual(q_new)
    pos_err = float(np.linalg.norm(final_err[:3]))
    orn_err = float(np.linalg.norm(final_err[3:]))

    # Clamp into joint range.
    clamped = []
    for i, addr in enumerate(arm_addrs):
        lo, hi = float(jnt_ranges[i, 0]), float(jnt_ranges[i, 1])
        before = q_new[i]
        q_new[i] = max(lo, min(hi, before))
        if abs(q_new[i] - before) > 1e-6:
            clamped.append((addr, before, q_new[i]))

    if clamped:
        print(f"  [IK-WARN] joints clamped at theta={math.degrees(theta_rad):+.2f}deg: "
              f"{[(a, f'{b:.4f}->{c:.4f}') for a, b, c in clamped]}")

    new_state = np.asarray(init_state, dtype=float).copy()
    for qi, addr in zip(q_new, arm_addrs):
        new_state[addr + 1] = qi
    return new_state, {"pos_err": pos_err, "orn_err": orn_err,
                       "nfev": int(res.nfev), "clamped": clamped}


def find_robot_wrist_joint(
    sim,
    mount_body_name: Optional[str] = None,
) -> tuple:
    """Locate the robot's wrist twist joint — the *last* hinge joint in the
    robot subtree, which on a Panda is joint7 (the EE-yaw hinge). Rotating
    this joint twists the gripper about its approach axis without moving the
    arm or the mount.

    Returns (joint_id, qpos_addr, joint_name).
    """
    if mount_body_name is None:
        mount_body_name = discover_robot_mount_body(sim)
    m = sim.model
    mount_id = _body_name_to_id(sim, mount_body_name)
    if mount_id < 0:
        raise RuntimeError(f"mount body {mount_body_name!r} not found in sim")

    def is_descendant(bid: int) -> bool:
        while bid > 0:
            if bid == mount_id:
                return True
            bid = int(m.body_parentid[bid])
        return False

    HINGE = 3  # mjtJoint.mjJNT_HINGE
    for j in range(int(m.njnt) - 1, -1, -1):
        if int(m.jnt_type[j]) != HINGE:
            continue
        if not is_descendant(int(m.jnt_bodyid[j])):
            continue
        return j, int(m.jnt_qposadr[j]), _joint_name(sim, j)
    raise RuntimeError(
        f"no hinge joint found under robot subtree {mount_body_name!r}"
    )


# Supported rotation modes for reset_env_rotated. Both bake the rotation into
# the freshly-set init_state (no post-hoc qpos mutation). Named cameras
# (default: "agentview") are co-rotated with the scene via XML edit.
ROTATION_MODE_FULL = "full"          # rotate robot mount body + object qpos
ROTATION_MODE_EE_YAW = "ee_yaw"      # rotate object qpos + twist wrist joint
_VALID_ROTATION_MODES = (ROTATION_MODE_FULL, ROTATION_MODE_EE_YAW)


def _bump_joint_in_state(
    state: np.ndarray,
    qpos_addr: int,
    delta: float,
    jnt_range: tuple,
    qpos_offset_in_state: int = 1,
    sign: int = 1,
) -> tuple:
    """Add `sign * delta` to a 1-D joint qpos in a flat init_state vector,
    wrapping to [-pi, pi] and clamping into jnt_range. Returns (new_value,
    was_clamped). Prints a WARNING when clamping actually fires."""
    a = qpos_addr + qpos_offset_in_state
    raw = float(state[a]) + sign * delta
    # Wrap into (-pi, pi].
    wrapped = (raw + math.pi) % (2.0 * math.pi) - math.pi
    lo, hi = float(jnt_range[0]), float(jnt_range[1])
    clamped = max(lo, min(hi, wrapped))
    was_clamped = abs(clamped - wrapped) > 1e-6
    if was_clamped:
        print(f"  [WARN] joint qpos_addr={qpos_addr} bump wrapped+clamped: "
              f"raw={raw:.4f} -> wrap={wrapped:.4f} -> clamp={clamped:.4f} "
              f"(range=[{lo:.4f}, {hi:.4f}])")
    state[a] = clamped
    return clamped, was_clamped


def reset_env_rotated(
    env,
    base_xml: str,
    init_state: np.ndarray,
    theta_rad: float,
    mode: str = ROTATION_MODE_FULL,
    mount_body_name: Optional[str] = None,
    pivot_xy=None,
    exclude_substrings: tuple = ("robot", "gripper", "mount", "hand"),
    ee_body_name: str = "gripper0_eef",
    camera_names: tuple = ("agentview",),
) -> dict:
    """Reset env into a rotated scene; rotation baked into init_state.

    Modes:
      "full"    — rewrite XML so the robot mount body is rotated around world
                  +Z about pivot_xy; rotate every object free-joint qpos by
                  the same angle. Relative robot↔object geometry preserved.
                  Reachability invariant.
      "ee_yaw"  — keep robot mount XML pose unchanged; rotate every object
                  free-joint qpos; solve IK so the EE world pose (position
                  AND orientation) matches what FULL mode would produce at the
                  same theta. Arm joints 1–7 bend to reach the rotated EE
                  pose with the mount kept put. May be infeasible at large
                  theta (EE target unreachable from fixed mount) — an
                  `[IK-WARN]` is printed if joints get clamped.

    camera_names: cameras to co-rotate with the scene (default: ("agentview",)).
        Pass an empty tuple to keep all cameras fixed (old behaviour).
        Wrist / eye-in-hand cameras should NOT be listed here — they are
        attached to the robot and co-rotate automatically when the mount rotates.

    Both modes call env.set_init_state(rotated_state) so the model sees a
    physically-valid initial configuration, not a post-hoc perturbation.

    Returns the post-reset obs dict.
    """
    if mode not in _VALID_ROTATION_MODES:
        raise ValueError(f"mode must be one of {_VALID_ROTATION_MODES}, got {mode!r}")

    pivot = (resolve_table_center(env)[0] if pivot_xy is None
             else np.asarray(pivot_xy, dtype=float))

    # Build XML and reset:
    #   - mode=full + |theta|>0   -> rotate mount body + cameras in XML,
    #                                 then reset_from_xml_string
    #   - else                     -> XML unchanged, plain env.reset() (avoids
    #                                 the costly sim recreate from reset_from_xml_string)
    inner = getattr(env, "env", env)
    if mode == ROTATION_MODE_FULL and abs(theta_rad) >= 1e-12:
        if mount_body_name is None:
            mount_body_name = discover_robot_mount_body(env.sim)
        xml = rotate_body_in_xml(base_xml, mount_body_name, theta_rad, pivot_xy=pivot)
        for cam_name in camera_names:
            try:
                xml = rotate_camera_in_xml(xml, cam_name, theta_rad, pivot_xy=pivot)
            except RuntimeError as exc:
                print(f"  [WARN] rotate_camera_in_xml skipped {cam_name!r}: {exc}")
        inner.reset_from_xml_string(xml)
    else:
        env.reset()

    # Object qpos rotation (both modes do this).
    free = list_free_joints(env.sim, exclude_substrings=exclude_substrings)
    addrs = [adr for _, _, adr, _ in free]
    rotated_state = rotate_init_state_qpos(init_state, addrs, theta_rad, pivot)

    # ee_yaw also bends the arm via IK so EE pose matches FULL mode's EE pose.
    if mode == ROTATION_MODE_EE_YAW and abs(theta_rad) >= 1e-12:
        rotated_state, ik_info = compute_ee_pose_match_init_state(
            env, rotated_state, theta_rad, pivot,
            ee_body_name=ee_body_name, mount_body_name=mount_body_name,
        )
        print(f"  [IK] theta={math.degrees(theta_rad):+.2f}deg "
              f"pos_err={ik_info['pos_err']:.4f}m "
              f"orn_err={ik_info['orn_err']:.4f}rad "
              f"nfev={ik_info['nfev']} "
              f"clamped={len(ik_info['clamped'])}")

    return env.set_init_state(rotated_state)


def check_ee_yaw_feasibility(
    env,
    init_state: np.ndarray,
    angles_deg: list,
    pivot_xy=None,
    pos_thr: float = 0.02,
    orn_thr: float = 0.05,
    max_nfev: int = 200,
) -> tuple:
    """Quick IK-only pass — no model, no inference.

    Returns (feasible: list[float], infeasible: list[float]).
    Call this before eval to pre-filter angles that ee_yaw mode cannot reach
    (e.g. 180–210° where the EE target is behind the fixed mount).
    """
    pivot = (resolve_table_center(env)[0] if pivot_xy is None
             else np.asarray(pivot_xy, dtype=float))
    env.reset()
    addrs = [adr for _, _, adr, _ in list_free_joints(env.sim)]
    mount = discover_robot_mount_body(env.sim)

    feasible, infeasible = [], []
    for deg in angles_deg:
        theta = math.radians(deg)
        if abs(theta) < 1e-12:
            feasible.append(deg)
            continue
        env.reset()
        rotated = rotate_init_state_qpos(init_state, addrs, theta, pivot)
        _, info = compute_ee_pose_match_init_state(
            env, rotated, theta, pivot,
            ee_body_name="gripper0_eef", mount_body_name=mount, max_nfev=max_nfev,
        )
        (feasible if info["pos_err"] < pos_thr and info["orn_err"] < orn_thr
         else infeasible).append(deg)
    return feasible, infeasible


# ----------------------------------------------------------------------------
# Self-tests on import (fail fast on math regressions)
# ----------------------------------------------------------------------------

def _self_test() -> None:
    eye = np.array([1.0, 0.0, 0.0, 0.0])
    i = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(quat_mul_wxyz(eye, i), i), "identity * i != i"

    assert np.allclose(quat_from_axis_angle_z(0.0), eye), "rotZ(0) != identity"
    assert np.allclose(quat_from_axis_angle_z(math.pi), [0.0, 0.0, 0.0, 1.0], atol=1e-9), \
        "rotZ(pi) != [0,0,0,1]"

    half = quat_from_axis_angle_z(math.pi / 2.0)
    full = quat_mul_wxyz(half, half)
    assert np.allclose(np.abs(full), np.abs(quat_from_axis_angle_z(math.pi)), atol=1e-9), \
        "rotZ(pi/2) composed with itself != rotZ(pi)"

    p = rotate_xy_about_z(np.array([1.0, 0.0]), math.pi / 2.0, np.zeros(2))
    assert np.allclose(p, [0.0, 1.0], atol=1e-9), f"rotate_xy_about_z wrong: {p}"

    q7 = np.array([0.5, 0.3, 0.9, 1.0, 0.0, 0.0, 0.0])
    out = rotated_free_qpos7(q7, math.pi / 2.0, np.zeros(2))
    assert np.allclose(out[:2], [-0.3, 0.5], atol=1e-9), f"xy rotation wrong: {out[:2]}"
    assert abs(out[2] - 0.9) < 1e-12, f"z must be preserved, got {out[2]}"
    assert abs(np.linalg.norm(out[3:7]) - 1.0) < 1e-9, "output quat not unit"

    # v2 helpers: XML edit round-trip with a synthetic mount body.
    fake_xml = (
        "<mujoco>"
        "<worldbody>"
        "<body name='robot0_base' pos='-0.6 0 0'/>"
        "<camera name='agentview' pos='1 0 1'/>"
        "</worldbody>"
        "</mujoco>"
    )
    rotated = rotate_body_in_xml(fake_xml, "robot0_base", math.pi / 2.0)
    import xml.etree.ElementTree as ET
    rb = ET.fromstring(rotated).find("worldbody").find("body")
    pos_after = _parse_xml_vec(rb.get("pos"), [0, 0, 0])
    quat_after = _parse_xml_vec(rb.get("quat"), [1, 0, 0, 0])
    # Rotating (-0.6, 0) by 90 deg around origin -> (0, -0.6).
    assert np.allclose(pos_after[:2], [0.0, -0.6], atol=1e-9), \
        f"xml pos rotation wrong: {pos_after}"
    expected_quat = quat_from_axis_angle_z(math.pi / 2.0)
    assert np.allclose(quat_after, expected_quat, atol=1e-9), \
        f"xml quat rotation wrong: {quat_after}"

    # v4: joint bump wrap+clamp logic. State layout [time, ..., joint, ...].
    panda_j7_range = (-2.897, 2.897)
    s = np.zeros(3); s[1] = 0.5
    val, clamped = _bump_joint_in_state(s.copy(), qpos_addr=0, delta=math.pi / 4,
                                        jnt_range=panda_j7_range)
    assert abs(val - (0.5 + math.pi / 4)) < 1e-9 and not clamped, \
        f"normal bump wrong: val={val} clamped={clamped}"
    # Bump past +pi -> wraps into negative range, then clamp (within Panda range).
    val, clamped = _bump_joint_in_state(s.copy(), qpos_addr=0, delta=math.pi,
                                        jnt_range=panda_j7_range)
    # 0.5 + pi = 3.6415 -> wrap to 3.6415 - 2*pi = -2.6416 -> within range, no clamp.
    expected_wrap = (0.5 + math.pi + math.pi) % (2 * math.pi) - math.pi
    assert abs(val - expected_wrap) < 1e-9 and not clamped, \
        f"wrap bump wrong: val={val} expected={expected_wrap} clamped={clamped}"


_self_test()