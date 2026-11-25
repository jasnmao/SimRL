"""Model introspection helpers: joint ordering, dependent joints, FK, projection."""

from __future__ import annotations

import math
from typing import List, Tuple

import mujoco
import numpy as np

from . import config as C


class ModelInfo:
    """Cached indices, limits and equality-constraint structure of the model."""

    def __init__(self, model: mujoco.MjModel, cfg: C.Config | None = None):
        self.m = model
        self.cfg = cfg or C.default_config()
        self.nu = model.nu

        self.jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.JOINTS]
        assert all(j >= 0 for j in self.jnt_ids), "joint names not found"
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jnt_ids], dtype=np.int64)
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jnt_ids], dtype=np.int64)
        self.q_lo = np.array([model.jnt_range[j][0] for j in self.jnt_ids])
        self.q_hi = np.array([model.jnt_range[j][1] for j in self.jnt_ids])

        # actuator -> joint mapping (order of <actuator> is the action order too)
        self.act_jnt = np.array([model.actuator_trnid[i][0] for i in range(model.nu)], dtype=np.int64)
        self.gear = np.asarray(model.actuator_gear[:, 0], dtype=np.float64)
        self.tau_lo = self.gear * np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64)
        self.tau_hi = self.gear * np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64)
        # actuators are declared in the same order as C.JOINTS in this model
        assert np.array_equal(self.act_jnt, np.array(self.jnt_ids))

        self.eef_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end-effector")
        self.humerus_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "humerus")
        self.elbow_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ulna")
        self.eef_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "accsite_end-effector")
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "screen_cam")
        self.target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
        self.cursor_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cursor_ball")
        self.target_mocap = int(model.body_mocapid[self.target_body])
        self.cursor_mocap = int(model.body_mocapid[self.cursor_body])
        self.ball_qpos = np.array([0.0, 0.0, 0.0, 1.0])

        # Every joint that appears as obj1 of a JOINT equality is a slave, written
        # from its master.  Joints whose equality has no master are locked at 0.
        self.slaves: List[Tuple[int, int, np.ndarray]] = []
        self.locked: List[int] = []
        act = set(int(a) for a in self.act_jnt)
        for e in range(model.neq):
            if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            j1 = int(model.eq_obj1id[e])
            j2 = int(model.eq_obj2id[e])
            poly = np.asarray(model.eq_data[e][:5], dtype=np.float64)
            if j1 in act:
                continue
            if j2 < 0:
                self.locked.append(j1)
            else:
                self.slaves.append((j1, j2, poly))

    # ------------------------------------------------------------------
    def dependent_qpos(self, q7: np.ndarray) -> np.ndarray:
        """Full nq-dof qpos vector for a given 7-dof arm configuration."""
        q = np.zeros(self.m.nq)
        for j, a in enumerate(self.qadr):
            q[a] = q7[j]
        for j1 in self.locked:
            q[self.m.jnt_qposadr[j1]] = 0.0
        for j1, j2, poly in self.slaves:
            master = q[self.m.jnt_qposadr[j2]]
            q[self.m.jnt_qposadr[j1]] = (
                poly[0]
                + poly[1] * master
                + poly[2] * master**2
                + poly[3] * master**3
                + poly[4] * master**4
            )
        return q

    def init_q7(self) -> np.ndarray:
        """Initial arm pose: elbow flexed so the fingertip points at the screen.

        The original ``Init`` keyframe of models/test.xml has the arm hanging
        straight down/outwards (fingertip at x=0.06, z=0.64), which projects far
        outside a screen whose normal is +X.  This pose was solved by IK so that
        the fingertip sits in front of the chest, roughly at the middle of the
        screen (see ``calibrate.py --init-pose``).
        """
        return np.array(C.INIT_Q7, dtype=np.float64)

    def clamp_q7(self, q7):
        return np.clip(q7, self.q_lo, self.q_hi)

    def fingertip(self, data: mujoco.MjData) -> np.ndarray:
        return np.array(data.xpos[self.eef_body], dtype=np.float64)


# ----------------------------------------------------------------------
# screen rig
# ----------------------------------------------------------------------
def _euler_xyz(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class ScreenRig:
    """Rigid sway of the (camera + screen plane) pair, plus the target's own sway.

    Conventions
    -----------
    * ``frame`` = the virtual screen frame: origin = screen centre, x = right =
      world **-Z**, y = up = world **+Y**, z = out of the screen towards the
      camera = world **+X**.  The screen plane is ``x = SCREEN_X`` and the body is
      entirely behind it, exactly like a screen the person is facing.
    * The camera is at frame_origin + R @ (0, 0, CAM_DISTANCE) and uses R as its
      own orientation, so it always looks along the frame's -z at the body.
    * A world point P is expressed in the frame as p = (P - o) @ R.  The ray from
      the camera (0,0,D) through p hits the plane z=0 at  s = D/(D - p_z).
    """

    #: rest rotation of the screen frame (columns: right, up, towards-camera)
    BASE_ROT = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    def __init__(self, cfg: C.SwayConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        c = self.cfg
        self.phase = self.rng.uniform(0, 2 * np.pi, size=(3, 2))   # 3 axes x 2 tones
        self.freq = self.rng.uniform(*c.freq_range, size=(3, 2))
        self.phase_t = self.rng.uniform(0, 2 * np.pi, size=2)
        self.freq_t = self.rng.uniform(*c.target_sway_freq, size=2)
        # fixed random sign pattern so the two tones do not cancel
        self.sign = self.rng.choice([-1.0, 1.0], size=(3, 2))

    def _sin(self, t, phase, freq):
        w = 2 * np.pi * freq
        return np.sin(w * t + phase)

    def rig(self, t):
        """Return (R, origin) of the screen frame in world coordinates."""
        c = self.cfg
        o0 = np.array([C.SCREEN_X, C.SCREEN_CENTER_Y, C.SCREEN_CENTER_Z], dtype=np.float64)
        R0 = self.BASE_ROT
        if not c.enabled:
            return R0, o0
        s = (self._sin(t, self.phase, self.freq) * self.sign).sum(axis=-1) / 1.6
        rot = np.deg2rad(np.asarray(c.rot_amp_deg)) * s
        trans = np.asarray(c.trans_amp) * s
        return R0 @ _euler_xyz(*rot), o0 + R0 @ trans

    def camera(self, t):
        R, o = self.rig(t)
        return o + R @ np.array([0.0, 0.0, C.CAM_DISTANCE]), R

    def target_sway(self, t):
        c = self.cfg
        if not c.enabled:
            return np.zeros(2)
        w = 2 * np.pi * self.freq_t
        return np.asarray(c.target_sway_amp) * np.sin(w * t + self.phase_t)

    # -- geometry -----------------------------------------------------
    def project(self, P_world, t):
        """Ray-project a world point onto the screen plane.

        Returns ``(xy_metres, uv_pixels)``; ``xy`` is relative to the screen
        centre in the screen frame.
        """
        R, o = self.rig(t)
        p = (np.asarray(P_world, dtype=np.float64) - o) @ R       # frame coords (batched)
        pz = np.clip(p[..., 2], -np.inf, C.CAM_DISTANCE - 0.02)
        s = C.CAM_DISTANCE / (C.CAM_DISTANCE - pz)
        xy = p[..., :2] * s[..., None]
        return xy, self.plane_to_pixel(xy)

    def plane_to_pixel(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        scale = C.IMAGE_SIZE / C.SCREEN_SIZE
        u = 0.5 * C.IMAGE_SIZE + scale * xy[..., 0]
        v = 0.5 * C.IMAGE_SIZE - scale * xy[..., 1]
        return np.stack([u, v], axis=-1)

    def pixel_to_plane(self, uv):
        uv = np.asarray(uv, dtype=np.float64)
        scale = C.IMAGE_SIZE / C.SCREEN_SIZE
        x = (uv[..., 0] - 0.5 * C.IMAGE_SIZE) / scale
        y = -(uv[..., 1] - 0.5 * C.IMAGE_SIZE) / scale
        return np.stack([x, y], axis=-1)

    def plane_to_world(self, xy, t):
        """Screen-plane point (metres relative to the screen centre) -> world."""
        R, o = self.rig(t)
        xy = np.asarray(xy, dtype=np.float64)
        local = np.zeros(xy.shape[:-1] + (3,), dtype=np.float64)
        local[..., :2] = xy
        return o + local @ R.T
