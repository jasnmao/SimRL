"""Vectorised pointing environment.

One control period = 20 MuJoCo physics steps of 2 ms (40 ms).

    s(t) --pi--> a(t) (7 joint position increments)
         --> inverse-dynamics torque servo --> 20 physics steps
         --> render the virtual screen at 640x640, YOLO -> two sphere centres
         --> s(t+1) = [purple(u,v), yellow(u,v), q7, dq7, ddq7]

Reward:  r = -( ||x_purple - x_yellow||^2 + r1 * ||tau||^2 + r2 * ||ddq||^2 ),
with every term scaled to O(1) by characteristic magnitudes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import mujoco
import numpy as np

from . import config as C
from .control import PoseController
from .detector import SphereDetector
from .kinematics import ModelInfo, ScreenRig
from .scene import load_model


@dataclass
class StepOut:
    obs: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    success: np.ndarray
    info: dict = field(default_factory=dict)


class VecPointingEnv:
    def __init__(
        self,
        cfg: C.Config | None = None,
        n_envs: int = 8,
        seed: int = 0,
        device: str = "cuda",
        render: bool = True,
        detector: SphereDetector | None = None,
        record_substeps: bool = False,
    ):
        self.cfg = cfg or C.default_config()
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.model = load_model(self.cfg.xml)
        self.info = ModelInfo(self.model, self.cfg)
        self.dt = float(self.model.opt.timestep)
        self.nsub = self.cfg.control.substeps
        self.render_enabled = render
        self.record_substeps = bool(record_substeps)
        self.step_substeps = None

        self.data = [mujoco.MjData(self.model) for _ in range(n_envs)]
        self.rigs = [ScreenRig(self.cfg.sway, seed=seed * 1000 + i) for i in range(n_envs)]
        self.ctrl = PoseController(self.model, self.info, self.cfg.control)
        self.obs_dim = 2 * 2 + 3 * 7
        self.act_dim = 7

        self.t = np.zeros(n_envs)
        self.steps = np.zeros(n_envs, dtype=np.int64)
        self.target_xy = np.zeros((n_envs, 2))
        self.hold = np.zeros(n_envs, dtype=np.int64)
        self.q7_prev = np.tile(self.info.init_q7(), (n_envs, 1))
        self.last_action = np.zeros((n_envs, 7))
        # bookkeeping for the reward / logging
        self.tau_sq = np.zeros(n_envs)
        self.qdd_sq = np.zeros(n_envs)
        self.det_fallback = 0
        self.det_occluded = 0
        self.det_offscreen = 0
        self.ep_return = np.zeros(n_envs)
        self.ep_len = np.zeros(n_envs, dtype=np.int64)
        self.episodes: list[tuple[float, int, bool]] = []

        self.renderer = None
        self.det = None
        if render:
            self.renderer = mujoco.Renderer(self.model, C.IMAGE_SIZE, C.IMAGE_SIZE)
            # Shadows alone cost ~35 ms/frame in software here; the two spheres are
            # matte so nothing is lost.  The demo video uses a separate, prettier
            # renderer.
            self.renderer.scene.flags[:] = 0
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 1
            self.det = detector or SphereDetector(
                self.cfg.det_weights, device=device, conf=self.cfg.det_conf, half=True,
                use_cuda_graph=self.cfg.det_use_cuda_graph,
            )
            self.det.warmup(n_envs)

        for i in range(n_envs):
            self._reset_physics(i)

    # ------------------------------------------------------------------
    # reset / observation
    # ------------------------------------------------------------------
    def _reset_physics(self, i: int):
        d = self.data[i]
        # the initial pose is *not* the test.xml "Init" keyframe any more: the arm
        # starts bent so that the fingertip already points at the screen
        d.qpos[:] = self.info.dependent_qpos(self.info.init_q7())
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        d.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, d)
        self.ctrl.reset(i)
        self.rigs[i].reset()
        self.t[i] = 0.0
        self.steps[i] = 0
        self.hold[i] = 0
        self.ep_return[i] = 0.0
        self.ep_len[i] = 0
        self.q7_prev[i] = self.info.init_q7()
        self.last_action[i] = 0.0
        reg = self.cfg.episode.target_region
        # the cursor starts at the projection of the initial fingertip; draw the
        # target uniformly in the region but never right on top of the cursor
        xy0, _ = self.rigs[i].project(self.info.fingertip(d), 0.0)
        for _ in range(50):
            cand = self.rng.uniform(reg[::2], reg[1::2])
            du = (cand[0] - xy0[0]) * (C.IMAGE_SIZE / C.SCREEN_SIZE)
            dv = (cand[1] - xy0[1]) * (C.IMAGE_SIZE / C.SCREEN_SIZE)
            if np.hypot(du, dv) >= self.cfg.episode.target_min_start_px:
                break
        self.target_xy[i] = cand

    def _view(self, idx):
        """Place the two balls + the camera for the given envs and refresh kinematics."""
        m = self.model
        for i in idx:
            d = self.data[i]
            rig = self.rigs[i]
            t = self.t[i]
            tip = self.info.fingertip(d)
            xy_c, _ = rig.project(tip, t)
            xy_t = self.target_xy[i] + rig.target_sway(t)
            d.mocap_pos[self.info.cursor_mocap] = rig.plane_to_world(xy_c, t)
            d.mocap_pos[self.info.target_mocap] = rig.plane_to_world(xy_t, t)
            campos, R = rig.camera(t)
            m.cam_pos[self.info.cam_id] = campos
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, R.flatten())
            m.cam_quat[self.info.cam_id] = quat
            mujoco.mj_forward(m, d)
            mujoco.mj_camlight(m, d)

    def _render_detect(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Render the screen view for envs ``idx`` and run YOLO on the batch."""
        if len(idx) == 0:
            return (np.zeros((0, 2, 2)), np.zeros((0, 2)), np.zeros((0, 2)),
                    np.zeros((0, 640, 640, 3), np.uint8))
        imgs = np.empty((len(idx), C.IMAGE_SIZE, C.IMAGE_SIZE, 3), dtype=np.uint8)
        for k, i in enumerate(idx):
            self.renderer.update_scene(self.data[i], camera=self.info.cam_id)
            imgs[k] = self.renderer.render()
        centers, conf, size = self.det.detect(imgs)
        return centers, conf, size, imgs

    def _analytic_centers(self, idx) -> np.ndarray:
        out = np.zeros((len(idx), 2, 2))
        for k, i in enumerate(idx):
            d, rig, t = self.data[i], self.rigs[i], self.t[i]
            xy_c, uv_c = rig.project(self.info.fingertip(d), t)
            uv_t = rig.plane_to_pixel(self.target_xy[i] + rig.target_sway(t))
            out[k, 0] = uv_t
            out[k, 1] = uv_c
        return out

    def _occluded(self, idx) -> np.ndarray:
        """Which spheres are hidden from the camera by the body (ray casting).

        The hand can reach past the virtual screen, and when it does, a large part of
        the arm sits between the camera and the screen.  YOLO then reports confident
        false positives on the bones for *both* classes, which collapses the measured
        distance to zero -- a reward hack.  Casting a ray from the camera to each ball
        centre tells us honestly whether that ball is visible at all.
        """
        out = np.zeros((len(idx), 2), dtype=bool)
        for k, i in enumerate(idx):
            d = self.data[i]
            cam = np.array(d.cam_xpos[self.info.cam_id], dtype=np.float64)
            for j, (body, mocap, radius) in enumerate(
                (
                    (self.info.target_body, self.info.target_mocap, C.TARGET_RADIUS),
                    (self.info.cursor_body, self.info.cursor_mocap, C.CURSOR_RADIUS),
                )
            ):
                vec = np.array(d.mocap_pos[mocap], dtype=np.float64) - cam
                dist = float(np.linalg.norm(vec))
                if dist < 1e-6:
                    continue
                gid = np.zeros(1, np.int32)
                hit = mujoco.mj_ray(self.model, d, cam, vec / dist, None, 1, body, gid)
                if hit > 0.0 and hit < dist - radius - 1e-3:
                    out[k, j] = True
        return out

    def _obs(self, centers: np.ndarray, idx=None) -> np.ndarray:
        idx = list(range(self.n)) if idx is None else list(idx)
        q = np.stack([self.data[i].qpos[self.info.qadr] for i in idx])
        v = np.stack([self.data[i].qvel[self.info.dadr] for i in idx])
        a = np.stack([self.data[i].qacc[self.info.dadr] for i in idx])
        qn = 2.0 * (q - self.info.q_lo) / (self.info.q_hi - self.info.q_lo) - 1.0
        cn = np.clip((centers.reshape(len(idx), -1) - 320.0) / 320.0, -3.0, 3.0)
        vn = np.clip(v / self.cfg.obs_vel_scale, -3.0, 3.0)
        an = np.clip(a / self.cfg.obs_acc_scale, -3.0, 3.0)
        return np.concatenate([cn, qn, vn, an], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        for i in range(self.n):
            self._reset_physics(i)
        self._view(range(self.n))
        centers, conf, _size, _ = self._render_detect(range(self.n)) if self.render_enabled else (
            self._analytic_centers(range(self.n)), np.ones((self.n, 2)), None, None)
        return self._obs(centers)

    def _physics(self, actions: np.ndarray):
        m = self.model
        self.tau_sq[:] = 0.0
        self.qdd_sq[:] = 0.0
        if self.record_substeps:
            self.step_substeps = [None] * self.n
        for i in range(self.n):
            d = self.data[i]
            q7_start = d.qpos[self.info.qadr].copy()
            dq7 = np.clip(actions[i], -1.0, 1.0) * self.cfg.control.max_action
            dq7 = np.clip(q7_start + dq7, self.info.q_lo, self.info.q_hi) - q7_start
            self.q7_prev[i] = q7_start
            self.last_action[i] = dq7
            tau_prev = (d.ctrl * self.info.gear).copy()
            self.ctrl.begin_period(d, q7_start, dq7, tau_prev, env_idx=i)
            tau_sq = 0.0
            qdd_sq = 0.0
            t0 = self.t[i]
            subs = None
            if self.record_substeps:
                subs = {"t": [], "q7": [], "qd7": [], "qacc7": [], "tau": [], "fingertip": []}
            for k in range(self.nsub):
                tau = self.ctrl(d, k / self.nsub, env_idx=i)
                d.ctrl[:] = tau / self.info.gear
                mujoco.mj_step(m, d)
                tau_sq += float(np.sum(tau**2))
                qdd_sq += float(np.sum(d.qacc[self.info.dadr] ** 2))
                if subs is not None:
                    # mj_step integrates qpos but leaves xpos/xmat from the pre-step
                    # kinematics; refresh them so the fingertip matches the state.
                    mujoco.mj_kinematics(m, d)
                    subs["t"].append(t0 + (k + 1) * self.dt)
                    subs["q7"].append(d.qpos[self.info.qadr].copy())
                    subs["qd7"].append(d.qvel[self.info.dadr].copy())
                    subs["qacc7"].append(d.qacc[self.info.dadr].copy())
                    subs["tau"].append(tau.copy())
                    subs["fingertip"].append(self.info.fingertip(d).copy())
            if subs is not None:
                self.step_substeps[i] = {
                    key: np.asarray(val, dtype=np.float64) for key, val in subs.items()
                }
            self.tau_sq[i] = tau_sq / self.nsub
            self.qdd_sq[i] = qdd_sq / self.nsub
            self.t[i] += self.nsub * self.dt
            self.steps[i] += 1

    def step(self, actions: np.ndarray) -> StepOut:
        self._physics(np.asarray(actions, dtype=np.float64))
        idx = list(range(self.n))
        self._view(idx)
        if self.render_enabled:
            centers, conf, size, imgs = self._render_detect(idx)
            ana = self._analytic_centers(idx)
            bad = conf < self.cfg.det_conf_min
            # the regression branch of the detector can silently degenerate (boxes
            # collapse to ~0 while the class scores stay high), so validate the box
            # size too and trust the analytic projection whenever it looks wrong
            lo = np.array([r[0] for r in self.cfg.det_size_range])
            hi = np.array([r[1] for r in self.cfg.det_size_range])
            bad = bad | (size < lo) | (size > hi)
            occ = self._occluded(idx)
            # The hand may point outside the screen image (e.g. at t=0 for a pose
            # that was never calibrated to be inside it).  YOLO cannot report a
            # purple centre then, and its false positives on the arm are unusable,
            # so use the proprioceptive projection whenever that lies off-image.
            uv_c = ana[:, 1]
            margin = 2.0
            off = (
                (uv_c[:, 0] < margin)
                | (uv_c[:, 0] > C.IMAGE_SIZE - margin)
                | (uv_c[:, 1] < margin)
                | (uv_c[:, 1] > C.IMAGE_SIZE - margin)
            )
            use_ana = bad | occ
            use_ana[:, 1] |= off
            self.det_offscreen += int(off.sum())
            if use_ana.any():
                self.det_fallback += int(bad.sum())
                self.det_occluded += int(occ.sum())
                centers = np.where(use_ana[..., None], ana, centers)
        else:
            centers, conf, imgs = self._analytic_centers(idx), np.ones((self.n, 2)), None

        dist_px = np.linalg.norm(centers[:, 0] - centers[:, 1], axis=-1)
        qvel = np.stack([self.data[i].qvel[self.info.dadr] for i in range(self.n)])
        maxv = np.abs(qvel).max(axis=1)

        rc = self.cfg.reward
        dist = (dist_px / rc.dist_scale_px) ** 2
        tau = self.tau_sq / rc.tau_scale**2
        qdd = self.qdd_sq / rc.qdd_scale**2
        qvel_sq = np.stack([np.sum(self.data[i].qvel[self.info.dadr] ** 2)
                            for i in range(self.n)])
        vel = qvel_sq / rc.qvel_scale**2
        reward = -(dist + rc.r1 * tau + rc.r2 * qdd + rc.r3 * vel)

        near = dist_px < self.cfg.episode.success_px
        still = maxv < self.cfg.episode.success_qvel
        ok = near & still
        self.hold = np.where(ok, self.hold + 1, 0)
        success = self.hold >= self.cfg.episode.success_hold
        # "stay there" reward: the agent gets a large per-step bonus exactly while it
        # points at the target *and* has (almost) stopped moving, so holding the ball on
        # the target is worth far more than anything else in the reward.  The episode
        # still ends after success_hold consecutive such steps.
        reward = reward + rc.hold_bonus * ok.astype(np.float64)
        reward = reward + rc.success_bonus * success
        info_hold = ok
        timeout = self.steps >= self.cfg.episode.max_steps
        done = success | timeout

        obs = self._obs(centers)
        info = {
            "dist_px": dist_px,
            "centers": centers,
            "conf": conf,
            "success": success,
            "dist_term": dist,
            "tau_term": tau,
            "qdd_term": qdd,
            "vel_term": vel,
            "hold": info_hold,
            "images": imgs,
            "occluded": occ if self.render_enabled else np.zeros((self.n, 2), bool),
            "substeps": self.step_substeps,
        }
        # true joint state of this step, captured *before* a done-env is reset
        # (evaluation plots need the state of the final, successful step)
        info["q7"] = np.stack([self.data[i].qpos[self.info.qadr] for i in range(self.n)])
        info["qd7"] = np.stack([self.data[i].qvel[self.info.dadr] for i in range(self.n)])
        info["fingertip"] = np.stack([self.info.fingertip(self.data[i]) for i in range(self.n)])

        # episode bookkeeping (for progress reporting)
        self.ep_return += reward
        self.ep_len += 1
        if done.any():
            for i in np.nonzero(done)[0]:
                self.episodes.append(
                    (float(self.ep_return[i]), int(self.ep_len[i]), bool(success[i]))
                )
                self.ep_return[i] = 0.0
                self.ep_len[i] = 0
        if done.any():
            idx_done = np.nonzero(done)[0]
            for i in idx_done:
                self._reset_physics(int(i))
            self._view(idx_done)
            if self.render_enabled:
                c2, _, _, _ = self._render_detect(idx_done)
            else:
                c2 = self._analytic_centers(idx_done)
            obs[idx_done] = self._obs(c2, idx_done)
        return StepOut(obs=obs, reward=reward, done=done, success=success, info=info)

    # ------------------------------------------------------------------
    def analytic_state(self) -> dict:
        """True (non-detected) quantities, for evaluation / plots."""
        centers = self._analytic_centers(range(self.n))
        q = np.stack([self.data[i].qpos[self.info.qadr] for i in range(self.n)])
        v = np.stack([self.data[i].qvel[self.info.dadr] for i in range(self.n)])
        a = np.stack([self.data[i].qacc[self.info.dadr] for i in range(self.n)])
        tip = np.stack([self.info.fingertip(self.data[i]) for i in range(self.n)])
        return {"centers": centers, "q7": q, "qd7": v, "qacc7": a, "fingertip": tip, "t": self.t.copy()}
