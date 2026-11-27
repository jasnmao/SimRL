"""Joint-space position control driven by inverse dynamics.

The policy emits *joint position increments* for the 7 actuated joints once per
40 ms control period.  For each of the 20 physics sub-steps the controller

1. follows a C1-continuous joint-space reference: a critically damped
   second-order reference governor low-pass filters the per-period position
   target, so the servo never chases a 40 ms step (the old behaviour),
2. runs a critically damped position servo (stiffness ``kp`` / damping ``kd``)
   around that moving reference,
3. adds the torque that inverse dynamics says is needed to *hold* the current
   pose (gravity + Coriolis + passive forces + equality-constraint reactions), and
4. clips the result to the actuator torque limits ``gear * ctrlrange``.

Why a governor rather than a 40 ms S-curve: a rest-to-rest 0.05 rad move in
40 ms needs a peak acceleration of roughly 180 rad/s^2 and saturates every
actuator, so the "S-curve" version (``action_smooth="traj"``) tracked an
infeasible reference and oscillated.  The governor instead moves the target at a
rate the actuators can actually follow; when the command is zero it comes to
rest.  ``traj`` is kept for experiments.
"""

from __future__ import annotations

import numpy as np
import mujoco


def minjerk(s: float) -> float:
    """Minimum-jerk normalised profile: zero velocity & acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


# Boundary conditions of a normalised quintic  q(s), s in [0, 1]:
#   q(0)=q0, q'(0)=v0*T, q''(0)=a0*T^2
#   q(1)=q1, q'(1)=v1*T, q''(1)=a1*T^2
_QUINTIC_A = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.0, 2.0, 6.0, 12.0, 20.0],
    ],
    dtype=np.float64,
)


def quintic_coeffs(q0, v0, a0, q1, v1, a1, T):
    """Coefficients (6, 7) of the normalised quintic for 7 joints."""
    rhs = np.stack(
        [q0, v0 * T, a0 * T * T, q1, v1 * T, a1 * T * T], axis=0
    )
    return np.linalg.solve(_QUINTIC_A, rhs)


def quintic_eval(c, s, T):
    """Return position, velocity and acceleration of the quintic at normalised s."""
    q = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * c[5]))))
    dq = c[1] + s * (2.0 * c[2] + s * (3.0 * c[3] + s * (4.0 * c[4] + s * 5.0 * c[5])))
    ddq = 2.0 * c[2] + s * (6.0 * c[3] + s * (12.0 * c[4] + s * 20.0 * c[5]))
    return q, dq / T, ddq / (T * T)


class PoseController:
    """Position servo + inverse-dynamics feed-forward for the 7 actuated joints.

    The stiffness/damping are scaled with the joint-space inertia of the model at
    the Init pose, so that every joint closes its position error in roughly
    ``1/omega_n`` seconds (``zeta = 1`` gives critical damping).  Torque limits are
    applied afterwards, i.e. the servo asks for as much as the actuators allow.

    Modes (``ControlConfig.action_smooth``):
      * ``"step"``    – target jumps to ``q0+dq`` immediately (original training)
      * ``"filt"``    – second-order reference governor + inverse dynamics (fine-tune/deploy)
      * ``"traj"``    – quintic reference + inverse dynamics (experimental)
      * ``"minjerk"`` – minimum-jerk interpolation of the target
      * ``"linear"``  – linear interpolation of the target
    """

    def __init__(self, model, info, cfg):
        self.m = model
        self.info = info
        self.tau_lo = info.tau_lo
        self.tau_hi = info.tau_hi
        self.mode = cfg.action_smooth
        self.T = float(cfg.substeps) * float(model.opt.timestep)
        d = mujoco.MjData(model)
        d.qpos[:] = info.dependent_qpos(info.init_q7())
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, d, M)
        inertia = np.diag(M)[info.dadr].copy()
        self.inertia = inertia
        omega, zeta = cfg.omega_n, cfg.zeta
        if cfg.kp is not None:
            self.kp = np.full(7, cfg.kp, dtype=np.float64)
            self.kd = np.full(
                7, cfg.kd if cfg.kd is not None else 2 * zeta * omega, dtype=np.float64
            )
        else:
            self.kp = inertia * omega**2
            self.kd = 2.0 * zeta * inertia * omega

        # Trajectory-mode gains can be tuned separately from the old step gains.
        self.traj_omega = float(getattr(cfg, "traj_omega_n", omega) or omega)
        self.traj_zeta = float(getattr(cfg, "traj_zeta", zeta))
        self.traj_vend = str(getattr(cfg, "traj_vend", "flow"))
        self.traj_qdd_max = float(getattr(cfg, "traj_qdd_max", 300.0))
        if cfg.kp is not None:
            self.kp_traj, self.kd_traj = self.kp, self.kd
        else:
            self.kp_traj = inertia * self.traj_omega**2
            self.kd_traj = 2.0 * self.traj_zeta * inertia * self.traj_omega

        self._pdata = mujoco.MjData(model) if self.mode == "traj" else None
        self._coef = None
        self.q7_start = None
        self.dq7 = None
        # per-env state of the second-order reference governor ("filt" mode)
        self.filt_omega = float(getattr(cfg, "filt_omega", 15.0))
        self.filt_zeta = float(getattr(cfg, "filt_zeta", 1.0))
        self.dt_sub = float(model.opt.timestep)
        self._f_ref: dict[int, np.ndarray] = {}
        self._f_vel: dict[int, np.ndarray] = {}
        self._f_goal: dict[int, np.ndarray] = {}
        self._f_goal0: dict[int, np.ndarray] = {}
        self._f_dgoal: dict[int, np.ndarray] = {}

    def reset(self, env_idx: int = 0):
        """Drop the persistent reference state of one env (called on episode reset)."""
        for d in (self._f_ref, self._f_vel, self._f_goal, self._f_goal0, self._f_dgoal):
            d.pop(int(env_idx), None)

    def hold_torque(self, data) -> np.ndarray:
        """Torque required to keep the *current* pose static (and the joints where they are)."""
        data.qfrc_applied[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_inverse(self.m, data)
        return data.qfrc_inverse[self.info.dadr].copy()

    def begin_period(self, data, q7_start, dq7, tau_prev, env_idx: int = 0):
        """Plan the trajectory for the next control period.

        ``q7_start``/``dq7`` are the measured joint positions and the commanded
        increment; ``tau_prev`` is the torque that was applied at the end of the
        previous period.  For the legacy modes only the target is stored.
        """
        self.q7_start = np.asarray(q7_start, dtype=np.float64).copy()
        self.dq7 = np.asarray(dq7, dtype=np.float64).copy()
        if self.mode == "filt":
            # Second-order reference governor: the target of this period is a
            # linear ramp from the previous reference target to the new one.
            # Planning from the *reference* (not from the measured q) keeps the
            # reference continuous even when the arm lags behind it.
            i = int(env_idx)
            if i not in self._f_ref:
                self._f_ref[i] = self.q7_start.copy()
                self._f_vel[i] = data.qvel[self.info.dadr].copy()
                self._f_goal[i] = self.q7_start.copy()
            q_ref = self._f_ref[i]
            g0 = self._f_goal[i]
            g1 = np.clip(q_ref + self.dq7, self.info.q_lo, self.info.q_hi)
            self._f_goal0[i] = g0
            self._f_goal[i] = g1
            self._f_dgoal[i] = g1 - g0
            return
        if self.mode != "traj":
            return

        q0 = self.q7_start
        v0 = data.qvel[self.info.dadr].copy()
        q1 = np.clip(q0 + self.dq7, self.info.q_lo, self.info.q_hi)
        if self.traj_vend == "rest":
            v1 = np.zeros(7, dtype=np.float64)
        else:
            v1 = (q1 - q0) / self.T

        # Initial acceleration: the acceleration that the torque applied at the
        # end of the previous period would produce now.  This keeps torque and
        # acceleration continuous across the control-period boundary.
        p = self._pdata
        mujoco.mj_copyData(p, self.m, data)
        p.qfrc_applied[:] = 0.0
        p.qfrc_applied[self.info.dadr] = np.asarray(tau_prev, dtype=np.float64)
        p.ctrl[:] = 0.0
        mujoco.mj_forward(self.m, p)
        a0 = p.qacc[self.info.dadr].copy()
        a0 = np.clip(a0, -self.traj_qdd_max, self.traj_qdd_max)

        self._coef = quintic_coeffs(q0, v0, a0, q1, v1, np.zeros(7), self.T)

    def __call__(self, data, kfrac, use_ff=True, env_idx: int = 0):
        """Return the (clipped) torque command for the 7 actuated joints."""
        mode = self.mode
        if mode == "filt":
            i = int(env_idx)
            s = min(max(float(kfrac), 0.0), 1.0)
            q_ref = self._f_ref[i]
            qd_ref = self._f_vel[i]
            g = self._f_goal0[i] + self._f_dgoal[i] * s
            # Critically damped second-order tracking of the ramped target.
            qdd = (
                self.filt_omega**2 * (g - q_ref)
                - 2.0 * self.filt_zeta * self.filt_omega * qd_ref
            )
            qd_ref = qd_ref + qdd * self.dt_sub
            q_ref = np.clip(q_ref + qd_ref * self.dt_sub, self.info.q_lo, self.info.q_hi)
            self._f_ref[i] = q_ref
            self._f_vel[i] = qd_ref
            q = data.qpos[self.info.qadr]
            v = data.qvel[self.info.dadr]
            tau_pd = self.kp * (q_ref - q) + self.kd * (qd_ref - v)
        elif mode == "traj":
            s = min(max(float(kfrac), 0.0), 1.0)
            q7_des, qd7_des, _ = quintic_eval(self._coef, s, self.T)
            q = data.qpos[self.info.qadr]
            v = data.qvel[self.info.dadr]
            tau_pd = self.kp_traj * (q7_des - q) + self.kd_traj * (qd7_des - v)
        else:
            if mode == "minjerk":
                frac = minjerk(kfrac)
            elif mode == "linear":
                frac = kfrac
            else:  # "step": the servo target jumps straight to the new position
                frac = 1.0
            q7_des = self.q7_start + self.dq7 * frac
            np.clip(q7_des, self.info.q_lo, self.info.q_hi, out=q7_des)
            q = data.qpos[self.info.qadr]
            v = data.qvel[self.info.dadr]
            tau_pd = self.kp * (q7_des - q) - self.kd * v

        if use_ff:
            tau = tau_pd + self.hold_torque(data)
        else:
            tau = tau_pd
        return np.clip(tau, self.tau_lo, self.tau_hi)
