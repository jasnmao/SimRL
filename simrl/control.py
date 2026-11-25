"""Operational-space style position controller driven by inverse dynamics.

The policy emits *joint position increments* for the 7 actuated joints.  For each
of the 20 physics sub-steps of one control period we

1. interpolate the commanded joint positions with a minimum-jerk profile,
2. run a critically damped position servo (stiffness ``kp`` / damping ``kd``),
3. add the torque that inverse dynamics says is needed to *hold* the pose
   (gravity + Coriolis + passive forces + equality-constraint reactions), and
4. clip the result to the actuator torque limits ``gear * ctrlrange``.

Calling :func:`mujoco.mj_inverse` with ``qacc = 0`` gives exactly the actuator
force that keeps the current pose static, so the servo only has to shape the
transient instead of fighting gravity.
"""

from __future__ import annotations

import numpy as np
import mujoco


def minjerk(s: float) -> float:
    """Minimum-jerk normalised profile: zero velocity & acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


class PoseController:
    """Position servo + inverse-dynamics feed-forward for the 7 actuated joints.

    The stiffness/damping are scaled with the joint-space inertia of the model at
    the Init pose, so that every joint closes its position error in roughly
    ``1/omega_n`` seconds (``zeta = 1`` gives critical damping).  Torque limits are
    applied afterwards, i.e. the servo asks for as much as the actuators allow.
    """

    def __init__(self, model, info, cfg):
        self.m = model
        self.info = info
        self.tau_lo = info.tau_lo
        self.tau_hi = info.tau_hi
        self.mode = cfg.action_smooth
        self.kfrac_max = 1.0
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
            self.kd = np.full(7, cfg.kd if cfg.kd is not None else 2 * zeta * omega, dtype=np.float64)
        else:
            self.kp = inertia * omega**2
            self.kd = 2.0 * zeta * inertia * omega

    def hold_torque(self, data) -> np.ndarray:
        """Torque required to keep the *current* pose static (and the joints where they are)."""
        data.qfrc_applied[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_inverse(self.m, data)
        return data.qfrc_inverse[self.info.dadr].copy()

    def __call__(self, data, q7_start, dq7, kfrac, use_ff=True):
        """Return the (clipped) torque command for the 7 actuated joints."""
        mode = self.mode
        if mode == "minjerk":
            frac = minjerk(kfrac)
        elif mode == "linear":
            frac = kfrac
        else:  # "step": the servo target jumps straight to the new position
            frac = 1.0
        q7_des = q7_start + dq7 * frac
        np.clip(q7_des, self.info.q_lo, self.info.q_hi, out=q7_des)
        q = data.qpos[self.info.qadr]
        v = data.qvel[self.info.dadr]
        tau_pd = self.kp * (q7_des - q) - self.kd * v
        if use_ff:
            tau = tau_pd + self.hold_torque(data)
        else:
            tau = tau_pd
        return np.clip(tau, self.tau_lo, self.tau_hi)
