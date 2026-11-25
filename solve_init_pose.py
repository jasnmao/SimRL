"""Solve an initial arm pose: upper arm hanging down, forearm pointing forward.

The screen normal is +X, so the forearm has to point at the screen while the upper
arm hangs naturally from the shoulder.  The elbow is bent slightly more than a
right angle (angle between upper arm and forearm ~100 deg), which puts the
fingertip in front of the chest at about chest height.

``--write`` stores the pose into simrl/config.py as INIT_Q7.
"""
import argparse
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

import simrl.config as C
from simrl.kinematics import ModelInfo, ScreenRig
from simrl.scene import ensure_scene

LIMIT_MARGIN = 0.08          # rad, stay at least this far from a joint stop
ELBOW_DEG = 100.0            # *interior* angle at the elbow (straight arm = 180)
W_DIR = 0.60                 # weight: upper arm pointing straight down
W_ELBOW = 0.60               # weight: elbow angle
W_TIP = 1.0                  # weight: fingertip in front of the chest


def chain(info, d, q7):
    d.qpos[:] = info.dependent_qpos(q7)
    d.qvel[:] = 0.0
    mujoco.mj_kinematics(info.m, d)
    S = np.array(d.xpos[info.humerus_body], float)
    E = np.array(d.xpos[info.elbow_body], float)
    F = np.array(d.xpos[info.eef_body], float)
    return S, E, F


def cost(q7, info, d, target, mid, span):
    S, E, F = chain(info, d, q7)
    up = (E - S) / max(np.linalg.norm(E - S), 1e-9)
    fore = (F - E) / max(np.linalg.norm(F - E), 1e-9)
    c = W_DIR * float(np.sum((up - np.array([0.0, -1.0, 0.0])) ** 2))
    # interior elbow angle: 180 deg with a straight arm, smaller as it bends
    ang = np.pi - np.arccos(np.clip(np.dot(up, fore), -1.0, 1.0))
    c += W_ELBOW * (ang - np.radians(ELBOW_DEG)) ** 2
    c += W_TIP * float(np.sum((F - target) ** 2))
    c += 1e-2 * float(np.sum(((q7 - mid) / span) ** 2))
    lo = q7 - (info.q_lo + LIMIT_MARGIN)
    hi = (info.q_hi - LIMIT_MARGIN) - q7
    c += 10.0 * float(np.sum(np.minimum(lo, 0.0) ** 2) + np.sum(np.minimum(hi, 0.0) ** 2))
    # keep the elbow out of the chest cavity (the ribs reach x=+0.09 near z=0)
    if E[0] < 0.0 and abs(E[2]) < 0.15:
        c += 40.0 * E[0] ** 2
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--starts", type=int, default=400)
    ap.add_argument("--tip", type=float, nargs=3, default=[0.44, -0.44, 0.18],
                    help="desired fingertip position [m, world]")
    args = ap.parse_args()

    ensure_scene()
    m = mujoco.MjModel.from_xml_path("assets/scene.xml")
    d = mujoco.MjData(m)
    info = ModelInfo(m)
    target = np.array(args.tip, float)
    mid = 0.5 * (info.q_lo + info.q_hi)
    span = info.q_hi - info.q_lo
    rig = ScreenRig(C.SwayConfig(enabled=False))

    rng = np.random.default_rng(2)
    best = None
    for k in range(args.starts):
        q0 = mid + rng.uniform(-0.4, 0.4, 7) * span
        res = minimize(
            cost, np.clip(q0, info.q_lo + 1e-3, info.q_hi - 1e-3),
            args=(info, d, target, mid, span),
            method="L-BFGS-B", bounds=list(zip(info.q_lo, info.q_hi)),
            options={"maxiter": 300},
        )
        if best is None or res.fun < best.fun:
            best = res

    q = np.asarray(best.x)
    S, E, F = chain(info, d, q)
    up = (E - S) / np.linalg.norm(E - S)
    fore = (F - E) / np.linalg.norm(F - E)
    ang = 180.0 - np.degrees(np.arccos(np.clip(np.dot(up, fore), -1, 1)))
    xy, uv = rig.project(F, 0.0)
    print("q7 =", list(np.round(q, 8)))
    print(f"cost {best.fun:.5f}")
    print("shoulder", np.round(S, 4), " elbow", np.round(E, 4), " fingertip", np.round(F, 4))
    print(f"upper-arm direction {np.round(up, 3)}  (want [0,-1,0])")
    print(f"forearm direction   {np.round(fore, 3)}  (want mostly +X)")
    print(f"elbow angle {ang:.1f} deg (want {ELBOW_DEG:.0f})")
    print("screen-plane", np.round(xy, 4), "-> px", np.round(uv, 1))
    print("q7      ", np.round(q, 3))
    print("limits lo", np.round(info.q_lo, 3))
    print("limits hi", np.round(info.q_hi, 3))
    print("slack   ", np.round(np.minimum(q - info.q_lo, info.q_hi - q), 3))

    if args.write:
        p = Path("simrl/config.py")
        t = p.read_text(encoding="utf-8")
        vals = ", ".join(f"{v:.8f}" for v in q)
        t = re.sub(r"INIT_Q7 = \([^)]*\)", f"INIT_Q7 = ({vals},)", t)
        p.write_text(t, encoding="utf-8")
        print("wrote INIT_Q7 to simrl/config.py")


if __name__ == "__main__":
    main()
