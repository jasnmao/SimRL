"""Calibrate the screen placement against the reachable pointing workspace.

Samples arm configurations around the initial pose, projects the fingertip onto
the (new, +X-normal) screen plane and reports
  * where the pointing workspace lands in the 640x640 image,
  * how much of it stays inside the image for a given SCREEN_SIZE / centre,
so ``config.SCREEN_SIZE``, ``SCREEN_CENTER_Y/Z`` and ``EpisodeConfig.target_region``
can be chosen from data instead of by hand.

Run ``python calibrate.py`` after changing INIT_Q7.
"""
import numpy as np

import mujoco

import simrl.config as C
from simrl.kinematics import ModelInfo, ScreenRig
from simrl.scene import ensure_scene

N = 4000


def sample(info, rng):
    base = info.init_q7()
    q_nat = np.clip(base + rng.normal(0, 0.35, size=(N, 7)), info.q_lo, info.q_hi)
    q_uni = rng.uniform(info.q_lo, info.q_hi, size=(N, 7))
    return q_nat, q_uni


def fingertips(m, info, qs):
    d = mujoco.MjData(m)
    out = np.empty((len(qs), 3))
    for i, q in enumerate(qs):
        d.qpos[:] = info.dependent_qpos(q)
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(m, d)
        out[i] = d.xpos[info.eef_body]
    return out


def main():
    ensure_scene()
    m = mujoco.MjModel.from_xml_path("assets/scene.xml")
    info = ModelInfo(m)
    rig = ScreenRig(C.SwayConfig(enabled=False))
    rng = np.random.default_rng(0)

    q_nat, q_uni = sample(info, rng)
    tip_nat = fingertips(m, info, q_nat)
    tip_uni = fingertips(m, info, q_uni)

    for name, tips in (("natural", tip_nat), ("uniform", tip_uni)):
        xy, uv = rig.project(tips, 0.0)
        print(f"\n[{name}] fingertip world  x {tips[:,0].min():.3f}..{tips[:,0].max():.3f}"
              f"  y {tips[:,1].min():.3f}..{tips[:,1].max():.3f}"
              f"  z {tips[:,2].min():.3f}..{tips[:,2].max():.3f}")
        print(f"  screen xy  x p5/p50/p95 = {np.percentile(xy[:,0],5):.3f}"
              f" / {np.percentile(xy[:,0],50):.3f} / {np.percentile(xy[:,0],95):.3f}"
              f"   y p5/p50/p95 = {np.percentile(xy[:,1],5):.3f}"
              f" / {np.percentile(xy[:,1],50):.3f} / {np.percentile(xy[:,1],95):.3f}")
        inside = (uv[:, 0] > 0) & (uv[:, 0] < C.IMAGE_SIZE) & (uv[:, 1] > 0) & (uv[:, 1] < C.IMAGE_SIZE)
        print(f"  fraction of poses whose cursor is inside the {C.IMAGE_SIZE}px image: {inside.mean():.3f}")
        print(f"  behind the screen plane (x < SCREEN_X): {(tips[:,0] < C.SCREEN_X).mean():.3f}")

    tip0 = fingertips(m, info, info.init_q7()[None, :])[0]
    xy0, uv0 = rig.project(tip0, 0.0)
    print(f"\ninitial pose: fingertip {np.round(tip0,4)}  plane {np.round(xy0,4)}  px {np.round(uv0,1)}")
    print(f"screen: SCREEN_SIZE={C.SCREEN_SIZE}  centre y={C.SCREEN_CENTER_Y} z={C.SCREEN_CENTER_Z}"
          f"  fovy={C.CAM_FOVY_DEG:.2f} deg  scale={C.IMAGE_SIZE/C.SCREEN_SIZE:.1f} px/m")
    print(f"target region {C.EpisodeConfig().target_region} [m, plane coords]")

    # coverage as a function of the screen size, centred on the natural cloud
    xy_nat = rig.project(tip_nat, 0.0)[0]
    print("\ncoverage if the screen were centred on the natural cloud centre:")
    for size in (0.55, 0.65, 0.72, 0.80, 0.90):
        half = size / 2
        frac = np.mean(
            (np.abs(xy_nat[:, 0] - np.median(xy_nat[:, 0])) < 0.5 * half * 2 * 0.95)
            & (np.abs(xy_nat[:, 1] - np.median(xy_nat[:, 1])) < 0.5 * half * 2 * 0.95)
        )
        print(f"  SCREEN_SIZE={size:.2f}: {frac*100:5.1f}% of natural poses inside 95% of the image")


if __name__ == "__main__":
    main()
