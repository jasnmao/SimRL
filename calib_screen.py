"""Choose the virtual screen size / centre so that

  * the initial purple ball (projection of the Init fingertip) is comfortably
    inside the 640x640 image,
  * the whole yellow ball stays inside the image for every sampled target,
  * the target sampling region is reachable by the arm.

Reports a grid over (SCREEN_SIZE, SCREEN_CENTER_Y).
"""

import numpy as np
import mujoco

import simrl
from simrl import config as C
from simrl.kinematics import ModelInfo, ScreenRig

m = mujoco.MjModel.from_xml_path("assets/scene.xml")
info = ModelInfo(m)
d = mujoco.MjData(m)
rng = np.random.default_rng(0)

# --- sample the workspace: uniform over joint limits + "natural" around Init ---
n = 20000
q_uni = rng.uniform(info.q_lo, info.q_hi, size=(n, 7))
q_nat = np.clip(info.init_q7() + rng.normal(0, 0.45, size=(n, 7)), info.q_lo, info.q_hi)
tips = {}
for tag, q7 in (("uni", q_uni), ("nat", q_nat)):
    t = np.zeros((n, 3))
    for i in range(n):
        d.qpos[:] = info.dependent_qpos(q7[i])
        mujoco.mj_kinematics(m, d)
        t[i] = info.fingertip(d)
    tips[tag] = t
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
tip0 = info.fingertip(d)
print("init fingertip", tip0)

rig = ScreenRig(C.SwayConfig(enabled=False), seed=0)
D = C.CAM_DISTANCE
ZS = C.SCREEN_Z


def plane_rel(P, yc):
    """screen-plane coords of the projection, relative to the screen centre."""
    pz = P[:, 2] - ZS
    pz = np.minimum(pz, D - 0.02)
    s = D / (D - pz)
    return np.stack([s * P[:, 0], s * (P[:, 1] - yc)], axis=1)


# max |sway| displacement in metres on the plane (rig rotation/translation)
sw = C.SwayConfig()
rot_px_m = np.tan(np.deg2rad(sw.rot_amp_deg)) * D          # plane shift from rotation
print("max rig sway on plane [m]", rot_px_m + np.abs(np.asarray(sw.trans_amp)))
target_region = np.array(C.EpisodeConfig().target_region)
print("target region", target_region)

print(f"\n{'S':>5} {'yc':>6} {'FOV':>6} {'r_y':>5} {'r_p':>5} {'v_init':>7} {'marg_p':>7} "
      f"{'tgt vmin/max':>16} {'marg_y':>7} {'reach%':>7} {'ws_in%':>7}")
best = []
for S in (0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75):
    for yc in (-0.30, -0.34, -0.38, -0.42, -0.46, -0.50):
        scale = C.IMAGE_SIZE / S
        ry = C.TARGET_RADIUS * scale
        rp = C.CURSOR_RADIUS * scale
        Y0 = plane_rel(tip0[None], yc)[0]
        v_init = 320 - scale * Y0[1]
        sway_px = (rot_px_m + np.abs(np.asarray(sw.trans_amp))).max() * scale
        marg_p = 320 - (abs(v_init - 320) + rp + sway_px)
        # target region corners
        xs = np.linspace(target_region[0], target_region[1], 5)
        ys = np.linspace(target_region[2], target_region[3], 5)
        X, Y = np.meshgrid(xs, ys)
        u = 320 + scale * X
        v = 320 - scale * Y
        marg_y = 320 - (np.abs(0.5 * (u.max() + u.min()) - 320) + 0.5 * (u.max() - u.min())
                        + ry + sway_px)
        marg_y2 = 320 - (np.abs(0.5 * (v.max() + v.min()) - 320) + 0.5 * (v.max() - v.min())
                         + ry + sway_px)
        marg_y = min(marg_y, marg_y2)
        # reachability
        Xu = plane_rel(tips["uni"], yc)
        inreg = ((Xu[:, 0] > target_region[0]) & (Xu[:, 0] < target_region[1]) &
                 (Xu[:, 1] > target_region[2]) & (Xu[:, 1] < target_region[3]))
        Xn = plane_rel(tips["nat"], yc)
        innat = ((Xn[:, 0] > target_region[0]) & (Xn[:, 0] < target_region[1]) &
                 (Xn[:, 1] > target_region[2]) & (Xn[:, 1] < target_region[3]))
        # workspace fraction visible in the image
        ws_in = np.mean((np.abs(Xn[:, 0]) * scale < 320 - rp) & (np.abs(Xn[:, 1]) * scale < 320 - rp))
        ok = (marg_p > 0) and (marg_y > 0) and inreg.mean() > 0.05
        fov = 2 * np.degrees(np.arctan(0.5 * S / D))
        print(f"{S:5.2f} {yc:6.2f} {fov:6.1f} {ry:5.1f} {rp:5.1f} {v_init:7.1f} {marg_p:7.1f} "
              f"{v.min():7.0f}/{v.max():<8.0f} {marg_y:7.1f} {100*inreg.mean():7.1f} {100*ws_in:7.1f}"
              + ("  <==" if ok else ""))
        if ok:
            best.append((S, yc, marg_p, marg_y, inreg.mean()))
print("\ncandidates:", len(best))
