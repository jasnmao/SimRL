"""Evaluate the trained policy: one real run + plots + demo video.

The policy is executed exactly as in the requirements: sample (or take the mean
of) Pi(a|s), turn the 7 joint position increments into torques with inverse
dynamics, clip them to the actuator limits, integrate 20 x 2 ms physics steps,
render the virtual screen, detect both spheres with YOLO, repeat until the
pointing task succeeds or the step limit is hit.

By default the deployment ``filt`` governor executes the increments; pass
``--smooth step`` to reproduce the original step-held executor.

Outputs (under log/):
    eval_curves.png   end-effector speed / acceleration + joint angles / velocities
    eval_trajectory.png   image-space path of the hand pointing sphere vs the target
    demo.mp4          third-person view of the whole human model + screen inset
    eval_stats.json   numeric summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import simrl  # noqa: F401  (MKL fix)
from simrl import config as C
from simrl.env import VecPointingEnv
from simrl.ppo import ActorCritic, RunningObs
from simrl.scene import ensure_scene


def load_policy(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    hidden = tuple(cfg["ppo"]["hidden"])
    pol = ActorCritic(25, 7, hidden, cfg["ppo"]["log_std_init"]).to(device)
    pol.load_state_dict(ckpt["model"])
    pol.eval()
    norm = RunningObs(25)
    norm.load_state_dict(ckpt["obs_norm"])
    return pol, norm, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="log/ppo_best.pt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--sample-std", type=float, default=0.0,
                    help="0 = use the policy mean; >0 = sample with this std")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--smooth", default="filt",
                    choices=["traj", "filt", "step", "minjerk", "linear"],
                    help="action_smooth; default filt (deployment/fine-tune), use step for the original executor")
    ap.add_argument("--omega", type=float, default=None,
                    help="override traj_omega_n")
    ap.add_argument("--vend", default=None, choices=["flow", "rest"],
                    help="override traj_vend")
    ap.add_argument("--qdd-max", type=float, default=None,
                    help="override traj_qdd_max")
    ap.add_argument("--filt-omega", type=float, default=None,
                    help="override filt_omega")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_scene()
    cfg = C.default_config()
    if args.smooth:
        cfg.control.action_smooth = args.smooth
    if args.omega is not None:
        cfg.control.traj_omega_n = args.omega
    if args.vend is not None:
        cfg.control.traj_vend = args.vend
    if args.qdd_max is not None:
        cfg.control.traj_qdd_max = args.qdd_max
    if args.filt_omega is not None:
        cfg.control.filt_omega = args.filt_omega
    cfg.ppo.num_envs = 1
    if args.steps:
        cfg.episode.max_steps = args.steps
    out = Path(cfg.out_dir)
    out.mkdir(exist_ok=True)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        alt = Path("log/ppo_ckpt.pt")
        print(f"{ckpt_path} not found, falling back to {alt}")
        ckpt_path = alt
    policy, obs_norm, ckpt = load_policy(ckpt_path, device)
    print(f"loaded {ckpt_path} (update {ckpt.get('update')})")

    env = VecPointingEnv(cfg, n_envs=1, seed=12345, device=str(device),
                         record_substeps=True)
    print(f"controller {cfg.control.action_smooth}")
    print(f"episode limit {cfg.episode.max_steps} steps "
          f"({cfg.episode.max_steps * cfg.control.substeps * env.dt:.2f} s of simulated time)")

    import cv2
    ep_frames = []          # per-episode screen frames (for the video)
    ep_mains = []           # per-episode third-person frames
    cur_frames, cur_mains = [], []
    big = None
    if not args.no_video:
        big, cam = make_camera(env)

    records = []
    best_rec = None         # frame kept for the video: most informative episode
    best_score = None
    best_frames, best_mains = [], []
    obs = env.reset()
    ep_sub = _new_sub(env)
    ep = 0
    t_sim = 0.0
    step_records = []
    t_wall = time.time()
    max_steps = cfg.episode.max_steps
    dt_step = cfg.control.substeps * env.dt
    while ep < args.episodes:
        with torch.no_grad():
            a, logp, val = policy.act(torch.as_tensor(obs_norm(obs), device=device))
            if args.sample_std > 0:
                a = torch.tanh(torch.atanh(a.clamp(-0.999, 0.999)) +
                               torch.randn_like(a) * args.sample_std)
        out_step = env.step(a.cpu().numpy())
        # one control period is 20 x 2 ms; keep every physics step for the plots
        sub = out_step.info["substeps"][0]
        for key in ep_sub:
            ep_sub[key].extend(np.asarray(sub[key]).tolist())
        step_records.append({
            "t": float(t_sim + dt_step),
            "q7": out_step.info["q7"][0].copy(),
            "qd7": out_step.info["qd7"][0].copy(),
            "fingertip": out_step.info["fingertip"][0].copy(),
            "centers": out_step.info["centers"][0].copy(),   # [yellow, purple]
            "dist_px": float(out_step.info["dist_px"][0]),
            "reward": float(out_step.reward[0]),
            "tau_sq": float(env.tau_sq[0]),
            "qdd_sq": float(env.qdd_sq[0]),
            "success": bool(out_step.success[0]),
            "action": a.cpu().numpy()[0].copy(),
        })
        t_sim += dt_step
        obs = out_step.obs
        if not args.no_video and env.renderer is not None:
            cur_frames.append(out_step.info["images"][0].copy())
            if out_step.done[0] and cur_mains:
                # the env was already reset inside step(); keep the previous render so
                # the last frame does not jump to the next episode's initial state
                cur_mains.append(cur_mains[-1].copy())
            else:
                cur_mains.append(render_main(big, env, cam))
        if out_step.done[0]:
            rec = {"episode": ep, "steps": len(step_records), "success": bool(out_step.success[0]),
                   "final_dist_px": float(out_step.info["dist_px"][0]),
                   "min_dist_px": float(min(r["dist_px"] for r in step_records)),
                   "settled_dist_px": float(np.mean([r["dist_px"] for r in step_records[-5:]])),
                   "return": float(sum(r["reward"] for r in step_records)),
                   "substeps": {k: np.asarray(v) for k, v in ep_sub.items()}}
            records.append((rec, step_records))
            print(f"episode {ep}: {'SUCCESS' if rec['success'] else 'timeout'} "
                  f"after {rec['steps']} steps ({rec['steps']*0.04:.2f} s), "
                  f"final distance {rec['final_dist_px']:.1f} px "
                  f"(min {rec['min_dist_px']:.1f}, last-5 mean {rec['settled_dist_px']:.1f}), "
                  f"return {rec['return']:.2f}")
            # keep the frames of the most informative episode only (memory friendly):
            # a success if possible, and the longest one (it shows the whole approach)
            score = (1 if rec["success"] else 0, rec["steps"])
            if best_score is None or score > best_score:
                best_score, best_rec = score, rec
                best_frames, best_mains = cur_frames, cur_mains
            cur_frames, cur_mains = [], []
            ep += 1
            step_records = []
            ep_sub = _new_sub(env)
            t_sim = 0.0
            t_wall = time.time()

    # ------------------------------------------------------------------ stats
    n_succ = sum(1 for r, _ in records if r["success"])
    stats = {
        "checkpoint": str(ckpt_path),
        "update": int(ckpt.get("update", -1)),
        "episodes": [{k: v for k, v in r.items() if k != "substeps"} for r, _ in records],
        "success_rate": n_succ / len(records),
        "mean_final_dist_px": float(np.mean([r["final_dist_px"] for r, _ in records])),
        "image_size": C.IMAGE_SIZE,
        "pixels_per_metre": C.IMAGE_SIZE / C.SCREEN_SIZE,
        "det_fallback": int(env.det_fallback),
        "det_occluded": int(env.det_occluded),
        "det_offscreen": int(env.det_offscreen),
        "env_steps": int(sum(r["steps"] for r, _ in records)),
    }
    (out / f"{args.tag}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "episodes"}, indent=2))

    # ------------------------------------------------------------------ plots
    best_i = next((i for i, (r, _) in enumerate(records) if r is best_rec), 0)
    best = records[best_i][1]
    best_sub = records[best_i][0]["substeps"]
    plot_run(best, best_sub, out / f"{args.tag}_curves.png", cfg)
    np.savez(out / f"{args.tag}_substeps.npz", **best_sub)
    plot_trajectory(best, out / f"{args.tag}_trajectory.png", cfg)

    # ------------------------------------------------------------------ video
    if not args.no_video and best_frames:
        print(f"video shows episode {best_i} "
              f"({'success' if records[best_i][0]['success'] else 'timeout'}, "
              f"{best_rec['steps']} steps, min distance {best_rec['min_dist_px']:.1f} px)")
        make_video(best_frames, best_mains, out / "demo.mp4",
                   label=(f"episode {best_i}: "
                          f"{'success' if best_rec['success'] else 'timeout'}"))
    print("done")
    sys.stdout.flush()
    os._exit(0)


# ---------------------------------------------------------------------- plots
def _t(rec):
    return np.array([r["t"] for r in rec])


def _new_sub(env):
    """Initial (t=0) state of a fresh episode, as lists ready for extension."""
    st = env.analytic_state()
    return {
        "t": [0.0],
        "q7": [st["q7"][0].copy()],
        "qd7": [st["qd7"][0].copy()],
        "qacc7": [np.zeros(7)],
        "tau": [np.zeros(7)],
        "fingertip": [st["fingertip"][0].copy()],
    }


def plot_run(rec, sub, path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(sub["t"], dtype=np.float64)
    if t.size < 2:
        return
    dt = float(np.median(np.diff(t)))          # 2 ms physics step
    tip = np.asarray(sub["fingertip"], dtype=np.float64)
    vel = np.gradient(tip, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    acc = np.gradient(vel, dt, axis=0)
    accn = np.linalg.norm(acc, axis=1)
    q = np.asarray(sub["q7"], dtype=np.float64)
    qd = np.asarray(sub["qd7"], dtype=np.float64)
    dist = np.array([r["dist_px"] for r in rec])
    nsub = int(cfg.control.substeps)

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.2))
    ax[0, 0].plot(t, speed, color="tab:blue", lw=0.9)
    ax[0, 0].set_title("end-effector speed  |v|  [m/s]")
    ax[0, 0].set_ylabel("m/s")
    ax[0, 1].plot(t, accn, color="tab:red", lw=0.9)
    ax[0, 1].set_title("end-effector acceleration  |a|  [m/s$^2$]")
    ax[0, 1].set_ylabel("m/s$^2$")
    for j, name in enumerate(C.JOINTS):
        ax[1, 0].plot(t, q[:, j], label=name, lw=1.0)
        ax[1, 1].plot(t, qd[:, j], label=name, lw=1.0)
    ax[1, 0].set_title("joint angles  [rad]")
    ax[1, 1].set_title("joint angular velocities  [rad/s]")
    ax[1, 1].legend(fontsize=7, ncol=2, loc="best")
    for a in ax.ravel():
        # 40 ms policy-decision boundaries
        for k in range(0, len(t), nsub):
            a.axvline(t[k], color="0.85", lw=0.6, zorder=0)
        a.set_xlabel("time [s]")
        a.grid(alpha=0.3)
    fig.suptitle(
        f"policy rollout  (2 ms physics samples, grey = 40 ms decisions; "
        f"final centre distance {dist[-1]:.1f} px)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_trajectory(rec, path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cen = np.array([r["centers"] for r in rec])      # (T, [yellow, purple], xy)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(np.zeros((C.IMAGE_SIZE, C.IMAGE_SIZE, 3), np.uint8))
    ax.plot(cen[:, 1, 0], cen[:, 1, 1], "-o", ms=2, color="#7e22ce", label="hand pointing (detected)")
    ax.plot(cen[:, 0, 0], cen[:, 0, 1], "s", ms=3, color="#d4a800", label="target (detected)")
    ax.plot(cen[0, 1, 0], cen[0, 1, 1], "o", ms=9, mfc="none", mew=2, color="#7e22ce")
    ax.set_xlim(0, C.IMAGE_SIZE)
    ax.set_ylim(C.IMAGE_SIZE, 0)
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")
    ax.legend(loc="upper right")
    ax.set_title("screen-space trajectory (640x640 view of the virtual screen)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------- video
def make_camera(env):
    import mujoco
    big = mujoco.Renderer(env.model, 720, 1280)
    big.scene.flags[:] = 0
    big.scene.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 1
    big.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    return big, "demo_cam"


def _cam_matrix(env, cam):
    import mujoco
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    R = np.zeros(9, np.float64)
    mujoco.mju_quat2Mat(R, env.model.cam_quat[cid])
    return np.array(env.model.cam_pos[cid], float), R.reshape(3, 3), float(env.model.cam_fovy[cid])


def _project(pts, o, R, fovy, w, h):
    """World points -> pixel coordinates for a MuJoCo camera (looks along -z of its frame)."""
    p = (np.asarray(pts, float) - o) @ R            # world -> camera frame
    f = (h / 2.0) / np.tan(np.radians(fovy) / 2.0)
    z = np.maximum(-p[:, 2], 1e-6)
    u = w / 2.0 + f * p[:, 0] / z
    v = h / 2.0 - f * p[:, 1] / z
    return np.stack([u, v], axis=-1)


def render_main(big, env, cam):
    big.update_scene(env.data[0], camera=cam)
    frame = np.asarray(big.render()).copy()
    # draw the pointing ray: fingertip -> purple ball (the cursor on the screen plane)
    import cv2

    d = env.data[0]
    tip = env.info.fingertip(d)[None, :]
    ball = np.array(d.mocap_pos[env.info.cursor_mocap], float)[None, :]
    o, R, fovy = _cam_matrix(env, cam)
    h, w = frame.shape[:2]
    uv = _project(np.concatenate([tip, ball], axis=0), o, R, fovy, w, h)
    ov = frame.copy()
    cv2.line(ov, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)),
             (255, 200, 60), 2, cv2.LINE_AA)          # BGR: warm yellow-green
    cv2.circle(ov, tuple(np.round(uv[0]).astype(int)), 5, (80, 255, 120), -1, cv2.LINE_AA)
    frame = cv2.addWeighted(ov, 0.65, frame, 0.35, 0.0)
    return frame


def make_video(screen_frames, mains, path, fps=25, slow=4, label=""):
    """Third-person view of the whole human model, with the policy's view as an inset."""
    import cv2

    H, W = mains[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    if not vw.isOpened():
        print(f"!! could not open video writer for {path}")
        return
    # soft vertical gradient that replaces the (empty, black) MuJoCo background
    grad = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
    bg = ((1.0 - grad) * np.array([238, 241, 245], np.float32) +
          grad * np.array([196, 202, 212], np.float32))
    bg = np.repeat(bg, W, axis=1)
    s = 256
    x0, y0 = W - s - 16, H - s - 16
    n_out = 0
    for k, (main, img) in enumerate(zip(mains, screen_frames)):
        # MuJoCo renders RGB, cv2 expects BGR
        m = main[:, :, ::-1].astype(np.float32)
        img = img[:, :, ::-1]
        a = np.clip((m.max(axis=2, keepdims=True) - 2.0) / 10.0, 0.0, 1.0)  # 0 = empty background
        main = (m * a + bg * (1.0 - a)).astype(np.uint8)
        small = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
        cv2.rectangle(main, (x0 - 3, y0 - 3), (x0 + s + 2, y0 + s + 2), (30, 30, 30), 2)
        main[y0 : y0 + s, x0 : x0 + s] = small
        cv2.putText(main, "policy view (640x640 -> 256)", (x0 - 2, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(main, f"t = {k * 0.04:5.2f} s", (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        if label:
            cv2.putText(main, label, (18, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
        for _ in range(slow):        # slow motion: the real steps are only 40 ms apart
            vw.write(main)
            n_out += 1
    vw.release()
    print(f"wrote {path} ({n_out} frames, {n_out/fps:.1f} s at {fps} fps, "
          f"{len(mains)} steps, {slow}x slow motion)")


if __name__ == "__main__":
    main()