"""Train the pointing policy with PPO.

Usage
-----
    python train.py                 # full run from scratch (original "step" executor)
    python train.py --updates 50    # short smoke run
    python train.py --resume log/ppo_best.pt --updates 100  # fine-tune ("filt" executor)

Everything (checkpoints, the CSV progress log, the learning curves) is written
under ``log/``.  Progress and timing are printed to the console as training goes.
The controller is selected automatically: ``step`` from scratch, ``filt`` with
``--resume``; ``--smooth`` overrides either choice.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import simrl  # noqa: F401  (sets the MKL threading fix before numpy is imported)
from simrl import config as C
from simrl.env import VecPointingEnv
from simrl.ppo import ActorCritic, RunningObs, gae, ppo_update
from simrl.scene import ensure_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=None, help="override total_updates")
    ap.add_argument("--envs", type=int, default=None, help="override num_envs")
    ap.add_argument("--rollout", type=int, default=None, help="override rollout_steps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", type=str, default="ppo")
    ap.add_argument("--lr", type=float, default=None,
                    help="override learning rate (useful for fine-tuning)")
    ap.add_argument("--ent", type=float, default=None,
                    help="override entropy coefficient")
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint to continue training from (model + obs normaliser)")
    ap.add_argument("--smooth", type=str, default=None,
                    choices=["step", "filt", "traj", "minjerk", "linear"],
                    help="override action_smooth; default: step from scratch, filt when --resume")
    args = ap.parse_args()

    cfg = C.default_config()
    p = cfg.ppo
    # Keep the original training recipe and the deployment/fine-tuning recipe separate:
    # a run from scratch reproduces the original step-held executor, while a fine-tune
    # starts from a policy that was trained with step and continues under the smooth
    # governor used at deployment time.
    if args.smooth is not None:
        cfg.control.action_smooth = args.smooth
    elif args.resume:
        cfg.control.action_smooth = "filt"
    else:
        cfg.control.action_smooth = "step"
    if args.updates is not None:
        p.total_updates = args.updates
    if args.envs is not None:
        p.num_envs = args.envs
    if args.rollout is not None:
        p.rollout_steps = args.rollout
    if args.seed is not None:
        cfg.seed = args.seed
    if args.lr is not None:
        p.lr = args.lr
    if args.ent is not None:
        p.ent_coef = args.ent

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(p.device if torch.cuda.is_available() else "cpu")

    ensure_scene(cfg.xml, cfg.raw_xml)
    out = Path(cfg.out_dir)
    out.mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8")

    print("=" * 78)
    print("MuJoCo visual pointing task -- PPO training")
    print(f"  envs {p.num_envs}  rollout {p.rollout_steps}  updates {p.total_updates}  "
          f"=> {p.num_envs * p.rollout_steps * p.total_updates:,} env steps")
    print(f"  controller {cfg.control.action_smooth}  "
          f"(auto: step from scratch, filt with --resume; --smooth overrides)")
    print(f"  device {device}  torch {torch.__version__}")
    print("=" * 78)

    t0 = time.time()
    env = VecPointingEnv(cfg, n_envs=p.num_envs, seed=cfg.seed, device=str(device))
    print(f"environment ready in {time.time() - t0:.1f} s "
          f"(obs {env.obs_dim}, act {env.act_dim}, model fps {env.model.opt.timestep})")

    obs_norm = RunningObs(env.obs_dim)
    policy = ActorCritic(env.obs_dim, env.act_dim, tuple(p.hidden), p.log_std_init).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=p.lr, eps=1e-5)
    n_params = sum(x.numel() for x in policy.parameters())
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ck["model"])
        obs_norm.load_state_dict(ck["obs_norm"])
        print(f"resumed from {args.resume} (update {ck.get('update')})")
    print(f"policy: {n_params:,} parameters, {p.hidden} hidden")

    obs = env.reset()
    obs_norm.update(obs)
    csv_path = out / f"{args.tag}_log.csv"
    fields = ["update", "env_steps", "wall_time", "steps_per_s", "return_mean", "ep_len",
              "success_rate", "dist_px_mean", "policy_loss", "value_loss", "entropy",
              "clipfrac", "det_fallback", "det_occluded", "det_offscreen", "lr"]
    append = bool(args.resume) and csv_path.exists()
    with open(csv_path, "a" if append else "w", newline="", encoding="utf-8") as fh:
        if not append:
            csv.writer(fh).writerow(fields)

    history = {k: [] for k in fields}
    best_score = -9.0
    t_start = time.time()
    env_steps = 0

    for update in range(1, p.total_updates + 1):
        buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val = [], [], [], [], [], []
        dist_acc = []
        t_roll = time.time()
        for _ in range(p.rollout_steps):
            obs_n = obs_norm(obs)
            with torch.no_grad():
                a, logp, val = policy.act(torch.as_tensor(obs_n, device=device))
            out_step = env.step(a.cpu().numpy())
            buf_obs.append(obs_n)
            buf_act.append(a.cpu().numpy())
            buf_logp.append(logp.cpu().numpy())
            buf_val.append(val.cpu().numpy())
            buf_rew.append(out_step.reward)
            buf_done.append(out_step.done.astype(np.float64))
            dist_acc.append(float(out_step.info["dist_px"].mean()))
            obs = out_step.obs
            obs_norm.update(obs)
            env_steps += p.num_envs
        roll_time = time.time() - t_roll

        with torch.no_grad():
            last_val = policy.value(torch.as_tensor(obs_norm(obs), device=device)).cpu().numpy()
        adv, ret = gae(np.array(buf_rew), np.array(buf_val), np.array(buf_done),
                       last_val, p.gamma, p.lam)
        flat = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(-1, *np.shape(x)[2:]),
                                        device=device)
        batch = (flat(buf_obs), flat(buf_act), flat(buf_logp), flat(adv), flat(ret))
        t_upd = time.time()
        stats = ppo_update(policy, optim, batch, p, device)
        upd_time = time.time() - t_upd

        # ---- progress -----------------------------------------------------
        eps = env.episodes
        if eps:
            rets = np.array([e[0] for e in eps])
            lens = np.array([e[1] for e in eps])
            succ = np.array([1.0 if e[2] else 0.0 for e in eps])
            ret_m, len_m, rate = rets.mean(), lens.mean(), succ.mean()
        else:
            ret_m = len_m = rate = float("nan")
        if eps:
            env.episodes = []
        dist_m = float(np.mean(dist_acc))
        elapsed = time.time() - t_start
        sps = env_steps / max(elapsed, 1e-9)
        row = dict(update=update, env_steps=env_steps, wall_time=round(elapsed, 2),
                   steps_per_s=round(sps, 1), return_mean=round(float(ret_m), 4),
                   ep_len=round(float(len_m), 2), success_rate=round(float(rate), 3),
                   dist_px_mean=round(dist_m, 2),
                   policy_loss=round(stats["policy_loss"], 5),
                   value_loss=round(stats["value_loss"], 5),
                   entropy=round(stats["entropy"], 4),
                   clipfrac=round(stats["clipfrac"], 4),
                   det_fallback=env.det_fallback, det_occluded=env.det_occluded,
                   det_offscreen=env.det_offscreen, lr=p.lr)
        for k, v in row.items():
            history[k].append(v)
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([row[k] for k in fields])

        if update % 5 == 0 or update == 1:
            r_s = f"{ret_m:8.2f}" if eps else "     n/a"
            l_s = f"{len_m:6.1f}" if eps else "   n/a"
            s_s = f"{rate*100:5.1f}" if eps else "  n/a"
            print(f"[{update:4d}/{p.total_updates}] steps {env_steps:7d}  {sps:5.1f} sps  "
                  f"t {elapsed/60:5.2f} min  R {r_s}  len {l_s}  "
                  f"succ {s_s}%  dist {dist_m:6.1f}px  "
                  f"pi {stats['policy_loss']:+.3f} v {stats['value_loss']:.3f}  "
                  f"{roll_time*1e3:5.0f}+{upd_time*1e3:4.0f} ms")
        sys_stdout_flush()

        if update % 25 == 0 or update == p.total_updates:
            torch.save({"model": policy.state_dict(), "obs_norm": obs_norm.state_dict(),
                        "update": update, "cfg": cfg.to_dict()}, out / f"{args.tag}_ckpt.pt")
            print(f"    checkpoint -> {out / f'{args.tag}_ckpt.pt'}")
        # a checkpoint is "best" when it succeeds often *and* points accurately:
        # success rate alone is noisy with only a handful of episodes per update
        score = (rate if eps else -9.0) - dist_m / 500.0
        if eps and score > best_score and update > 10:
            best_score = score
            torch.save({"model": policy.state_dict(), "obs_norm": obs_norm.state_dict(),
                        "update": update, "cfg": cfg.to_dict()}, out / f"{args.tag}_best.pt")

    total = time.time() - t_start
    print("=" * 78)
    print(f"training finished: {env_steps:,} env steps in {total/60:.2f} min "
          f"({env_steps/total:.1f} steps/s), {total/p.total_updates:.2f} s per update")
    print(f"detector fallbacks: {env.det_fallback}  occluded-ball overrides: {env.det_occluded}  "
          f"off-screen-hand overrides: {env.det_offscreen}")
    plot_curves(history, out / f"{args.tag}_curves.png")


def sys_stdout_flush():
    import sys
    sys.stdout.flush()


def plot_curves(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    u = hist["update"]
    ax[0, 0].plot(u, hist["return_mean"], color="tab:blue")
    ax[0, 0].set_title("episode return")
    ax[0, 1].plot(u, np.array(hist["success_rate"]) * 100, color="tab:green")
    ax[0, 1].set_title("success rate (%)")
    ax[0, 1].set_ylim(-2, 102)
    ax[1, 0].plot(u, hist["dist_px_mean"], color="tab:red")
    ax[1, 0].set_title("final centre distance (px)")
    ax[1, 1].plot(u, hist["policy_loss"], label="policy")
    ax[1, 1].plot(u, hist["value_loss"], label="value")
    ax[1, 1].legend()
    ax[1, 1].set_title("losses")
    for a in ax.ravel():
        a.set_xlabel("update")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"curves -> {path}")


if __name__ == "__main__":
    main()
    import os
    import sys
    sys.stdout.flush()
    os._exit(0)   # avoid a noisy GLFW teardown traceback from Renderer.__del__
