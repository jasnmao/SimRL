"""Feasibility probe: how expensive is one policy call (batch of 8 envs)?

Requirement 2 asks the policy to run every 2 ms of simulated time (20 calls per
40 ms decision step).  That only works if a policy call is cheap compared with one
physics substep (~0.1 ms for 8 envs).
"""
import time

import numpy as np
import torch

import simrl  # noqa: F401  (sets the MKL env vars)
from simrl.ppo import ActorCritic

B, OBS, ACT = 8, 25, 7


def bench(name, fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"{name:38s} {dt*1e3:8.3f} ms/call   -> {dt*1e3/20:6.3f} ms per substep budget")
    return dt


def main():
    torch.manual_seed(0)
    pol = ActorCritic(OBS, ACT, (128, 128), -1.0).eval()
    obs_np = np.random.randn(B, OBS).astype(np.float32)

    obs_gpu = torch.from_numpy(obs_np).cuda()
    pol_gpu = pol.cuda()

    @torch.no_grad()
    def gpu_full():
        d = pol_gpu.distribution(obs_gpu)
        a = d.mean
        return a.cpu()

    @torch.no_grad()
    def gpu_actor_only():
        h = obs_gpu
        for lin in pol_gpu.actor:
            h = torch.tanh(lin(h))
        return h.cpu()

    pol_cpu = ActorCritic(OBS, ACT, (128, 128), -1.0).eval()
    pol_cpu.load_state_dict(pol.state_dict())
    obs_cpu = torch.from_numpy(obs_np)

    @torch.no_grad()
    def cpu_full():
        d = pol_cpu.distribution(obs_cpu)
        return d.mean.numpy()

    @torch.no_grad()
    def cpu_act_method():
        return pol_cpu.act(obs_cpu, deterministic=True)

    print("== policy call cost, batch of 8 ==")
    bench("CUDA: distribution().mean -> cpu", gpu_full)
    bench("CUDA: actor layers only", gpu_actor_only)
    bench("CPU : distribution().mean", cpu_full)
    bench("CPU : act(deterministic=True)", cpu_act_method)

    # a smaller net as a fallback option
    small = ActorCritic(OBS, ACT, (64, 64), -1.0).eval()
    obs_s = torch.from_numpy(obs_np)

    @torch.no_grad()
    def cpu_small():
        return small.act(obs_s, deterministic=True)

    bench("CPU : 64x64 net", cpu_small)

    # graph-captured GPU call
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                gpu_full()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            out_g = pol_gpu.distribution(obs_gpu).mean
        @torch.no_grad()
        def graph_call():
            g.replay()
            torch.cuda.synchronize()
            return out_g
        bench("CUDA: graph-captured actor", graph_call)
    except Exception as e:                                   # pragma: no cover
        print("cuda graph capture failed:", e)

    print("\n== per-substep budget ==")
    print("one physics substep for 8 envs is ~0.13 ms (mj_step+control+obs)")


if __name__ == "__main__":
    main()
