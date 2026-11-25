"""Continuous-action PPO: actor-critic networks, GAE and the clipped objective."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class RunningObs:
    """Running mean/std normaliser for the observation vector."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.eps = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        b = x.shape[0]
        bmean, bvar = x.mean(0), x.var(0)
        delta = bmean - self.mean
        tot = self.count + b
        self.mean += delta * b / tot
        m2 = self.var * self.count + bvar * b + delta**2 * self.count * b / tot
        self.var = m2 / tot
        self.count = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + self.eps), -5.0, 5.0).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd):
        self.mean = np.asarray(sd["mean"], dtype=np.float64)
        self.var = np.asarray(sd["var"], dtype=np.float64)
        self.count = float(sd["count"])


def _mlp(sizes, out_gain=1.0):
    layers, last = [], sizes[0]
    for s in sizes[1:]:
        layers += [nn.Linear(last, s), nn.Tanh()]
        last = s
    layers = layers[:-1]  # drop the last tanh: linear output head
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2.0))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, out_gain)
    return net


class ActorCritic(nn.Module):
    """Diagonal Gaussian policy with tanh squashing plus a state-value head."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128), log_std_init: float = -1.0):
        super().__init__()
        self.act_dim = act_dim
        self.actor = _mlp([obs_dim, *hidden, act_dim], out_gain=0.01)
        self.critic = _mlp([obs_dim, *hidden, 1], out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def distribution(self, obs_n: torch.Tensor) -> Normal:
        return Normal(self.actor(obs_n), self.log_std.clamp(-3.0, 0.5).exp().expand_as(self.actor(obs_n)))

    def value(self, obs_n: torch.Tensor) -> torch.Tensor:
        return self.critic(obs_n).squeeze(-1)

    @staticmethod
    def squash(u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u)

    @staticmethod
    def log_prob(dist: Normal, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """log pi(a|s) for a = tanh(u), including the tanh Jacobian."""
        return (dist.log_prob(u) - torch.log(1.0 - a**2 + 1e-6)).sum(-1)

    @torch.no_grad()
    def act(self, obs_n: torch.Tensor, deterministic: bool = False):
        dist = self.distribution(obs_n)
        u = dist.mean if deterministic else dist.sample()
        a = self.squash(u)
        logp = self.log_prob(dist, u, a)
        return a, logp, self.value(obs_n)


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation over a (T, N) rollout."""
    T = len(rewards)
    adv = np.zeros((T,) + rewards[0].shape, dtype=np.float64)
    gae_run = np.zeros_like(rewards[0], dtype=np.float64)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae_run = delta + gamma * lam * nonterminal * gae_run
        adv[t] = gae_run
    return adv, adv + values


def ppo_update(policy, optim, batch, cfg, device):
    """One PPO round over a flattened batch of transitions."""
    obs, act, logp_old, adv, ret = (b.to(device) for b in batch)
    n = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clipfrac": 0.0}
    n_mb = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, cfg.minibatch):
            idx = perm[start : start + cfg.minibatch]
            if idx.numel() < 8:
                continue
            dist = policy.distribution(obs[idx])
            u = torch.atanh(act[idx].clamp(-0.999999, 0.999999))
            logp = policy.log_prob(dist, u, act[idx])
            ratio = (logp - logp_old[idx]).exp()
            s1 = ratio * adv[idx]
            s2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv[idx]
            policy_loss = -torch.min(s1, s2).mean()
            value = policy.value(obs[idx])
            value_loss = 0.5 * (value - ret[idx]).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optim.step()
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            stats["clipfrac"] += float(((ratio - 1.0).abs() > cfg.clip).float().mean())
            n_mb += 1
    for k in stats:
        stats[k] /= max(n_mb, 1)
    return stats
