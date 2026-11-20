"""Agents: TOP-TD3 (+ fixed-beta variants), TD3, SAC, OAC.

Every agent exposes the same interface used by train.py:
    start_episode() -> dict        called at the start of each training episode
    end_episode(episode_reward)    called at the end of each training episode
    get_action(state, deterministic) -> np.ndarray
    optimize() -> dict             one gradient step (critic, + actor when scheduled)

Sources (maths kept identical to the originals):
    TOP_Agent : TOP repo mujoco/top.py
    TD3_Agent : fiorenza2/TD3_PyTorch td3.py   (paper: "same as Ball and Roberts [2021]")
    SAC_Agent : fiorenza2/SAC_PyTorch sac.py   (paper: "same as Ball and Roberts [2021]")
    OAC       : SAC_Agent + exploration rule ported line-by-line from
                microsoft/oac-explore optimistic_exploration.py (beta_UB=4.66, delta=23.53,
                the values in their reproduce.sh)
"""
import copy
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from bandit import ExpWeights
from utils import (DeterministicPolicy, DoubleQFunc, MLPNetwork, QuantileDoubleQFunc,
                   ReplayPool, calculate_quantile_huber_loss)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def _freeze(net: nn.Module) -> None:
    for p in net.parameters():
        p.requires_grad = False


# =========================================================================== TOP-TD3
class TOP_Agent:
    """TOP-TD3. `fixed_beta=None` -> bandit over arms [-1, 0] (the paper's TOP).
    `fixed_beta=b` -> single-arm bandit, beta=b every episode
    (b=0: "Optimistic", b=-1: "Pessimistic" in Fig. 2)."""

    def __init__(self, seed: int, state_dim: int, action_dim: int, device: torch.device,
                 action_lim: float = 1.0, lr: float = 3e-4, gamma: float = 0.99, tau: float = 5e-3,
                 batchsize: int = 256, hidden_size: int = 256, n_hidden: int = 2,
                 update_interval: int = 2, buffer_size: int = int(1e6),
                 target_noise: float = 0.2, target_noise_clip: float = 0.5, explore_noise: float = 0.1,
                 n_quantiles: int = 50, kappa: float = 1.0, bandit_lr: float = 0.1,
                 fixed_beta: Optional[float] = None) -> None:
        self.device = device
        self.gamma, self.tau, self.batchsize = gamma, tau, batchsize
        self.update_interval, self.action_lim = update_interval, action_lim
        self.target_noise, self.target_noise_clip, self.explore_noise = target_noise, target_noise_clip, explore_noise

        torch.manual_seed(seed)
        self.q_funcs = QuantileDoubleQFunc(state_dim, action_dim, n_quantiles, hidden_size, n_hidden).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        _freeze(self.target_q_funcs)
        self.policy = DeterministicPolicy(state_dim, action_dim, hidden_size, n_hidden).to(device)
        self.target_policy = copy.deepcopy(self.policy)
        _freeze(self.target_policy)

        taus = torch.arange(0, n_quantiles + 1, device=device, dtype=torch.float32) / n_quantiles
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, n_quantiles)
        self.kappa = kappa

        # bandit "top-down controller" (unchanged). A single arm == fixed beta.
        arms = [-1, 0] if fixed_beta is None else [fixed_beta]
        self.TDC = ExpWeights(arms=arms, lr=bandit_lr, init=0.0, use_std=True)
        self.beta = None
        self._prev_episode_reward = 0.0

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.replay_pool = ReplayPool(state_dim, action_dim, buffer_size, seed=seed)
        self._update_counter = 0

    # ---- bandit, once per episode (same as original train_top_agent.py)
    def start_episode(self) -> Dict:
        self.beta = self.TDC.sample()
        return {"beta": float(self.beta)}

    def end_episode(self, episode_reward: float) -> None:
        feedback = episode_reward - self._prev_episode_reward
        self.TDC.update_dists(feedback)
        self._prev_episode_reward = episode_reward

    def arm_probs(self):
        return list(self.TDC.get_probs())

    # ---- acting
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            action = self.policy(state)
            if not deterministic:
                action += self.explore_noise * torch.randn_like(action)
            action.clamp_(-self.action_lim, self.action_lim)
        return np.atleast_1d(action.squeeze(0).cpu().numpy())

    # ---- learning (maths identical to mujoco/top.py)
    def _critic_loss(self, b, beta):
        with torch.no_grad():
            nextaction = self.target_policy(b["nextstate"])
            noise = (self.target_noise * torch.randn_like(nextaction)).clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction = (nextaction + noise).clamp_(-self.action_lim, self.action_lim)
            qt1, qt2 = self.target_q_funcs(b["nextstate"], nextaction)
            mu = torch.stack([qt1, qt2], dim=-1).mean(dim=-1)
            sigma = torch.sqrt((torch.pow(qt1 - mu, 2) + torch.pow(qt2 - mu, 2)) + 1e-4)
            belief = mu + beta * sigma                                           # [B, N]
            target = b["reward"][..., None] + (1.0 - b["real_done"][..., None]) * self.gamma * belief[:, None, :]
        q1, q2 = self.q_funcs(b["state"], b["action"])
        loss_1 = calculate_quantile_huber_loss(target - q1[..., None], self.tau_hats, kappa=self.kappa)
        loss_2 = calculate_quantile_huber_loss(target - q2[..., None], self.tau_hats, kappa=self.kappa)
        return loss_1 + loss_2

    def _policy_loss(self, state, beta):
        qb1, qb2 = self.q_funcs(state, self.policy(state))
        mu = torch.stack([qb1, qb2], dim=-1).mean(dim=-1)
        eps1, eps2 = 1e-4, 1.1e-4
        sigma = torch.sqrt((torch.pow(qb1 + eps1 - mu, 2) + torch.pow(qb2 + eps2 - mu, 2)) + eps1)
        belief = mu + beta * sigma
        return (-belief.mean(dim=-1)).mean()

    def optimize(self) -> Dict:
        b = self.replay_pool.sample(self.batchsize, self.device)
        q_loss = self._critic_loss(b, self.beta)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self._update_counter += 1
        out = {"q_loss": q_loss.detach()}
        if self._update_counter % self.update_interval == 0:
            _freeze(self.q_funcs)
            pi_loss = self._policy_loss(b["state"], self.beta)
            self.policy_optimizer.zero_grad()
            pi_loss.backward()
            self.policy_optimizer.step()
            for p in self.q_funcs.parameters():
                p.requires_grad = True
            _soft_update(self.target_q_funcs, self.q_funcs, self.tau)
            _soft_update(self.target_policy, self.policy, self.tau)
            out["pi_loss"] = pi_loss.detach()
        return out


# =========================================================================== TD3
class TD3_Agent:
    """Ball's TD3: min of two critics, MSE loss, delayed actor/target updates."""

    def __init__(self, seed, state_dim, action_dim, device, action_lim=1.0, lr=3e-4, gamma=0.99, tau=5e-3,
                 batchsize=256, hidden_size=256, n_hidden=2, update_interval=2, buffer_size=int(1e6),
                 target_noise=0.2, target_noise_clip=0.5, explore_noise=0.1):
        self.device = device
        self.gamma, self.tau, self.batchsize = gamma, tau, batchsize
        self.update_interval, self.action_lim = update_interval, action_lim
        self.target_noise, self.target_noise_clip, self.explore_noise = target_noise, target_noise_clip, explore_noise

        torch.manual_seed(seed)
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size, n_hidden).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        _freeze(self.target_q_funcs)
        self.policy = DeterministicPolicy(state_dim, action_dim, hidden_size, n_hidden).to(device)
        self.target_policy = copy.deepcopy(self.policy)
        _freeze(self.target_policy)
        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.replay_pool = ReplayPool(state_dim, action_dim, buffer_size, seed=seed)
        self._update_counter = 0

    def start_episode(self):
        return {}

    def end_episode(self, episode_reward):
        pass

    def get_action(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            action = self.policy(state)
            if not deterministic:
                action += self.explore_noise * torch.randn_like(action)
            action.clamp_(-self.action_lim, self.action_lim)
        return np.atleast_1d(action.squeeze(0).cpu().numpy())

    def optimize(self):
        b = self.replay_pool.sample(self.batchsize, self.device)
        with torch.no_grad():
            nextaction = self.target_policy(b["nextstate"])
            noise = (self.target_noise * torch.randn_like(nextaction)).clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction = (nextaction + noise).clamp_(-self.action_lim, self.action_lim)
            qt1, qt2 = self.target_q_funcs(b["nextstate"], nextaction)
            value_target = b["reward"] + (1.0 - b["real_done"]) * self.gamma * torch.min(qt1, qt2)
        q1, q2 = self.q_funcs(b["state"], b["action"])
        q_loss = F.mse_loss(q1, value_target) + F.mse_loss(q2, value_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self._update_counter += 1
        out = {"q_loss": q_loss.detach()}
        if self._update_counter % self.update_interval == 0:
            _freeze(self.q_funcs)
            qb1, qb2 = self.q_funcs(b["state"], self.policy(b["state"]))
            pi_loss = (-torch.min(qb1, qb2)).mean()
            self.policy_optimizer.zero_grad()
            pi_loss.backward()
            self.policy_optimizer.step()
            for p in self.q_funcs.parameters():
                p.requires_grad = True
            _soft_update(self.target_q_funcs, self.q_funcs, self.tau)
            _soft_update(self.target_policy, self.policy, self.tau)
            out["pi_loss"] = pi_loss.detach()
        return out


# =========================================================================== SAC / OAC
class GaussianPolicy(nn.Module):
    """Ball's SAC policy: tanh-squashed Gaussian, log-std clamped to [-20, 2]."""

    def __init__(self, state_dim, action_dim, hidden_size=256, n_hidden=2):
        super().__init__()
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size, n_hidden)

    def mu_std(self, x):
        mu, logstd = self.network(x).chunk(2, dim=1)
        return mu, torch.clamp(logstd, -20, 2).exp()

    def forward(self, x, get_logprob=False):
        mu, std = self.mu_std(x)
        dist = TransformedDistribution(Normal(mu, std), [TanhTransform(cache_size=1)])
        action = dist.rsample()
        logprob = dist.log_prob(action).sum(dim=-1, keepdim=True) if get_logprob else None
        return action, logprob, torch.tanh(mu)


class SAC_Agent:
    """Ball's SAC (automatic temperature, target entropy = -|A|).
    With oac=True, exploration actions use OAC's optimistic shift (Ciosek et al. 2019)."""

    def __init__(self, seed, state_dim, action_dim, device, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256,
                 hidden_size=256, n_hidden=2, update_interval=1, buffer_size=int(1e6), target_entropy=None,
                 oac=False, beta_UB=4.66, delta=23.53):
        self.device = device
        self.gamma, self.tau, self.batchsize, self.update_interval = gamma, tau, batchsize, update_interval
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.oac, self.beta_UB, self.delta = oac, beta_UB, delta

        torch.manual_seed(seed)
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size, n_hidden).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        _freeze(self.target_q_funcs)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_size, n_hidden).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.replay_pool = ReplayPool(state_dim, action_dim, buffer_size, seed=seed)
        self._update_counter = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def start_episode(self):
        return {}

    def end_episode(self, episode_reward):
        pass

    def get_action(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        if self.oac and not deterministic:
            return self._oac_action(state)
        with torch.no_grad():
            action, _, mean = self.policy(state)
        out = mean if deterministic else action
        return np.atleast_1d(out.squeeze(0).cpu().numpy())

    def _oac_action(self, state):
        """Port of oac-explore/optimistic_exploration.py:get_optimistic_exploration_action."""
        with torch.no_grad():
            pre_tanh_mu, std = self.policy.mu_std(state)
        pre_tanh_mu = pre_tanh_mu.detach().requires_grad_()
        q1, q2 = self.q_funcs(state, torch.tanh(pre_tanh_mu))
        mu_Q = (q1 + q2) / 2.0
        sigma_Q = torch.abs(q1 - q2) / 2.0
        Q_UB = mu_Q + self.beta_UB * sigma_Q
        grad = torch.autograd.grad(Q_UB.sum(), pre_tanh_mu)[0]
        with torch.no_grad():
            Sigma = torch.pow(std, 2)
            denom = torch.sqrt(torch.sum(torch.pow(grad, 2) * Sigma)) + 10e-6
            mu_C = math.sqrt(2.0 * self.delta) * Sigma * grad / denom
            dist = TransformedDistribution(Normal(pre_tanh_mu + mu_C, std), [TanhTransform(cache_size=1)])
            action = dist.sample()
        return np.atleast_1d(action.squeeze(0).cpu().numpy())

    def optimize(self):
        b = self.replay_pool.sample(self.batchsize, self.device)
        with torch.no_grad():
            nextaction, logprob_next, _ = self.policy(b["nextstate"], get_logprob=True)
            qt1, qt2 = self.target_q_funcs(b["nextstate"], nextaction)
            value_target = b["reward"] + (1.0 - b["real_done"]) * self.gamma * (
                torch.min(qt1, qt2) - self.alpha * logprob_next)
        q1, q2 = self.q_funcs(b["state"], b["action"])
        q_loss = F.mse_loss(q1, value_target) + F.mse_loss(q2, value_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        _freeze(self.q_funcs)
        action, logprob, _ = self.policy(b["state"], get_logprob=True)
        qb1, qb2 = self.q_funcs(b["state"], action)
        pi_loss = (self.alpha * logprob - torch.min(qb1, qb2)).mean()
        temp_loss = -self.alpha * (logprob.detach() + self.target_entropy).mean()
        self.policy_optimizer.zero_grad()
        pi_loss.backward()
        self.policy_optimizer.step()
        self.temp_optimizer.zero_grad()
        temp_loss.backward()
        self.temp_optimizer.step()
        for p in self.q_funcs.parameters():
            p.requires_grad = True

        # Ball's code: `if i % update_interval == 0` with i=0 every call -> target update every step
        if self._update_counter % self.update_interval == 0:
            _soft_update(self.target_q_funcs, self.q_funcs, self.tau)
        self._update_counter += 1
        return {"q_loss": q_loss.detach(), "pi_loss": pi_loss.detach(), "alpha": self.alpha.detach()}


# =========================================================================== factory
ALGOS = ("top", "td3", "sac", "oac")


def make_agent(algo: str, seed: int, state_dim: int, action_dim: int, device, args):
    common = dict(seed=seed, state_dim=state_dim, action_dim=action_dim, device=device,
                  hidden_size=args.hidden_size, n_hidden=args.n_hidden, batchsize=args.batch_size)
    if algo == "top":
        return TOP_Agent(**common, n_quantiles=args.n_quantiles, bandit_lr=args.bandit_lr,
                         fixed_beta=args.fixed_beta)
    if algo == "td3":
        return TD3_Agent(**common)
    if algo == "sac":
        return SAC_Agent(**common)
    if algo == "oac":
        return SAC_Agent(**common, oac=True, beta_UB=args.beta_UB, delta=args.delta)
    raise ValueError(algo)
