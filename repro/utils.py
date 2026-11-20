"""Shared pieces: networks, replay buffer, quantile loss.

Sources:
  - networks / quantile loss: TOP repo (mujoco/networks.py, mujoco/utils.py), unchanged maths.
  - ReplayPool: Philip Ball's numpy buffer (fiorenza2/TD3_PyTorch utils.py), which the
    TOP repo's slow deque buffer was derived from. Same data, same uniform sampling
    without replacement -> same algorithm, only faster.
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- networks
class MLPNetwork(nn.Module):
    """MLP with `n_hidden` hidden layers of `hidden_size` units and ReLU.

    Paper (Table 3): hidden layers 256:256 -> n_hidden=2 (default here).
    The released code used 3 hidden layers; pass n_hidden=3 to get that back.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 256, n_hidden: int = 2) -> None:
        super().__init__()
        layers, d = [], input_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_size), nn.ReLU()]
            d = hidden_size
        layers.append(nn.Linear(d, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeterministicPolicy(nn.Module):
    """TD3 / TOP actor: tanh-squashed deterministic action."""

    def __init__(self, state_dim, action_dim, hidden_size=256, n_hidden=2):
        super().__init__()
        self.network = MLPNetwork(state_dim, action_dim, hidden_size, n_hidden)

    def forward(self, x):
        return torch.tanh(self.network(x))


class DoubleQFunc(nn.Module):
    """Two scalar critics (TD3, SAC, OAC)."""

    def __init__(self, state_dim, action_dim, hidden_size=256, n_hidden=2):
        super().__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size, n_hidden)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size, n_hidden)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class QuantileDoubleQFunc(nn.Module):
    """Two quantile critics (TOP), each outputs n_quantiles values."""

    def __init__(self, state_dim, action_dim, n_quantiles=50, hidden_size=256, n_hidden=2):
        super().__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, n_quantiles, hidden_size, n_hidden)
        self.network2 = MLPNetwork(state_dim + action_dim, n_quantiles, hidden_size, n_hidden)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


# --------------------------------------------------------------------------- replay buffer
class ReplayPool:
    """Preallocated numpy ring buffer (Ball's implementation).

    `sample` draws `batch_size` distinct indices uniformly at random, exactly like the
    original `random.sample(deque, batch_size)` (sampling WITHOUT replacement).
    """

    def __init__(self, state_dim: int, action_dim: int, capacity: int = int(1e6), seed: int = 0) -> None:
        self.capacity = int(capacity)
        self._pointer = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.nextstate = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.real_done = np.zeros((self.capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, nextstate, real_done) -> None:
        i = self._pointer
        self.state[i] = state
        self.action[i] = action
        self.reward[i] = reward
        self.nextstate[i] = nextstate
        self.real_done[i] = float(real_done)
        self._pointer = (self._pointer + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = self._rng.choice(self._size, size=batch_size, replace=False)
        return {
            k: torch.from_numpy(getattr(self, k)[idx]).to(device)
            for k in ("state", "action", "reward", "nextstate", "real_done")
        }

    def __len__(self) -> int:
        return self._size


# --------------------------------------------------------------------------- quantile loss (TOP repo, unchanged)
def calculate_huber_loss(td_errors: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))


def calculate_quantile_huber_loss(td_errors: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """td_errors: [batch, N, N'] ; taus: [1, N]."""
    assert not taus.requires_grad
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
    ) * element_wise_huber_loss / kappa
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
    return batch_quantile_huber_loss.mean()
