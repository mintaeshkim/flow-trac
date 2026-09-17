import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0
EPS = 1e-6


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-1.0 + EPS, 1.0 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class Actor(nn.Module):
    """
    TRAC tanh-squashed Gaussian policy.
    """

    def __init__(self, env: gym.vector.VectorEnv, hidden_dim: int = 256):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_sigma = nn.Linear(hidden_dim, act_dim)

        for layer in self.trunk[::2]:
            nn.init.constant_(layer.bias, 0.1)
        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = act_dim
        self.max_action = float(np.max(np.abs(env.single_action_space.high)))

    def _params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(obs.float())
        mu = self.mu(hidden)
        log_sigma = torch.clip(self.log_sigma(hidden), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_sigma

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mu, log_sigma = self._params(obs)
        dist = Normal(mu, log_sigma.exp())
        raw_action = mu if deterministic else dist.rsample()
        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.max_action

        log_prob = None
        if need_log_prob:
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + EPS).sum(dim=-1)
            log_prob = log_prob - self.action_dim * np.log(self.max_action)

        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = self._params(obs)
        dist = Normal(mu, log_sigma.exp())
        scaled_action = action / self.max_action
        raw_action = atanh(scaled_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob = log_prob - torch.log(1 - scaled_action.pow(2) + EPS).sum(dim=-1)
        log_prob = log_prob - self.action_dim * np.log(self.max_action)
        return log_prob

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        device = next(self.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        squeeze_batch = obs_t.ndim == 1
        if squeeze_batch:
            obs_t = obs_t.unsqueeze(0)

        action = self(obs_t, deterministic=deterministic)[0].cpu().numpy().astype(np.float32)
        return action[0] if squeeze_batch else action
