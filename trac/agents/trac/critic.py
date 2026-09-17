# trac/agents/trac/critic.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, env: gym.vector.VectorEnv, hidden_dim: int = 256):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))

        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs.float(), action.float()], dim=-1))


class Value(nn.Module):
    def __init__(self, env: gym.vector.VectorEnv, hidden_dim: int = 256):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())
