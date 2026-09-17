from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    """Fixed offline replay buffer preserving the reference sampler."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: torch.device,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.n_envs = n_envs
        self.obs = np.zeros(
            (buffer_size, n_envs, *observation_space.shape),
            dtype=np.float32,
        )
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros(
            (buffer_size, n_envs, *action_space.shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        max_idx = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, max_idx, size=batch_size)
        env_inds = np.random.randint(0, self.n_envs, size=batch_size)

        obs = self.obs[batch_inds, env_inds]
        next_obs = self.next_obs[batch_inds, env_inds]
        actions = self.actions[batch_inds, env_inds]
        rewards = self.rewards[batch_inds, env_inds].reshape(-1, 1)
        dones = self.dones[batch_inds, env_inds].reshape(-1, 1)

        return ReplayBufferSamples(
            torch.tensor(obs, device=self.device, dtype=torch.float32),
            torch.tensor(actions, device=self.device, dtype=torch.float32),
            torch.tensor(next_obs, device=self.device, dtype=torch.float32),
            torch.tensor(dones, device=self.device, dtype=torch.float32),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
        )

    def load_transitions(self, transitions: dict, use_truncations_as_dones: bool = False):
        """
        Load a fixed offline dataset into the replay buffer.
        Expected keys are the same as `minari_dataset_to_transitions`:
        observations, actions, rewards, next_observations, terminations, truncations.
        """
        observations = transitions["observations"].astype(np.float32)
        actions = transitions["actions"].astype(np.float32)
        rewards = transitions["rewards"].astype(np.float32)
        next_observations = transitions["next_observations"].astype(np.float32)
        terminations = transitions["terminations"].astype(np.float32)
        truncations = transitions["truncations"].astype(np.float32)

        n_transitions = observations.shape[0]
        if self.n_envs != 1:
            raise ValueError("Offline dataset loading expects n_envs=1.")
        if self.pos != 0 or self.full:
            raise ValueError("Trying to load data into a non-empty replay buffer.")
        if n_transitions > self.buffer_size:
            raise ValueError(
                f"Replay buffer size {self.buffer_size} is smaller than dataset size {n_transitions}."
            )

        dones = terminations
        if use_truncations_as_dones:
            dones = np.logical_or(terminations, truncations).astype(np.float32)

        self.obs[:n_transitions, 0] = observations
        self.actions[:n_transitions, 0] = actions
        self.rewards[:n_transitions, 0] = rewards
        self.next_obs[:n_transitions, 0] = next_observations
        self.dones[:n_transitions, 0] = dones

        self.pos = n_transitions
        self.full = n_transitions == self.buffer_size

    def size(self) -> int:
        return self.buffer_size if self.full else self.pos
