# trac/utils/buffers.py
from dataclasses import dataclass
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


class BaseReplayBuffer:
    """Base class for replay buffers with vectorized environments."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape,
        action_shape,
        device: torch.device,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.n_envs = n_envs

        self.obs = np.zeros((buffer_size, n_envs, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, n_envs, *obs_shape), dtype=np.float32)
        self.actions = (
            np.zeros((buffer_size, n_envs, *action_shape), dtype=np.float32)
            if action_shape is not None
            else None
        )
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, actions, rewards, dones):
        """Append a batch of transitions (vector envs)."""
        self.obs[self.pos] = obs.astype(np.float32)
        self.next_obs[self.pos] = next_obs.astype(np.float32)

        if self.actions is not None:
            self.actions[self.pos] = actions.astype(np.float32)

        self.rewards[self.pos] = rewards.astype(np.float32)
        self.dones[self.pos] = dones.astype(np.float32)

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def _sample_indices(self, batch_size: int):
        max_idx = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, max_idx, size=batch_size)
        env_inds = np.random.randint(0, self.n_envs, size=batch_size)
        return batch_inds, env_inds

    def can_sample(self, batch_size: int) -> bool:
        """Return True if enough samples are stored."""
        max_idx = self.buffer_size if self.full else self.pos
        return max_idx * self.n_envs >= batch_size and max_idx > 0


class ReplayBuffer(BaseReplayBuffer):
    """Standard off-policy replay buffer for SAC/TD3/DQN."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: torch.device,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size,
            observation_space.shape,
            action_space.shape,
            device,
            n_envs,
        )

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        batch_inds, env_inds = self._sample_indices(batch_size)

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

    def add_transitions(self, transitions: dict, use_truncations_as_dones: bool = False):
        """
        Append a transition dict into the buffer using ring-buffer semantics.

        This is intended for generated/model transitions whose size can change
        throughout training. Existing samples are overwritten when the buffer is
        full.
        """
        observations = transitions["observations"].astype(np.float32)
        actions = transitions["actions"].astype(np.float32)
        rewards = transitions["rewards"].astype(np.float32)
        next_observations = transitions["next_observations"].astype(np.float32)
        terminations = transitions["terminations"].astype(np.float32)
        truncations = transitions["truncations"].astype(np.float32)

        n_transitions = observations.shape[0]
        if self.n_envs != 1:
            raise ValueError("Transition dict loading expects n_envs=1.")
        if n_transitions == 0:
            return

        dones = terminations
        if use_truncations_as_dones:
            dones = np.logical_or(terminations, truncations).astype(np.float32)

        for idx in range(n_transitions):
            self.obs[self.pos, 0] = observations[idx]
            self.actions[self.pos, 0] = actions[idx]
            self.rewards[self.pos, 0] = rewards[idx]
            self.next_obs[self.pos, 0] = next_observations[idx]
            self.dones[self.pos, 0] = dones[idx]

            self.pos += 1
            if self.pos >= self.buffer_size:
                self.full = True
                self.pos = 0

    def clear(self):
        self.pos = 0
        self.full = False

    def size(self) -> int:
        return self.buffer_size if self.full else self.pos


@dataclass
class MixedReplayBufferMetrics:
    offline_batch_size: int
    imagined_batch_size: int
    offline_size: int
    imagined_size: int
    imagined_ratio: float


class MixedReplayBuffer:
    """
    Samples a batch from an offline buffer and an imagined buffer.

    The returned object has the same ReplayBufferSamples format expected by
    existing agents, so agents do not need to know whether model data is used.
    """

    def __init__(
        self,
        offline_buffer: ReplayBuffer,
        imagined_buffer: ReplayBuffer,
        imagined_batch_ratio: float = 0.25,
    ):
        if not 0.0 <= imagined_batch_ratio <= 1.0:
            raise ValueError("imagined_batch_ratio must be in [0, 1].")
        self.offline_buffer = offline_buffer
        self.imagined_buffer = imagined_buffer
        self.imagined_batch_ratio = imagined_batch_ratio
        self.last_metrics = MixedReplayBufferMetrics(
            offline_batch_size=0,
            imagined_batch_size=0,
            offline_size=offline_buffer.size(),
            imagined_size=imagined_buffer.size(),
            imagined_ratio=imagined_batch_ratio,
        )

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        imagined_batch_size = int(round(batch_size * self.imagined_batch_ratio))
        if not self.imagined_buffer.can_sample(max(imagined_batch_size, 1)):
            imagined_batch_size = 0
        imagined_batch_size = min(imagined_batch_size, batch_size)
        offline_batch_size = batch_size - imagined_batch_size

        if offline_batch_size <= 0 and imagined_batch_size > 0:
            samples = self.imagined_buffer.sample(imagined_batch_size)
            self._set_metrics(0, imagined_batch_size)
            return samples
        if imagined_batch_size <= 0:
            samples = self.offline_buffer.sample(batch_size)
            self._set_metrics(batch_size, 0)
            return samples

        offline_samples = self.offline_buffer.sample(offline_batch_size)
        imagined_samples = self.imagined_buffer.sample(imagined_batch_size)
        samples = ReplayBufferSamples(
            observations=torch.cat([offline_samples.observations, imagined_samples.observations], dim=0),
            actions=torch.cat([offline_samples.actions, imagined_samples.actions], dim=0),
            next_observations=torch.cat(
                [offline_samples.next_observations, imagined_samples.next_observations],
                dim=0,
            ),
            dones=torch.cat([offline_samples.dones, imagined_samples.dones], dim=0),
            rewards=torch.cat([offline_samples.rewards, imagined_samples.rewards], dim=0),
        )
        permutation = torch.randperm(batch_size, device=samples.observations.device)
        samples = ReplayBufferSamples(
            observations=samples.observations[permutation],
            actions=samples.actions[permutation],
            next_observations=samples.next_observations[permutation],
            dones=samples.dones[permutation],
            rewards=samples.rewards[permutation],
        )
        self._set_metrics(offline_batch_size, imagined_batch_size)
        return samples

    def _set_metrics(self, offline_batch_size: int, imagined_batch_size: int):
        total = max(offline_batch_size + imagined_batch_size, 1)
        self.last_metrics = MixedReplayBufferMetrics(
            offline_batch_size=offline_batch_size,
            imagined_batch_size=imagined_batch_size,
            offline_size=self.offline_buffer.size(),
            imagined_size=self.imagined_buffer.size(),
            imagined_ratio=imagined_batch_size / total,
        )

    def metrics(self) -> dict[str, float]:
        return {
            "idi/offline_batch_size": float(self.last_metrics.offline_batch_size),
            "idi/imagined_batch_size": float(self.last_metrics.imagined_batch_size),
            "idi/offline_buffer_size": float(self.last_metrics.offline_size),
            "idi/imagined_buffer_size": float(self.last_metrics.imagined_size),
            "idi/actual_imagined_batch_ratio": float(self.last_metrics.imagined_ratio),
        }

    def size(self) -> int:
        return self.offline_buffer.size() + self.imagined_buffer.size()
