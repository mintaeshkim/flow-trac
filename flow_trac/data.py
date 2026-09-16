from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import utils as space_utils


@dataclass(frozen=True)
class Batch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


def flatten_observation_batch(
    observation_space: gym.spaces.Space,
    observations,
) -> np.ndarray:
    if isinstance(observation_space, gym.spaces.Box):
        array = np.asarray(observations, dtype=np.float32)
        if array.shape == observation_space.shape:
            return array.reshape(1, -1)
        return array.reshape(array.shape[0], -1)

    if isinstance(observation_space, gym.spaces.Dict):
        parts = []
        for key, subspace in observation_space.spaces.items():
            values = (
                observations[key]
                if isinstance(observations, Mapping)
                else [obs[key] for obs in observations]
            )
            parts.append(flatten_observation_batch(subspace, values))
        return np.concatenate(parts, axis=-1).astype(np.float32)

    if isinstance(observation_space, gym.spaces.Tuple):
        parts = [
            flatten_observation_batch(subspace, [obs[index] for obs in observations])
            for index, subspace in enumerate(observation_space.spaces)
        ]
        return np.concatenate(parts, axis=-1).astype(np.float32)

    if isinstance(observations, Sequence):
        return np.stack(
            [space_utils.flatten(observation_space, obs) for obs in observations]
        ).astype(np.float32)
    raise TypeError(f"Unsupported observation space: {observation_space}")


def _select_observations(observation_space, observations, observation_key: str | None):
    if observation_key is None:
        return observation_space, observations
    if not isinstance(observation_space, gym.spaces.Dict):
        raise TypeError(f"observation_key={observation_key!r} requires a Dict observation space.")
    if observation_key not in observation_space.spaces:
        raise KeyError(
            f"Observation key {observation_key!r} is not in {list(observation_space.spaces)}."
        )
    selected = (
        observations[observation_key]
        if isinstance(observations, Mapping)
        else [obs[observation_key] for obs in observations]
    )
    return observation_space.spaces[observation_key], selected


def minari_to_transitions(
    dataset,
    observation_space: gym.spaces.Space,
    observation_key: str | None = None,
) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "terminations": [],
        "truncations": [],
    }

    for episode in dataset.iterate_episodes():
        selected_space, raw_observations = _select_observations(
            observation_space,
            episode.observations,
            observation_key,
        )
        observations = flatten_observation_batch(selected_space, raw_observations)
        actions = np.asarray(episode.actions, dtype=np.float32)
        actions = actions.reshape(actions.shape[0], -1)
        rewards = np.asarray(episode.rewards, dtype=np.float32)
        terminations = np.asarray(
            getattr(episode, "terminations", np.zeros_like(rewards)),
            dtype=np.float32,
        )
        truncations = np.asarray(
            getattr(episode, "truncations", np.zeros_like(rewards)),
            dtype=np.float32,
        )

        if observations.shape[0] == actions.shape[0] + 1:
            current_observations = observations[:-1]
            next_observations = observations[1:]
        elif observations.shape[0] == actions.shape[0]:
            current_observations = observations[:-1]
            next_observations = observations[1:]
            actions = actions[:-1]
            rewards = rewards[:-1]
            terminations = terminations[:-1]
            truncations = truncations[:-1]
        else:
            raise ValueError(
                "Unexpected episode lengths: "
                f"observations={observations.shape[0]}, actions={actions.shape[0]}."
            )

        fields["observations"].append(current_observations)
        fields["actions"].append(actions)
        fields["rewards"].append(rewards)
        fields["next_observations"].append(next_observations)
        fields["terminations"].append(terminations)
        fields["truncations"].append(truncations)

    if not fields["observations"]:
        raise ValueError("The Minari dataset contains no episodes.")
    return {
        key: np.concatenate(values, axis=0).astype(np.float32) for key, values in fields.items()
    }


def observation_stats(
    transitions: dict[str, np.ndarray],
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    observations = transitions["observations"]
    return observations.mean(axis=0).astype(np.float32), (observations.std(axis=0) + eps).astype(
        np.float32
    )


def normalize_observations(
    transitions: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    transitions["observations"] = (transitions["observations"] - mean) / std
    transitions["next_observations"] = (transitions["next_observations"] - mean) / std


def normalize_rewards(transitions: dict[str, np.ndarray], dataset_name: str) -> None:
    name = dataset_name.lower()
    if "antmaze" in name:
        transitions["rewards"] -= 1.0
        return
    if not any(task in name for task in ("halfcheetah", "hopper", "walker2d")):
        return

    done = np.logical_or(
        transitions["terminations"].astype(bool),
        transitions["truncations"].astype(bool),
    )
    episode_returns = []
    episode_return = 0.0
    for reward, boundary in zip(transitions["rewards"], done):
        episode_return += float(reward)
        if boundary:
            episode_returns.append(episode_return)
            episode_return = 0.0
    if episode_return or not episode_returns:
        episode_returns.append(episode_return)
    return_range = max(episode_returns) - min(episode_returns)
    if return_range > 0:
        transitions["rewards"] *= 1000.0 / return_range


class OfflineDataset:
    def __init__(
        self,
        transitions: dict[str, np.ndarray],
        device: torch.device,
        use_truncations_as_dones: bool = False,
        seed: int = 0,
    ):
        self.transitions = transitions
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.num_samples = len(transitions["observations"])
        terminations = transitions["terminations"].astype(bool)
        truncations = transitions["truncations"].astype(bool)
        self.dones = (
            np.logical_or(terminations, truncations) if use_truncations_as_dones else terminations
        ).astype(np.float32)

    def sample(self, batch_size: int, indices: np.ndarray | None = None) -> Batch:
        population = self.num_samples if indices is None else len(indices)
        sampled = self.rng.integers(0, population, size=batch_size)
        batch_indices = sampled if indices is None else indices[sampled]

        def tensor(key: str) -> torch.Tensor:
            return torch.as_tensor(
                self.transitions[key][batch_indices],
                dtype=torch.float32,
                device=self.device,
            )

        return Batch(
            observations=tensor("observations"),
            actions=tensor("actions"),
            rewards=tensor("rewards").reshape(-1, 1),
            next_observations=tensor("next_observations"),
            dones=torch.as_tensor(
                self.dones[batch_indices],
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1, 1),
        )

    def train_validation_indices(
        self,
        validation_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        permutation = self.rng.permutation(self.num_samples)
        validation_size = int(round(self.num_samples * validation_fraction))
        if validation_fraction > 0.0:
            validation_size = max(1, validation_size)
        return permutation[validation_size:], permutation[:validation_size]


def wrap_environment(
    env: gym.Env,
    observation_key: str | None,
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
) -> gym.Env:
    if observation_key is not None:
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError(
                f"observation_key={observation_key!r} requires a Dict observation space."
            )
        selected_space = env.observation_space.spaces[observation_key]
        env = gym.wrappers.TransformObservation(
            env,
            lambda observation: observation[observation_key],
            observation_space=selected_space,
        )
    env = gym.wrappers.FlattenObservation(env)
    normalized_space = gym.spaces.Box(
        low=((env.observation_space.low - observation_mean) / observation_std).astype(np.float32),
        high=((env.observation_space.high - observation_mean) / observation_std).astype(np.float32),
        dtype=np.float32,
    )
    return gym.wrappers.TransformObservation(
        env,
        lambda observation: (np.asarray(observation, dtype=np.float32) - observation_mean)
        / observation_std,
        observation_space=normalized_space,
    )
