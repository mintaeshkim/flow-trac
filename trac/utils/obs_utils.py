# trac/utils/obs_utils.py
from __future__ import annotations

from collections.abc import Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.spaces import utils as space_utils

ObservationKey = str | Sequence[str] | None


def flatten_observation_space(observation_space: gym.spaces.Space) -> gym.spaces.Box:
    return space_utils.flatten_space(observation_space)


def flatten_observation(
    observation_space: gym.spaces.Space,
    observation,
) -> np.ndarray:
    return space_utils.flatten(observation_space, observation).astype(np.float32)


def flatten_observation_batch(
    observation_space: gym.spaces.Space,
    observations,
) -> np.ndarray:
    """
    Flatten a batch/sequence of observations using Gymnasium's space order.

    Handles Box observations, Dict observations represented as dict-of-arrays,
    and sequences/object arrays of individual dict observations.
    """

    if isinstance(observation_space, gym.spaces.Box):
        arr = np.asarray(observations, dtype=np.float32)
        if arr.shape == observation_space.shape:
            return arr.reshape(1, -1)
        return arr.reshape(arr.shape[0], -1)

    if isinstance(observation_space, gym.spaces.Dict):
        parts = []
        for key, subspace in observation_space.spaces.items():
            if isinstance(observations, Mapping):
                sub_observations = observations[key]
            else:
                sub_observations = [obs[key] for obs in observations]
            parts.append(flatten_observation_batch(subspace, sub_observations))
        return np.concatenate(parts, axis=-1).astype(np.float32)

    if isinstance(observation_space, gym.spaces.Tuple):
        parts = []
        for idx, subspace in enumerate(observation_space.spaces):
            sub_observations = [obs[idx] for obs in observations]
            parts.append(flatten_observation_batch(subspace, sub_observations))
        return np.concatenate(parts, axis=-1).astype(np.float32)

    if isinstance(observations, Sequence) and not isinstance(observations, np.ndarray):
        return np.stack(
            [flatten_observation(observation_space, obs) for obs in observations],
            axis=0,
        ).astype(np.float32)

    arr = np.asarray(observations)
    if arr.dtype == object:
        return np.stack(
            [flatten_observation(observation_space, obs) for obs in arr],
            axis=0,
        ).astype(np.float32)

    raise TypeError(f"Unsupported observation space for flattening: {observation_space}")


def _normalize_observation_keys(observation_key: ObservationKey) -> tuple[str, ...] | None:
    if observation_key is None:
        return None
    if isinstance(observation_key, str):
        return (observation_key,)
    return tuple(observation_key)


def _concat_box_spaces(spaces: Sequence[gym.spaces.Space]) -> gym.spaces.Box:
    flat_spaces = [space_utils.flatten_space(space) for space in spaces]
    lows = [space.low.reshape(-1) for space in flat_spaces]
    highs = [space.high.reshape(-1) for space in flat_spaces]
    return gym.spaces.Box(
        low=np.concatenate(lows, axis=0).astype(np.float32),
        high=np.concatenate(highs, axis=0).astype(np.float32),
        dtype=np.float32,
    )


def select_observation_space(
    observation_space: gym.spaces.Space,
    observation_key: ObservationKey,
) -> gym.spaces.Space:
    observation_keys = _normalize_observation_keys(observation_key)
    if observation_keys is None:
        return observation_space
    if not isinstance(observation_space, gym.spaces.Dict):
        raise TypeError(
            f"observation_key={observation_key!r} requires a Dict observation space, "
            f"got {observation_space}."
        )
    missing_keys = [
        key for key in observation_keys if key not in observation_space.spaces
    ]
    if missing_keys:
        raise KeyError(
            f"Observation keys {missing_keys!r} not found in {list(observation_space.spaces.keys())}."
        )
    if len(observation_keys) == 1:
        return observation_space.spaces[observation_keys[0]]
    return _concat_box_spaces([observation_space.spaces[key] for key in observation_keys])


def select_observation_value(observations, observation_key: ObservationKey):
    observation_keys = _normalize_observation_keys(observation_key)
    if observation_keys is None:
        return observations
    if len(observation_keys) == 1:
        key = observation_keys[0]
        if isinstance(observations, Mapping):
            return observations[key]
        return [obs[key] for obs in observations]

    def flatten_selected(obs):
        parts = [
            flatten_observation_batch(
                select_observation_space(gym.spaces.Box(-np.inf, np.inf, shape=np.asarray(obs[key]).shape[-1:], dtype=np.float32), None),
                np.asarray(obs[key], dtype=np.float32),
            )
            for key in observation_keys
        ]
        return np.concatenate(parts, axis=-1).squeeze(0)

    if isinstance(observations, Mapping):
        parts = [np.asarray(observations[key], dtype=np.float32) for key in observation_keys]
        return np.concatenate([part.reshape(part.shape[0], -1) for part in parts], axis=-1)
    return [flatten_selected(obs) for obs in observations]


def select_observation_batch(
    observation_space: gym.spaces.Space,
    observations,
    observation_key: ObservationKey,
) -> np.ndarray:
    selected_space = select_observation_space(observation_space, observation_key)
    selected_observations = select_observation_value(observations, observation_key)
    return flatten_observation_batch(selected_space, selected_observations)


def select_observation_env(env: gym.Env, observation_key: ObservationKey) -> gym.Env:
    observation_keys = _normalize_observation_keys(observation_key)
    if observation_keys is None:
        return env
    observation_space = select_observation_space(env.observation_space, observation_keys)

    if len(observation_keys) == 1:
        key = observation_keys[0]

        def transform(obs):
            return obs[key]

    else:
        def transform(obs):
            return np.concatenate(
                [
                    flatten_observation_batch(
                        env.observation_space.spaces[key],
                        obs[key],
                    ).reshape(-1)
                    for key in observation_keys
                ],
                axis=0,
            ).astype(np.float32)

    return gym.wrappers.TransformObservation(
        env,
        transform,
        observation_space=observation_space,
    )


def flatten_env(env: gym.Env, observation_key: ObservationKey = None) -> gym.Env:
    env = select_observation_env(env, observation_key)
    if isinstance(env.observation_space, gym.spaces.Dict):
        return gym.wrappers.FlattenObservation(env)
    return env
