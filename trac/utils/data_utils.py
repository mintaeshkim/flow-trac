from typing import Dict

import gymnasium as gym
import numpy as np
from trac.utils.obs_utils import flatten_observation_batch, select_observation_batch


def minari_dataset_to_transitions(
    dataset,
    observation_space: gym.spaces.Space | None = None,
    flatten_observations: bool = False,
    observation_key: str | tuple[str, ...] | list[str] | None = None,
) -> Dict[str, np.ndarray]:
    observations, actions, rewards = [], [], []
    next_observations, terminations, truncations = [], [], []

    for episode in dataset.iterate_episodes():
        raw_obs = episode.observations
        act = np.asarray(episode.actions)
        rew = np.asarray(episode.rewards)

        if hasattr(episode, "terminations") and episode.terminations is not None:
            term = np.asarray(episode.terminations)
        elif hasattr(episode, "terminated") and episode.terminated is not None:
            term = np.asarray(episode.terminated)
        else:
            term = np.zeros_like(rew, dtype=bool)
            term[-1] = True

        if hasattr(episode, "truncations") and episode.truncations is not None:
            trunc = np.asarray(episode.truncations)
        elif hasattr(episode, "truncated") and episode.truncated is not None:
            trunc = np.asarray(episode.truncated)
        else:
            trunc = np.zeros_like(rew, dtype=bool)

        if flatten_observations:
            if observation_space is None:
                raise ValueError("observation_space is required when flatten_observations=True.")
            if observation_key is None:
                obs = flatten_observation_batch(observation_space, raw_obs)
            else:
                obs = select_observation_batch(observation_space, raw_obs, observation_key)
        else:
            if isinstance(raw_obs, dict):
                raise ValueError(
                    "Dict observations require flatten_observations=True and observation_space."
                )
            obs = np.asarray(raw_obs)

        if obs.shape[0] == act.shape[0] + 1:
            cur_obs = obs[:-1]
            next_obs = obs[1:]
        elif obs.shape[0] == act.shape[0]:
            cur_obs = obs[:-1]
            next_obs = obs[1:]
            act = act[:-1]
            rew = rew[:-1]
            term = term[:-1]
            trunc = trunc[:-1]
        else:
            raise ValueError(f"Unexpected episode shapes: observations={obs.shape}, actions={act.shape}")

        observations.append(cur_obs)
        next_observations.append(next_obs)
        actions.append(act)
        rewards.append(rew)
        terminations.append(term.astype(np.float32))
        truncations.append(trunc.astype(np.float32))

    return {
        "observations": np.concatenate(observations, axis=0).astype(np.float32),
        "actions": np.concatenate(actions, axis=0).astype(np.float32),
        "rewards": np.concatenate(rewards, axis=0).astype(np.float32),
        "next_observations": np.concatenate(next_observations, axis=0).astype(np.float32),
        "terminations": np.concatenate(terminations, axis=0).astype(np.float32),
        "truncations": np.concatenate(truncations, axis=0).astype(np.float32),
    }


def validate_dataset_shapes(transitions: dict, envs: gym.vector.VectorEnv):
    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

    if transitions["observations"].shape[1:] != obs_shape:
        raise ValueError(
            f"Dataset observation shape {transitions['observations'].shape[1:]} "
            f"does not match env observation shape {obs_shape}."
        )
    if transitions["actions"].shape[1:] != action_shape:
        raise ValueError(
            f"Dataset action shape {transitions['actions'].shape[1:]} "
            f"does not match env action shape {action_shape}."
        )


def return_reward_range(transitions: dict, max_episode_steps: int = 1000):
    returns, lengths = [], []
    ep_return, ep_length = 0.0, 0
    dones = np.logical_or(transitions["terminations"], transitions["truncations"])

    for reward, done in zip(transitions["rewards"], dones):
        ep_return += float(reward)
        ep_length += 1
        if done or ep_length == max_episode_steps:
            returns.append(ep_return)
            lengths.append(ep_length)
            ep_return, ep_length = 0.0, 0

    if ep_length > 0:
        returns.append(ep_return)
        lengths.append(ep_length)

    assert sum(lengths) == len(transitions["rewards"])
    return min(returns), max(returns)


def normalize_dataset_reward(transitions: dict, dataset_name: str, max_episode_steps: int = 1000):
    name = dataset_name.lower()
    if any(task_name in name for task_name in ("halfcheetah", "hopper", "walker2d")):
        min_return, max_return = return_reward_range(transitions, max_episode_steps)
        transitions["rewards"] /= max_return - min_return
        transitions["rewards"] *= max_episode_steps
    elif "antmaze" in name:
        transitions["rewards"] -= 1.0


def make_n_step_transitions(
    transitions: dict,
    n_step: int,
    gamma: float,
    use_truncations_as_dones: bool = False,
) -> tuple[dict, dict[str, float]]:
    """
    Convert one-step transitions into fixed n-step Bellman transitions.

    The output keeps the original (s_t, a_t) pairs, replaces rewards with
    discounted n-step returns, and replaces next_observations with s_{t+n}
    unless an episode boundary is reached first.
    """
    if n_step < 1:
        raise ValueError("n_step must be >= 1.")
    if n_step == 1:
        return transitions, {
            "n_step": 1.0,
            "n_step_effective_mean": 1.0,
            "n_step_done_fraction": float(
                np.logical_or(
                    transitions["terminations"].astype(bool),
                    transitions["truncations"].astype(bool)
                    if use_truncations_as_dones
                    else np.zeros_like(transitions["terminations"], dtype=bool),
                ).mean()
            ),
        }

    observations = transitions["observations"].astype(np.float32)
    actions = transitions["actions"].astype(np.float32)
    rewards = transitions["rewards"].astype(np.float32)
    next_observations = transitions["next_observations"].astype(np.float32)
    terminations = transitions["terminations"].astype(bool)
    truncations = transitions["truncations"].astype(bool)
    boundaries = terminations | (truncations if use_truncations_as_dones else False)

    num_transitions = rewards.shape[0]
    n_step_rewards = np.zeros_like(rewards, dtype=np.float32)
    n_step_next_observations = np.zeros_like(next_observations, dtype=np.float32)
    n_step_terminations = np.zeros_like(rewards, dtype=np.float32)
    n_step_truncations = np.zeros_like(rewards, dtype=np.float32)
    effective_steps = np.zeros_like(rewards, dtype=np.float32)

    for start_idx in range(num_transitions):
        discounted_return = 0.0
        discount = 1.0
        end_idx = start_idx
        hit_boundary = False

        for step_idx in range(n_step):
            idx = start_idx + step_idx
            if idx >= num_transitions:
                hit_boundary = True
                n_step_truncations[start_idx] = 1.0
                break

            discounted_return += discount * float(rewards[idx])
            end_idx = idx
            if boundaries[idx]:
                hit_boundary = True
                n_step_terminations[start_idx] = float(terminations[idx])
                n_step_truncations[start_idx] = float(truncations[idx])
                break
            discount *= gamma

        n_step_rewards[start_idx] = discounted_return
        n_step_next_observations[start_idx] = next_observations[end_idx]
        effective_steps[start_idx] = float(end_idx - start_idx + 1)
        if not hit_boundary:
            n_step_terminations[start_idx] = 0.0
            n_step_truncations[start_idx] = 0.0

    n_step_transitions = {
        "observations": observations,
        "actions": actions,
        "rewards": n_step_rewards.astype(np.float32),
        "next_observations": n_step_next_observations.astype(np.float32),
        "terminations": n_step_terminations.astype(np.float32),
        "truncations": n_step_truncations.astype(np.float32),
    }
    done_flags = np.logical_or(
        n_step_transitions["terminations"].astype(bool),
        n_step_transitions["truncations"].astype(bool)
        if use_truncations_as_dones
        else np.zeros_like(n_step_transitions["terminations"], dtype=bool),
    )
    info = {
        "n_step": float(n_step),
        "n_step_effective_mean": float(effective_steps.mean()),
        "n_step_effective_min": float(effective_steps.min()),
        "n_step_effective_max": float(effective_steps.max()),
        "n_step_done_fraction": float(done_flags.mean()),
        "n_step_reward_mean": float(n_step_rewards.mean()),
        "n_step_reward_std": float(n_step_rewards.std()),
        "n_step_bootstrap_gamma": float(gamma ** n_step),
    }
    return n_step_transitions, info


def compute_obs_mean_std(transitions: dict, eps: float = 1e-3):
    obs_mean = transitions["observations"].mean(axis=0)
    obs_std = transitions["observations"].std(axis=0) + eps
    return obs_mean.astype(np.float32), obs_std.astype(np.float32)


def normalize_dataset_obs(
    transitions: dict,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
) -> dict:
    transitions["observations"] = (
        transitions["observations"] - obs_mean
    ) / obs_std
    transitions["next_observations"] = (
        transitions["next_observations"] - obs_mean
    ) / obs_std
    return transitions


def wrap_obs_normalization(env, obs_mean, obs_std):
    observation_space = env.observation_space
    observation_space = gym.spaces.Box(
        low=((observation_space.low - obs_mean) / obs_std).astype(np.float32),
        high=((observation_space.high - obs_mean) / obs_std).astype(np.float32),
        shape=observation_space.shape,
        dtype=np.float32,
    )
    return gym.wrappers.TransformObservation(
        env,
        lambda obs: (obs - obs_mean) / obs_std,
        observation_space=observation_space,
    )
