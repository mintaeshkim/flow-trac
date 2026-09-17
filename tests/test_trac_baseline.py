from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from trac.agents.trac import TRACAgent, TRACConfig
from trac.train.train_kitchen import Args
from trac.utils.buffers import ReplayBuffer


def make_env_spec():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(3,),
            dtype=np.float32,
        ),
        single_action_space=gym.spaces.Box(
            -1.0,
            1.0,
            shape=(2,),
            dtype=np.float32,
        ),
    )


def make_buffer(env_spec, size: int = 32) -> ReplayBuffer:
    buffer = ReplayBuffer(
        buffer_size=size,
        observation_space=env_spec.single_observation_space,
        action_space=env_spec.single_action_space,
        device=torch.device("cpu"),
    )
    buffer.load_transitions(
        {
            "observations": np.random.randn(size, 3).astype(np.float32),
            "actions": np.random.uniform(-0.9, 0.9, (size, 2)).astype(np.float32),
            "rewards": np.random.randn(size).astype(np.float32),
            "next_observations": np.random.randn(size, 3).astype(np.float32),
            "terminations": np.zeros(size, dtype=np.float32),
            "truncations": np.zeros(size, dtype=np.float32),
        }
    )
    return buffer


def test_reproduced_trac_pretrains_prior_then_warm_starts_actor():
    torch.manual_seed(0)
    env_spec = make_env_spec()
    agent = TRACAgent(
        env_spec,
        TRACConfig(
            hidden_dim=16,
            num_value_samples=4,
            prior_pretrain_steps=1,
            prior_update_steps=1,
            actor_start_steps=1,
            actor_warm_start_from_prior=True,
            policy_frequency=1,
            cql_alpha=1.0,
            cql_num_actions=4,
            use_actor_ema=True,
        ),
        torch.device("cpu"),
    )
    buffer = make_buffer(env_spec)

    warmup_metrics = agent.update(buffer, batch_size=8, global_step=0)
    train_metrics = agent.update(buffer, batch_size=8, global_step=1)

    assert warmup_metrics["warmup/prior_only"] == 1.0
    assert train_metrics["warmup/actor_warm_started_now"] == 1.0
    assert "critic_loss" in train_metrics
    assert "actor_loss" in train_metrics
    assert agent.actor_ema is not None
    assert "value" not in agent.state_dict()


def test_kitchen_defaults_match_reference_run():
    args = Args()

    assert args.exp_name == "train_trac_kitchen"
    assert args.cql_alpha == 0.1
    assert args.prior_pretrain_steps == 50_000
    assert args.actor_start_steps == 50_000
    assert args.use_actor_ema is True
    assert args.n_step == 1
