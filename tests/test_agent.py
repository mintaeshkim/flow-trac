import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from flow_trac.agent import FlowTRACAgent, FlowTRACConfig
from flow_trac.data import Batch
from flow_trac.train import Args


class ConstantCritic(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0], 1), self.value, device=obs.device)


def make_agent(**overrides) -> FlowTRACAgent:
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    config_values = {
        "hidden_dim": 16,
        "num_value_samples": 4,
        "actor_num_candidates": 4,
        "flow_steps": 2,
        "critic_warmup_steps": 0,
        "policy_frequency": 1,
        **overrides,
    }
    config = FlowTRACConfig(**config_values)
    return FlowTRACAgent(
        observation_space,
        action_space,
        config,
        torch.device("cpu"),
    )


def make_batch(batch_size: int = 16) -> Batch:
    return Batch(
        observations=torch.randn(batch_size, 3),
        actions=torch.empty(batch_size, 2).uniform_(-0.9, 0.9),
        rewards=torch.randn(batch_size, 1),
        next_observations=torch.randn(batch_size, 3),
        dones=torch.zeros(batch_size, 1),
    )


def test_trac_value_is_exact_for_constant_q():
    agent = make_agent()
    agent.critic_1_target = ConstantCritic(3.25)
    agent.critic_2_target = ConstantCritic(4.0)
    value, _ = agent._target_value(torch.randn(8, 3))
    torch.testing.assert_close(value, torch.full((8, 1), 3.25))


def test_behavior_is_unchanged_after_freeze_and_rl_update():
    torch.manual_seed(0)
    agent = make_agent()
    batch = make_batch()
    pretrain_metrics = agent.pretrain_behavior(batch)
    agent.freeze_behavior()
    frozen_parameters = {
        name: parameter.detach().clone() for name, parameter in agent.behavior.named_parameters()
    }
    metrics = agent.update(batch, global_step=0)

    assert "flow_actor/ess_fraction" in metrics
    assert "flow_actor/flow_loss" in metrics
    assert "flow_actor/readout_loss" in metrics
    assert "behavior/flow_grad_norm" in pretrain_metrics
    assert "behavior/readout_grad_norm" in pretrain_metrics
    assert "flow_actor/flow_grad_norm" in metrics
    assert "flow_actor/readout_grad_norm" in metrics
    for name, parameter in agent.behavior.named_parameters():
        assert parameter.requires_grad is False
        torch.testing.assert_close(parameter, frozen_parameters[name], rtol=0, atol=0)


def test_resampling_metrics_have_expected_ranges():
    agent = make_agent(actor_num_candidates=8)
    agent.freeze_behavior()
    _, metrics = agent._resampled_actor_targets(torch.randn(12, 3))
    assert 1.0 <= metrics["flow_actor/ess"] <= 8.0
    assert 1.0 / 8.0 <= metrics["flow_actor/ess_fraction"] <= 1.0
    assert 1.0 / 8.0 <= metrics["flow_actor/weight_max"] <= 1.0
    assert metrics["flow_actor/action_diversity"] >= 0.0


def test_weighted_actor_uses_all_candidates():
    torch.manual_seed(0)
    agent = make_agent(actor_mode="weighted", actor_num_candidates=8)
    agent.freeze_behavior()
    batch = make_batch(batch_size=4)

    metrics = agent.update(batch, global_step=0)

    assert np.isfinite(metrics["flow_actor/flow_loss"])
    assert metrics["flow_actor/weighted_objective"] == 1.0
    assert 1.0 <= metrics["flow_actor/ess"] <= 8.0


def test_actor_updates_can_be_disabled():
    torch.manual_seed(0)
    agent = make_agent(actor_updates=False)
    agent.freeze_behavior()
    actor_parameters = {
        name: parameter.detach().clone() for name, parameter in agent.actor.named_parameters()
    }

    metrics = agent.update(make_batch(batch_size=4), global_step=100_000)

    assert metrics["warmup/actor_enabled"] == 0.0
    assert "flow_actor/flow_loss" not in metrics
    for name, parameter in agent.actor.named_parameters():
        torch.testing.assert_close(parameter, actor_parameters[name], rtol=0, atol=0)


def test_legacy_flow_checkpoint_uses_zero_noise_readout():
    agent = make_agent()
    agent.freeze_behavior()
    state = agent.state_dict()
    for key in ("behavior", "actor", "actor_ema"):
        state[key] = {
            name: value for name, value in state[key].items() if not name.startswith("readout.")
        }

    restored = make_agent()
    restored.load_state_dict(state)

    assert restored.behavior.use_mean_head is False
    assert restored.actor.use_mean_head is False
    assert restored.actor_ema.use_mean_head is False


def test_prior_action_can_reuse_fixed_latent():
    agent = make_agent()
    observation = np.zeros(3, dtype=np.float32)
    base_latent = np.array([0.5, -0.25], dtype=np.float32)

    first = agent.act_prior(observation, base_latent=base_latent)
    torch.randn(100)
    second = agent.act_prior(observation, base_latent=base_latent)

    np.testing.assert_allclose(first, second)


def test_cql_can_use_behavior_candidates_only(monkeypatch):
    agent = make_agent(cql_alpha=0.1, cql_num_actions=3, cql_include_uniform=False)
    agent.freeze_behavior()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("uniform actions must not be sampled")

    monkeypatch.setattr(agent.behavior.action_transform, "uniform", fail_if_called)
    batch = make_batch(batch_size=4)
    loss, metrics = agent._cql_loss(
        batch.observations,
        batch.actions,
        agent.critic_1(batch.observations, batch.actions),
        agent.critic_2(batch.observations, batch.actions),
    )

    assert torch.isfinite(loss)
    assert np.isfinite(metrics["critic/cql_loss"])
    assert metrics["critic/cql_candidate_count"] == 4.0


def test_data_action_anchor_bounds_cql_gap():
    agent = make_agent(
        cql_alpha=0.1,
        cql_num_actions=3,
        cql_temperature=1.0,
        cql_include_uniform=False,
        cql_include_data_action=True,
    )
    agent.freeze_behavior()
    batch = make_batch(batch_size=8)
    q1_data = agent.critic_1(batch.observations, batch.actions)
    q2_data = agent.critic_2(batch.observations, batch.actions)

    _, metrics = agent._cql_loss(
        batch.observations,
        batch.actions,
        q1_data,
        q2_data,
    )

    lower_bound = -np.log(4.0)
    assert metrics["critic/cql_q1_gap"] >= lower_bound - 1e-6
    assert metrics["critic/cql_q2_gap"] >= lower_bound - 1e-6
    assert metrics["critic/cql_gap_lower_bound"] == lower_bound


def test_resampled_actor_remains_the_default():
    agent = make_agent()
    agent.freeze_behavior()

    metrics = agent.update(make_batch(batch_size=4), global_step=0)

    assert agent.cfg.actor_mode == "resampled"
    assert metrics["flow_actor/weighted_objective"] == 0.0


def test_new_training_runs_default_to_weighted():
    assert Args().actor_mode == "weighted"


def test_invalid_actor_mode_is_rejected():
    with np.testing.assert_raises_regex(ValueError, "actor_mode"):
        make_agent(actor_mode="invalid")


def test_single_actor_candidate_has_finite_zero_diversity():
    agent = make_agent(actor_num_candidates=1)
    agent.freeze_behavior()
    _, metrics = agent._resampled_actor_targets(torch.randn(4, 3))
    assert metrics["flow_actor/ess"] == 1.0
    assert metrics["flow_actor/ess_fraction"] == 1.0
    assert metrics["flow_actor/action_diversity"] == 0.0
