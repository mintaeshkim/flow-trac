import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from flow_trac.agent import FlowTRACAgent, FlowTRACConfig
from flow_trac.data import Batch


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
    agent.pretrain_behavior(batch)
    agent.freeze_behavior()
    frozen_parameters = {
        name: parameter.detach().clone() for name, parameter in agent.behavior.named_parameters()
    }
    metrics = agent.update(batch, global_step=0)

    assert "flow_actor/ess_fraction" in metrics
    assert "flow_actor/flow_loss" in metrics
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


def test_single_actor_candidate_has_finite_zero_diversity():
    agent = make_agent(actor_num_candidates=1)
    agent.freeze_behavior()
    _, metrics = agent._resampled_actor_targets(torch.randn(4, 3))
    assert metrics["flow_actor/ess"] == 1.0
    assert metrics["flow_actor/ess_fraction"] == 1.0
    assert metrics["flow_actor/action_diversity"] == 0.0
