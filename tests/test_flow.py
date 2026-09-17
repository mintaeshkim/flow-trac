import numpy as np
import torch

from flow_trac.flow import ActionTransform, ConditionalFlow


def test_action_transform_round_trip_for_asymmetric_bounds():
    transform = ActionTransform(
        action_low=np.array([-2.0, 1.0], dtype=np.float32),
        action_high=np.array([4.0, 5.0], dtype=np.float32),
    )
    actions = torch.tensor([[-1.5, 1.5], [0.0, 3.0], [3.5, 4.5]])
    reconstructed = transform.from_latent(transform.to_latent(actions))
    torch.testing.assert_close(reconstructed, actions)


def test_flow_sample_shape_and_bounds():
    flow = ConditionalFlow(
        obs_dim=3,
        action_dim=2,
        action_low=np.array([-2.0, 1.0], dtype=np.float32),
        action_high=np.array([4.0, 5.0], dtype=np.float32),
        hidden_dim=16,
    )
    observations = torch.randn(5, 3)
    samples = flow.sample(observations, num_samples=7, num_steps=3)
    assert samples.shape == (5, 7, 2)
    assert torch.all(samples[..., 0] >= -2.0)
    assert torch.all(samples[..., 0] <= 4.0)
    assert torch.all(samples[..., 1] >= 1.0)
    assert torch.all(samples[..., 1] <= 5.0)


def test_flow_matching_loss_updates_parameters():
    flow = ConditionalFlow(
        obs_dim=3,
        action_dim=2,
        action_low=-np.ones(2, dtype=np.float32),
        action_high=np.ones(2, dtype=np.float32),
        hidden_dim=16,
    )
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    observations = torch.randn(32, 3)
    actions = torch.empty(32, 2).uniform_(-0.8, 0.8)
    before = [parameter.detach().clone() for parameter in flow.parameters()]
    loss = flow.flow_matching_loss(observations, actions).loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, flow.parameters()))


def test_deterministic_samples_use_mean_action_head():
    flow = ConditionalFlow(
        obs_dim=3,
        action_dim=2,
        action_low=-np.ones(2, dtype=np.float32),
        action_high=np.ones(2, dtype=np.float32),
        hidden_dim=16,
    )
    observations = torch.randn(5, 3)

    expected = flow.mean_action(observations)
    samples = flow.sample(observations, num_samples=3, num_steps=2, deterministic=True)

    torch.testing.assert_close(samples, expected[:, None, :].expand(-1, 3, -1))


def test_readout_loss_trains_deterministic_mean():
    torch.manual_seed(0)
    flow = ConditionalFlow(
        obs_dim=3,
        action_dim=2,
        action_low=-np.ones(2, dtype=np.float32),
        action_high=np.ones(2, dtype=np.float32),
        hidden_dim=16,
    )
    observations = torch.randn(32, 3)
    actions = torch.tanh(observations[:, :2])
    optimizer = torch.optim.Adam(flow.readout.parameters(), lr=3e-3)
    initial = flow.mean_action_mse(observations, actions).item()

    for _ in range(200):
        loss = flow.readout_loss(observations, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert flow.mean_action_mse(observations, actions).item() < initial * 0.2
