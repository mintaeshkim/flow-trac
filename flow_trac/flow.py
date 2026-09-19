from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_trac.models import mlp

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass(frozen=True)
class FlowLoss:
    loss: torch.Tensor
    velocity_norm: torch.Tensor
    latent_norm: torch.Tensor


class ActionTransform(nn.Module):
    """Affine Box normalization followed by a tanh/atanh transform."""

    def __init__(self, action_low: np.ndarray, action_high: np.ndarray, clip: float = 0.999):
        super().__init__()
        low = torch.as_tensor(action_low, dtype=torch.float32).reshape(-1)
        high = torch.as_tensor(action_high, dtype=torch.float32).reshape(-1)
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("Flow-TRAC requires finite Box action bounds.")
        if not torch.all(high > low):
            raise ValueError("Every action upper bound must exceed its lower bound.")
        if not 0.0 < clip < 1.0:
            raise ValueError("clip must be in (0, 1).")

        self.register_buffer("center", (high + low) / 2.0)
        self.register_buffer("scale", (high - low) / 2.0)
        self.clip = clip

    def to_latent(self, action: torch.Tensor) -> torch.Tensor:
        normalized = ((action - self.center) / self.scale).clamp(-self.clip, self.clip)
        return torch.atanh(normalized)

    def from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.center + self.scale * torch.tanh(latent)

    def uniform(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        normalized = torch.empty(shape, device=device).uniform_(-1.0, 1.0)
        return self.center + self.scale * normalized


class GaussianReadout(nn.Module):
    """Squashed-Gaussian projection used only for deterministic control."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        for layer in self.trunk[::2]:
            nn.init.constant_(layer.bias, 0.1)
        for layer in (self.mu, self.log_std):
            nn.init.uniform_(layer.weight, -1e-3, 1e-3)
            nn.init.uniform_(layer.bias, -1e-3, 1e-3)

    def parameters_for(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(obs.float())
        return self.mu(hidden), self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def loss(
        self,
        obs: torch.Tensor,
        target_latent: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mu, log_std = self.parameters_for(obs)
        inverse_variance = torch.exp(-2.0 * log_std)
        per_sample = (
            0.5 * (target_latent - mu).square() * inverse_variance + log_std
        ).sum(dim=-1)
        if sample_weight is None:
            return per_sample.mean()
        weight = sample_weight.reshape(-1)
        weight = weight / weight.sum().clamp_min(1e-8)
        return (weight * per_sample).sum()


class ConditionalFlow(nn.Module):
    """Conditional rectified flow in unconstrained action coordinates."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_transform = ActionTransform(action_low, action_high)
        self.velocity = mlp(obs_dim + action_dim + 3, hidden_dim, action_dim)
        self.readout = GaussianReadout(obs_dim, action_dim, hidden_dim)
        self.use_mean_head = True

    def forward(
        self,
        obs: torch.Tensor,
        latent_action: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        time_features = torch.cat(
            (time, torch.sin(torch.pi * time), torch.cos(torch.pi * time)), dim=-1
        )
        return self.velocity(torch.cat((obs.float(), latent_action, time_features), dim=-1))

    def flow_matching_loss(
        self,
        obs: torch.Tensor,
        target_action: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        base_noise: torch.Tensor | None = None,
    ) -> FlowLoss:
        target_latent = self.action_transform.to_latent(target_action)
        if base_noise is None:
            noise = torch.randn_like(target_latent)
        else:
            if base_noise.shape != target_latent.shape:
                raise ValueError("base_noise must have the same shape as target_action.")
            noise = base_noise.to(device=obs.device, dtype=obs.dtype)
        time = torch.rand((obs.shape[0], 1), device=obs.device)
        path = (1.0 - time) * noise + time * target_latent
        target_velocity = target_latent - noise
        predicted_velocity = self(obs, path, time)
        per_sample_loss = F.mse_loss(
            predicted_velocity,
            target_velocity,
            reduction="none",
        ).mean(dim=-1)
        velocity_norm = predicted_velocity.norm(dim=-1)
        latent_norm = target_latent.norm(dim=-1)
        if sample_weight is None:
            loss = per_sample_loss.mean()
            mean_velocity_norm = velocity_norm.mean()
            mean_latent_norm = latent_norm.mean()
        else:
            weight = sample_weight.reshape(-1)
            weight = weight / weight.sum().clamp_min(1e-8)
            loss = (weight * per_sample_loss).sum()
            mean_velocity_norm = (weight * velocity_norm).sum()
            mean_latent_norm = (weight * latent_norm).sum()
        return FlowLoss(
            loss=loss,
            velocity_norm=mean_velocity_norm.detach(),
            latent_norm=mean_latent_norm.detach(),
        )

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic conditional-mean readout for closed-loop control."""
        mu, _ = self.readout.parameters_for(obs)
        return self.action_transform.from_latent(mu)

    def readout_loss(
        self,
        obs: torch.Tensor,
        target_action: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_latent = self.action_transform.to_latent(target_action)
        return self.readout.loss(obs, target_latent, sample_weight)

    def mean_action_mse(
        self,
        obs: torch.Tensor,
        target_action: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(self.mean_action(obs), target_action)

    @torch.no_grad()
    def sample_from_latent(
        self,
        obs: torch.Tensor,
        base_latent: torch.Tensor,
        num_steps: int = 8,
    ) -> torch.Tensor:
        """Transport caller-provided base noise through the conditional flow."""
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")
        if base_latent.ndim == 2:
            base_latent = base_latent.unsqueeze(1)
        if base_latent.ndim != 3:
            raise ValueError(
                "base_latent must have shape [batch, action] or [batch, sample, action]."
            )
        if base_latent.shape[0] != obs.shape[0]:
            raise ValueError("base_latent and obs must have the same batch size.")
        if base_latent.shape[-1] != self.action_dim:
            raise ValueError("base_latent has the wrong action dimension.")

        batch_size, num_samples, _ = base_latent.shape
        obs_dim = obs.shape[-1]
        repeated_obs = (
            obs[:, None, :]
            .expand(batch_size, num_samples, obs_dim)
            .reshape(batch_size * num_samples, obs_dim)
        )
        latent = base_latent.to(device=obs.device, dtype=obs.dtype).reshape(
            batch_size * num_samples,
            self.action_dim,
        )
        step_size = 1.0 / num_steps
        for step in range(num_steps):
            time = torch.full(
                (latent.shape[0], 1),
                step * step_size,
                device=obs.device,
            )
            latent = latent + step_size * self(repeated_obs, latent, time)

        action = self.action_transform.from_latent(latent)
        return action.reshape(batch_size, num_samples, self.action_dim)

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        num_samples: int = 1,
        num_steps: int = 8,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")

        batch_size = obs.shape[0]
        if deterministic and self.use_mean_head:
            action = self.mean_action(obs)
            return action[:, None, :].expand(batch_size, num_samples, self.action_dim)
        if deterministic:
            base_latent = torch.zeros(
                (batch_size, num_samples, self.action_dim),
                device=obs.device,
            )
        else:
            base_latent = torch.randn(
                (batch_size, num_samples, self.action_dim),
                device=obs.device,
            )
        return self.sample_from_latent(obs, base_latent, num_steps=num_steps)
