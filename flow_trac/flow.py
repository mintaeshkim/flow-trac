from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_trac.models import mlp


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
    ) -> FlowLoss:
        target_latent = self.action_transform.to_latent(target_action)
        noise = torch.randn_like(target_latent)
        time = torch.rand((obs.shape[0], 1), device=obs.device)
        path = (1.0 - time) * noise + time * target_latent
        target_velocity = target_latent - noise
        predicted_velocity = self(obs, path, time)
        per_sample_loss = F.mse_loss(
            predicted_velocity,
            target_velocity,
            reduction="none",
        ).mean(dim=-1)
        if sample_weight is None:
            loss = per_sample_loss.mean()
        else:
            weight = sample_weight.reshape(-1)
            weight = weight / weight.sum().clamp_min(1e-8)
            loss = (weight * per_sample_loss).sum()
        return FlowLoss(
            loss=loss,
            velocity_norm=predicted_velocity.norm(dim=-1).mean().detach(),
            latent_norm=target_latent.norm(dim=-1).mean().detach(),
        )

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

        batch_size, obs_dim = obs.shape
        repeated_obs = (
            obs[:, None, :]
            .expand(batch_size, num_samples, obs_dim)
            .reshape(batch_size * num_samples, obs_dim)
        )
        if deterministic:
            latent = torch.zeros(
                (batch_size * num_samples, self.action_dim),
                device=obs.device,
            )
        else:
            latent = torch.randn(
                (batch_size * num_samples, self.action_dim),
                device=obs.device,
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
