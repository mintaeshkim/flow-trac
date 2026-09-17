from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from flow_trac.data import Batch
from flow_trac.flow import ConditionalFlow
from flow_trac.models import Critic, soft_update


@dataclass
class FlowTRACConfig:
    gamma: float = 0.99
    tau: float = 5e-3
    hidden_dim: int = 256
    behavior_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    lambda_: float = 1.0
    num_value_samples: int = 32
    actor_num_candidates: int = 32
    actor_mode: Literal["resampled", "weighted"] = "resampled"
    flow_steps: int = 8
    critic_warmup_steps: int = 25_000
    actor_updates: bool = True
    policy_frequency: int = 2
    max_grad_norm: float = 10.0
    cql_alpha: float = 0.0
    cql_num_actions: int = 16
    cql_temperature: float = 1.0
    cql_include_uniform: bool = True
    target_q_clip_min: float | None = None
    target_q_clip_max: float | None = None
    actor_ema_decay: float = 0.995
    advantage_clip: float | None = None
    deterministic_loss_coef: float = 1.0


class FlowTRACAgent:
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        config: FlowTRACConfig,
        device: torch.device,
    ):
        if config.lambda_ <= 0.0:
            raise ValueError("lambda_ must be positive.")
        if config.num_value_samples < 1 or config.actor_num_candidates < 1:
            raise ValueError("Candidate counts must be positive.")
        if config.actor_mode not in {"resampled", "weighted"}:
            raise ValueError("actor_mode must be resampled or weighted.")
        if config.flow_steps < 1:
            raise ValueError("flow_steps must be positive.")
        if config.cql_alpha < 0.0 or config.cql_num_actions < 1:
            raise ValueError("CQL alpha must be non-negative and its action count positive.")
        if config.cql_temperature <= 0.0:
            raise ValueError("cql_temperature must be positive.")
        if not 0.0 <= config.actor_ema_decay < 1.0:
            raise ValueError("actor_ema_decay must be in [0, 1).")
        if config.deterministic_loss_coef < 0.0:
            raise ValueError("deterministic_loss_coef must be non-negative.")

        self.cfg = config
        self.device = device
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        self.action_shape = action_space.shape
        action_low = np.asarray(action_space.low, dtype=np.float32)
        action_high = np.asarray(action_space.high, dtype=np.float32)

        flow_args = (
            obs_dim,
            action_dim,
            action_low,
            action_high,
            config.hidden_dim,
        )
        self.behavior = ConditionalFlow(*flow_args).to(device)
        self.actor = ConditionalFlow(*flow_args).to(device)
        self.actor_ema = deepcopy(self.actor).to(device).eval()
        for parameter in self.actor_ema.parameters():
            parameter.requires_grad_(False)

        self.critic_1 = Critic(obs_dim, action_dim, config.hidden_dim).to(device)
        self.critic_2 = Critic(obs_dim, action_dim, config.hidden_dim).to(device)
        self.critic_1_target = deepcopy(self.critic_1).to(device).eval()
        self.critic_2_target = deepcopy(self.critic_2).to(device).eval()
        for target in (self.critic_1_target, self.critic_2_target):
            for parameter in target.parameters():
                parameter.requires_grad_(False)

        self.behavior_optimizer: torch.optim.Optimizer | None = torch.optim.Adam(
            self.behavior.parameters(), lr=config.behavior_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=config.critic_lr,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.behavior_frozen = False

    def pretrain_behavior(self, batch: Batch) -> dict[str, float]:
        if self.behavior_frozen or self.behavior_optimizer is None:
            raise RuntimeError("The behavior flow is already frozen.")
        result = self.behavior.flow_matching_loss(batch.observations, batch.actions)
        readout_loss = self.behavior.readout_loss(batch.observations, batch.actions)
        mean_action_mse = self.behavior.mean_action_mse(batch.observations, batch.actions)
        loss = result.loss + self.cfg.deterministic_loss_coef * readout_loss
        self.behavior_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.behavior.parameters(), self.cfg.max_grad_norm
        )
        self.behavior_optimizer.step()
        return {
            "behavior/loss": float(loss.item()),
            "behavior/flow_loss": float(result.loss.item()),
            "behavior/readout_loss": float(readout_loss.item()),
            "behavior/mean_action_mse": float(mean_action_mse.item()),
            "behavior/path_velocity_norm": float(result.velocity_norm.item()),
            "behavior/target_latent_norm": float(result.latent_norm.item()),
            "behavior/grad_norm": float(grad_norm),
        }

    @torch.no_grad()
    def behavior_validation_metrics(self, batch: Batch) -> dict[str, float]:
        result = self.behavior.flow_matching_loss(batch.observations, batch.actions)
        readout_loss = self.behavior.readout_loss(batch.observations, batch.actions)
        mean_action_mse = self.behavior.mean_action_mse(batch.observations, batch.actions)
        generated = self.behavior.sample(
            batch.observations,
            num_steps=self.cfg.flow_steps,
        ).squeeze(1)
        return {
            "behavior/validation_flow_loss": float(result.loss.item()),
            "behavior/validation_readout_loss": float(readout_loss.item()),
            "behavior/validation_mean_action_mse": float(mean_action_mse.item()),
            "behavior/generated_action_mean": float(generated.mean().item()),
            "behavior/generated_action_std": float(generated.std().item()),
            "behavior/data_action_mean": float(batch.actions.mean().item()),
            "behavior/data_action_std": float(batch.actions.std().item()),
        }

    def freeze_behavior(self, initialize_actor: bool = True) -> None:
        if initialize_actor:
            self.actor.load_state_dict(self.behavior.state_dict())
            self.actor_ema.load_state_dict(self.behavior.state_dict())
            self.actor.use_mean_head = self.behavior.use_mean_head
            self.actor_ema.use_mean_head = self.behavior.use_mean_head
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.behavior.eval()
        for parameter in self.behavior.parameters():
            parameter.requires_grad_(False)
        self.behavior_optimizer = None
        self.behavior_frozen = True

    def _candidate_q(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        target: bool = True,
    ) -> torch.Tensor:
        batch_size, num_candidates, action_dim = actions.shape
        repeated_obs = (
            obs[:, None, :]
            .expand(batch_size, num_candidates, obs.shape[-1])
            .reshape(batch_size * num_candidates, obs.shape[-1])
        )
        flat_actions = actions.reshape(batch_size * num_candidates, action_dim)
        critic_1 = self.critic_1_target if target else self.critic_1
        critic_2 = self.critic_2_target if target else self.critic_2
        q1 = critic_1(repeated_obs, flat_actions)
        q2 = critic_2(repeated_obs, flat_actions)
        return torch.minimum(q1, q2).reshape(batch_size, num_candidates)

    @torch.no_grad()
    def _target_value(self, next_obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        actions = self.behavior.sample(
            next_obs,
            num_samples=self.cfg.num_value_samples,
            num_steps=self.cfg.flow_steps,
        )
        q = self._candidate_q(next_obs, actions, target=True)
        value = self.cfg.lambda_ * (
            torch.logsumexp(q / self.cfg.lambda_, dim=1) - np.log(self.cfg.num_value_samples)
        )
        return value.unsqueeze(-1), {
            "critic/target_value": float(value.mean().item()),
            "critic/target_candidate_q_mean": float(q.mean().item()),
            "critic/target_candidate_q_std": float(q.std(unbiased=False).item()),
        }

    def _cql_loss(
        self,
        obs: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.cfg.cql_alpha == 0.0:
            return torch.zeros((), device=self.device), {
                "critic/cql_loss": 0.0,
                "critic/cql_q1_gap": 0.0,
                "critic/cql_q2_gap": 0.0,
            }

        with torch.no_grad():
            prior_actions = self.behavior.sample(
                obs,
                num_samples=self.cfg.cql_num_actions,
                num_steps=self.cfg.flow_steps,
            )
            candidate_actions = prior_actions
            if self.cfg.cql_include_uniform:
                uniform_actions = self.behavior.action_transform.uniform(
                    prior_actions.shape,
                    obs.device,
                )
                candidate_actions = torch.cat((candidate_actions, uniform_actions), dim=1)

        batch_size, num_candidates, action_dim = candidate_actions.shape
        repeated_obs = (
            obs[:, None, :]
            .expand(batch_size, num_candidates, obs.shape[-1])
            .reshape(batch_size * num_candidates, obs.shape[-1])
        )
        flat_actions = candidate_actions.reshape(batch_size * num_candidates, action_dim)
        q1 = self.critic_1(repeated_obs, flat_actions).reshape(batch_size, num_candidates)
        q2 = self.critic_2(repeated_obs, flat_actions).reshape(batch_size, num_candidates)
        temperature = self.cfg.cql_temperature
        normalization = temperature * np.log(num_candidates)
        q1_lme = temperature * torch.logsumexp(q1 / temperature, dim=1) - normalization
        q2_lme = temperature * torch.logsumexp(q2 / temperature, dim=1) - normalization
        q1_gap = q1_lme - q1_data.squeeze(-1)
        q2_gap = q2_lme - q2_data.squeeze(-1)
        raw_loss = q1_gap.mean() + q2_gap.mean()
        loss = self.cfg.cql_alpha * raw_loss
        return loss, {
            "critic/cql_loss": float(loss.item()),
            "critic/cql_raw_loss": float(raw_loss.item()),
            "critic/cql_q1_gap": float(q1_gap.mean().item()),
            "critic/cql_q2_gap": float(q2_gap.mean().item()),
        }

    def _update_critic(self, batch: Batch) -> dict[str, float]:
        with torch.no_grad():
            target_value, target_metrics = self._target_value(batch.next_observations)
            target_q_unclipped = batch.rewards + (1.0 - batch.dones) * self.cfg.gamma * target_value
            if self.cfg.target_q_clip_min is None and self.cfg.target_q_clip_max is None:
                target_q = target_q_unclipped
            else:
                target_q = torch.clamp(
                    target_q_unclipped,
                    min=self.cfg.target_q_clip_min,
                    max=self.cfg.target_q_clip_max,
                )

        q1 = self.critic_1(batch.observations, batch.actions)
        q2 = self.critic_2(batch.observations, batch.actions)
        bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        cql_loss, cql_metrics = self._cql_loss(batch.observations, q1, q2)
        critic_loss = bellman_loss + cql_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.cfg.max_grad_norm,
        )
        self.critic_optimizer.step()
        soft_update(self.critic_1, self.critic_1_target, self.cfg.tau)
        soft_update(self.critic_2, self.critic_2_target, self.cfg.tau)

        metrics = {
            "critic/loss": float(critic_loss.item()),
            "critic/bellman_loss": float(bellman_loss.item()),
            "critic/q1": float(q1.mean().item()),
            "critic/q2": float(q2.mean().item()),
            "critic/q_target": float(target_q.mean().item()),
            "critic/q_target_unclipped": float(target_q_unclipped.mean().item()),
            "critic/grad_norm": float(grad_norm),
            **target_metrics,
            **cql_metrics,
        }
        if self.cfg.target_q_clip_min is not None or self.cfg.target_q_clip_max is not None:
            metrics["critic/q_target_clipped_fraction"] = float(
                (target_q != target_q_unclipped).float().mean().item()
            )
        return metrics

    @torch.no_grad()
    def _actor_candidates_and_weights(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        candidates = self.behavior.sample(
            obs,
            num_samples=self.cfg.actor_num_candidates,
            num_steps=self.cfg.flow_steps,
        )
        q = self._candidate_q(obs, candidates, target=True)
        logits = q / self.cfg.lambda_
        if self.cfg.advantage_clip is not None:
            logits = logits - logits.mean(dim=1, keepdim=True)
            logits = logits.clamp(-self.cfg.advantage_clip, self.cfg.advantage_clip)
        weights = torch.softmax(logits, dim=1)
        ess = 1.0 / weights.square().sum(dim=1)
        diversity = candidates.std(dim=1, unbiased=False).norm(dim=-1)
        return candidates, weights, {
            "flow_actor/ess": float(ess.mean().item()),
            "flow_actor/ess_fraction": float((ess / self.cfg.actor_num_candidates).mean().item()),
            "flow_actor/weight_max": float(weights.max(dim=1).values.mean().item()),
            "flow_actor/q_weighted": float((weights * q).sum(dim=1).mean().item()),
            "flow_actor/q_prior": float(q.mean().item()),
            "flow_actor/action_diversity": float(diversity.mean().item()),
        }

    @torch.no_grad()
    def _resampled_actor_targets(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        candidates, weights, metrics = self._actor_candidates_and_weights(obs)
        indices = torch.multinomial(weights, num_samples=1)
        gather_indices = indices.unsqueeze(-1).expand(-1, 1, candidates.shape[-1])
        targets = torch.gather(candidates, dim=1, index=gather_indices).squeeze(1)
        return targets, metrics

    def _update_actor(self, obs: torch.Tensor) -> dict[str, float]:
        if self.cfg.actor_mode == "resampled":
            target_actions, metrics = self._resampled_actor_targets(obs)
            result = self.actor.flow_matching_loss(obs, target_actions)
            readout_loss = self.actor.readout_loss(obs, target_actions)
            mean_action_mse = self.actor.mean_action_mse(obs, target_actions)
        else:
            candidates, weights, metrics = self._actor_candidates_and_weights(obs)
            batch_size, num_candidates, action_dim = candidates.shape
            repeated_obs = (
                obs[:, None, :]
                .expand(batch_size, num_candidates, obs.shape[-1])
                .reshape(batch_size * num_candidates, obs.shape[-1])
            )
            result = self.actor.flow_matching_loss(
                repeated_obs,
                candidates.reshape(batch_size * num_candidates, action_dim),
                sample_weight=weights.reshape(-1),
            )
            flat_candidates = candidates.reshape(batch_size * num_candidates, action_dim)
            readout_loss = self.actor.readout_loss(
                repeated_obs,
                flat_candidates,
                sample_weight=weights.reshape(-1),
            )
            weighted_mean = (weights.unsqueeze(-1) * candidates).sum(dim=1)
            mean_action_mse = self.actor.mean_action_mse(obs, weighted_mean)
        loss = result.loss + self.cfg.deterministic_loss_coef * readout_loss
        self.actor_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()
        self._update_actor_ema()
        metrics.update(
            {
                "flow_actor/loss": float(loss.item()),
                "flow_actor/flow_loss": float(result.loss.item()),
                "flow_actor/readout_loss": float(readout_loss.item()),
                "flow_actor/mean_action_mse": float(mean_action_mse.item()),
                "flow_actor/path_velocity_norm": float(result.velocity_norm.item()),
                "flow_actor/target_latent_norm": float(result.latent_norm.item()),
                "flow_actor/grad_norm": float(grad_norm),
                "flow_actor/weighted_objective": float(self.cfg.actor_mode == "weighted"),
            }
        )
        return metrics

    @torch.no_grad()
    def _update_actor_ema(self) -> None:
        decay = self.cfg.actor_ema_decay
        for ema_parameter, parameter in zip(self.actor_ema.parameters(), self.actor.parameters()):
            ema_parameter.lerp_(parameter, 1.0 - decay)

    def update(self, batch: Batch, global_step: int) -> dict[str, float]:
        if not self.behavior_frozen:
            raise RuntimeError("Call freeze_behavior() before critic training.")
        metrics = self._update_critic(batch)
        actor_enabled = self.cfg.actor_updates and global_step >= self.cfg.critic_warmup_steps
        metrics["warmup/actor_enabled"] = float(actor_enabled)
        if actor_enabled and (global_step + 1) % self.cfg.policy_frequency == 0:
            metrics.update(self._update_actor(batch.observations))
        return metrics

    def _obs_tensor(self, observation: np.ndarray) -> tuple[torch.Tensor, bool]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        squeeze = obs.ndim == 1
        return (obs.unsqueeze(0) if squeeze else obs), squeeze

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs, squeeze = self._obs_tensor(observation)
        actions = self.actor_ema.sample(
            obs,
            num_steps=self.cfg.flow_steps,
            deterministic=deterministic,
        ).squeeze(1)
        result = actions.cpu().numpy().astype(np.float32)
        if squeeze:
            return result[0].reshape(self.action_shape)
        return result.reshape((result.shape[0], *self.action_shape))

    @torch.no_grad()
    def act_prior(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs, squeeze = self._obs_tensor(observation)
        actions = self.behavior.sample(
            obs,
            num_steps=self.cfg.flow_steps,
            deterministic=deterministic,
        ).squeeze(1)
        result = actions.cpu().numpy().astype(np.float32)
        if squeeze:
            return result[0].reshape(self.action_shape)
        return result.reshape((result.shape[0], *self.action_shape))

    @torch.no_grad()
    def act_prior_resampled(
        self,
        observation: np.ndarray,
        num_candidates: int = 64,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs, squeeze = self._obs_tensor(observation)
        candidates = self.behavior.sample(
            obs,
            num_samples=num_candidates,
            num_steps=self.cfg.flow_steps,
        )
        q = self._candidate_q(obs, candidates, target=True)
        weights = torch.softmax(q / self.cfg.lambda_, dim=1)
        indices = (
            weights.argmax(dim=1, keepdim=True)
            if deterministic
            else torch.multinomial(weights, num_samples=1)
        )
        gather_indices = indices.unsqueeze(-1).expand(-1, 1, candidates.shape[-1])
        actions = torch.gather(candidates, 1, gather_indices).squeeze(1)
        result = actions.cpu().numpy().astype(np.float32)
        if squeeze:
            return result[0].reshape(self.action_shape)
        return result.reshape((result.shape[0], *self.action_shape))

    def state_dict(self) -> dict[str, Any]:
        return {
            "behavior": self.behavior.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_ema": self.actor_ema.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_1_target": self.critic_1_target.state_dict(),
            "critic_2_target": self.critic_2_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "behavior_frozen": self.behavior_frozen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        behavior_has_mean = self._load_flow_state(self.behavior, state["behavior"])
        actor_has_mean = self._load_flow_state(self.actor, state["actor"])
        actor_ema_has_mean = self._load_flow_state(self.actor_ema, state["actor_ema"])
        self.behavior.use_mean_head = behavior_has_mean
        self.actor.use_mean_head = actor_has_mean
        self.actor_ema.use_mean_head = actor_ema_has_mean
        self.critic_1.load_state_dict(state["critic_1"])
        self.critic_2.load_state_dict(state["critic_2"])
        self.critic_1_target.load_state_dict(state["critic_1_target"])
        self.critic_2_target.load_state_dict(state["critic_2_target"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if actor_has_mean:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if state.get("behavior_frozen", True):
            self.freeze_behavior(initialize_actor=False)

    @staticmethod
    def _load_flow_state(flow: ConditionalFlow, state: dict[str, Any]) -> bool:
        incompatible = flow.load_state_dict(state, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        readout_keys = {name for name in flow.state_dict() if name.startswith("readout.")}
        legacy_mean_keys = {name for name in unexpected if name.startswith("mean_head.")}
        if unexpected - legacy_mean_keys or missing - readout_keys:
            raise RuntimeError(
                f"Incompatible flow checkpoint: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        return not missing and not legacy_mean_keys
