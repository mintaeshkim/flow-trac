from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from trac.actor import Actor
from trac.gmm_prior import GMMBehaviorPrior
from trac.models import Critic


@dataclass
class TRACConfig:
    gamma: float = 0.99
    target_gamma: float | None = None
    tau: float = 0.005
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    prior_lr: float = 3e-4
    lambda_: float = 1.0
    num_value_samples: int = 10
    policy_frequency: int = 2
    max_grad_norm: float = 10.0
    prior_pretrain_steps: int = 0
    prior_update_steps: int | None = None
    actor_start_steps: int = 0
    actor_warm_start_from_prior: bool = False
    target_q_clip_min: float | None = None
    target_q_clip_max: float | None = None
    prior_type: str = "gaussian"
    prior_num_components: int = 5
    prior_log_std_min: float = -2.5
    prior_log_std_max: float = 2.0
    target_include_prior_det: bool = False
    target_high_density_oversample: int = 1
    actor_bc_coef: float = 0.0
    use_actor_ema: bool = False
    actor_ema_decay: float = 0.995
    cql_alpha: float = 0.0
    cql_num_actions: int = 16
    cql_temperature: float = 1.0
    cql_include_prior_det: bool = True


class TRACAgent:
    """
    TRAC: dataset-aware soft actor-critic with a learned behavior prior.
    """

    def __init__(self, envs: gym.vector.VectorEnv, config: TRACConfig, device: torch.device):
        self.envs = envs
        self.cfg = config
        self.device = device
        if not hasattr(self.cfg, "use_actor_ema"):
            self.cfg.use_actor_ema = False
        if not hasattr(self.cfg, "actor_ema_decay"):
            self.cfg.actor_ema_decay = 0.995
        if not hasattr(self.cfg, "target_gamma"):
            self.cfg.target_gamma = None
        if not hasattr(self.cfg, "cql_alpha"):
            self.cfg.cql_alpha = 0.0
        if not hasattr(self.cfg, "cql_num_actions"):
            self.cfg.cql_num_actions = 16
        if not hasattr(self.cfg, "cql_temperature"):
            self.cfg.cql_temperature = 1.0
        if not hasattr(self.cfg, "cql_include_prior_det"):
            self.cfg.cql_include_prior_det = True

        self.actor = Actor(envs, hidden_dim=self.cfg.hidden_dim).to(device)
        if not 0.0 <= self.cfg.actor_ema_decay < 1.0:
            raise ValueError("actor_ema_decay must be in [0, 1).")
        self.actor_ema = deepcopy(self.actor).to(device) if self.cfg.use_actor_ema else None
        if self.actor_ema is not None:
            for param in self.actor_ema.parameters():
                param.requires_grad_(False)
            self.actor_ema.eval()

        if self.cfg.prior_type == "gaussian":
            self.behavior_prior = Actor(envs, hidden_dim=self.cfg.hidden_dim).to(device)
        elif self.cfg.prior_type == "gmm":
            self.behavior_prior = GMMBehaviorPrior(
                envs,
                hidden_dim=self.cfg.hidden_dim,
                num_components=self.cfg.prior_num_components,
                log_std_min=self.cfg.prior_log_std_min,
                log_std_max=self.cfg.prior_log_std_max,
            ).to(device)
        else:
            raise ValueError(f"Unsupported TRAC behavior prior type: {self.cfg.prior_type}")
        self.critic_1 = Critic(envs, hidden_dim=self.cfg.hidden_dim).to(device)
        self.critic_2 = Critic(envs, hidden_dim=self.cfg.hidden_dim).to(device)
        self.critic_1_target = deepcopy(self.critic_1).to(device)
        self.critic_2_target = deepcopy(self.critic_2).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.prior_optimizer = optim.Adam(self.behavior_prior.parameters(), lr=self.cfg.prior_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.cfg.critic_lr,
        )
        self.actor_warm_started = False
        self.actor_warm_start_component = -1

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        policy = self.actor_ema if deterministic and self.actor_ema is not None else self.actor
        return policy.act(obs, deterministic=deterministic)

    @torch.no_grad()
    def act_prior(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        squeeze_batch = obs_t.ndim == 1
        if squeeze_batch:
            obs_t = obs_t.unsqueeze(0)

        action = self.behavior_prior(obs_t, deterministic=deterministic)[0]
        action_np = action.cpu().numpy().astype(np.float32)
        return action_np[0] if squeeze_batch else action_np

    @torch.no_grad()
    def act_prior_candidate(
        self,
        obs: np.ndarray,
        num_candidates: int = 64,
        include_det: bool = True,
        score_mode: str = "energy",
        return_info: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        """
        Implicit policy improvement by choosing among behavior-prior actions.

        score_mode="energy" uses Q(s, a) + lambda * log mu(a|s), which is the
        mode objective of the TRAC Boltzmann-improved policy. score_mode="q"
        ignores the density term and greedily maximizes Q over prior samples.
        """
        if score_mode not in {"energy", "q"}:
            raise ValueError("score_mode must be one of {'energy', 'q'}.")

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        squeeze_batch = obs_t.ndim == 1
        if squeeze_batch:
            obs_t = obs_t.unsqueeze(0)

        batch_size = obs_t.shape[0]
        num_candidates = max(1, int(num_candidates))
        num_det = 1 if include_det else 0
        num_stochastic = max(num_candidates - num_det, 0)

        candidate_actions = []
        candidate_log_mu = []
        if include_det:
            det_actions = self.behavior_prior(obs_t, deterministic=True)[0]
            candidate_actions.append(det_actions.unsqueeze(1))
            candidate_log_mu.append(self.behavior_prior.log_prob(obs_t, det_actions).unsqueeze(1))

        if num_stochastic > 0:
            obs_repeated = obs_t.unsqueeze(1).expand(-1, num_stochastic, -1).reshape(
                batch_size * num_stochastic,
                -1,
            )
            sampled_actions = self.behavior_prior(obs_repeated, deterministic=False)[0]
            sampled_log_mu = self.behavior_prior.log_prob(obs_repeated, sampled_actions).view(
                batch_size,
                num_stochastic,
            )
            sampled_actions = sampled_actions.view(batch_size, num_stochastic, -1)
            candidate_actions.append(sampled_actions)
            candidate_log_mu.append(sampled_log_mu)

        actions = torch.cat(candidate_actions, dim=1)
        log_mu = torch.cat(candidate_log_mu, dim=1)
        candidate_count = actions.shape[1]
        obs_repeated = obs_t.unsqueeze(1).expand(-1, candidate_count, -1).reshape(
            batch_size * candidate_count,
            -1,
        )
        actions_flat = actions.reshape(batch_size * candidate_count, -1)
        q1 = self.critic_1(obs_repeated, actions_flat)
        q2 = self.critic_2(obs_repeated, actions_flat)
        min_q = torch.min(q1, q2).view(batch_size, candidate_count)

        if score_mode == "energy":
            scores = min_q + self.cfg.lambda_ * log_mu
        else:
            scores = min_q

        best_indices = scores.argmax(dim=1)
        gather_indices = best_indices.view(batch_size, 1, 1).expand(-1, 1, actions.shape[-1])
        best_actions = torch.gather(actions, dim=1, index=gather_indices).squeeze(1)
        best_q = torch.gather(min_q, dim=1, index=best_indices.view(batch_size, 1)).squeeze(1)
        best_log_mu = torch.gather(log_mu, dim=1, index=best_indices.view(batch_size, 1)).squeeze(1)
        best_scores = torch.gather(scores, dim=1, index=best_indices.view(batch_size, 1)).squeeze(1)

        action_np = best_actions.cpu().numpy().astype(np.float32)
        if squeeze_batch:
            action_np = action_np[0]

        if not return_info:
            return action_np

        info = {
            "eval_policy/prior_candidate_count": float(candidate_count),
            "eval_policy/prior_selected_q": float(best_q.mean().item()),
            "eval_policy/prior_selected_log_mu": float(best_log_mu.mean().item()),
            "eval_policy/prior_selected_score": float(best_scores.mean().item()),
            "eval_policy/prior_candidate_q": float(min_q.mean().item()),
            "eval_policy/prior_candidate_q_std": float(min_q.std().item()),
            "eval_policy/prior_candidate_log_mu": float(log_mu.mean().item()),
            "eval_policy/prior_candidate_log_mu_std": float(log_mu.std().item()),
        }
        return action_np, info

    def _soft_update(self):
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(
                self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data
            )
        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(
                self.cfg.tau * param.data + (1.0 - self.cfg.tau) * target_param.data
            )

    @torch.no_grad()
    def _sync_actor_ema(self):
        if self.actor_ema is None:
            return
        self.actor_ema.load_state_dict(self.actor.state_dict())

    @torch.no_grad()
    def _update_actor_ema(self):
        if self.actor_ema is None:
            return
        decay = self.cfg.actor_ema_decay
        for ema_param, param in zip(self.actor_ema.parameters(), self.actor.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(self.actor_ema.buffers(), self.actor.buffers()):
            ema_buffer.data.copy_(buffer.data)

    def _prior_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return -self.behavior_prior.log_prob(obs, actions).mean()

    def _maybe_warm_start_actor(self, global_step: int, obs: torch.Tensor) -> bool:
        if not self.cfg.actor_warm_start_from_prior:
            return False
        if self.actor_warm_started:
            return False
        if global_step < self.cfg.actor_start_steps:
            return False

        if isinstance(self.behavior_prior, Actor):
            self.actor.load_state_dict(self.behavior_prior.state_dict())
            self.actor_warm_start_component = -1
        elif hasattr(self.behavior_prior, "copy_component_to_actor"):
            self.actor_warm_start_component = self.behavior_prior.copy_component_to_actor(
                self.actor,
                obs,
            )
        else:
            raise TypeError(
                "actor_warm_start_from_prior requires a Gaussian Actor prior or a prior "
                "with copy_component_to_actor()."
            )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.actor_warm_started = True
        self._sync_actor_ema()
        return True

    def _target_candidate_actions(
        self,
        next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        batch_size = next_obs.shape[0]
        num_candidates = max(1, int(self.cfg.num_value_samples))
        include_det = bool(self.cfg.target_include_prior_det)
        num_det = 1 if include_det else 0
        num_stochastic = max(num_candidates - num_det, 0)

        candidate_actions = []
        candidate_log_mu = []
        if include_det:
            det_action = self.behavior_prior(next_obs, deterministic=True)[0]
            candidate_actions.append(det_action.unsqueeze(1))
            candidate_log_mu.append(self.behavior_prior.log_prob(next_obs, det_action).unsqueeze(1))

        if num_stochastic > 0:
            oversample = max(1, int(self.cfg.target_high_density_oversample))
            raw_num_samples = num_stochastic * oversample
            obs_repeated = next_obs.unsqueeze(1).expand(-1, raw_num_samples, -1).reshape(
                batch_size * raw_num_samples,
                -1,
            )
            raw_actions = self.behavior_prior(obs_repeated, deterministic=False)[0]
            raw_log_mu = self.behavior_prior.log_prob(obs_repeated, raw_actions).view(
                batch_size,
                raw_num_samples,
            )
            raw_actions = raw_actions.view(batch_size, raw_num_samples, -1)

            if oversample > 1:
                top_indices = torch.topk(
                    raw_log_mu,
                    k=num_stochastic,
                    dim=1,
                    largest=True,
                ).indices
                gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, raw_actions.shape[-1])
                sampled_actions = torch.gather(raw_actions, dim=1, index=gather_indices)
                sampled_log_mu = torch.gather(raw_log_mu, dim=1, index=top_indices)
            else:
                sampled_actions = raw_actions
                sampled_log_mu = raw_log_mu

            candidate_actions.append(sampled_actions)
            candidate_log_mu.append(sampled_log_mu)

        actions = torch.cat(candidate_actions, dim=1)
        log_mu = torch.cat(candidate_log_mu, dim=1)
        info = {
            "target_candidate_count": float(actions.shape[1]),
            "target_include_prior_det": float(include_det),
            "target_high_density_oversample": float(max(1, int(self.cfg.target_high_density_oversample))),
            "target_candidate_log_mu": float(log_mu.mean().item()),
            "target_candidate_log_mu_std": float(log_mu.std().item()),
            "target_candidate_log_mu_min": float(log_mu.min().item()),
        }
        return actions, log_mu, info

    def _target_value_with_info(
        self,
        next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size = next_obs.shape[0]
        next_actions, _, info = self._target_candidate_actions(next_obs)
        num_candidates = next_actions.shape[1]
        obs_repeated = next_obs.unsqueeze(1).expand(-1, num_candidates, -1).reshape(
            batch_size * num_candidates,
            -1,
        )
        next_actions_flat = next_actions.reshape(batch_size * num_candidates, -1)
        q1 = self.critic_1_target(obs_repeated, next_actions_flat)
        q2 = self.critic_2_target(obs_repeated, next_actions_flat)
        min_q = torch.min(q1, q2).view(batch_size, num_candidates)
        value = self.cfg.lambda_ * torch.logsumexp(
            min_q / self.cfg.lambda_,
            dim=1,
        )
        value = value - self.cfg.lambda_ * np.log(num_candidates)
        info.update(
            {
                "target_candidate_q": float(min_q.mean().item()),
                "target_candidate_q_std": float(min_q.std().item()),
                "target_candidate_q_min": float(min_q.min().item()),
                "target_candidate_q_max": float(min_q.max().item()),
            }
        )
        return value.unsqueeze(-1), info

    def _target_value(self, next_obs: torch.Tensor) -> torch.Tensor:
        return self._target_value_with_info(next_obs)[0]

    def _cql_loss(
        self,
        obs: torch.Tensor,
        data_actions: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.cfg.cql_alpha <= 0.0:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {
                "cql_loss": 0.0,
                "cql_alpha": float(self.cfg.cql_alpha),
                "cql_q1_gap": 0.0,
                "cql_q2_gap": 0.0,
            }

        batch_size = obs.shape[0]
        num_actions = max(1, int(self.cfg.cql_num_actions))
        include_det = bool(self.cfg.cql_include_prior_det)
        num_det = 1 if include_det else 0
        num_stochastic = max(num_actions - num_det, 0)

        candidate_actions = []
        candidate_log_mu = []
        with torch.no_grad():
            if include_det:
                det_actions = self.behavior_prior(obs, deterministic=True)[0]
                candidate_actions.append(det_actions.unsqueeze(1))
                candidate_log_mu.append(self.behavior_prior.log_prob(obs, det_actions).unsqueeze(1))

            if num_stochastic > 0:
                obs_repeated = obs.unsqueeze(1).expand(-1, num_stochastic, -1).reshape(
                    batch_size * num_stochastic,
                    -1,
                )
                sampled_actions = self.behavior_prior(obs_repeated, deterministic=False)[0]
                sampled_log_mu = self.behavior_prior.log_prob(obs_repeated, sampled_actions).view(
                    batch_size,
                    num_stochastic,
                )
                sampled_actions = sampled_actions.view(batch_size, num_stochastic, -1)
                candidate_actions.append(sampled_actions)
                candidate_log_mu.append(sampled_log_mu)

        actions = torch.cat(candidate_actions, dim=1)
        log_mu = torch.cat(candidate_log_mu, dim=1)
        candidate_count = actions.shape[1]
        obs_repeated = obs.unsqueeze(1).expand(-1, candidate_count, -1).reshape(
            batch_size * candidate_count,
            -1,
        )
        actions_flat = actions.reshape(batch_size * candidate_count, -1)

        q1_candidates = self.critic_1(obs_repeated, actions_flat).view(batch_size, candidate_count)
        q2_candidates = self.critic_2(obs_repeated, actions_flat).view(batch_size, candidate_count)
        temperature = max(float(self.cfg.cql_temperature), 1e-6)
        q1_logmeanexp = temperature * torch.logsumexp(
            q1_candidates / temperature,
            dim=1,
        ) - temperature * np.log(candidate_count)
        q2_logmeanexp = temperature * torch.logsumexp(
            q2_candidates / temperature,
            dim=1,
        ) - temperature * np.log(candidate_count)

        q1_gap = q1_logmeanexp - q1_data.view(-1)
        q2_gap = q2_logmeanexp - q2_data.view(-1)
        raw_cql_loss = q1_gap.mean() + q2_gap.mean()
        cql_loss = self.cfg.cql_alpha * raw_cql_loss

        info = {
            "cql_loss": float(cql_loss.item()),
            "cql_raw_loss": float(raw_cql_loss.item()),
            "cql_alpha": float(self.cfg.cql_alpha),
            "cql_num_actions": float(candidate_count),
            "cql_temperature": float(temperature),
            "cql_include_prior_det": float(include_det),
            "cql_q1_gap": float(q1_gap.mean().item()),
            "cql_q2_gap": float(q2_gap.mean().item()),
            "cql_candidate_q1": float(q1_candidates.mean().item()),
            "cql_candidate_q2": float(q2_candidates.mean().item()),
            "cql_candidate_q1_max": float(q1_candidates.max().item()),
            "cql_candidate_q2_max": float(q2_candidates.max().item()),
            "cql_candidate_log_mu": float(log_mu.mean().item()),
            "cql_candidate_log_mu_std": float(log_mu.std().item()),
        }
        return cql_loss, info

    def _critic_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            target_value, target_info = self._target_value_with_info(next_obs)
            target_gamma = self.cfg.gamma if self.cfg.target_gamma is None else self.cfg.target_gamma
            unclipped_target_q = rewards + (1.0 - dones) * target_gamma * target_value
            target_q = unclipped_target_q
            if self.cfg.target_q_clip_min is not None or self.cfg.target_q_clip_max is not None:
                target_q = torch.clamp(
                    target_q,
                    min=self.cfg.target_q_clip_min,
                    max=self.cfg.target_q_clip_max,
                )

        q1 = self.critic_1(obs, actions)
        q2 = self.critic_2(obs, actions)
        bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        cql_loss, cql_info = self._cql_loss(obs, actions, q1, q2)
        critic_loss = bellman_loss + cql_loss

        info = {
            "critic_loss": float(critic_loss.item()),
            "bellman_loss": float(bellman_loss.item()),
            "q1_values": float(q1.mean().item()),
            "q2_values": float(q2.mean().item()),
            "q_target": float(target_q.mean().item()),
            "q_target_std": float(target_q.std().item()),
            "q_target_unclipped": float(unclipped_target_q.mean().item()),
            "target_value": float(target_value.mean().item()),
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std().item()),
            "target_gamma": float(target_gamma),
            **target_info,
            **cql_info,
        }
        if self.cfg.target_q_clip_min is not None or self.cfg.target_q_clip_max is not None:
            info["q_target_clipped_fraction"] = float(
                (target_q != unclipped_target_q).float().mean().item()
            )
        return critic_loss, info

    def _actor_loss(
        self,
        obs: torch.Tensor,
        data_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        actions, log_pi = self.actor(obs, need_log_prob=True)
        prior_requires_grad = [param.requires_grad for param in self.behavior_prior.parameters()]
        for param in self.behavior_prior.parameters():
            param.requires_grad_(False)
        try:
            log_mu = self.behavior_prior.log_prob(obs, actions)
        finally:
            for param, requires_grad in zip(self.behavior_prior.parameters(), prior_requires_grad):
                param.requires_grad_(requires_grad)
        q1_pi = self.critic_1(obs, actions)
        q2_pi = self.critic_2(obs, actions)
        min_q_pi = torch.min(q1_pi, q2_pi).view(-1)
        policy_mu, policy_log_sigma = self.actor._params(obs)
        mean_actions = torch.tanh(policy_mu) * self.actor.max_action

        kl_term = log_pi - log_mu
        sac_actor_loss = (self.cfg.lambda_ * kl_term - min_q_pi).mean()
        bc_loss = F.mse_loss(mean_actions, data_actions)
        actor_loss = sac_actor_loss + self.cfg.actor_bc_coef * bc_loss

        info = {
            "actor_loss": float(actor_loss.item()),
            "actor/sac_loss": float(sac_actor_loss.item()),
            "actor/bc_loss": float(bc_loss.item()),
            "actor/bc_coef": float(self.cfg.actor_bc_coef),
            "actor/kl_term": float(kl_term.mean().item()),
            "actor/log_pi": float(log_pi.mean().item()),
            "actor/log_mu": float(log_mu.mean().item()),
            "actor/q_values": float(min_q_pi.mean().item()),
            "actor/policy_action_mean": float(actions.mean().item()),
            "actor/policy_action_std": float(actions.std().item()),
            "actor/mean_action_mse_to_data": float(bc_loss.item()),
            "actor/mean_action_l1_to_data": float((mean_actions - data_actions).abs().mean().item()),
            "actor/policy_mean_abs": float(policy_mu.abs().mean().item()),
            "actor/policy_log_std_mean": float(policy_log_sigma.mean().item()),
        }
        return actor_loss, info

    def update(self, replay_buffer, batch_size: int, global_step: int) -> dict[str, Any]:
        data = replay_buffer.sample(batch_size)
        obs = data.observations
        actions = data.actions
        next_obs = data.next_observations
        rewards = data.rewards
        dones = data.dones

        prior_loss = self._prior_loss(obs, actions)
        update_prior = (
            self.cfg.prior_update_steps is None
            or global_step < self.cfg.prior_update_steps
        )
        prior_grad_norm = torch.tensor(0.0, device=self.device)
        if update_prior:
            self.prior_optimizer.zero_grad()
            prior_loss.backward()
            prior_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.behavior_prior.parameters(),
                self.cfg.max_grad_norm,
            )
            self.prior_optimizer.step()

        with torch.no_grad():
            prior_mu, prior_log_sigma = self.behavior_prior._params(obs)

        metrics: dict[str, Any] = {
            "prior_loss": float(prior_loss.item()),
            "prior_grad_norm": float(prior_grad_norm),
            "prior/log_prob_data": float((-prior_loss).item()),
            "prior/log_std_mean": float(prior_log_sigma.mean().item()),
            "prior/mean_abs": float(prior_mu.abs().mean().item()),
            "warmup/prior_updates_enabled": float(update_prior),
            "warmup/prior_only": float(global_step < self.cfg.prior_pretrain_steps),
            "actor/use_ema": float(self.actor_ema is not None),
            "actor/ema_decay": float(self.cfg.actor_ema_decay if self.actor_ema is not None else 0.0),
        }
        if hasattr(self.behavior_prior, "component_entropy"):
            metrics["prior/component_entropy"] = float(
                self.behavior_prior.component_entropy(obs).mean().item()
            )
        if hasattr(self.behavior_prior, "max_component_prob"):
            metrics["prior/max_component_prob"] = float(
                self.behavior_prior.max_component_prob(obs).mean().item()
            )

        if global_step < self.cfg.prior_pretrain_steps:
            return metrics

        actor_warm_started_now = self._maybe_warm_start_actor(global_step, obs)
        metrics["warmup/actor_warm_started"] = float(self.actor_warm_started)
        metrics["warmup/actor_warm_started_now"] = float(actor_warm_started_now)
        metrics["warmup/actor_warm_start_component"] = float(self.actor_warm_start_component)

        critic_loss, critic_info = self._critic_loss(obs, actions, rewards, next_obs, dones)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.cfg.max_grad_norm,
        )
        self.critic_optimizer.step()

        metrics.update(
            {
                "critic_grad_norm": float(critic_grad_norm),
                **critic_info,
            }
        )

        if (
            global_step >= self.cfg.actor_start_steps
            and (global_step + 1) % self.cfg.policy_frequency == 0
        ):
            actor_loss, actor_info = self._actor_loss(obs, actions)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.cfg.max_grad_norm,
            )
            self.actor_optimizer.step()
            metrics["actor_grad_norm"] = float(actor_grad_norm)
            metrics.update(actor_info)

            with torch.no_grad():
                self._update_actor_ema()
                self._soft_update()
                if self.actor_ema is not None:
                    actor_det_actions = self.actor(obs, deterministic=True)[0]
                    ema_det_actions = self.actor_ema(obs, deterministic=True)[0]
                    metrics["actor/ema_action_l1_to_actor"] = float(
                        (ema_det_actions - actor_det_actions).abs().mean().item()
                    )
        elif (global_step + 1) % self.cfg.policy_frequency == 0:
            with torch.no_grad():
                self._soft_update()

        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_ema": self.actor_ema.state_dict() if self.actor_ema is not None else None,
            "behavior_prior": self.behavior_prior.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_1_target": self.critic_1_target.state_dict(),
            "critic_2_target": self.critic_2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "prior_optimizer": self.prior_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_warm_started": self.actor_warm_started,
            "actor_warm_start_component": self.actor_warm_start_component,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        if self.actor_ema is not None:
            actor_ema_state = state_dict.get("actor_ema", None)
            if actor_ema_state is None:
                self._sync_actor_ema()
            else:
                self.actor_ema.load_state_dict(actor_ema_state)
        self.behavior_prior.load_state_dict(state_dict["behavior_prior"])
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_1_target.load_state_dict(state_dict["critic_1_target"])
        self.critic_2_target.load_state_dict(state_dict["critic_2_target"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.prior_optimizer.load_state_dict(state_dict["prior_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.actor_warm_started = bool(state_dict.get("actor_warm_started", False))
        self.actor_warm_start_component = int(state_dict.get("actor_warm_start_component", -1))
