# trac/agents/trac/gmm_prior.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from trac.agents.trac.actor import EPS, atanh


class GMMBehaviorPrior(nn.Module):
    """
    Tanh-squashed Gaussian mixture behavior prior.

    This is intended for TRAC's behavior model mu(a|s), not for the main actor.
    It provides sampling and exact mixture log-probability under the tanh
    change-of-variables correction.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        hidden_dim: int = 256,
        num_components: int = 5,
        log_std_min: float = -2.5,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if num_components < 1:
            raise ValueError("num_components must be >= 1.")

        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.logits = nn.Linear(hidden_dim, num_components)
        self.mu = nn.Linear(hidden_dim, num_components * act_dim)
        self.log_sigma = nn.Linear(hidden_dim, num_components * act_dim)

        for layer in self.trunk[::2]:
            nn.init.constant_(layer.bias, 0.1)
        nn.init.uniform_(self.logits.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.logits.bias, -1e-3, 1e-3)
        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = act_dim
        self.num_components = num_components
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = float(np.max(np.abs(env.single_action_space.high)))

    def _component_params(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(obs.float())
        logits = self.logits(hidden)
        mu = self.mu(hidden).view(-1, self.num_components, self.action_dim)
        log_sigma = self.log_sigma(hidden).view(-1, self.num_components, self.action_dim)
        log_sigma = torch.clip(log_sigma, self.log_std_min, self.log_std_max)
        return logits, mu, log_sigma

    def _params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, mu, log_sigma = self._component_params(obs)
        probs = F.softmax(logits, dim=-1).unsqueeze(-1)
        mixture_mu = (probs * mu).sum(dim=1)
        mixture_log_sigma = (probs * log_sigma).sum(dim=1)
        return mixture_mu, mixture_log_sigma

    def component_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self._component_params(obs)
        return Categorical(logits=logits).entropy()

    def max_component_prob(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self._component_params(obs)
        return F.softmax(logits, dim=-1).max(dim=-1).values

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits, mu, log_sigma = self._component_params(obs)

        if deterministic:
            component = logits.argmax(dim=-1)
        else:
            component = Categorical(logits=logits).sample()

        batch_indices = torch.arange(obs.shape[0], device=obs.device)
        selected_mu = mu[batch_indices, component]
        selected_log_sigma = log_sigma[batch_indices, component]
        dist = Normal(selected_mu, selected_log_sigma.exp())
        raw_action = selected_mu if deterministic else dist.rsample()
        action = torch.tanh(raw_action) * self.max_action

        log_prob = self.log_prob(obs, action) if need_log_prob else None
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        logits, mu, log_sigma = self._component_params(obs)
        scaled_action = action / self.max_action
        raw_action = atanh(scaled_action)

        dist = Normal(mu, log_sigma.exp())
        component_log_probs = dist.log_prob(raw_action.unsqueeze(1)).sum(dim=-1)
        mixture_log_probs = torch.logsumexp(
            F.log_softmax(logits, dim=-1) + component_log_probs,
            dim=-1,
        )
        correction = torch.log(1 - scaled_action.pow(2) + EPS).sum(dim=-1)
        correction = correction + self.action_dim * np.log(self.max_action)
        return mixture_log_probs - correction

    @torch.no_grad()
    def copy_component_to_actor(
        self,
        actor: nn.Module,
        obs: torch.Tensor | None = None,
    ) -> int:
        if obs is None:
            component_idx = 0
        else:
            logits, _, _ = self._component_params(obs)
            component_idx = int(logits.mean(dim=0).argmax().item())

        start = component_idx * self.action_dim
        end = start + self.action_dim
        actor.trunk.load_state_dict(self.trunk.state_dict())
        actor.mu.weight.copy_(self.mu.weight[start:end])
        actor.mu.bias.copy_(self.mu.bias[start:end])
        actor.log_sigma.weight.copy_(self.log_sigma.weight[start:end])
        actor.log_sigma.bias.copy_(self.log_sigma.bias[start:end])
        return component_idx
