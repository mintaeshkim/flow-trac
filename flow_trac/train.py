from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal

import gymnasium as gym
import minari
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from flow_trac.agent import FlowTRACAgent, FlowTRACConfig
from flow_trac.data import (
    OfflineDataset,
    minari_to_transitions,
    normalize_observations,
    normalize_rewards,
    observation_stats,
    wrap_environment,
)


@dataclass
class Args:
    exp_name: str = "flow_trac"
    seed: int = 10
    eval_seed: int = 42
    torch_deterministic: bool = False
    cuda: bool = False

    # Dataset and environment.
    dataset_name: str = "D4RL/kitchen/complete-v2"
    download_dataset: bool = True
    observation_key: str | None = "observation"
    use_truncations_as_dones: bool = True
    normalize_obs: bool = True
    normalize_reward: bool = False
    reward_scale: float = 0.1

    # Training phases.
    behavior_pretrain_steps: int = 50_000
    behavior_validation_fraction: float = 0.05
    total_updates: int = 1_000_000
    batch_size: int = 256
    critic_warmup_steps: int = 25_000
    actor_updates: bool = True

    # Flow-TRAC.
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    behavior_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    lambda_: Annotated[float, tyro.conf.arg(name="lambda")] = 1.0
    num_value_samples: int = 32
    actor_num_candidates: int = 32
    actor_mode: Literal["resampled", "weighted"] = "weighted"
    flow_steps: int = 8
    policy_frequency: int = 2
    max_grad_norm: float = 10.0
    cql_alpha: float = 1.0
    cql_num_actions: int = 16
    cql_temperature: float = 1.0
    cql_include_uniform: bool = True
    target_q_clip_min: float | None = 0.0
    target_q_clip_max: float | None = 100.0
    actor_ema_decay: float = 0.995
    advantage_clip: float | None = None
    deterministic_loss_coef: float = 1.0

    # Evaluation and logging.
    log_freq: int = 1_000
    eval_freq: int = 5_000
    num_eval_episodes: int = 10
    eval_policy: str = "actor"
    eval_deterministic: bool = False
    eval_prior_num_candidates: int = 64
    save_model: bool = True
    runs_dir: str = "runs"


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _completed_tasks(info: dict[str, Any]) -> set[str]:
    value = info.get("episode_task_completions", info.get("step_task_completions"))
    if value is None:
        return set()
    if isinstance(value, dict):
        return {str(key) for key, completed in value.items() if completed}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


@torch.no_grad()
def evaluate(
    agent: FlowTRACAgent,
    env: gym.Env,
    seed: int,
    num_episodes: int,
    policy: str,
    deterministic: bool,
    prior_num_candidates: int,
) -> dict[str, float]:
    policy = policy.replace("-", "_")
    if policy not in {"actor", "prior", "prior_resample"}:
        raise ValueError("eval_policy must be actor, prior, or prior_resample.")
    returns, lengths, actions, task_counts = [], [], [], []
    for episode_index in range(num_episodes):
        observation, _ = env.reset(seed=seed + episode_index)
        terminated = truncated = False
        episode_return = 0.0
        episode_length = 0
        completed_tasks: set[str] = set()
        while not (terminated or truncated):
            observation_array = np.asarray(observation, dtype=np.float32)
            if policy == "actor":
                action = agent.act(observation_array, deterministic=deterministic)
            elif policy == "prior":
                action = agent.act_prior(observation_array, deterministic=deterministic)
            else:
                action = agent.act_prior_resampled(
                    observation_array,
                    num_candidates=prior_num_candidates,
                    deterministic=deterministic,
                )
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            episode_return += float(reward)
            episode_length += 1
            completed_tasks.update(_completed_tasks(info))
        returns.append(episode_return)
        lengths.append(episode_length)
        task_counts.append(len(completed_tasks))

    action_array = np.asarray(actions, dtype=np.float32)
    return {
        "eval/return": float(np.mean(returns)),
        "eval/length": float(np.mean(lengths)),
        "eval/action_mean": float(action_array.mean()),
        "eval/action_std": float(action_array.std()),
        "eval/action_min": float(action_array.min()),
        "eval/action_max": float(action_array.max()),
        "eval/completed_tasks": float(np.mean(task_counts)),
    }


def _write_metrics(
    writer: SummaryWriter,
    metrics: dict[str, float],
    step: int,
) -> None:
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)


def _checkpoint(
    agent: FlowTRACAgent,
    config: FlowTRACConfig,
    args: Args,
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    step: int,
) -> dict[str, Any]:
    return {
        "agent": agent.state_dict(),
        "config": asdict(config),
        "args": asdict(args),
        "observation_mean": observation_mean,
        "observation_std": observation_std,
        "step": step,
    }


def train(args: Args) -> None:
    if args.behavior_pretrain_steps < 0 or args.total_updates < 0:
        raise ValueError("Training step counts must be non-negative.")
    if args.log_freq < 1:
        raise ValueError("log_freq must be positive.")
    set_seed(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    print(f"Loading Minari dataset: {args.dataset_name}")
    dataset = minari.load_dataset(
        args.dataset_name,
        download=args.download_dataset,
    )
    raw_env = dataset.recover_environment(eval_env=False)
    if not isinstance(raw_env.action_space, gym.spaces.Box):
        raise TypeError("Flow-TRAC requires a continuous Box action space.")
    observation_key = args.observation_key
    if not isinstance(raw_env.observation_space, gym.spaces.Dict):
        observation_key = None
    transitions = minari_to_transitions(
        dataset,
        raw_env.observation_space,
        observation_key,
    )
    action_space = raw_env.action_space
    raw_env.close()

    observation_mean = np.zeros(transitions["observations"].shape[1], dtype=np.float32)
    observation_std = np.ones_like(observation_mean)
    if args.normalize_obs:
        observation_mean, observation_std = observation_stats(transitions)
        normalize_observations(transitions, observation_mean, observation_std)
    if args.normalize_reward:
        normalize_rewards(transitions, args.dataset_name)
    transitions["rewards"] *= args.reward_scale

    offline_dataset = OfflineDataset(
        transitions,
        device=device,
        use_truncations_as_dones=args.use_truncations_as_dones,
        seed=args.seed,
    )
    train_indices, validation_indices = offline_dataset.train_validation_indices(
        args.behavior_validation_fraction
    )
    if len(train_indices) == 0:
        raise ValueError("The behavior training split is empty.")
    validation_indices = validation_indices if len(validation_indices) else train_indices

    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(transitions["observations"].shape[1],),
        dtype=np.float32,
    )
    config = FlowTRACConfig(
        gamma=args.gamma,
        tau=args.tau,
        hidden_dim=args.hidden_dim,
        behavior_lr=args.behavior_lr,
        critic_lr=args.critic_lr,
        actor_lr=args.actor_lr,
        lambda_=args.lambda_,
        num_value_samples=args.num_value_samples,
        actor_num_candidates=args.actor_num_candidates,
        actor_mode=args.actor_mode,
        flow_steps=args.flow_steps,
        critic_warmup_steps=args.critic_warmup_steps,
        actor_updates=args.actor_updates,
        policy_frequency=args.policy_frequency,
        max_grad_norm=args.max_grad_norm,
        cql_alpha=args.cql_alpha,
        cql_num_actions=args.cql_num_actions,
        cql_temperature=args.cql_temperature,
        cql_include_uniform=args.cql_include_uniform,
        target_q_clip_min=args.target_q_clip_min,
        target_q_clip_max=args.target_q_clip_max,
        actor_ema_decay=args.actor_ema_decay,
        advantage_clip=args.advantage_clip,
        deterministic_loss_coef=args.deterministic_loss_coef,
    )
    agent = FlowTRACAgent(observation_space, action_space, config, device)

    timestamp = int(time.time())
    run_name = (
        f"{args.dataset_name.replace('/', '_')}__{args.exp_name}__" f"{args.seed}__{timestamp}"
    )
    writer = SummaryWriter(os.path.join(args.runs_dir, run_name))
    os.makedirs(writer.log_dir, exist_ok=True)
    with open(os.path.join(writer.log_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(asdict(args), file, indent=2, sort_keys=True)
    writer.add_text("config/dataset_name", args.dataset_name, 0)
    writer.add_text("config/observation_key", str(observation_key), 0)
    writer.add_scalar("data/num_transitions", offline_dataset.num_samples, 0)
    writer.add_scalar("data/behavior_train_size", len(train_indices), 0)
    writer.add_scalar("data/behavior_validation_size", len(validation_indices), 0)
    print(
        f"Loaded {offline_dataset.num_samples} transitions on {device}; "
        f"behavior train/validation={len(train_indices)}/{len(validation_indices)}"
    )

    for step in trange(args.behavior_pretrain_steps, desc="Behavior flow"):
        batch = offline_dataset.sample(args.batch_size, train_indices)
        metrics = agent.pretrain_behavior(batch)
        if step % args.log_freq == 0:
            validation_batch = offline_dataset.sample(args.batch_size, validation_indices)
            metrics.update(agent.behavior_validation_metrics(validation_batch))
            _write_metrics(writer, metrics, step)
    agent.freeze_behavior(initialize_actor=True)
    writer.add_scalar("behavior/frozen", 1.0, args.behavior_pretrain_steps)

    eval_env = dataset.recover_environment(eval_env=True)
    eval_env = wrap_environment(
        eval_env,
        observation_key,
        observation_mean,
        observation_std,
    )
    checkpoint_dir = os.path.join(writer.log_dir, "checkpoints")
    if args.save_model:
        os.makedirs(checkpoint_dir, exist_ok=True)

    best_return = -float("inf")
    start_time = time.time()
    for step in trange(args.total_updates, desc="Flow-TRAC"):
        metrics = agent.update(offline_dataset.sample(args.batch_size), step)
        if (step + 1) % args.log_freq == 0:
            metrics["charts/sps"] = (step + 1) / max(time.time() - start_time, 1e-6)
            _write_metrics(writer, metrics, step)

        if args.eval_freq > 0 and (step + 1) % args.eval_freq == 0:
            eval_metrics = evaluate(
                agent,
                eval_env,
                args.eval_seed,
                args.num_eval_episodes,
                args.eval_policy,
                args.eval_deterministic,
                args.eval_prior_num_candidates,
            )
            _write_metrics(writer, eval_metrics, step)
            print(
                f"step={step + 1} return={eval_metrics['eval/return']:.2f} "
                f"length={eval_metrics['eval/length']:.1f} "
                f"tasks={eval_metrics['eval/completed_tasks']:.2f}"
            )
            if args.save_model:
                payload = _checkpoint(
                    agent,
                    config,
                    args,
                    observation_mean,
                    observation_std,
                    step,
                )
                torch.save(payload, os.path.join(checkpoint_dir, "latest.pt"))
                if eval_metrics["eval/return"] > best_return:
                    best_return = eval_metrics["eval/return"]
                    torch.save(payload, os.path.join(checkpoint_dir, "best.pt"))

    if args.save_model:
        torch.save(
            _checkpoint(
                agent,
                config,
                args,
                observation_mean,
                observation_std,
                args.total_updates - 1,
            ),
            os.path.join(checkpoint_dir, "final.pt"),
        )
    eval_env.close()
    writer.close()


def cli() -> None:
    train(tyro.cli(Args))


if __name__ == "__main__":
    cli()
