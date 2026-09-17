# trac/train/train_kitchen.py
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import gymnasium as gym
import minari
import numpy as np
import torch
import tyro
from tqdm import trange

from trac.agents.trac import TRACAgent, TRACConfig
from trac.utils.buffers import ReplayBuffer
from trac.utils.data_utils import (
    compute_obs_mean_std,
    make_n_step_transitions,
    minari_dataset_to_transitions,
    normalize_dataset_obs,
    normalize_dataset_reward,
    validate_dataset_shapes,
    wrap_obs_normalization,
)
from trac.utils.obs_utils import flatten_env
from trac.utils.train_utils import (
    make_tensorboard_writer,
    save_config_txt,
    set_seed,
)


@dataclass
class Args:
    exp_name: str = "train_trac_kitchen"
    seed: int = 10
    eval_seed: int = 42
    torch_deterministic: bool = False
    cuda: bool = False

    # Dataset / environment
    dataset_name: str = "D4RL/kitchen/complete-v2"
    download_dataset: bool = True
    use_truncations_as_dones: bool = True
    normalize_obs: bool = True
    normalize_reward: bool = False
    reward_scale: float = 0.1
    include_goal_observations: bool = False
    observation_key: str = "observation"

    # Offline training
    total_updates: int = 1_000_000
    buffer_size: Optional[int] = None
    batch_size: int = 256
    n_step: int = 1

    # TRAC
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    prior_lr: float = 3e-4
    lambda_: float = 1.0
    num_value_samples: int = 32
    policy_frequency: int = 2
    max_grad_norm: float = 10.0
    prior_pretrain_steps: int = 50_000
    prior_update_steps: Optional[int] = 50_000
    actor_start_steps: int = 50_000
    actor_warm_start_from_prior: bool = True
    target_q_clip_min: Optional[float] = 0.0
    target_q_clip_max: Optional[float] = 100.0
    prior_type: str = "gaussian"
    prior_num_components: int = 5
    prior_log_std_min: float = -2.5
    prior_log_std_max: float = 2.0
    target_include_prior_det: bool = False
    target_high_density_oversample: int = 1
    actor_bc_coef: float = 0.0
    use_actor_ema: bool = True
    actor_ema_decay: float = 0.995
    cql_alpha: float = 0.1
    cql_num_actions: int = 16
    cql_temperature: float = 1.0
    cql_include_prior_det: bool = True

    # Logging / evaluation
    log_freq: int = 1000
    eval_freq: int = 5000
    num_eval_episodes: int = 10
    eval_policy: str = "actor"
    eval_prior_num_candidates: int = 64
    eval_prior_include_det: bool = True
    capture_video: bool = False
    save_model: bool = True
    save_best_model: bool = True
    best_model_metric: str = "eval_episodic_return"
    checkpoints_path: str = "checkpoints"


def make_kitchen_env(
    dataset: minari.MinariDataset,
    seed: int,
    idx: int,
    observation_key: str | None,
):
    def thunk():
        env = dataset.recover_environment(eval_env=False)
        env = flatten_env(env, observation_key=observation_key)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk


def make_kitchen_eval_env(
    dataset: minari.MinariDataset,
    obs_mean,
    obs_std,
    observation_key: str | None,
):
    eval_env = dataset.recover_environment(eval_env=True)
    eval_env = flatten_env(eval_env, observation_key=observation_key)
    eval_env = wrap_obs_normalization(eval_env, obs_mean, obs_std)
    sample_obs, _ = eval_env.reset()
    print("Kitchen eval env created:", np.asarray(sample_obs, dtype=np.float32)[:5])
    return eval_env


def _task_names(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        return {str(key) for key, is_done in value.items() if bool(is_done)}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def _completed_tasks(info: dict[str, Any]) -> set[str]:
    completed = info.get("episode_task_completions", None)
    if completed is None:
        completed = info.get("step_task_completions", None)
    return _task_names(completed)


def _remaining_tasks(info: dict[str, Any]) -> set[str] | None:
    remaining = info.get("tasks_to_complete", None)
    if remaining is None:
        return None
    if isinstance(remaining, dict):
        return {str(key) for key, is_remaining in remaining.items() if bool(is_remaining)}
    if isinstance(remaining, (list, tuple, set)):
        return {str(item) for item in remaining}
    return {str(remaining)}


@torch.no_grad()
def evaluate_kitchen(
    agent: TRACAgent,
    eval_env: gym.Env,
    seed: int,
    num_episodes: int,
    eval_policy: str = "actor",
    eval_prior_num_candidates: int = 64,
    eval_prior_include_det: bool = True,
) -> dict[str, float]:
    if eval_policy not in {"actor", "prior_energy", "prior_q"}:
        raise ValueError("eval_policy must be one of {'actor', 'prior_energy', 'prior_q'}.")

    returns = []
    lengths = []
    actions = []
    completed_counts = []
    remaining_counts = []
    successes = []
    task_completion_counts: dict[str, int] = {}
    policy_info_values: dict[str, list[float]] = {}

    for episode_idx in range(num_episodes):
        obs, reset_info = eval_env.reset(seed=seed + episode_idx)
        done = False
        ep_return = 0.0
        ep_length = 0
        final_info: dict[str, Any] = {}
        completed_tasks = set()
        initial_tasks = _remaining_tasks(reset_info)

        while not done:
            obs_arr = np.asarray(obs, dtype=np.float32)
            if eval_policy == "actor":
                action = agent.act(obs_arr, deterministic=True)
            else:
                score_mode = "energy" if eval_policy == "prior_energy" else "q"
                action, policy_info = agent.act_prior_candidate(
                    obs_arr,
                    num_candidates=eval_prior_num_candidates,
                    include_det=eval_prior_include_det,
                    score_mode=score_mode,
                    return_info=True,
                )
                for key, value in policy_info.items():
                    policy_info_values.setdefault(key, []).append(value)
            actions.append(action)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            ep_length += 1
            final_info = info
            completed_tasks.update(_completed_tasks(info))
            if initial_tasks is None:
                initial_tasks = _remaining_tasks(info)

        final_remaining = _remaining_tasks(final_info)
        num_completed = len(completed_tasks)
        if initial_tasks is not None:
            num_total_tasks = len(initial_tasks)
            num_remaining = max(num_total_tasks - num_completed, 0)
        elif final_remaining is not None:
            num_remaining = len(final_remaining)
            num_total_tasks = num_completed + num_remaining
        else:
            num_remaining = 0
            num_total_tasks = num_completed

        returns.append(ep_return)
        lengths.append(ep_length)
        completed_counts.append(num_completed)
        remaining_counts.append(num_remaining)
        successes.append(float(num_total_tasks > 0 and num_completed >= num_total_tasks))
        for task_name in completed_tasks:
            task_completion_counts[task_name] = task_completion_counts.get(task_name, 0) + 1

    actions_arr = np.asarray(actions, dtype=np.float32)
    metrics = {
        "eval_episodic_return": float(np.mean(returns)),
        "eval_episodic_length": float(np.mean(lengths)),
        "eval_action_mean": float(actions_arr.mean()),
        "eval_action_std": float(actions_arr.std()),
        "eval_action_min": float(actions_arr.min()),
        "eval_action_max": float(actions_arr.max()),
        "eval_completed_tasks": float(np.mean(completed_counts)),
        "eval_remaining_tasks": float(np.mean(remaining_counts)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_policy/is_prior_energy": float(eval_policy == "prior_energy"),
        "eval_policy/is_prior_q": float(eval_policy == "prior_q"),
    }
    for key, values in policy_info_values.items():
        metrics[key] = float(np.mean(values))
    for task_name, count in sorted(task_completion_counts.items()):
        safe_name = task_name.replace("/", "_").replace(" ", "_")
        metrics[f"eval_task/{safe_name}_completion_rate"] = float(count / num_episodes)
    return metrics


def checkpoint_payload(
    agent: TRACAgent,
    trac_cfg: TRACConfig,
    args: Args,
    obs_mean,
    obs_std,
    selected_observation_key: str | None,
    best_eval_metric: float | None = None,
    best_eval_step: int | None = None,
) -> dict[str, Any]:
    return {
        "agent": agent.state_dict(),
        "trac_config": trac_cfg,
        "args": asdict(args),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "observation_is_flattened": True,
        "observation_key": selected_observation_key,
        "best_model_metric": args.best_model_metric,
        "best_eval_metric": best_eval_metric,
        "best_eval_step": best_eval_step,
    }


def main(args: Args):
    if args.n_step < 1:
        raise ValueError("n_step must be >= 1.")
    if args.cql_alpha < 0.0:
        raise ValueError("cql_alpha must be >= 0.")
    if args.cql_num_actions < 1:
        raise ValueError("cql_num_actions must be >= 1.")
    if args.cql_temperature <= 0.0:
        raise ValueError("cql_temperature must be > 0.")
    set_seed(args.seed, args.torch_deterministic)
    writer, run_name = make_tensorboard_writer(args.dataset_name, args.exp_name, args.seed)
    save_config_txt(args, writer.log_dir)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    print(f"Loading Minari Kitchen dataset: {args.dataset_name}")
    dataset = minari.load_dataset(args.dataset_name, download=args.download_dataset)
    raw_env = dataset.recover_environment(eval_env=False)
    raw_observation_space_text = str(raw_env.observation_space)
    selected_observation_key = None if args.include_goal_observations else args.observation_key

    transitions = minari_dataset_to_transitions(
        dataset,
        observation_space=raw_env.observation_space,
        flatten_observations=True,
        observation_key=selected_observation_key,
    )
    raw_env.close()

    obs_mean = np.array(0.0, dtype=np.float32)
    obs_std = np.array(1.0, dtype=np.float32)
    if args.normalize_obs:
        obs_mean, obs_std = compute_obs_mean_std(transitions)
        normalize_dataset_obs(transitions, obs_mean, obs_std)
    if args.normalize_reward:
        normalize_dataset_reward(transitions, args.dataset_name)
    if args.reward_scale != 1.0:
        transitions["rewards"] *= args.reward_scale
    if args.n_step > 1:
        transitions, n_step_info = make_n_step_transitions(
            transitions,
            n_step=args.n_step,
            gamma=args.gamma,
            use_truncations_as_dones=args.use_truncations_as_dones,
        )
        print(
            "Applied n-step targets: "
            f"n={args.n_step}, "
            f"effective_mean={n_step_info['n_step_effective_mean']:.2f}, "
            f"done_fraction={n_step_info['n_step_done_fraction']:.4f}, "
            f"bootstrap_gamma={n_step_info['n_step_bootstrap_gamma']:.4f}"
        )
    else:
        n_step_info = {
            "n_step": 1.0,
            "n_step_effective_mean": 1.0,
            "n_step_done_fraction": float(
                np.logical_or(
                    transitions["terminations"].astype(bool),
                    transitions["truncations"].astype(bool)
                    if args.use_truncations_as_dones
                    else np.zeros_like(transitions["terminations"], dtype=bool),
                ).mean()
            ),
            "n_step_reward_mean": float(transitions["rewards"].mean()),
            "n_step_reward_std": float(transitions["rewards"].std()),
            "n_step_bootstrap_gamma": float(args.gamma),
        }

    dataset_size = transitions["observations"].shape[0]
    buffer_size = args.buffer_size or dataset_size
    print(f"Loaded {dataset_size} flattened Kitchen transitions")

    envs = gym.vector.SyncVectorEnv(
        [make_kitchen_env(dataset, args.seed, 0, selected_observation_key)]
    )
    eval_env = make_kitchen_eval_env(dataset, obs_mean, obs_std, selected_observation_key)
    assert isinstance(envs.single_observation_space, gym.spaces.Box)
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    assert isinstance(eval_env.action_space, gym.spaces.Box)
    validate_dataset_shapes(transitions, envs)

    trac_cfg = TRACConfig(
        gamma=args.gamma,
        target_gamma=args.gamma ** args.n_step,
        tau=args.tau,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        prior_lr=args.prior_lr,
        lambda_=args.lambda_,
        num_value_samples=args.num_value_samples,
        policy_frequency=args.policy_frequency,
        max_grad_norm=args.max_grad_norm,
        prior_pretrain_steps=args.prior_pretrain_steps,
        prior_update_steps=args.prior_update_steps,
        actor_start_steps=args.actor_start_steps,
        actor_warm_start_from_prior=args.actor_warm_start_from_prior,
        target_q_clip_min=args.target_q_clip_min,
        target_q_clip_max=args.target_q_clip_max,
        prior_type=args.prior_type,
        prior_num_components=args.prior_num_components,
        prior_log_std_min=args.prior_log_std_min,
        prior_log_std_max=args.prior_log_std_max,
        target_include_prior_det=args.target_include_prior_det,
        target_high_density_oversample=args.target_high_density_oversample,
        actor_bc_coef=args.actor_bc_coef,
        use_actor_ema=args.use_actor_ema,
        actor_ema_decay=args.actor_ema_decay,
        cql_alpha=args.cql_alpha,
        cql_num_actions=args.cql_num_actions,
        cql_temperature=args.cql_temperature,
        cql_include_prior_det=args.cql_include_prior_det,
    )
    agent = TRACAgent(envs, trac_cfg, device)

    rb = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        n_envs=1,
    )
    rb.load_transitions(
        transitions,
        use_truncations_as_dones=args.use_truncations_as_dones,
    )

    writer.add_text("config/dataset_name", args.dataset_name, 0)
    writer.add_text("config/raw_observation_space", raw_observation_space_text, 0)
    writer.add_text("config/flat_observation_space", str(envs.single_observation_space), 0)
    writer.add_text(
        "config/observation_mode",
        "full_dict_flatten" if selected_observation_key is None else f"dict_key:{selected_observation_key}",
        0,
    )
    writer.add_scalar("charts/dataset_size", rb.size(), 0)
    for key, value in n_step_info.items():
        writer.add_scalar(f"config/{key}", value, 0)
    checkpoint_dir = os.path.join(writer.log_dir, args.checkpoints_path)
    if args.save_model:
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer.add_text("config/checkpoint_dir", checkpoint_dir, 0)
    writer.add_text("config/best_model_metric", args.best_model_metric, 0)
    writer.add_text("config/eval_policy", args.eval_policy, 0)
    writer.add_scalar("config/eval_prior_num_candidates", args.eval_prior_num_candidates, 0)
    writer.add_scalar("config/eval_prior_include_det", float(args.eval_prior_include_det), 0)
    writer.add_scalar("config/cql_alpha", args.cql_alpha, 0)
    writer.add_scalar("config/cql_num_actions", args.cql_num_actions, 0)
    writer.add_scalar("config/cql_temperature", args.cql_temperature, 0)
    writer.add_scalar("config/cql_include_prior_det", float(args.cql_include_prior_det), 0)

    best_eval_metric = -float("inf")
    best_eval_step = -1
    train_start_time = time.time()
    for global_step in trange(args.total_updates, desc="TRAC-Kitchen"):
        metrics = agent.update(rb, args.batch_size, global_step)

        if global_step % args.log_freq == 0:
            sps = int((global_step + 1) / (time.time() - train_start_time))
            print(f"global_step={global_step}, SPS={sps}")
            writer.add_scalar("charts/SPS", sps, global_step)
            for key, value in metrics.items():
                tag = key if "/" in key else f"losses/{key}"
                writer.add_scalar(tag, value, global_step)

        if args.eval_freq > 0 and (global_step + 1) % args.eval_freq == 0:
            eval_metrics = evaluate_kitchen(
                agent,
                eval_env,
                args.eval_seed,
                args.num_eval_episodes,
                eval_policy=args.eval_policy,
                eval_prior_num_candidates=args.eval_prior_num_candidates,
                eval_prior_include_det=args.eval_prior_include_det,
            )
            print(
                f"eval_step={global_step}, "
                f"policy={args.eval_policy}, "
                f"return={eval_metrics['eval_episodic_return']:.2f}, "
                f"length={eval_metrics['eval_episodic_length']:.1f}, "
                f"tasks={eval_metrics['eval_completed_tasks']:.2f}"
            )
            for key, value in eval_metrics.items():
                writer.add_scalar(f"charts/{key}", value, global_step)

            if args.save_model and args.save_best_model:
                if args.best_model_metric not in eval_metrics:
                    available = ", ".join(sorted(eval_metrics.keys()))
                    raise KeyError(
                        f"best_model_metric={args.best_model_metric!r} was not produced by "
                        f"evaluation. Available metrics: {available}"
                    )
                eval_metric = eval_metrics[args.best_model_metric]
                if eval_metric > best_eval_metric:
                    best_eval_metric = eval_metric
                    best_eval_step = global_step
                    best_path = os.path.join(checkpoint_dir, "trac_kitchen_best.pt")
                    torch.save(
                        checkpoint_payload(
                            agent,
                            trac_cfg,
                            args,
                            obs_mean,
                            obs_std,
                            selected_observation_key,
                            best_eval_metric=best_eval_metric,
                            best_eval_step=best_eval_step,
                        ),
                        best_path,
                    )
                    print(
                        f"new_best_step={global_step}, "
                        f"{args.best_model_metric}={best_eval_metric:.2f}, "
                        f"path={best_path}"
                    )
                writer.add_scalar(
                    f"charts/best_{args.best_model_metric}",
                    best_eval_metric,
                    global_step,
                )
                writer.add_scalar("charts/best_eval_step", best_eval_step, global_step)

        if args.save_model and args.eval_freq > 0 and (global_step + 1) % args.eval_freq == 0:
            torch.save(
                checkpoint_payload(
                    agent,
                    trac_cfg,
                    args,
                    obs_mean,
                    obs_std,
                    selected_observation_key,
                    best_eval_metric=best_eval_metric if best_eval_step >= 0 else None,
                    best_eval_step=best_eval_step if best_eval_step >= 0 else None,
                ),
                os.path.join(checkpoint_dir, f"trac_kitchen_{global_step}.pt"),
            )

    envs.close()
    eval_env.close()
    writer.close()


def cli() -> None:
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
