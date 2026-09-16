from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import minari
import numpy as np
import torch
import tyro

from flow_trac.agent import FlowTRACAgent, FlowTRACConfig
from flow_trac.data import wrap_environment
from flow_trac.train import evaluate, set_seed


@dataclass
class Args:
    checkpoint: str
    num_episodes: int = 10
    seed: int | None = None
    policy: str | None = None
    deterministic: bool | None = None
    prior_num_candidates: int | None = None
    cuda: bool = True
    video_dir: str | None = None
    num_videos: int = 1
    video_fps: int | None = None


def _checkpoint_arg(checkpoint: dict[str, Any], name: str, fallback: Any) -> Any:
    return checkpoint.get("args", {}).get(name, fallback)


def run(args: Args) -> dict[str, float]:
    if args.num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    if not 0 <= args.num_videos <= args.num_episodes:
        raise ValueError("num_videos must be between zero and num_episodes.")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_step = int(checkpoint.get("step", -1))

    dataset_name = _checkpoint_arg(checkpoint, "dataset_name", None)
    if dataset_name is None:
        raise KeyError("The checkpoint does not contain args.dataset_name.")
    seed = args.seed if args.seed is not None else _checkpoint_arg(checkpoint, "eval_seed", 42)
    policy = args.policy or _checkpoint_arg(checkpoint, "eval_policy", "actor")
    deterministic = (
        args.deterministic
        if args.deterministic is not None
        else _checkpoint_arg(checkpoint, "eval_deterministic", False)
    )
    prior_num_candidates = (
        args.prior_num_candidates
        if args.prior_num_candidates is not None
        else _checkpoint_arg(checkpoint, "eval_prior_num_candidates", 64)
    )
    set_seed(seed, deterministic=False)

    dataset = minari.load_dataset(dataset_name, download=False)
    env_kwargs = {"render_mode": "rgb_array"} if args.num_videos else {}
    raw_env = dataset.recover_environment(eval_env=True, **env_kwargs)
    if not isinstance(raw_env.action_space, gym.spaces.Box):
        raise TypeError("Flow-TRAC requires a continuous Box action space.")
    action_space = raw_env.action_space

    if args.num_videos:
        if args.video_dir is None:
            raise ValueError("video_dir is required when num_videos is positive.")
        video_dir = Path(args.video_dir).expanduser().resolve()
        video_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{checkpoint_path.stem}-step-{checkpoint_step + 1}"
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: episode_id < args.num_videos,
            name_prefix=prefix,
            fps=args.video_fps,
        )
    else:
        video_dir = None

    observation_key = _checkpoint_arg(checkpoint, "observation_key", "observation")
    if not isinstance(raw_env.observation_space, gym.spaces.Dict):
        observation_key = None
    env = wrap_environment(
        raw_env,
        observation_key,
        np.asarray(checkpoint["observation_mean"], dtype=np.float32),
        np.asarray(checkpoint["observation_std"], dtype=np.float32),
    )

    config = FlowTRACConfig(**checkpoint["config"])
    agent = FlowTRACAgent(env.observation_space, action_space, config, device)
    agent.load_state_dict(checkpoint["agent"])
    metrics = evaluate(
        agent,
        env,
        seed=seed,
        num_episodes=args.num_episodes,
        policy=policy,
        deterministic=deterministic,
        prior_num_candidates=prior_num_candidates,
    )
    env.close()

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step + 1,
        "dataset_name": dataset_name,
        "device": str(device),
        "seed": seed,
        "num_episodes": args.num_episodes,
        "policy": policy,
        "deterministic": deterministic,
        **metrics,
    }
    if video_dir is not None:
        result["videos"] = sorted(str(path) for path in video_dir.glob("*.mp4"))
        with (video_dir / "evaluation.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return metrics


def cli() -> None:
    run(tyro.cli(Args))


if __name__ == "__main__":
    cli()
