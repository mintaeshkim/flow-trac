from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, is_dataclass

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def make_tensorboard_writer(
    dataset_name: str,
    exp_name: str,
    seed: int,
) -> tuple[SummaryWriter, str]:
    run_name = f"{dataset_name.replace('/', '_')}__{exp_name}__{seed}__{int(time.time())}"
    return SummaryWriter(f"runs/{run_name}"), run_name


def save_config_txt(args, log_dir: str, filename: str = "config.txt") -> str:
    config = asdict(args) if is_dataclass(args) else vars(args)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True, default=str)
        file.write("\n")
    return path


def set_seed(seed: int, torch_deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
