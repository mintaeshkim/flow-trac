import numpy as np
import torch

from flow_trac.data import OfflineDataset


def test_offline_dataset_split_and_sample():
    size = 100
    transitions = {
        "observations": np.zeros((size, 3), dtype=np.float32),
        "actions": np.zeros((size, 2), dtype=np.float32),
        "rewards": np.arange(size, dtype=np.float32),
        "next_observations": np.ones((size, 3), dtype=np.float32),
        "terminations": np.zeros(size, dtype=np.float32),
        "truncations": np.zeros(size, dtype=np.float32),
    }
    dataset = OfflineDataset(transitions, torch.device("cpu"), seed=7)
    train_indices, validation_indices = dataset.train_validation_indices(0.1)
    assert len(train_indices) == 90
    assert len(validation_indices) == 10
    assert len(np.intersect1d(train_indices, validation_indices)) == 0

    batch = dataset.sample(16, validation_indices)
    assert batch.observations.shape == (16, 3)
    assert batch.actions.shape == (16, 2)
    assert set(batch.rewards.squeeze(-1).tolist()).issubset(set(validation_indices.astype(float)))
