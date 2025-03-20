import numpy as np
import torch
from typing import Any
from torchvision.datasets.folder import DatasetFolder
from elephant_identity_dataset import ElephantIdentityDataset
from collections import defaultdict


def sort_samples_by_class(targets):
    samples_by_class = defaultdict(list)
    for sample_idx, class_idx in enumerate(targets):
        samples_by_class[class_idx].append(sample_idx)
    return dict(samples_by_class)


def vector_entropy(x: torch.Tensor):
    eps = 1e-6
    return -torch.sum(x * torch.log(x + eps))


class IdentitySequenceDataset:
    def __init__(
        self,
        time_count: int,
        uncertain_rate: float,
        image_dataset: DatasetFolder | ElephantIdentityDataset,
    ):
        self.time_count = time_count
        self.uncertain_rate = uncertain_rate
        self.image_dataset = image_dataset

        # Build lists of samples per class
        self.samples_by_class = sort_samples_by_class(self.image_dataset.targets)

        self.classes = self.image_dataset.classes
        self.targets = self.image_dataset.targets

        self.uncertain_dataset = (
            image_dataset.datasets["terrible"]
            if isinstance(self.image_dataset, ElephantIdentityDataset)
            else None
        )

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        sample0, target0 = self.image_dataset[index]
        target_idx = self.image_dataset.targets[index]

        channels, height, width = sample0.shape
        sample = torch.empty(
            (self.time_count, channels, height, width), dtype=sample0.dtype
        )
        target = (
            torch.empty((self.time_count, target0.shape[0]), dtype=target0.dtype)
            if isinstance(target0, torch.Tensor)
            else torch.empty((self.time_count,), dtype=torch.int64)
        )

        sample[0] = sample0
        target[0] = target0

        sample_indices = self.samples_by_class[target_idx]
        for t in range(1, self.time_count):
            if np.random.uniform(0, 1) < self.uncertain_rate:
                index_t = int(np.random.uniform(0, len(self.uncertain_dataset) - 1))
                sample_t, target_t = self.uncertain_dataset[index_t]
            else:
                index_t = int(np.random.uniform(0, len(sample_indices) - 1))
                sample_t, target_t = self.image_dataset[sample_indices[index_t]]

            sample[t] = sample_t
            target[t] = target_t

        # Make sure entropy always decreases

        if target.dtype != torch.int64:
            prob = target[0]
            entropy = vector_entropy(prob)
            for t in range(1, self.time_count):
                new_entropy = vector_entropy(target[t])
                if new_entropy < entropy:
                    prob = target[t]
                    entropy = new_entropy
                target[t] = prob

        return sample, target

    def __len__(self) -> int:
        return len(self.image_dataset)
