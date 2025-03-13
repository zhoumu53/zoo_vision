import numpy as np
import torch
from typing import Any
from torchvision.datasets.folder import DatasetFolder
from collections import defaultdict


def sort_samples_by_class(targets):
    samples_by_class = defaultdict(list)
    for sample_idx, class_idx in enumerate(targets):
        samples_by_class[class_idx].append(sample_idx)
    return samples_by_class


class IdentitySequenceDataset:
    def __init__(self, time_count: int, image_dataset: DatasetFolder):
        self.time_count = time_count
        self.image_dataset = image_dataset

        # Build lists of samples per class
        self.samples_by_class = sort_samples_by_class(self.image_dataset.targets)

        self.classes = self.image_dataset.classes
        self.targets = self.image_dataset.targets

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        sample0, target = self.image_dataset[index]

        channels, height, width = sample0.shape
        sample = torch.empty(
            (self.time_count, channels, height, width), dtype=sample0.dtype
        )
        sample[0] = sample0

        sample_indices = self.samples_by_class[target]
        for t in range(1, self.time_count):
            index_t = int(np.random.uniform(0, len(sample_indices) - 1))
            sample_t, target_t = self.image_dataset[sample_indices[index_t]]
            assert target_t == target

            sample[t] = sample_t

        return sample, target

    def __len__(self) -> int:
        return len(self.image_dataset)
