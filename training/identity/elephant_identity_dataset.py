import numpy as np
import torch
from pathlib import Path
from typing import Any, Callable, Union, Optional
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import ConcatDataset
from collections import defaultdict


class UncertainImageFolder(ImageFolder):
    def __init__(
        self,
        root: Union[str, Path],
        uncertainty: float,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        super().__init__(
            root, transform, target_transform, loader, is_valid_file, allow_empty
        )
        self.uncertainty = uncertainty

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        sample, target = super().__getitem__(index)

        class_count = len(self.classes)
        min_prob = 1 / class_count
        max_prob = 1 + (min_prob - 1) * self.uncertainty
        other_probs = (1 - max_prob) / (class_count - 1)
        probs = torch.full([class_count], other_probs, dtype=torch.float32)
        probs[target] = max_prob

        return sample, probs


class ElephantIdentityDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path] = None,  # type: ignore[assignment]
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        super().__init__(root, None, transform, target_transform)

        sub_dirs = ["good", "bad", "terrible"]
        uncertainties = [0.3, 0.8, 1.0]

        self.datasets = {
            name: UncertainImageFolder(
                Path(root) / name, uncertainty, transform, target_transform
            )
            for name, uncertainty in zip(sub_dirs, uncertainties)
        }
        self.classes = self.datasets["good"].classes
        self.targets = (
            self.datasets["good"].targets
            + self.datasets["bad"].targets
            + self.datasets["terrible"].targets
        )
        self.dataset = ConcatDataset(self.datasets.values())

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)
