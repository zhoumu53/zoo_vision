import numpy as np
import torch
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Callable

from project_root import PROJECT_ROOT, DATASETS_ROOT

import fiftyone as fo
import torchvision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loads all of our datasets into the fiftyone database for visualization/evaluation."
    )
    parser.add_argument("--force", "-f", nargs="+", default=[], type=str)
    parser.add_argument("--force_all", "-fa", action="store_true")
    return parser.parse_args()


def process_dataset(
    name: str, make_func: Callable[[str], fo.Dataset], args: argparse.Namespace
):
    exists = fo.dataset_exists(name)
    if args.force_all or name in args.force or not exists:
        if exists:
            print(f"Deleting {name}...")
            fo.delete_dataset(name)
        print(f"Building {name}...")
        dataset = make_func(name)
        return dataset

    return None


def process_classification_dataset(
    name: str, make_func: Callable[[str], fo.Dataset], args: argparse.Namespace
):
    dataset = process_dataset(name, make_func, args)
    if dataset is not None:
        dataset.classes = {
            "ground_truth": list(
                {
                    s.get_field("ground_truth").get_field("label")
                    for s in dataset.iter_samples()
                }
            )
        }


def main():
    args = parse_args()

    coco_root = DATASETS_ROOT / "coco"
    elephant_ds_root = DATASETS_ROOT / "elephants"

    print(f"Existing datasets: {fo.list_datasets()}")

    # Coco dataset
    process_dataset(
        "coco-elephants-train2017",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path="/home/dherrera/data/coco/train2017",
            labels_path=coco_root / "annotations/elephants_train2017.json",
            name=name,
            persistent=True,
        ),
        args,
    )

    process_dataset(
        "coco-elephants-val2017",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path="/home/dherrera/data/coco/val2017",
            labels_path=coco_root / "annotations/elephants_val2017.json",
            name=name,
            persistent=True,
        ),
        args,
    )

    # Detection dataset
    process_dataset(
        "zoo-elephants-detection-train",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=elephant_ds_root / "training_data",
            labels_path=elephant_ds_root / "training_data/annotations.json",
            name=name,
            persistent=True,
        ),
        args,
    )

    # Identity dataset
    process_classification_dataset(
        "zoo-elephants-identity-train",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            dataset_dir=elephant_ds_root / "identity/dataset/certainty/train/good",
            name=name,
            persistent=True,
        ),
        args,
    )

    process_classification_dataset(
        "zoo-elephants-identity-val",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            dataset_dir=elephant_ds_root / "identity/dataset/certainty/val",
            name=name,
            persistent=True,
        ),
        args,
    )

    # Behaviour dataset
    process_classification_dataset(
        "zoo-elephants-sleep-train",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            dataset_dir=elephant_ds_root / "behaviour/train",
            name=name,
            persistent=True,
        ),
        args,
    )

    process_classification_dataset(
        "zoo-elephants-sleep-val",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            dataset_dir=elephant_ds_root / "behaviour/val",
            name=name,
            persistent=True,
        ),
        args,
    )


if __name__ == "__main__":
    main()
