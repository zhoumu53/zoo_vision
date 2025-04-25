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
        print(f"Size of {name}={len(dataset)}")
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


def merge_datasets(name: str, inputs: list[str], args: argparse.Namespace):
    exists = fo.dataset_exists(name)
    force = (
        args.force_all
        or name in args.force
        or np.any([input in args.force for input in inputs])
    )
    if not exists or force:
        if exists:
            print(f"Deleting {name}...")
            fo.delete_dataset(name)
        print(f"Building {name}...")
        ds = None
        for input in inputs:
            ds_input = fo.load_dataset(input)
            if ds is None:
                ds = ds_input.clone(name, persistent=True)
            else:
                ds.merge_samples(ds_input)

        print(f"Size of {name}={len(ds)}")
        return ds
    return None


def make_certainty_dataset(name):
    good_dirs = [
        "/home/dherrera/data/elephants/identity/dataset/certainty/train/good",
        "/home/dherrera/data/elephants/identity/dataset/certainty/val",
        "/home/dherrera/data/elephants/identity/dataset/id3",
        "/home/dherrera/data/elephants/identity/dataset/v4",
    ]
    bad_dirs = [
        "/home/dherrera/data/elephants/certainty/v1",
        "/home/dherrera/data/elephants/certainty/v2",
        "/home/dherrera/data/elephants/certainty/v4",
    ]

    samples = []
    for label, dirs in zip(["good", "bad"], [good_dirs, bad_dirs]):
        for dir in dirs:
            for file in Path(dir).glob("**/*.jpg"):
                sample = fo.Sample(filepath=file)
                sample["ground_truth"] = fo.Classification(label=label)
                samples.append(sample)
    dataset = fo.Dataset(name, persistent=True)
    dataset.add_samples(samples)
    return dataset


def main():
    args = parse_args()

    coco_root = DATASETS_ROOT / "elephants/segmentation/coco"
    elephant_ds_root = DATASETS_ROOT / "elephants"

    print(f"Existing datasets: {fo.list_datasets()}")

    # Coco dataset
    process_dataset(
        "coco-elephants-train2017",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=coco_root / "train2017",
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
            data_path=coco_root / "val2017",
            labels_path=coco_root / "annotations/elephants_val2017.json",
            name=name,
            persistent=True,
        ),
        args,
    )

    # Detection dataset
    process_dataset(
        "zoo-elephants-detection",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=elephant_ds_root / "segmentation",
            labels_path=elephant_ds_root / "segmentation/all_pan.json",
            name=name,
            persistent=True,
        ),
        args,
    )

    ###############################################################
    # Identity dataset
    def make_identity_dataset(name, dir):
        process_classification_dataset(
            name,
            lambda name: fo.Dataset.from_dir(
                dataset_type=fo.types.ImageClassificationDirectoryTree,
                dataset_dir=elephant_ds_root / dir,
                name=name,
                persistent=True,
            ),
            args,
        )

    make_identity_dataset(
        "zoo-elephants-identity-v1-curated", "identity/dataset/v1/train_curated"
    )
    make_identity_dataset("zoo-elephants-identity-v1-val", "identity/dataset/v1/val")
    make_identity_dataset("zoo-elephants-identity-d2", "identity/dataset/d2/train")
    make_identity_dataset(
        "zoo-elephants-identity-certainty-good", "identity/dataset/certainty/train/good"
    )
    make_identity_dataset(
        "zoo-elephants-identity-certainty-val", "identity/dataset/certainty/val"
    )
    make_identity_dataset("zoo-elephants-identity-id3", "identity/dataset/id3")
    make_identity_dataset("zoo-elephants-identity-v4", "identity/dataset/v4")
    merge_datasets(
        "zoo-elephants-identity",
        [
            "zoo-elephants-identity-v1-curated",
            "zoo-elephants-identity-v1-val",
            "zoo-elephants-identity-d2",
            "zoo-elephants-identity-certainty-good",
            "zoo-elephants-identity-certainty-val",
            "zoo-elephants-identity-id3",
            "zoo-elephants-identity-v4",
        ],
        args,
    )

    # process_classification_dataset(
    #     "zoo-elephants-identity-tracks",
    #     lambda name: fo.Dataset.from_dir(
    #         dataset_type=fo.types.ImageClassificationDirectoryTree,
    #         dataset_dir=Path(
    #             "/media/dherrera/ElephantExternal/elephants/tracks/tracks_apr03/maybe"
    #         ),
    #         name=name,
    #         persistent=True,
    #     ),
    #     args,
    # )

    ##############################################################
    # Behaviour dataset
    process_classification_dataset(
        "zoo-elephants-sleep-v1",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            dataset_dir=elephant_ds_root / "behaviour/sleep_v1",
            name=name,
            persistent=True,
        ),
        args,
    )

    process_classification_dataset(
        "zoo-elephants-sleep-v2",
        lambda name: fo.Dataset.from_dir(
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            dataset_dir=elephant_ds_root / "behaviour/sleep_v2",
            name=name,
            persistent=True,
        ),
        args,
    )

    merge_datasets(
        "zoo-elephants-sleep",
        ["zoo-elephants-sleep-v1", "zoo-elephants-sleep-v2"],
        args,
    )

    ##############################################################
    # Certainty dataset
    process_classification_dataset(
        "zoo-elephants-certainty",
        make_certainty_dataset,
        args,
    )


if __name__ == "__main__":
    main()
