import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Any

from project_root import PROJECT_ROOT, DATASETS_ROOT

import fiftyone as fo
import fiftyone.brain as fob


def delete_similar_images(path: Path, keep_count: int) -> None:
    ds = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageDirectory,
        dataset_dir=path,
        persistent=False,
    )

    print(f"Dataset at {path} has {len(ds)} images")

    fob.compute_uniqueness(ds)

    view = ds.sort_by("uniqueness")
    names = [sample["filepath"] for sample in view]

    if len(names) <= keep_count:
        print(f"Too few files in directory ({len(names)}, keeping all of them)")
        return

    delete_count = len(names) - keep_count
    res = input(
        f"About to delete {delete_count} ({delete_count/len(view):.0%}) files, are you sure? (y/N)"
    )
    res = res.lower()
    if res != "y":
        print("Aborting")
        return

    names = names[0:delete_count]
    for name in tqdm(names):
        Path.unlink(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=Path)
    parser.add_argument("--keep", "-k", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input_path
    keep_count = args.keep
    delete_similar_images(input_path, keep_count)


if __name__ == "__main__":
    main()
