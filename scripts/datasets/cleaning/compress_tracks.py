import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Any, Generator
import shutil

from project_root import PROJECT_ROOT, DATASETS_ROOT

import fiftyone as fo
import fiftyone.brain as fob

DRY_RUN = False
IMAGE_EXTENSIONS = {".jpg", ".png"}


def delete_similar_images(path: Path, min_count: int, keep_count: int) -> None:
    image_count = len([f for f in path.iterdir() if f.suffix in IMAGE_EXTENSIONS])
    print(f"Dataset at {path} has {image_count} images")
    if image_count < min_count:
        print(f" Track is too small, deleting folder")
        if not DRY_RUN:
            shutil.rmtree(path)
        return

    if image_count <= keep_count:
        return

    ds = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageDirectory,
        dataset_dir=path,
        persistent=False,
    )
    assert len(ds) == image_count

    fob.compute_uniqueness(ds)

    view = ds.sort_by("uniqueness")
    names = [sample["filepath"] for sample in view]

    if len(names) <= keep_count:
        print(f"Too few files in directory ({len(names)}, keeping all of them)")
        return

    delete_count = len(names) - keep_count
    print(f"Deleting {delete_count} ({delete_count/len(view):.0%}) files...")

    names = names[0:delete_count]
    for name in tqdm(names):
        if not DRY_RUN:
            Path.unlink(name)


def glob_dirs_with_images(root: Path) -> Generator[Path, None, None]:
    for path in root.glob("**"):
        if not path.is_dir():
            # Not a directory, ignore
            continue
        has_images = False
        for file in path.iterdir():
            if file.suffix in IMAGE_EXTENSIONS:
                has_images = True
                break
        if not has_images:
            # No images in this directory, ignore
            continue

        yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=Path)
    parser.add_argument(
        "--merge_all",
        action="store_true",
        help="Merge all subfolders into a single ds and compute uniqueness between all images. Otherwise each subfolder is treated separately.",
    )
    parser.add_argument(
        "--min",
        "-n",
        type=int,
        default=15,
        help="If the track has less than this count, all files are deleted.",
    )
    parser.add_argument("--keep", "-k", type=int, default=200)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip the actual deletion. To make sure things work before deleting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path: Path = args.input_path
    min_count: int = args.min
    keep_count: int = args.keep

    global DRY_RUN
    DRY_RUN = args.dry_run

    if args.merge_all:
        delete_similar_images(input_path, min_count, keep_count)
    else:
        print("Collecting all dirs...")
        dirs_with_images = list(tqdm(glob_dirs_with_images(input_path)))
        for path in dirs_with_images:
            delete_similar_images(path, min_count, keep_count)


if __name__ == "__main__":
    main()
