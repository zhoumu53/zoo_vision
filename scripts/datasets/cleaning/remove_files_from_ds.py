import numpy as np
import shutil
from pathlib import Path
import enlighten
import argparse

from project_root import PROJECT_ROOT

pbar_manager = enlighten.get_manager()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receives a list of valid image names and removes all images from a dataset that are not in that list."
    )
    parser.add_argument("--input_path", "-i", type=Path)
    parser.add_argument("--names", "-n", type=Path)
    parser.add_argument("--discard_path", "-d", type=Path)
    return parser.parse_args()


def get_root(name: str) -> str:
    return name.replace("_img.jpg", "").replace("_seg.png", "")


def main():
    args = parse_args()
    input_path: Path = args.input_path
    names_file: Path = args.names
    discard_path: Path = args.discard_path

    discard_path.mkdir(exist_ok=True, parents=True)

    with names_file.open() as f:
        names = f.readlines()
    names = [l.strip().replace("_img.jpg", "") for l in names]

    print(f"Valid names: {len(names)}")

    files = [f for f in input_path.glob("**/*")]
    print(f"Total files in dataset: {len(files)}")

    invalid_files = [f for f in files if f.is_file() and get_root(f.name) not in names]
    if len(invalid_files) == 0:
        print("No files to remove")
        return

    print(
        f"Files to remove: {len(invalid_files)} ({len(invalid_files)/len(files):.1%})"
    )

    pbar = pbar_manager.counter(total=len(invalid_files), unit="file")
    for file in pbar(invalid_files):
        shutil.move(file, discard_path / file.name)


if __name__ == "__main__":
    main()
