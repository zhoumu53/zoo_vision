import shutil
import argparse
from enlighten import Counter
from pathlib import Path
from typing import Any

from project_root import PROJECT_ROOT, DATASETS_ROOT

import fiftyone as fo
import fiftyone.brain as fob


def flatten_folders(path: Path) -> None:
    classes = [f.name for f in path.glob("*") if f.is_dir()]
    print(f"Detected classes under {path}: {classes}")

    class_bar = Counter(total=len(classes), desc="Classes")
    for c in class_bar(classes):
        files = [f for f in (path / c).glob("**/*") if not f.is_dir()]
        file_bar = Counter(total=len(files), desc="Files")
        for i, file in enumerate(file_bar(files)):
            new_name = path / c / f"{i:08}{file.suffix}"
            shutil.move(file, new_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input_path
    flatten_folders(input_path)


if __name__ == "__main__":
    main()
