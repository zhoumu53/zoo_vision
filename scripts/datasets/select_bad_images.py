import numpy as np
import sys
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import enlighten
import torch
import argparse
import shutil

from typing import Any

from project_root import PROJECT_ROOT
from scripts.datasets.segmentation_utils import bbox_from_mask, label_from_grey

pbar_manager = enlighten.get_manager()


def process_image(img_file: Path, good_path: Path, bad_path: Path) -> None:
    c = img_file.parent.name

    # Load
    im_color: np.ndarray = cv2.imread(img_file)

    # Make colored mask
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", im_color)

    keepAsking = True
    print(f"{str(img_file.name)}")
    while keepAsking:
        print("bad='a', good='d' , quit='q'?")
        key = cv2.waitKey()
        key = chr(key)
        print(key)
        if key == "q":
            sys.exit()
        elif key == "d":
            # Good, do nothing
            shutil.move(img_file, good_path / c / img_file.name)
            keepAsking = False
        elif key == "a":
            # Bad , move to good folder
            shutil.move(img_file, bad_path / c / img_file.name)
            keepAsking = False
        else:
            print(f"Unkonwn: {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        "-i",
        type=Path,
        default=Path.home() / "data/elephants/identity/dataset/mix/train",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=Path,
        default=Path.home() / "data/elephants/identity/dataset/certainty",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path: Path = args.input_path
    output_path: Path = args.output_path
    good_path = output_path / "good"
    bad_path = output_path / "bad"
    good_path.mkdir(exist_ok=True, parents=True)
    bad_path.mkdir(exist_ok=True, parents=True)

    # Create class
    classes = [f.name for f in input_path.glob("*")]
    print(f"{classes=}")
    for c in classes:
        (good_path / c).mkdir(exist_ok=True)
        (bad_path / c).mkdir(exist_ok=True)

    files = sorted(list(input_path.glob("**/*_img.jpg")))

    pbar = pbar_manager.counter(total=len(files), desc="", unit="file")
    for img_file in pbar(files):
        pbar.desc = f"{img_file.name}"
        pbar.update(incr=0, force=True)

        process_image(img_file, good_path, bad_path)


if __name__ == "__main__":
    main()
