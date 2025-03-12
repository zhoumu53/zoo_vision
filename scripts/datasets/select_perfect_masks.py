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


def parse_color(hex_color: str) -> np.ndarray:
    assert len(hex_color) == 7
    assert hex_color[0] == "#"
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return np.array([r, g, b], dtype=np.uint8)


def process_image(
    img_file: Path, seg_file: Path, good_path: Path, bad_path: Path
) -> None:
    # Load
    im_color: np.ndarray = cv2.imread(img_file)
    im_segmentation: np.ndarray = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)

    config_individuals = {
        "Chandra": {"id": 1, "color": "#F5FFC6"},
        "Indi": {"id": 2, "color": "#B4E1FF"},
        "Fahra": {"id": 3, "color": "#AB87FF"},
        "Panang": {"id": 4, "color": "#EDBBB4"},
        "Thai": {"id": 5, "color": "#C1FF9B"},
        "Ceyla": {"id": 6, "color": "#FFFF00"},
    }
    color_from_id = {
        v["id"]: parse_color(v["color"]) for v in config_individuals.values()
    }
    name_from_id = {v["id"]: k for k, v in config_individuals.items()}

    # Make colored mask
    merge = im_color.copy()
    ALPHA = 0.4
    greys = np.unique(im_segmentation)
    for grey in greys:
        if grey == 0:
            # Background id
            continue
        id = label_from_grey(grey)
        mask = im_segmentation == grey
        color = color_from_id[id]
        name = name_from_id[id]

        for c in range(3):
            merge_c = merge[:, :, c]
            merge_c[mask] = (1 - ALPHA) * merge_c[mask] + ALPHA * color[c]
        merge = merge.astype(np.uint8)

        bbox = bbox_from_mask(mask)
        cv2.putText(merge, name, [bbox[0], bbox[1]], 0, 1, color=[255, 255, 255])
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", merge)

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
            # Good, move to good folder
            shutil.move(img_file, good_path / img_file.name)
            shutil.move(seg_file, good_path / seg_file.name)
            keepAsking = False
        elif key == "a":
            # Bad , move to good folder
            shutil.move(img_file, bad_path / img_file.name)
            shutil.move(seg_file, bad_path / seg_file.name)
            keepAsking = False
        else:
            print(f"Unkonwn: {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        "-i",
        type=Path,
        default=Path.home() / "data/elephants/d2/images",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=Path,
        default=Path.home() / "data/elephants/d2/images_clean",
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

    files = list(input_path.glob("**/*_img.jpg"))

    pbar = pbar_manager.counter(total=len(files), desc="", unit="file")
    for img_file in pbar(files):
        pbar.desc = f"{img_file.name}"
        pbar.update(incr=0, force=True)

        seg_file = Path(str(img_file).replace("_img.jpg", "_seg.png"))

        process_image(img_file, seg_file, good_path, bad_path)


if __name__ == "__main__":
    main()
