import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import enlighten
import torch
import argparse

from typing import Any

from project_root import PROJECT_ROOT
from scripts.datasets.segmentation_utils import bbox_from_mask, label_from_grey

pbar_manager = enlighten.get_manager()

CROP_SIZE = 256


def crop_bbox(im_color: np.ndarray, bbox: tuple[int, int, int, int]):
    x0, y0, w, h = bbox
    bbox_aspect = w / h
    if bbox_aspect > 1:
        rescale_size = [CROP_SIZE, int(np.round(CROP_SIZE / bbox_aspect))]
    else:
        rescale_size = [int(np.round(CROP_SIZE * bbox_aspect)), CROP_SIZE]
    cx0 = int(np.floor((CROP_SIZE - rescale_size[0]) / 2))
    cy0 = int(np.floor((CROP_SIZE - rescale_size[1]) / 2))

    patch = im_color[y0 : (y0 + h), x0 : (x0 + w), :]

    # cv2 and torch methods should be equivalent
    USE_CV2 = False
    if USE_CV2:
        rescale = cv2.resize(patch, rescale_size, interpolation=cv2.INTER_LINEAR)
    else:
        patch_tensor = torch.from_numpy(patch).permute([2, 0, 1])
        patch_tensor = patch_tensor[None, ...]
        rescale_tensor = torch.nn.functional.interpolate(
            patch_tensor,
            [rescale_size[1], rescale_size[0]],
            mode="bilinear",
            antialias=True,
        )
        rescale = rescale_tensor[0].permute([1, 2, 0]).numpy()

    crop = np.zeros([CROP_SIZE, CROP_SIZE, 3], dtype=np.uint8)
    crop[cy0 : (cy0 + rescale_size[1]), cx0 : (cx0 + rescale_size[0]), :] = rescale
    return crop


def output_folder_for_class(id: int, name: str) -> str:
    return f"{id:02}_{name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=Path)
    parser.add_argument("--output_path", "-o", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path

    # Load config
    with (PROJECT_ROOT / "data/config.json").open() as f:
        config = json.load(f)
    name_from_id = {props["id"]: name for name, props in config["individuals"].items()}

    for id, name in name_from_id.items():
        (output_path / output_folder_for_class(id, name)).mkdir(
            parents=True, exist_ok=True
        )

    files = list(input_path.glob("**/*_img.jpg"))

    pbar = pbar_manager.counter(total=len(files), desc="Extracting crops", unit="file")
    for img_file in pbar(files):
        seg_file = Path(str(img_file).replace("_img.jpg", "_seg.png"))

        # Load
        im_color: np.ndarray = cv2.imread(img_file)
        im_segmentation: np.ndarray = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)

        greys = np.unique(im_segmentation)
        for grey in greys:
            if grey == 0:
                # Background id
                continue
            id = label_from_grey(grey)
            mask = im_segmentation == grey
            bbox = bbox_from_mask(mask)

            crop = crop_bbox(im_color, bbox)
            out_path = (
                output_path
                / output_folder_for_class(id, name_from_id[id])
                / img_file.name
            )
            cv2.imwrite(out_path, crop)


if __name__ == "__main__":
    main()
