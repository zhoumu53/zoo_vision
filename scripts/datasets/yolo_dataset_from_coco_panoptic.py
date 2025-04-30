from project_root import PROJECT_ROOT, DATASETS_ROOT

import json
from tqdm import tqdm
from pathlib import Path
from simple_parsing import ArgumentParser
from dataclasses import dataclass
import shutil
import numpy as np
import cv2

YOLO_ELEPHANT_CLASS_IDX: int = 0


@dataclass
class Args:
    input_json: Path
    output_path: Path


def parse_args() -> Args:
    parser = ArgumentParser()
    parser.add_arguments(Args, "args")
    ns = parser.parse_args()
    return ns.args


def name_from_id(id: int, suffix: str) -> str:
    return f"{id:06}{suffix}"


def main():
    args = parse_args()
    print(args)
    assert args.input_json.exists()

    src_path = args.input_json.parent

    with args.input_json.open() as f:
        ds_json = json.load(f)

    # Copy images
    image_id_map = {}
    next_image_id = 0
    dst_path = args.output_path / "images"
    dst_path.mkdir(parents=True, exist_ok=True)

    for image in tqdm(ds_json["images"], desc="Images"):
        id = next_image_id
        image_id_map[image["id"]] = id
        next_image_id += 1

        src_filename = src_path / image["file_name"]
        dst_filename = dst_path / name_from_id(id, src_filename.suffix)
        shutil.copy2(src_filename, dst_filename)

    # Copy labels
    dst_path = args.output_path / "labels"
    dst_path.mkdir(parents=True, exist_ok=True)
    for annotation in tqdm(ds_json["annotations"], desc="Annotations"):
        image_id = image_id_map[annotation["image_id"]]

        dst_filename = dst_path / name_from_id(image_id, ".txt")
        if dst_filename.exists():
            print(f"Duplicate annotation {annotation['file_name']}")
            continue

        segmentation = cv2.imread(
            src_path / annotation["file_name"], flags=cv2.IMREAD_GRAYSCALE
        )
        height = segmentation.shape[0]
        width = segmentation.shape[1]

        labels = []

        # Go over every instance in the segmentation
        instance_ids = [
            int(id) for id in set(np.unique(segmentation)) if id is not None
        ]
        for instance_id in instance_ids:
            if instance_id == 0:
                continue
            mask = (segmentation == instance_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )  # Find contours

            for contour in contours:
                if (
                    len(contour) >= 3
                ):  # YOLO requires at least 3 points for a valid segmentation
                    contour = contour.squeeze()  # Remove single-dimensional entries
                    label = [YOLO_ELEPHANT_CLASS_IDX]
                    for point in contour:
                        # Normalize the coordinates
                        label.append(
                            round(point[0] / width, 6)
                        )  # Rounding to 6 decimal places
                        label.append(round(point[1] / height, 6))
                    labels.append(label)

        # Write out labels to disk
        with dst_filename.open("w") as f:
            for label in labels:
                line = " ".join(map(str, label))
                f.write(line + "\n")


if __name__ == "__main__":
    main()
