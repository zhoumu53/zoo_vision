from project_root import PROJECT_ROOT, DATASETS_ROOT

import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

from segmentation_utils import bbox_from_mask

BACKGROUND_CATEGORY_ID = 0
ELEPHANT_CATEGORY_ID = 1


class IDGenerator:
    def __init__(self):
        self.next_id = 1

    def create(self) -> int:
        id = self.next_id
        self.next_id += 1
        return id


def create_elephant_annotation():
    return {
        "info": {
            "description": "Zoo Zurich Elephants 2025 Dataset",
            "url": "n/a",
            "version": "0.1",
            "year": 2025,
            "contributor": "Zoo Zurich",
            "date_created": "2025/02/01",
        },
        "licenses": [{"url": "non-public", "id": 1, "name": "Non-public"}],
        "categories": [
            {
                "supercategory": "background",
                "id": BACKGROUND_CATEGORY_ID,
                "name": "background",
            },
            {"supercategory": "animal", "id": ELEPHANT_CATEGORY_ID, "name": "elephant"},
        ],
        "images": [],
        "annotations": [],
    }


def make_image(id: int, name: str, height: int, width: int, date_captured: str):
    return {
        "license": 1,
        "file_name": name,
        "height": height,
        "width": width,
        "date_capture": date_captured,
        "id": id,
    }


def annotation_from_segmentation(
    input_path: Path, segmentation_file: Path, image_id: int
):
    annotation = {
        "image_id": image_id,
        "file_name": str(segmentation_file.relative_to(input_path)),
        "segments_info": [],
    }

    segmentation: np.ndarray = cv2.imread(segmentation_file, flags=cv2.IMREAD_GRAYSCALE)
    instance_ids = [int(id) for id in set(np.unique(segmentation))]
    for instance_id in instance_ids:
        # See https://cocodataset.org/#format-data
        segment_id = instance_id + instance_id * 256 + instance_id * 256 * 256

        mask = segmentation == instance_id
        area = int(np.sum(mask))
        bbox = bbox_from_mask(mask)

        segment = {
            "area": area,
            "iscrowd": instance_id == 0,
            "image_id": image_id,
            "bbox": bbox,
            "category_id": (
                BACKGROUND_CATEGORY_ID if instance_id == 0 else ELEPHANT_CATEGORY_ID
            ),
            "id": segment_id,
        }
        annotation["segments_info"].append(segment)
    return annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input_path

    elephant_annotations = create_elephant_annotation()
    gen_image_id = IDGenerator()

    files = [f.relative_to(input_path) for f in input_path.glob("**/*_img.jpg")]
    print(f"Total {len(files)} files")
    print(f"For example: {files[0]}")

    for file in tqdm(files):
        image_id = gen_image_id.create()
        segmentation_file = input_path / str(file).replace("_img.jpg", "_seg.png")

        # Need to read image to get size
        im = cv2.imread(input_path / file)

        image_json = make_image(
            id=image_id,
            name=str(file),
            height=im.shape[0],
            width=im.shape[1],
            date_captured="2025-02-01",
        )

        annotation_json = annotation_from_segmentation(
            input_path=input_path,
            segmentation_file=segmentation_file,
            image_id=image_id,
        )

        elephant_annotations["images"].append(image_json)
        elephant_annotations["annotations"].append(annotation_json)

    annotation_count = len(elephant_annotations["annotations"])
    print(f"{annotation_count=}")
    with (input_path / "annotations_pan.json").open("w") as f:
        json.dump(elephant_annotations, f, indent=1)


if __name__ == "__main__":
    main()
