from project_root import PROJECT_ROOT, DATASETS_ROOT

import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

from segmentation_utils import bbox_from_mask

# ELEPHANT_CATEGORY_ID = 22  # Same as in COCO
ELEPHANT_CATEGORY_ID = 1  # COCO with only one category


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
            {"supercategory": "animal", "id": ELEPHANT_CATEGORY_ID, "name": "elephant"}
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


def binary_mask_to_rle_np(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle


def annotations_from_segmentation(
    id: IDGenerator, segmentation_file: Path, image_id: int
):
    segmentation: np.ndarray = cv2.imread(segmentation_file, flags=cv2.IMREAD_GRAYSCALE)
    instance_ids = list(set(np.unique(segmentation)) - {0})
    annotations = []
    for instance_id in instance_ids:
        mask = segmentation == instance_id
        area = int(np.sum(mask))
        bbox = bbox_from_mask(mask)

        h, w = mask.shape
        mask_rle = binary_mask_to_rle_np(mask)
        # mask = mask.transpose().reshape(h,w)
        # mask = np.asfortranarray(mask)

        # mask_crle = coco_mask.encode(coco_mask.frPyObjects(mask_rle,h,w))
        # mask_crle["counts"]=list(mask_rle["counts"])
        # mask_compressed = coco_mask.frPyObjects(mask_rle, mask_rle.get('size')[0], mask_rle.get('size')[1])
        # mask_compressed =mask_rle

        annotation = {
            "segmentation": [mask_rle],
            "area": area,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": bbox,
            "category_id": ELEPHANT_CATEGORY_ID,
            "id": id.create(),
        }
        annotations.append(annotation)
    return annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input_path

    elephant_annotations = create_elephant_annotation()
    gen_image_id = IDGenerator()
    gen_annotation_id = IDGenerator()

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

        annotations_json = annotations_from_segmentation(
            id=gen_annotation_id, segmentation_file=segmentation_file, image_id=image_id
        )

        elephant_annotations["images"].append(image_json)
        elephant_annotations["annotations"].extend(annotations_json)

    annotation_count = len(elephant_annotations["annotations"])
    print(f"{annotation_count=}")
    with (input_path / "annotations.json").open("w") as f:
        json.dump(elephant_annotations, f)


if __name__ == "__main__":
    main()
