from project_root import PROJECT_ROOT, DATASETS_ROOT

import json
from pathlib import Path
from enlighten import Counter as ECounter
import argparse
import datetime

BACKGROUND_CATEGORY_ID = 0
ELEPHANT_CATEGORY_ID = 1


class IDGenerator:
    def __init__(self):
        self.next_id = 1

    def create(self) -> int:
        id = self.next_id
        self.next_id += 1
        return id


def create_elephant_annotation(input_paths: list[Path]):
    return {
        "info": {
            "description": f"Zoo Zurich Elephants - merge of {','.join(map(str, input_paths))}",
            "url": "n/a",
            "version": "0.1",
            "year": 2025,
            "contributor": "Zoo Zurich",
            "date_created": str(datetime.datetime.now()),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", nargs="*", type=Path, required=True)
    parser.add_argument("--output_path", "-o", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths: list[Path] = args.input_path
    output_path = Path.resolve(args.output_path)
    output_root = output_path.parent

    # Validate
    assert output_path.suffix == ".json"
    if len(input_paths) < 2:
        raise RuntimeError(f"At least two inputs are needed")
    for path in input_paths:
        if not path.exists():
            raise RuntimeError(f"Input path {path} does not exist")

    print(f"Storing merged ds in {str(output_path)}")

    # Initialize ds
    merged_ds = create_elephant_annotation(input_paths)
    gen_image_id = IDGenerator()

    pbar = ECounter(desc="Merging datasets", total=len(input_paths))
    for input_path in pbar(input_paths):
        with input_path.open() as f:
            input_ds = json.load(f)

        assert input_ds["categories"][0]["id"] in [
            BACKGROUND_CATEGORY_ID,
            ELEPHANT_CATEGORY_ID,
        ]

        print(
            f"Dataset: {input_path.name}, images={len(input_ds['images'])}, annotations={len(input_ds['annotations'])}"
        )
        # Copy image records
        input_root = input_path.parent
        image_new_id_from_input_id = {}
        for image in input_ds["images"]:
            # Create new image id
            new_id = gen_image_id.create()
            image_new_id_from_input_id[image["id"]] = new_id
            image["id"] = new_id

            # Update path
            image_path: Path = input_root / image["file_name"]
            image["file_name"] = str(image_path.relative_to(output_root))

            # Add to merged ds
            merged_ds["images"].append(image)

        # Copy annotations
        for annotation in input_ds["annotations"]:
            # Update image id
            annotation["image_id"] = image_new_id_from_input_id[annotation["image_id"]]

            # Update path
            segmentation_path: Path = input_root / annotation["file_name"]
            annotation["file_name"] = str(segmentation_path.relative_to(output_root))

            # Add to merged ds
            merged_ds["annotations"].append(annotation)

    print(
        f"Dataset: merged, images={len(merged_ds['images'])}, annotations={len(merged_ds['annotations'])}"
    )
    with (output_path).open("w") as f:
        json.dump(merged_ds, f, indent=1)


if __name__ == "__main__":
    main()
