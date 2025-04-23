from dataclasses import asdict
from database import active_db, Database, Record
from pathlib import Path
import cv2
from typing import Any, cast
import numpy as np
import numpy.typing as npt

import project_root  # type: ignore
from labelling.common.drawing import update_frame_image
from labelling.common.utils import unwrap


def get_db_serialization_path(video_path: Path, labels_path: Path) -> Path:
    return labels_path / (video_path.with_suffix("").name + "_points.json")


def convert_np_to_list(a: dict[str, Any]) -> dict[str, Any]:
    for key in a.keys():
        value = a[key]
        if isinstance(value, np.ndarray):
            a[key] = value.tolist()
    return a


def get_fields(a: object, fields: list[str]) -> dict[str, Any]:
    ad = asdict(a)  # type:ignore
    return convert_np_to_list({k: ad[k] for k in fields})


def segmentation_path_from_image_path(image_path: Path) -> Path:
    return image_path.parent / image_path.name.replace(image_path.suffix, "_seg.png")


def grey_from_label(label, label_count):
    MIN_VALUE = 100
    MAX_VALUE = 255
    return MIN_VALUE + (MAX_VALUE - MIN_VALUE) / label_count * (label + 1)


def serialize_database() -> None:
    db = active_db()
    image_path = db.image_path
    seg_path = segmentation_path_from_image_path(image_path)
    if seg_path.exists():
        seg_path.unlink()

    if db.segmented_image is not None:
        # Build grayscale segmentation image from records
        seg_image = np.zeros((db.image.shape[0], db.image.shape[1]), dtype=np.uint8)
        record_count = len(db.records)
        for i, r in enumerate(db.records.values()):
            grey_value = grey_from_label(i, record_count)
            assert r.segmentation is not None
            seg_image[r.segmentation != 0] = grey_value

        cv2.imwrite(str(seg_path), seg_image)
        print(f"Saved segmentation image to {str(seg_path)}")
        # Only mark as clean if we saved an image
        active_db().is_dirty = False


def deserialize_database(image_path: Path) -> Database:
    image = cv2.imread(str(image_path))
    image = cast(npt.NDArray[np.uint8], image)
    db = Database(image_path=image_path, image=image)

    seg_path = segmentation_path_from_image_path(image_path)
    if seg_path.exists():
        seg_image = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        seg_image = cast(npt.NDArray[np.uint8], seg_image)
        # Decompose grayscale segmentation image into records
        labels = set(np.unique(seg_image).tolist()) - {0}  # Remove background label
        for i, label in enumerate(labels):
            mask = seg_image == label
            record = Record(segmentation=mask)
            db.records[i] = record

        db.segmented_image = update_frame_image(
            image, [unwrap(r.segmentation) for r in db.records.values()], [], []
        )
        db.is_dirty = False
    else:
        db.segmented_image = db.image
        # We should write the segmentation even if its an empty image
        db.is_dirty = True

    return db
