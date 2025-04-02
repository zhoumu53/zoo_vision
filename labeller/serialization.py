from dataclasses import asdict
import os
import json
from database import active_db, Database, DatabaseFrame, Record
from pathlib import Path
import cv2
from typing import Any
import numpy as np


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


def serialize_database() -> None:
    json_path = active_db().json_path
    with json_path.open("w") as f:
        json.dump(
            [
                {
                    "frame": frame.frame,
                    "records": [
                        {
                            "name": record.name,
                            "ppoints": record.positive_points.tolist(),
                            "npoints": record.negative_points.tolist(),
                        }
                        for record in frame.records.values()
                    ],
                }
                for frame in active_db().frames.values()
            ],
            f,
            indent=2,
        )
    print(f"Saved points to {str(json_path)}")
    active_db().is_dirty = False


def deserialize_database(
    video_path: Path, labels_path: Path, video_reader: cv2.VideoCapture
) -> Database:
    json_path = get_db_serialization_path(video_path, labels_path)
    db = Database(video_path=video_path, json_path=json_path)

    if json_path.exists():
        try:
            with json_path.open("r") as f:
                data = json.load(f)

            for fdata in data:
                frame_index: int = fdata["frame"]
                records = {
                    idx: Record(
                        instance_id=idx,
                        frame=frame_index,
                        name=rdata["name"],
                        positive_points=np.array(rdata["ppoints"]).reshape((-1, 2)),
                        negative_points=np.array(rdata["npoints"]).reshape((-1, 2)),
                    )
                    for idx, rdata in enumerate(fdata["records"])
                }

                video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                _, image = video_reader.read()

                db.frames[frame_index] = DatabaseFrame(
                    frame=frame_index, original_image=image, records=records
                )

        except Exception as e:
            print(f"Error loading db from {str(json_path)}.\nError: {e}")

    db.is_dirty = False
    return db
