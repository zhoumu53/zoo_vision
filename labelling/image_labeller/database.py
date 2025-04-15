from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import TypeAlias

TInstanceId: TypeAlias = int


@dataclass
class Record:
    positive_points: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )  # [point, coords]
    negative_points: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )  # [point, coords]

    segmentation: npt.NDArray[np.uint8] | None = None  # [H,W]


@dataclass
class Database:
    image_path: Path
    image: npt.NDArray[np.uint8]  # [H,W,3]

    is_dirty: bool = False

    records: dict[TInstanceId, Record] = field(default_factory=dict)
    segmented_image: npt.NDArray[np.uint8] | None = None  # [H,W,3]

    def get_or_add_record(
        self,
        instance_id: TInstanceId,
    ) -> Record:
        record = self.records.get(instance_id)
        if record is None:
            # Matching node not found, create a new one
            record = Record()
            self.records[instance_id] = record
            self.is_dirty = True
        return record

    def add_point(
        self,
        instance_id: int,
        point: npt.NDArray[np.float32],
        is_positive: bool,
    ) -> None:
        assert point.shape == (1, 2)

        record = self.get_or_add_record(instance_id)

        # Check if the point cancels out with another point
        other_points = record.negative_points if is_positive else record.positive_points
        point_cancelled = False
        for i, other_point in enumerate(other_points):
            diff = np.linalg.norm(other_point - point)
            if diff < 10:
                other_points = np.delete(other_points, (i), axis=0)
                point_cancelled = True
                break

        if point_cancelled:
            if is_positive:
                record.negative_points = other_points
            else:
                record.positive_points = other_points
        else:
            # Add new point
            if is_positive:
                record.positive_points = np.concatenate([record.positive_points, point])
            else:
                record.negative_points = np.concatenate([record.negative_points, point])

        # Check if record is empty
        if (
            record.positive_points.shape[0] == 0
            and record.negative_points.shape[0] == 0
        ):
            self.records.pop(instance_id)

        # Mask needs recalculating
        record.segmentation = None
        self.segmented_image = None

        self.is_dirty = True


_database = Database(image_path=Path(), image=np.empty((0, 0), dtype=np.uint8))


def active_db() -> Database:
    global _database
    return _database


def set_db(db: Database) -> None:
    global _database
    _database = db
