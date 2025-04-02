from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
from typing import TypeAlias

TInstanceId: TypeAlias = int


@dataclass
class Record:
    frame: int
    instance_id: TInstanceId
    name: str
    positive_points: np.ndarray = field(
        default_factory=lambda: np.ndarray((0, 2))
    )  # [point, coords]
    negative_points: np.ndarray = field(
        default_factory=lambda: np.ndarray((0, 2))
    )  # [point, coords]

    segmentation: np.ndarray | None = None  # [H,W]


@dataclass
class DatabaseFrame:
    frame: int
    original_image: np.ndarray  # [H,W,3]

    records: dict[TInstanceId, Record] = field(default_factory=dict)
    segmented_image: np.ndarray | None = None  # [H,W,3]


@dataclass
class Database:
    video_path: Path
    json_path: Path
    frames: dict[int, DatabaseFrame] = field(default_factory=dict)
    is_dirty: bool = False

    def get_or_add_frame(
        self,
        frame_index: int,
        original_image: np.ndarray,
    ) -> DatabaseFrame:
        frame_data = self.frames.get(frame_index)
        if frame_data is None:
            frame_data = DatabaseFrame(
                frame=frame_index,
                records={},
                original_image=original_image,
                segmented_image=None,
            )
            self.frames[frame_index] = frame_data
            self.is_dirty = True
        return frame_data

    def get_or_add_record(
        self,
        frame_data: DatabaseFrame,
        instance_id: TInstanceId,
        name: str,
    ) -> Record:
        record = frame_data.records.get(instance_id)
        if record is not None and record.name != name:
            # Instance id found but name doesn't match.
            # Create a new record with a new instance id
            record = None
            instance_id = (
                np.max([r.instance_id for r in frame_data.records.values()]) + 1
            )

        if record is None:
            # Matching node not found, create a new one
            record = Record(
                frame=frame_data.frame,
                instance_id=instance_id,
                name=name,
            )
            frame_data.records[instance_id] = record
            self.is_dirty = True
        return record

    def add_point(
        self,
        record: Record,
        point: np.ndarray,
        is_positive: bool,
    ) -> None:
        assert point.shape == (1, 2)

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
        # Mask needs recalculating
        record.segmentation = None
        self.is_dirty = True


_database = Database(video_path=Path(), json_path=Path())


def active_db() -> Database:
    global _database
    return _database


def set_db(db: Database) -> None:
    global _database
    _database = db
