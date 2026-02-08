from datetime import datetime, time
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import bisect
from dataclasses import dataclass
import psycopg2
import cv2
import pytz
import asyncio
import re

logger = logging.getLogger(__name__)

MAX_DISTANCE_SEC = 5
CAMERA_ID_FROM_NAME = {
    "zag_elp_cam_016": 0,
    "zag_elp_cam_017": 1,
    "zag_elp_cam_018": 2,
    "zag_elp_cam_019": 3,
}
INDIVIDUAL_FROM_ID = {
    0: "Invalid",
    1: "Chandra",
    3: "Farha",
    2: "Indi",
    4: "Panang",
    5: "Thai",
}
INDIVIDUAL_ID_FROM_NAME = {name.lower(): id for id, name in INDIVIDUAL_FROM_ID.items()}

COLOR_FROM_INDIVIDUAL_ID = {
    0: "#777777",
    1: "#73BF69",
    2: "#F2CC0C",
    3: "#5794F2",
    4: "#FF9830",
    5: "#F2495C",
}


@dataclass
class DayData:
    csv_paths: list[Path]
    start_timestamps: list[datetime]
    end_timestamps: list[datetime]
    csv_data: list[pd.DataFrame]


@dataclass
class Detection:
    csv_path: str
    timestamp: datetime
    image: np.ndarray
    bbox_tlhw: tuple[int, int, int, int]
    color: str
    identity_id: int
    identity_name: str
    behaviour_id: int
    behaviour_name: str


def get_day_path(track_root_path: Path, camera: str, timestamp: datetime) -> Path:
    return track_root_path / camera / timestamp.strftime("%Y-%m-%d")


def get_video_path(csv_path: Path) -> Path:
    path = csv_path.with_suffix(".mkv")
    if path.exists():
        return path
    path = csv_path.with_suffix(".mp4")
    if path.exists():
        return path
    raise RuntimeError(f"Video for track {csv_path} does not exist on disk.")


def read_track_ranges(path: Path) -> DayData:
    data = DayData([], [], [], [])
    for f in sorted(path.glob("*.csv")):
        # Check that name exactly matches our format, e.g. T082241_ID007719.csv
        if not re.match(r"T\d{6}_ID\d{6}.csv", f.name):
            continue

        # Read track details
        df_timestamps = pd.read_csv(
            f,
            usecols=lambda x: x
            in [
                "timestamp",
                "bbox_top",
                "bbox_left",
                "bbox_bottom",
                "bbox_right",
                "bbox_top2",
                "bbox_left2",
                "bbox_bottom2",
                "bbox_right2",
            ],
            parse_dates=["timestamp"],
        )
        if len(df_timestamps) == 0:
            # Empty file, ignore silently
            continue
        if not pd.api.types.is_datetime64_dtype(df_timestamps["timestamp"]):
            # Error in file, ignore
            logger.error(
                f"Ignoring csv {f}. Timestamp is of type {df_timestamps['timestamp'].dtype}"
            )
            continue

        # All server data is stored in CET timezone
        # FIXME: we should store timezone in the timestamp itself
        df_timestamps["timestamp"] = df_timestamps["timestamp"].dt.tz_localize(
            pytz.timezone("Europe/Zurich")
        )

        data.csv_paths.append(f)
        data.csv_data.append(df_timestamps)
        data.start_timestamps.append(df_timestamps["timestamp"].iloc[0])
        data.end_timestamps.append(df_timestamps["timestamp"].iloc[-1])

    return data


async def find_track_images(
    camera: str, day_data: DayData, timestamp: datetime
) -> list[Detection]:
    detections = []
    count = len(day_data.start_timestamps)
    for i in range(count):
        start = day_data.start_timestamps[i]
        if start > timestamp:
            # All tracks start too late from this index on
            break

        end = day_data.end_timestamps[i]
        if end < timestamp:
            # Timestamp is not in the range, skip
            continue

        csv_data = day_data.csv_data[i]
        frame_timestamps = csv_data["timestamp"]
        ind = bisect.bisect_right(frame_timestamps.to_list(), timestamp)
        if ind == 0:
            continue
        csv_index = ind - 1
        video_timestamp = frame_timestamps[csv_index]
        distance = abs((video_timestamp - timestamp).total_seconds())
        if distance > MAX_DISTANCE_SEC:
            continue

        video_frameid = csv_index

        # Read info from csv
        if "bbox_top2" in csv_data:
            bbox_tlbr = (
                csv_data["bbox_top2"].iloc[csv_index],
                csv_data["bbox_left2"].iloc[csv_index],
                csv_data["bbox_bottom2"].iloc[csv_index],
                csv_data["bbox_right2"].iloc[csv_index],
            )
        else:
            # TODO: top and left dimensions normalized with the wrong values!!!
            width = 1060 / 2688 * 1520
            height = 600 / 1520 * 2688
            # TODO: top and left dimensions are flipped in the csv!!!
            bbox_tlbr = (
                csv_data["bbox_left"].iloc[csv_index] / width,
                csv_data["bbox_top"].iloc[csv_index] / height,
                csv_data["bbox_right"].iloc[csv_index] / width,
                csv_data["bbox_bottom"].iloc[csv_index] / height,
            )
        bbox_tlhw = (
            bbox_tlbr[0],
            bbox_tlbr[1],
            bbox_tlbr[2] - bbox_tlbr[0],
            bbox_tlbr[3] - bbox_tlbr[1],
        )

        # Open the actual video and skip to desired frame
        csv_path = day_data.csv_paths[i]
        video_path = get_video_path(csv_path)
        video = cv2.VideoCapture(str(video_path))
        video.set(cv2.CAP_PROP_POS_FRAMES, video_frameid)
        ok, image = video.read()
        if not ok:
            logger.error(
                "Error reading video: %s, frame: %d", video_path, video_frameid
            )
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections.append(
            Detection(
                csv_path=str(csv_path),
                timestamp=video_timestamp,
                image=image,
                bbox_tlhw=bbox_tlhw,
                color="",
                identity_id=0,
                identity_name="",
                behaviour_id=0,
                behaviour_name="",
            )
        )
    # Query database for the identities
    if len(detections) > 0:
        all_tracknames = [Path(d.csv_path).stem for d in detections]

        with psycopg2.connect(
            "dbname=zoo_vision user=grafanareader password=asdf"
        ) as db_connection:
            with db_connection.cursor() as db_cursor:
                db_cursor.execute(
                    "SELECT track_filename, identity_id "
                    + "FROM tracks "
                    + "WHERE camera_id=%s AND track_filename IN ("
                    + ",".join(["%s" for _ in all_tracknames])
                    + ")",
                    (CAMERA_ID_FROM_NAME[camera.lower()], *all_tracknames),
                )
                results = db_cursor.fetchall()
                identity_from_stem = {x[0]: x[1] for x in results}
        for detection, csv_stem in zip(detections, all_tracknames):
            detection.identity_id = identity_from_stem.get(csv_stem, 0)
            detection.identity_name = INDIVIDUAL_FROM_ID[detection.identity_id]
            detection.color = COLOR_FROM_INDIVIDUAL_ID[detection.identity_id]

    return detections
