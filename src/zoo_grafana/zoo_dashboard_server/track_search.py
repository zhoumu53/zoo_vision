from datetime import datetime, time
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import bisect
from dataclasses import dataclass
import cv2

logger = logging.getLogger(__name__)

ROOT_FOLDER = Path("/data/xmas/tracks")
# CAMERAS = ["zag_elp_cam_016", "zag_elp_cam_017", "zag_elp_cam_018", "zag_elp_cam_019"]
CAMERAS = ["zag_elp_cam_016"]
MAX_DISTANCE_SEC = 5


@dataclass
class DayData:
    csv_paths: list[Path]
    start_timestamps: list[datetime]
    end_timestamps: list[datetime]
    frame_timestamps: list[pd.Series]


def get_day_path(camera: str, timestamp: datetime) -> Path:
    return ROOT_FOLDER / camera / timestamp.strftime("%Y-%m-%d")


def read_track_ranges(path: Path) -> DayData:
    data = DayData([], [], [], [])
    for f in sorted(path.glob("*.csv")):
        # Read track details
        df_timestamps = pd.read_csv(
            f,
            usecols=["timestamp"],
            parse_dates=["timestamp"],
        )
        ds_timestamps = df_timestamps["timestamp"]

        data.csv_paths.append(f)
        data.frame_timestamps.append(ds_timestamps)
        data.start_timestamps.append(ds_timestamps.iloc[0])
        data.end_timestamps.append(ds_timestamps.iloc[-1])

    return data


def find_track_images(day_data: DayData, timestamp: datetime) -> list[np.ndarray]:
    images = []
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

        frame_timestamps = day_data.frame_timestamps[i]
        ind = bisect.bisect_right(frame_timestamps.to_list(), timestamp)
        if ind == 0:
            continue
        video_timestamp = frame_timestamps[ind - 1]
        distance = abs((video_timestamp - timestamp).total_seconds())
        if distance > MAX_DISTANCE_SEC:
            continue

        video_frameid = ind - 1

        # Open the actual video and skip to desired frame
        video_path = day_data.csv_paths[i].with_suffix(".mkv")
        video = cv2.VideoCapture(str(video_path))
        video.set(cv2.CAP_PROP_POS_FRAMES, video_frameid)
        ok, image = video.read()
        if not ok:
            logger.error(
                "Error reading video: %s, frame: %d", video_path, video_frameid
            )
            continue
        images.append(image)
    return images
