from datetime import datetime, time
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import bisect
import cv2

logger = logging.getLogger(__name__)

ROOT_FOLDER = Path("/data/xmas/tracks")
# CAMERAS = ["zag_elp_cam_016", "zag_elp_cam_017", "zag_elp_cam_018", "zag_elp_cam_019"]
CAMERAS = ["zag_elp_cam_016"]
MAX_DISTANCE_SEC = 5


def get_track_image(timestamp: datetime, day_path: Path):
    if not day_path.exists():
        # Day is not in database, return silently
        return None
    if not day_path.is_dir():
        logger.error("Path should be a directory: %s", day_path)
        return None

    # Get all tracks
    track_paths = [f for f in day_path.glob("*.csv")]
    times = [f.name[1:7] for f in track_paths]

    sorted_indices = np.argsort(times)
    track_paths = list(map(track_paths.__getitem__, sorted_indices))
    times = list(map(times.__getitem__, sorted_indices))

    target_time = timestamp.strftime("%H%M%S")
    ind = bisect.bisect_right(times, target_time)
    if ind == 0:
        return None

    csv_path = track_paths[ind - 1]
    video_path = csv_path.with_suffix(".mkv")

    # Read track details and find the relevant frame
    data = pd.read_csv(
        csv_path,
        usecols=["frame_id", "timestamp"],
        dtype={"frame_id": np.int64},
        parse_dates=["timestamp"],
    )

    ind = bisect.bisect_right(data["timestamp"].to_list(), timestamp)
    if ind == 0:
        return None
    video_timestamp = data["timestamp"][ind - 1]
    distance = abs((video_timestamp - timestamp).total_seconds())
    if distance > MAX_DISTANCE_SEC:
        return None

    video_frameid = data.index[ind - 1]

    # Open the actual video and skip to desired frame
    video = cv2.VideoCapture(str(video_path))
    video.set(cv2.CAP_PROP_POS_FRAMES, video_frameid)
    ok, image = video.read()
    if not ok:
        logger.error("Error reading video: %s, frame: %d", video_path, video_frameid)
    return image


def get_all_track_images(timestamp: datetime):
    images = []
    for camera in CAMERAS:
        image = get_track_image(
            timestamp,
            ROOT_FOLDER / camera / timestamp.strftime("%Y-%m-%d"),
        )
        if image is not None:
            images.append(image)

    return images
