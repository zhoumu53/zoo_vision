from video_db import load_and_update_db, VideoDB

from datetime import datetime
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from dataclasses import dataclass
import cv2
import pytz

logger = logging.getLogger(__name__)


def parse_timestamp(timestamp_str: datetime) -> datetime:
    # Assume date is in swiss timezone if no timezone is given
    tz = pytz.timezone("Europe/Zurich")
    timestamp = tz.localize(timestamp_str, is_dst=False)
    return timestamp


async def find_camera_image(camera: str, timestamp: datetime) -> np.ndarray | None:
    camera_db = load_and_update_db().cameras[camera]
    for video_path, start_time_str, end_time_str in zip(
        camera_db.filenames, camera_db.start_times, camera_db.end_times
    ):
        start_time = parse_timestamp(start_time_str)
        if start_time > timestamp:
            # Videos are sorted, we can skip all the rest
            return None

        end_time = parse_timestamp(end_time_str)
        if end_time < timestamp:
            continue

        # Found a match
        # Calculate position in video based on duration
        video_length_s = (end_time - start_time).total_seconds()
        position_s = (timestamp - start_time).total_seconds()
        position_ratio = position_s / video_length_s

        # Load video
        cvCapture = cv2.VideoCapture()
        if not cvCapture.open(video_path):
            logger.warning(f"Could not load video from disk: {video_path}")
            return None
        total_frames = cvCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        position_frames = position_ratio * total_frames
        cvCapture.set(cv2.CAP_PROP_POS_FRAMES, position_frames)

        ok, img = cvCapture.read()
        if not ok:
            logger.warning(
                f"Could not read frame #{position_frames} from video: {video_path}"
            )
            return None

        # Convert to rgb
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    return None
