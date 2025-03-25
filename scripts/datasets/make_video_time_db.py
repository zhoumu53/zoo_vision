import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
from enlighten import Counter as ECounter
import torch
import argparse
import datetime
from dataclasses import dataclass
import re

from project_root import PROJECT_ROOT

# We need to know the camera names in advance to remove them from the video file name
CAMERA_NAMES = [
    "zag_elp_cam_016",
    "zag_elp_cam_017",
    "zag_elp_cam_018",
    "zag_elp_cam_019",
]

CAMERA_PREFIXES = {
    "ELP - Sandbox_mit_Buero-": "zag_elp_cam_017",
    "ELP - Sandbox_mit_MGMT-": "zag_elp_cam_018",
    "ELP - Sandbox_ohne_Büro-": "zag_elp_cam_016",
    "ELP - Sandbox_ohne_MGMT-": "zag_elp_cam_019",
    "zag_elp_cam_016-": "zag_elp_cam_016",
    "zag_elp_cam_017-": "zag_elp_cam_017",
    "zag_elp_cam_018-": "zag_elp_cam_018",
    "zag_elp_cam_019-": "zag_elp_cam_019",
}
CAMERA_PREFIXES = {k.lower().replace("-", "_"): v for k, v in CAMERA_PREFIXES.items()}


@dataclass
class VideoInfo:
    filename: Path
    camera: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    fps: float


def parse_video_name(filename: Path) -> VideoInfo:
    original_filename = filename
    filename = str(filename.name).lower()

    # Name is in of these shapes:
    # {camera}-{dd.mm.yyyy}-{time_start}-{time_end}.mp4
    # {camera}-{yyyymmdd}-{time_start}.mp4

    # Parse {camera}
    filename_underscore = filename.replace("-", "_")
    camera_names = [
        CAMERA_PREFIXES[prefix]
        for prefix in CAMERA_PREFIXES.keys()
        if filename_underscore.startswith(prefix)
    ]
    if len(camera_names) != 1:
        raise RuntimeError(
            f"Filename {str(filename)} could not be parsed: camera prefix error"
        )
    camera_name = camera_names[0]

    # Remove camera prefix
    filename = filename[len(camera_name) + 1 :]

    # Parse {date}
    if filename[2] == ".":
        # Name is in format {dd.mm.yyyy}-{hhmmss}-{hhmmss}.mp4
        match = re.search(
            "(?P<day>\d{2}).(?P<month>\d{2}).(?P<year>\d{4})-(?P<hour0>\d{2})(?P<min0>\d{2})(?P<sec0>\d{2})-(?P<hour1>\d{2})(?P<min1>\d{2})(?P<sec1>\d{2})",
            filename,
        )
        if match is None:
            raise RuntimeError(f"Could not parse {filename}")
    else:
        # Name is in format {yyyymmdd}-{hhmmss}.mp4
        match = re.search(
            "(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})-(?P<hour0>\d{2})(?P<min0>\d{2})(?P<sec0>\d{2})",
            filename,
        )
        if match is None:
            raise RuntimeError(f"Could not parse {filename}")
    fields = {k: int(v) for k, v in match.groupdict().items()}

    start_time = datetime.datetime(
        *[fields[k] for k in ["year", "month", "day", "hour0", "min0", "sec0"]]
    )

    # Open the video to check the frame count
    video = cv2.VideoCapture(original_filename)
    assert video.isOpened()
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Find end
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 2)
    video.read()  # Read dummy frame to update props
    video_length_s = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
    if video_length_s == 0:
        fps = 25
        video_length_s = frame_count / fps
    else:
        fps = frame_count / video_length_s
    video = None

    if False and "hour1" in fields:
        # Filename has end time, calculate fps
        end_day = fields["day"]
        if fields["hour0"] == 23 and fields["hour1"] < 23:
            end_day += 1
        end_time = datetime.datetime(
            year=fields["year"],
            month=fields["month"],
            day=end_day,
            hour=fields["hour1"],
            minute=fields["min1"],
            second=fields["sec1"],
        )
        fps = frame_count / (end_time - start_time).total_seconds()
    else:
        # No end time in filename. Assume fps and calculate end time
        # end_time = start_time + datetime.timedelta(seconds=frame_count / fps)
        end_time = start_time + datetime.timedelta(seconds=video_length_s)
    print(f"Video={original_filename.name}, length={end_time - start_time}")
    return VideoInfo(original_filename, camera_name, start_time, end_time, fps)


def parse_video_files(files: list[Path]) -> dict[str, list[VideoInfo]]:
    video_info_by_camera = {}
    for camera in CAMERA_NAMES:
        video_info_by_camera[camera] = []

    pbar = ECounter(total=len(files), desc="Parsing videos", unit="file")
    for video_file in pbar(files):
        pbar.desc = str(video_file)
        info = parse_video_name(video_file)
        video_info_by_camera[info.camera].append(info)

    return video_info_by_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        "-i",
        type=Path,
        default="/home/dherrera/data/elephants/identity/videos/src/identity_days",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path: Path = args.input_path

    # Parse all video files
    files = list(input_path.glob("**/*.mp4"))
    print(f"Found {len(files)} videos")
    video_info_by_camera = parse_video_files(files)

    # Find min/max
    start_min = None
    start_max = None
    for camera in CAMERA_NAMES:
        for info in video_info_by_camera[camera]:
            if start_min is None or start_min > info.start_time:
                start_min = info.start_time

            last_start = info.end_time - datetime.timedelta(hours=1)
            if start_max is None or start_max < last_start:
                start_max = last_start

    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
    cameras_db = {}
    for camera in CAMERA_NAMES:
        infos = sorted(video_info_by_camera[camera], key=lambda x: x.start_time)
        cameras_db[camera] = {
            "videos": [str(info.filename.relative_to(input_path)) for info in infos],
            "start_times": [info.start_time.strftime(DATE_FORMAT) for info in infos],
            "end_times": [info.end_time.strftime(DATE_FORMAT) for info in infos],
        }

    video_db = {"start_time": start_min.strftime(DATE_FORMAT), "cameras": cameras_db}

    with (input_path / "video_db.json").open("w") as f:
        json.dump(video_db, f, indent=1)


if __name__ == "__main__":
    main()
