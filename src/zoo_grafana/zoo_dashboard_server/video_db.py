from project_root import PROJECT_ROOT
from src.zoo_grafana.zoo_dashboard_server.project_config import get_config


import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import re
import cv2
import logging
import json
from cachetools.func import ttl_cache

logger = logging.getLogger(__name__)

CAMERA_NAMES = [
    "zag_elp_cam_016",
    "zag_elp_cam_017",
    "zag_elp_cam_018",
    "zag_elp_cam_019",
]

INVALID_DATE = datetime(year=1900, month=1, day=1)
VIDEO_DB_FILENAME = PROJECT_ROOT / "data/all_nas_videos.json"


class ParsedVideoInfo(BaseModel):
    filename: Path
    camera: str
    start_time: datetime
    end_time: datetime
    fps: float


class CameraVideos(BaseModel):
    filenames: list[Path] = Field(default_factory=list)
    start_times: list[datetime] = Field(default_factory=list)
    end_times: list[datetime] = Field(default_factory=list)


class VideoDB(BaseModel):
    last_update_time: datetime = INVALID_DATE

    first_timestamp: datetime = INVALID_DATE
    last_timestamp: datetime = INVALID_DATE
    cameras: dict[str, CameraVideos] = Field(default_factory=dict)


def load_db() -> VideoDB:
    if not VIDEO_DB_FILENAME.exists():
        return VideoDB(cameras={name: CameraVideos() for name in CAMERA_NAMES})
    data = VIDEO_DB_FILENAME.read_text()
    return VideoDB.model_validate_json(data)


def save_db(db: VideoDB):
    with VIDEO_DB_FILENAME.open("w") as f:
        f.write(db.model_dump_json())


def get_video_duration(filename: Path) -> tuple[float, float]:
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,r_frame_rate",
    ]
    res = subprocess.check_output(ffprobe_cmd + [str(filename)])
    output_txt = res.decode()
    output_json = json.loads(output_txt)

    stream_info = output_json["streams"][0]
    r_frame_rate = stream_info["r_frame_rate"]
    rates = r_frame_rate.split("/")
    fps = float(rates[0]) / float(rates[1])

    duration_s = float(stream_info["duration"])
    return fps, duration_s


def parse_video_name(filename: Path) -> ParsedVideoInfo:
    name = str(filename.name).lower()

    # Name is in of these shapes:
    # {camera}-{dd.mm.yyyy}-{time_start}-{time_end}.mp4
    # {camera}-{yyyymmdd}-{time_start}.mp4

    # Parse {camera}
    name_underscore = name.replace("-", "_")
    camera_names = [name for name in CAMERA_NAMES if name_underscore.startswith(name)]
    if len(camera_names) != 1:
        raise RuntimeError(
            f"Filename {str(filename)} could not be parsed: camera prefix error"
        )
    camera_name = camera_names[0]

    # Remove camera prefix
    name = name[len(camera_name) + 1 :]

    # Parse {date}
    if name[2] == ".":
        # Name is in format {dd.mm.yyyy}-{hhmmss}-{hhmmss}.mp4
        # Name is in format {dd.mm.yyyy}-{hhmmss}-{hhmmss}.mp4
        match = re.search(
            "(?P<day>\d{2}).(?P<month>\d{2}).(?P<year>\d{4})-(?P<hour0>\d{2})(?P<min0>\d{2})(?P<sec0>\d{2})-(?P<hour1>\d{2})(?P<min1>\d{2})(?P<sec1>\d{2})",
            name,
        )
        if match is None:
            raise RuntimeError(f"Could not parse {filename}")
    else:
        # Name is in format {yyyymmdd}-{hhmmss}.mp4
        match = re.search(
            r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})-(?P<hour0>\d{2})(?P<min0>\d{2})(?P<sec0>\d{2})",
            name,
        )
        if match is None:
            raise RuntimeError(f"Could not parse {filename}")
    fields = {k: int(v) for k, v in match.groupdict().items()}

    start_time = datetime(
        *[fields[k] for k in ["year", "month", "day", "hour0", "min0", "sec0"]]
    )

    # Open the video to check the frame count
    fps, video_length_s = get_video_duration(filename)

    # No end time in filename. Assume fps and calculate end time
    # end_time = start_time + datetime.timedelta(seconds=frame_count / fps)
    end_time = start_time + timedelta(seconds=video_length_s)
    print(f"Video={filename}, length={end_time - start_time}")
    return ParsedVideoInfo(
        filename=filename,
        camera=camera_name,
        start_time=start_time,
        end_time=end_time,
        fps=fps,
    )


def get_video_root():
    config = get_config()
    if config["video_root"] == "":
        return Path(config["video_db"]).parent
    else:
        return Path(config["video_root"])


def update_dir(db: VideoDB, dir: Path):
    for path in dir.iterdir():
        if path.name[0] in ["@", "."]:
            # Skip special dirs
            continue

        if path.is_dir():
            update_dir(db, path)
        elif path.suffix == ".mp4":
            update_file(db, path)


def update_file(db: VideoDB, path: Path):
    found = False
    for camera_data in db.cameras.values():
        if path in camera_data.filenames:
            found = True
            break
    if found:
        # Already in db, no need to parse again
        return

    # Need to update!
    try:
        info = parse_video_name(path)
    except Exception as e:
        logger.error(f"Error parsing {str(path)}, error: {e}")
        # Create dummy entry so we don't keep trying in the future
        info = ParsedVideoInfo(
            filename=path,
            camera="zag_elp_cam_016",
            start_time=datetime(1970, 1, 1),
            end_time=datetime(1970, 1, 1),
            fps=0,
        )
    camera_data = db.cameras[info.camera]
    camera_data.filenames.append(path)
    camera_data.start_times.append(info.start_time)
    camera_data.end_times.append(info.end_time)


def update_db(db: VideoDB):
    video_root = get_video_root()
    logging.info(f"Listing top dirs of {str(video_root)}...")
    dirs = [dir for dir in video_root.iterdir() if dir.is_dir()]

    # Remove the last video from each camera because it might have been incomplete the last time we updated
    for camera_data in db.cameras.values():
        if len(camera_data.filenames) > 0:
            del camera_data.filenames[-1]
            del camera_data.start_times[-1]
            del camera_data.end_times[-1]

    # Remove files that don't exist
    logging.info(f"Removing non-existent files...")
    removed_count = 0
    for camera_data in db.cameras.values():
        for i in range(len(camera_data.filenames) - 1, -1, -1):
            if not camera_data.filenames[i].exists():
                removed_count += 1
                del camera_data.filenames[i]
                del camera_data.start_times[i]
                del camera_data.end_times[i]
    if removed_count > 0:
        logger.warning(
            f"Removed {removed_count} videos from db that are no longer on disk"
        )

    # Normalize names
    dir_dict = {str(name.name).replace("-", "_").lower(): name for name in dirs}

    # Search for new files
    # Two options: either it is the NAS and the first level is the camera names, or we need to to a glob
    if all([name in dir_dict for name in CAMERA_NAMES]):
        logger.info(
            "Found camera names in top directory, assuming this is the NAS and skipping other dirs"
        )
        for name, camera_data in db.cameras.items():
            if not name in dir_dict:
                logger.error(f"Camera {name} not found in {str(video_root)}")
                continue

            update_dir(db, dir_dict[name])
    else:
        logger.info(
            f"Did not find camera names in top directory {str(video_root)}, globbing for mp4s"
        )
        for file in video_root.glob("**/*.mp4"):
            update_file(db, file)

    cameras_with_files = [d for d in db.cameras.values() if len(d.filenames) > 0]

    # Sort lists
    for camera_data in cameras_with_files:
        start_times, end_times, filenames = zip(
            *sorted(
                zip(
                    camera_data.start_times,
                    camera_data.end_times,
                    camera_data.filenames,
                )
            )
        )
        camera_data.filenames = list(filenames)
        camera_data.start_times = list(start_times)
        camera_data.end_times = list(end_times)

    # Update start and end times
    if cameras_with_files:
        db.first_timestamp = min([d.start_times[0] for d in cameras_with_files])
        db.last_timestamp = max([d.end_times[-1] for d in cameras_with_files])
    else:
        db.first_timestamp = INVALID_DATE
        db.last_timestamp = INVALID_DATE
    db.last_update_time = datetime.now()

    logger.info(
        f"Video db updated at {datetime.now()}, total videos: "
        + ", ".join([f"{name}={len(d.filenames)}" for name, d in db.cameras.items()])
    )


@ttl_cache(ttl=30 * 60)
def load_cached_db() -> VideoDB:
    logger.info("Loading video db...")

    start_time = time.perf_counter()
    db = load_db()
    duration_s = time.perf_counter() - start_time
    logger.info(f"Load video db: {duration_s} sec")

    return db


def load_and_update_db() -> VideoDB:
    logger.info("Updating video db video db...")

    start_time = time.perf_counter()
    db = load_db()
    duration_s = time.perf_counter() - start_time
    logger.info(f"Load video db: {duration_s} sec")

    start_time = time.perf_counter()
    update_db(db)
    duration_s = time.perf_counter() - start_time
    logger.info(f"Update video db: {duration_s} sec")

    save_db(db)

    return db


def video_db_update_daemon() -> None:

    UPDATE_PERIOD_SEC = 30 * 60
    logger.info(f"Video db update thread active, period={UPDATE_PERIOD_SEC // 60} min")

    while True:
        load_and_update_db()

        # Sleep until we need to update again
        time.sleep(UPDATE_PERIOD_SEC)
