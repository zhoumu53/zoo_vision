"""
This is the webserver that accompanies the zoo_grafana panel plugin.
It serves the track images associated with a timestamp.

To start:
flask --app zoo_dashboard_server run --host 0.0.0.0 --debug

"""

from project_root import PROJECT_ROOT
from track_search import *
from camera_images import find_camera_image
from track_heatmap import make_map_heatmap
from dataclasses import asdict

import io
import json
import base64
from PIL import Image
from quart import Quart, request, send_file, Response
from quart_cors import cors
from quart_schema import validate_request, validate_querystring, QuartSchema
from cachetools.func import ttl_cache
from datetime import datetime, timedelta
import dateutil
import asyncio

DEFAULT_CAMERA = "zag_elp_cam_016"
DEFAULT_TIMESTAMP = "2025-02-09T20:56:00"
DEFAULT_END_TIMESTAMP = "2025-02-10T20:56:00"

JPEG_PREFIX = "data:image/jpeg;base64,"

config = {
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300,
}
app = Quart(__name__)
app.config.from_mapping(config)
cors(app)
QuartSchema(app)

################################################
# Image utils


def compress_jpeg(image: Image.Image) -> io.BytesIO:
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="JPEG")  # convert the PIL image to byte array
    byte_arr.seek(0)
    return byte_arr


def encode_base64(byte_arr: io.BytesIO) -> str:
    encoded_img = JPEG_PREFIX + base64.encodebytes(byte_arr.getvalue()).decode("ascii")
    return encoded_img


################################################
# Track images


@ttl_cache(maxsize=128, ttl=30 * 60)
def read_track_ranges_cached(path: Path) -> DayData:
    return read_track_ranges(path)


@ttl_cache(maxsize=1, ttl=30 * 60)
def get_config():
    config_path = PROJECT_ROOT / "data" / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    return config


def parse_timestamp(timestamp_str: str) -> datetime:
    # Server is en zurich so we want all dates in this timezone
    tz = pytz.timezone("Europe/Zurich")

    if ":" in timestamp_str:
        all_tzinfos = {x: pytz.timezone(x) for x in pytz.all_timezones}
        timestamp = dateutil.parser.parse(timestamp_str, tzinfos=all_tzinfos)

        if timestamp.tzinfo is None:
            # Assume date is in swiss timezone if no timezone is given
            timestamp = tz.localize(timestamp, is_dst=False)
        else:
            # Convert to zurich timezone
            timestamp = timestamp.astimezone(tz)
    else:
        timestamp_s = float(timestamp_str)
        timestamp = datetime.fromtimestamp(timestamp_s)
        # Assume a numeric date is in utc
        timestamp = pytz.utc.localize(timestamp, is_dst=False)
        # Convert to zurich timezone
        timestamp = timestamp.astimezone(tz)

    return timestamp


async def track_images(camera: str, timestamp: datetime) -> list[Detection]:
    config = get_config()
    track_root_path = Path(config["record_root"]) / "tracks"

    # Get images from previous, current, and next days
    # to avoid any time zone issues.
    promises = []
    for offset in [-1, 0, +1]:
        timeoffset = timestamp + timedelta(days=offset)
        day_path = get_day_path(track_root_path, camera, timeoffset)
        day_data = read_track_ranges_cached(day_path)
        promises.append(find_track_images(camera, day_data, timestamp))
    # Await on all days simultaneously
    detections = await asyncio.gather(*promises)
    detections = sum(detections, [])

    return detections


async def stream_or_cancel_json(task: asyncio.Task[str]):
    try:
        while True:
            try:
                done, _ = await asyncio.wait([task], timeout=0.1)
                if task in done:
                    try:
                        content = await task
                    except Exception as e:
                        # TODO: this should return a 505 Internal Error response but we cannot change it now
                        content = app.json.dumps({"error": str(e)})
                    yield content
                    break
                else:
                    yield " "  # Output something so we test the connection
                    pass
            except asyncio.exceptions.CancelledError:
                # This usually never happens because
                # when we cancel in GeneratorExit we never get called again.
                yield "cancelled"
                break
    except GeneratorExit:
        # Flask raises this exception when a client disconnects and we try to write to the stream
        task.cancel()


@dataclass
class TrackImagesParams:
    camera: str = DEFAULT_CAMERA
    timestamp: str = DEFAULT_TIMESTAMP


@app.route("/track_images", methods=["GET"])
@validate_querystring(TrackImagesParams)
async def track_images_get(query_args: TrackImagesParams):
    camera = query_args.camera
    timestamp = parse_timestamp(query_args.timestamp)

    async def impl():
        detections = await track_images(camera, timestamp)

        detections = [asdict(x) for x in detections]

        # Compress and encode images
        for detection in detections:
            detection["image"] = encode_base64(
                compress_jpeg(Image.fromarray(detection["image"]))
            )
        return app.json.dumps(
            {
                "timestamp": timestamp.strftime("%a, %d %b %Y %H:%M:%S %Z"),
                "timestamp2": timestamp.timestamp(),
                "detections": detections,
            }
        )

    task = asyncio.create_task(impl())

    # Return json with all images for this camera
    return Response(stream_or_cancel_json(task), mimetype="application/json")


@app.route("/test_track_image", methods=["GET"])
@validate_querystring(TrackImagesParams)
async def test_track_image_get(query_args: TrackImagesParams):
    camera = query_args.camera
    timestamp = parse_timestamp(query_args.timestamp)
    detections = await track_images(camera, timestamp)

    if len(detections) == 0:
        return await send_file("static/no_image.jpg")

    # Compress and return
    raw_bytes = compress_jpeg(Image.fromarray(detections[0].image))
    return await send_file(raw_bytes, mimetype="image/jpg")


################################################
# Camera images


@ttl_cache(ttl=30 * 60)
def get_video_db():
    config = get_config()
    video_db_file = Path(config["video_db"])
    with video_db_file.open() as fd:
        video_db = json.load(fd)
    return video_db


async def camera_image(camera: str, timestamp: datetime):
    config = get_config()
    if "video_root" in config and config["video_root"] != "":
        video_root = Path(config["video_root"])
    else:
        video_root = Path(config["video_db"]).parent

    image = await find_camera_image(get_video_db(), video_root, camera, timestamp)
    if image is None:
        with open("static/no_image.jpg", "rb") as f:
            raw_bytes = io.BytesIO(f.read())
    else:
        raw_bytes = compress_jpeg(Image.fromarray(image))

    return app.json.dumps(
        {
            "timestamp": timestamp.strftime("%a, %d %b %Y %H:%M:%S %Z"),
            "timestamp2": timestamp.timestamp(),
            "image": encode_base64(raw_bytes),
        }
    )


@dataclass
class CameraImageParams:
    camera: str = DEFAULT_CAMERA
    timestamp: str = DEFAULT_TIMESTAMP


@app.route("/camera_image", methods=["GET"])
@validate_querystring(CameraImageParams)
async def camera_image_get(query_args: CameraImageParams):
    camera = query_args.camera
    timestamp = parse_timestamp(query_args.timestamp)

    task = asyncio.create_task(camera_image(camera, timestamp))

    # Return json with all images for this camera
    return Response(stream_or_cancel_json(task), mimetype="application/json")


@app.route("/test_camera_image", methods=["GET"])
@validate_querystring(CameraImageParams)
async def test_camera_image_get(query_args: CameraImageParams):
    camera = query_args.camera
    timestamp = parse_timestamp(query_args.timestamp)
    json_str = await camera_image(camera, timestamp)

    json_data = json.loads(json_str)
    image_base64 = json_data["image"][len(JPEG_PREFIX) :]
    image_jpg = base64.decodebytes(image_base64.encode("ascii"))
    return await send_file(io.BytesIO(image_jpg), mimetype="image/jpg")


def create_app(*args) -> Quart:
    """
    Used to call from command line:
        waitress-serve --port 5000 --call zoo_dashboard_server:create_app
    """
    return app


################################################
# World heatmap images


@dataclass
class HeatmapParams:
    start_timestamp: str = DEFAULT_TIMESTAMP
    end_timestamp: str = DEFAULT_END_TIMESTAMP
    identity_id: int | None = None


@app.route("/heatmaps/world", methods=["GET"])
@validate_querystring(HeatmapParams)
async def heamap_world_get(query_args: HeatmapParams):
    start_timestamp = parse_timestamp(query_args.start_timestamp)
    end_timestamp = parse_timestamp(query_args.end_timestamp)

    bytes = make_map_heatmap(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        identity_ids=[query_args.identity_id] if query_args.identity_id else None,
    )

    return await send_file(io.BytesIO(bytes), mimetype="image/png")


def create_app(*args) -> Quart:
    """
    Used to call from command line:
        waitress-serve --port 5000 --call zoo_dashboard_server:create_app
    """
    return app
