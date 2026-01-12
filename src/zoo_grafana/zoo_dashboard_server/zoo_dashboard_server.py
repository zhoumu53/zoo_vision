"""
This is the webserver that accompanies the zoo_grafana panel plugin.
It serves the track images associated with a timestamp.

To start:
flask --app zoo_dashboard_server run --host 0.0.0.0 --debug

"""

from project_root import PROJECT_ROOT
from track_search import *
from camera_images import find_camera_image
from dataclasses import asdict

import io
import json
from base64 import encodebytes
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_caching import Cache
from datetime import datetime, timedelta
import dateutil

DEFAULT_CAMERA = "zag_elp_cam_016"
DEFAULT_TIMESTAMP = "2025-02-09T20:56:00"

config = {
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300,
}
app = Flask(__name__)
app.config.from_mapping(config)
CORS(app)
cache = Cache(app)


################################################
# Image utils


def compress_jpeg(image: Image.Image) -> io.BytesIO:
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="JPEG")  # convert the PIL image to byte array
    byte_arr.seek(0)
    return byte_arr


def encode_base64(byte_arr: io.BytesIO) -> str:
    encoded_img = "data:image/jpeg;base64," + encodebytes(byte_arr.getvalue()).decode(
        "ascii"
    )
    return encoded_img


################################################
# Track images


@cache.memoize(30 * 60)
def read_track_ranges_cached(path: Path) -> DayData:
    return read_track_ranges(path)


@cache.memoize(30 * 60)
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


def track_images(camera: str, timestamp: datetime) -> list[Detection]:
    config = get_config()
    track_root_path = Path(config["record_root"]) / "tracks"

    # Get images from previous, current, and next days
    # to avoid any time zone issues.
    detections = []
    for offset in [-1, 0, +1]:
        timeoffset = timestamp + timedelta(days=offset)
        day_path = get_day_path(track_root_path, camera, timeoffset)
        day_data = read_track_ranges_cached(day_path)
        detections_i = find_track_images(camera, day_data, timestamp)
        detections.extend(detections_i)

    return detections


@app.route("/track_images", methods=["GET"])
def track_images_get():
    camera = request.args.get("camera", "zag_elp_cam_016")
    timestamp = parse_timestamp(request.args.get("timestamp", DEFAULT_TIMESTAMP))
    detections = track_images(camera, timestamp)

    detections = [asdict(x) for x in detections]

    # Compress and encode images
    for detection in detections:
        detection["image"] = encode_base64(
            compress_jpeg(Image.fromarray(detection["image"]))
        )

    # Return json with all images for this camera
    return jsonify(
        {
            "timestamp": timestamp.strftime("%a, %d %b %Y %H:%M:%S %Z"),
            "timestamp2": timestamp.timestamp(),
            "detections": detections,
        }
    )


@app.route("/test_track_image", methods=["GET"])
def test_track_image_get():
    camera = request.args.get("camera", "zag_elp_cam_016")
    timestamp = parse_timestamp(request.args.get("timestamp", DEFAULT_TIMESTAMP))
    detections = track_images(camera, timestamp)

    if len(detections) == 0:
        return send_file("static/no_image.jpg")

    # Compress and return
    raw_bytes = compress_jpeg(Image.fromarray(detections[0].image))
    return send_file(raw_bytes, mimetype="image/jpg")


################################################
# Camera images


@cache.memoize(30 * 60)
def get_video_db():
    config = get_config()
    video_db_file = Path(config["video_db"])
    with video_db_file.open() as fd:
        video_db = json.load(fd)
    return video_db


def camera_image(camera: str, timestamp: datetime):
    config = get_config()
    if "video_root" in config and config["video_root"] != "":
        video_root = Path(config["video_root"])
    else:
        video_root = Path(config["video_db"]).parent

    image = find_camera_image(get_video_db(), video_root, camera, timestamp)
    if image is None:
        with open("static/no_image.jpg", "rb") as f:
            raw_bytes = io.BytesIO(f.read())
    else:
        raw_bytes = compress_jpeg(Image.fromarray(image))
    return raw_bytes


@app.route("/camera_image", methods=["GET"])
def camera_image_get():
    camera = request.args.get("camera", DEFAULT_CAMERA)
    timestamp = parse_timestamp(request.args.get("timestamp", DEFAULT_TIMESTAMP))
    raw_bytes = camera_image(camera, timestamp)

    return jsonify(
        {
            "timestamp": timestamp.strftime("%a, %d %b %Y %H:%M:%S %Z"),
            "timestamp2": timestamp.timestamp(),
            "image": encode_base64(raw_bytes),
        }
    )


@app.route("/test_camera_image", methods=["GET"])
def test_camera_image_get():
    camera = request.args.get("camera", DEFAULT_CAMERA)
    timestamp = parse_timestamp(request.args.get("timestamp", DEFAULT_TIMESTAMP))
    raw_bytes = camera_image(camera, timestamp)

    return send_file(raw_bytes, mimetype="image/jpg")


def create_app() -> Flask:
    """
    Used to call from command line:
        waitress-serve --port 5000 --call zoo_dashboard_server:create_app
    """
    return app
