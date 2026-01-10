"""
This is the webserver that accompanies the zoo_grafana panel plugin.
It serves the track images associated with a timestamp.

To start:
flask --app zoo_dashboard_server run --host 0.0.0.0 --debug

"""

from project_root import PROJECT_ROOT
from track_search import *

import io
import json
from base64 import encodebytes
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_caching import Cache
from datetime import datetime, timedelta
import dateutil

config = {
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300,
}
app = Flask(__name__)
app.config.from_mapping(config)
CORS(app)
cache = Cache(app)


def compress_and_encode(image: Image.Image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="JPEG")  # convert the PIL image to byte array
    byte_arr.seek(0)
    encoded_img = encodebytes(byte_arr.getvalue()).decode("ascii")  # encode as base64
    return encoded_img


@cache.memoize(30 * 60)
def read_track_ranges_cached(path: Path) -> DayData:
    return read_track_ranges(path)


@cache.memoize(30 * 60)
def get_config():
    config_path = PROJECT_ROOT / "data" / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    return config


def parse_timestamp() -> datetime:
    # Server is en zurich so we want all dates in this timezone
    tz = pytz.timezone("Europe/Zurich")

    timestamp_str = request.args.get("timestamp", "2025-02-09T20:56:00")
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


def track_images(camera: str, timestamp: datetime):
    config = get_config()
    track_root_path = Path(config["record_root"]) / "tracks"

    # Get images from previous, current, and next days
    # to avoid any time zone issues.
    images = []
    for offset in [-1, 0, +1]:
        timeoffset = timestamp + timedelta(days=offset)
        day_path = get_day_path(track_root_path, camera, timeoffset)
        day_data = read_track_ranges_cached(day_path)
        images_i = find_track_images(day_data, timestamp)
        images.extend(images_i)

    return images


@app.route("/track_images", methods=["GET"])
def track_images_get():
    camera = request.args.get("camera", "zag_elp_cam_016")
    timestamp = parse_timestamp()
    images = track_images(camera, timestamp)

    # Compress and encode
    images_base64 = [compress_and_encode(Image.fromarray(image)) for image in images]
    # Return json with all images for this camera
    return jsonify(
        {
            "timestamp": timestamp.strftime("%a, %d %b %Y %H:%M:%S %Z"),
            "timestamp2": timestamp.timestamp(),
            "images": images_base64,
        }
    )


@app.route("/test_track_image", methods=["GET"])
def test_track_image_get():
    camera = request.args.get("camera", "zag_elp_cam_016")
    timestamp = parse_timestamp()
    images = track_images(camera, timestamp)

    if len(images) == 0:
        return send_file("static/no_image.jpg")

    # Compress and return
    image_pil = Image.fromarray(images[0])
    raw_bytes = io.BytesIO()
    image_pil.save(raw_bytes, "JPEG")
    raw_bytes.seek(0)
    return send_file(raw_bytes, mimetype="image/jpg")


def create_app() -> Flask:
    """
    Used to call from command line:
        waitress-serve --port 5000 --call zoo_dashboard_server:create_app
    """
    return app
