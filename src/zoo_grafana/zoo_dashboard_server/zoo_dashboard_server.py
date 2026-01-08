"""
This is the webserver that accompanies the zoo_grafana panel plugin.
It serves the track images associated with a timestamp.

To start:
flask --app zoo_dashboard_server run --host 0.0.0.0 --debug

"""

from .track_search import *

import io
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


def parse_timestamp() -> datetime:
    timestamp_str = request.args.get("timestamp", "2025-03-16 00:53:59")
    if ":" in timestamp_str:
        timestamp = dateutil.parser.parse(timestamp_str)

        ##### THIS IS NOT USED. Timezones are a mess ####
        # all_tzinfos = {x: pytz.timezone(x) for x in pytz.all_timezones}
        # timestamp = dateutil.parser.parse(timestamp_str, tzinfos=all_tzinfos)

        # Assume date is in swiss timezone if no timezone is given
        # if timestamp.tzinfo is None:
        #     tz = pytz.timezone("Europe/Zurich")
        #     timestamp = tz.localize(timestamp, is_dst=False)

        # Drop the timezone offset by moving to utc
        # We want to use all timestamps as utc
        # timestamp = timestamp.astimezone(pytz.utc)
        #########
    else:
        timestamp_s = float(timestamp_str)
        timestamp = datetime.fromtimestamp(timestamp_s)
    return timestamp


def track_images(camera: str, timestamp: datetime):

    # Get images from previous, current, and next days
    # to avoid any time zone issues.
    images = []
    for offset in [-1, 0, +1]:
        timeoffset = timestamp + timedelta(days=offset)
        day_path = get_day_path(camera, timeoffset)
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
            "timestamp": timestamp,
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
