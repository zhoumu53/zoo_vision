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
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_caching import Cache
from datetime import datetime, timedelta

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


@app.route("/track_images", methods=["GET"])
def track_images():
    # Parse args
    camera = request.args.get("camera", "zag_elp_cam_016")
    timestamp_str = request.args.get("timestamp", "2025-03-16T00:53:59")
    print(timestamp_str)
    if ":" in timestamp_str:
        timestamp = datetime.fromisoformat(timestamp_str)
    else:
        timestamp_num = float(timestamp_str) / 1000
        print(timestamp_num)
        timestamp = datetime.fromtimestamp(timestamp_num)

    # Get image
    images = []
    for offset in [-1, 0]:
        timeoffset = timestamp + timedelta(days=offset)
        day_path = get_day_path(camera, timeoffset)
        day_data = read_track_ranges_cached(day_path)
        images_i = find_track_images(day_data, timestamp)
        images.extend(images_i)

    # Compress and encode
    images_base64 = [compress_and_encode(Image.fromarray(image)) for image in images]
    # Return json with all images for this camera
    return jsonify({"result": images_base64})
