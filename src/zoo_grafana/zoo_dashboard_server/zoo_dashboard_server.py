"""
This is the webserver that accompanies the zoo_grafana panel plugin.
It serves the track images associated with a timestamp.

To start:
flask --app zoo_dashboard_server run --host 0.0.0.0 --debug

"""

import io
from base64 import encodebytes
from PIL import Image
from flask import Flask, request, Response, send_file
from .track_search import find_track_image
from datetime import datetime

app = Flask(__name__)


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode="r")  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format="PNG")  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode("ascii")  # encode as base64
    return encoded_img


@app.route("/find_images", methods=["GET"])
def find_images():
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
    image = find_track_image(camera, timestamp)
    if image is None:
        return send_file("static/no_image.jpg")

    # Compress and return
    image_pil = Image.fromarray(image)
    raw_bytes = io.BytesIO()
    image_pil.save(raw_bytes, "JPEG")
    raw_bytes.seek(0)
    return send_file(raw_bytes, mimetype="image/jpg")
