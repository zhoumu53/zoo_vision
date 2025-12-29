"""
This is the webserver that accompanies the zoo_grafana panel plugin.
It serves the track images associated with a timestamp.

To start:
flask --app zoo_dashboard_server run --host 0.0.0.0 --debug

"""

import io
from base64 import encodebytes
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from .track_search import find_track_image
from datetime import datetime

app = Flask(__name__)
CORS(app)


def compress_and_encode(image: Image.Image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="JPEG")  # convert the PIL image to byte array
    byte_arr.seek(0)
    encoded_img = encodebytes(byte_arr.getvalue()).decode("ascii")  # encode as base64
    return encoded_img


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
    image = find_track_image(camera, timestamp)
    if image is not None:
        images.append(image)

    # Compress and encode
    images_base64 = [compress_and_encode(Image.fromarray(image)) for image in images]
    # Return json with all images for this camera
    return jsonify({"result": images_base64})
