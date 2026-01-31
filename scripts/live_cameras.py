import cv2
import json
from pathlib import Path
from matplotlib import pyplot as plt

with Path("/home/dherrera/git/zoo_vision/data/config.json").open() as f:
    config = json.load(f)

images = {}
for name,camera in config["cameras"].items():
    stream = camera["stream"]
    protocol = stream["protocol"]
    ip = stream["ip"]
    url = stream["url"]
    user = "daniel"
    pwd = "PwU82-!MnG"
    url = f"{protocol}://{user}:{pwd}@{ip}/{url}"
    video = cv2.VideoCapture(url)
    ok,im = video.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    images[name] = im

for name,im in images.items():
    cv2.imwrite(f"/media/ElephantsWD/elephants/test_dan/results/{name}.png",im)
