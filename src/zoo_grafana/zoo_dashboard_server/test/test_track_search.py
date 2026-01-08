from ..track_search import *
from datetime import datetime
import logging
import numpy as np
import cv2

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = find_all_track_images(datetime.fromisoformat("2025-03-16T00:53:59"))
    print(result)
