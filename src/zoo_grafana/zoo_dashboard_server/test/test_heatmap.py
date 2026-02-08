import project_root
from src.zoo_grafana.zoo_dashboard_server.track_heatmap import *
from datetime import datetime
import logging
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    start_timestamp = datetime.fromisoformat("2025-11-15 00:00:00")
    end_timestamp = datetime.fromisoformat("2025-11-16 00:00:00")
    camera_ids = [2]
    identity_ids = []
    bytes = make_map_heatmap(start_timestamp, end_timestamp, camera_ids, identity_ids)

    output_file = Path("test.png")
    output_file.write_bytes(bytes)
