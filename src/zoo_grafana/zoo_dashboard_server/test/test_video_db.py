import project_root
from src.zoo_grafana.zoo_dashboard_server.video_db import *
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    load_and_update_db()
