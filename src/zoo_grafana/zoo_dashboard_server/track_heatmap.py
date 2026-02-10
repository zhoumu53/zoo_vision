from project_root import PROJECT_ROOT
from track_search import INDIVIDUAL_FROM_ID
from src.zoo_grafana.zoo_dashboard_server.project_config import get_config

from datetime import datetime
import logging
import numpy as np
from scipy.ndimage import gaussian_filter
import psycopg2
import cv2
from io import BytesIO
from typing import Any
import matplotlib.pyplot as plt
import json
from cachetools.func import ttl_cache


@ttl_cache(ttl=30 * 60)
def get_submap():
    SUBMAP_X = 1450
    SUBMAP_Y = 1300
    SUBMAP_WIDTH = 1250
    SUBMAP_HEIGHT = 900
    SUBMAP_SCALE = 0.25

    config = get_config()

    im_map = cv2.imread(str(PROJECT_ROOT / "data/kkep_floorplan.png"))
    im_submap = im_map[
        SUBMAP_Y : (SUBMAP_Y + SUBMAP_HEIGHT),
        SUBMAP_X : (SUBMAP_X + SUBMAP_WIDTH) :,
    ]
    im_submap = cv2.resize(
        im_submap,
        dsize=None,
        fx=SUBMAP_SCALE,
        fy=SUBMAP_SCALE,
        interpolation=cv2.INTER_AREA,
    )
    cv2.cvtColor(im_submap, cv2.COLOR_BGR2RGB, im_submap)

    # Load submap pose
    T_map_from_world2 = np.asarray(config["map"]["T_map_from_world2"])
    T_submap_from_world2 = (
        np.array(
            [
                [SUBMAP_SCALE, 0, -SUBMAP_SCALE * SUBMAP_X],
                [0, SUBMAP_SCALE, -SUBMAP_SCALE * SUBMAP_Y],
                [0, 0, 1],
            ]
        )
        @ T_map_from_world2
    )
    return im_submap, T_submap_from_world2


def _identities_to_title(identity_ids: list[int] | None):
    if identity_ids:
        return ", ".join([INDIVIDUAL_FROM_ID[id] for id in identity_ids])
    else:
        return "All individuals"


def make_map_heatmap(
    start_timestamp: datetime,
    end_timestamp: datetime,
    camera_ids: list[int] | None = None,
    identity_ids: list[int] | None = None,
) -> bytes:
    # Load submap
    im_submap, T_submap_from_world2 = get_submap()
    submap_height = im_submap.shape[0]
    submap_width = im_submap.shape[1]

    # Load detections
    sql = """
WITH cte AS (
  SELECT 
      t.camera_id
    , t.identity_id
    , date_bin('1 second', o.time, TIMESTAMP %(start_timestamp)s) AS time_bined
    , count(*) as observation_count
    , avg(location[0]) as location_x
    , avg(location[1]) as location_y
  FROM observations AS o
  INNER JOIN tracks AS t ON t.id=o.track_id 
  INNER JOIN identities AS i on t.identity_id=i.id
  WHERE  o.time BETWEEN TIMESTAMP %(start_timestamp)s AND TIMESTAMP %(end_timestamp)s
        and EXTRACT(EPOCH from t.end_time-t.start_time) > 1
  GROUP  BY t.id, t.camera_id, t.identity_id, time_bined
)
SELECT location_x, location_y FROM cte
WHERE cte.observation_count > 1
"""
    sql_args: dict[str, Any] = {
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
    }
    if camera_ids:
        sql += " and cte.camera_id = ANY(%(camera_ids)s)"
        sql_args["camera_ids"] = camera_ids
    if identity_ids:
        sql += " and cte.identity_id = ANY(%(identity_ids)s)"
        sql_args["identity_ids"] = identity_ids
    with psycopg2.connect(
        "dbname=zoo_vision user=grafanareader password=asdf"
    ) as db_connection:
        with db_connection.cursor() as db_cursor:
            db_cursor.execute(
                sql,
                sql_args,
            )
            results = db_cursor.fetchall()

    # Create heat map
    im_heat = np.zeros((submap_height, submap_width), dtype=np.float32)
    for location_world_x, location_world_y in results:
        location_world_h = np.asarray(
            [location_world_x, location_world_y, 1], dtype=np.float32
        )
        location_map_h = T_submap_from_world2 @ location_world_h
        location_map = location_map_h[0:2].astype(np.int32)
        u, v = location_map[0:2]

        if u < 0 or u >= submap_width:
            continue
        if v < 0 or v >= submap_height:
            continue
        im_heat[v, u] += 1

    # Use log() so the static poses don't dominate the heatmap
    im_heat_log = np.log(im_heat + 0.0000001)
    im_heat_log = gaussian_filter(im_heat_log, 2.5, mode="nearest")

    # Add alpha but with sqrt() so we still get contrast for low values
    alpha = np.full(im_heat.shape, 0.5)
    alpha = im_heat_log
    alpha -= alpha.min()
    alpha_max = alpha.max()
    if alpha_max == 0:
        alpha = 0
    else:
        alpha *= 1 / alpha.max()
    alpha = np.sqrt(alpha)

    ax: plt.Axes
    fig, ax = plt.subplots(1, 1)
    ax.imshow(im_submap)
    ax.imshow(im_heat_log, alpha=alpha, cmap="jet")
    ax.set_axis_off()
    ax.text(
        5.5,
        5.5,
        _identities_to_title(identity_ids),
        va="top",
        color="white",
        backgroundcolor="gray",
    )

    stream = BytesIO()
    fig.savefig(stream, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return stream.getvalue()
