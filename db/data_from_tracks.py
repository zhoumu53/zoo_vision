"""Fills the db with dummy data by reading the tracks folder"""

from pathlib import Path
from datetime import datetime
import enlighten
import random
import pandas as pd
import psycopg2

BEHAVIOUR_STAND_ID = 1
BEHAVIOUR_SLEEP_LEFT_ID = 2
BEHAVIOUR_SLEEP_RIGHT_ID = 3

SOCIAL_GROUPS = ["IC", "FP", "T"]
INDIVIDUALS_TO_GROUPS = {
    "Chandra": "IC",
    "Farha": "FP",
    "Indi": "IC",
    "Panang": "FP",
    "Thai": "T",
}
INDIVIDUALS_TO_ID = {
    "Chandra": 1,
    "Farha": 3,
    "Indi": 2,
    "Panang": 4,
    "Thai": 5,
    "Invalid": 0,
}
ROOMS = ["sand_box_ohne", "sand_box_mit", "other"]
CAMERA_TO_ROOM = {
    "zag_elp_cam_016": "sand_box_ohne",
    "zag_elp_cam_019": "sand_box_ohne",
    "zag_elp_cam_017": "sand_box_mit",
    "zag_elp_cam_018": "sand_box_mit",
}

CAMERA_TO_ID = {
    "zag_elp_cam_016": 0,
    "zag_elp_cam_017": 1,
    "zag_elp_cam_018": 2,
    "zag_elp_cam_019": 3,
}

pbar_manager = enlighten.get_manager()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Create dummy db data from tracks", add_help=add_help
    )

    parser.add_argument(
        "--dir", "-d", type=Path, help="Path to tracks data", required=True
    )
    parser.add_argument(
        "--dates", type=str, nargs="+", help="Specific dates to process", default=None
    )
    return parser


def gather_all_dates(root_dir: Path) -> list[str]:
    all_dates = set()
    for camera_dir in root_dir.glob("*"):
        for date_dir in camera_dir.glob("*"):
            if not date_dir.is_dir():
                # Skip empty.csv
                continue
            all_dates.add(date_dir.name)
    return list(sorted(all_dates))


def assign_individuals_to_cameras():
    groups = SOCIAL_GROUPS.copy()
    random.shuffle(groups)
    groups_to_room = {g: r for g, r in zip(groups, ROOMS)}
    individuals_to_camera = {}
    camera_to_individuals = {c: [] for c in CAMERA_TO_ROOM.keys()}
    for camera, room in CAMERA_TO_ROOM.items():
        for individual, group in INDIVIDUALS_TO_GROUPS.items():
            if groups_to_room[group] == room:
                camera_to_individuals[camera].append(individual)
    return camera_to_individuals


def select_sleeping_hours() -> list[int]:
    MIN_HOUR = 22
    MAX_HOUR = 8
    sleep_range_size = MAX_HOUR - MIN_HOUR
    if sleep_range_size < 0:
        sleep_range_size += 24
    sleep_hours = int(random.normalvariate(mu=3, sigma=1))
    if sleep_hours < 1:
        sleep_hours = 1
    if sleep_hours > sleep_range_size:
        sleep_hours = sleep_range_size

    hours = [MIN_HOUR + h for h in range(sleep_range_size)]
    hours = [h if h < 24 else h - 24 for h in hours]
    random.shuffle(hours)
    return hours[0:sleep_hours]


def log_track(
    db_cursor, camera: str, individual: str, sleeping_hours: list[int], track_file: Path
):
    camera_id = CAMERA_TO_ID[camera]
    individual_id = INDIVIDUALS_TO_ID[individual]
    df_track = pd.read_csv(
        track_file,
        parse_dates=["timestamp"],
    )
    row_count = len(df_track)
    if row_count == 0:
        return

    ts_start = df_track["timestamp"].iloc[0]
    ts_end = df_track["timestamp"].iloc[-1]

    # Insert into tracks table
    db_cursor.execute(
        "INSERT INTO tracks(camera_id, start_time, end_time, frame_count, identity_id, track_filename) "
        "VALUES(%s,%s,%s,%s,%s,%s)"
        "RETURNING id;",
        (camera_id, ts_start, ts_end, row_count, individual_id, track_file.stem),
    )
    (track_id,) = db_cursor.fetchone()

    # Insert into observations
    last_ts = None
    for i in range(row_count):
        ts: datetime = df_track["timestamp"].iloc[i]
        if ts == last_ts:
            # Skip duplicates
            continue
        last_ts = ts

        world_x = float(df_track["world_x"].iloc[i])
        world_y = float(df_track["world_y"].iloc[i])

        if ts.hour in sleeping_hours:
            if ts.minute <= 30:
                behaviour_id = BEHAVIOUR_SLEEP_LEFT_ID
            else:
                behaviour_id = BEHAVIOUR_SLEEP_RIGHT_ID
        else:
            behaviour_id = BEHAVIOUR_STAND_ID

        db_cursor.execute(
            """
            INSERT INTO observations(track_id, "time", location, behaviour_id)
            VALUES(%s, %s, point(%s, %s), %s)
            ON CONFLICT (track_id, "time")
            DO UPDATE SET
                location = EXCLUDED.location,
                behaviour_id = EXCLUDED.behaviour_id
            """,
            (track_id, ts, world_x, world_y, behaviour_id),
        )


def load_individual_from_csv(track_file: Path) -> str:
    df_track = pd.read_csv(track_file)
    row_count = len(df_track)
    if row_count == 0:  ### Empty track csv
        return None
    if "identity_label" not in df_track.columns:
        return "Invalid"
    individual = df_track["identity_label"].mode()[0]
    return individual


def main(args):
    root_dir: Path = args.dir
    # First gather all dates
    # so we can make consistent data for all cameras
    if args.dates is not None:
        all_dates = args.dates
    else:
        all_dates = gather_all_dates(root_dir)
    all_dates = [
        "2025-11-15",
        "2025-11-16",
        "2025-11-30",
        "2025-11-31",
        "2025-12-01",
        "2025-12-02",
        "2025-12-15",
        "2025-12-16",
    ]

    db_connection = psycopg2.connect("dbname=zoo_vision user=dherrera")
    db_cursor = db_connection.cursor()

    pbar = pbar_manager.counter(
        total=len(all_dates), desc="Processing dates", unit="day"
    )
    for date in pbar(all_dates):
        # Randomly select which individuals are where
        camera_to_individuals = assign_individuals_to_cameras()
        individual_to_sleeping_hours = {
            i: select_sleeping_hours() for i in INDIVIDUALS_TO_ID.keys()
        }
        for camera_dir in root_dir.glob("*"):
            camera = camera_dir.name
            possible_individuals = camera_to_individuals[camera]
            track_files = list((camera_dir / date).glob("*.csv"))

            pbar2 = pbar_manager.counter(
                total=len(track_files), desc="Processing tracks", unit="track"
            )
            for track_file in pbar2(track_files):
                individual = load_individual_from_csv(track_file)
                if individual is None:
                    continue
                log_track(
                    db_cursor,
                    camera,
                    individual,
                    individual_to_sleeping_hours[individual],
                    track_file,
                )
                db_connection.commit()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
