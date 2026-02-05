"""Fills the db with dummy data by reading the tracks folder"""

from pathlib import Path
from datetime import datetime
import enlighten
import pandas as pd
import psycopg2

INDIVIDUALS_TO_ID = {
    "Chandra": 1,
    "Farha": 3,
    "Indi": 2,
    "Panang": 4,
    "Thai": 5,
    "Invalid": 0,
}
BEHAVIOURS_TO_ID = {
    "00_invalid": 0,
    "01_standing": 1,
    "02_sleeping_left": 2,
    "03_sleeping_right": 3,
}

CAMERA_TO_ID = {
    "zag_elp_cam_016": 0,
    "zag_elp_cam_017": 1,
    "zag_elp_cam_018": 2,
    "zag_elp_cam_019": 3,
}
TYPO = {
    'Fahra': 'Farha',
}

pbar_manager = enlighten.get_manager()



def load_identity_labels_from_json(
    record_root: Path,
    cam_id: str,
    date : str,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
    id_col: str = "voted_track_label", ### now using voted labels before fixing the stitching issue
) -> pd.DataFrame:
    """
    Load identity labels from stitched tracklets JSON files.
    
    Returns:
        DataFrame with columns: track_filename, stitched_label, voted_track_label, 
                                smoothed_label, identity_label
    """
    import json
    
    # Format date string (assuming date is in format "YYYY-MM-DD")
    date_str = date
    # add '-' to date string if not present
    if '-' not in date_str:
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    # Format time strings from timestamps
    start_time_str = start_datetime.strftime("%H%M%S")
    end_time_str = end_datetime.strftime("%H%M%S")
    ## 
    all_labels = []

    json_dir = record_root / 'demo' / f'zag_elp_cam_{cam_id}' / date_str
    json_pattern = f'stitched_tracklets_cam{cam_id}_{start_time_str}_{end_time_str}.json'
    
    # Find the JSON file
    json_files = list(json_dir.glob(json_pattern))
    # print(f"Searching for JSON files in: {json_dir} with pattern: {json_pattern}")
    if not json_files:
        json_files = list(json_dir.glob(f'*.json'))

    if not json_files:
        print(f"Warning: No JSON file found for camera {cam_id} at {json_dir}")
        return pd.DataFrame()
    
    json_path = json_files[0]
    print(f"Loading identity labels from: {json_path}")
    
    # Load JSON
    with open(json_path, 'r') as f:
        tracklets_data = json.load(f)
    
    track_file2_label = {}
    for tracklet in tracklets_data:
        track_file2_label[tracklet['track_filename']] = tracklet.get(id_col, 'Invalid')
        
    return track_file2_label


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
    parser.add_argument(
        "--start_timestamp", type=str, help="Start timestamp for processing, e.g., '180000'", required=True
    )
    parser.add_argument(
        "--end_timestamp", type=str, help="End timestamp for processing, e.g., '080000'", required=True
    )
    return parser


def gather_all_nights(root_dir: Path) -> list[str]:
    all_nights = set()
    for camera_dir in root_dir.glob("*"):
        for night_dir in camera_dir.glob("*"):
            if not night_dir.is_dir():
                # Skip empty.csv
                continue
            all_nights.add(night_dir.name)
    return list(sorted(all_nights))


def merge_track_behavior(track_file: Path) -> pd.DataFrame:
    df_track = pd.read_csv(track_file)
    behavior_file = track_file.with_name(track_file.stem + "_behavior.csv")
    if not behavior_file.exists():
        return df_track
    df_behavior = pd.read_csv(behavior_file)
    df_merged = pd.merge(df_track, df_behavior, on='timestamp', how='left')
    return df_merged


def log_track(db_cursor, camera: str, individual: str, track_file: Path):
    camera_id = CAMERA_TO_ID[camera]
    if individual == 'invalid':
        individual = 'Invalid'
    individual_id = INDIVIDUALS_TO_ID[individual]
    # df_track = pd.read_csv(
    #     track_file,
    #     parse_dates=["timestamp"],
    # )
    df_track = merge_track_behavior(track_file)
    row_count = len(df_track)
    if row_count == 0:
        return
    
    ### filter out bad quality frames if any
    if 'quality_label' in df_track.columns:
        df_track = df_track[df_track['quality_label'] == 'good']
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

        if "behavior_label" in df_track:
            behaviour = df_track["behavior_label"].iloc[i]
        else:
            behaviour = "00_invalid"
        behaviour_id = BEHAVIOURS_TO_ID[behaviour.lower()]

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
        return "Invalid"
    if "identity_label" not in df_track.columns:
        return "Invalid"
    individual = df_track["identity_label"].mode()[0]
    return individual


def load_behaviour_from_csv(track_file: Path) -> str:
    df_track = pd.read_csv(track_file)
    row_count = len(df_track)
    if row_count == 0:  ### Empty track csv
        return "Invalid"
    if "identity_label" not in df_track.columns:
        return "Invalid"
    individual = df_track["identity_label"].mode()[0]
    return individual


def normalize_time_string(time_str: str) -> str:
    """Convert various time formats to HHMMSS format.
    
    Handles: 18, 1800, 180000, 18:00:00 -> all convert to 180000
    """
    # Remove colons if present
    time_str = time_str.replace(':', '')
    
    # Pad to 6 digits (HHMMSS)
    if len(time_str) == 1:
        time_str = time_str.zfill(2) + '0000'  # "18" -> "180000"
    elif len(time_str) == 2:
        time_str = time_str + '0000'  # "18" -> "180000"
    elif len(time_str) == 4:
        time_str = time_str + '00'  # "1800" -> "180000"
    elif len(time_str) == 6:
        pass  # Already correct format
    else:
        raise ValueError(f"Invalid time format: {time_str}")
    
    return time_str

def list_track_files_for_night(camera_dir: Path, 
                               date: str) -> list[Path]:
    """List all track files for a given night (date + time range)"""
    track_files = []
    date_str = date
    # add '-' to date string if not present
    if '-' not in date_str:
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    date_dir = camera_dir / date_str
    next_date = (pd.to_datetime(date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    next_date_dir = camera_dir / next_date
    # List files from date_dir
    
    if date_dir.exists():
        track_files.extend(list(date_dir.glob("*.csv")))
    # List files from next_date_dir
    if next_date_dir.exists():
        track_files.extend(list(next_date_dir.glob("*.csv")))
        
    return track_files
      

def delete_existing_data_for_night(db_cursor, date: str, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp):
    """Delete all observations and tracks for a specific night before re-inserting."""
    
    # Convert date string to datetime objects for the night range
    if '-' not in date:
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    
    date_dt = pd.to_datetime(date)
    
    # Night starts at start_time on date and ends at end_time next day
    night_start = date_dt + pd.Timedelta(
        hours=start_timestamp.hour,
        minutes=start_timestamp.minute,
        seconds=start_timestamp.second
    )
    
    if end_timestamp < start_timestamp:
        # End time is on next day
        night_end = date_dt + pd.Timedelta(days=1) + pd.Timedelta(
            hours=end_timestamp.hour,
            minutes=end_timestamp.minute,
            seconds=end_timestamp.second
        )
    else:
        # End time is on same day
        night_end = date_dt + pd.Timedelta(
            hours=end_timestamp.hour,
            minutes=end_timestamp.minute,
            seconds=end_timestamp.second
        )
    
    print(f"Deleting existing data for night: {night_start} to {night_end}")
    
    # First delete observations for tracks in this time range
    db_cursor.execute(
        """
        DELETE FROM observations
        WHERE track_id IN (
            SELECT id FROM tracks
            WHERE start_time >= %s AND start_time < %s
        )
        """,
        (night_start, night_end)
    )
    deleted_observations = db_cursor.rowcount
    
    # Then delete the tracks themselves
    db_cursor.execute(
        """
        DELETE FROM tracks
        WHERE start_time >= %s AND start_time < %s
        """,
        (night_start, night_end)
    )
    deleted_tracks = db_cursor.rowcount
    
    print(f"Deleted {deleted_observations} observations and {deleted_tracks} tracks")


def main(args):
    root_dir: Path = args.dir
    # First gather all dates
    # so we can make consistent data for all cameras
    if args.dates is not None:
        all_dates = args.dates
    else:
        all_dates = gather_all_nights(root_dir)
    
    # Normalize and convert timestamp strings to pd.Timestamp
    start_timestamp = normalize_time_string(args.start_timestamp)
    end_timestamp = normalize_time_string(args.end_timestamp)
    
    start_timestamp = pd.to_datetime(start_timestamp, format="%H%M%S")
    end_timestamp = pd.to_datetime(end_timestamp, format="%H%M%S")

    db_connection = psycopg2.connect("dbname=zoo_vision user=dherrera")
    db_cursor = db_connection.cursor()

    pbar = pbar_manager.counter(
        total=len(all_dates), desc="Processing nights", unit="night"
    )
    
    ### upload data from one night (date)
    for date in pbar(all_dates):
        ## date format if YYYYMMDD -> YYYY-MM-DD
        if '-' not in date:
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        
        # DELETE existing data for this night BEFORE inserting new data -- to avoid duplicates when re-running the script
        delete_existing_data_for_night(db_cursor, date, start_timestamp, end_timestamp)
        db_connection.commit()
        
        for camera_dir in root_dir.glob("*"):
            camera = camera_dir.name
            ### all track files for this night (date 18h - next day 8h)
            track_files = list_track_files_for_night(camera_dir, date)

            ### one night data are saved in one json file
            trackfile2labels = load_identity_labels_from_json(
                record_root=root_dir.parent,
                cam_id=camera.split('_')[-1],
                date = date,
                start_datetime=start_timestamp,
                end_datetime=end_timestamp,
            )
            pbar2 = pbar_manager.counter(
                total=len(track_files), desc="Processing tracks", unit="track"
            )
            for track_file in pbar2(track_files):
                # skip behavior files, and part_ files
                if '_behavior' in track_file.stem or 'part_' in track_file.stem:
                    continue

                individual = trackfile2labels.get(track_file.stem, None)
                # fix typos
                if individual in TYPO:
                    individual = TYPO[individual]

                if individual is None:
                    continue
                log_track(
                    db_cursor,
                    camera,
                    individual,
                    track_file,
                )
    db_connection.commit()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
