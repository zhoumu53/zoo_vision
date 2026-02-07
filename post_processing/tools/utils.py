import logging
from pathlib import Path

from pyparsing import Optional
from post_processing.tools.videoloader import VideoLoader
from post_processing.core.behavior_inference import BehaviorInference
import pandas as pd
import numpy as np

def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("post_processing")


def run_behavior_on_track(
    video_path: Path,
    behavior_model: BehaviorInference,
    batch_size: int = 64,
) -> list[tuple[str, float]]:
    """Run frame-by-frame behavior classification on a track clip."""
    loader = VideoLoader(str(video_path), verbose=False)
    if not loader.ok():
        raise RuntimeError(f"Could not open video for behavior inference: {video_path}")

    results: list[tuple[str, float]] = []
    batch: list = []

    for frame in loader:
        batch.append(frame)
        if len(batch) >= batch_size:
            preds = behavior_model.predict(batch, batch_size=batch_size)
            results.extend(preds)
            batch.clear()

    if batch:
        preds = behavior_model.predict(batch, batch_size=batch_size)
        results.extend(preds)

    return results


def load_gallery_features(
    reid_model,
    checkpoint_path: Path,
    provided_path: Path | None,
    logger: logging.Logger,
    known_labels: list[str] | None = None,
):
    """Resolve gallery path and load gallery features."""
    gallery_path = provided_path
    if gallery_path is None:
        gallery_path = (
            checkpoint_path.parent / "pred_features" / "train_iid" / "pytorch_result_e.npz"
        )
        logger.info("No gallery path provided. Using default: %s", gallery_path)
    return reid_model.load_features(gallery_path, known_labels=known_labels)


def load_behavior_model(
    behavior_model_path: Path | None, device: str, logger: logging.Logger
) -> BehaviorInference | None:
    """Load behavior model if path is provided."""
    if behavior_model_path is None:
        return None
    logger.info("Loading behavior model from %s", behavior_model_path)
    return BehaviorInference(
        model_path=str(behavior_model_path),
        device=device,
        logger=logger,
    )
    
def compute_box_wh_ratio(bbox_right, bbox_left, bbox_bottom, bbox_top) -> float:
    """Compute width/height ratio of the bounding box at a given index in the dataframe."""
    width = bbox_right - bbox_left
    height = bbox_bottom - bbox_top
    if height == 0 or width == 0:
        return 0.0  
    wh_ratio = width / height
    return wh_ratio


def tlbr2fullsize(df_tracks, img_width: int, img_height: int) -> tuple[float, float, float, float]:
    """Convert YOLO format bbox (tlbr) in 0-1 to (x1, y1, x2, y2) in full size."""
    #  frame_id	timestamp	bbox_top2	bbox_left2	bbox_bottom2	bbox_right2
    # 96269	2025-02-05 23:04:27.047000000	0.197667	0.493405	0.270667	0.508482
    df_tracks_converted = df_tracks.copy()


    # ### DEBUG 
    # import cv2
    # from pathlib import Path
    # video_path = Path('/mnt/camera_nas/ZAG-ELP-CAM-016/20250209PM/ZAG-ELP-CAM-016-20250209-180018-1739120418220-7.mp4')
    # cap = cv2.VideoCapture(str(video_path))
    # ret, frame = cap.read()
    # print(frame.shape)
    # img_width = frame.shape[1]
    # img_height = frame.shape[0]

    # if 'bbox_top2' in df_tracks.columns and 'bbox_top' not in df_tracks.columns:
    df_tracks['bbox_top'] = df_tracks['bbox_top2'] * img_height
    df_tracks['bbox_left'] = df_tracks['bbox_left2'] * img_width
    df_tracks['bbox_bottom'] = df_tracks['bbox_bottom2'] * img_height
    df_tracks['bbox_right'] = df_tracks['bbox_right2'] * img_width

    print("Converted bbox columns to full size.", df_tracks)

    # if ret:
    #     box = df_tracks.iloc[0]
    #     print(" +++++++++++++++++++++++++++++++++++++++ ")
    #     print(box)
    #     x1, y1, x2, y2 = int(box['bbox_top']), int(box['bbox_left']), int(box['bbox_bottom']), int(box['bbox_right'])
    #     cv2.rectangle(frame, (y1, x1), (y2, x2), (0, 255, 0), 2)  # green
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red
    #     cv2.imwrite('debug_box.jpg', frame)
    #     print("debug_box.jpg")
    #     import sys; sys.exit()


    return df_tracks_converted



def filter_by_box_quality(df_tracks,
                           bbox_ratio_lower: float = 1/3,
                           bbox_ratio_upper: float = 3.0
                           ) -> list[int]:
    ### Identify good frame indices based on box quality 
    wh_ratios = df_tracks.apply(
        lambda row: compute_box_wh_ratio(row['bbox_right'], 
                                         row['bbox_left'], 
                                         row['bbox_bottom'], 
                                         row['bbox_top']),
        axis=1
    )
    good_quality_mask = (wh_ratios >= bbox_ratio_lower) & (wh_ratios <= bbox_ratio_upper)
    
    return good_quality_mask


def get_good_frame_indices(df_tracks, ) -> list[int]:
    """Get frame indices with good quality boxes."""
    good_quality_mask = filter_by_box_quality(df_tracks)
    good_frame_indices = df_tracks.index[good_quality_mask].tolist()
    return good_frame_indices



def load_tracklets_dataframe(json_path: Path) -> pd.DataFrame:
    """Load tracklet data from a JSON file."""
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)

    columns = ['track_filename', 'track_csv_path', 'stitched_label', 'stitched_id', 'voted_track_label', 'start_timestamp', 'end_timestamp']
    df = pd.DataFrame(data, columns=columns)

    return df



def load_identity_labels_from_json(
    record_root: Path,
    camera_ids: list,
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load identity labels from stitched tracklets JSON files.
    
    Returns:
        DataFrame with columns: track_filename, stitched_label, voted_track_label, 
                                smoothed_label, identity_label
    """
    import json
    
    # Format datetime for JSON filename
    start_time_str = start_datetime.strftime("%Y%m%d_%H%M%S")
    end_time_str = end_datetime.strftime("%Y%m%d_%H%M%S")
    date_str = start_datetime.strftime("%Y-%m-%d")
    
    all_labels = []
    
    for cam_id in camera_ids:
        # Construct JSON path
        json_dir = record_root / 'demo' / f'zag_elp_cam_{cam_id}' / date_str
        json_pattern = f'stitched_tracklets_cam{cam_id}_{start_time_str}_{end_time_str}.json'
        
        # Find the JSON file
        json_files = list(json_dir.glob(json_pattern))
        # print(f"Searching for JSON files in: {json_dir} with pattern: {json_pattern}")

        if not json_files:
            print(f"Warning: No JSON file found for camera {cam_id} at {json_dir}")
            continue
        
        json_path = json_files[0]
        # print(f"Loading identity labels from: {json_path}")
        
        # Load JSON
        with open(json_path, 'r') as f:
            tracklets_data = json.load(f)
        
        # Extract labels for each track
        for tracklet in tracklets_data:
            all_labels.append({
                'track_filename': tracklet.get('track_filename', ''),
                'track_csv_path': tracklet.get('track_csv_path', ''),
                'camera_id': tracklet.get('camera_id', cam_id),
                'stitched_label': tracklet.get('stitched_label', 'invalid'),
                'voted_track_label': tracklet.get('voted_track_label', 'invalid'),
                'smoothed_label': tracklet.get('smoothed_label', 'invalid'),
                'identity_label': tracklet.get('identity_label', 'invalid'),
            })
    
    if not all_labels:
        print("Warning: No identity labels loaded from JSON files")
        return pd.DataFrame()
    
    df_labels = pd.DataFrame(all_labels)
    return df_labels



def get_tracklet_json_path(tracklet_dir: Path, 
                           cam_id: str, 
                           date: str,
                           start_datetime: pd.Timestamp,
                            end_datetime: pd.Timestamp) -> Path:
    
    ### load stitched results & voted results
    json_path = tracklet_dir / f'zag_elp_cam_{cam_id}' / date / f'*{start_datetime.strftime("%H%M%S")}*{end_datetime.strftime("%H%M%S")}*.json'
    # if not exist, raise error
    try:
        json_path = list(json_path.parent.glob(json_path.name))[0]
    except IndexError:
        raise FileNotFoundError(f"No JSON file found for cam {cam_id} on date {date} between {start_datetime} and {end_datetime}")

    return json_path



def load_valid_tracks(record_root,
                      camera_ids,
                      start_datetime: pd.Timestamp,
                      end_datetime: pd.Timestamp,
                      behavior_csv_suffix: str = '_behavior.csv'
                     ) -> pd.DataFrame:
    
    from post_processing.core.file_manager import (
        offline_track_dir,
        list_track_files,
    )
    # check if date comes from 2 days
    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)
    start_date = start_datetime.date()
    end_date = end_datetime.date()
    date_list = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
    
    all_track_csvs = {}
    
    all_track_files: Dict[str, Tuple[List[Path], List[Path]]] = {}
    for cam_id in camera_ids:
        for date in date_list:
            td = offline_track_dir(record_root, cam_id, date)
            csv_list = list_track_files(td)
            csv_list = [f for f in csv_list if behavior_csv_suffix not in str(f)]  ### track csv only
            if cam_id not in all_track_csvs:
                all_track_csvs[cam_id] = []
            all_track_csvs[cam_id].extend(csv_list)
        
    valid_track_df = pd.DataFrame()
    valid_count = 0
    for cam_id, track_files in all_track_csvs.items():
        for track_file in track_files:
            # print("Processing track file:", track_file)
            date = track_file.parent.name
            behavior_csv = track_file.with_name(track_file.stem + behavior_csv_suffix)
            if not behavior_csv.exists():
                continue
            
            # filter out the tracks that are outside the time range (start_datetime, end_datetime) based on the timestamp in the filename

            track_data = extract_track_data(track_file)
            if track_data is None:
                continue
            ### check if track is within the time range
            track_start_time = track_data['timestamp'].min()
            if track_start_time < start_datetime or track_start_time > end_datetime:
                continue
            
            behavior_data = extract_track_data(behavior_csv)
            if behavior_data is None or 'behavior_label' not in behavior_data.columns:
                continue

            ### check if length matches
            # if len(track_data) != len(behavior_data):
                # print(f"Warning: Length mismatch between track data ({len(track_data)}) and behavior data ({len(behavior_data)}) for {track_file.name}.")
                # continue

            track_data = pd.merge(track_data, behavior_data, on='timestamp', how='left')

            valid_count += 1

            track_data['track_filename'] = track_file.name.replace('.csv', '')
            track_data['track_csv_path'] = str(track_file)
            track_data['camera_id'] = cam_id
            valid_track_df = pd.concat([valid_track_df, track_data], ignore_index=True)
    # sort by timestamp
    valid_track_df = valid_track_df.sort_values(by=['camera_id', 'timestamp']).reset_index(drop=True)   
            
    return valid_track_df



def extract_track_data(csv_path: Path) -> pd.DataFrame:
    """
    Return:
      - first non-empty 'identity_label' in the CSV (or None)
      - total number of data rows in the CSV (excluding header)
    """

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    if 'timestamp' in df.columns:
        # filter short tracks
        df = filter_short_tracks(df)
        if df is None:
            return None
    
    # if quality all bad quality_label == 'bad', return None
    if 'quality_label' in df.columns:
        if (df['quality_label'] == 'bad').all():
            return None

    return df




def filter_short_tracks(df: pd.DataFrame,
                        min_duration_minutes: float = 1.0,
                        min_num_frames: int = 20) -> pd.DataFrame:
    """
    Filter out tracks that are too short in duration or have too few frames.

    Args:
        df: DataFrame containing track data with 'timestamp' column.
        min_duration_minutes: Minimum duration of the track in minutes.
        min_num_frames: Minimum number of frames in the track.
        
    Returns:
        Filtered DataFrame with only valid tracks.
    """
    if df.empty:
        return df
    
    df['timestamp'] = pd.to_datetime(df['timestamp'],
                                    format="%Y-%m-%d %H:%M:%S.%f",
                                    errors="coerce")
    start_timestamp = df['timestamp'].min()
    end_timestamp = df['timestamp'].max()
    duration = (end_timestamp - start_timestamp).total_seconds() / 60.0
    
    if duration < min_duration_minutes or len(df) < min_num_frames:
        return None
    
    return df



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

