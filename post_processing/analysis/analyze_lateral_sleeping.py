
"""
functions:
1) load track csvs - filter short tracks + bad tracks (narrow rectangle, low conf)
2) get behavior predictions from tracks
   - count sleeping behavior predictions (count based on per-frame predictions)
   - aggreate
   
3) if evaluate:
  - load GT data (gt: id, behavior)
  - compare predictions to GT 

"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import os
import argparse
import logging
import sys
from typing import List, Tuple, Optional
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from post_processing.core.temporal_smooth import *

from post_processing.core.file_manager import (
    offline_track_dir,
    list_track_files,
)

from post_processing.analysis.evaluate_behavior import *
from post_processing.utils import *

_SLEEP_PREFIX = ("02_sleeping_left", "03_sleeping_right")
_STAND = "01_standing"

LABEL2NAME = {
    "01_standing": "standing",
    "02_sleeping_left": "sleep_L",
    "03_sleeping_right": "sleep_R",
}


def _to_dt64(s: pd.Series) -> pd.Series:
    return s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")

def _is_sleep(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    return s.str.startswith(_SLEEP_PREFIX[0]) | s.str.startswith(_SLEEP_PREFIX[1])

def _is_stand(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    return s.str.startswith(_STAND)



def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("post_processing")


def compute_avg_bbox_wh_ratio(df: pd.DataFrame) -> float:
    """
    Compute average (width / height) ratio of bounding boxes.

    width  = bbox_right  - bbox_left
    height = bbox_bottom - bbox_top

    Invalid or zero-height boxes are ignored.
    """
    width = df["bbox_right"] - df["bbox_left"]
    height = df["bbox_bottom"] - df["bbox_top"]

    valid = height > 0
    if not valid.any():
        return np.nan

    wh_ratio = width[valid] / height[valid]
    std = wh_ratio.std()
    mean = wh_ratio.mean()
    # print(f"Avg bbox W/H ratio: mean={mean:.2f}, std={std:.2f}")
    return float(wh_ratio.mean())


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
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_timestamp = df['timestamp'].min()
    end_timestamp = df['timestamp'].max()
    duration = (end_timestamp - start_timestamp).total_seconds() / 60.0
    
    if duration < min_duration_minutes or len(df) < min_num_frames:
        return None
    
    return df


def extract_track_data(csv_path: Path) -> Optional[pd.DataFrame]:
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
    # filter short tracks
    df = filter_short_tracks(df)
    if df is None:
        return None
    
    # filter bad tracks based on bbox wh ratio
    avg_wh_ratio = compute_avg_bbox_wh_ratio(df)
    if avg_wh_ratio < 0.3 or avg_wh_ratio > 3.0:
            # print(f"Filtered bad track {csv_path} with avg wh ratio {avg_wh_ratio:.2f}")
        return None

    return df


# def load_valid_tracks(record_root,
#                       camera_ids,
#                       start_datetime: pd.Timestamp,
#                       end_datetime: pd.Timestamp,
#                       new_beh_model: bool = False,
#                       logger: Optional[logging.Logger] = None,) -> pd.DataFrame:
    
#     # check if date comes from 2 days
#     if isinstance(start_datetime, str):
#         start_datetime = pd.to_datetime(start_datetime)
#     if isinstance(end_datetime, str):
#         end_datetime = pd.to_datetime(end_datetime)
#     start_date = start_datetime.date()
#     end_date = end_datetime.date()
#     date_list = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
    
#     all_track_csvs = {}
    
#     all_track_files: Dict[str, Tuple[List[Path], List[Path]]] = {}
#     for cam_id in camera_ids:
#         for date in date_list:
#             td = offline_track_dir(record_root, cam_id, date)
#             csv_list = list_track_files(td, logger=logger)
#             if cam_id not in all_track_csvs:
#                 all_track_csvs[cam_id] = []
#             all_track_csvs[cam_id].extend(csv_list)
        
#     valid_track_df = pd.DataFrame()
#     valid_count = 0
#     for cam_id, track_files in all_track_csvs.items():
#         for track_file in track_files:
#             # print("Processing track file:", track_file)
#             timestamp = track_file.stem.split('_')[0].replace('T', ' ')
#             date = track_file.parent.name
#             ## if date+timestamp not in range (start_datetime, end_datetime): continue
#             track_datetime = pd.to_datetime(f"{date} {timestamp}")
#             if track_datetime < start_datetime or track_datetime > end_datetime:
#                 # print(f"Skipping track {track_datetime} outside datetime range.", start_datetime, end_datetime)
#                 continue

#             if '_id_behavior' in str(track_file):
#                 continue
#             _track_file = track_file

#             new_beh_csv = str(track_file).replace('.csv', '_id_behavior.csv') if '_id_behavior' not in str(track_file) else None

#             if new_beh_csv is not None and Path(new_beh_csv).exists():
#                 if new_beh_model:
#                     _track_file = Path(new_beh_csv)
#                 else:
#                     _track_file = Path(track_file)
#             else:
#                 continue

#             track_data, identity = extract_track_data(_track_file)

#             if track_data is None:
#                 continue
#             valid_count += 1

#             track_data['track_filename'] = track_file.name
#             track_data['track_csv_path'] = str(_track_file)
#             track_data['camera_id'] = cam_id
#             valid_track_df = pd.concat([valid_track_df, track_data], ignore_index=True)
    
#     # sort by timestamp
#     valid_track_df = valid_track_df.sort_values(by=['camera_id', 'timestamp']).reset_index(drop=True)   
#     print(f"Total valid tracks loaded between {start_datetime} and {end_datetime}: {valid_count}")
    
#     return valid_track_df

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
            # Try alternative directory
            json_dir = record_root / 'demo-' / f'zag_elp_cam_{cam_id}' / date_str
            json_files = list(json_dir.glob(json_pattern))
        
        if not json_files:
            print(f"Warning: No JSON file found for camera {cam_id} at {json_dir}")
            continue
        
        json_path = json_files[0]
        print(f"Loading identity labels from: {json_path}")
        
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


def load_valid_tracks(record_root,
                      camera_ids,
                      start_datetime: pd.Timestamp,
                      end_datetime: pd.Timestamp,
                      new_beh_model: bool = False,
                      logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    
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
            csv_list = list_track_files(td, logger=logger)
            if cam_id not in all_track_csvs:
                all_track_csvs[cam_id] = []
            all_track_csvs[cam_id].extend(csv_list)
        
    valid_track_df = pd.DataFrame()
    valid_count = 0
    for cam_id, track_files in all_track_csvs.items():
        for track_file in track_files:
            # print("Processing track file:", track_file)
            timestamp = track_file.stem.split('_')[0].replace('T', ' ')
            date = track_file.parent.name
            track_datetime = pd.to_datetime(f"{date} {timestamp}")
            if track_datetime < start_datetime or track_datetime > end_datetime:
                # print(f"Skipping track {track_datetime} outside datetime range.", start_datetime, end_datetime)
                continue

            if '_id_behavior' in str(track_file):
                continue
            _track_file = track_file

            new_beh_csv = str(track_file).replace('.csv', '_id_behavior.csv') if '_id_behavior' not in str(track_file) else None

            if new_beh_csv is not None and Path(new_beh_csv).exists():
                if new_beh_model:
                    _track_file = Path(new_beh_csv)
                else:
                    _track_file = Path(track_file)
            else:
                continue

            track_data = extract_track_data(_track_file)
            if track_data is None:
                continue
            valid_count += 1

            track_data['track_filename'] = track_file.name.replace('.csv', '')
            track_data['track_csv_path'] = str(_track_file)
            track_data['camera_id'] = cam_id
            valid_track_df = pd.concat([valid_track_df, track_data], ignore_index=True)
    
    # sort by timestamp
    valid_track_df = valid_track_df.sort_values(by=['camera_id', 'timestamp']).reset_index(drop=True)   
            
    return valid_track_df

def load_gt_id(dir='/media/mu/zoo_vision/data/GT_id_behavior/id_GTs',
               camera_id='016',
               date='2025-11-15'):
    next_day = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    csv = Path (dir) / f'{camera_id}_{date}_{next_day}.csv'
    try:
        df = pd.read_csv(csv)
        columns = ['filename', 'gt']
        df = df[df['gt'].notna()]
        # remove gt == 'invalid' 
        df = df[df['gt'] != 'invalid']
        return df[columns]
    except Exception as e:
        print(f"Error loading ID GT from {csv}: {e}")
        return None


def load_gt_ids(dir='/media/mu/zoo_vision/data/GT_id_behavior/id_GTs',
               camera_ids=['016', '019'],
               date='2025-11-15'):
    next_day = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    columns = ['filename', 'gt']
    df_all = pd.DataFrame(columns=columns)
    for camera_id in camera_ids:
        csv = Path (dir) / f'{camera_id}_{date}_{next_day}.csv'
        try:
            df = pd.read_csv(csv)
            df = df[df['gt'].notna()]
            # remove gt == 'invalid' 
            df = df[df['gt'] != 'invalid']
            df_all = pd.concat([df_all, df[columns]], ignore_index=True)
        except Exception as e:
            print(f"Error loading ID GT from {csv}: {e}")
            
    return df_all

def load_gt_behavior(dir='/media/mu/zoo_vision/data/GT_id_behavior/behavior_GTs',
                     camera_ids=['016', '019'],
                     date='2025-11-15'):
    next_day = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    if len(camera_ids) == 1:
        cam = CAMERA_PARIS[ camera_ids[0] ]
        camera_ids.append(cam)
        # sort camera ids
        camera_ids = sorted(camera_ids)
    csv = Path (dir) / f'{camera_ids[0]}_{camera_ids[1]}_{date}_{next_day}.csv'
    try:
        df = pd.read_csv(csv)
        return df
    except Exception as e:
        print(f"Error loading behavior GT from {csv}: {e}")
        return None


def update_csv_from_df(df: pd.DataFrame, 
                       record_root: Path) -> None:
    
    track_files = df['track_filename'].unique().tolist()
    
    for track_file in track_files:
        _df = df[ df['track_filename'] == track_file ]
        df_original = pd.read_csv(track_file)
        original_columns = df_original.columns.tolist()
        _df = _df[original_columns]
        _df.to_csv(track_file, index=False)



def get_args():
    parser = argparse.ArgumentParser(description="Analyze lateral sleeping behavior from multi-camera tracks.")
    parser.add_argument('--record_root', type=str, default="/media/ElephantsWD/elephants/xmas", help='Root directory of the recordings.')
    parser.add_argument('--camera_ids', type=str, nargs='+', help='List of camera IDs to analyze.')
    parser.add_argument('--date', type=str, help='Date of the recordings (YYYY-MM-DD).')
    return parser.parse_args()


def main(record_root, date, camera_ids, is_smooth=False, eval_type='voted_track_label'):
    
    next_day = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    start_datetime=pd.Timestamp(f"{date} 15:30:00")
    end_datetime=pd.Timestamp(f"{next_day} 08:00:00")
    
    
    for new_beh_model in [False]:

        df_results = load_valid_tracks(
            record_root=record_root,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            camera_ids=camera_ids,
            new_beh_model=new_beh_model
        )
        
        df_label_predictions = load_identity_labels_from_json(
            record_root=Path(record_root),
            camera_ids=camera_ids,
            start_datetime=start_datetime,
            end_datetime=end_datetime
        )
        # merge identity labels into df_results, on track_filename
        df_results = df_results.merge(
            df_label_predictions[['track_filename', 'stitched_label', 'voted_track_label', 'smoothed_label', 'identity_label']],
            on='track_filename',
            how='left'
        )
        
        print(f"===== Identity Label TYPE: {eval_type} =====")
        df_results['identity_label'] = df_results[eval_type]
        
        if is_smooth:
            # df_results = behavior_label_smooth(df_results, window_size=21, min_conf_threshold=0.7)
            ### do cross-camera standing - matching
            df_results = smooth_behavior_cross_cameras(df_results)
            print("Applied temporal smoothing to behavior labels.", len(df_results))
            
            
        model_type = 'swinb' if new_beh_model else 'swinb_2_heads'
        print("\n" + "=" * 80 + "\n")
        print(f"========== Evaluation: between {start_datetime} and {end_datetime} on cameras {camera_ids} + [{model_type}] model + [is_smooth={is_smooth}] ========== ")

        semi_group_gt = load_semi_gt_ids(
            sandbox_gt_path=Path('/media/mu/zoo_vision/data/semi_supervised_gt/semi_supervised_gt_ids.csv'),
            date=date,
            camera_id=camera_ids[0]
        )
        # print(f"Semi GT IDs for camera {cam_id} on date {date}: {semi_group_gt}")

        if 'Thai' in semi_group_gt:
            # print("Thai is in the semi-supervised GT IDs.")
            df_gt_id = pd.DataFrame({
                'filename': [str(f).replace('.csv', '') for f in df_results['track_filename'].unique().tolist()],
                'gt': ['Thai'] * len(df_results['track_filename'].unique().tolist())
            })
        else:
            ## EVALUATION
            df_gt_id = load_gt_ids(
                dir='/media/mu/zoo_vision/data/GT_id_behavior/id_GTs',
                camera_ids=camera_ids,
                date=date
            )
        # print("df_gt_id head:")
        # print(df_gt_id.head())
        
        df_gt_behavior = load_gt_behavior(
            dir='/media/mu/zoo_vision/data/GT_id_behavior/behavior_GTs',
            camera_ids=camera_ids,
            date=date
        )
        # print("df_gt_behavior head:", df_gt_id['gt'].unique().tolist())
        # print(df_gt_behavior.head())

        # print("df_results shape:", df_results.shape)
        
    
        df_eval = build_gt_behavior_for_results(df_results, df_gt_id, df_gt_behavior)

        # print("df_eval shape:", df_eval.shape)

        # print("Evaluation DataFrame head:")
        # print(df_eval['pred_behavior'])

        dir = '/media/mu/zoo_vision/post_processing/analysis/csvs/id_behaviors/data'
        Path(dir).mkdir(parents=True, exist_ok=True)

        df_eval.to_csv(f"{dir}/eval_behavior_cam{camera_ids[0]}_cam{camera_ids[1]}_{date}_model_is_smooth{is_smooth}.csv", index=False)

        df_eval['pred_behavior'] = df_eval['pred_behavior'].apply(lambda x: x.replace(f"_{x.split('_')[-1]}", "") if 'good' in x or 'bad' in x else x)

        df_eval_correct_ids = df_eval[ df_eval['gt_identity'] == df_eval['identity_label'] ]
        df_eval_correct_ids_wrong_beahvior = df_eval_correct_ids[ df_eval_correct_ids['gt_behavior'] != df_eval_correct_ids['pred_behavior'] ]
        df_eval_correct_ids_wrong_beahvior.to_csv(f"eval_correct_ids_wrong_behavior_cam{camera_ids[0]}_cam{camera_ids[1]}_{date}_model_is_smooth{is_smooth}.csv", index=False)
        

        if 'Thai' in semi_group_gt:
            df = df_eval.copy()
            gts = df['gt_behavior'].to_list()
            preds = df['pred_behavior'].to_list()

            ## compute accuracy -- cm.
            accuracy = accuracy_score(gts, preds)
            print(f"Accuracy: {accuracy:.4f}")
            
            cm = confusion_matrix(gts, preds)
            print("\nConfusion Matrix (counts):")
            print(cm)
            
            # Get unique labels for better display
            labels = sorted(set(gts + preds))
            print(f"\nLabels: {labels}")
            
            # Classification report for detailed metrics
            print("\nClassification Report:")
            print(classification_report(gts, preds, zero_division=0))
        else:
            # print(df_eval.head())
            # print("df eval columns:", df_eval.columns.tolist())
            
            out = behavior_metrics_per_id(df_eval)
            print("\nBehavior metrics per ID:")
            for _id, metrics in out.items():
                print(f"ID: {_id}, Accuracy: {metrics['accuracy']:.4f}")
                # print(f", n: {metrics['n']}")
                
            out = behavior_metrics_per_id_per_cam(df_eval)
            print("\nBehavior metrics per ID per Camera:")
            for (_id, _cam), metrics in out.items():
                print(f"ID: {_id}, Camera: {_cam}, Accuracy: {metrics['accuracy']:.4f}, n: {metrics['n']}")
                
            """
            out = compute_gt_joint_wrongid_hours(df_eval, fps=25, log_every_n_frames=5, extra_group_cols=["camera_id"])
            pp = postprocess_time_outputs(out)
            
            predicted_hours = compute_predicted_hours_per_id_behavior(
                df_eval,
                dt_seconds=out["dt_seconds"],      # IMPORTANT: reuse same dt
                extra_group_cols=["camera_id"],    # optional, but keep consistent
            )
             
            # Wrong-ID hours per PREDICTED behavior per GT ID
            wrong_by_pred = wrong_id_hours_per_behavior(df_eval, dt_seconds=out["dt_seconds"], behavior_source="pred", per_id=True)
            # print("\nWrong-ID hours per PREDICTED behavior per GT ID:")
            # print(wrong_by_pred)
            
            # merge wrong_by_pred into pp["per_id_behavior_merged"]
            pp["per_id_behavior_merged"] = pp["per_id_behavior_merged"].merge(
                wrong_by_pred.rename(columns={'wrong_id_hours': "wrong_id_hours"})[['gt_identity', 'gt_behavior', "wrong_id_hours"]],
                on=['gt_identity', 'gt_behavior'],
                how="left",
            )
            pp["per_id_behavior_merged"]["wrong_id_hours"] = pp["per_id_behavior_merged"]["wrong_id_hours"].fillna(0.0) 

            correct_id_wrong_beh = correct_id_wrong_behavior_hours_per_id_behavior(
                df_eval,
                dt_seconds=out["dt_seconds"],      # IMPORTANT: reuse same dt
                extra_group_cols=["camera_id"],    # if you used this elsewhere
            )
            pp["per_id_behavior_merged"] = add_correct_id_wrong_behavior_to_per_id_behavior(
                pp["per_id_behavior_merged"],
                correct_id_wrong_beh,
            )
            
            columns = pp["per_id_behavior_merged"].columns.tolist()
            # ['gt_identity', 'gt_behavior', 'gt_hours', 'both_correct_h', 'wrong_id_hours', 'cor_id_wrong_beh']
            rename_columns = {
                'gt_identity': 'ID',
                'gt_behavior': 'gt_beh',
                'gt_hours': 'gt_hours',
                'both_correct_h': 'both_correct_hrs',
                'wrong_id_hours': 'wrong_id_hrs',
                'cor_id_wrong_beh': 'cor_id_wrong_beh_hrs (sleeping)',
            }
            pp["per_id_behavior_merged"] = pp["per_id_behavior_merged"].rename(columns=rename_columns)[list(rename_columns.values())]
            pp["per_id_behavior_merged"]["gt_beh"] = pp["per_id_behavior_merged"]["gt_beh"].apply(lambda x: LABEL2NAME.get(x, x))

            print("\n Per ID Behavior with Wrong-ID Hours by PREDICTED behavior:")
            ### print the column names, rows
            for col in pp["per_id_behavior_merged"].columns:
                print(f"{col:20}", end="")
            print()
            for _, row in pp["per_id_behavior_merged"].iterrows():
                for col in pp["per_id_behavior_merged"].columns:
                    item = row[col]
                    if isinstance(row[col], float):
                        item = f"{item:.2f}"
                    print(f"{str(item):20}", end="")
                print()
            print()
            """
            # 

        
if __name__ == "__main__":


    args = get_args()
    
    record_root = args.record_root
    camera_ids=args.camera_ids
    date = args.date


    gt_dates_cams= {
        '2025-11-15': [['017', '018'], ['016', '019']],
        '2025-11-30': [['017', '018'], ['016', '019']],
        '2025-12-01': [['016', '019']],
        '2025-12-15': [['017', '018']],
    }



    # ### THAI - data/GT_id_behavior/behavior_GTs/017_018_2025-12-15_2025-12-16.csv, ./data/GT_id_behavior/behavior_GTs/016_019_2025-12-01_2025-12-02.csv
    # camera_ids = ["017", "018"]
    # date = "2025-12-15"

    # date = "2025-12-01"
    # camera_ids=["016", "019"]
    
    # # ### IC group
    
    # camera_ids=["016", "019"]
    # date = "2025-11-15"
    # camera_ids = ["017", "018"]
    # date = "2025-11-30"

    for date, camera_ids_list in gt_dates_cams.items():
        for camera_ids in camera_ids_list:
            print(f"\n\nProcessing date {date} for cameras {camera_ids}...\n")
            for is_smooth in [True]:
                main(record_root, date, camera_ids, is_smooth=is_smooth)