"""
Goal: active learning for reid model
   - evaluate reid performance on tracks with GT -> if correct - put into labeled pool
"""

import pandas as pd
import numpy as np
import json
import shutil
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from post_processing.analysis.analyze_lateral_sleeping import load_gt_ids
from post_processing.tools.videoloader import VideoLoader


ID2LABELS = {
    'Chandra': '01_Chandra',
    'Indi': '02_Indi',
    'Fahra': '03_Fahra',
    'Panang': '04_Panang',
    'Thai': '05_Thai',
    'Zali': '06_Zali',
}

OUTPUT_DIR = Path('/media/mu/zoo_vision/data/reid_time_split/active_labels')
GT_DIR = Path('/media/mu/zoo_vision/data/GT_id_behavior/id_GTs')
TRACK_ROOT = Path('/media/mu/zoo_vision/xmas')
STITCHED_DIR = TRACK_ROOT / 'demo'


def load_stitched_results(
    date: str,
    cam_id: str,
    start_time: str = '15:30:00',
    end_time: str = '08:00:00'
) -> pd.DataFrame:
    """Load stitched results from JSON file.
    
    Returns:
        DataFrame with columns: track_filename, track_csv_path, stitched_id, voted_track_label
    """
    json_path = STITCHED_DIR / f'zag_elp_cam_{cam_id}' / date / f'*{start_time.replace(":", "")}*{end_time.replace(":", "")}*.json'
    json_paths = list(json_path.parent.glob(json_path.name))
    
    if not json_paths:
        raise FileNotFoundError(f"No stitched results found: {json_path}")
    
    json_path = json_paths[0]
    print(f"Loading stitched results from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        raise ValueError(f"No data found in JSON file: {json_path}")
    
    data_items = [i for _d in data.values() for i in _d]
    
    columns = ['track_filename', 'track_csv_path', 'identity_label', 'voted_track_label']
    df_stitched = pd.DataFrame(data_items, columns=columns)
    df_stitched = df_stitched.rename(columns={'identity_label': 'stitched_id'})
    
    return df_stitched


def get_correct_predictions(
    date: str,
    cam_id: str,
    start_time: str = '15:30:00',
    end_time: str = '08:00:00',
    prediction_type: str = 'voted_track_label'  # or 'stitched_id'
) -> pd.DataFrame:
    """Get tracks where predictions match ground truth.
    
    Args:
        date: Date string in format 'YYYY-MM-DD'
        cam_id: Camera ID (e.g., '016')
        start_time: Start time for filtering
        end_time: End time for filtering
        prediction_type: Column name to use for prediction ('voted_track_label' or 'stitched_id')
    
    Returns:
        DataFrame with correctly predicted tracks
    """
    # Load GT annotations
    df_gt_id = load_gt_ids(
        dir=GT_DIR,
        camera_ids=[cam_id],
        date=date
    )
    df_gt_id['filename'] = df_gt_id['filename'].apply(lambda x: x.replace('.csv', ''))
    
    if df_gt_id.empty:
        print(f"Warning: No GT IDs found for date {date} and cam_id {cam_id}")
        return pd.DataFrame()
    
    # Load stitched/voted results
    df_stitched = load_stitched_results(date, cam_id, start_time, end_time)
    
    # Merge with GT
    df_eval = pd.merge(
        df_stitched,
        df_gt_id,
        how='inner',
        left_on='track_filename',
        right_on='filename'
    )
    
    if df_eval.empty:
        print(f"Warning: No matching tracks found between predictions and GT")
        return pd.DataFrame()
    
    # Remove NaN GTs
    df_eval = df_eval[~df_eval['gt'].isna()]
    
    # Filter for correct predictions
    df_correct = df_eval[df_eval['gt'] == df_eval[prediction_type]].copy()
    
    print(f"Date {date}, Cam {cam_id}: {len(df_correct)}/{len(df_eval)} correct predictions ({len(df_correct)/len(df_eval)*100:.1f}%)")
    
    return df_correct


def copy_correct_tracks_to_labeled_pool(
    df_correct: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    max_frames: int = 100
) -> Dict[str, int]:
    """Extract frames from correctly predicted track videos and save to labeled pool.
    
    Args:
        df_correct: DataFrame with correctly predicted tracks
        output_dir: Output directory for labeled pool
        max_frames: Maximum number of random frames to extract (100 by default)
    
    Returns:
        Dictionary mapping identity labels to count of tracks copied
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    counts = {}
    
    for _, row in df_correct.iterrows():
        gt_label = row['gt']
        track_csv_path = Path(row['track_csv_path'])
        
        # Get mapped label (e.g., 'Chandra' -> '01_Chandra')
        mapped_label = ID2LABELS.get(gt_label, gt_label)
        
        # Create output directory for this identity
        label_dir = output_dir / mapped_label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        track_name = track_csv_path.stem
        track_dir = track_csv_path.parent
        
        # Find track video file
        track_video_path = track_dir / track_csv_path.with_suffix('.mkv').name
        if not track_video_path.exists():
            track_video_path = track_dir / track_csv_path.with_suffix('.mp4').name
        
        if not track_video_path.exists():
            print(f"  Warning: Video not found for {track_name}")
            continue
        
        dest_track_dir = label_dir
        ## check if dest_track_dir already has frames for this track
        existing_frames = list(dest_track_dir.glob(f"{track_name}_frame_*.jpg"))
        if existing_frames:
            print(f"  Skipping {track_name}, frames already exist in {dest_track_dir}")
            counts[mapped_label] = counts.get(mapped_label, 0) + 1
            continue    
        
        # Load video using VideoLoader
        print(f"  Loading video: {track_video_path}")
        video_loader = VideoLoader(str(track_video_path), verbose=False)
        
        if not video_loader.ok():
            print(f"  Warning: Failed to load video {track_video_path}")
            continue
        
        video_length = len(video_loader)
        
        # Determine which frames to extract
        if video_length <= max_frames:
            # Extract all frames
            frame_indices = list(range(video_length))
        else:
            # Sample random K frames
            frame_indices = sorted(np.random.choice(video_length, max_frames, replace=False))
        
        dest_track_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract and save frames
        extracted_count = 0
        for idx in frame_indices:
            try:
                frame = video_loader[idx]
                # Convert RGB to BGR for cv2.imwrite
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Save frame
                output_path = dest_track_dir / f"{track_name}_frame_{idx:06d}.jpg"
                cv2.imwrite(str(output_path), frame_bgr)
                extracted_count += 1
                
            except Exception as e:
                print(f"  Warning: Failed to extract frame {idx} from {track_name}: {e}")
                continue
        
        if extracted_count > 0:
            counts[mapped_label] = counts.get(mapped_label, 0) + 1
            print(f"  Extracted {extracted_count}/{len(frame_indices)} frames from {track_name} to {mapped_label}")
        else:
            # Clean up empty directory
            dest_track_dir.rmdir()
            print(f"  Warning: No frames extracted from {track_name}")
    
    return counts


def process_all_dates(
    dates_with_cams: Dict[str, List[str]],
    start_time: str = '15:30:00',
    end_time: str = '08:00:00',
    prediction_type: str = 'voted_track_label',
    max_frames: int = 100
) -> Dict[str, int]:
    """Process all dates and cameras to build active learning labeled pool.
    
    Args:
        dates_with_cams: Dictionary mapping dates to list of camera IDs
        start_time: Start time for filtering
        end_time: End time for filtering
        prediction_type: Prediction column to use ('voted_track_label' or 'stitched_id')
        max_frames: Maximum number of frames to extract per video
    
    Returns:
        Dictionary mapping identity labels to total count of tracks
    """
    total_counts = {}
    
    for date, cam_ids in dates_with_cams.items():
        print(f"\n{'='*80}")
        print(f"Processing date: {date}")
        print(f"{'='*80}")
        
        for cam_id in cam_ids:
            print(f"\nCamera: {cam_id}")
            
            try:
                df_correct = get_correct_predictions(
                    date=date,
                    cam_id=cam_id,
                    start_time=start_time,
                    end_time=end_time,
                    prediction_type=prediction_type
                )
                
                if not df_correct.empty:
                    counts = copy_correct_tracks_to_labeled_pool(
                        df_correct=df_correct,
                        output_dir=OUTPUT_DIR,
                        max_frames=100
                    )
                    
                    # Update total counts
                    for label, count in counts.items():
                        total_counts[label] = total_counts.get(label, 0) + count
                        
            except Exception as e:
                print(f"  Error processing {date}/{cam_id}: {e}")
                continue
    
    return total_counts


if __name__ == "__main__":
    
    # Dates with ground truth annotations
    data_with_gts = {
        '2025-11-15': ['016', '019'],
        '2025-11-30': ['017', '018'],
    }
    
    # Process all dates and copy correct predictions to labeled pool
    print("Starting active learning data collection...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Prediction type: voted_track_label")
    
    total_counts = process_all_dates(
        dates_with_cams=data_with_gts,
        start_time='15:30:00',
        end_time='08:00:00',
        prediction_type='voted_track_label',
        max_frames=100
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY: Active Learning Labeled Pool")
    print(f"{'='*80}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nTracks per identity:")
    for label in sorted(total_counts.keys()):
        print(f"  {label}: {total_counts[label]} tracks")
    print(f"\nTotal tracks: {sum(total_counts.values())}")
    print(f"{'='*80}\n")