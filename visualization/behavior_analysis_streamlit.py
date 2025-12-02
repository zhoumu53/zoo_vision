"""
Elephant Behavior Analysis Dashboard (Streamlit App)

Features:
- Multi-Camera Interactive Dashboard: View all 4 cameras with behavior timelines
- Room Occupancy Analysis: Track elephant presence in each room over time
- Camera Occupancy Analysis: Monitor individual camera activity
- Behavior Distribution: Analyze behavior patterns per individual
- Frame Inspector: Scrub through time to see annotated video frames
- Data Processing: Includes all smoothing and fixing logic from behavior_analysis.py

Usage:
    streamlit run behavior_analysis_streamlit.py
"""

import streamlit as st
import json
import logging
import copy
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# Configuration
# ==============================================================================

st.set_page_config(page_title="Elephant Behavior Analysis", layout="wide")

# Room configuration with ordered camera lists
ROOM_PAIRS = {
    "room1": ["016", "019"],  # Cameras in display order
    "room2": ["017", "018"],  # Cameras in display order
}

# Ordered list of rooms for consistent display
ROOM_ORDER = ["room1", "room2"]

CAMERA_TO_ROOM = {
    "016": "room1",
    "019": "room1",
    "017": "room2",
    "018": "room2",
}

BEHAVIOR_COLORS = {
    "standing": "#4CAF50",      # Green
    "walking": "#2196F3",       # Blue
    "eating": "#FF9800",        # Orange
    "sleeping": "#9C27B0",      # Purple
    "laying": "#E91E63",        # Pink
    "invalid": "#9E9E9E",       # Gray
    "unknown": "#757575",       # Dark Gray
}

# ==============================================================================
# Data Loading Functions (from behavior_analysis.py)
# ==============================================================================

def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string (YYYYMMDD_HHMMSS) to datetime."""
    try:
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except Exception:
        return datetime.now()


def extract_camera_id(video_path: str) -> str:
    """Extract camera ID from video path."""
    parts = video_path.split("/")
    for part in parts:
        if "CAM-" in part:
            return part.split("-")[-1]
    return "unknown"


def load_invalid_zones(invalid_zones_dir: Path, camera_id: str) -> Optional[List[np.ndarray]]:
    """Load invalid zone polygons for a camera.
    
    Args:
        invalid_zones_dir: Directory containing invalid zone JSON files
        camera_id: Camera ID (e.g., '016')
    
    Returns:
        List of polygon arrays (each is Nx2), or None if no invalid zones file exists
    """
    if not invalid_zones_dir:
        return None
    
    json_path = invalid_zones_dir / f"cam{camera_id}_invalid_zones.json"
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        polygons = []
        for poly in data.get("polygons", []):
            pts = np.array(poly["points"], dtype=np.int32)
            polygons.append(pts)
        
        return polygons if polygons else None
    
    except Exception as e:
        st.warning(f"Failed to load invalid zones for camera {camera_id}: {e}")
        return None


def bbox_in_invalid_zone(bbox: List[float], invalid_polygons: List[np.ndarray]) -> bool:
    """Check if a bounding box overlaps with any invalid zone polygon.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        invalid_polygons: List of polygon arrays
    
    Returns:
        True if bbox overlaps with any invalid zone, False otherwise
    """
    if not invalid_polygons or not bbox or len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    
    # Get bbox center point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_point = (int(center_x), int(center_y))
    
    # Check if center point is inside any invalid polygon
    for polygon in invalid_polygons:
        # Use cv2.pointPolygonTest: returns positive if inside, 0 if on edge, negative if outside
        result = cv2.pointPolygonTest(polygon, center_point, False)
        if result >= 0:  # Inside or on edge
            return True
    
    return False


@st.cache_data
def load_stitched_data(jsonl_path: Path, invalid_zones_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load stitched tracking results from JSONL file into a DataFrame.
    
    Args:
        jsonl_path: Path to stitched_tracks.jsonl
        invalid_zones_dir: Optional directory containing invalid zone JSON files
    
    Returns:
        DataFrame with columns: timestamp, camera_id, stitched_id, gallery_identity, 
                               gallery_score, behavior, bbox, frame_idx
    """
    records = []
    current_camera = None
    frame_idx = 0
    current_invalid_zones = None
    total_filtered = 0
    camera_filtered_counts = {}
    
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            
            # Meta line - indicates new video
            if "meta" in data:
                video_path = data["meta"].get("video", "")
                current_camera = extract_camera_id(video_path)
                frame_idx = 0
                
                # Load invalid zones for this camera
                if invalid_zones_dir:
                    current_invalid_zones = load_invalid_zones(invalid_zones_dir, current_camera)
                else:
                    current_invalid_zones = None
                
                camera_filtered_counts[current_camera] = 0
                
            # Results line
            elif "results" in data:
                timestamp = parse_timestamp(data["results"].get("timestamp", ""))
                tracks = data["results"].get("tracks", [])
                
                for track in tracks:
                    stitched_id = track.get("stitched_track_id", -1)
                    if stitched_id == -1:
                        continue
                    
                    # Filter tracks in invalid zones
                    bbox = track.get("bbox", [])
                    if current_invalid_zones and bbox_in_invalid_zone(bbox, current_invalid_zones):
                        camera_filtered_counts[current_camera] += 1
                        total_filtered += 1
                        continue  # Skip this track
                    
                    behavior_dict = track.get("behavior", {})
                    if isinstance(behavior_dict, dict):
                        behavior = behavior_dict.get("label", "unknown")
                    else:
                        behavior = "unknown"
                    
                    records.append({
                        "timestamp": timestamp,
                        "camera_id": current_camera,
                        "stitched_id": stitched_id,
                        "gallery_identity": track.get("gallery_identity"),
                        "gallery_score": track.get("gallery_score", 0.0),
                        "behavior": behavior,
                        "bbox": bbox,
                        "frame_idx": frame_idx,
                    })
                
                frame_idx += 1
    
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["camera_id", "timestamp"]).reset_index(drop=True)
    
    return df


def fix_single_track_identity_switches(
    df: pd.DataFrame,
    time_window_minutes: int = 5,
    min_spurious_duration_seconds: int = 30,
) -> pd.DataFrame:
    """Fix identity switches and remove spurious detections (vectorized).
    
    Handles two scenarios:
    1. Single detection frames: When only 1 detection exists, fix identity switches
    2. Spurious detections: When a track ID appears briefly alongside a dominant track
    """
    df = df.copy()
    
    # Track statistics per camera
    for camera_id in df['camera_id'].unique():
        cam_df = df[df['camera_id'] == camera_id].copy()
        
        # Calculate track durations
        track_stats = cam_df.groupby('stitched_id').agg({
            'timestamp': ['min', 'max', 'count']
        })
        track_stats.columns = ['first_time', 'last_time', 'num_frames']
        track_stats['duration'] = (track_stats['last_time'] - track_stats['first_time']).dt.total_seconds()
        
        # Find spurious tracks (short-lived overlapping with longer tracks)
        tracks_to_remove = set()
        
        for stitched_id, stats in track_stats.iterrows():
            if stats['duration'] < min_spurious_duration_seconds:
                # Check overlap with longer tracks
                for other_id, other_stats in track_stats.iterrows():
                    if other_id == stitched_id:
                        continue
                    
                    # Temporal overlap check
                    overlaps = (stats['first_time'] <= other_stats['last_time']) and \
                              (stats['last_time'] >= other_stats['first_time'])
                    
                    if overlaps and other_stats['duration'] > stats['duration'] * 3:
                        tracks_to_remove.add(stitched_id)
                        break
        
        # Remove spurious tracks
        if tracks_to_remove:
            df = df[~((df['camera_id'] == camera_id) & (df['stitched_id'].isin(tracks_to_remove)))]
        
        # Fix identity switches in single-detection frames
        cam_df = df[df['camera_id'] == camera_id].copy()
        
        # Count detections per frame
        frame_counts = cam_df.groupby('frame_idx').size()
        single_detection_frames = frame_counts[frame_counts == 1].index
        
        if len(single_detection_frames) > 0:
            single_df = cam_df[cam_df['frame_idx'].isin(single_detection_frames)].copy()
            single_df = single_df.sort_values('timestamp')
            
            # Group into temporal sequences
            single_df['time_gap'] = single_df['timestamp'].diff().dt.total_seconds()
            single_df['new_sequence'] = (single_df['time_gap'] > time_window_minutes * 60) | single_df['time_gap'].isna()
            single_df['sequence_id'] = single_df['new_sequence'].cumsum()
            
            # Fix identity switches within each sequence
            for seq_id, seq_group in single_df.groupby('sequence_id'):
                if len(seq_group) < 2:
                    continue
                
                # Find most common identity
                identities = seq_group['gallery_identity'].dropna()
                if len(identities) == 0:
                    continue
                
                most_common_identity = identities.mode()
                if len(most_common_identity) > 0:
                    most_common = most_common_identity.iloc[0]
                    
                    # Update all identities in this sequence to the most common
                    indices = seq_group.index
                    df.loc[indices, 'gallery_identity'] = most_common
    
    return df


def smooth_behavior_labels(
    df: pd.DataFrame,
    time_window_seconds: float = 1.5,
    invalid_window_seconds: float = 10.0,
    max_consecutive_frames: int = 3,
) -> pd.DataFrame:
    """Smooth behavior labels within time windows for the same track (vectorized).
    
    This function:
    1. Smooths 'invalid' labels with nearby valid behaviors
    2. Detects and smooths brief behavior spikes (≤max_consecutive_frames)
    
    Args:
        df: DataFrame with tracking data
        time_window_seconds: Time window for general smoothing (default: 1.5s)
        invalid_window_seconds: Larger window for invalid labels (default: 10.0s)
        max_consecutive_frames: Max consecutive frames to consider as spike (default: 3)
    """
    df = df.copy()
    df['behavior_original'] = df['behavior']
    
    # Process each camera-track combination
    for (camera_id, stitched_id), group in df.groupby(['camera_id', 'stitched_id']):
        if len(group) < 3:
            continue
        
        group_sorted = group.sort_values('timestamp')
        indices = group_sorted.index.tolist()
        
        for i, idx in enumerate(indices):
            current_behavior = df.loc[idx, 'behavior']
            current_time = df.loc[idx, 'timestamp']
            
            # Use larger window for 'invalid' labels and spike detection
            window = invalid_window_seconds if current_behavior == "invalid" else time_window_seconds
            spike_detection_window = invalid_window_seconds
            
            # Find neighbors within time window (vectorized)
            time_diffs = (group_sorted['timestamp'] - current_time).abs().dt.total_seconds()
            neighbors = group_sorted[(time_diffs <= window) & (time_diffs > 0)]
            
            # Get valid neighbor behaviors
            valid_behaviors = neighbors[~neighbors['behavior'].isin(['invalid', 'unknown'])]['behavior']
            
            # 1. Always smooth 'invalid' labels with valid neighbors
            if current_behavior == "invalid" and len(valid_behaviors) > 0:
                most_common = valid_behaviors.mode()
                if len(most_common) > 0:
                    df.loc[idx, 'behavior'] = most_common.iloc[0]
                continue
            
            # 2. Detect brief spikes (≤max_consecutive_frames with same behavior)
            # Count consecutive frames with current behavior
            consecutive_count = 1  # Current frame
            
            # Count backward
            for j in range(i - 1, -1, -1):
                prev_idx = indices[j]
                prev_time = df.loc[prev_idx, 'timestamp']
                if (current_time - prev_time).total_seconds() > spike_detection_window:
                    break
                if df.loc[prev_idx, 'behavior'] == current_behavior:
                    consecutive_count += 1
                else:
                    break
            
            # Count forward
            for j in range(i + 1, len(indices)):
                next_idx = indices[j]
                next_time = df.loc[next_idx, 'timestamp']
                if (next_time - current_time).total_seconds() > spike_detection_window:
                    break
                if df.loc[next_idx, 'behavior'] == current_behavior:
                    consecutive_count += 1
                else:
                    break
            
            # If it's a brief spike, check if neighbors agree on different behavior
            if consecutive_count <= max_consecutive_frames:
                # Get behaviors before the spike
                behaviors_before = []
                for j in range(i - 1, -1, -1):
                    prev_idx = indices[j]
                    prev_time = df.loc[prev_idx, 'timestamp']
                    if (current_time - prev_time).total_seconds() > spike_detection_window:
                        break
                    prev_behavior = df.loc[prev_idx, 'behavior']
                    if prev_behavior != current_behavior and prev_behavior != "invalid":
                        behaviors_before.append(prev_behavior)
                        if len(behaviors_before) >= 2:
                            break
                
                # Get behaviors after the spike
                behaviors_after = []
                for j in range(i + 1, len(indices)):
                    next_idx = indices[j]
                    next_time = df.loc[next_idx, 'timestamp']
                    if (next_time - current_time).total_seconds() > spike_detection_window:
                        break
                    next_behavior = df.loc[next_idx, 'behavior']
                    if next_behavior != current_behavior and next_behavior != "invalid":
                        behaviors_after.append(next_behavior)
                        if len(behaviors_after) >= 2:
                            break
                
                # If we have at least 2 neighbors on each side and they agree, smooth the spike
                if len(behaviors_before) >= 2 and len(behaviors_after) >= 2:
                    all_surrounding = behaviors_before + behaviors_after
                    most_common = max(set(all_surrounding), key=all_surrounding.count)
                    if all_surrounding.count(most_common) >= 3:  # Majority agreement
                        df.loc[idx, 'behavior'] = most_common
    
    return df


def remove_remaining_invalid_labels(
    df: pd.DataFrame,
    lookback_seconds: float = 60.0,
    default_behavior: str = "standing",
) -> pd.DataFrame:
    """Remove any remaining 'invalid' labels using track history and fallback.
    
    Strategy:
    1. Look back in the same track's history for the most recent valid behavior
    2. If no valid history exists, use the default behavior
    
    Args:
        df: DataFrame with tracking data
        lookback_seconds: How far back to look for valid behavior (default: 60s)
        default_behavior: Fallback behavior if no valid history (default: 'standing')
    """
    df = df.copy()
    
    # Process each camera-track combination
    for (camera_id, stitched_id), group in df.groupby(['camera_id', 'stitched_id']):
        group_sorted = group.sort_values('timestamp')
        invalid_mask = group_sorted['behavior'] == 'invalid'
        
        if not invalid_mask.any():
            continue
        
        # For each invalid entry, find replacement
        for idx in group_sorted[invalid_mask].index:
            current_time = df.loc[idx, 'timestamp']
            
            # Look back for most recent valid behavior
            earlier = group_sorted[
                (group_sorted['timestamp'] < current_time) &
                (group_sorted['timestamp'] >= current_time - pd.Timedelta(seconds=lookback_seconds)) &
                (~group_sorted['behavior'].isin(['invalid', 'unknown']))
            ]
            
            if len(earlier) > 0:
                # Use most recent valid behavior
                most_recent = earlier.iloc[-1]
                df.loc[idx, 'behavior'] = most_recent['behavior']
            else:
                # Look forward if no history
                later = group_sorted[
                    (group_sorted['timestamp'] > current_time) &
                    (group_sorted['timestamp'] <= current_time + pd.Timedelta(seconds=lookback_seconds)) &
                    (~group_sorted['behavior'].isin(['invalid', 'unknown']))
                ]
                
                if len(later) > 0:
                    # Use next valid behavior
                    next_valid = later.iloc[0]
                    df.loc[idx, 'behavior'] = next_valid['behavior']
                else:
                    # Use default as last resort
                    df.loc[idx, 'behavior'] = default_behavior
    
    return df


def fix_invalid_behaviors_cross_camera(
    df: pd.DataFrame,
    time_tolerance_seconds: float = 3.0,
) -> pd.DataFrame:
    """Fix invalid behaviors by checking paired cameras in the same room (vectorized)."""
    df = df.copy()
    
    # Add room information
    df['room'] = df['camera_id'].map(CAMERA_TO_ROOM)
    
    # Round timestamps to seconds for grouping
    df['timestamp_sec'] = df['timestamp'].dt.floor('1S')
    
    # Find invalid behaviors
    invalid_mask = df['behavior'] == 'invalid'
    invalid_rows = df[invalid_mask].copy()
    
    if len(invalid_rows) == 0:
        return df.drop(columns=['room', 'timestamp_sec'])
    
    total_fixed = 0
    
    # Group by room, timestamp, and stitched_id to find cross-camera matches
    for idx, row in invalid_rows.iterrows():
        room = row['room']
        stitched_id = row['stitched_id']
        timestamp = row['timestamp']
        
        # Find same elephant in same room at similar time
        time_window = pd.Timedelta(seconds=time_tolerance_seconds)
        same_elephant = df[
            (df['room'] == room) &
            (df['stitched_id'] == stitched_id) &
            (df['timestamp'] >= timestamp - time_window) &
            (df['timestamp'] <= timestamp + time_window) &
            (df['camera_id'] != row['camera_id']) &  # Different camera
            (~df['behavior'].isin(['invalid', 'unknown']))  # Valid behavior
        ]
        
        if len(same_elephant) > 0:
            # Use the most common valid behavior from paired camera
            valid_behavior = same_elephant['behavior'].mode()
            if len(valid_behavior) > 0:
                df.loc[idx, 'behavior'] = valid_behavior.iloc[0]
                total_fixed += 1
    
    return df.drop(columns=['room', 'timestamp_sec'])


def build_individual_timelines(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build timeline DataFrame for each identified individual (fast groupby)."""
    if df.empty:
        return {}
    
    # Add display_id column if it doesn't exist
    df = df.copy()
    if 'display_id' not in df.columns:
        df['display_id'] = df['gallery_identity'].fillna('ID_' + df['stitched_id'].astype(str))
    
    # Group by stitched_id and return dict of DataFrames
    timelines = {
        str(stitched_id): group.sort_values('timestamp').reset_index(drop=True)
        for stitched_id, group in df.groupby('stitched_id')
    }
    
    return timelines


# ==============================================================================
# Plotting Functions (Streamlit versions)
# ==============================================================================

def generate_camera_dashboard(
    camera_id: str,
    df: pd.DataFrame,
) -> go.Figure:
    """Generate interactive behavior timeline for a single camera."""
    
    # Filter data for this camera
    camera_df = df[df['camera_id'] == camera_id].copy()
    
    if camera_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Camera {camera_id} - No Data")
        return fig
    
    # Get room for this camera
    room = CAMERA_TO_ROOM.get(camera_id, "unknown")
    
    # Create elephant_id column
    camera_df['elephant_id'] = camera_df['gallery_identity'].fillna('Invalid')
    camera_df['track_key'] = camera_df['elephant_id'] + '_' + camera_df['stitched_id'].astype(str)
    
    # Sort by timestamp
    camera_df = camera_df.sort_values('timestamp')
    
    # Process each track to create segments
    segments = []
    
    for track_key, group in camera_df.groupby('track_key'):
        group = group.sort_values('timestamp')
        elephant_id = group['elephant_id'].iloc[0]
        
        # Identify segment boundaries (behavior change or time gap > 1.5s)
        group['time_diff'] = group['timestamp'].diff().dt.total_seconds()
        group['behavior_change'] = group['behavior'] != group['behavior'].shift()
        group['new_segment'] = (group['behavior_change']) | (group['time_diff'] > 1.5)
        group['segment_id'] = group['new_segment'].cumsum()
        
        # Create segments
        for seg_id, seg_group in group.groupby('segment_id'):
            start_time = seg_group['timestamp'].iloc[0]
            end_time = seg_group['timestamp'].iloc[-1] + pd.Timedelta(seconds=1)
            behavior = seg_group['behavior'].iloc[0]
            duration = (end_time - start_time).total_seconds()
            
            segments.append({
                "elephant_id": elephant_id,
                "start_time": start_time,
                "end_time": end_time,
                "behavior": behavior,
                "duration_seconds": duration,
            })
    
    if not segments:
        fig = go.Figure()
        fig.update_layout(title=f"Camera {camera_id} - No Segments")
        return fig
    
    segments_df = pd.DataFrame(segments)
    
    elephants_list = sorted(segments_df["elephant_id"].unique())
    title_text = f"Camera {camera_id} ({room.upper()}) - Elephants: {', '.join(elephants_list)}"
    
    fig = px.timeline(
        segments_df,
        x_start="start_time",
        x_end="end_time",
        y="elephant_id",
        color="behavior",
        color_discrete_map=BEHAVIOR_COLORS,
        title=title_text,
        labels={"elephant_id": "Elephant", "behavior": "Behavior"},
        hover_data=["duration_seconds"],
    )
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Time (HH:MM:SS)",
        yaxis_title="Elephant ID",
        legend_title_text="Behavior",
        height=400,
        hovermode="closest",
        xaxis=dict(tickformat="%H:%M:%S", type="date"),
    )
    
    return fig


def plot_room_occupancy(
    df: pd.DataFrame,
    df_original: pd.DataFrame,
) -> go.Figure:
    """Create room occupancy visualization (vectorized)."""
    
    # Add room information
    df = df.copy()
    df['room'] = df['camera_id'].map(CAMERA_TO_ROOM)
    df_original = df_original.copy()
    df_original['room'] = df_original['camera_id'].map(CAMERA_TO_ROOM)
    
    # Use ordered room list
    rooms = [room for room in ROOM_ORDER if room in df['room'].unique()]
    num_rooms = len(rooms)
    
    if num_rooms == 0:
        fig = go.Figure()
        fig.update_layout(title="No room data available")
        return fig
    
    fig = make_subplots(
        rows=num_rooms, cols=1,
        subplot_titles=[f'{room.upper()} - Cameras: {", ".join(ROOM_PAIRS.get(room, []))}' 
                       for room in rooms],
        vertical_spacing=0.12,
    )
    
    for idx, room in enumerate(rooms, start=1):
        room_df = df[df['room'] == room]
        room_df_orig = df_original[df_original['room'] == room]
        
        if room_df.empty:
            continue
        
        # Create 5-minute time bins
        min_time = room_df['timestamp'].min()
        max_time = room_df['timestamp'].max()
        time_bins = pd.date_range(min_time, max_time, freq='5min')
        
        # Vectorized binning
        room_df['time_bin'] = pd.cut(room_df['timestamp'], bins=time_bins, labels=time_bins[:-1], include_lowest=True)
        room_df_orig['time_bin'] = pd.cut(room_df_orig['timestamp'], bins=time_bins, labels=time_bins[:-1], include_lowest=True)
        
        # Count unique identities and track IDs per bin
        identity_counts = room_df[room_df['gallery_identity'].notna()].groupby('time_bin')['gallery_identity'].nunique()
        track_counts_orig = room_df_orig.groupby('time_bin')['stitched_id'].nunique()
        
        # Reindex to include all bins
        identity_counts = identity_counts.reindex(time_bins[:-1], fill_value=0)
        track_counts_orig = track_counts_orig.reindex(time_bins[:-1], fill_value=0)
        
        show_legend = (idx == 1)
        
        fig.add_trace(
            go.Scatter(
                x=identity_counts.index, y=identity_counts.values,
                mode='lines',
                name='Unique identities (gallery matched)',
                line=dict(color='#4CAF50', width=2, dash='dash'),
                showlegend=show_legend,
                legendgroup='identities',
            ),
            row=idx, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=track_counts_orig.index, y=track_counts_orig.values,
                mode='lines',
                name='Track IDs (original)',
                line=dict(color='#FF9800', width=1.5, dash='dot'),
                opacity=0.7,
                showlegend=show_legend,
                legendgroup='original',
            ),
            row=idx, col=1
        )
        
        fig.update_yaxes(title_text='Count', row=idx, col=1, rangemode='tozero')
    
    fig.update_xaxes(title_text='Time', row=num_rooms, col=1)
    fig.update_layout(
        title_text='Room Occupancy Over Time',
        height=400 * num_rooms,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    
    return fig


def plot_behavior_distribution(
    timelines: Dict[str, pd.DataFrame],
) -> go.Figure:
    """Create behavior distribution visualization for each identified elephant (vectorized)."""
    
    if not timelines:
        fig = go.Figure()
        fig.update_layout(title="No timeline data available")
        return fig
    
    # Combine all timelines and group by identity
    all_data = pd.concat(timelines.values(), ignore_index=True)
    
    # Determine primary identity for each stitched_id
    identity_map = (
        all_data[all_data['gallery_identity'].notna()]
        .groupby('stitched_id')['gallery_identity']
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    )
    
    # Add display_id
    all_data['display_id'] = all_data['stitched_id'].map(identity_map).fillna('ID_' + all_data['stitched_id'].astype(str))
    
    # Filter out invalid behaviors
    valid_data = all_data[all_data['behavior'] != 'invalid']
    
    # Count behaviors per identity
    behavior_counts = valid_data.groupby(['display_id', 'behavior']).size().reset_index(name='count')
    
    identities = sorted(behavior_counts['display_id'].unique())
    num_identities = len(identities)
    cols = min(3, num_identities)
    rows = (num_identities + cols - 1) // cols
    
    # Create subplot titles
    subplot_titles = []
    for identity in identities:
        total = behavior_counts[behavior_counts['display_id'] == identity]['count'].sum()
        subplot_titles.append(f'{identity}<br>({total} observations)')
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'pie'}] * cols for _ in range(rows)],
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.05,
    )
    
    for idx, identity in enumerate(identities):
        row = idx // cols + 1
        col = idx % cols + 1
        
        identity_data = behavior_counts[behavior_counts['display_id'] == identity]
        
        if identity_data.empty:
            continue
        
        behaviors = identity_data['behavior'].tolist()
        counts = identity_data['count'].tolist()
        colors = [BEHAVIOR_COLORS.get(b, "#000000") for b in behaviors]
        
        fig.add_trace(
            go.Pie(
                labels=behaviors,
                values=counts,
                marker=dict(colors=colors),
                textposition='inside',
                textinfo='label+percent',
                hovertemplate='%{label}<br>%{value} observations<br>%{percent}<extra></extra>',
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text='Behavior Distribution by Individual',
        height=400 * rows,
        showlegend=False,
    )
    
    return fig


def plot_camera_occupancy(
    df: pd.DataFrame,
    df_original: pd.DataFrame,
) -> go.Figure:
    """Create camera occupancy visualization (vectorized)."""
    
    # Order cameras by room: room1 (016, 019), room2 (017, 018)
    camera_order = []
    for room in ROOM_ORDER:
        camera_order.extend(ROOM_PAIRS[room])
    camera_ids = [cam for cam in camera_order if cam in df['camera_id'].unique()]
    num_cameras = len(camera_ids)
    
    if num_cameras == 0:
        return go.Figure()
    
    subplot_titles = [f'Camera {cam_id} ({CAMERA_TO_ROOM.get(cam_id, "unknown").upper()})' 
                      for cam_id in camera_ids]
    
    fig = make_subplots(
        rows=num_cameras, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )
    
    for idx, camera_id in enumerate(camera_ids, start=1):
        cam_df = df[df['camera_id'] == camera_id].copy()
        cam_df_orig = df_original[df_original['camera_id'] == camera_id].copy()
        
        if cam_df.empty or cam_df_orig.empty:
            continue
        
        # Create 5-minute time bins
        min_time = cam_df['timestamp'].min()
        max_time = cam_df['timestamp'].max()
        time_bins = pd.date_range(min_time, max_time, freq='5min')
        
        # Vectorized binning
        cam_df['time_bin'] = pd.cut(cam_df['timestamp'], bins=time_bins, labels=time_bins[:-1], include_lowest=True)
        cam_df_orig['time_bin'] = pd.cut(cam_df_orig['timestamp'], bins=time_bins, labels=time_bins[:-1], include_lowest=True)
        
        # Count metrics
        identity_counts = cam_df[cam_df['gallery_identity'].notna()].groupby('time_bin')['gallery_identity'].nunique()
        track_counts_orig = cam_df_orig.groupby('time_bin')['stitched_id'].nunique()
        
        # Reindex to include all bins
        identity_counts = identity_counts.reindex(time_bins[:-1], fill_value=0)
        track_counts_orig = track_counts_orig.reindex(time_bins[:-1], fill_value=0)
        
        show_legend = (idx == 1)
        
        fig.add_trace(
            go.Scatter(
                x=identity_counts.index, y=identity_counts.values,
                mode='lines',
                name='Unique identities (gallery matched)',
                line=dict(color='#4CAF50', width=2, dash='dash'),
                showlegend=show_legend,
                legendgroup='identities',
            ),
            row=idx, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=track_counts_orig.index, y=track_counts_orig.values,
                mode='lines',
                name='Track IDs (original)',
                line=dict(color='#FF9800', width=1.5, dash='dot'),
                opacity=0.7,
                showlegend=show_legend,
                legendgroup='original',
            ),
            row=idx, col=1
        )
        
        fig.update_yaxes(title_text='Count', row=idx, col=1, rangemode='tozero')
    
    fig.update_xaxes(title_text='Time', row=num_cameras, col=1)
    fig.update_layout(
        title_text='Camera Occupancy Over Time',
        height=350 * num_cameras,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    
    return fig


# ==============================================================================
# Frame Inspector Functions
# ==============================================================================

def get_annotated_frame(video_path: str, timestamp: datetime, tracks_now: List[Dict[str, Any]]) -> np.ndarray:
    """Generate annotated frame with bounding boxes."""
    
    # Generate synthetic frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(frame, f"Time: {timestamp.strftime('%H:%M:%S')}", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Annotate tracks
    for track in tracks_now:
        bbox = track.get("bbox", [])
        identity = track.get("display_id", "Unknown")
        behavior = track.get("behavior", "unknown")
        color_hex = BEHAVIOR_COLORS.get(behavior, "#FFFFFF")
        
        c = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
            
            label = f"{identity} | {behavior}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), c, -1)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


# ==============================================================================
# Main Streamlit App
# ==============================================================================


def scan_available_results(root_dir: Path) -> List[Tuple[str, str]]:
    """Scan root directory for available processed results.
    
    Args:
        root_dir: Root directory containing tracking results
    
    Returns:
        List of (date, hour) tuples for available results with JSONL files
    """
    available = []
    
    if not root_dir.exists():
        return available
    
    # Scan date directories (YYYYMMDD format)
    for date_dir in sorted(root_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        
        date_name = date_dir.name
        if not date_name.isdigit() or len(date_name) != 8:
            continue
        
        # Scan hour directories (YYYYMMDD_HH format)
        for hour_dir in sorted(date_dir.iterdir()):
            if not hour_dir.is_dir():
                continue
            
            # Check if JSONL file exists
            jsonl_path = hour_dir / "stitched_tracks" / "stitched_tracks.jsonl"
            if jsonl_path.exists():
                # Extract hour from directory name (YYYYMMDD_HH)
                dir_parts = hour_dir.name.split('_')
                if len(dir_parts) == 2 and dir_parts[1].isdigit():
                    hour = dir_parts[1]
                    available.append((date_name, hour))
    
    return available


def get_result_files_from_datetime(
        root_dir: Path = Path("/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams"), 
        date: str = "20250729", 
        hour: str = "18") -> Tuple[Path, Path]:
    """Get JSONL file and timeline CSV path based on date and hour inputs.
    
    Args:
        root_dir: Root directory containing tracking results
        date: Date string in YYYYMMDD format
        hour: Hour string in HH format (00-23)
    
    Returns:
        Tuple of (result_json_path, timeline_csv_path)
        
    Example:
        result_json, timeline_csv = get_result_files_from_datetime(
            root_dir=Path("/data/tracking"),
            date="20250729",
            hour="18"
        )
    """
    base_dir = root_dir / f"{date}" / f"{date}_{hour}"
    result_json = base_dir / "stitched_tracks" / "stitched_tracks.jsonl"
    timeline_csv = base_dir / "behavior_analysis" / "timeline_data.csv"
    return result_json, timeline_csv


def post_processing(
        result_json: Path, 
        timeline_csv: Path,
        time_window_seconds: float = 1.5,
        invalid_window_seconds: float = 10.0,
        max_consecutive_frames: int = 3,
        time_tolerance_seconds: float = 3.0,
        lookback_seconds: float = 60.0,
        verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Perform post-processing on the given JSONL file and save timeline CSV.
    
    This function handles the complete processing pipeline:
    1. Load data from JSONL (or cached CSV if exists)
    2. Fix identity switches
    3. Smooth behavior labels
    4. Fix invalid behaviors using cross-camera data
    5. Remove remaining invalid labels
    6. Save processed data to CSV
    
    Args:
        result_json: Path to stitched_tracks.jsonl file
        timeline_csv: Path where timeline CSV should be saved
        time_window_seconds: Time window for behavior smoothing (default: 1.5s)
        invalid_window_seconds: Larger window for invalid labels (default: 10.0s)
        max_consecutive_frames: Max consecutive frames to consider as spike (default: 3)
        time_tolerance_seconds: Time tolerance for cross-camera matching (default: 3.0s)
        lookback_seconds: How far to look back for valid behavior (default: 60.0s)
        verbose: Whether to print progress messages (default: True)
    
    Returns:
        Tuple of (processed DataFrame, timing statistics dict)
    """
    timing_stats = {}
    start_time = time.time()
    
    # Check if CSV file exists - if so, skip processing and load directly
    if timeline_csv.exists():
        print(f"Loading cached timeline CSV: {timeline_csv}")
        load_start = time.time()
        if verbose:
            print(f"✓ Loading cached data from {timeline_csv.name}")
        df = pd.read_csv(timeline_csv, parse_dates=['timestamp'])
        
        # Ensure all required columns exist
        required_cols = ['timestamp', 'camera_id', 'stitched_id', 'behavior']
        if not all(col in df.columns for col in required_cols):
            if verbose:
                print("WARNING: Cached CSV missing required columns. Reprocessing...")
        else:
            timing_stats['load_cached'] = time.time() - load_start
            timing_stats['total'] = time.time() - start_time
            if verbose:
                print(f"✓ Loaded in {timing_stats['load_cached']:.2f}s")
            return df, timing_stats
    
    if verbose:
        print("Processing raw tracking data...")
    
    # Ensure output directory exists
    timeline_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    load_start = time.time()
    df = load_stitched_data(result_json)
    timing_stats['load_jsonl'] = time.time() - load_start
    
    if df.empty:
        if verbose:
            print("ERROR: No data loaded from file.")
        return pd.DataFrame(), timing_stats
    
    if verbose:
        print(f"✓ Loaded {len(df)} records in {timing_stats['load_jsonl']:.2f}s")
        print("Applying corrections...")
    
    # 1. Fix single-track identity switches
    step_start = time.time()
    df = fix_single_track_identity_switches(df)
    timing_stats['fix_identity_switches'] = time.time() - step_start
    if verbose:
        print(f"  - Identity switches: {timing_stats['fix_identity_switches']:.2f}s")
    
    # 2. Smooth behavior labels (detect and fix brief spikes)
    step_start = time.time()
    df = smooth_behavior_labels(
        df,
        time_window_seconds=time_window_seconds,
        invalid_window_seconds=invalid_window_seconds,
        max_consecutive_frames=max_consecutive_frames
    )
    timing_stats['smooth_behaviors'] = time.time() - step_start
    if verbose:
        print(f"  - Smooth behaviors: {timing_stats['smooth_behaviors']:.2f}s")
    
    # 3. Fix invalid behaviors using cross-camera information
    step_start = time.time()
    df = fix_invalid_behaviors_cross_camera(
        df, 
        time_tolerance_seconds=time_tolerance_seconds
    )
    timing_stats['cross_camera_fix'] = time.time() - step_start
    if verbose:
        print(f"  - Cross-camera fix: {timing_stats['cross_camera_fix']:.2f}s")
    
    # 4. Remove any remaining invalid labels (final cleanup)
    step_start = time.time()
    df = remove_remaining_invalid_labels(
        df, 
        lookback_seconds=lookback_seconds
    )
    timing_stats['remove_invalid'] = time.time() - step_start
    if verbose:
        print(f"  - Remove invalid: {timing_stats['remove_invalid']:.2f}s")
    
    # Save processed timeline data as CSV
    save_start = time.time()
    export_df = df.copy()
    export_df['display_id'] = export_df['gallery_identity'].fillna(
        'ID_' + export_df['stitched_id'].astype(str)
    )
    # Select only the columns we need for export
    export_cols = ['timestamp', 'camera_id', 'stitched_id', 'gallery_identity', 
                   'gallery_score', 'behavior', 'bbox', 'frame_idx', 'display_id']
    export_df = export_df[export_cols]
    export_df.to_csv(timeline_csv, index=False)
    timing_stats['save_csv'] = time.time() - save_start
    
    timing_stats['total'] = time.time() - start_time
    
    if verbose:
        print(f"✓ Saved processed data to {timeline_csv.name}")
        print(f"✓ Total processing time: {timing_stats['total']:.2f}s")
    
    return df, timing_stats



def app(date: str = "20250729", hour: str = "22", root_dir: Optional[Path] = None, **kwargs):
    """Main Streamlit app for elephant behavior analysis.
    
    Args:
        date: Date string in YYYYMMDD format
        hour: Hour string in HH format (00-23)
        root_dir: Root directory for tracking results
        **kwargs: Additional configuration options
    """
    st.title("🐘 Elephant Behavior Analysis Dashboard")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Root directory input
    if root_dir is None:
        root_dir = Path("/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams")
    
    root_dir_input = st.sidebar.text_input("Root directory:", str(root_dir))
    root_dir = Path(root_dir_input)
    
    # Scan for available results
    available_results = scan_available_results(root_dir)
    
    if not available_results:
        st.error(f"No processed results found in {root_dir}")
        st.info("Please ensure the directory contains subdirectories in format: YYYYMMDD/YYYYMMDD_HH/stitched_tracks/stitched_tracks.jsonl")
        return
    
    # Date/Time selection
    st.sidebar.subheader("Analysis Period")
    
    # Get unique dates and create mapping
    dates_available = sorted(set(d for d, h in available_results))
    date_options = {d: f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in dates_available}
    
    # Find default date index
    default_date_idx = 0
    if date in dates_available:
        default_date_idx = dates_available.index(date)
    
    selected_date_display = st.sidebar.selectbox(
        "Date:",
        options=list(date_options.values()),
        index=default_date_idx
    )
    
    # Get actual date value (YYYYMMDD)
    date_input = [k for k, v in date_options.items() if v == selected_date_display][0]
    
    # Get available hours for selected date
    hours_available = sorted([h for d, h in available_results if d == date_input])
    hour_options = {h: f"{h}:00" for h in hours_available}
    
    # Find default hour index
    default_hour_idx = 0
    if hour in hours_available:
        default_hour_idx = hours_available.index(hour)
    
    selected_hour_display = st.sidebar.selectbox(
        "Hour:",
        options=list(hour_options.values()),
        index=default_hour_idx
    )
    
    # Get actual hour value (HH)
    hour_input = [k for k, v in hour_options.items() if v == selected_hour_display][0]
    
    # Get file paths from date/time
    try:
        result_json, timeline_csv = get_result_files_from_datetime(
            root_dir=root_dir,
            date=date_input,
            hour=hour_input
        )
    except Exception as e:
        st.error(f"Error getting file paths: {e}")
        return
    
    # Display file paths
    st.sidebar.info(f"JSONL: {result_json.name}")
    st.sidebar.info(f"CSV: {timeline_csv.name}")
    
    # Check if JSONL exists
    if not result_json.exists():
        st.error(f"Data file not found: {result_json}")
        return
    
    # Load and process data
    with st.spinner("Loading and processing tracking data..."):
        try:
            # Use post_processing function which handles caching and pipeline
            df, timing_stats = post_processing(result_json, timeline_csv, verbose=False)
            
            if df is None or df.empty:
                st.error("No data loaded from file.")
                return
            
            # Show timing statistics in sidebar
            st.sidebar.success("✓ Data processing complete")
            with st.sidebar.expander("⏱️ Processing Timings", expanded=False):
                if 'load_cached' in timing_stats:
                    st.metric("Cache Load", f"{timing_stats['load_cached']:.2f}s")
                else:
                    if 'load_jsonl' in timing_stats:
                        st.metric("Load JSONL", f"{timing_stats['load_jsonl']:.2f}s")
                    if 'fix_identity_switches' in timing_stats:
                        st.metric("Fix Identity", f"{timing_stats['fix_identity_switches']:.2f}s")
                    if 'smooth_behaviors' in timing_stats:
                        st.metric("Smooth Behaviors", f"{timing_stats['smooth_behaviors']:.2f}s")
                    if 'cross_camera_fix' in timing_stats:
                        st.metric("Cross-Camera Fix", f"{timing_stats['cross_camera_fix']:.2f}s")
                    if 'remove_invalid' in timing_stats:
                        st.metric("Remove Invalid", f"{timing_stats['remove_invalid']:.2f}s")
                    if 'save_csv' in timing_stats:
                        st.metric("Save CSV", f"{timing_stats['save_csv']:.2f}s")
                
                st.metric("**Total Time**", f"{timing_stats.get('total', 0):.2f}s", 
                         help="Total processing time including all steps")
            
            # Load original data for comparison
            df_original = load_stitched_data(result_json)
            
            # Build timelines
            timelines = build_individual_timelines(df)
            
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return
    
    st.sidebar.success(f"✓ Loaded {df['camera_id'].nunique()} cameras, {len(timelines)} individuals")
    st.sidebar.success(f"✓ Data saved to {timeline_csv.parent.name}/")
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📹 Camera Dashboards", 
        "🏠 Room Occupancy", 
        "📊 Behavior Distribution",
        "📷 Camera Occupancy",
        "🔍 Frame Inspector"
    ])
    
    # Tab 1: Camera Dashboards
    with tab1:
        st.header("Camera Behavior Timelines")
        # Order cameras by room: room1 (016, 019), room2 (017, 018)
        camera_order = []
        for room in ROOM_ORDER:
            camera_order.extend(ROOM_PAIRS[room])
        camera_ids = [cam for cam in camera_order if cam in df['camera_id'].unique()]
        
        for idx, camera_id in enumerate(camera_ids):
            with st.container():
                fig = generate_camera_dashboard(camera_id, df)
                st.plotly_chart(fig, width='stretch')
                
                # Add divider after room1 cameras (016, 019) before room2 cameras
                if camera_id == "019":
                    st.markdown("---")  # Horizontal divider between rooms
                elif camera_id != camera_ids[-1]:  # Not the last camera
                    st.divider()
    
    # Tab 2: Room Occupancy
    with tab2:
        st.header("Room Occupancy Analysis")
        fig = plot_room_occupancy(df, df_original)
        st.plotly_chart(fig, width='stretch')
    
    # Tab 3: Behavior Distribution
    with tab3:
        st.header("Behavior Distribution by Individual")
        fig = plot_behavior_distribution(timelines)
        st.plotly_chart(fig, width='stretch')
    
    # Tab 4: Camera Occupancy
    with tab4:
        st.header("Camera Occupancy Analysis")
        fig = plot_camera_occupancy(df, df_original)
        st.plotly_chart(fig, width='stretch')
    
    # Tab 5: Frame Inspector
    with tab5:
        st.header("Frame Inspector")
        
        # Camera selector (ordered by room)
        camera_order = []
        for room in ROOM_ORDER:
            camera_order.extend(ROOM_PAIRS[room])
        cameras = [cam for cam in camera_order if cam in df['camera_id'].unique()]
        selected_cam = st.selectbox("Select Camera", cameras)
        
        # Get camera data
        cam_df = df[df['camera_id'] == selected_cam].copy()
        
        if not cam_df.empty:
            # Add display_id
            cam_df['display_id'] = cam_df['gallery_identity'].fillna('ID_' + cam_df['stitched_id'].astype(str))
            
            # Get unique timestamps
            timestamps = sorted(cam_df['timestamp'].unique())
            
            if timestamps:
                min_time = timestamps[0].to_pydatetime() if isinstance(timestamps[0], pd.Timestamp) else timestamps[0]
                max_time = timestamps[-1].to_pydatetime() if isinstance(timestamps[-1], pd.Timestamp) else timestamps[-1]
                
                selected_time = st.slider(
                    "Select Time",
                    min_value=min_time,
                    max_value=max_time,
                    value=min_time,
                    format="HH:mm:ss"
                )
                
                # Get observations at selected time (within 1 second)
                time_mask = (cam_df['timestamp'] >= selected_time - pd.Timedelta(seconds=1)) & \
                           (cam_df['timestamp'] <= selected_time + pd.Timedelta(seconds=1))
                tracks_now = cam_df[time_mask].to_dict('records')
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    frame_img = get_annotated_frame("", selected_time, tracks_now)
                    st.image(frame_img, channels="BGR", 
                            caption=f"Time: {selected_time.strftime('%H:%M:%S')} | Elephants detected: {len(tracks_now)}")
                
                with col2:
                    st.write(f"**Behaviors at {selected_time.strftime('%H:%M:%S')}**")
                    if tracks_now:
                        for track in tracks_now:
                            color = BEHAVIOR_COLORS.get(track["behavior"], "#333")
                            st.markdown(
                                f'''
                                <div style="padding:10px; border-radius:5px; background-color:{color}; color:white; margin-bottom:5px;">
                                    <b>{track['display_id']}</b>: {track['behavior'].upper()}
                                </div>
                                ''', 
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("No tracking data for this timestamp.")


if __name__ == "__main__":
    # Default parameters - can be overridden via command line or environment
    app(date="20250729", hour="22")
