"""
Behavior Analysis for Stitched Tracking Results

This script analyzes elephant behavior per room and per individual from stitched tracking results.
Designed to help zookeepers quickly understand:
  - Which elephants are in each room
  - What behaviors they're exhibiting throughout the day
  - Activity patterns and social interactions
  - Time spent in different behaviors
  - Movement between cameras within a room

Usage:
    python behavior_analysis.py --input stitched_tracks.jsonl --output-dir analysis_output
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, HourLocator

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    px = None
    go = None

# Room configuration
ROOM_PAIRS = {
    "room1": ["016", "019"],
    "room2": ["017", "018"],
}

CAMERA_TO_ROOM = {
    "016": "room1",
    "019": "room1",
    "017": "room2",
    "018": "room2",
}

# Behavior color mapping for visualization
BEHAVIOR_COLORS = {
    "standing": "#4CAF50",      # Green
    "walking": "#2196F3",       # Blue
    "eating": "#FF9800",        # Orange
    "sleeping": "#9C27B0",      # Purple
    "laying": "#E91E63",        # Pink
    "invalid": "#9E9E9E",       # Gray
    "unknown": "#757575",       # Dark Gray
}

# Identity colors (consistent with visualization)
IDENTITY_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#16A085", "#C0392B"
]


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("behavior_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze elephant behavior from stitched tracking results"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to stitched_tracks.jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for analysis results (default: same as input dir)",
    )
    parser.add_argument(
        "--invalid-zones-dir",
        default=None,
        help="Directory containing invalid zone JSON files (e.g., cam016_invalid_zones.json)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity",
    )
    return parser.parse_args()


# ==============================================================================
# Data Loading and Parsing
# ==============================================================================


def load_invalid_zones(invalid_zones_dir: Path, camera_id: str, logger: logging.Logger) -> Optional[List[np.ndarray]]:
    """Load invalid zone polygons for a camera.
    
    Args:
        invalid_zones_dir: Directory containing invalid zone JSON files
        camera_id: Camera ID (e.g., '016')
        logger: Logger instance
    
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
        
        if polygons:
            logger.debug(f"Loaded {len(polygons)} invalid zones for camera {camera_id}")
        return polygons if polygons else None
    
    except Exception as e:
        logger.warning(f"Failed to load invalid zones for camera {camera_id}: {e}")
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


def load_stitched_data(
    jsonl_path: Path, 
    logger: logging.Logger,
    invalid_zones_dir: Optional[Path] = None,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Load stitched tracking results from JSONL file and filter detections in invalid zones.
    
    Args:
        jsonl_path: Path to stitched_tracks.jsonl
        logger: Logger instance
        invalid_zones_dir: Optional directory containing invalid zone JSON files
    
    Returns:
        List of (camera_id, frames) tuples with invalid detections filtered out
    """
    video_data = []
    current_camera = None
    current_frames = []
    current_invalid_zones = None
    
    total_filtered = 0
    camera_filtered_counts = {}
    
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if "meta" in record:
                # New video section - save previous if exists
                if current_camera and current_frames:
                    video_data.append((current_camera, current_frames))
                
                # Extract camera from new metadata
                video_path = record["meta"].get("video", "")
                current_camera = extract_camera_id(video_path)
                current_frames = []
                
                # Load invalid zones for this camera
                if invalid_zones_dir:
                    current_invalid_zones = load_invalid_zones(invalid_zones_dir, current_camera, logger)
                    if current_invalid_zones:
                        logger.info(f"  Loaded {len(current_invalid_zones)} invalid zones for camera {current_camera}")
                else:
                    current_invalid_zones = None
                
                camera_filtered_counts[current_camera] = 0
            
            elif "results" in record:
                if current_camera:
                    frame_data = record["results"]
                    
                    # Filter tracks in invalid zones if needed
                    if current_invalid_zones and "tracks" in frame_data:
                        original_count = len(frame_data["tracks"])
                        filtered_tracks = []
                        
                        for track in frame_data["tracks"]:
                            bbox = track.get("bbox")
                            if not bbox_in_invalid_zone(bbox, current_invalid_zones):
                                filtered_tracks.append(track)
                            else:
                                camera_filtered_counts[current_camera] += 1
                                total_filtered += 1
                        
                        frame_data["tracks"] = filtered_tracks
                    
                    current_frames.append(frame_data)
    
    # Save last video
    if current_camera and current_frames:
        video_data.append((current_camera, current_frames))
    
    total_frames = sum(len(frames) for _, frames in video_data)
    logger.info(f"Loaded {total_frames} frames from {len(video_data)} cameras")
    for camera, frames in video_data:
        logger.info(f"  Camera {camera}: {len(frames)} frames")
    
    # Report filtering statistics
    if total_filtered > 0:
        logger.info(f"Filtered {total_filtered} detections in invalid zones:")
        for camera, count in sorted(camera_filtered_counts.items()):
            if count > 0:
                logger.info(f"  Camera {camera}: {count} detections filtered")
    
    return video_data


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string (YYYYMMDD_HHMMSS) to datetime."""
    try:
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except Exception:
        # Fallback for other formats
        try:
            return datetime.fromisoformat(ts_str)
        except Exception:
            return datetime.now()


def extract_camera_id(video_path: str) -> str:
    """Extract camera ID from video path."""
    # Example: /path/ZAG-ELP-CAM-016/... -> 016
    parts = video_path.split("/")
    for part in parts:
        if "CAM-" in part:
            return part.split("-")[-1]
    return "unknown"


# ==============================================================================
# Data Processing
# ==============================================================================


def build_individual_timelines(
    video_data: List[Tuple[str, List[Dict[str, Any]]]], 
    logger: logging.Logger
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build timeline for each identified individual.
    
    Returns:
        Dict mapping stitched_track_id -> list of observations sorted by time
    """
    timelines = defaultdict(list)
    
    for camera_id, frames in video_data:
        for frame in frames:
            timestamp = parse_timestamp(frame.get("timestamp", ""))
            tracks = frame.get("tracks", [])
            
            for track in tracks:
                stitched_id = track.get("stitched_track_id", -1)
                if stitched_id == -1:
                    continue  # Skip unstitched tracks
                
                observation = {
                    "timestamp": timestamp,
                    "stitched_id": stitched_id,
                    "gallery_identity": track.get("gallery_identity"),
                    "gallery_score": track.get("gallery_score"),
                    "behavior_label": track.get("behavior", {}).get("label", "unknown"),
                    "behavior_prob": track.get("behavior", {}).get("prob", 0.0),
                    "bbox": track.get("bbox"),
                    "camera_id": camera_id,
                    "frame_idx": frame.get("frame_idx", 0),
                }
                
                timelines[stitched_id].append(observation)
    
    # Sort each timeline by timestamp
    for stitched_id in timelines:
        timelines[stitched_id].sort(key=lambda x: x["timestamp"])
    
    logger.info(f"Built timelines for {len(timelines)} individuals")
    return dict(timelines)


def compute_behavior_statistics(
    timeline: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute behavior statistics for a single individual's timeline."""
    if not timeline:
        return {}
    
    # Count behavior occurrences
    behavior_counts = defaultdict(int)
    for obs in timeline:
        behavior = obs["behavior_label"]
        behavior_counts[behavior] += 1
    
    total = len(timeline)
    
    # Calculate percentages
    behavior_percentages = {
        behavior: (count / total) * 100
        for behavior, count in behavior_counts.items()
    }
    
    # Time span
    start_time = timeline[0]["timestamp"]
    end_time = timeline[-1]["timestamp"]
    duration = end_time - start_time
    
    # Camera coverage
    cameras = set(obs.get("camera_id", "unknown") for obs in timeline)
    
    # Identity information
    identities = [obs.get("gallery_identity") for obs in timeline if obs.get("gallery_identity")]
    most_common_identity = max(set(identities), key=identities.count) if identities else "Unknown"
    
    identity_confidence = np.mean([
        obs.get("gallery_score", 0.0) 
        for obs in timeline 
        if obs.get("gallery_score")
    ]) if any(obs.get("gallery_score") for obs in timeline) else 0.0
    
    return {
        "stitched_id": timeline[0]["stitched_id"],
        "identity": most_common_identity,
        "identity_confidence": identity_confidence,
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": duration.total_seconds() / 60,
        "num_observations": total,
        "cameras": sorted(cameras),
        "behavior_counts": dict(behavior_counts),
        "behavior_percentages": behavior_percentages,
        "dominant_behavior": max(behavior_counts.items(), key=lambda x: x[1])[0],
    }


def group_by_room(
    timelines: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group individual timelines by room."""
    room_timelines = defaultdict(lambda: defaultdict(list))
    
    for stitched_id, timeline in timelines.items():
        # Determine which room(s) this individual appears in
        cameras = set(obs.get("camera_id", "unknown") for obs in timeline)
        
        for camera in cameras:
            room = CAMERA_TO_ROOM.get(camera, "unknown")
            # Filter timeline to only this room's cameras
            room_timeline = [
                obs for obs in timeline 
                if obs.get("camera_id") == camera or CAMERA_TO_ROOM.get(obs.get("camera_id")) == room
            ]
            if room_timeline:
                room_timelines[room][stitched_id].extend(room_timeline)
    
    # Sort each room timeline
    for room in room_timelines:
        for stitched_id in room_timelines[room]:
            room_timelines[room][stitched_id].sort(key=lambda x: x["timestamp"])
    
    return dict(room_timelines)


# ==============================================================================
# Analysis Output
# ==============================================================================


def write_summary_report(
    timelines: Dict[str, List[Dict[str, Any]]],
    room_timelines: Dict[str, Dict[str, List[Dict[str, Any]]]],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Write human-readable summary report for zookeepers."""
    
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ELEPHANT BEHAVIOR ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total unique individuals tracked: {len(timelines)}\n")
        
        # Identify named elephants
        named_elephants = {}
        for stitched_id, timeline in timelines.items():
            identities = [obs.get("gallery_identity") for obs in timeline if obs.get("gallery_identity")]
            if identities:
                most_common = max(set(identities), key=identities.count)
                named_elephants[stitched_id] = most_common
        
        f.write(f"Identified elephants: {len(named_elephants)}\n")
        for stitched_id, name in sorted(named_elephants.items(), key=lambda x: x[1]):
            f.write(f"  - ID {stitched_id}: {name}\n")
        
        f.write(f"\nRooms with activity: {len(room_timelines)}\n")
        f.write("\n\n")
        
        # Per-room analysis
        for room, room_individuals in sorted(room_timelines.items()):
            f.write("=" * 80 + "\n")
            f.write(f"ROOM: {room.upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Cameras: {', '.join(ROOM_PAIRS.get(room, []))}\n")
            f.write(f"Individuals present: {len(room_individuals)}\n\n")
            
            # Per-individual analysis in this room
            for stitched_id, timeline in sorted(room_individuals.items()):
                stats = compute_behavior_statistics(timeline)
                
                f.write("-" * 80 + "\n")
                f.write(f"Individual ID: {stitched_id}\n")
                f.write(f"Identity: {stats['identity']}")
                if stats['identity_confidence'] > 0:
                    f.write(f" (confidence: {stats['identity_confidence']:.2%})\n")
                else:
                    f.write("\n")
                
                f.write(f"Time range: {stats['start_time'].strftime('%H:%M:%S')} - {stats['end_time'].strftime('%H:%M:%S')}\n")
                f.write(f"Duration: {stats['duration_minutes']:.1f} minutes\n")
                f.write(f"Observations: {stats['num_observations']} frames\n")
                f.write(f"Cameras used: {', '.join(stats['cameras'])}\n")
                f.write(f"Dominant behavior: {stats['dominant_behavior']}\n\n")
                
                f.write("Behavior breakdown:\n")
                for behavior, percentage in sorted(stats['behavior_percentages'].items(), key=lambda x: x[1], reverse=True):
                    count = stats['behavior_counts'][behavior]
                    f.write(f"  {behavior:15s}: {percentage:5.1f}% ({count:5d} frames)\n")
                
                f.write("\n")
            
            f.write("\n")
        
        # Activity summary by time
        f.write("=" * 80 + "\n")
        f.write("TEMPORAL ACTIVITY SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Build hourly activity summary
        hourly_activity = defaultdict(lambda: defaultdict(set))
        for stitched_id, timeline in timelines.items():
            identity = named_elephants.get(stitched_id, f"ID {stitched_id}")
            for obs in timeline:
                hour = obs["timestamp"].hour
                room = CAMERA_TO_ROOM.get(obs.get("camera_id", ""), "unknown")
                hourly_activity[hour][room].add(identity)
        
        for hour in sorted(hourly_activity.keys()):
            f.write(f"{hour:02d}:00 - {hour:02d}:59\n")
            for room, individuals in sorted(hourly_activity[hour].items()):
                if individuals:
                    f.write(f"  {room}: {', '.join(sorted(individuals))}\n")
            f.write("\n")
    
    logger.info(f"Wrote summary report to {output_path}")


def export_csv_data(
    timelines: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Export detailed timeline data to CSV for further analysis."""
    
    rows = []
    for stitched_id, timeline in timelines.items():
        for obs in timeline:
            rows.append({
                "timestamp": obs["timestamp"].isoformat(),
                "stitched_id": stitched_id,
                "gallery_identity": obs.get("gallery_identity", "Unknown"),
                "gallery_score": obs.get("gallery_score", 0.0),
                "behavior_label": obs["behavior_label"],
                "behavior_prob": obs["behavior_prob"],
                "camera_id": obs.get("camera_id", "unknown"),
                "room": CAMERA_TO_ROOM.get(obs.get("camera_id", ""), "unknown"),
                "frame_idx": obs.get("frame_idx", 0),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(rows)} observations to {output_path}")


# ==============================================================================
# Visualization
# ==============================================================================


def generate_camera_dashboard(
    camera_id: str,
    video_data: List[Tuple[str, List[Dict[str, Any]]]],
    logger: logging.Logger,
) -> go.Figure:
    """Generate interactive behavior timeline for a single camera."""
    
    if px is None or go is None:
        raise ImportError("Plotly is required for dashboards. Install with: pip install plotly")
    
    # Find data for this camera
    camera_frames = None
    for cam_id, frames in video_data:
        if cam_id == camera_id:
            camera_frames = frames
            break
    
    if not camera_frames:
        # Empty figure
        fig = go.Figure()
        fig.update_layout(title=f"Camera {camera_id} - No Data")
        return fig
    
    # Get room for this camera
    room = CAMERA_TO_ROOM.get(camera_id, "unknown")
    
    # Build timeline data for this camera - only include elephants detected in this camera's room
    timeline_records = []
    elephants_in_room = set()  # Track which elephants appear in this room
    
    for frame in camera_frames:
        timestamp = parse_timestamp(frame.get("timestamp", ""))
        tracks = frame.get("tracks", [])
        
        for track in tracks:
            stitched_id = track.get("stitched_track_id", -1)
            if stitched_id == -1:
                continue
            
            gallery_identity = track.get("gallery_identity", f"ID {stitched_id}")
            behavior_label = track.get("behavior", {}).get("label", "unknown")
            
            elephants_in_room.add(gallery_identity)
            
            timeline_records.append({
                "timestamp": timestamp,
                "elephant_id": gallery_identity,
                "behavior": behavior_label,
            })
    
    if not timeline_records:
        fig = go.Figure()
        fig.update_layout(title=f"Camera {camera_id} - No Tracks")
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(timeline_records)
    
    # Compute behavior segments (consecutive same behavior)
    segments = []
    df_sorted = df.sort_values(by=["elephant_id", "timestamp"])
    
    for elephant_id in df_sorted["elephant_id"].unique():
        elephant_data = df_sorted[df_sorted["elephant_id"] == elephant_id].copy()
        elephant_data = elephant_data.sort_values("timestamp")
        
        current_behavior = None
        segment_start = None
        
        for _, row in elephant_data.iterrows():
            behavior = row["behavior"]
            timestamp = row["timestamp"]
            
            if behavior != current_behavior:
                # End previous segment
                if current_behavior is not None:
                    segments.append({
                        "elephant_id": elephant_id,
                        "start_time": segment_start,
                        "end_time": timestamp,
                        "behavior": current_behavior,
                    })
                
                # Start new segment
                current_behavior = behavior
                segment_start = timestamp
        
        # Add final segment
        if current_behavior is not None:
            segments.append({
                "elephant_id": elephant_id,
                "start_time": segment_start,
                "end_time": elephant_data["timestamp"].iloc[-1],
                "behavior": current_behavior,
            })
    
    if not segments:
        fig = go.Figure()
        fig.update_layout(title=f"Camera {camera_id} - No Segments")
        return fig
    
    segments_df = pd.DataFrame(segments)
    
    # Ensure datetime columns are properly formatted
    segments_df["start_time"] = pd.to_datetime(segments_df["start_time"])
    segments_df["end_time"] = pd.to_datetime(segments_df["end_time"])
    
    # Create Gantt chart
    room = CAMERA_TO_ROOM.get(camera_id, "unknown")
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
    )
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Time (HH:MM:SS)",
        yaxis_title="Elephant ID",
        legend_title_text="Behavior",
        height=400,
        hovermode="closest",
        xaxis=dict(
            tickformat="%H:%M:%S",
            type="date"
        ),
    )
    
    return fig


def save_combined_camera_dashboard(
    video_data: List[Tuple[str, List[Dict[str, Any]]]],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Create HTML dashboard with 4 camera views grouped by room."""
    
    if px is None or go is None:
        logger.error("Plotly is required for dashboards. Install with: pip install plotly")
        return
    
    # Get unique camera IDs and group by room
    camera_ids = [cam_id for cam_id, _ in video_data]
    
    # Group cameras by room
    room_cameras = defaultdict(list)
    for camera_id in camera_ids:
        room = CAMERA_TO_ROOM.get(camera_id, "unknown")
        room_cameras[room].append(camera_id)
    
    # Sort cameras within each room for consistent display
    for room in room_cameras:
        room_cameras[room].sort()
    
    logger.info(f"Generating dashboard for {len(camera_ids)} cameras grouped by room")
    for room, cams in sorted(room_cameras.items()):
        logger.info(f"  {room.upper()}: {', '.join(cams)}")
    
    # Build combined HTML
    html_parts = []
    html_parts.append("<html><head>")
    html_parts.append("<title>Elephant Behavior Dashboard</title>")
    html_parts.append('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
    html_parts.append("<style>")
    html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }")
    html_parts.append("h1 { text-align: center; color: #333; }")
    html_parts.append("h2 { color: #555; border-bottom: 2px solid #2196F3; padding-bottom: 10px; margin-top: 30px; }")
    html_parts.append(".room-section { background: white; margin: 30px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }")
    html_parts.append(".camera-section { margin: 20px 0; }")
    html_parts.append("</style>")
    html_parts.append("</head><body>")
    html_parts.append("<h1>Elephant Behavior Dashboard - All Cameras</h1>")
    
    # Process each room
    for room in sorted(room_cameras.keys()):
        cameras = room_cameras[room]
        html_parts.append(f'<div class="room-section">')
        html_parts.append(f"<h2>{room.upper()} - Cameras: {', '.join(cameras)}</h2>")
        
        # Add figures for each camera in this room
        for camera_id in cameras:
            fig = generate_camera_dashboard(camera_id, video_data, logger)
            html_parts.append(f'<div class="camera-section">')
            fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
            html_parts.append(fig_html)
            html_parts.append("</div>")
        
        html_parts.append("</div>")
    
    html_parts.append("</body></html>")
    
    # Write to file
    combined_html = "\n".join(html_parts)
    with open(output_path, "w") as f:
        f.write(combined_html)
    
    logger.info(f"Created interactive dashboard: {output_path}")


def plot_room_occupancy(
    room_timelines: Dict[str, Dict[str, List[Dict[str, Any]]]],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Create room occupancy visualization showing which elephants are in each room over time."""
    
    num_rooms = len(room_timelines)
    fig, axes = plt.subplots(num_rooms, 1, figsize=(16, 4 * num_rooms), sharex=True)
    
    if num_rooms == 1:
        axes = [axes]
    
    for idx, (room, individuals) in enumerate(sorted(room_timelines.items())):
        ax = axes[idx]
        
        # Create time bins (5-minute intervals)
        all_times = []
        for timeline in individuals.values():
            all_times.extend([obs["timestamp"] for obs in timeline])
        
        if not all_times:
            continue
        
        min_time = min(all_times)
        max_time = max(all_times)
        time_bins = pd.date_range(min_time, max_time, freq='5min')
        
        # Count track IDs and unique identities per time bin
        track_id_counts = []
        identity_counts = []
        
        for bin_time in time_bins:
            bin_end = bin_time + pd.Timedelta(minutes=5)
            present_track_ids = set()
            present_identities = set()
            
            for stitched_id, timeline in individuals.items():
                for obs in timeline:
                    if bin_time <= obs["timestamp"] < bin_end:
                        present_track_ids.add(stitched_id)
                        if obs.get("gallery_identity"):
                            present_identities.add(obs["gallery_identity"])
                        break
            
            track_id_counts.append(len(present_track_ids))
            identity_counts.append(len(present_identities))
        
        # Plot both metrics
        ax.plot(time_bins, track_id_counts, linewidth=2, label='Total track IDs', color='#2196F3')
        ax.fill_between(time_bins, track_id_counts, alpha=0.3, color='#2196F3')
        ax.plot(time_bins, identity_counts, linewidth=2, label='Unique identities (gallery matched)', 
               color='#4CAF50', linestyle='--')
        
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'{room.upper()} - Cameras: {", ".join(ROOM_PAIRS.get(room, []))}', 
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylim(bottom=0)
    
    # Format x-axis
    axes[-1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    axes[-1].xaxis.set_major_locator(HourLocator())
    axes[-1].set_xlabel('Time', fontsize=12, fontweight='bold')
    
    plt.suptitle('Room Occupancy Over Time', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created room occupancy plot: {output_path}")


def plot_behavior_distribution(
    timelines: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Create behavior distribution pie charts for each identified elephant."""
    
    # Group by identity
    identity_timelines = defaultdict(list)
    for stitched_id, timeline in timelines.items():
        identities = [obs.get("gallery_identity") for obs in timeline if obs.get("gallery_identity")]
        if identities:
            identity = max(set(identities), key=identities.count)
            identity_timelines[identity].extend(timeline)
        else:
            identity_timelines[f"ID {stitched_id}"].extend(timeline)
    
    num_identities = len(identity_timelines)
    cols = min(3, num_identities)
    rows = (num_identities + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    
    if num_identities == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (identity, timeline) in enumerate(sorted(identity_timelines.items())):
        ax = axes[idx]
        
        # Count behaviors
        behavior_counts = defaultdict(int)
        for obs in timeline:
            behavior = obs["behavior_label"]
            if behavior != "invalid":  # Skip invalid detections
                behavior_counts[behavior] += 1
        
        if not behavior_counts:
            ax.axis('off')
            continue
        
        # Prepare data
        behaviors = list(behavior_counts.keys())
        counts = list(behavior_counts.values())
        colors = [BEHAVIOR_COLORS.get(b, "#000000") for b in behaviors]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=behaviors, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(f'{identity}\n({sum(counts)} observations)', 
                    fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(num_identities, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Behavior Distribution by Individual', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created behavior distribution plot: {output_path}")


# ==============================================================================
# Main
# ==============================================================================


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)
    
    logger.info("=" * 80)
    logger.info("ELEPHANT BEHAVIOR ANALYSIS")
    logger.info("=" * 80)
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info(f"Input: {input_path}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / "behavior_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Parse invalid zones directory if provided
    invalid_zones_dir = None
    if args.invalid_zones_dir:
        invalid_zones_dir = Path(args.invalid_zones_dir)
        if invalid_zones_dir.exists():
            logger.info(f"Invalid zones directory: {invalid_zones_dir}")
        else:
            logger.warning(f"Invalid zones directory not found: {invalid_zones_dir}")
            invalid_zones_dir = None
    
    # Load stitched data
    video_data = load_stitched_data(input_path, logger, invalid_zones_dir)
    
    # Build individual timelines
    timelines = build_individual_timelines(video_data, logger)
    
    # Group by room
    room_timelines = group_by_room(timelines)
    logger.info(f"Room distribution: {', '.join(f'{room}: {len(indiv)} individuals' for room, indiv in room_timelines.items())}")
    logger.info("")
    
    # Generate outputs
    logger.info("Generating analysis outputs...")
    
    # 1. Summary report
    write_summary_report(
        timelines,
        room_timelines,
        output_dir / "behavior_summary.txt",
        logger,
    )
    
    # 2. CSV export
    export_csv_data(
        timelines,
        output_dir / "timeline_data.csv",
        logger,
    )
    
    # 3. Interactive HTML dashboard (4 cameras)
    save_combined_camera_dashboard(
        video_data,
        output_dir / "behavior_dashboard.html",
        logger,
    )
    
    # 4. Room occupancy visualization
    plot_room_occupancy(
        room_timelines,
        output_dir / "room_occupancy.png",
        logger,
    )
    
    # 5. Behavior distribution
    plot_behavior_distribution(
        timelines,
        output_dir / "behavior_distribution.png",
        logger,
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("  - behavior_summary.txt     : Human-readable report for zookeepers")
    logger.info("  - timeline_data.csv        : Detailed data for further analysis")
    logger.info("  - behavior_dashboard.html  : Interactive dashboard (4 cameras)")
    logger.info("  - room_occupancy.png       : Room occupancy over time")
    logger.info("  - behavior_distribution.png: Behavior breakdown per elephant")


if __name__ == "__main__":
    main()
