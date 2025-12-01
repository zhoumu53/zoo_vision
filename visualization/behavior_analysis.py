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
        "--filter-identity-jumps",
        action="store_true",
        help="Enable filtering of short-lived identity jumps (spurious identifications)",
    )
    parser.add_argument(
        "--identity-jump-window",
        type=int,
        default=120,
        help="Time window in seconds to check for identity jumps (default: 120s = 2min)",
    )
    parser.add_argument(
        "--identity-jump-min-duration",
        type=int,
        default=10,
        help="Minimum duration in seconds for an identity to be valid (default: 10s)",
    )
    parser.add_argument(
        "--bbox-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for bbox similarity when filtering identity jumps (default: 0.5)",
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


def compute_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0
    
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Compute intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Compute union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


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


def fix_single_track_identity_switches(
    video_data: List[Tuple[str, List[Dict[str, Any]]]],
    logger: logging.Logger,
    debug_camera: Optional[str] = None,
    debug_time_ranges: Optional[List[Tuple[int, int]]] = None,
    time_window_minutes: int = 5,
    min_spurious_duration_seconds: int = 30,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Fix identity switches and remove spurious detections.
    
    This function handles two scenarios:
    1. Single detection frames: When only 1 detection exists, fix identity switches
    2. Spurious detections: When a track ID appears briefly alongside a dominant track,
       remove it as a false positive
    
    Args:
        video_data: List of (camera_id, frames) tuples
        logger: Logger instance
        debug_camera: Optional camera ID to debug
        debug_time_ranges: Optional list of (hour, minute) tuples to debug
        time_window_minutes: Time window in minutes to analyze (default: 5)
        min_spurious_duration_seconds: Minimum duration for a track to be valid (default: 30s)
    
    Returns:
        Corrected video_data with identity switches fixed and spurious tracks removed
    """
    logger.info("Fixing identity switches and removing spurious detections...")
    logger.info(f"  Time window: {time_window_minutes} minutes")
    logger.info(f"  Min track duration: {min_spurious_duration_seconds} seconds")
    
    fixed_data = []
    total_fixed = 0
    total_removed = 0
    
    for camera_id, frames in video_data:
        # Collect all track appearances with timestamps
        track_timeline = defaultdict(list)  # stitched_id -> list of (timestamp, frame_idx, track)
        
        for frame_idx, frame in enumerate(frames):
            timestamp = parse_timestamp(frame.get("timestamp", ""))
            tracks = frame.get("tracks", [])
            
            for track in tracks:
                stitched_id = track.get("stitched_track_id", -1)
                if stitched_id != -1:
                    track_timeline[stitched_id].append({
                        "timestamp": timestamp,
                        "frame_idx": frame_idx,
                        "track": track,
                        "num_detections_in_frame": len(tracks),
                    })
        
        # Analyze each track's duration and temporal pattern
        track_stats = {}
        for stitched_id, appearances in track_timeline.items():
            appearances.sort(key=lambda x: x["timestamp"])
            first_time = appearances[0]["timestamp"]
            last_time = appearances[-1]["timestamp"]
            duration = (last_time - first_time).total_seconds()
            
            track_stats[stitched_id] = {
                "first_time": first_time,
                "last_time": last_time,
                "duration": duration,
                "num_frames": len(appearances),
                "appearances": appearances,
            }
        
        # Find spurious tracks (short-lived tracks that overlap with longer tracks)
        tracks_to_remove = set()
        
        for stitched_id, stats in track_stats.items():
            if stats["duration"] < min_spurious_duration_seconds:
                # Check if this track overlaps with a longer track
                for other_id, other_stats in track_stats.items():
                    if other_id == stitched_id:
                        continue
                    
                    # Check temporal overlap
                    if (stats["first_time"] >= other_stats["first_time"] and 
                        stats["first_time"] <= other_stats["last_time"]) or \
                       (stats["last_time"] >= other_stats["first_time"] and 
                        stats["last_time"] <= other_stats["last_time"]):
                        
                        # If other track is significantly longer, mark this as spurious
                        if other_stats["duration"] > stats["duration"] * 3:
                            tracks_to_remove.add(stitched_id)
                            
                            # Debug output
                            in_debug_range = False
                            if debug_camera and camera_id == debug_camera and debug_time_ranges:
                                for hour, minute in debug_time_ranges:
                                    if (stats["first_time"].hour == hour and abs(stats["first_time"].minute - minute) <= 5):
                                        in_debug_range = True
                                        logger.info("=" * 80)
                                        logger.info(f"DEBUG SPURIOUS TRACK REMOVAL: Camera {camera_id}")
                                        logger.info(f"  Removing track {stitched_id}:")
                                        logger.info(f"    Duration: {stats['duration']:.1f}s, Frames: {stats['num_frames']}")
                                        logger.info(f"    Time: {stats['first_time'].strftime('%H:%M:%S')} - {stats['last_time'].strftime('%H:%M:%S')}")
                                        logger.info(f"  Dominant track {other_id}:")
                                        logger.info(f"    Duration: {other_stats['duration']:.1f}s, Frames: {other_stats['num_frames']}")
                                        logger.info(f"    Time: {other_stats['first_time'].strftime('%H:%M:%S')} - {other_stats['last_time'].strftime('%H:%M:%S')}")
                                        break
                            break
        
        # Remove spurious tracks from frames
        if tracks_to_remove:
            for frame_idx, frame in enumerate(frames):
                tracks = frame.get("tracks", [])
                original_count = len(tracks)
                frame["tracks"] = [t for t in tracks if t.get("stitched_track_id", -1) not in tracks_to_remove]
                removed_count = original_count - len(frame["tracks"])
                total_removed += removed_count
        
        # Now handle single-detection frames and identity switches
        single_detection_frames = []
        
        for frame_idx, frame in enumerate(frames):
            tracks = frame.get("tracks", [])
            
            if len(tracks) == 1:
                track = tracks[0]
                timestamp = parse_timestamp(frame.get("timestamp", ""))
                single_detection_frames.append({
                    "frame_idx": frame_idx,
                    "frame": frame,
                    "track": track,
                    "timestamp": timestamp,
                })
        
        if single_detection_frames:
            # Sort by timestamp
            single_detection_frames.sort(key=lambda x: x["timestamp"])
            
            # Group into temporal sequences
            sequences = []
            current_sequence = [single_detection_frames[0]]
            
            for item in single_detection_frames[1:]:
                time_gap = (item["timestamp"] - current_sequence[-1]["timestamp"]).total_seconds()
                
                if time_gap <= time_window_minutes * 60:
                    current_sequence.append(item)
                else:
                    if len(current_sequence) >= 2:
                        sequences.append(current_sequence)
                    current_sequence = [item]
            
            if len(current_sequence) >= 2:
                sequences.append(current_sequence)
            
            # Fix identity switches in each sequence
            for sequence in sequences:
                identities = []
                for item in sequence:
                    identity = item["track"].get("gallery_identity")
                    if identity:
                        identities.append(identity)
                
                if not identities:
                    continue
                
                # Find the most common identity
                identity_counts = defaultdict(int)
                for identity in identities:
                    identity_counts[identity] += 1
                
                most_common_identity = max(identity_counts.items(), key=lambda x: x[1])[0]
                most_common_count = identity_counts[most_common_identity]
                
                unique_identities = set(identities)
                if len(unique_identities) > 1:
                    start_time = sequence[0]["timestamp"]
                    end_time = sequence[-1]["timestamp"]
                    
                    in_debug_range = False
                    if debug_camera and camera_id == debug_camera and debug_time_ranges:
                        for hour, minute in debug_time_ranges:
                            if (start_time.hour == hour and abs(start_time.minute - minute) <= 5) or \
                               (end_time.hour == hour and abs(end_time.minute - minute) <= 5):
                                in_debug_range = True
                                logger.info("=" * 80)
                                logger.info(f"DEBUG IDENTITY FIX: Camera {camera_id}")
                                logger.info(f"  Time range: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}")
                                logger.info(f"  Sequence: {len(sequence)} single-detection frames")
                                logger.info(f"  Identity switches: {dict(identity_counts)}")
                                logger.info(f"  Most common: {most_common_identity} ({most_common_count}/{len(identities)})")
                                break
                    
                    # Fix identities
                    fixes_made = 0
                    for item in sequence:
                        current_identity = item["track"].get("gallery_identity")
                        if current_identity != most_common_identity:
                            item["track"]["gallery_identity"] = most_common_identity
                            fixes_made += 1
                            total_fixed += 1
                    
                    if in_debug_range:
                        logger.info(f"  DECISION: Fixed {fixes_made} identity switches to '{most_common_identity}'")
        
        fixed_data.append((camera_id, frames))
    
    logger.info(f"Removed {total_removed} spurious detections")
    logger.info(f"Fixed {total_fixed} identity switches")
    return fixed_data


def filter_identity_jumps(
    video_data: List[Tuple[str, List[Dict[str, Any]]]],
    time_window: int,
    min_duration: int,
    iou_threshold: float,
    logger: logging.Logger,
    debug_camera: Optional[str] = None,
    debug_time_ranges: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Filter short-lived identity jumps that are likely spurious.
    
    If an identity appears for less than min_duration seconds, and the bounding boxes
    before and after are similar (high IoU), assume it's a spurious identification
    and remove the gallery_identity.
    
    Args:
        video_data: List of (camera_id, frames) tuples
        time_window: Time window in seconds to check around identity changes
        min_duration: Minimum duration in seconds for valid identity
        iou_threshold: Minimum IoU to consider bboxes similar
        logger: Logger instance
        debug_camera: Optional camera ID to debug (e.g., "019")
        debug_time_ranges: Optional list of (hour, minute) tuples to debug (e.g., [(18, 49), (19, 21)])
    
    Returns:
        Filtered video_data with spurious identities removed
    """
    logger.info("Filtering identity jumps...")
    logger.info(f"  Time window: {time_window}s")
    logger.info(f"  Min duration: {min_duration}s")
    logger.info(f"  IoU threshold: {iou_threshold}")
    
    if debug_camera:
        logger.info(f"  DEBUG MODE: Camera {debug_camera}, Time ranges: {debug_time_ranges}")
    
    filtered_data = []
    total_removed = 0
    
    for camera_id, frames in video_data:
        # Build temporal index: timestamp -> list of (track_id, bbox, identity)
        temporal_index = defaultdict(list)
        for frame in frames:
            timestamp = parse_timestamp(frame.get("timestamp", ""))
            for track in frame.get("tracks", []):
                stitched_id = track.get("stitched_track_id", -1)
                if stitched_id != -1:
                    temporal_index[timestamp].append({
                        "track_id": stitched_id,
                        "bbox": track.get("bbox"),
                        "identity": track.get("gallery_identity"),
                    })
        
        # Group frames by stitched_track_id
        track_frames = defaultdict(list)
        for frame_idx, frame in enumerate(frames):
            for track in frame.get("tracks", []):
                stitched_id = track.get("stitched_track_id", -1)
                if stitched_id != -1:
                    track_frames[stitched_id].append((frame_idx, frame, track))
        
        # Process each track's timeline
        for stitched_id, track_timeline in track_frames.items():
            # Sort by frame index
            track_timeline.sort(key=lambda x: x[0])
            
            # Find identity segments (consecutive frames with same identity)
            identity_segments = []
            current_identity = None
            segment_start = 0
            
            for i, (frame_idx, frame, track) in enumerate(track_timeline):
                identity = track.get("gallery_identity")
                
                if identity != current_identity:
                    # End previous segment
                    if current_identity is not None:
                        identity_segments.append({
                            "identity": current_identity,
                            "start_idx": segment_start,
                            "end_idx": i - 1,
                            "indices": list(range(segment_start, i)),
                        })
                    
                    # Start new segment
                    current_identity = identity
                    segment_start = i
            
            # Add final segment
            if current_identity is not None:
                identity_segments.append({
                    "identity": current_identity,
                    "start_idx": segment_start,
                    "end_idx": len(track_timeline) - 1,
                    "indices": list(range(segment_start, len(track_timeline))),
                })
            
            # Check each non-None identity segment for spurious jumps
            for seg in identity_segments:
                if seg["identity"] is None:
                    continue
                
                # Get temporal duration
                start_time = parse_timestamp(track_timeline[seg["start_idx"]][1].get("timestamp", ""))
                end_time = parse_timestamp(track_timeline[seg["end_idx"]][1].get("timestamp", ""))
                duration = (end_time - start_time).total_seconds()
                
                # Debug output for specific camera and time ranges
                in_debug_range = False
                if debug_camera and camera_id == debug_camera and debug_time_ranges:
                    for hour, minute in debug_time_ranges:
                        if (start_time.hour == hour and abs(start_time.minute - minute) <= 5) or \
                           (end_time.hour == hour and abs(end_time.minute - minute) <= 5):
                            in_debug_range = True
                            logger.info("=" * 80)
                            logger.info(f"DEBUG: Camera {camera_id}, Track {stitched_id}")
                            logger.info(f"  Identity: {seg['identity']}")
                            logger.info(f"  Time range: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}")
                            logger.info(f"  Duration: {duration:.1f}s")
                            logger.info(f"  Num frames: {len(seg['indices'])}")
                            logger.info(f"  Segment indices: {seg['start_idx']} to {seg['end_idx']} (out of {len(track_timeline)})")
                            break
                
                # If duration is too short, check if it's a spurious jump
                if duration < min_duration:
                    # Check bbox similarity with neighbors
                    should_remove = False
                    
                    # Get bboxes and timestamps in this segment
                    segment_bboxes = []
                    segment_timestamps = []
                    segment_centers = []
                    for idx in seg["indices"]:
                        bbox = track_timeline[idx][2].get("bbox")
                        segment_bboxes.append(bbox)
                        segment_timestamps.append(parse_timestamp(track_timeline[idx][1].get("timestamp", "")))
                        if bbox and len(bbox) == 4:
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2
                            segment_centers.append((center_x, center_y))
                    
                    # Find neighboring tracks at similar times (using temporal index)
                    # Look for other tracks that exist before and after this segment
                    prev_bboxes = []
                    prev_identities = []
                    prev_centers = []
                    next_bboxes = []
                    next_identities = []
                    next_centers = []
                    
                    # Get all timestamps in a time window around this segment
                    all_timestamps = sorted(temporal_index.keys())
                    
                    # Find timestamps before segment start
                    for ts in all_timestamps:
                        if ts < start_time and (start_time - ts).total_seconds() <= time_window:
                            for track_info in temporal_index[ts]:
                                # Skip the same track
                                if track_info["track_id"] != stitched_id:
                                    bbox = track_info["bbox"]
                                    prev_bboxes.append(bbox)
                                    prev_identities.append(track_info["identity"])
                                    if bbox and len(bbox) == 4:
                                        center_x = (bbox[0] + bbox[2]) / 2
                                        center_y = (bbox[1] + bbox[3]) / 2
                                        prev_centers.append((center_x, center_y))
                    
                    # Find timestamps after segment end
                    for ts in all_timestamps:
                        if ts > end_time and (ts - end_time).total_seconds() <= time_window:
                            for track_info in temporal_index[ts]:
                                # Skip the same track
                                if track_info["track_id"] != stitched_id:
                                    bbox = track_info["bbox"]
                                    next_bboxes.append(bbox)
                                    next_identities.append(track_info["identity"])
                                    if bbox and len(bbox) == 4:
                                        center_x = (bbox[0] + bbox[2]) / 2
                                        center_y = (bbox[1] + bbox[3]) / 2
                                        next_centers.append((center_x, center_y))
                    
                    if in_debug_range:
                        # Compute average center for this segment
                        if segment_centers:
                            avg_center_x = np.mean([c[0] for c in segment_centers])
                            avg_center_y = np.mean([c[1] for c in segment_centers])
                            logger.info(f"  Segment center: ({avg_center_x:.1f}, {avg_center_y:.1f})")
                            logger.info(f"  Segment bboxes sample: {segment_bboxes[:3]}")
                        
                        logger.info(f"  Prev frames: {len(prev_bboxes)}, identities: {set(prev_identities)}")
                        if prev_centers:
                            avg_prev_x = np.mean([c[0] for c in prev_centers[:5]])
                            avg_prev_y = np.mean([c[1] for c in prev_centers[:5]])
                            logger.info(f"  Prev centers sample: ({avg_prev_x:.1f}, {avg_prev_y:.1f})")
                        
                        logger.info(f"  Next frames: {len(next_bboxes)}, identities: {set(next_identities)}")
                        if next_centers:
                            avg_next_x = np.mean([c[0] for c in next_centers[:5]])
                            avg_next_y = np.mean([c[1] for c in next_centers[:5]])
                            logger.info(f"  Next centers sample: ({avg_next_x:.1f}, {avg_next_y:.1f})")
                    
                    # If we have neighbors and high IoU, it's likely spurious
                    if prev_bboxes and next_bboxes and segment_bboxes:
                        # Compute average IoU with neighbors
                        ious = []
                        for seg_bbox in segment_bboxes:
                            for prev_bbox in prev_bboxes:
                                iou = compute_bbox_iou(seg_bbox, prev_bbox)
                                ious.append(iou)
                            for next_bbox in next_bboxes:
                                iou = compute_bbox_iou(seg_bbox, next_bbox)
                                ious.append(iou)
                        
                        if ious:
                            avg_iou = np.mean(ious)
                            max_iou = np.max(ious)
                            
                            if in_debug_range:
                                logger.info(f"  IoU stats: avg={avg_iou:.3f}, max={max_iou:.3f}")
                                logger.info(f"  All IoUs: {[f'{iou:.3f}' for iou in ious[:10]]}...")
                            
                            if avg_iou >= iou_threshold:
                                should_remove = True
                                if in_debug_range:
                                    logger.info(f"  DECISION: REMOVE (avg IoU {avg_iou:.3f} >= {iou_threshold})")
                                else:
                                    logger.debug(
                                        f"Removing spurious identity '{seg['identity']}' for track {stitched_id} "
                                        f"(duration: {duration:.1f}s, avg IoU: {avg_iou:.2f})"
                                    )
                            elif in_debug_range:
                                logger.info(f"  DECISION: KEEP (avg IoU {avg_iou:.3f} < {iou_threshold})")
                    elif in_debug_range:
                        logger.info(f"  DECISION: KEEP (insufficient neighbors: prev={len(prev_bboxes)}, next={len(next_bboxes)})")
                    
                    # Remove gallery_identity from these frames
                    if should_remove:
                        for idx in seg["indices"]:
                            frame_idx, frame, track = track_timeline[idx]
                            track["gallery_identity"] = None
                            track["gallery_score"] = None
                            total_removed += 1
                elif in_debug_range:
                    logger.info(f"  DECISION: KEEP (duration {duration:.1f}s >= min {min_duration}s)")
        
        filtered_data.append((camera_id, frames))
    
    logger.info(f"Filtered {total_removed} spurious identity detections")
    return filtered_data


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
                
                # Get behavior from nested dict safely
                behavior_dict = track.get("behavior", {})
                if isinstance(behavior_dict, dict):
                    behavior_label = behavior_dict.get("label", "unknown")
                    behavior_prob = behavior_dict.get("prob", 0.0)
                else:
                    behavior_label = "unknown"
                    behavior_prob = 0.0
                
                observation = {
                    "timestamp": timestamp,
                    "stitched_id": stitched_id,
                    "gallery_identity": track.get("gallery_identity"),
                    "gallery_score": track.get("gallery_score"),
                    "behavior_label": behavior_label,
                    "behavior_prob": behavior_prob,
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
            
            # Use gallery_identity if available, otherwise use stitched_id
            gallery_identity = track.get("gallery_identity")
            if gallery_identity:
                elephant_id = gallery_identity
            else:
                elephant_id = f"Invalid"

            # Get behavior from nested dict
            behavior_dict = track.get("behavior", {})
            if isinstance(behavior_dict, dict):
                behavior_label = behavior_dict.get("label", "unknown")
            else:
                behavior_label = "unknown"
            
            elephants_in_room.add(elephant_id)
            
            timeline_records.append({
                "timestamp": timestamp,
                "elephant_id": elephant_id,
                "behavior": behavior_label,
            })

    if not timeline_records:
        fig = go.Figure()
        fig.update_layout(title=f"Camera {camera_id} - No Tracks")
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(timeline_records)
    
    # Compute behavior segments (consecutive same behavior with temporal continuity)
    # Max gap of 5 seconds to consider frames as continuous
    MAX_TEMPORAL_GAP = pd.Timedelta(seconds=5)
    
    segments = []
    df_sorted = df.sort_values(by=["elephant_id", "timestamp"])
    
    for elephant_id in df_sorted["elephant_id"].unique():
        elephant_data = df_sorted[df_sorted["elephant_id"] == elephant_id].copy()
        elephant_data = elephant_data.sort_values("timestamp").reset_index(drop=True)
        
        current_behavior = None
        segment_start = None
        prev_timestamp = None
        
        for idx, row in elephant_data.iterrows():
            behavior = row["behavior"]
            timestamp = row["timestamp"]
            
            # Check if there's a temporal gap (missing frames)
            temporal_gap = False
            if prev_timestamp is not None:
                time_diff = timestamp - prev_timestamp
                if time_diff > MAX_TEMPORAL_GAP:
                    temporal_gap = True
            
            # Start new segment if behavior changes OR temporal gap detected
            if behavior != current_behavior or temporal_gap:
                # End previous segment
                if current_behavior is not None and segment_start is not None:
                    segments.append({
                        "elephant_id": elephant_id,
                        "start_time": segment_start,
                        "end_time": prev_timestamp,
                        "behavior": current_behavior,
                    })
                
                # Start new segment
                current_behavior = behavior
                segment_start = timestamp
            
            prev_timestamp = timestamp
        
        # Add final segment
        if current_behavior is not None and segment_start is not None and prev_timestamp is not None:
            segments.append({
                "elephant_id": elephant_id,
                "start_time": segment_start,
                "end_time": prev_timestamp,
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


def plot_camera_occupancy(
    video_data: List[Tuple[str, List[Dict[str, Any]]]],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Create camera occupancy visualization showing elephant count per camera over time."""
    
    # Extract all cameras
    camera_ids = sorted([cam_id for cam_id, _ in video_data])
    num_cameras = len(camera_ids)
    
    if num_cameras == 0:
        logger.warning("No cameras found for occupancy plot")
        return
    
    fig, axes = plt.subplots(num_cameras, 1, figsize=(16, 4 * num_cameras), sharex=True)
    
    if num_cameras == 1:
        axes = [axes]
    
    for idx, camera_id in enumerate(camera_ids):
        ax = axes[idx]
        
        # Find camera data
        camera_frames = None
        for cam_id, frames in video_data:
            if cam_id == camera_id:
                camera_frames = frames
                break
        
        if not camera_frames:
            continue
        
        # Extract all timestamps and detections
        all_times = []
        camera_detections = []
        
        for frame in camera_frames:
            timestamp = parse_timestamp(frame.get("timestamp", ""))
            tracks = frame.get("tracks", [])
            
            all_times.append(timestamp)
            camera_detections.append({
                "timestamp": timestamp,
                "tracks": tracks,
            })
        
        if not all_times:
            continue
        
        # Create time bins (5-minute intervals)
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
            
            for frame_data in camera_detections:
                timestamp = frame_data["timestamp"]
                if bin_time <= timestamp < bin_end:
                    for track in frame_data["tracks"]:
                        stitched_id = track.get("stitched_track_id", -1)
                        if stitched_id != -1:
                            present_track_ids.add(stitched_id)
                            
                            gallery_identity = track.get("gallery_identity")
                            if gallery_identity:
                                present_identities.add(gallery_identity)
            
            track_id_counts.append(len(present_track_ids))
            identity_counts.append(len(present_identities))
        
        # Plot both metrics
        ax.plot(time_bins, track_id_counts, linewidth=2, label='Total track IDs', color='#2196F3')
        ax.fill_between(time_bins, track_id_counts, alpha=0.3, color='#2196F3')
        ax.plot(time_bins, identity_counts, linewidth=2, label='Unique identities (gallery matched)', 
               color='#4CAF50', linestyle='--')
        
        room = CAMERA_TO_ROOM.get(camera_id, "unknown")
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Camera {camera_id} ({room.upper()})', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylim(bottom=0)
    
    # Format x-axis
    axes[-1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    axes[-1].xaxis.set_major_locator(HourLocator())
    axes[-1].set_xlabel('Time', fontsize=12, fontweight='bold')
    
    plt.suptitle('Camera Occupancy Over Time', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created camera occupancy plot: {output_path}")


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
    
    # Debug mode: focus on camera 019 at specific times
    debug_camera = "019"
    debug_time_ranges = [(18, 49), (19, 21)]  # (hour, minute) tuples
    
    # Fix single-track identity switches (when only 1 elephant in frame)
    video_data = fix_single_track_identity_switches(
        video_data,
        logger,
        debug_camera=debug_camera,
        debug_time_ranges=debug_time_ranges,
    )
    
    # # # TODO -- improve this Filter identity jumps if requested
    # if args.filter_identity_jumps:
    #     video_data = filter_identity_jumps(
    #         video_data,
    #         args.identity_jump_window,
    #         args.identity_jump_min_duration,
    #         args.bbox_iou_threshold,
    #         logger,
    #         debug_camera=debug_camera,
    #         debug_time_ranges=debug_time_ranges,
    #     )
    
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
    
    # 5. Camera occupancy visualization
    plot_camera_occupancy(
        video_data,
        output_dir / "camera_occupancy.png",
        logger,
    )
    
    # 6. Behavior distribution
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
    logger.info("  - camera_occupancy.png     : Camera occupancy over time")
    logger.info("  - behavior_distribution.png: Behavior breakdown per elephant")


if __name__ == "__main__":
    main()
