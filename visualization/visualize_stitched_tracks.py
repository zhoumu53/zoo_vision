"""
Visualize Stitched Tracks in 2x2 Grid

This script creates synchronized multi-camera visualizations of stitched tracking results.
Displays 4 camera views in a 2x2 grid with annotated bounding boxes showing:
- Track ID (stitched)
- Original Track ID
- Behavior label

Camera layout:
  Row 1: CAM-016 | CAM-019
  Row 2: CAM-017 | CAM-018
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Import color utilities
import sys
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from utils import COLOR_PALETTE, build_label_color_map


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("visualize_stitched")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize stitched tracks in 2x2 multi-camera grid"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to stitched_tracks.jsonl file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for visualization frames",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Target FPS for output video (samples frames accordingly)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Maximum number of frames to generate",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def extract_camera_id(frame_name: str) -> Optional[str]:
    """Extract camera ID from frame name (e.g., 'ZAG-ELP-CAM-016-...')."""
    match = re.search(r"CAM-(\d+)", frame_name)
    return match.group(1) if match else None


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string (format: '20250729_183238')."""
    try:
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except Exception:
        # Try without seconds
        try:
            return datetime.strptime(ts_str[:13], "%Y%m%d_%H%M")
        except Exception:
            return datetime.now()


def load_stitched_tracks(jsonl_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load stitched tracks JSONL and organize by camera and timestamp.
    
    Returns:
        {
            'metadata': {camera_id: metadata_dict},
            'frames': {camera_id: {timestamp: frame_data}}
        }
    """
    metadata = {}
    frames = defaultdict(dict)
    
    current_camera = None
    current_video = None
    
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            if "meta" in data:
                # Metadata record - extract camera ID from video path
                meta = data["meta"]
                video_path = meta.get("video", "")
                
                # Extract camera ID from video filename
                match = re.search(r"CAM-(\d+)", video_path)
                if match:
                    cam_id = match.group(1)
                    metadata[cam_id] = meta
                    current_camera = cam_id
                    current_video = video_path
                    logger.debug("Loaded metadata for camera %s", cam_id)
                
            elif "results" in data:
                # Frame results
                results = data["results"]
                frame_idx = results.get("frame_idx", 0)
                timestamp = results.get("timestamp", "")
                tracks = results.get("tracks", [])
                
                # Try to get camera ID from tracks
                cam_id = None
                if tracks:
                    for track in tracks:
                        frame_name = track.get("frame_name", "")
                        cam_id = extract_camera_id(frame_name)
                        if cam_id:
                            break
                
                # Fall back to current camera
                if not cam_id:
                    cam_id = current_camera
                
                if cam_id and timestamp:
                    frames[cam_id][timestamp] = {
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "tracks": tracks,
                        "video_path": current_video,
                    }
    
    logger.info("Loaded tracks for cameras: %s", sorted(metadata.keys()))
    for cam_id in sorted(metadata.keys()):
        logger.info("  Camera %s: %d frames", cam_id, len(frames[cam_id]))
    
    return {
        "metadata": metadata,
        "frames": frames,
    }


def synchronize_timestamps(
    frames: Dict[str, Dict[str, Any]], 
    logger: logging.Logger
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Synchronize frames across cameras by timestamp.
    
    Returns list of (timestamp, {camera_id: frame_data}) tuples.
    """
    # Get all unique timestamps across all cameras
    all_timestamps = set()
    for cam_frames in frames.values():
        all_timestamps.update(cam_frames.keys())
    
    sorted_timestamps = sorted(all_timestamps)
    
    logger.info("Total unique timestamps: %d", len(sorted_timestamps))
    
    # Build synchronized frames
    synced_frames = []
    for ts in sorted_timestamps:
        frame_group = {}
        for cam_id, cam_frames in frames.items():
            if ts in cam_frames:
                frame_group[cam_id] = cam_frames[ts]
        
        # Only include if at least one camera has this timestamp
        if frame_group:
            synced_frames.append((ts, frame_group))
    
    logger.info("Synchronized %d timestamps", len(synced_frames))
    
    return synced_frames


def get_track_color(track_id: int, identity_label: Optional[str] = None, identity_color_map: Optional[Dict[str, Tuple[int, int, int]]] = None) -> Tuple[int, int, int]:
    """Generate consistent BGR color for a track.
    
    Priority:
    1. If identity_label provided and in color_map, use that color (consistent across cameras)
    2. Otherwise, use track_id to generate color
    """
    # Use identity color if available (ensures same elephant has same color across cameras)
    if identity_label and identity_color_map and identity_label in identity_color_map:
        return identity_color_map[identity_label]
    
    # Fallback to track ID color
    if track_id < 0:
        return (128, 128, 128)  # Gray for invalid IDs
    
    # Use hash to ensure seed is within valid range
    seed = abs(hash(track_id)) % (2**32)
    np.random.seed(seed)
    return tuple(np.random.randint(50, 255, 3).tolist())


def draw_tracks_on_frame(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    camera_id: str,
    identity_color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    annotated = frame.copy()
    
    for track in tracks:
        bbox = track.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Get IDs and behavior
        stitched_id = track.get("stitched_track_id", -1)
        original_id = track.get("original_track_id", -1)
        behavior = track.get("behavior", {})
        behavior_label = behavior.get("label", "unknown")
        
        # Get gallery identity if available
        gallery_identity = track.get("gallery_identity")
        gallery_score = track.get("gallery_score")
        
        # Color based on identity label (if available) or stitched ID
        color = get_track_color(stitched_id, identity_label=gallery_identity, identity_color_map=identity_color_map)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label text
        label_lines = [
            f"ID: {stitched_id}",
            f"Ori: {original_id}",
        ]
        
        # Add gallery identity if available
        if gallery_identity:
            if gallery_score is not None:
                label_lines.append(f"{gallery_identity} ({gallery_score:.2f})")
            else:
                label_lines.append(f"{gallery_identity}")
        
        # Add behavior
        label_lines.append(f"{behavior_label}")
        
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        y_offset = y1 - 10
        for i, line in enumerate(label_lines):
            (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            
            # Background rectangle
            cv2.rectangle(
                annotated,
                (x1, y_offset - text_h - 5),
                (x1 + text_w + 5, y_offset + 5),
                color,
                -1,
            )
            
            # Text
            cv2.putText(
                annotated,
                line,
                (x1 + 2, y_offset),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )
            
            y_offset -= (text_h + 8)
    
    # Add camera label
    cam_label = f"CAM-{camera_id}"
    cv2.putText(
        annotated,
        cam_label,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3,
    )
    
    return annotated


def create_2x2_grid(
    frames: Dict[str, np.ndarray],
    target_size: Tuple[int, int] = (960, 540),
) -> np.ndarray:
    """
    Create 2x2 grid layout.
    
    Layout:
      016 | 019
      017 | 018
    """
    # Resize all frames to target size
    resized = {}
    for cam_id, frame in frames.items():
        if frame is not None:
            resized[cam_id] = cv2.resize(frame, target_size)
        else:
            # Create blank frame
            resized[cam_id] = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(
                resized[cam_id],
                f"CAM-{cam_id}",
                (target_size[0] // 2 - 100, target_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (128, 128, 128),
                2,
            )
            cv2.putText(
                resized[cam_id],
                "No Frame",
                (target_size[0] // 2 - 80, target_size[1] // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (128, 128, 128),
                2,
            )
    
    # Arrange in 2x2 grid
    # Row 1: 016, 019
    row1 = np.hstack([
        resized.get("016", resized.get(list(resized.keys())[0])),
        resized.get("019", resized.get(list(resized.keys())[0])),
    ])
    
    # Row 2: 017, 018
    row2 = np.hstack([
        resized.get("017", resized.get(list(resized.keys())[0])),
        resized.get("018", resized.get(list(resized.keys())[0])),
    ])
    
    # Stack rows
    grid = np.vstack([row1, row2])
    
    return grid


def visualize_stitched_tracks(
    jsonl_path: Path,
    output_dir: Path,
    target_fps: float,
    max_frames: int,
    logger: logging.Logger,
) -> None:
    """Generate multi-camera visualization from stitched tracks."""
    
    # Load data
    logger.info("Loading stitched tracks from: %s", jsonl_path)
    data = load_stitched_tracks(jsonl_path, logger)
    
    metadata = data["metadata"]
    frames = data["frames"]
    
    # Synchronize timestamps
    synced_frames = synchronize_timestamps(frames, logger)
    
    # Sample frames based on target FPS
    if target_fps > 0:
        # Assume original videos are ~25 fps
        original_fps = 25.0
        frame_interval = int(original_fps / target_fps)
        sampled_frames = synced_frames[::max(1, frame_interval)]
    else:
        sampled_frames = synced_frames
    
    # Limit to max_frames
    sampled_frames = sampled_frames[:max_frames]
    
    logger.info("Generating %d visualization frames", len(sampled_frames))
    
    # Build identity color map from all gallery identities
    all_identities = set()
    for frame_group_list in frames.values():
        for frame_data in frame_group_list.values():
            for track in frame_data.get("tracks", []):
                identity = track.get("gallery_identity")
                if identity:
                    all_identities.add(identity)
    
    identity_color_map = build_label_color_map(sorted(all_identities)) if all_identities else {}
    logger.info("Built color map for %d unique identities: %s", len(identity_color_map), sorted(identity_color_map.keys()))
    
    # Open video captures
    video_caps = {}
    for cam_id, meta in metadata.items():
        video_path = meta.get("video", "")
        if video_path and Path(video_path).exists():
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                video_caps[cam_id] = cap
                logger.info("Opened video for camera %s: %s", cam_id, video_path)
            else:
                logger.warning("Cannot open video for camera %s: %s", cam_id, video_path)
        else:
            logger.warning("Video not found for camera %s: %s", cam_id, video_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate frames
    for idx, (timestamp, frame_group) in enumerate(tqdm(sampled_frames, desc="Generating frames")):
        camera_frames = {}
        
        for cam_id in ["016", "017", "018", "019"]:
            if cam_id not in frame_group:
                camera_frames[cam_id] = None
                continue
            
            frame_data = frame_group[cam_id]
            frame_idx = frame_data["frame_idx"]
            tracks = frame_data["tracks"]
            
            # Read frame from video
            if cam_id in video_caps:
                cap = video_caps[cam_id]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Draw tracks
                    annotated = draw_tracks_on_frame(frame, tracks, cam_id, identity_color_map)
                    camera_frames[cam_id] = annotated
                else:
                    logger.debug("Cannot read frame %d from camera %s", frame_idx, cam_id)
                    camera_frames[cam_id] = None
            else:
                camera_frames[cam_id] = None
        
        # Create 2x2 grid
        grid = create_2x2_grid(camera_frames)
        
        # Add timestamp overlay
        cv2.putText(
            grid,
            f"Time: {timestamp}",
            (20, grid.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        
        # Save frame
        output_path = output_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(output_path), grid)
    
    # Release video captures
    for cap in video_caps.values():
        cap.release()
    
    logger.info("Visualization complete!")
    logger.info("Output frames saved to: %s", output_dir)
    logger.info("Total frames: %d", len(sampled_frames))


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)
    
    jsonl_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    if not jsonl_path.exists():
        logger.error("Input file not found: %s", jsonl_path)
        return
    
    visualize_stitched_tracks(
        jsonl_path,
        output_dir,
        args.fps,
        args.max_frames,
        logger,
    )


if __name__ == "__main__":
    main()
