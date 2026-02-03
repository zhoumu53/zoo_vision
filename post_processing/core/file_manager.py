"""
File helper utilities (functional style).

Provides helpers for locating and loading video/track files and config data.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CAMERA_NAMES = [
    "zag_elp_cam_016",
    "zag_elp_cam_017",
    "zag_elp_cam_018",
    "zag_elp_cam_019",
]



def parse_video_start_from_name(path: Path) -> Optional[datetime]:
    """Parse datetime from raw video filename like ZAG-ELP-CAM-016-20251129-001949-....mp4."""
    name = path.name
    import re

    m = re.search(r"CAM-(?P<cam>\d{3})-(?P<date>\d{8})-(?P<hms>\d{6})", name)
    if not m:
        return None
    dt_str = f"{m.group('date')}{m.group('hms')}"
    try:
        return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    except Exception:
        return None


def find_raw_video(
    camera_id: str,
    timestamp: Optional[datetime],
    root: Path = Path("/mnt/camera_nas"),
) -> Optional[Path]:
    """
    Find the closest raw video under /mnt/camera_nas/ZAG-ELP-CAM-{cam}/{date}{AM|PM}/.
    Chooses the video whose parsed start time is closest to the given timestamp.
    """
    if timestamp is None:
        return None

    date_str = timestamp.strftime("%Y%m%d")
    ampm = "AM" if timestamp.hour < 12 else "PM"
    base_dir = root / f"ZAG-ELP-CAM-{camera_id}" / f"{date_str}{ampm}"

    if not base_dir.exists():
        return None

    candidates = list(base_dir.glob("*.mp4"))
    if not candidates:
        candidates = list(base_dir.rglob("*.mp4"))
    if not candidates:
        return None

    best_path = None
    best_delta = None
    for p in candidates:
        start_dt = parse_video_start_from_name(p)
        if start_dt is None:
            continue
        delta = abs((start_dt - timestamp).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_path = p

    return best_path or candidates[0]


def cam_id2name(cam_id: str) -> str:
    """Convert camera ID to camera name."""
    cam_id = cam_id.zfill(3)
    return f"zag_elp_cam_{cam_id}"


def read_csv(csv_path: Path) -> pd.DataFrame:
    """Read CSV file into DataFrame."""
    return pd.read_csv(csv_path)


def get_track_dir(record_root: Path, cam_id: str, date: str) -> Path:
    """Get track directory path for given camera ID and date."""
    date_norm = normalize_date(date)
    cam_name = cam_id2name(cam_id)
    return record_root / "tracks" / cam_name / date_norm


def load_config_file(config_path: Path) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


def normalize_date(date: str) -> str:
    """Return date in YYYY-MM-DD format when possible."""
    if len(date) == 8 and date.isdigit():
        return datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
    return date


def offline_track_dir(record_root: Path, cam_id: str, date: str) -> Path:
    """Pure helper to build offline track dir path."""
    date_norm = normalize_date(date)
    return Path(record_root) / "tracks" / cam_id2name(cam_id) / date_norm


def list_track_files(track_dir: Path, logger: Optional[logging.Logger] = None) -> List[Path]:
    """List CSV files in a track directory."""
    if not track_dir.exists():
        if logger:
            logger.warning("Track directory does not exist: %s", track_dir)
        return []
    track_files = sorted(track_dir.glob("*.csv"))
    # Filter out part files and behavior files
    track_files = [f for f in track_files if "part_" not in f.name and "behavior.csv" not in f.name]
    
    if logger:
        logger.info("Found %d track files in %s", len(track_files), track_dir)
    return track_files


def list_npz_files(track_dir: Path) -> List[Path]:
    """List NPZ feature files in a track directory."""
    return sorted(track_dir.glob("*.npz"))


def load_camera_names(config_path: Path) -> List[str]:
    """Load camera names from configuration."""
    data = load_config_file(config_path)
    camera_names = data.get("enabled_cameras", [])
    return camera_names if camera_names else CAMERA_NAMES


def list_track_files_all_cams(
    record_root: Path,
    camera_ids: List[str],
    date: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Tuple[List[Path], List[Path]]]:
    """List track files for all specified cameras."""
    all_track_files: Dict[str, Tuple[List[Path], List[Path]]] = {}
    for cam_id in camera_ids:
        td = offline_track_dir(record_root, cam_id, date)
        all_track_files[cam_id] = list_track_files(td, logger=logger)
    return all_track_files


def get_track_files_by_timestamp(
    record_root: Path,
    camera_id: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """Get track files within the specified timestamp range."""
    dates = pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq="D")
    
    files = []
    for date in dates:
        track_dir = get_track_dir(record_root, camera_id, date.strftime("%Y-%m-%d"))
        if not track_dir.exists():
            continue
        
        track_files = list_track_files(track_dir, logger=logger)
        
        for tf in track_files:
            start_time_str = tf.stem.split("_")[0].split("T")[1]
            start_time = pd.Timestamp(f"{date.strftime('%Y-%m-%d')} {start_time_str}")
            if start_dt <= start_time <= end_dt:
                files.append(tf)
        
    return files
    
    
if __name__ == "__main__":
    
    
    files = get_track_files_by_timestamp(
        record_root=Path("/media/ElephantsWD/elephants/xmas/"),
        camera_id="016",
        start_dt=pd.Timestamp("2025-11-30 18:00:00"),
        end_dt=pd.Timestamp("2025-12-01 08:00:00"),
    )
    
    print("Found files:")
    for f in files:
        print(f)