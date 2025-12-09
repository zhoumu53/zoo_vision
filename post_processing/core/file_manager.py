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


def cam_id2name(cam_id: str) -> str:
    """Convert camera ID to camera name."""
    cam_id = cam_id.zfill(3)
    return f"zag_elp_cam_{cam_id}"


def read_csv(csv_path: Path) -> pd.DataFrame:
    """Read CSV file into DataFrame."""
    return pd.read_csv(csv_path)


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


def list_track_files(track_dir: Path, logger: Optional[logging.Logger] = None) -> Tuple[List[Path], List[Path]]:
    """List CSV/MKV files in a track directory."""
    if not track_dir.exists():
        if logger:
            logger.warning("Track directory does not exist: %s", track_dir)
        return [], []
    track_files = sorted(track_dir.glob("*.csv"))
    track_videos = sorted(track_dir.glob("*.mkv"))
    if logger:
        logger.info("Found %d track files in %s", len(track_files), track_dir)
    return track_files, track_videos


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
