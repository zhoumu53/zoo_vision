"""
File Loader Module

Provides FileLoader class for locating and loading video files and their 
corresponding processing results (online tracking, offline stitching, behavior analysis)
based on date, time, and camera ID.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


CAMERA_NAMES = [
    "zag_elp_cam_016",
    "zag_elp_cam_017",
    "zag_elp_cam_018",
    "zag_elp_cam_019"
]


def cam_id2name(cam_id: str) -> str:
    """Convert camera ID to camera name."""
    cam_id = cam_id.zfill(3)
    return f"zag_elp_cam_{cam_id}"


def read_csv(csv_path: Path) -> pd.DataFrame:
    """Read CSV file into DataFrame."""
    return pd.read_csv(csv_path)


class FileManager:
    """
    """
    
    def __init__(
        self,
        date: str,
        raw_video_dir: str = "/mnt/camera_nas",
        config: str = '/media/mu/zoo_vision/data/config.json',
        logger: Optional[logging.Logger] = None,
        record_root: Optional[str] = None,
    ):
        """
        Initialize FileManager.
        
        """
        self.date = date
        self.raw_video_dir = Path(raw_video_dir)
        self.config = Path(config)
        self.config_data = self.load_config()
        self.logger = logger or logging.getLogger(__name__)
        self.record_root = Path(record_root) if record_root else Path(self.config_data.get("record_root", "/media/dherrera/ElephantsWD/elephants/improve"))
        
        # Validate date format
        try:
            datetime.strptime(date, "%Y%m%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date}. Expected YYYYMMDD")
    

    def load_config(self) -> Dict:
        """Load configuration from JSON file."""
        with open(self.config, "r") as f:
            config_data = json.load(f)
        return config_data
    

    def load_camera_names(self) -> List[str]:
        """Load camera names from configuration."""
        
        camera_names = self.config_data.get("enabled_cameras", [])
        
        return camera_names if camera_names else CAMERA_NAMES
    

    def load_offline_track_dir(self, cam_id, date=None) -> Path:
        """Get offline track directory from config."""
        date = date or self.date
        #format date to YYYY-MM-DD if not already
        if len(date) == 8 and date.isdigit():
            date = datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
        cam_name = cam_id2name(cam_id)
        offline_track_dir = Path(self.record_root) / "tracks" / cam_name / date
        return offline_track_dir
    

    def load_offline_track_dirs(self, camera_ids: List[str], date=None) -> Dict[str, Path]:
        """Get offline track directories for multiple cameras."""
        date = date or self.date
        track_dirs = {}
        for cam_id in camera_ids:
            track_dirs[cam_id] = self.load_offline_track_dir(cam_id, date=date)
        return track_dirs
    
    
    def list_track_files(self, track_dir = None) -> List[Path]:
        """List all track files in the given directory."""
        track_dir = self.track_dir if track_dir is None else track_dir
        if not Path(track_dir).exists():
            self.logger.warning("Track directory does not exist: %s", track_dir)
            return []
        
        track_files = sorted(track_dir.glob("*.csv"))
        track_videos = sorted([f for f in track_dir.glob("*.mkv")])
        
        self.logger.info("Found %d track files in %s", len(track_files), track_dir)
        return track_files, track_videos


    def list_track_files_all_cams(self, camera_ids: List[str], date=None) -> Dict[str, Tuple[List[Path], List[Path]]]:
        """List track files for all specified cameras."""
        date = date or self.date
        track_dirs = self.load_offline_track_dirs(camera_ids, date=date)
        all_track_files = {}
        for cam_id, track_dir in track_dirs.items():
            track_files, track_videos = self.list_track_files(track_dir=track_dir)
            all_track_files[cam_id] = (track_files, track_videos)
        return all_track_files


if __name__ == "__main__":
    # Example usage
    loader = FileManager(config='/media/mu/zoo_vision/data/config.json',
                         date="20250318")

    print("Cameras:", loader.load_camera_names())    
    print("Record root:", loader.record_root)

    track_dir = loader.load_offline_track_dir(cam_id="016")
    print("Track dir:", track_dir)

    lists = loader.list_track_files(track_dir=track_dir)
    print(lists)

