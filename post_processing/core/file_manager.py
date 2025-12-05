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


class FileManager:
    """
    """
    
    def __init__(
        self,
        date: str,
        raw_video_dir: str = "/mnt/camera_nas",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize FileManager.
        
        """
        self.date = date
        self.raw_video_dir = Path(raw_video_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate date format
        try:
            datetime.strptime(date, "%Y%m%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date}. Expected YYYYMMDD")
        

    def get_online_tracking_json_path(
        self,
        cam_id: str,
        hour: Optional[int] = None,
        outdir: Path = Path('/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams'),
    ) -> Path:
        if hour is None:
            raise ValueError("hour is required to locate online tracking results")
        
        json_path = outdir / Path(f"{self.date}/{self.date}_{hour:02d}/ZAG-ELP-CAM-{cam_id}-{self.date}-{hour:02d}*/*.jsonl")
        # Find the first matching file
        matching_files = sorted(outdir.glob(f"{self.date}/{self.date}_{hour:02d}/ZAG-ELP-CAM-{cam_id}-{self.date}-{hour:02d}*/*.jsonl"))
        if not matching_files:
            raise FileNotFoundError(f"No online tracking JSONL file found for camera(s) {cam_id} at hour {hour} on date {self.date}")
        json_path = matching_files[0]
        print(f"Found online tracking JSONL file: {json_path}")
        
        return json_path


    def load_online_tracking_jsonl(self, jsonl_path: Path) -> Tuple[Dict, List[Dict]]:
        """
        Load online tracking JSONL file.
        
        Args:
            jsonl_path: Path to tracking JSONL file
        
        Returns:
            Tuple of (metadata, records)
            - metadata: Dict with video info (first line)
            - records: List of frame records
        
        Example:
            >>> loader = FileLoader("20250830", 14)
            >>> metadata, records = loader.load_online_tracking_jsonl(track_path)
            >>> print(f"Video: {metadata['video']}, FPS: {metadata['fps']}")
            >>> print(f"Frames: {len(records)}")
        """
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
        
        if len(lines) < 1:
            raise ValueError(f"Invalid JSONL file: {jsonl_path}")
        
        metadata = json.loads(lines[0])
        records = [json.loads(line) for line in lines[1:]]
        
        self.logger.info("Loaded %d frames from %s", len(records), jsonl_path)
        return metadata, records
    
    
    def load_video_metadata(self, video_path: Path) -> Dict:
        """
        Extract metadata from video file path.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with metadata:
            - camera_id: str
            - date: str (YYYYMMDD)
            - time: str (HHMMSS)
            - timestamp: int (milliseconds)
            - segment: int
            - ampm: str (AM/PM)
        
        Example:
            >>> loader = FileLoader("20250830", 14)
            >>> metadata = loader.load_video_metadata(Path("ZAG-ELP-CAM-016-20250830-140000-123456-7.mp4"))
            >>> print(metadata)
            {'camera_id': '016', 'date': '20250830', 'time': '140000', ...}
        """
        filename = video_path.stem
        parts = filename.split("-")
        
        if len(parts) < 7:
            raise ValueError(f"Invalid video filename format: {filename}")
        
        camera_id = parts[3]
        date = parts[4]
        time = parts[5]
        timestamp = int(parts[6])
        segment = int(parts[7]) if len(parts) > 7 else 0
        
        hour = int(time[:2])
        ampm = "AM" if hour < 12 else "PM"
        
        return {
            "camera_id": camera_id,
            "date": date,
            "time": time,
            "timestamp": timestamp,
            "segment": segment,
            "ampm": ampm,
            "filename": filename,
            "path": str(video_path),
        }
    
    

    def get_all_videos_for_day(
        self,
        camera_ids: List[str] = None,
    ) -> List[Path]:
        """
        Get all videos for the entire day (AM and PM).
        
        Args:
            cam_id: Camera ID to search. If None, uses instance cam_id or all cameras.
        
        Returns:
            List of video file paths for the entire day
        
        Example:
            >>> loader = FileLoader("20250830", 14, "016")
            >>> all_videos = loader.get_all_videos_for_day()
            >>> print(f"Found {len(all_videos)} videos for the day")
        """
        
        video_files: List[Path] = []
        
        for cam_id in camera_ids:
            for ampm in ["AM", "PM"]:
                video_dir = self.raw_video_dir / f"ZAG-ELP-CAM-{cam_id}" / f"{self.date}{ampm}"
                
                if not video_dir.exists():
                    continue
                
                pattern = f"ZAG-ELP-CAM-{cam_id}-{self.date}-*.mp4"
                videos = sorted(video_dir.glob(pattern))
                video_files.extend(videos)
        
        self.logger.info("Found %d videos for entire day", len(video_files))
        return video_files
    


if __name__ == "__main__":
    # Example usage
    loader = FileManager("20251129")
    

    cam_ids = ["016", "017"]
    hour = 00
    # for cam_id in cam_ids:
    #     online_tracking_file = loader.get_online_tracking_json_path(cam_id, hour=hour)

    #     meta, records = loader.load_online_tracking_jsonl(online_tracking_file)
        
    #     print(records[0])

    #     print("=====")

    all_videos = loader.get_all_videos_for_day(camera_ids=cam_ids)
    for video in all_videos:
        print(video)
    
