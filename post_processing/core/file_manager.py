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
        hour: int,
        cam_id: Optional[str] = None,
        raw_video_dir: str = "/mnt/camera_nas",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize FileManager.
        
        """
        self.date = date
        self.hour = hour
        self.cam_id = cam_id
        self.raw_video_dir = Path(raw_video_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate date format
        try:
            datetime.strptime(date, "%Y%m%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date}. Expected YYYYMMDD")
        
        # Validate hour
        if not 0 <= hour <= 23:
            raise ValueError(f"Invalid hour: {hour}. Expected 0-23")
        
        # Determine AM/PM
        self.ampm = "AM" if hour < 12 else "PM"
        
        self.logger.info("FileLoader initialized: date=%s, hour=%d (%s), cam_id=%s",
                        date, hour, self.ampm, cam_id or "all")
    

    def get_online_tracking_files(
        self,
        camera_ids: List[str],
        oneline_outdir: Path = None,
        date: Optional[str] = None,
        hour: Optional[int] = None,
    ) -> List[Path]:
        """
        """
        if date is None:
            date = self.date
        if hour is None:
            hour = self.hour
            
        tracking_files: List[Path] = []
        
        for cam_id in camera_ids:
            # Search for tracking files matching pattern
            pattern = f"ZAG-ELP-CAM-{cam_id}-{self.date}-*_tracks.jsonl"
            all_tracks = sorted(oneline_outdir.glob(pattern))
            
            # Also search in subdirectories
            all_tracks.extend(sorted(oneline_outdir.glob(f"*/{pattern}")))
            
            for track_path in all_tracks:
                # Extract time from filename
                video_time = self._extract_time_from_jsonl_path(track_path)
                if video_time and self._is_within_time_window(
                    video_time, self.hour, time_window_minutes
                ):
                    tracking_files.append(track_path)
                    self.logger.debug("Found tracking file: %s", track_path)
        
        self.logger.info("Found %d online tracking files", len(tracking_files))
        return tracking_files
    
    def find_offline_stitching_file(self) -> Optional[Path]:
        """
        Find offline stitching result file.
        
        Searches for: stitched_tracks.jsonl in offline_output_dir
        
        Returns:
            Path to stitched_tracks.jsonl or None if not found
        
        Example:
            >>> loader = FileLoader("20250830", 14, 
            ...                    offline_output_dir="/path/to/offline_results")
            >>> stitched = loader.find_offline_stitching_file()
        """
        if not self.offline_output_dir:
            self.logger.warning("offline_output_dir not set")
            return None
        
        stitched_file = self.offline_output_dir / "stitched_tracks.jsonl"
        
        if stitched_file.exists():
            self.logger.info("Found offline stitching file: %s", stitched_file)
            return stitched_file
        
        self.logger.warning("Offline stitching file not found: %s", stitched_file)
        return None
    
    def find_behavior_analysis_files(self) -> Dict[str, Optional[Path]]:
        """
        Find behavior analysis result files.
        
        Searches for:
        - behavior_timeline_raw.csv
        - behavior_segments.csv
        - behavior_summary.txt
        
        Returns:
            Dictionary mapping file types to paths
        
        Example:
            >>> loader = FileLoader("20250830", 14,
            ...                    behavior_output_dir="/path/to/behavior_results")
            >>> files = loader.find_behavior_analysis_files()
            >>> timeline = files["timeline"]
            >>> segments = files["segments"]
        """
        if not self.behavior_output_dir:
            self.logger.warning("behavior_output_dir not set")
            return {"timeline": None, "segments": None, "summary": None}
        
        if not self.behavior_output_dir.exists():
            self.logger.warning("Behavior output directory does not exist: %s",
                              self.behavior_output_dir)
            return {"timeline": None, "segments": None, "summary": None}
        
        files = {
            "timeline": self.behavior_output_dir / "behavior_timeline_raw.csv",
            "segments": self.behavior_output_dir / "behavior_segments.csv",
            "summary": self.behavior_output_dir / "behavior_summary.txt",
        }
        
        result = {}
        for key, path in files.items():
            if path.exists():
                result[key] = path
                self.logger.info("Found %s file: %s", key, path)
            else:
                result[key] = None
                self.logger.warning("%s file not found: %s", key, path)
        
        return result
    
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
    
    def load_offline_stitching_jsonl(
        self,
        jsonl_path: Optional[Path] = None,
    ) -> Tuple[Dict, List[Dict]]:
        """
        Load offline stitching JSONL file.
        
        Args:
            jsonl_path: Path to stitched JSONL file. If None, uses find_offline_stitching_file()
        
        Returns:
            Tuple of (metadata, records)
        
        Example:
            >>> loader = FileLoader("20250830", 14,
            ...                    offline_output_dir="/path/to/offline_results")
            >>> metadata, records = loader.load_offline_stitching_jsonl()
        """
        if jsonl_path is None:
            jsonl_path = self.find_offline_stitching_file()
            if jsonl_path is None:
                raise FileNotFoundError("Offline stitching file not found")
        
        return self.load_online_tracking_jsonl(jsonl_path)
    
    def load_behavior_timeline_csv(
        self,
        csv_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Load behavior timeline CSV file.
        
        Args:
            csv_path: Path to CSV file. If None, uses find_behavior_analysis_files()
        
        Returns:
            DataFrame with behavior timeline data
        
        Example:
            >>> loader = FileLoader("20250830", 14,
            ...                    behavior_output_dir="/path/to/behavior_results")
            >>> df = loader.load_behavior_timeline_csv()
            >>> print(df.head())
        """
        if csv_path is None:
            files = self.find_behavior_analysis_files()
            csv_path = files.get("timeline")
            if csv_path is None:
                raise FileNotFoundError("Behavior timeline CSV not found")
        
        df = pd.read_csv(csv_path)
        
        # Convert timestamp to datetime if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        self.logger.info("Loaded behavior timeline: %d rows", len(df))
        return df
    
    def load_behavior_segments_csv(
        self,
        csv_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Load behavior segments CSV file.
        
        Args:
            csv_path: Path to CSV file. If None, uses find_behavior_analysis_files()
        
        Returns:
            DataFrame with behavior segments data
        
        Example:
            >>> loader = FileLoader("20250830", 14,
            ...                    behavior_output_dir="/path/to/behavior_results")
            >>> df = loader.load_behavior_segments_csv()
            >>> print(df.head())
        """
        if csv_path is None:
            files = self.find_behavior_analysis_files()
            csv_path = files.get("segments")
            if csv_path is None:
                raise FileNotFoundError("Behavior segments CSV not found")
        
        df = pd.read_csv(csv_path)
        
        # Convert timestamps to datetime if present
        for col in ["start_timestamp", "end_timestamp"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        self.logger.info("Loaded behavior segments: %d rows", len(df))
        return df
    
    def get_all_videos_for_day(
        self,
        cam_id: Optional[str] = None,
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
        camera_ids = [cam_id] if cam_id else self.get_camera_ids()
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
    
    # Helper methods
    
    def _extract_time_from_video_path(self, video_path: Path) -> Optional[int]:
        """Extract hour from video path."""
        try:
            parts = video_path.stem.split("-")
            if len(parts) >= 6:
                time_str = parts[5]  # HHMMSS
                return int(time_str[:2])  # Hour
        except (ValueError, IndexError):
            pass
        return None
    
    def _extract_time_from_jsonl_path(self, jsonl_path: Path) -> Optional[int]:
        """Extract hour from JSONL path (removes _tracks.jsonl suffix)."""
        stem = jsonl_path.stem.replace("_tracks", "")
        try:
            parts = stem.split("-")
            if len(parts) >= 6:
                time_str = parts[5]  # HHMMSS
                return int(time_str[:2])  # Hour
        except (ValueError, IndexError):
            pass
        return None
    
    def _is_within_time_window(
        self,
        video_hour: int,
        target_hour: int,
        window_minutes: int,
    ) -> bool:
        """Check if video hour is within time window of target hour."""
        # Simple hour-based check (can be enhanced for minute-level precision)
        window_hours = window_minutes / 60.0
        return abs(video_hour - target_hour) <= window_hours
    
    def __repr__(self) -> str:
        return (f"FileLoader(date={self.date}, hour={self.hour}, "
                f"cam_id={self.cam_id}, ampm={self.ampm})")
