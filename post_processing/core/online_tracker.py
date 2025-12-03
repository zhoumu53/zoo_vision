"""
Online Tracking Module

Handles real-time YOLO detection + ByteTrack tracking + ReID stitching + Behavior classification.
This is Stage 1 of the pipeline: Online tracking for individual videos.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    load_class_names,
    maybe_resize,
    run_yolo_byteTrack,
)


@dataclass
class DetectionResult:
    """Structure storing a single detection + reid match information."""

    bbox: Tuple[int, int, int, int] # x1, y1, x2, y2
    score: float | None
    cls_id: int | None
    cls_name: str | None
    track_id: int | None


class TrackJSONLogger:
    """Write per-frame tracking results to a JSONL file."""

    def __init__(
        self,
        path: Path,
        video: str,
        fps: float,
        width: int,
        height: int,
        class_names: List[str],
    ) -> None:
        self.path = path
        self.file = open(path, "w")
        
        metadata = {
            "video": video,
            "fps": fps,
            "width": width,
            "height": height,
            "class_names": class_names,
        }
        self.file.write(json.dumps(metadata) + "\n")

    def log_frame(
        self,
        frame_idx: int,
        tracks: List[Dict],
        frame_timestamp: str | None = None,
    ) -> None:
        record = {
            "frame_idx": frame_idx,
            "timestamp": frame_timestamp,
            "tracks": tracks,
        }
        self.file.write(json.dumps(self._ensure_json_serializable(record)) + "\n")

    def close(self) -> None:
        if self.file:
            self.file.close()

    def _ensure_json_serializable(self, obj):
        """Recursively ensure object is JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._ensure_json_serializable(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj


class OnlineTracker:
    """
    Online tracking pipeline: YOLO + ByteTrack only.
    
    Simplified version with only detection and tracking.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: Path,
        yolo_model_path: str,
        class_names_path: str,
        yolo_device: str = "cuda",
        conf_thres: float = 0.65,
        iou_thres: float = 0.65,
        max_dets: int = 50,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        resize_width: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        
        self.yolo_device = yolo_device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_dets = max_dets
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        self.resize_width = resize_width

        self.class_names = load_class_names(class_names_path)
        from ultralytics import YOLO
        
        self.logger.info("Loading YOLO model from: %s", yolo_model_path)
        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.logger.info("YOLO model instance created successfully")
        except Exception as e:
            self.logger.error("Failed to create YOLO model: %s", str(e), exc_info=True)
            raise
        
        # Validate device availability - force CPU to avoid segfault
        # There appears to be a compatibility issue with CUDA for this YOLO model
        if self.yolo_device and self.yolo_device.lower() != "cpu":
            self.logger.warning("CUDA device requested but forcing CPU due to known segfault issue")
            self.logger.warning("Original device requested: %s", self.yolo_device)
            self.yolo_device = "cpu"
        else:
            self.logger.info("Using CPU for YOLO model")
            self.yolo_device = "cpu"
        
        self.logger.info("YOLO model ready on device: %s", self.yolo_device)

    def run(self, skip_if_exists: bool = True) -> Path:
        """
        Run YOLO + ByteTrack tracking pipeline.
        
        Args:
            skip_if_exists: If True, skip processing if valid output already exists
        
        Returns:
            Path to the output JSONL file
        """

        self.logger.info("=" * 80)
        self.logger.info("YOLO + ByteTrack Pipeline")
        self.logger.info("=" * 80)
        self.logger.info("Video: %s", self.video_path)
        self.logger.info("Output: %s", self.output_dir)
        
        video_name = Path(self.video_path).stem
        jsonl_path = self.output_dir / f"{video_name}_tracks.jsonl"

        self.logger.info("ready to process video and save to %s", jsonl_path)
        
        if skip_if_exists and self._is_valid_output(jsonl_path):
            self.logger.warning("=" * 80)
            self.logger.warning("OUTPUT ALREADY EXISTS - SKIPPING PROCESSING")
            self.logger.warning("=" * 80)
            self.logger.warning("Output file: %s", jsonl_path)
            self.logger.warning("To reprocess, use --no-skip-existing flag")
            self.logger.warning("=" * 80)
            return jsonl_path
        
        vr = VideoReader(self.video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        width, height = vr[0].shape[1], vr[0].shape[0]
        
        self.logger.info("Video info: %d frames, %.2f fps, %dx%d", total_frames, fps, width, height)
        
        video_start = self._parse_video_start_datetime(self.video_path)
        
        json_logger = TrackJSONLogger(
            jsonl_path,
            self.video_path,
            fps,
            width,
            height,
            self.class_names,
        )
        
        processed_frames = 0
        max_to_process = min(total_frames, self.max_frames) if self.max_frames else total_frames
        
        self.logger.info("Starting frame processing: %d frames to process", max_to_process // self.frame_skip)
        
        with tqdm(total=max_to_process // self.frame_skip, desc="Processing") as pbar:
            for frame_idx in range(0, max_to_process, self.frame_skip):
                try:
                    if processed_frames % 100 == 0:
                        self.logger.info("Processing frame %d/%d", frame_idx, max_to_process)
                    
                    frame = vr[frame_idx].asnumpy()
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    if self.resize_width:
                        frame_bgr = maybe_resize(frame_bgr, self.resize_width)
                        self.logger.debug("Resized frame to width %d", self.resize_width)
                    
                    self.logger.debug("Running YOLO detection on frame %d, shape: %s", frame_idx, frame_bgr.shape)
                    boxes, scores, cls_ids, track_ids = run_yolo_byteTrack(
                        self.yolo_model,
                        frame_bgr,
                        conf_thres=self.conf_thres,
                        iou_thres=self.iou_thres,
                        max_dets=self.max_dets,
                        tracker_cfg='bytetrack.yaml',
                        device=self.yolo_device,
                    )
                    
                    if processed_frames % 100 == 0:
                        self.logger.info("Frame %d: Found %d detections", frame_idx, len(boxes))
                    
                    # Convert to DetectionResult objects
                    detections = []
                    for i in range(len(boxes)):
                        cls_id = int(cls_ids[i]) if i < len(cls_ids) else -1
                        cls_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"id_{cls_id}"
                        det = DetectionResult(
                            bbox=boxes[i].astype(int).tolist(),
                            score=float(scores[i]) if i < len(scores) else 0.0,
                            cls_id=cls_id,
                            cls_name=cls_name,
                            track_id=int(track_ids[i]) if i < len(track_ids) else -1,
                        )
                        detections.append(det)
                    
                    frame_timestamp = self._format_frame_timestamp(video_start, frame_idx, fps)
                    tracks = self._detections_to_json(detections)
                    json_logger.log_frame(frame_idx, tracks, frame_timestamp)
                    
                    processed_frames += 1
                    pbar.update(1)
                    
                except Exception as e:
                    self.logger.error("Error processing frame %d: %s", frame_idx, str(e), exc_info=True)
                    # Continue to next frame instead of crashing
                    processed_frames += 1
                    pbar.update(1)
                    continue
        
        json_logger.close()
        
        self.logger.info("YOLO + ByteTrack tracking complete. Output: %s", jsonl_path)
        return jsonl_path



    def _detections_to_json(self, detections: List[DetectionResult]) -> List[Dict]:
        """Convert detections to JSON-serializable format."""
        tracks = []
        for det in detections:
            tracks.append({
                "track_id": int(det.track_id) if det.track_id is not None else -1,
                "bbox": [int(x) for x in det.bbox],
                "score": float(det.score) if det.score is not None else 0.0,
                "class_id": int(det.cls_id) if det.cls_id is not None else -1,
                "class_name": det.cls_name,
            })
        
        return tracks

    def _parse_video_start_datetime(self, video_path: str) -> Optional[datetime]:
        """Parse start datetime from video filename."""
        stem = Path(video_path).stem
        parts = stem.split("-")
        if len(parts) >= 6:
            try:
                date_str = parts[4]  # YYYYMMDD
                time_str = parts[5]  # HHMMSS
                dt_str = f"{date_str}{time_str}"
                return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
            except Exception:
                pass
        return None

    def _format_frame_timestamp(
        self,
        video_start: Optional[datetime],
        frame_idx: int,
        fps: float,
    ) -> str:
        """Format frame timestamp for logging."""
        offset_seconds = frame_idx / fps
        
        if video_start is not None:
            frame_time = video_start + timedelta(seconds=offset_seconds)
            return frame_time.strftime("%Y%m%d_%H%M%S")
        
        total_ms = int(offset_seconds * 1000)
        seconds, millis = divmod(total_ms, 1000)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f"{hours:02d}{minutes:02d}{seconds:02d}_{millis:03d}"

    def _is_valid_output(self, jsonl_path: Path) -> bool:
        """Check if output file exists and is valid."""
        if not jsonl_path.exists():
            return False
        
        try:
            with open(jsonl_path, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:
                    return False
                import json
                json.loads(lines[0])
                return True
        except Exception:
            return False

