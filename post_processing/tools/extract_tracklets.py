"""
Extract per-track clips and CSVs from a tracking JSONL file.

Outputs:
  <json_parent>/tracks/{track_id}.mkv
  <json_parent>/tracks/{track_id}.csv
CSV columns: frame_id, timestamp, bbox_top, bbox_left, bbox_bottom, bbox_right, score
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import decord
import numpy as np
import torch

logger = logging.getLogger(__name__)

BBOX_LEFT, BBOX_TOP, BBOX_RIGHT, BBOX_BOTTOM = range(4)


class OpenCVVideoReader:
    """Minimal VideoReader fallback using OpenCV when decord cannot parse metadata."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = self._count_frames_safely()
        self._length = frame_count

    def _count_frames_safely(self) -> int:
        total = 0
        while True:
            ok, _ = self.cap.read()
            if not ok:
                break
            total += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Frame index {idx} out of range for {self._length} frames")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx} from {self.video_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame_rgb)

    def __del__(self) -> None:
        if hasattr(self, "cap"):
            self.cap.release()


def load_video(video_path: str):
    """Load video using decord with a safe fallback to OpenCV when metadata is broken."""
    decord.bridge.set_bridge("torch")
    try:
        return decord.VideoReader(video_path, ctx=decord.cpu(0))
    except Exception as err:  # pragma: no cover - runtime fallback
        logger.warning("Decord failed to read %s: %s. Falling back to OpenCV.", video_path, err)
        return OpenCVVideoReader(video_path)


def read_track_jsonl(
    jsonl_path: Path,
) -> tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """Parse a tracking JSONL and group detections by raw_track_id."""
    meta: Dict[str, Any] | None = None
    tracks_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with jsonl_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj:
                meta = obj["meta"]
                continue
            result = obj.get("results", obj)
            frame_idx = result.get("frame_idx")
            timestamp = result.get("timestamp")
            for track in result.get("tracks", []):
                track_id = str(track.get("raw_track_id"))
                tracks_by_id[track_id].append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "bbox": track.get("bbox"),
                        "score": track.get("score"),
                    }
                )

    if meta is None:
        raise ValueError(f"No meta line found in {jsonl_path}")

    for entries in tracks_by_id.values():
        entries.sort(key=lambda x: x["frame_idx"])

    return meta, tracks_by_id


def _to_numpy_rgb(frame: Any) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return frame


def crop_square_resize(frame: np.ndarray, bbox: Iterable[float], output_size: int = 512) -> np.ndarray:
    """
    Letterbox the bbox region to a square centered on the box, then resize.

    Keeps the box aspect ratio by padding to a square around its center before resizing.
    Pixels outside the bbox inside the square are zeroed (black), so only the
    detected area remains visible.
    """
    frame = _to_numpy_rgb(frame)
    h, w = frame.shape[:2]

    left = int(bbox[BBOX_LEFT])
    top = int(bbox[BBOX_TOP])
    right = int(bbox[BBOX_RIGHT])
    bottom = int(bbox[BBOX_BOTTOM])

    width = right - left
    height = bottom - top
    side = int(max(width, height))
    if side <= 0:
        raise ValueError(f"Invalid bbox with non-positive side: {bbox}")

    cx = left + width / 2.0
    cy = top + height / 2.0
    half = side / 2.0

    # Square around center
    square_left = int(round(cx - half))
    square_top = int(round(cy - half))
    square_right = square_left + side
    square_bottom = square_top + side

    # Prepare empty square (black background)
    square = np.zeros((side, side, frame.shape[2]), dtype=frame.dtype)

    # Source bbox clipped to image bounds
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, w)
    src_bottom = min(bottom, h)

    if src_right <= src_left or src_bottom <= src_top:
        raise ValueError(f"Invalid clipped bbox: {(src_left, src_top, src_right, src_bottom)} from {bbox}")

    # Destination coordinates in the square
    dst_left = src_left - square_left
    dst_top = src_top - square_top
    dst_right = dst_left + (src_right - src_left)
    dst_bottom = dst_top + (src_bottom - src_top)

    # Ensure destination indices are within square bounds
    dst_left_clipped = max(dst_left, 0)
    dst_top_clipped = max(dst_top, 0)
    dst_right_clipped = min(dst_right, side)
    dst_bottom_clipped = min(dst_bottom, side)

    src_left_adjusted = src_left + (dst_left_clipped - dst_left)
    src_top_adjusted = src_top + (dst_top_clipped - dst_top)
    src_right_adjusted = src_right - (dst_right - dst_right_clipped)
    src_bottom_adjusted = src_bottom - (dst_bottom - dst_bottom_clipped)

    if (
        src_right_adjusted <= src_left_adjusted
        or src_bottom_adjusted <= src_top_adjusted
        or dst_right_clipped <= dst_left_clipped
        or dst_bottom_clipped <= dst_top_clipped
    ):
        raise ValueError("BBox or square alignment resulted in empty region.")

    square[dst_top_clipped:dst_bottom_clipped, dst_left_clipped:dst_right_clipped] = frame[
        src_top_adjusted:src_bottom_adjusted, src_left_adjusted:src_right_adjusted
    ]

    square = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(square, cv2.COLOR_RGB2BGR)  # VideoWriter expects BGR


def save_track_csv(csv_path: Path, rows: List[Tuple[int, str, int, int, int, int, float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["frame_id", "timestamp", "bbox_top", "bbox_left", "bbox_bottom", "bbox_right", "score"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def to_xyxy(
    bbox: Iterable[float],
    frame_w: int,
    frame_h: int,
    bbox_format: str = "auto",
) -> Tuple[int, int, int, int]:
    """Convert bbox to xyxy (left, top, right, bottom)."""
    vals = list(bbox)
    if len(vals) != 4:
        raise ValueError(f"Expected 4 bbox values, got {vals}")

    def as_xyxy_xyxy():
        l, t, r, b = vals
        return int(l), int(t), int(r), int(b)

    def as_xyxy_xywh():
        x, y, w, h = vals
        return int(x), int(y), int(x + w), int(y + h)

    def valid(lb, tb, rb, bb) -> bool:
        return rb > lb and bb > tb and rb <= frame_w + 4 and bb <= frame_h + 4

    fmt = bbox_format.lower()
    if fmt == "xyxy":
        return as_xyxy_xyxy()
    if fmt == "xywh":
        return as_xyxy_xywh()
    if fmt != "auto":
        raise ValueError(f"bbox_format must be 'xyxy', 'xywh', or 'auto', got {bbox_format}")

    l1, t1, r1, b1 = as_xyxy_xyxy()
    l2, t2, r2, b2 = as_xyxy_xywh()
    xyxy_valid = valid(l1, t1, r1, b1)
    xywh_valid = valid(l2, t2, r2, b2)

    if xyxy_valid and not xywh_valid:
        return l1, t1, r1, b1
    if xywh_valid and not xyxy_valid:
        return l2, t2, r2, b2
    if xyxy_valid and xywh_valid:
        area_xyxy = (r1 - l1) * (b1 - t1)
        area_xywh = (r2 - l2) * (b2 - t2)
        return (l1, t1, r1, b1) if area_xyxy <= area_xywh else (l2, t2, r2, b2)

    l, t, r, b = l1, t1, r1, b1
    return (
        max(0, l),
        max(0, t),
        min(frame_w, max(r, l + 1)),
        min(frame_h, max(b, t + 1)),
    )


def export_tracklets(
    jsonl_path: Path,
    output_size: int = 512,
    bbox_format: str = "auto",
    output_dir: Path | None = None,
) -> Path:
    """Create per-track video clips and CSVs next to the JSONL or in a custom output directory."""
    meta, tracks_by_id = read_track_jsonl(jsonl_path)
    video_path = meta.get("video")
    fps = float(meta.get("fps", 30.0))
    meta_w = int(meta.get("width", 0))
    meta_h = int(meta.get("height", 0))

    tracks_dir = Path(output_dir) if output_dir is not None else jsonl_path.parent / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    video_reader = load_video(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    crops_to_reid_training = tracks_dir / "crops_to_reid_training"
    crops_to_reid_training.mkdir(parents=True, exist_ok=True)
    video_name = jsonl_path.stem

    for track_id, entries in tracks_by_id.items():
        start_time = entries[0]["timestamp"]
        # format time as hhmmss - remove yyymmdd if present '20251129_001949'
        if start_time and "_" in start_time:
            date_part, time_part = start_time.split("_", 1)
        track_filename = f"T{time_part}_ID{int(track_id):06d}"

        ## skip if track video already exists
        track_video_path = tracks_dir / f"{track_filename}.mkv"
        if track_video_path.exists():
            logger.info("Skipping existing track video: %s", track_video_path)
            continue

        rows: List[Tuple[int, str, int, int, int, int, float]] = []
        writer = cv2.VideoWriter(
            str(track_video_path),
            fourcc,
            fps,
            (output_size, output_size),
        )

        ## save cropped frames into folder, under tracks_dir / crops

        crop_dir = crops_to_reid_training / f"{track_filename}"
        crop_dir.mkdir(parents=True, exist_ok=True)

        try:
            for entry in entries:
                frame_idx = entry["frame_idx"]
                bbox = entry.get("bbox")
                timestamp = entry.get("timestamp", "")
                score = entry.get("score", 0.0)

                if bbox is None or len(bbox) != 4:
                    logger.warning("Skipping track %s frame %s due to invalid bbox %s", track_id, frame_idx, bbox)
                    continue

                try:
                    frame = video_reader[frame_idx]
                except Exception as err:
                    logger.warning("Skipping frame %s for track %s: %s", frame_idx, track_id, err)
                    continue

                fh, fw = frame.shape[:2]
                l, t, r, b = to_xyxy(bbox, frame_w=fw or meta_w, frame_h=fh or meta_h, bbox_format=bbox_format)
                normalized_bbox = (l, t, r, b)

                clipped = crop_square_resize(frame, normalized_bbox, output_size=output_size)
                writer.write(clipped)


                ### save to crop dir as well
                crop_save_path = crop_dir / f"{track_filename}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(crop_save_path), clipped)

                rows.append(
                    (
                        frame_idx,
                        timestamp,
                        int(t),
                        int(l),
                        int(b),
                        int(r),
                        float(score),
                    )
                )
        finally:
            writer.release()

        save_track_csv(tracks_dir / f"{track_filename}.csv", rows)
        logger.info("Saved track %s with %d frames", track_filename, len(rows))

    return tracks_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract track clips and CSVs from tracking JSONL.")
    parser.add_argument("--date", default='20251129', help="Input JSONL file (stitched)")
    parser.add_argument("--jsonl", type=Path, default='/media/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20251129/20251129_01/ZAG-ELP-CAM-016-20251129-011949-1764375589549-7/ZAG-ELP-CAM-016-20251129-011949-1764375589549-7_tracks.jsonl', help="Path to tracking JSONL file.")
    parser.add_argument("--output-size", type=int, default=224, help="Square crop/resize size.")
    parser.add_argument("--bbox-format", type=str, default="auto", choices=["auto", "xyxy", "xywh"], help="BBox format in JSON.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    export_tracklets(args.jsonl, output_size=args.output_size, bbox_format=args.bbox_format)


if __name__ == "__main__":
    main()
