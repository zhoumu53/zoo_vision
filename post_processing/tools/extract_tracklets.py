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
    """Crop a bbox to a square with zero padding outside the image, then resize."""
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

    new_left = int(round(cx - half))
    new_top = int(round(cy - half))
    new_right = new_left + side
    new_bottom = new_top + side

    pad_left = max(0, -new_left)
    pad_top = max(0, -new_top)
    pad_right = max(0, new_right - w)
    pad_bottom = max(0, new_bottom - h)

    new_left = max(new_left, 0)
    new_top = max(new_top, 0)
    new_right = min(new_right, w)
    new_bottom = min(new_bottom, h)

    cropped = frame[new_top:new_bottom, new_left:new_right]
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        cropped = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    cropped = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)  # VideoWriter expects BGR


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


def export_tracklets(jsonl_path: Path, output_size: int = 512, bbox_format: str = "auto") -> Path:
    """Create per-track video clips and CSVs next to the JSONL."""
    meta, tracks_by_id = read_track_jsonl(jsonl_path)
    video_path = meta.get("video")
    fps = float(meta.get("fps", 30.0))
    meta_w = int(meta.get("width", 0))
    meta_h = int(meta.get("height", 0))

    tracks_dir = jsonl_path.parent / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    video_reader = load_video(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    crops_to_reid_training = tracks_dir / "crops_to_reid_training"
    crops_to_reid_training.mkdir(parents=True, exist_ok=True)
    video_name = jsonl_path.stem

    for track_id, entries in tracks_by_id.items():
        track_id_str = str(track_id)
        rows: List[Tuple[int, str, int, int, int, int, float]] = []
        writer = cv2.VideoWriter(
            str(tracks_dir / f"{track_id_str}.mkv"),
            fourcc,
            fps,
            (output_size, output_size),
        )

        ## TODO: save cropped frames into folder, under tracks_dir / crops

        crop_dir = crops_to_reid_training / f"track_{track_id_str}"
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
                crop_save_path = crop_dir / f"{video_name}_{frame_idx:06d}.jpg"
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

        save_track_csv(tracks_dir / f"{track_id_str}.csv", rows)
        logger.info("Saved track %s with %d frames", track_id_str, len(rows))

    return tracks_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract track clips and CSVs from tracking JSONL.")
    parser.add_argument("--jsonl", type=Path, default='/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20251129/20251129_01/ZAG-ELP-CAM-016-20251129-011949-1764375589549-7/ZAG-ELP-CAM-016-20251129-011949-1764375589549-7_tracks.jsonl', help="Path to tracking JSONL file.")
    parser.add_argument("--output-size", type=int, default=512, help="Square crop/resize size.")
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
