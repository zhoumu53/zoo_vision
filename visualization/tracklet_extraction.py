from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrameDetection:
    frame_idx: int
    bbox: Tuple[float, float, float, float]
    frame_name: Optional[str] = None
    behavior: str = "unknown"


def _frame_column(tracks: pd.DataFrame) -> str:
    for candidate in ("frame_idx", "frame_id"):
        if candidate in tracks.columns:
            return candidate
    raise ValueError("tracks JSON must contain either 'frame_idx' or 'frame_id'.")


def _normalize_bbox(raw_bbox) -> Tuple[float, float, float, float]:
    if raw_bbox is None:
        raise ValueError("Missing bbox entry.")
    if isinstance(raw_bbox, dict):
        if all(k in raw_bbox for k in ("x1", "y1", "x2", "y2")):
            return (
                float(raw_bbox["x1"]),
                float(raw_bbox["y1"]),
                float(raw_bbox["x2"]),
                float(raw_bbox["y2"]),
            )
        if all(k in raw_bbox for k in ("left", "top", "width", "height")):
            left = float(raw_bbox["left"])
            top = float(raw_bbox["top"])
            return (left, top, left + float(raw_bbox["width"]), top + float(raw_bbox["height"]))
    if isinstance(raw_bbox, (list, tuple)):
        values = list(raw_bbox)
        if len(values) == 4:
            return tuple(float(v) for v in values)  # type: ignore[return-value]
    raise ValueError(f"Unsupported bbox format: {raw_bbox}")


def _behavior_label(row: pd.Series) -> str:
    if "behavior" not in row or pd.isna(row["behavior"]):
        if "behavior_label" in row and isinstance(row["behavior_label"], str):
            return row["behavior_label"] or "unknown"
        return "unknown"
    behavior = row["behavior"]
    if isinstance(behavior, str):
        return behavior or "unknown"
    if isinstance(behavior, dict):
        label = behavior.get("label") or behavior.get("behavior") or behavior.get("name")
        if isinstance(label, str) and label:
            return label
    return "unknown"


def _sanitize_behavior(name: str) -> str:
    base = (name or "unknown").strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]+", "_", base)
    cleaned = cleaned.strip("_")
    return cleaned if cleaned else "unknown"


def _stable_window_size(
    boxes: Iterable[Tuple[float, float, float, float]], padding_ratio: float
) -> Tuple[int, int]:
    boxes = list(boxes)
    if not boxes:
        return 0, 0
    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    min_x1 = min(box[0] for box in boxes)
    max_x2 = max(box[2] for box in boxes)
    min_y1 = min(box[1] for box in boxes)
    max_y2 = max(box[3] for box in boxes)
    span_w = max_x2 - min_x1
    span_h = max_y2 - min_y1
    base_w = max(span_w, max(widths))
    base_h = max(span_h, max(heights))
    pad_w = base_w * padding_ratio
    pad_h = base_h * padding_ratio
    stable_w = int(round(base_w + 2 * pad_w))
    stable_h = int(round(base_h + 2 * pad_h))
    return max(stable_w, 1), max(stable_h, 1)


def _stable_center(boxes: Iterable[Tuple[float, float, float, float]]) -> Tuple[float, float]:
    boxes = list(boxes)
    if not boxes:
        return 0.0, 0.0
    centers_x = [(box[0] + box[2]) * 0.5 for box in boxes]
    centers_y = [(box[1] + box[3]) * 0.5 for box in boxes]
    return float(sum(centers_x) / len(centers_x)), float(sum(centers_y) / len(centers_y))


def _crop_with_center(
    frame, cx: float, cy: float, width: int, height: int
) -> Tuple[int, int, int, int]:
    frame_height, frame_width = frame.shape[:2]
    half_w = width / 2.0
    half_h = height / 2.0
    x1 = int(round(cx - half_w))
    y1 = int(round(cy - half_h))
    x2 = int(round(cx + half_w))
    y2 = int(round(cy + half_h))

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_width:
        shift = x2 - frame_width
        x1 = max(0, x1 - shift)
        x2 = frame_width
    if y2 > frame_height:
        shift = y2 - frame_height
        y1 = max(0, y1 - shift)
        y2 = frame_height

    x1 = max(0, min(frame_width, x1))
    x2 = max(0, min(frame_width, x2))
    y1 = max(0, min(frame_height, y1))
    y2 = max(0, min(frame_height, y2))
    return x1, y1, x2, y2


def _parse_json_lines(text: str, path: Path) -> List[Any]:
    records: List[Any] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON line {line_no} in {path}: {exc}") from exc
    return records


def _load_tracks(tracks_json_path: str) -> List[dict]:
    path = Path(tracks_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Tracks file not found: {tracks_json_path}")
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        entries = _parse_json_lines(text, path)
    else:
        stripped = text.lstrip()
        if not stripped:
            entries = []
        else:
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError:
                entries = _parse_json_lines(text, path)
            else:
                if isinstance(loaded, list):
                    entries = loaded
                else:
                    entries = [loaded]

    track_entries: List[dict] = []

    def _capture_tracks(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, list):
            for item in obj:
                _capture_tracks(item)
            return
        if not isinstance(obj, dict):
            return
        if "meta" in obj and isinstance(obj["meta"], dict):
            return
        if obj.get("type") == "meta":
            return
        if "results" in obj and isinstance(obj["results"], dict):
            _capture_tracks(obj["results"])
            return
        if "tracks" in obj and isinstance(obj["tracks"], list):
            frame_idx = obj.get("frame_idx")
            timestamp = obj.get("timestamp")
            for track in obj["tracks"]:
                if not isinstance(track, dict):
                    continue
                record = dict(track)
                if frame_idx is not None:
                    record.setdefault("frame_idx", frame_idx)
                if timestamp is not None:
                    record.setdefault("timestamp", timestamp)
                track_entries.append(record)
            return
        track_entries.append(obj)

    for entry in entries:
        _capture_tracks(entry)

    return track_entries


def extract_tracklets(
    tracks_json_path: str,
    video_path: str,
    output_dir: str,
    track_id: Optional[int] = None,
    max_frames: Optional[int] = None,
    padding_ratio: float = 0.15,
):
    """Extract tracklets from video based on tracks JSON file.

    Args:
        tracks_json_path (str): Path to the tracks JSON file.
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted tracklet images.
        track_id (Optional[int]): Specific track ID to extract. If None, extract all tracks.
        max_frames (Optional[int]): Maximum number of frames to process. If None, process all frames.
        padding_ratio (float): Extra scale applied when building stable crops to avoid jitter.
    """
    os.makedirs(output_dir, exist_ok=True)

    tracks_data = _load_tracks(tracks_json_path)
    print(f"Loaded {len(tracks_data)} track entries from {tracks_json_path}")
    tracks_df = pd.DataFrame(tracks_data)
    if tracks_df.empty:
        print("No tracks available in the provided JSON.")
        return
    
    print(tracks_df.columns.tolist())
        
    track_col = "raw_track_id"
    frame_col = _frame_column(tracks_df)

    if track_id is not None:
        tracks_df = tracks_df[tracks_df[track_col] == track_id]

    if max_frames is not None:
        tracks_df = tracks_df[tracks_df[frame_col] < max_frames]

    if tracks_df.empty:
        print("No tracks found for the specified criteria.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video at {video_path}")

    raw_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames: Optional[int] = raw_total_frames if raw_total_frames > 0 else None
    if max_frames is not None and total_frames is not None:
        frame_limit = min(max_frames, total_frames)
    elif max_frames is not None:
        frame_limit = max_frames
    else:
        frame_limit = total_frames

    grouped: Dict[int | str, List[FrameDetection]] = {}
    for tid, group in tracks_df.groupby(track_col):
        tid_int = int(tid) if isinstance(tid, (int, float)) or str(tid).isdigit() else tid
        entries: List[FrameDetection] = []
        for _, row in group.iterrows():
            bbox = _normalize_bbox(row.get("bbox"))
            frame_idx = int(row[frame_col])
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if frame_limit is not None and frame_idx >= frame_limit:
                continue
            entries.append(
                FrameDetection(
                    frame_idx=frame_idx,
                    bbox=bbox,
                    frame_name=row.get("frame_name"),
                    behavior=_behavior_label(row),
                )
            )
        if entries:
            entries.sort(key=lambda det: det.frame_idx)
            grouped[tid_int] = entries

    if not grouped:
        print("No usable track entries found after filtering.")
        cap.release()
        return

    frame_requests: Dict[int, List[Tuple[int | str, FrameDetection, Tuple[int, int], Tuple[float, float]]]] = {}
    saved_summary: Dict[int | str, int] = {}

    for tid, detections in grouped.items():
        boxes = [det.bbox for det in detections]
        crop_w, crop_h = _stable_window_size(boxes, padding_ratio)
        if crop_w == 0 or crop_h == 0:
            continue
        center_x, center_y = _stable_center(boxes)
        saved_summary[tid] = 0
        for det in detections:
            frame_idx = det.frame_idx
            if frame_limit is not None and frame_idx >= frame_limit:
                continue
            requests = frame_requests.setdefault(frame_idx, [])
            requests.append((tid, det, (crop_w, crop_h), (center_x, center_y)))

    ordered_indices = sorted(frame_requests.keys())
    current_position: Optional[int] = None

    for frame_idx in ordered_indices:
        if current_position is None or frame_idx < current_position:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_position = frame_idx

        failed = False
        while current_position is not None and current_position < frame_idx:
            success, _ = cap.read()
            if not success:
                failed = True
                break
            current_position += 1

        if failed or current_position is None:
            print(f"Warning: could not seek to frame {frame_idx}.")
            continue

        success, frame = cap.read()
        if not success:
            print(f"Warning: could not read frame {frame_idx}.")
            continue
        current_position = frame_idx + 1
        for tid, det, (crop_w, crop_h), (center_x, center_y) in frame_requests[frame_idx]:
            x1, y1, x2, y2 = _crop_with_center(frame, center_x, center_y, crop_w, crop_h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            frame_name = det.frame_name or f"frame_{frame_idx:06d}"
            behavior_dir = Path(output_dir) / _sanitize_behavior(det.behavior)
            track_dir = behavior_dir / f"tracklets_{tid}"
            track_dir.mkdir(parents=True, exist_ok=True)
            output_path = track_dir / f"{frame_name}"
            cv2.imwrite(str(output_path), crop)
            saved_summary[tid] += 1

    cap.release()
    for tid, count in saved_summary.items():
        print(f"Saved {count} crops for track {tid}.")
    print(f"Tracklets extracted to {output_dir}")



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract tracklets from video based on tracks JSON file."
    )
    parser.add_argument("tracks_json", type=str, help="Path to the tracks JSON file.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, help="Directory to save extracted tracklet images.")
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Specific track ID to extract. If not set, extract all tracks.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process. If not set, process all frames.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.15,
        help="Extra scale applied when building stable crops to avoid jitter.",
    )

    args = parser.parse_args()

    extract_tracklets(
        tracks_json_path=args.tracks_json,
        video_path=args.video_path,
        output_dir=args.output_dir,
        track_id=args.track_id,
        max_frames=args.max_frames,
        padding_ratio=args.padding_ratio,
    )



# python visualization/tracklet_extraction.py \
#     '/home/mu/Desktop/comparison_videos/video_tracks_reid_improved_with_behavior_4cams/20240905/ZAG-ELP-CAM-016-20240905-024719-1725497239539-7/ZAG-ELP-CAM-016-20240905-024719-1725497239539-7_tracks/ZAG-ELP-CAM-016-20240905-024719-1725497239539-7_tracks.jsonl'\
#     '/mnt/camera_nas/ZAG-ELP-CAM-016/20240905AM/ZAG-ELP-CAM-016-20240905-024719-1725497239539-7.mp4' \
#     '/media/mu/test' \
#     --max-frames 5000 
