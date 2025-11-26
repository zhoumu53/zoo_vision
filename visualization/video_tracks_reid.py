"""
Video visualization script for Elephant ReID.

The script loads:
  1. A video file
  2. A YOLO segmentation/detection model (Ultralytics)
  3. A pretrained PoseGuidedReID model checkpoint
  4. A gallery features npz file generated via PoseGuidedReID inference

Pipeline:
  Video frame -> YOLO detection -> crop patches -> ReID feature extraction -> cosine similarity
  -> draw annotations -> save annotated video.
"""


from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu, gpu
from tqdm import tqdm
from utils import *

LABEL_TO_FOLDER: Dict[str, str] = {}
for identity in DEFAULT_IDENTITY_NAMES:
    parts = identity.split("_", 1)
    if len(parts) == 2:
        LABEL_TO_FOLDER[parts[1]] = identity
    LABEL_TO_FOLDER[identity] = identity

MATCH_THUMB_SIZE = 140
MATCH_PANEL_PADDING = 14



# def format_frame_timestamp(
#     video_start: datetime | None,
#     frame_idx: int,
#     fps: float,
# ) -> str:
#     """Return a timestamp string for a frame index using video metadata when available."""
#     effective_fps = fps if fps > 0 else 30.0
#     offset_seconds = frame_idx / max(effective_fps, 1e-6)
#     if video_start is not None:
#         frame_time = video_start + timedelta(seconds=offset_seconds)
#         return frame_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
#     total_ms = int(offset_seconds * 1000)
#     seconds, millis = divmod(total_ms, 1000)
#     hours, seconds = divmod(seconds, 3600)
#     minutes, seconds = divmod(seconds, 60)
#     return f"{hours:02d}{minutes:02d}{seconds:02d}_{millis:03d}"


# def build_frame_output_path(
#     frames_dir: Path,
#     video_path: str,
#     frame_idx: int,
#     fps: float,
#     video_start: datetime | None,
# ) -> Path:
#     """Generate a deterministic frame path keeping the original video stem."""
#     base_name = Path(video_path).stem or "frame"
#     timestamp = format_frame_timestamp(video_start, frame_idx, fps)
#     return frames_dir / f"{base_name}_{timestamp}.jpg"


def _file2date(file_path: str) -> str:
    """Extract YYYY_MM portion from gallery file name."""
    segments = file_path.split("_")
    token = segments[4] if len(segments) > 4 else segments[-1]
    token = token.split(".")[0]
    if len(token) < 6:
        return token
    return f"{token[:4]}_{token[4:6]}"


@lru_cache(maxsize=512)
def _load_gallery_thumbnail(path: str, size: int) -> np.ndarray:
    """Load and resize a gallery image for visualization."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def annotate_frame_with_matches(
    frame: np.ndarray,
    detections: List[DetectionResult],
    matched_paths: List[List[str]],
    gallery: GalleryDB,
    top_k: int,
    thumb_size: int,
    padding: int,
) -> np.ndarray:
    """
    Append a right-hand panel that visualizes the top-K gallery matches per detection.
    """
    effective_top_k = max(top_k, max((len(row) for row in matched_paths), default=0))
    if effective_top_k <= 0:
        return frame

    panel_width = effective_top_k * thumb_size + (effective_top_k + 1) * padding
    panel = np.full((frame.shape[0], panel_width, 3), 25, dtype=np.uint8)
    y = padding
    block_height = thumb_size + padding + 24

    if not detections or not matched_paths:
        return np.hstack([frame, panel])

    for det_idx, detection in enumerate(detections):
        if det_idx >= len(matched_paths):
            break
        matches = detection.matches or []
        row_paths = matched_paths[det_idx]
        if not row_paths or not matches:
            continue

        for rank, path in enumerate(row_paths[:top_k]):
            label = matches[rank][0] if rank < len(matches) else "Unknown"
            display_label = LABEL_TO_FOLDER.get(label, label)
            color = gallery.label_to_color.get(label, (0, 255, 0))
            abs_path = str(path)
            thumb = _load_gallery_thumbnail(abs_path, thumb_size)
            x = padding + rank * (thumb_size + padding)
            if y + thumb_size > panel.shape[0]:
                break
            panel[y : y + thumb_size, x : x + thumb_size] = thumb
            cv2.rectangle(panel, (x, y), (x + thumb_size, y + thumb_size), color, 2)
            text_y = max(y + 20, 18)
            cv2.putText(
                panel,
                display_label,
                (x + 6, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        y += block_height
        if y + thumb_size > panel.shape[0]:
            break

    return np.hstack([frame, panel])

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - guard for missing dependency
    raise ImportError(
        "Ultralytics is required for visualization. Install via `pip install ultralytics`."
    ) from exc

# Allow importing PoseGuidedReID modules when running from repo root
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Elephant ReID on a video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Path to save the annotated video.")
    parser.add_argument(
        "--yolo-model",
        required=True,
        help="Path to YOLO model weights (Ultralytics .pt / .onnx / TorchScript).",
    )
    parser.add_argument(
        "--class-names",
        required=True,
        help="Text file with YOLO class names (one per line).",
    )
    parser.add_argument(
        "--reid-config",
        required=True,
        help="Path to PoseGuidedReID config (.yml).",
    )
    parser.add_argument(
        "--reid-checkpoint",
        required=True,
        help="PoseGuidedReID checkpoint (.pth).",
    )
    parser.add_argument(
        "--gallery",
        required=True,
        help="Gallery features npz (generated via PoseGuidedReID inference).",
    )
    parser.add_argument(
        "--yolo-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for YOLO inference (cuda / cuda:0 / cpu).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for ReID model inference.",
    )
    parser.add_argument(
        "--gallery-device",
        default="cpu",
        help="Device to keep gallery embeddings on (cpu or cuda).",
    )
    parser.add_argument("--conf-thres", type=float, default=0.4, help="YOLO confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="YOLO IOU threshold.")
    parser.add_argument("--max-dets", type=int, default=50, help="Max detections per frame.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K ReID matches to visualize.")
    parser.add_argument(
        "--tracker-config",
        default="bytetrack.yaml",
        help="Tracker configuration for Ultralytics ByteTrack (use absolute path or name resolvable by Ultralytics).",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.5,
        help="Minimum cosine similarity to lock a track's identity.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every N-th frame (>=1). Use >1 to speed up.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on processed frames (for quick debugging).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optionally resize video frames to this width while preserving aspect ratio.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity.",
    )
    return parser.parse_args()


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("visualize")



def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[DetectionResult],
    gallery: GalleryDB,
) -> np.ndarray:
    for det in detections:
        label = det.identity_label or (det.matches[0][0] if det.matches else "Unknown")
        score = det.identity_score if det.identity_score is not None else (det.matches[0][1] if det.matches else 0.0)
        color = gallery.label_to_color.get(label, (0, 255, 0))
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        track_text = f"T{det.display_track_id}" if det.display_track_id >= 0 else "T?"
        identity_text = f"{label} ({score:.2f})" if label else "Unknown"
        text = f"{track_text} | {identity_text} | det {det.score:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        for idx, (match_label, match_score) in enumerate(det.matches[1:], start=1):
            caption = f"{idx+1}. {match_label} {match_score:.2f}"
            cv2.putText(
                frame,
                caption,
                (x1, y1 + 18 + idx * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return frame


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    device = torch.device(args.device)
    gallery_device = torch.device(args.gallery_device)
    logger.info("Using device=%s, gallery_device=%s", device, gallery_device)

    class_names = load_class_names(args.class_names)
    yolo_model = YOLO(args.yolo_model)
    if args.yolo_device:
        yolo_model.to(args.yolo_device)
    logger.info("Loaded YOLO model from %s on %s", args.yolo_model, args.yolo_device)

    gallery = load_gallery_database(args.gallery, gallery_device)
    num_classes = len(set(gallery.ids.tolist()))
    logger.info("Loaded gallery embeddings: %s entries", len(gallery.labels))

    reid_model, transform = build_reid_model(
        args.reid_config,
        args.reid_checkpoint,
        num_classes=num_classes,
        device=device,
        logger=logger,
    )
    logger.info("Loaded ReID checkpoint from %s", args.reid_checkpoint)

    # --- decord video reader ---
    try:
        vr = VideoReader(args.video, ctx=cpu(0))  # or gpu(0) if you want GPU decode
    except Exception as e:
        raise FileNotFoundError(f"Unable to open video with decord: {args.video} ({e})")

    total_frames = len(vr)
    logger.info("Processing video (decord): %s (%d frames)", args.video, total_frames)

    # fps / size from decord
    try:
        fps = float(vr.get_avg_fps())
        if fps <= 0:
            fps = 30.0
    except Exception:
        fps = 30.0

    first_frame = vr[0].asnumpy()
    height, width = first_frame.shape[:2]
    if args.resize_width:
        writer_width = args.resize_width
        writer_height = int(height * (args.resize_width / width))
    else:
        writer_width, writer_height = width, height

    panel_top_k = max(args.top_k, 1)
    match_panel_width = panel_top_k * MATCH_THUMB_SIZE + (panel_top_k + 1) * MATCH_PANEL_PADDING
    writer_size = (writer_width + match_panel_width, writer_height)

    try:
        _, date_str, time_str, ampm = extract_metadata_from_video_path(args.video)
        video_start_time = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Unable to parse timestamp metadata from %s (%s); using relative timestamps.",
            args.video,
            exc,
        )
        video_start_time = None
        ampm = None

    output_path = Path(args.output)
    frames_dir = output_path.with_suffix("")
    if frames_dir.exists():
        if frames_dir.is_dir():
            shutil.rmtree(frames_dir)
        else:
            frames_dir.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving intermediate frames to %s", frames_dir)
    saved_frame_paths: List[Path] = []

    processed = 0
    bad_frame_count = 0
    track_id_to_identity: Dict[int, str] = {}
    identity_to_track: Dict[str, int] = {}
    track_id_alias: Dict[int, int] = {}
    start_time = time.perf_counter()

    if args.max_frames is not None:
        max_frame_idx = min(total_frames, args.max_frames * args.frame_skip)
    else:
        max_frame_idx = total_frames

    with torch.no_grad():
        with tqdm(total=total_frames if total_frames > 0 else None, desc="Frames") as pbar:
            for frame_idx in range(0, max_frame_idx):
                pbar.update(1)
                if args.frame_skip > 1 and frame_idx % args.frame_skip != 0:
                    continue

                if args.max_frames is not None and processed >= args.max_frames:
                    break

                try:
                    frame = vr[frame_idx].asnumpy()
                except Exception as e:
                    bad_frame_count += 1
                    if bad_frame_count <= 5 or bad_frame_count % 100 == 0:
                        logger.warning(
                            "Skipping damaged frame %d via decord (total skipped: %d) - %s",
                            frame_idx, bad_frame_count, e
                        )
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = maybe_resize(frame, args.resize_width)

                boxes, det_scores, cls_ids, track_ids = run_yolo_byteTrack(
                    yolo_model,
                    frame,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    max_dets=args.max_dets,
                    tracker_cfg=args.tracker_config,
                    device=args.yolo_device,
                )
                if boxes.size == 0:
                    processed += 1
                    continue

                tensors, kept_boxes, kept_indices = preprocess_patches(frame, boxes, transform)
                if not tensors:
                    processed += 1
                    continue

                batch = torch.stack(tensors).to(device)
                feats = extract_features(reid_model, batch)
                feats = feats.to(gallery.features.device)
                scores_mat, indices_mat = match_gallery(feats, gallery, args.top_k, ampm=ampm)
                match_indices_np = indices_mat.cpu().numpy()
                matched_gallery_paths = gallery.paths[match_indices_np]
                matched_gallery_labels = gallery.labels[match_indices_np]
                matched_gallery_folders = [
                    [os.path.join(GT_IMAGES_DIR, _file2date(path), LABEL_TO_FOLDER.get(str(label), str(label)), path) for label, path in zip(row, path_row)]
                    for row, path_row in zip(matched_gallery_labels, matched_gallery_paths)
                ]
                
                detections: List[DetectionResult] = []
                for idx, bbox in enumerate(kept_boxes):
                    det_idx = kept_indices[idx]
                    cls_id = cls_ids[det_idx] if det_idx < len(cls_ids) else -1
                    cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"id_{cls_id}"
                    det_score = float(det_scores[det_idx]) if det_idx < len(det_scores) else 0.0
                    track_id = int(track_ids[det_idx]) if det_idx < len(track_ids) else -1
                    match_scores = scores_mat[idx].tolist()
                    match_indices = indices_mat[idx].tolist()
                    matches = [
                        (str(gallery.labels[m_idx]), float(match_scores[j]))
                        for j, m_idx in enumerate(match_indices)
                    ]

                    ### 
                    identity_label = track_id_to_identity.get(track_id)
                    identity_score = None
                    if matches:
                        best_label, best_score = matches[0]
                        if identity_label is None and track_id != -1 and best_score >= args.min_similarity:
                            identity_label = best_label
                            track_id_to_identity[track_id] = best_label
                            if best_label not in identity_to_track:
                                identity_to_track[best_label] = track_id
                                track_id_alias[track_id] = track_id
                            else:
                                track_id_alias[track_id] = identity_to_track[best_label]
                        if identity_label == best_label:
                            identity_score = best_score
                        elif identity_label is None:
                            identity_score = best_score

                    display_track_id = track_id
                    if track_id != -1:
                        if identity_label and identity_label in identity_to_track:
                            display_track_id = identity_to_track[identity_label]
                        elif track_id in track_id_alias:
                            display_track_id = track_id_alias[track_id]
                        else:
                            track_id_alias[track_id] = track_id
                    detections.append(
                        DetectionResult(
                            bbox=bbox,
                            score=det_score,
                            cls_id=cls_id,
                            cls_name=cls_name,
                            track_id=track_id,
                            display_track_id=display_track_id,
                            identity_label=identity_label,
                            identity_score=identity_score,
                            matches=matches,
                            predictions=[],
                        )
                    )
                

                ### add the visualization of matched gallery images with bounding boxes
                annotated = annotate_frame(frame.copy(), detections, gallery)
                annotated = annotate_frame_with_matches(
                    annotated,
                    detections,
                    matched_gallery_folders,
                    gallery,
                    panel_top_k,
                    MATCH_THUMB_SIZE,
                    MATCH_PANEL_PADDING,
                )
                frame_path = build_frame_output_path(
                    frames_dir, args.video, frame_idx, fps, video_start_time
                )
                if not cv2.imwrite(str(frame_path), annotated):
                    raise RuntimeError(f"Failed to save frame {frame_path}")
                saved_frame_paths.append(frame_path)
                processed += 1

    if not saved_frame_paths:
        raise RuntimeError("No frames were saved; cannot build output video.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        writer_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    logger.info("Encoding %d frames into %s", len(saved_frame_paths), output_path)
    for frame_path in saved_frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning("Skipping unreadable frame %s", frame_path)
            continue
        writer.write(frame)
    writer.release()
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    logger.info(
        "Visualization saved to %s (processed %d frames in %.2fs, %.2f FPS)",
        args.output,
        processed,
        elapsed,
        processed / elapsed,
    )
              

if __name__ == "__main__":
    main()
