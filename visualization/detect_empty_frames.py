"""
Detect empty frames (no YOLO detections) for a given video and save per-frame stats to CSV.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
from datetime import datetime
from pathlib import Path

import cv2
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from utils import (
    extract_metadata_from_video_path,
    format_frame_timestamp,
    maybe_resize,
    run_yolo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO on a video (no tracking) and record frames with zero detections."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the CSV file with per-frame detection counts.",
    )
    parser.add_argument(
        "--yolo-model",
        required=True,
        help="Path to YOLO model weights (Ultralytics .pt / .onnx / TorchScript).",
    )
    parser.add_argument(
        "--yolo-device",
        default="cuda:0",
        help="Device for YOLO inference (cuda / cuda:0).",
    )
    parser.add_argument("--conf-thres", type=float, default=0.4, help="YOLO confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="YOLO IOU threshold.")
    parser.add_argument("--max-dets", type=int, default=50, help="Max detections per frame.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=30,
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
    parser.add_argument(
        "--cuda-empty-cache-interval",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache() every N processed frames (0 disables).",
    )
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=0,
        help="Call Python garbage collector every N processed frames (0 disables).",
    )
    return parser.parse_args()


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("detect_empty_frames")


def infer_video_start(video_path: str, logger: logging.Logger) -> datetime | None:
    """Best-effort attempt to parse the video start datetime from filename."""
    try:
        _, date_str, time_str = extract_metadata_from_video_path(video_path)[:3]
        return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
    except Exception as exc:  # pragma: no cover - best effort parsing
        logger.debug("Unable to infer video start time from %s: %s", video_path, exc)
        return None


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    if not args.yolo_device.startswith("cuda"):
        raise ValueError(
            f"YOLO device must be CUDA for this script (got {args.yolo_device}). "
            "Override --yolo-device with a CUDA device string such as cuda:0."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available but a CUDA device was requested for YOLO inference."
        )

    from ultralytics import YOLO

    yolo_model = YOLO(args.yolo_model)
    if args.yolo_device:
        yolo_model.to(args.yolo_device)
    logger.info("Loaded YOLO model from %s on %s", args.yolo_model, args.yolo_device)

    try:
        vr = VideoReader(args.video, ctx=cpu(0))
    except Exception as e:
        raise FileNotFoundError(f"Unable to open video with decord: {args.video} ({e})")

    total_frames = len(vr)
    logger.info("Processing video (decord): %s (%d frames)", args.video, total_frames)

    try:
        fps = float(vr.get_avg_fps())
        if fps <= 0:
            fps = 30.0
    except Exception:
        fps = 30.0

    video_start = infer_video_start(args.video, logger)

    frame_step = max(args.frame_skip, 1)
    if args.max_frames is not None:
        max_frame_idx = min(total_frames, args.max_frames * frame_step)
    else:
        max_frame_idx = total_frames
    frame_indices = range(0, max_frame_idx, frame_step)

    empty_frame_count = 0
    bad_frame_count = 0
    processed_rows = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_frames: set[str] = set()
    output_exists = output_path.exists()
    if output_exists:
        with output_path.open("r", newline="") as existing_file:
            reader = csv.DictReader(existing_file)
            for row in reader:
                frame_name = row.get("frame_name")
                if frame_name:
                    existing_frames.add(frame_name)

    write_mode = "a" if output_exists else "w"
    with output_path.open(write_mode, newline="") as csv_file:
        fieldnames = [
            "frame_idx",
            "frame_name",
            "timestamp",
            "num_detections",
            "is_empty",
            "is_bad_frame",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not output_exists:
            writer.writeheader()
            csv_file.flush()

        total_iters = len(frame_indices)
        effective_fps = fps if fps > 0 else 30.0
        with torch.no_grad():
            with tqdm(total=total_iters, desc="Frames") as pbar:
                for frame_idx in frame_indices:
                    pbar.update(1)
                    frame_name = format_frame_timestamp(video_start, frame_idx, fps)
                    timestamp_seconds = frame_idx / effective_fps
                    if frame_name in existing_frames:
                        processed_rows += 1
                        continue

                    try:
                        vr.seek(frame_idx)
                        decord_frame = vr.next()
                        frame_np = decord_frame.asnumpy()
                    except Exception as e:
                        bad_frame_count += 1
                        if bad_frame_count <= 5 or bad_frame_count % 100 == 0:
                            logger.warning(
                                "Skipping damaged frame %d via decord (total skipped: %d) - %s",
                                frame_idx,
                                bad_frame_count,
                                e,
                            )
                        writer.writerow(
                            {
                                "frame_idx": frame_idx,
                                "frame_name": frame_name,
                                "timestamp": timestamp_seconds,
                                "num_detections": "",
                                "is_empty": "",
                                "is_bad_frame": True,
                            }
                        )
                        csv_file.flush()
                        processed_rows += 1
                        continue

                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    frame_bgr = maybe_resize(frame_bgr, args.resize_width)

                    boxes, _, _ = run_yolo(
                        yolo_model,
                        frame_bgr,
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        max_dets=args.max_dets,
                        device=args.yolo_device,
                    )
                    num_dets = int(boxes.shape[0]) if boxes.size != 0 else 0
                    if num_dets == 0:
                        empty_frame_count += 1

                    writer.writerow(
                        {
                            "frame_idx": frame_idx,
                            "frame_name": frame_name,
                            "timestamp": timestamp_seconds,
                            "num_detections": num_dets,
                            "is_empty": bool(num_dets == 0),
                            "is_bad_frame": False,
                        }
                    )
                    csv_file.flush()
                    processed_rows += 1
                    existing_frames.add(frame_name)

                    if (
                        args.cuda_empty_cache_interval
                        and args.cuda_empty_cache_interval > 0
                        and processed_rows % args.cuda_empty_cache_interval == 0
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()

                    if (
                        args.gc_interval
                        and args.gc_interval > 0
                        and processed_rows % args.gc_interval == 0
                    ):
                        gc.collect()

    logger.info(
        "Saved detection stats for %d frames to %s (empty frames: %d)",
        processed_rows,
        output_path,
        empty_frame_count,
    )


if __name__ == "__main__":
    main()
