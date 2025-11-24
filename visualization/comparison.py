"""
Generate side-by-side comparison video for multiple pipelines (ReID, ReID+Track, ID classification).

The script sequentially runs each pipeline on the same video source, collects their outputs,
and stitches the resulting frames horizontally with padding + labels for easier comparison.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple visualization pipelines.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Path to save the combined comparison video.")
    parser.add_argument("--reid-output", default="runs/reid_vis.mp4", help="Output path for pure ReID pipeline.")
    parser.add_argument("--track-output", default="runs/reid_track_vis.mp4", help="Output path for tracking+ReID pipeline.")
    parser.add_argument("--id-output", default="runs/id_vis.mp4", help="Output path for ID classification pipeline.")
    parser.add_argument("--class-names", required=True, help="Path to YOLO class names file.")
    parser.add_argument("--yolo-model", required=True, help="Path to YOLO checkpoint.")
    parser.add_argument("--reid-config", required=True, help="Path to PoseGuidedReID config.")
    parser.add_argument("--reid-checkpoint", required=True, help="Path to PoseGuidedReID checkpoint.")
    parser.add_argument("--gallery", required=True, help="Path to gallery npz.")
    parser.add_argument("--id-checkpoint", required=True, help="Identity classifier checkpoint (.pth or .ptc).")
    parser.add_argument("--titles", nargs=3, default=["ReID", "ReID+Track", "ID Classifier"], help="Titles for each pipeline.")
    parser.add_argument("--device", default="cuda", help="Device to run inference on.")
    parser.add_argument("--gallery-device", default="cpu", help="Device to store gallery features on.")
    parser.add_argument("--resize-width", type=int, default=None, help="Optional width override for individual outputs.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Frame skip for pipelines that support it.")
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most N frames per pipeline.")
    parser.add_argument("--tracker-config", default="bytetrack.yaml", help="Tracker config for track pipeline.")
    parser.add_argument("--min-similarity", type=float, default=0.5, help="Min similarity to lock identity in track pipeline.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logger verbosity.")
    return parser.parse_args()


def run_pipeline(command: List[str], logger: logging.Logger) -> None:
    logger.info("Running pipeline: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Pipeline failed: %s", exc)
        raise


def read_video_frames(path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def pad_title(frame: np.ndarray, title: str, height=40) -> np.ndarray:
    title_bar = np.full((height, frame.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(
        title_bar,
        title,
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([title_bar, frame])


def align_frames(frames_list: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    min_len = min(len(frames) for frames in frames_list)
    return [frames[:min_len] for frames in frames_list]


def write_video(frames: List[np.ndarray], output_path: str, fps: float) -> None:
    if not frames:
        raise RuntimeError("No frames to write.")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    comparison_commands = [
        [
            sys.executable,
            str(THIS_DIR / "video_reid.py"),
            "--video",
            args.video,
            "--output",
            args.reid_output,
            "--yolo-model",
            args.yolo_model,
            "--class-names",
            args.class_names,
            "--reid-config",
            args.reid_config,
            "--reid-checkpoint",
            args.reid_checkpoint,
            "--gallery",
            args.gallery,
            "--device",
            args.device,
            "--gallery-device",
            args.gallery_device,
            "--max-frames",
            str(args.max_frames) if args.max_frames is not None else "",
        ],
        [
            sys.executable,
            str(THIS_DIR / "video_tracks_reid.py"),
            "--video",
            args.video,
            "--output",
            args.track_output,
            "--yolo-model",
            args.yolo_model,
            "--class-names",
            args.class_names,
            "--reid-config",
            args.reid_config,
            "--reid-checkpoint",
            args.reid_checkpoint,
            "--gallery",
            args.gallery,
            "--device",
            args.device,
            "--gallery-device",
            args.gallery_device,
            "--tracker-config",
            args.tracker_config,
            "--min-similarity",
            str(args.min_similarity),
            "--max-frames",
            str(args.max_frames) if args.max_frames is not None else "",
        ],
        [
            sys.executable,
            str(THIS_DIR / "video_id_classification.py"),
            "--video",
            args.video,
            "--output",
            args.id_output,
            "--yolo-model",
            args.yolo_model,
            "--class-names",
            args.class_names,
            "--id-checkpoint",
            args.id_checkpoint,
            "--device",
            args.device,
            "--max-frames",
            str(args.max_frames) if args.max_frames is not None else "",
        ],
    ]

    for cmd in comparison_commands:
        run_pipeline(cmd, logger)

    outputs = [args.reid_output, args.track_output, args.id_output]
    frames_list = []
    fps_values = []
    for path in outputs:
        frames, fps = read_video_frames(path)
        frames_list.append(frames)
        fps_values.append(fps)

    base_fps = fps_values[0] if fps_values else 30.0
    combined_frames = []
    aligned = align_frames(frames_list)
    for frame_idx in tqdm(range(len(aligned[0])), desc="Combining videos"):
        row_frames = []
        for idx, frames in enumerate(aligned):
            frame = frames[frame_idx]
            row_frames.append(pad_title(frame, args.titles[idx]))
        combined_frames.append(np.vstack(row_frames))

    write_video(combined_frames, args.output, base_fps)
    logger.info("Comparison video saved to %s", args.output)


if __name__ == "__main__":
    main()
