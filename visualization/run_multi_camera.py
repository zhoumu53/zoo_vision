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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm
from utils import *

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
    parser.add_argument("--cmd", required=True, choices=['video_tracks_reid_improved', 'video_id_classification', 'video_tracks_reid_improved_with_behavior'], help="Which visualization pipeline to run.")
    parser.add_argument("--track-outdir", default="/home/mu/Desktop/comparison_videos", help="Output path for tracking+ReID pipeline.")
    parser.add_argument("--class-names", required=True, help="Path to YOLO class names file.")
    parser.add_argument("--yolo-model", required=True, help="Path to YOLO checkpoint.")
    parser.add_argument("--reid-config", required=True, help="Path to PoseGuidedReID config.")
    parser.add_argument("--reid-checkpoint", required=True, help="Path to PoseGuidedReID checkpoint.")
    parser.add_argument("--gallery", required=True, help="Path to gallery npz.")
    parser.add_argument("--id-checkpoint", required=True, help="Identity classifier checkpoint (.pth or .ptc).")
    parser.add_argument("--yolo-device", default="cuda", help="Device for YOLO inference (cuda / cuda:0 / cpu).")
    parser.add_argument("--device", default="cuda", help="Device to run ReID/ID models on.")
    parser.add_argument("--gallery-device", default="cpu", help="Device to store gallery features on.")
    parser.add_argument("--resize-width", type=int, default=None, help="Optional width override for individual outputs.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Frame skip for pipelines that support it.")
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most N frames per pipeline.")
    parser.add_argument("--tracker-config", default="bytetrack.yaml", help="Tracker config for track pipeline.")
    parser.add_argument("--min-similarity", type=float, default=0.5, help="Min similarity to lock identity in track pipeline.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logger verbosity.")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Maximum number of tracking jobs to run in parallel.",
    )
    return parser.parse_args()


def add_runtime_flags(cmd: List[str], args: argparse.Namespace) -> None:
    """Append shared runtime knobs if the downstream script supports them."""
    cmd.extend(["--frame-skip", str(args.frame_skip)])
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.resize_width is not None:
        cmd.extend(["--resize-width", str(args.resize_width)])


def run_pipeline(command: List[str], logger: logging.Logger) -> None:
    logger.info("Running pipeline: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Pipeline failed: %s", exc)
        raise


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


def stream_and_write(outputs: List[str], titles: List[str], output_path: str) -> None:
    if len(outputs) != len(titles):

        raise ValueError(f"Number of titles {len(titles)} must match the number of outputs "
                         f"={len(outputs)}.")
    if len(outputs) != 4:
        raise ValueError("run_multi_camera expects exactly 4 camera outputs for a 2x2 grid.")

    captures = []
    writer = None
    progress = None
    frame_count = 0

    try:
        for path in outputs:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {path}")
            captures.append(cap)

        fps_values = [cap.get(cv2.CAP_PROP_FPS) or 30.0 for cap in captures]
        base_fps = fps_values[0] if fps_values else 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        progress = tqdm(desc="Combining videos", unit="frame")
        while True:
            frames = []
            for cap in captures:
                ret, frame = cap.read()
                if not ret:
                    frames = []
                    break
                frames.append(frame)

            if not frames:
                break

            padded = [pad_title(frame, titles[idx]) for idx, frame in enumerate(frames)]
            top_row = np.hstack(padded[:2])
            bottom_row = np.hstack(padded[2:])
            combined = np.vstack([top_row, bottom_row])

            if writer is None:
                h, w = combined.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, base_fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {output_path}")

            writer.write(combined)
            frame_count += 1
            progress.update(1)

        if frame_count == 0:
            raise RuntimeError("No overlapping frames between comparison videos.")
    finally:
        if progress is not None:
            progress.close()
        for cap in captures:
            cap.release()
        if writer is not None:
            writer.release()



def get_output_filename(video_path, track_outdir, max_frames, date=None) -> str:
    filename = os.path.basename(video_path)
    suffix = '' if max_frames is None else f'_{max_frames}'
    filename = filename.replace('.mp4', suffix)
    if date is None:
        date = extract_metadata_from_video_path(video_path)[1]
    return os.path.join(track_outdir, date, filename)



def run_cameras(args, video, date=None) -> None:
    logger = setup_logger(args.log_level)

    camera_id, date, time, ampm = extract_metadata_from_video_path(video)
    other_camera_ids = set(CAMERA_PARIS) - {camera_id}

    print("other camera ids:", other_camera_ids)

    other_video_paths = [extract_other_cameras(camera_id, date, time, ampm, raw_video_dir='/mnt/camera_nas') for camera_id in other_camera_ids]    
    all_video_paths = [video] + other_video_paths
    track_output_paths = [get_output_filename(path, args.track_outdir, args.max_frames) for path in all_video_paths]

    commands_to_run: List[tuple[str, str, List[str]]] = []

    for idx, (video_path, track_output) in enumerate(zip(all_video_paths, track_output_paths)):

        if not os.path.exists(video_path):
            logger.warning("Video path does not exist: %s. Skipping.", video_path)
            continue
        if os.path.exists(track_output):
            logger.info("Output already exists for %s at %s. Skipping.", video_path, track_output)
            continue

        ### if track_output is dir, and exists, skip
        if os.path.isdir(track_output) and os.path.exists(track_output):
            logger.info("Output directory already exists for %s at %s. Skipping.", video_path, track_output)
            continue

         ### ReID + Track

        if args.cmd == 'video_tracks_reid':
            track_cmd = [
                sys.executable,
                str(THIS_DIR / f"{args.cmd}.py"),
                "--video",
                video_path,
                "--output",
                track_output,
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
                "--yolo-device",
                args.yolo_device,
                "--device",
                args.device,
                "--gallery-device",
                args.gallery_device,
                "--tracker-config",
                args.tracker_config,
                "--min-similarity",
                str(args.min_similarity),
            ]

        elif args.cmd == 'video_id_classification':
            track_cmd = [
                sys.executable,
                str(THIS_DIR / f"{args.cmd}.py"),
                "--video",
                video_path,
                "--output",
                track_output,
                "--yolo-model",
                args.yolo_model,
                "--class-names",
                args.class_names,
                "--id-checkpoint",
                args.id_checkpoint,
                "--yolo-device",
                args.yolo_device,
                "--device",
                args.device,
                "--max-frames",
                str(args.max_frames),
                "--save-jpg",
                "--jpg-interval",
                "20",
                "--jpg-max-count",
                "20000",
            ]

        elif args.cmd == 'video_tracks_reid_improved':  
            track_cmd = [
                sys.executable,
                str(THIS_DIR / f"{args.cmd}.py"),
                "--video",
                video_path,
                "--output",
                track_output,
                "--yolo-model",
                args.yolo_model,
                "--class-names",
                args.class_names,
                "--reid-config",
                args.reid_config,
                "--reid-checkpoint",
                args.reid_checkpoint,
                "--yolo-device",
                args.yolo_device,
                "--device",
                args.device,
                "--no-new-stitching",
                "--frame-skip",
                "5",
                "--max-dets",
                "20",
                "--reid-sim-thres",
                "0.7",
                "--reid-max-gap-frames",
                "300",
                "--save-jpg",
                "--jpg-interval",
                "20",
                "--jpg-max-count",
                "20000",
                "--online-reid-from-hub",
                "--max-new-reid-per-frame",
                "5"
            ]
        elif args.cmd == 'video_tracks_reid_improved_with_behavior':
            track_cmd = [
                sys.executable,
                str(THIS_DIR / f"{args.cmd}.py"),
                "--video",
                video_path,
                "--output",
                track_output,
                "--yolo-model",
                args.yolo_model,
                "--class-names",
                args.class_names,
                "--reid-config",
                args.reid_config,
                "--reid-checkpoint",
                args.reid_checkpoint,
                "--yolo-device",
                args.yolo_device,
                "--device",
                args.device,
                "--frame-skip",
                "1",
                "--max-dets",
                "20",
                "--reid-sim-thres",
                "0.9",
                "--reid-max-gap-frames",
                "300",
                "--save-jpg",
                "--jpg-interval",
                "25",
                "--jpg-max-count",
                "50000",
                "--online-reid-from-hub",
                "--max-new-reid-per-frame",
                "5",
                "--behavior-model",
                "models/sleep/vit/v2_no_validation/config.ptc"
            ]


        add_runtime_flags(track_cmd, args)
        commands_to_run.append((video_path, track_output, track_cmd))

    if commands_to_run:
        max_workers = max(1, min(args.max_parallel, len(commands_to_run)))
        logger.info("Running %d tracking jobs in parallel (max_workers=%d)", len(commands_to_run), max_workers)
        failures: List[tuple[str, Exception]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(run_pipeline, cmd, logger): (video_path, track_output)
                for video_path, track_output, cmd in commands_to_run
            }
            for future in as_completed(future_to_task):
                video_path, track_output = future_to_task[future]
                try:
                    future.result()
                    logger.info("Completed tracking for %s -> %s", video_path, track_output)
                except Exception as exc:
                    logger.error("Tracking failed for %s: %s", video_path, exc)
                    failures.append((video_path, exc))
        if failures:
            raise RuntimeError(f"{len(failures)} tracking job(s) failed; see logs for details.")

    ### titles

    ### if any output path is dir, get new mp4 path inside dir
    new_track_output_paths = []
    for path in track_output_paths:
        if os.path.isdir(path):
            suffix = 'tracks' if args.cmd == 'video_tracks_reid_improved' else 'idcls'
            path = os.path.join(path, os.path.basename(path).replace(str(args.max_frames),  suffix))
        new_track_output_paths.append(path)
    track_output_paths = new_track_output_paths if len(new_track_output_paths) == len(all_output_paths) else all_output_paths

    all_titles = [f"Camera {extract_metadata_from_video_path(path)[0]} "  for path in all_video_paths]
    stream_and_write(track_output_paths, all_titles, os.path.join(args.track_outdir, 'comparison_video.mp4'))
    logger.info("Comparison video saved to %s", os.path.join(args.track_outdir, 'comparison_video.mp4'))


if __name__ == "__main__":

    args = parse_args()

    video = args.video

    ### if run whole day -- extract camera id, date, time, ampm
    camera_id, date, time, ampm = extract_metadata_from_video_path(video)

    ### 
    ## extract all videos from this camera id, from this day

    single_cam_single_day_videos = extract_all_videos_single_camera_single_day(camera_id, date, raw_video_dir='/mnt/camera_nas')

    for video in single_cam_single_day_videos:
        print("Processing video:", video)
        run_cameras(args, video=video)
