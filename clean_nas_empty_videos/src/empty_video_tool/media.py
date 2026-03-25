from __future__ import annotations

import math
import subprocess
from pathlib import Path


def probe_video_duration(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe failed for {video_path}: {stderr}")

    raw_output = completed.stdout.strip()
    if not raw_output:
        raise RuntimeError(f"ffprobe returned an empty duration for {video_path}")

    try:
        duration = float(raw_output)
    except ValueError as exc:
        raise RuntimeError(f"Could not parse duration for {video_path}: {raw_output}") from exc

    if duration <= 0:
        raise RuntimeError(f"Video duration must be positive for {video_path}: {duration}")
    return duration


def build_sample_timestamps(duration_sec: float, interval_minutes: int) -> list[float]:
    if duration_sec <= 0:
        raise ValueError("Video duration must be positive.")
    if interval_minutes < 1:
        raise ValueError("Interval minutes must be at least 1.")

    interval_seconds = interval_minutes * 60
    segment_count = max(1, math.ceil(duration_sec / interval_seconds))
    timestamps: list[float] = []

    for index in range(segment_count):
        segment_start = index * interval_seconds
        segment_end = min((index + 1) * interval_seconds, duration_sec)
        midpoint = (segment_start + segment_end) / 2.0
        safe_midpoint = min(max(midpoint, 0.0), max(duration_sec - 0.001, 0.0))
        timestamps.append(round(safe_midpoint, 3))

    return timestamps


def extract_frame(video_path: Path, timestamp_sec: float, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-update",
        "1",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed for {video_path} at {timestamp_sec:.3f}s: {stderr}")
    if not output_path.exists():
        raise RuntimeError(f"ffmpeg did not create a frame for {video_path} at {timestamp_sec:.3f}s")
    return output_path

