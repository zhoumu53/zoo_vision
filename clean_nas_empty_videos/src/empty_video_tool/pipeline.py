from __future__ import annotations

import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil

from .detection import Detector, YoloDetector
from .discovery import iter_video_files
from .fs import relative_to_root
from .media import build_sample_timestamps, extract_frame, probe_video_duration
from .models import FrameSampleResult, ScanConfig, ScanProgress, VideoResult
from .preview import build_preview_contact_sheet
from .exporting import (
    EMPTY_VIDEO_EXPORT_FIELDNAMES,
    build_empty_video_export_rows,
    _append_unique_csv_rows,
)
from .reporting import (
    append_result_to_report,
    load_processed_video_paths,
    run_dir_for_video,
)

ProgressCallback = Callable[[ScanProgress], None]
DetectorFactory = Callable[[ScanConfig], Detector]


def _emit(progress_callback: ProgressCallback | None, progress: ScanProgress) -> None:
    if progress_callback is not None:
        progress_callback(progress)


def _successful_samples(
    frame_samples: list[FrameSampleResult],
    phases: set[str] | None = None,
) -> list[FrameSampleResult]:
    return [
        sample
        for sample in frame_samples
        if sample.error is None and (phases is None or sample.phase in phases)
    ]


def _default_detector_factory(config: ScanConfig) -> Detector:
    return YoloDetector(
        weights_path=config.weights_path,
        default_model=config.default_model,
        target_labels=config.target_labels,
    )


def _positive_ratio(frame_samples: list[FrameSampleResult], phases: set[str] | None = None) -> float:
    relevant_samples = _successful_samples(frame_samples, phases)
    if not relevant_samples:
        return 0.0
    positive_count = sum(1 for sample in relevant_samples if sample.detected)
    return round(positive_count / len(relevant_samples), 3)


def _build_between_timestamps(timestamps: list[float]) -> list[float]:
    between = []
    for left, right in zip(timestamps, timestamps[1:]):
        midpoint = round((left + right) / 2.0, 3)
        if midpoint not in between:
            between.append(midpoint)
    return between


def _build_dense_validation_timestamps(
    positive_timestamps: list[float],
    *,
    duration_sec: float,
    evidence_target_sec: int,
    stride_seconds: int,
    existing_timestamps: set[float],
) -> list[float]:
    timestamps: set[float] = set()
    half_window = evidence_target_sec / 2.0
    safe_duration_limit = max(duration_sec - 0.001, 0.0)

    for positive_timestamp in positive_timestamps:
        start = max(0.0, positive_timestamp - half_window)
        end = min(duration_sec, positive_timestamp + half_window)

        if end - start < evidence_target_sec:
            if start == 0.0:
                end = min(duration_sec, float(evidence_target_sec))
            elif end == duration_sec:
                start = max(0.0, duration_sec - evidence_target_sec)

        cursor = start
        while cursor <= end + 1e-9:
            safe_timestamp = round(min(max(cursor, 0.0), safe_duration_limit), 3)
            if safe_timestamp not in existing_timestamps:
                timestamps.add(safe_timestamp)
            cursor += stride_seconds

    return sorted(timestamps)


def _confirmed_detection_duration_sec(frame_samples: list[FrameSampleResult], stride_seconds: int) -> float:
    positive_timestamps = sorted(
        sample.timestamp_sec
        for sample in frame_samples
        if sample.detected
    )
    if not positive_timestamps:
        return 0.0

    cluster_start = positive_timestamps[0]
    cluster_end = positive_timestamps[0]
    longest_span = 0.0

    for timestamp in positive_timestamps[1:]:
        if timestamp - cluster_end <= stride_seconds * 1.5:
            cluster_end = timestamp
            continue
        longest_span = max(longest_span, cluster_end - cluster_start)
        cluster_start = timestamp
        cluster_end = timestamp

    longest_span = max(longest_span, cluster_end - cluster_start)
    return round(longest_span, 3)


def _run_phase_samples(
    *,
    phase: str,
    video_path: Path,
    relative_path: str,
    timestamps: list[float],
    frame_dir: Path,
    detector: Detector,
    config: ScanConfig,
    progress_callback: ProgressCallback | None,
    video_index: int,
) -> list[FrameSampleResult]:
    # Step 1: extract all frames (parallel ffmpeg calls)
    frame_paths: list[Path] = []
    frame_errors: dict[int, str] = {}  # index -> error message
    for sample_index, timestamp_sec in enumerate(timestamps):
        frame_paths.append(frame_dir / f"{phase}_{sample_index + 1:03d}_{int(timestamp_sec):06d}s.jpg")

    def _extract(idx_ts: tuple[int, float]) -> tuple[int, str | None]:
        idx, ts = idx_ts
        try:
            extract_frame(video_path, ts, frame_paths[idx])
            return idx, None
        except Exception as exc:
            return idx, str(exc)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for idx, err in pool.map(_extract, enumerate(timestamps)):
            if err is not None:
                frame_errors[idx] = err

    # Step 2: batch detect on successfully extracted frames
    valid_indices = [i for i in range(len(timestamps)) if i not in frame_errors]
    valid_paths = [frame_paths[i] for i in valid_indices]

    batch_results: list = []
    if valid_paths and hasattr(detector, "detect_batch"):
        batch_results = detector.detect_batch(valid_paths, config.confidence_threshold)
    elif valid_paths:
        batch_results = [detector.detect(p, config.confidence_threshold) for p in valid_paths]

    # Step 3: assemble results
    detection_map = dict(zip(valid_indices, batch_results))
    samples: list[FrameSampleResult] = []
    for sample_index, timestamp_sec in enumerate(timestamps):
        if sample_index in frame_errors:
            samples.append(
                FrameSampleResult(
                    timestamp_sec=timestamp_sec,
                    frame_path=str(frame_paths[sample_index]),
                    detected=False,
                    phase=phase,
                    error=frame_errors[sample_index],
                )
            )
        else:
            det = detection_map[sample_index]
            samples.append(
                FrameSampleResult(
                    timestamp_sec=timestamp_sec,
                    frame_path=str(frame_paths[sample_index]),
                    detected=det.detected,
                    phase=phase,
                    matched_labels=det.matched_labels,
                    matched_confidences=det.matched_confidences,
                )
            )
    return samples


def _process_single_video(
    *,
    video_path: Path,
    video_index: int,
    config: ScanConfig,
    detector: Detector,
    progress_callback: ProgressCallback | None,
) -> VideoResult:
    """Process one video: extract frames, run detection, classify, generate preview."""
    relative_path = relative_to_root(config.data_root, video_path)
    vid_run_dir = run_dir_for_video(config.output_root, config.data_root, video_path)
    vid_run_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = None
    frame_samples: list[FrameSampleResult] = []
    try:
        duration_sec = probe_video_duration(video_path)
        coarse_timestamps = build_sample_timestamps(duration_sec, config.interval_minutes)
        evidence_target_sec = min(config.min_non_empty_minutes * 60, max(int(duration_sec), 1))

        _emit(
            progress_callback,
            ScanProgress(
                phase="video_started",
                video_index=video_index,
                total_samples=len(coarse_timestamps),
                video_path=str(video_path),
                message=f"Scanning {relative_path}",
            ),
        )

        # use a temp dir for frames (cleaned up after processing)
        frame_dir = Path(tempfile.mkdtemp(prefix="evtool_frames_"))
        frame_samples = _run_phase_samples(
            phase="coarse",
            video_path=video_path,
            relative_path=relative_path,
            timestamps=coarse_timestamps,
            frame_dir=frame_dir,
            detector=detector,
            config=config,
            progress_callback=progress_callback,
            video_index=video_index,
        )

        successful_frame_samples = _successful_samples(frame_samples)
        if not successful_frame_samples:
            raise RuntimeError(f"No readable frames could be extracted from {video_path}")

        coarse_positive_ratio = _positive_ratio(frame_samples, {"coarse"})
        uniform_positive_ratio = coarse_positive_ratio

        if coarse_positive_ratio <= config.refine_ratio_threshold:
            between_timestamps = _build_between_timestamps(coarse_timestamps)
            if between_timestamps:
                frame_samples.extend(
                    _run_phase_samples(
                        phase="between",
                        video_path=video_path,
                        relative_path=relative_path,
                        timestamps=between_timestamps,
                        frame_dir=frame_dir,
                        detector=detector,
                        config=config,
                        progress_callback=progress_callback,
                        video_index=video_index,
                    )
                )
                uniform_positive_ratio = _positive_ratio(frame_samples, {"coarse", "between"})

        positive_uniform_timestamps = [
            sample.timestamp_sec
            for sample in frame_samples
            if sample.phase in {"coarse", "between"} and sample.detected
        ]

        existing_timestamps = {sample.timestamp_sec for sample in frame_samples}
        dense_timestamps = _build_dense_validation_timestamps(
            positive_uniform_timestamps,
            duration_sec=duration_sec,
            evidence_target_sec=evidence_target_sec,
            stride_seconds=config.dense_validation_stride_seconds,
            existing_timestamps=existing_timestamps,
        )

        if dense_timestamps:
            frame_samples.extend(
                _run_phase_samples(
                    phase="dense",
                    video_path=video_path,
                    relative_path=relative_path,
                    timestamps=dense_timestamps,
                    frame_dir=frame_dir,
                    detector=detector,
                    config=config,
                    progress_callback=progress_callback,
                    video_index=video_index,
                )
            )

        confirmed_detection_duration_sec = _confirmed_detection_duration_sec(
            frame_samples,
            config.dense_validation_stride_seconds,
        )
        successful_frame_samples = _successful_samples(frame_samples)
        skipped_sample_count = sum(1 for sample in frame_samples if sample.error)
        is_uniform_non_empty = uniform_positive_ratio >= config.non_empty_ratio_threshold
        is_duration_non_empty = confirmed_detection_duration_sec >= evidence_target_sec

        to_delete = not (is_uniform_non_empty or is_duration_non_empty)

        # build classification reason
        if is_uniform_non_empty:
            classification_reason = (
                f"uniform non-empty ratio {uniform_positive_ratio:.3f} "
                f">= threshold {config.non_empty_ratio_threshold:.3f}"
            )
        elif is_duration_non_empty:
            classification_reason = (
                f"confirmed elephant detections span {confirmed_detection_duration_sec:.1f}s "
                f">= target {evidence_target_sec}s"
            )
        else:
            classification_reason = (
                f"uniform non-empty ratio {uniform_positive_ratio:.3f} and confirmed span "
                f"{confirmed_detection_duration_sec:.1f}s stayed below thresholds"
            )
        if skipped_sample_count:
            classification_reason += f"; skipped {skipped_sample_count} unreadable sample(s)"

        # generate preview for empty videos
        preview_path: str | None = None
        if to_delete and successful_frame_samples:
            preview_dir = vid_run_dir / "to_delete_preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_file = preview_dir / f"{video_path.stem}.jpg"
            built_preview = build_preview_contact_sheet(
                successful_frame_samples,
                preview_file,
                title=relative_path,
            )
            preview_path = str(built_preview) if built_preview is not None else None

        result = VideoResult(
            video_path=str(video_path),
            duration_sec=round(duration_sec, 3),
            sample_interval_min=config.interval_minutes,
            n_frame_samples=len(successful_frame_samples),
            to_delete=to_delete,
            classification_reason=classification_reason,
            error=None,
            relative_path=relative_path,
            preview_path=preview_path,
            frame_samples=frame_samples,
        )
    except Exception as exc:
        relative_path = relative_to_root(config.data_root, video_path)
        result = VideoResult(
            video_path=str(video_path),
            duration_sec=locals().get("duration_sec"),
            sample_interval_min=config.interval_minutes,
            n_frame_samples=len(_successful_samples(frame_samples)),
            to_delete=None,
            classification_reason="processing_error",
            error=str(exc),
            relative_path=relative_path,
            frame_samples=frame_samples,
        )
    finally:
        if frame_dir is not None:
            shutil.rmtree(frame_dir, ignore_errors=True)

    return result


def scan_videos(
    config: ScanConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    detector_factory: DetectorFactory | None = None,
) -> list[VideoResult]:
    _emit(
        progress_callback,
        ScanProgress(
            phase="discovering",
            message=f"Scanning videos in {config.target_folder} ...",
        ),
    )

    detector_factory = detector_factory or _default_detector_factory
    try:
        detector = detector_factory(config)
    except Exception as exc:
        _emit(
            progress_callback,
            ScanProgress(phase="scan_completed", message=f"Detector initialization failed: {exc}"),
        )
        raise

    # cache of already-loaded processed sets per run_dir
    processed_cache: dict[Path, set[str]] = {}
    results: list[VideoResult] = []

    video_iter = iter_video_files(
        config.target_folder,
        recursive=config.recursive,
        filename_substring=config.filename_substring,
    )

    for video_index, video_path in enumerate(video_iter, start=1):
        vid_run_dir = run_dir_for_video(config.output_root, config.data_root, video_path)
        # check if already processed (skip check when force_rescan is enabled)
        if not config.force_rescan:
            if vid_run_dir not in processed_cache:
                processed_cache[vid_run_dir] = load_processed_video_paths(vid_run_dir)
            if str(video_path) in processed_cache[vid_run_dir]:
                _emit(
                    progress_callback,
                    ScanProgress(
                        phase="video_completed",
                        video_index=video_index,
                        video_path=str(video_path),
                        message=f"Skipped (already processed) {video_path}",
                    ),
                )
                continue

        result = _process_single_video(
            video_path=video_path,
            video_index=video_index,
            config=config,
            detector=detector,
            progress_callback=progress_callback,
        )
        results.append(result)

        # persist result to report.json immediately
        append_result_to_report(vid_run_dir, result, config=config)
        processed_cache.setdefault(vid_run_dir, set()).add(str(video_path))

        # persist to grouped CSV immediately if to_delete
        if result.to_delete:
            date_folder = vid_run_dir.name  # e.g. "20260410AM"
            csv_path = vid_run_dir / f"{date_folder}.csv"
            export_rows = build_empty_video_export_rows([result.to_report_row()])
            if export_rows:
                _append_unique_csv_rows(
                    csv_path,
                    export_rows,
                    fieldnames=EMPTY_VIDEO_EXPORT_FIELDNAMES,
                    unique_field="host_path",
                )

        _emit(
            progress_callback,
            ScanProgress(
                phase="video_completed",
                video_index=video_index,
                video_path=str(video_path),
                message=f"Completed {result.relative_path} | to_delete={result.to_delete}",
            ),
        )

    _emit(
        progress_callback,
        ScanProgress(
            phase="scan_completed",
            total_videos=len(results),
            message=f"Completed scan of {len(results)} new video(s).",
        ),
    )
    return results
