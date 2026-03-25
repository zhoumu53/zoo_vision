from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil

from .detection import Detector, YoloDetector
from .discovery import discover_video_files
from .fs import relative_to_root
from .media import build_sample_timestamps, extract_frame, probe_video_duration
from .models import FrameSampleResult, ScanConfig, ScanProgress, ScanRunResult, VideoResult
from .preview import build_preview_contact_sheet
from .reporting import (
    append_empty_video_artifacts,
    create_run_directory,
    empty_video_csv_path,
    write_scan_report,
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


def _build_error_result(
    config: ScanConfig,
    video_path: Path,
    error: Exception | str,
    *,
    duration_sec: float | None = None,
    frame_samples: list[FrameSampleResult] | None = None,
) -> VideoResult:
    relative_path = relative_to_root(config.data_root, video_path)
    successful_samples = _successful_samples(frame_samples or [])
    return VideoResult(
        video_path=str(video_path),
        host_video_path=config.host_video_path(relative_path),
        relative_path=relative_path,
        duration_sec=round(duration_sec, 3) if duration_sec is not None else None,
        sample_interval_min=config.interval_minutes,
        sample_timestamps_sec=[sample.timestamp_sec for sample in successful_samples],
        sample_count=len(successful_samples),
        detected_frames=sum(1 for sample in successful_samples if sample.detected),
        coarse_positive_ratio=None,
        uniform_positive_ratio=None,
        confirmed_detection_duration_sec=0.0,
        empty_video=None,
        preview_path=None,
        model_source=config.model_source,
        status="error",
        classification_reason="processing_error",
        error=str(error),
        frame_samples=list(frame_samples or []),
    )


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
    total_videos: int,
) -> list[FrameSampleResult]:
    samples: list[FrameSampleResult] = []
    for sample_index, timestamp_sec in enumerate(timestamps, start=1):
        _emit(
            progress_callback,
            ScanProgress(
                phase="sample_started",
                video_index=video_index,
                total_videos=total_videos,
                sample_index=sample_index,
                total_samples=len(timestamps),
                video_path=str(video_path),
                message=f"{relative_path} | {phase} sample {sample_index}/{len(timestamps)}",
            ),
        )

        frame_path = frame_dir / f"{phase}_{sample_index:03d}_{int(timestamp_sec):06d}s.jpg"
        try:
            extracted_frame_path = extract_frame(video_path, timestamp_sec, frame_path)
            detection_result = detector.detect(extracted_frame_path, config.confidence_threshold)
        except Exception as exc:
            samples.append(
                FrameSampleResult(
                    timestamp_sec=timestamp_sec,
                    frame_path=str(frame_path),
                    detected=False,
                    phase=phase,
                    error=str(exc),
                )
            )
            continue

        samples.append(
            FrameSampleResult(
                timestamp_sec=timestamp_sec,
                frame_path=str(extracted_frame_path),
                detected=detection_result.detected,
                phase=phase,
                matched_labels=detection_result.matched_labels,
                matched_confidences=detection_result.matched_confidences,
            )
        )
    return samples


def scan_videos(
    config: ScanConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    detector_factory: DetectorFactory | None = None,
) -> ScanRunResult:
    videos = discover_video_files(
        config.target_folder,
        recursive=config.recursive,
        filename_substring=config.filename_substring,
    )
    run_dir = create_run_directory(config.output_root)
    frames_root = run_dir / "frames"
    previews_root = run_dir / "previews"
    previews_root.mkdir(parents=True, exist_ok=True)

    _emit(
        progress_callback,
        ScanProgress(
            phase="discovering",
            total_videos=len(videos),
            message=f"Discovered {len(videos)} video(s) in {config.target_folder}",
        ),
    )

    if not videos:
        report_json_path, empty_video_log_path = write_scan_report(run_dir, [], config=config)
        shutil.rmtree(frames_root, ignore_errors=True)
        _emit(
            progress_callback,
            ScanProgress(phase="scan_completed", total_videos=0, message="No videos found."),
        )
        return ScanRunResult(
            run_dir=str(run_dir),
            report_json_path=str(report_json_path),
            empty_video_log_path=str(empty_video_log_path),
            empty_video_csv_path=str(empty_video_csv_path(run_dir)),
            results=[],
        )

    detector_factory = detector_factory or _default_detector_factory

    try:
        detector = detector_factory(config)
    except Exception as exc:
        results = [_build_error_result(config, video_path, f"Detector initialization failed: {exc}") for video_path in videos]
        report_json_path, empty_video_log_path = write_scan_report(run_dir, results, config=config)
        shutil.rmtree(frames_root, ignore_errors=True)
        _emit(
            progress_callback,
            ScanProgress(
                phase="scan_completed",
                total_videos=len(videos),
                message=f"Detector initialization failed: {exc}",
            ),
        )
        return ScanRunResult(
            run_dir=str(run_dir),
            report_json_path=str(report_json_path),
            empty_video_log_path=str(empty_video_log_path),
            empty_video_csv_path=str(empty_video_csv_path(run_dir)),
            results=results,
        )

    results: list[VideoResult] = []

    for video_index, video_path in enumerate(videos, start=1):
        relative_path = relative_to_root(config.data_root, video_path)
        host_video_path = config.host_video_path(relative_path)
        frame_dir: Path | None = None
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
                    total_videos=len(videos),
                    total_samples=len(coarse_timestamps),
                    video_path=str(video_path),
                    message=f"Scanning {relative_path}",
                ),
            )

            frame_dir = frames_root / Path(relative_path).with_suffix("")
            frame_dir.mkdir(parents=True, exist_ok=True)
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
                total_videos=len(videos),
            )

            successful_frame_samples = _successful_samples(frame_samples)
            if not successful_frame_samples:
                raise RuntimeError(f"No readable frames could be extracted from {video_path}")

            coarse_positive_ratio = _positive_ratio(frame_samples, {"coarse"})
            uniform_positive_ratio = coarse_positive_ratio
            classification_reason = "uniform samples below non-empty threshold"

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
                            total_videos=len(videos),
                        )
                    )
                    uniform_positive_ratio = _positive_ratio(frame_samples, {"coarse", "between"})
                    classification_reason = (
                        "uniform ratio updated after midpoint refinement"
                    )

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
                        total_videos=len(videos),
                    )
                )

            confirmed_detection_duration_sec = _confirmed_detection_duration_sec(
                frame_samples,
                config.dense_validation_stride_seconds,
            )
            successful_frame_samples = _successful_samples(frame_samples)
            skipped_sample_count = sum(1 for sample in frame_samples if sample.error)
            detected_frames = sum(1 for sample in successful_frame_samples if sample.detected)
            is_uniform_non_empty = uniform_positive_ratio >= config.non_empty_ratio_threshold
            is_duration_non_empty = confirmed_detection_duration_sec >= evidence_target_sec

            empty_video = not (is_uniform_non_empty or is_duration_non_empty)
            preview_path: str | None = None

            if empty_video and successful_frame_samples:
                preview_file = previews_root / Path(relative_path).with_suffix(".jpg")
                built_preview = build_preview_contact_sheet(
                    successful_frame_samples,
                    preview_file,
                    title=relative_path,
                )
                preview_path = str(built_preview) if built_preview is not None else None

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

            result = VideoResult(
                video_path=str(video_path),
                host_video_path=host_video_path,
                relative_path=relative_path,
                duration_sec=round(duration_sec, 3),
                sample_interval_min=config.interval_minutes,
                sample_timestamps_sec=[sample.timestamp_sec for sample in successful_frame_samples],
                sample_count=len(successful_frame_samples),
                detected_frames=detected_frames,
                coarse_positive_ratio=coarse_positive_ratio,
                uniform_positive_ratio=uniform_positive_ratio,
                confirmed_detection_duration_sec=confirmed_detection_duration_sec,
                empty_video=empty_video,
                preview_path=preview_path,
                model_source=config.model_source,
                status="ok",
                classification_reason=classification_reason,
                error=None,
                frame_samples=frame_samples,
            )
        except Exception as exc:
            result = _build_error_result(
                config,
                video_path,
                exc,
                duration_sec=locals().get("duration_sec"),
                frame_samples=locals().get("frame_samples", []),
            )
        try:
            results.append(result)
            append_empty_video_artifacts(run_dir, result, config=config)
            _emit(
                progress_callback,
                ScanProgress(
                    phase="video_completed",
                    video_index=video_index,
                    total_videos=len(videos),
                    total_samples=result.sample_count,
                    video_path=str(video_path),
                    message=f"Completed {relative_path} with status {result.status}",
                ),
            )
        finally:
            if frame_dir is not None:
                shutil.rmtree(frame_dir, ignore_errors=True)

    report_json_path, empty_video_log_path = write_scan_report(run_dir, results, config=config)
    shutil.rmtree(frames_root, ignore_errors=True)
    _emit(
        progress_callback,
        ScanProgress(
            phase="scan_completed",
            total_videos=len(videos),
            message=f"Completed scan of {len(videos)} video(s).",
        ),
    )
    return ScanRunResult(
        run_dir=str(run_dir),
        report_json_path=str(report_json_path),
        empty_video_log_path=str(empty_video_log_path),
        empty_video_csv_path=str(empty_video_csv_path(run_dir)),
        results=results,
    )
