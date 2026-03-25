from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


@dataclass(frozen=True)
class ScanConfig:
    data_root: Path
    target_folder: Path
    output_root: Path
    empty_export_root: Path | None = None
    host_data_root_display: str | None = None
    filename_substring: str | None = None
    recursive: bool = True
    interval_minutes: int = 2
    confidence_threshold: float = 0.65
    weights_path: str | None = None
    default_model: str = "yolov8n.pt"
    target_labels: tuple[str, ...] = ("elephant",)
    non_empty_ratio_threshold: float = 0.8
    refine_ratio_threshold: float = 0.2
    min_non_empty_minutes: int = 2
    dense_validation_stride_seconds: int = 30

    def __post_init__(self) -> None:
        resolved_data_root = _resolve_path(self.data_root)
        resolved_target_folder = _resolve_path(self.target_folder)
        resolved_output_root = _resolve_path(self.output_root)

        if not resolved_data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {resolved_data_root}")
        if not resolved_target_folder.exists():
            raise FileNotFoundError(f"Target folder does not exist: {resolved_target_folder}")
        if not resolved_target_folder.is_dir():
            raise NotADirectoryError(f"Target folder is not a directory: {resolved_target_folder}")
        if self.interval_minutes < 1:
            raise ValueError("Interval minutes must be at least 1.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if not self.target_labels:
            raise ValueError("At least one target label is required.")
        if not 0.0 <= self.refine_ratio_threshold <= 1.0:
            raise ValueError("Refine ratio threshold must be between 0 and 1.")
        if not 0.0 <= self.non_empty_ratio_threshold <= 1.0:
            raise ValueError("Non-empty ratio threshold must be between 0 and 1.")
        if self.dense_validation_stride_seconds < 1:
            raise ValueError("Dense validation stride must be at least 1 second.")
        if self.min_non_empty_minutes < 1:
            raise ValueError("Minimum non-empty evidence must be at least 1 minute.")
        resolved_target_folder.relative_to(resolved_data_root)

        object.__setattr__(self, "data_root", resolved_data_root)
        object.__setattr__(self, "target_folder", resolved_target_folder)
        object.__setattr__(self, "output_root", resolved_output_root)
        if self.empty_export_root is not None:
            object.__setattr__(self, "empty_export_root", _resolve_path(Path(self.empty_export_root)))
        host_data_root_display = self.host_data_root_display or str(resolved_data_root)
        object.__setattr__(self, "host_data_root_display", str(Path(host_data_root_display).expanduser()))
        if self.weights_path:
            object.__setattr__(
                self,
                "weights_path",
                str(Path(self.weights_path).expanduser()),
            )
        if self.filename_substring is not None:
            object.__setattr__(self, "filename_substring", str(self.filename_substring))

    @property
    def model_source(self) -> str:
        return self.weights_path or self.default_model

    def host_video_path(self, relative_path: str) -> str:
        return str((Path(self.host_data_root_display) / relative_path).as_posix())


@dataclass
class FrameSampleResult:
    timestamp_sec: float
    frame_path: str
    detected: bool
    phase: str = "coarse"
    matched_labels: list[str] = field(default_factory=list)
    matched_confidences: list[float] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoResult:
    video_path: str
    host_video_path: str
    relative_path: str
    duration_sec: float | None
    sample_interval_min: int
    sample_timestamps_sec: list[float] = field(default_factory=list)
    sample_count: int = 0
    detected_frames: int = 0
    coarse_positive_ratio: float | None = None
    uniform_positive_ratio: float | None = None
    confirmed_detection_duration_sec: float = 0.0
    empty_video: bool | None = None
    preview_path: str | None = None
    model_source: str = ""
    status: str = "ok"
    classification_reason: str = ""
    error: str | None = None
    frame_samples: list[FrameSampleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["frame_samples"] = [sample.to_dict() for sample in self.frame_samples]
        return payload

    def to_report_row(self) -> dict:
        return {
            "video_path": self.video_path,
            "host_video_path": self.host_video_path,
            "relative_path": self.relative_path,
            "duration_sec": self.duration_sec,
            "sample_interval_min": self.sample_interval_min,
            "sample_timestamps_sec": self.sample_timestamps_sec,
            "sample_count": self.sample_count,
            "detected_frames": self.detected_frames,
            "coarse_positive_ratio": self.coarse_positive_ratio,
            "uniform_positive_ratio": self.uniform_positive_ratio,
            "confirmed_detection_duration_sec": self.confirmed_detection_duration_sec,
            "empty_video": self.empty_video,
            "preview_path": self.preview_path,
            "model_source": self.model_source,
            "status": self.status,
            "classification_reason": self.classification_reason,
            "error": self.error,
        }


@dataclass(frozen=True)
class ScanProgress:
    phase: Literal[
        "discovering",
        "video_started",
        "sample_started",
        "video_completed",
        "scan_completed",
    ]
    video_index: int = 0
    total_videos: int = 0
    sample_index: int = 0
    total_samples: int = 0
    video_path: str = ""
    message: str = ""


@dataclass
class ScanRunResult:
    run_dir: str
    report_json_path: str
    empty_video_log_path: str
    empty_video_csv_path: str = ""
    results: list[VideoResult] = field(default_factory=list)
