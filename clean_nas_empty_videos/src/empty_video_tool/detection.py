from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


def _normalize_label(label: str) -> str:
    return label.strip().lower()


@dataclass(frozen=True)
class DetectionResult:
    detected: bool
    matched_labels: list[str]
    matched_confidences: list[float]


class Detector(Protocol):
    def detect(self, image_path: Path, confidence_threshold: float) -> DetectionResult:
        ...


class YoloDetector:
    def __init__(
        self,
        *,
        weights_path: str | None = None,
        default_model: str = "yolov8n.pt",
        target_labels: Sequence[str] = ("elephant",),
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Install requirements before running scans.") from exc

        if weights_path:
            candidate = Path(weights_path).expanduser()
            if not candidate.exists():
                raise FileNotFoundError(f"Configured weights file does not exist: {candidate}")
            model_source = str(candidate)
        else:
            model_source = default_model

        self.model_source = model_source
        self.model = YOLO(model_source)
        self.names = self._extract_name_map(self.model.names)
        normalized_targets = {_normalize_label(label) for label in target_labels}
        self.target_class_ids = {
            class_id
            for class_id, class_name in self.names.items()
            if _normalize_label(class_name) in normalized_targets
        }

        if not self.target_class_ids:
            available = ", ".join(self.names.values())
            raise ValueError(
                "Configured model does not expose the target label(s). "
                f"Expected one of {sorted(normalized_targets)}, available labels: {available}"
            )

    @staticmethod
    def _extract_name_map(names: dict | list) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def detect(self, image_path: Path, confidence_threshold: float) -> DetectionResult:
        results = self.model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            verbose=False,
            device="cpu",
            classes=sorted(self.target_class_ids),
        )
        if not results:
            return DetectionResult(detected=False, matched_labels=[], matched_confidences=[])

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return DetectionResult(detected=False, matched_labels=[], matched_confidences=[])

        raw_class_ids = [int(value) for value in boxes.cls.cpu().tolist()]
        raw_confidences = [float(value) for value in boxes.conf.cpu().tolist()]

        matched_labels: list[str] = []
        matched_confidences: list[float] = []
        for class_id, confidence in zip(raw_class_ids, raw_confidences, strict=True):
            if class_id in self.target_class_ids:
                matched_labels.append(self.names[class_id])
                matched_confidences.append(round(confidence, 6))

        return DetectionResult(
            detected=bool(matched_labels),
            matched_labels=matched_labels,
            matched_confidences=matched_confidences,
        )

