"""
Elephant detection using YOLOv8.

Adapted from:
  - zoo_vision/clean_nas_empty_videos/src/empty_video_tool/detection.py
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ElephantDetector:
    """YOLOv8-based elephant detector."""

    def __init__(
        self,
        weights_path: str | None = None,
        default_model: str = "yolov8n.pt",
        target_labels: tuple[str, ...] = ("elephant",),
        confidence_threshold: float = 0.65,
    ):
        from ultralytics import YOLO

        model_source = weights_path if weights_path else default_model
        if weights_path:
            p = Path(weights_path).expanduser()
            if not p.exists():
                logger.warning("Weights not found at %s, using default", weights_path)
                model_source = default_model

        self.model = YOLO(model_source)
        self.confidence_threshold = confidence_threshold

        # Map target labels to class IDs
        self.target_class_ids: set[int] = set()
        name_to_id = {v.lower(): k for k, v in self.model.names.items()}
        for label in target_labels:
            cid = name_to_id.get(label.lower())
            if cid is not None:
                self.target_class_ids.add(cid)
            else:
                logger.warning("Label '%s' not in model classes", label)

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Detect elephants in an image.

        Args:
            image: BGR numpy array (OpenCV format)

        Returns:
            List of detections: [{"bbox": [x1, y1, x2, y2], "confidence": float, "label": str}]
        """
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
            device="cpu",
            classes=sorted(self.target_class_ids) if self.target_class_ids else None,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.model.names.get(cls_id, "unknown")
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "label": label,
                    })

        return detections

    def detect_and_crop(self, image: np.ndarray, expand: float = 0.05) -> list[dict]:
        """
        Detect elephants and return body crops.

        Args:
            image: BGR numpy array
            expand: Expand bbox by this fraction (default 5%)

        Returns:
            List of {"bbox": [x1,y1,x2,y2], "crop": np.ndarray, "confidence": float}
        """
        detections = self.detect(image)
        h, w = image.shape[:2]
        results = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Expand bbox
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, x1 - bw * expand)
            y1 = max(0, y1 - bh * expand)
            x2 = min(w, x2 + bw * expand)
            y2 = min(h, y2 + bh * expand)

            crop = image[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue

            # Normalize bbox to [0, 1] (YOLO center format)
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h

            results.append({
                "bbox": {"x": cx, "y": cy, "w": nw, "h": nh},
                "bbox_abs": [int(x1), int(y1), int(x2), int(y2)],
                "crop": crop,
                "confidence": det["confidence"],
            })

        return results
