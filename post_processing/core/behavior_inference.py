"""
Behavior Classification Inference Module
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification


class BehaviorInference:
    """Behavior classification for elephant activity recognition."""
    
    def __init__(
        self,
        model_path: str,
        device: torch.device | str = "cpu",
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        model_dir = self._resolve_model_path(model_path)
        
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(model_dir)
        self.model.eval()
        self.model.to(device)
        
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.label2id = {v: int(k) for k, v in self.id2label.items()}
        self.num_classes = len(self.id2label)
        
        self.logger.info(
            "Behavior classifier loaded: %s (%d classes) on %s",
            model_dir, self.num_classes, device
        )
        self.logger.info("Behavior classes: %s", list(self.id2label.values()))
    
    def _resolve_model_path(self, path_str: str) -> Path:
        """Resolve model path from directory, config.json, or .ptc file."""
        path = Path(path_str).expanduser()
        
        if path.is_dir():
            return path
        
        if path.is_file():
            if path.name == "config.json":
                return path.parent
            
            if path.suffix == ".ptc":
                parent = path.parent
                config_path = parent / "config.json"
                if config_path.exists():
                    self.logger.info(
                        "Resolved TorchScript checkpoint %s to HuggingFace directory %s",
                        path, parent
                    )
                    return parent
        
        raise FileNotFoundError(
            f"Unable to resolve behavior model from {path_str}. "
            f"Provide a directory or config.json exported by HuggingFace."
        )
    
    def prepare_crops(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        context: float = 1.1,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Crop detection boxes with context padding, returns (RGB crops, valid_indices)."""
        h, w = frame.shape[:2]
        context = max(context, 1.0)
        
        crops = []
        valid_indices = []
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            
            half_w = 0.5 * box_w * context
            half_h = 0.5 * box_h * context
            
            new_x1 = max(0, int(np.floor(cx - half_w)))
            new_y1 = max(0, int(np.floor(cy - half_h)))
            new_x2 = min(w, int(np.ceil(cx + half_w)))
            new_y2 = min(h, int(np.ceil(cy + half_h)))
            
            if new_x2 <= new_x1 or new_y2 <= new_y1:
                continue
            
            crop = frame[new_y1:new_y2, new_x1:new_x2]
            if crop.size == 0:
                continue
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop_rgb)
            valid_indices.append(idx)
        
        return crops, valid_indices
    
    def predict(
        self,
        crops: List[np.ndarray],
        batch_size: int = 16,
    ) -> List[Tuple[str, float]]:
        """Run batch inference on RGB crops, returns [(label, confidence), ...]."""
        if not crops:
            return []
        
        predictions = []
        effective_batch = max(batch_size, 1)
        
        for start in range(0, len(crops), effective_batch):
            batch = list(crops[start : start + effective_batch])
            if not batch:
                continue
            
            encoded = self.processor(images=batch, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                max_probs, max_ids = torch.max(probs, dim=-1)
            
            for prob, idx in zip(max_probs, max_ids):
                label = self.id2label.get(int(idx.item()), str(int(idx.item())))
                predictions.append((label, float(prob.item())))
        
        return predictions
    
    def predict_from_frame(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        batch_size: int = 16,
        context: float = 1.1,
    ) -> Tuple[List[Tuple[str, float]], List[int]]:
        """Crop + predict from frame, returns (predictions, valid_indices)."""
        crops, valid_indices = self.prepare_crops(frame, boxes, context)
        
        if not crops:
            return [], []
        
        predictions = self.predict(crops, batch_size)
        return predictions, valid_indices
    
    def predict_batch_frames(
        self,
        frames: List[np.ndarray],
        boxes_per_frame: List[List[Tuple[int, int, int, int]]],
        batch_size: int = 16,
        context: float = 1.1,
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Predict behaviors for multiple frames in batch.
        
        Args:
            frames: List of BGR frames
            boxes_per_frame: List of box lists, one per frame
            batch_size: Batch size for inference
            context: Context scale factor
        
        Returns:
            List of prediction lists, one per frame.
            Each prediction is (label, confidence, box_index)
        
        Example:
            >>> frames = [cv2.imread(f"frame{i}.jpg") for i in range(5)]
            >>> boxes = [[(100, 100, 200, 300)] for _ in range(5)]
            >>> results = behavior.predict_batch_frames(frames, boxes)
            >>> for frame_idx, preds in enumerate(results):
            ...     print(f"Frame {frame_idx}: {preds}")
        """
        # Collect all crops with frame/box indices
        all_crops = []
        crop_metadata = []  # (frame_idx, box_idx)
        
        for frame_idx, (frame, boxes) in enumerate(zip(frames, boxes_per_frame)):
            crops, valid_indices = self.prepare_crops(frame, boxes, context)
            for crop_idx, box_idx in enumerate(valid_indices):
                all_crops.append(crops[crop_idx])
                crop_metadata.append((frame_idx, box_idx))
        
        # Run batch inference
    def predict_batch_frames(
        self,
        frames: List[np.ndarray],
        boxes_per_frame: List[List[Tuple[int, int, int, int]]],
        batch_size: int = 16,
        context: float = 1.1,
    ) -> List[List[Tuple[str, float, int]]]:
        """Predict behaviors for multiple frames, returns List[List[(label, conf, box_idx)]]."""
        all_crops = []
        crop_metadata = []
        
        for frame_idx, (frame, boxes) in enumerate(zip(frames, boxes_per_frame)):
            crops, valid_indices = self.prepare_crops(frame, boxes, context)
            for crop_idx, box_idx in enumerate(valid_indices):
                all_crops.append(crops[crop_idx])
                crop_metadata.append((frame_idx, box_idx))
        
        if not all_crops:
            return [[] for _ in frames]
        
        all_predictions = self.predict(all_crops, batch_size)
        
        results = [[] for _ in frames]
        for (label, conf), (frame_idx, box_idx) in zip(all_predictions, crop_metadata):
            results[frame_idx].append((label, conf, box_idx))
        
        return results
    
    def get_class_distribution(
        self,
        predictions: List[Tuple[str, float]],
    ) -> Dict[str, int]:
        """Get distribution of predicted classes."""
        distribution = {}
        for label, _ in predictions:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def filter_by_confidence(
        self,
        predictions: List[Tuple[str, float]],
        min_confidence: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Filter predictions by minimum confidence threshold."""
        return [(label, conf) for label, conf in predictions if conf >= min_confidence]