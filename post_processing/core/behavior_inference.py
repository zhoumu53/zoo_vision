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
        device: torch.device | str = "cuda:0",
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
    
    def _convert_to_rgb(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Convert BGR images to RGB if needed."""
        rgb_images = []
        for img in images:
            if img is None or img.size == 0:
                continue
            # Check if image is BGR (OpenCV format) by checking if it's a color image
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Assume BGR, convert to RGB
                rgb_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                rgb_images.append(img)
        return rgb_images
    
    def predict(
        self,
        images: List[np.ndarray],
        batch_size: int = 16,
    ) -> List[Tuple[str, float]]:
        """Run batch inference on pre-cropped images, returns [(label, confidence), ...]."""
        if not images:
            return []
        
        # Convert to RGB if needed
        rgb_images = self._convert_to_rgb(images)
        if not rgb_images:
            return []
        
        predictions = []
        effective_batch = max(batch_size, 1)
        
        for start in range(0, len(rgb_images), effective_batch):
            batch = list(rgb_images[start : start + effective_batch])
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
    
    def predict_tracks(
        self,
        frames: List[np.ndarray],
        batch_size: int = 16,
        n_frames: Optional[int] = None,
    ) -> Tuple[str, float, Dict[str, int]]:
        """
        Predict behavior for a track by aggregating predictions across frames.
        
        Args:
            frames: List of pre-cropped images (BGR or RGB)
            batch_size: Batch size for inference
            n_frames: Number of frames to sample. If None, use all frames.
                     If specified, uniformly sample n_frames from the track.
        
        Returns:
            Tuple of (final_label, avg_confidence, vote_distribution)
            - final_label: Most frequent predicted label
            - avg_confidence: Average confidence across all predictions
            - vote_distribution: Dict mapping labels to vote counts
        
        Example:
            >>> frames = [cv2.imread(f"track_{i}.jpg") for i in range(100)]
            >>> label, conf, votes = behavior.predict_tracks(frames, n_frames=10)
            >>> print(f"Behavior: {label} (conf={conf:.3f}, votes={votes})")
        """
        if not frames:
            return "unknown", 0.0, {}
        
        # Sample frames if n_frames is specified
        if n_frames is not None and n_frames > 0 and len(frames) > n_frames:
            # Uniform sampling
            indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames
        
        # Run predictions on all sampled frames
        predictions = self.predict(sampled_frames, batch_size)
        
        if not predictions:
            return "unknown", 0.0, {}
        
        # Count votes for each label
        vote_counts = {}
        total_confidence = 0.0
        
        for label, conf in predictions:
            vote_counts[label] = vote_counts.get(label, 0) + 1
            total_confidence += conf
        
        # Find most frequent label
        final_label = max(vote_counts, key=vote_counts.get)
        avg_confidence = total_confidence / len(predictions)
        
        return predictions, final_label, avg_confidence, vote_counts
    
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
    
    def predict_single(
        self,
        image: np.ndarray,
    ) -> Tuple[str, float]:
        """Predict behavior for a single pre-cropped image."""
        predictions = self.predict([image], batch_size=1)
        if not predictions:
            return "unknown", 0.0
        return predictions[0]