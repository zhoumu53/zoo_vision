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
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
import timm
from torchvision import transforms

QUALITY_CLASSES = {
    "good": 0,
    "bad": 1,
}

QUALITY_ID2LABEL = {v: k for k, v in QUALITY_CLASSES.items()}

BEHAVIOR_CLASSES = {
    "00_invalid": 0,
    "01_standing": 1,
    "02_sleeping_left": 2,
    "03_sleeping_right": 3,
}

class DualHeadModel(nn.Module):
    """
    Dual-head model with shared backbone.
    - Head 1: behavior classification (4 classes)
    - Head 2: quality classification (2 classes)
    """
    def __init__(self, backbone_name: str, num_behavior_classes: int = 4, num_quality_classes: int = 2, img_size: int = 224):
        super().__init__()
        
        # Create backbone (without classifier head)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,  # No classifier head
            img_size=img_size
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            feat_dim = self.backbone(dummy).shape[1]
        
        # Create two classifier heads
        self.head_behavior = nn.Linear(feat_dim, num_behavior_classes)
        self.head_quality = nn.Linear(feat_dim, num_quality_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits_behavior = self.head_behavior(features)
        logits_quality = self.head_quality(features)
        return logits_behavior, logits_quality


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
        
        model_path_obj = Path(model_path).expanduser()
        
        # Detect model type: HuggingFace (config.json) vs two-stage timm checkpoint (.pt)
        self.is_timm_model, self.is_two_heads = self._is_timm_checkpoint(model_path_obj)
        
        if self.is_timm_model:
            # Load new Swin-B model
            self._load_timm_model(model_path_obj)
        else:
            # Load old HuggingFace model
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
            model_path, self.num_classes, device
        )
        self.logger.info("Behavior classes: %s", list(self.id2label.values()))
    
    def _is_timm_checkpoint(self, path: Path) -> bool:
        """Check if path points to a timm checkpoint (.pt file from two-stage training)."""
        # Check if it's a .pt file
        is_timm = False
        is_two_heads = False
        if path.is_file() and path.suffix == ".pt":
            is_timm = True
        
        # Check if it's a directory containing stage1 checkpoint
        if path.is_dir():
            
            if 'two_stage' in path.name:
                best_model = path / "stage1_4class" / "best_stage1_4class.pt"
                last_model = path / "stage1_4class" / "last_stage1_4class.pt"
                
                if best_model.exists() or last_model.exists():
                    is_timm = True
                    is_two_heads = False
            elif 'two_heads' in path.name:
                best_model = path / "best_dual_head.pt"
                last_model = path / "last_dual_head.pt"
                
                if best_model.exists() or last_model.exists():
                    is_timm = True
                    is_two_heads = True
            
        return is_timm, is_two_heads
    
    def _load_timm_model(self, path: Path):
        """Load timm model from two-stage training checkpoint."""
        # Find the checkpoint file
        if path.is_file() and path.suffix == ".pt":
            ckpt_path = path
        elif path.is_dir():
            if not self.is_two_heads:
                best_model = path / "stage1_4class" / "best_stage1_4class.pt"
                last_model = path / "stage1_4class" / "last_stage1_4class.pt"
            else:
                best_model = path / "best_dual_head.pt"
                last_model = path / "last_dual_head.pt"
            if best_model.exists():
                ckpt_path = best_model
            elif last_model.exists():
                ckpt_path = last_model
            else:
                raise FileNotFoundError(
                    f"No checkpoint found in {path}. "
                    f"Expected: best_dual_head.pt or stage1_4class/best_stage1_4class.pt"
                )
        else:
            raise FileNotFoundError(f"Invalid path for timm model: {path}")
        
        self.logger.info(f"Loading timm checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_name = ckpt.get("model_name", "swin_base_patch4_window7_224")
        img_size = ckpt.get("img_size", 224)
        
        self.num_classes = 4
        
        if self.is_two_heads:
            # Load dual-head model
            self.model = DualHeadModel(
                backbone_name=model_name,
                num_behavior_classes=4,
                num_quality_classes=2,
                img_size=img_size
            )
            self.model.load_state_dict(ckpt["state_dict"], strict=True)
            self.logger.info("Loaded dual-head model (behavior + quality)")
        else:
            # Load single-head model (stage 1 only)
            self.model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=self.num_classes,
                img_size=img_size
            )
            self.model.load_state_dict(ckpt["state_dict"], strict=True)
            self.logger.info("Loaded single-head model (behavior only)")
        
        self.model.eval()
        self.model.to(self.device)
        
        self.id2label = {
            0: "00_invalid",
            1: "01_standing",
            2: "02_sleeping_left",
            3: "03_sleeping_right",
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Create preprocessing transform matching the training pipeline
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.processor = None  # Not used for timm models
    
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
        
        if self.is_timm_model:
            return self._predict_timm(rgb_images, batch_size)
        else:
            return self._predict_huggingface(rgb_images, batch_size)
    
    def _predict_huggingface(
        self,
        rgb_images: List[np.ndarray],
        batch_size: int,
    ) -> List[Tuple[str, float]]:
        """Predict using HuggingFace model."""
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
    
    def _predict_timm(
        self,
        rgb_images: List[np.ndarray],
        batch_size: int,
    ) -> List[Tuple[str, float]]:
        """Predict using timm model."""
        predictions = []
        effective_batch = max(batch_size, 1)
        
        for start in range(0, len(rgb_images), effective_batch):
            batch = rgb_images[start : start + effective_batch]
            if not batch:
                continue
            
            # Preprocess images
            tensors = []
            for img in batch:
                if img is None or img.size == 0:
                    continue
                tensor = self.transform(img)
                tensors.append(tensor)
            
            if not tensors:
                continue
            
            # Stack into batch
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                if self.is_two_heads:
                    # Dual-head model returns (logits_behavior, logits_quality)
                    logits_behavior, logits_quality = self.model(batch_tensor)
                    beh_probs = F.softmax(logits_behavior, dim=-1)
                    beh_max_probs, beh_max_ids = torch.max(beh_probs, dim=-1)

                    qual_probs = F.softmax(logits_quality, dim=-1)
                    qual_max_probs, qual_max_ids = torch.max(qual_probs, dim=-1)

                else:
                    # Single-head model returns logits directly
                    logits = self.model(batch_tensor)
                    probs = F.softmax(logits, dim=-1)
                    max_probs, max_ids = torch.max(probs, dim=-1)

                    quality_ids = None

            
            # for prob, idx in zip(max_probs, max_ids):
            #     label = self.id2label.get(int(idx.item()), str(int(idx.item())))
            #     predictions.append((label, float(prob.item())))
                
            # for i, (prob, idx) in enumerate(zip(max_probs, max_ids)):
            #     label_id = int(idx.item())
            #     # quality gating
            #     if self.is_two_heads:
            #         qid = int(quality_ids[i].item())
            #         if qid == QUALITY_CLASSES["bad"]:
            #             label_id = 0  # force to invalid / background

            #     label = self.id2label.get(label_id, str(label_id))
            #     predictions.append((label, float(prob.item())))
                        
            for i in range(len(beh_max_ids)):
                beh_id = int(beh_max_ids[i].item())
                beh_prob = float(beh_max_probs[i].item())
                beh_label = self.id2label.get(beh_id, str(beh_id))

                if self.is_two_heads:
                    qual_id = int(qual_max_ids[i].item())
                    qual_prob = float(qual_max_probs[i].item())
                    qual_label = QUALITY_ID2LABEL.get(qual_id, str(qual_id))
                else:
                    qual_label = "unknown"
                    qual_prob = 1.0

                # ---- combined outputs ----
                combined_label = f"{beh_label}_{qual_label}"
                combined_prob = f"{beh_prob:.4f}_{qual_prob:.4f}"

                predictions.append((combined_label, combined_prob))

        return predictions
    

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