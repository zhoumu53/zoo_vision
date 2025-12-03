"""
ReID Inference Module
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

import sys
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
POSE_REID_ROOT = PROJECT_ROOT / "training" / "PoseGuidedReID"

for path in (PROJECT_ROOT, POSE_REID_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from project.config import cfg as base_cfg
from project.datasets.make_dataloader import get_transforms
from project.models import make_model
from project.utils.tools import load_model


class ReIDInference:
    """ReID feature extraction and matching for elephant re-identification."""
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: torch.device | str = "cpu",
        num_classes: int = 6,
        logger: Optional[logging.Logger] = None,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.num_classes = num_classes
        
        self.cfg = base_cfg.clone()
        self.cfg.defrost()
        self.cfg.merge_from_file(config_path)
        self.cfg.TEST.WEIGHT = checkpoint_path
        self.cfg.freeze()
        
        self.model = make_model(
            self.cfg,
            num_classes=num_classes,
            logger=self.logger,
            return_feature=True,
            device=device,
            camera_num=0,
            view_num=0,
        )
        
        self.model = load_model(
            self.model,
            checkpoint_path,
            logger=self.logger,
            remove_fc=True,
            local_rank=0,
            is_swin=("swin" in self.cfg.MODEL.TYPE),
        )
        
        self.model.eval()
        self.model.to(device)
        
        self.transform = get_transforms(self.cfg, is_train=False)
        self.feature_dim = self._infer_feature_dim()
        
        self.logger.info(
            "ReID model loaded: config=%s, checkpoint=%s, device=%s, feat_dim=%d",
            config_path, checkpoint_path, device, self.feature_dim
        )
    
    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 128).to(self.device)
            dummy_feat = self.extract_features(dummy_input)
            return dummy_feat.shape[1]
    
    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract normalized features from preprocessed tensor batch (N, 3, H, W)."""
        with torch.no_grad():
            outputs = self.model(batch)
        
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 4:
                feat = outputs[2]
            elif len(outputs) > 1:
                feat = outputs[1]
            else:
                feat = outputs[0]
        else:
            feat = outputs
        
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat
    
    def compute_similarity(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity matrix (N, M) between two feature sets."""
        features_a = torch.nn.functional.normalize(features_a, dim=1)
        features_b = torch.nn.functional.normalize(features_b, dim=1)
        return torch.mm(features_a, features_b.t())
    
    def match_to_gallery(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match query features to gallery, return top-k (scores, indices)."""
        similarity = self.compute_similarity(query_features, gallery_features)
        effective_k = min(top_k, gallery_features.shape[0])
        scores, indices = torch.topk(similarity, effective_k, dim=1)
        return scores, indices
    
    def save_features(
        self,
        features: torch.Tensor,
        labels: List[str],
        ids: List[int],
        save_path: str | Path,
        metadata: Optional[dict] = None,
    ) -> None:
        """Save feature database to NPZ file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "features": features.cpu().numpy(),
            "labels": np.array(labels),
            "ids": np.array(ids),
        }
        
        if metadata:
            save_dict["metadata"] = metadata
        
        np.savez_compressed(save_path, **save_dict)
        self.logger.info("Saved %d features to %s", len(features), save_path)
    
    def load_features(
        self,
        load_path: str | Path,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Load feature database from NPZ file, returns (features, labels, ids)."""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Feature file not found: {load_path}")
        
        data = np.load(load_path, allow_pickle=True)
        
        features = torch.from_numpy(data["features"]).float()
        features = torch.nn.functional.normalize(features, dim=1)
        features = features.to(self.device)
        
        self.logger.info("Loaded %d features from %s", len(features), load_path)
        return features, data["labels"], data["ids"]