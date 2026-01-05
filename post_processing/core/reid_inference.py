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
from project.model import make_model
from project.utils.tools import load_model
from project.utils.metrics import eval_func_gpu, compute_cosine_distance


class ReIDInference:
    """ReID feature extraction and matching for elephant re-identification."""
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: torch.device | str = "cpu",
        num_classes: int = 6,
        logger: Optional[logging.Logger] = None,
        mode: str = "feature",  # "feature" or "classification"
    ):
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.num_classes = num_classes
        self.mode = mode
        
        self.cfg = base_cfg.clone()
        self.cfg.defrost()
        self.cfg.merge_from_file(config_path)
        self.cfg.TEST.WEIGHT = checkpoint_path
        self.cfg.freeze()
        
        # For classification mode, we need the FC layer
        remove_fc = (mode == "feature")
        
        self.model = make_model(
            self.cfg,
            load_weights=False,
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
            remove_fc=remove_fc,
            local_rank=0,
            is_swin=("swin" in self.cfg.MODEL.TYPE),
        )
        
        self.model.eval()
        self.model.to(device)
        
        self.transform = get_transforms(self.cfg, is_train=False)
        self.feature_dim = self._infer_feature_dim()
        
        self.logger.info(
            "ReID model loaded: config=%s, checkpoint=%s, device=%s, mode=%s, feat_dim=%d",
            config_path, checkpoint_path, device, mode, self.feature_dim
        )
    
    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
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
        # check device compatibility
        if isinstance(features_a, np.ndarray):
            features_a = torch.from_numpy(features_a).to(self.device)
        if isinstance(features_b, np.ndarray):
            features_b = torch.from_numpy(features_b).to(self.device)
        features_a = torch.nn.functional.normalize(features_a, dim=1)
        features_b = torch.nn.functional.normalize(features_b, dim=1)
        return torch.mm(features_a, features_b.t())
    
    def match_to_gallery(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor,
        gallery_labels: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match query features to gallery, return top-k (scores, indices)."""
        similarity = self.compute_similarity(query_features, gallery_features)
        effective_k = min(top_k, gallery_features.shape[0])
        scores, indices = torch.topk(similarity, effective_k, dim=1)

        if gallery_labels is not None:
            matched_labels = []
            for i in range(indices.shape[0]):
                matched = []
                for j in range(effective_k):
                    idx = indices[i, j].item()
                    matched.append(gallery_labels[idx])
                matched_labels.append(matched)
            # print("np.array(matched_labels).shape: ", np.array(matched_labels).shape, np.array(scores.cpu().numpy()).shape)
            return scores, indices, matched_labels
        return scores, indices, None
    
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
        """Load feature database from NPZ file, returns (features, label, ids)."""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Feature file not found: {load_path}")
        
        data = np.load(load_path, allow_pickle=True)

        labels = data.get("label", None)
        features = torch.from_numpy(data["feature"]).float()
        features = features.to(self.device)
        self.logger.info("Loaded %d features from %s", len(features), load_path)
        return features, labels
    
    def run_prediction(
        self,
        batch: torch.Tensor,
        gt_labels: Optional[List[int]] = None,
        is_evaluate: bool = False,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Run prediction on a batch of images.
        Note: This requires the model to be loaded in 'classification' mode.
        
        Args:
            batch: Preprocessed image tensor (N, 3, H, W)
            gt_labels: Ground truth labels for evaluation (optional)
            is_evaluate: Whether to compute accuracy
            
        Returns:
            predictions: Predicted class IDs (N,) - mapped to 0 to num_classes-1
            accuracy: Classification accuracy if is_evaluate=True, else None
        """
        if self.mode == "feature":
            raise ValueError(
                "Classification not available in 'feature' mode. "
                "Initialize ReIDInference with mode='classification'."
            )
        
        with torch.no_grad():
            outputs = self.model(batch)
        
        # Extract logits from model output
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 4:
                logits = outputs[0]  # classification logits
            else:
                logits = outputs[0] if len(outputs) > 0 else outputs
        else:
            logits = outputs
        
        # Get predicted class
        confidences, predictions = torch.max(logits.data, 1)
        predictions = predictions.cpu().numpy()
        
        # Predictions are already in range [0, num_classes-1] when FC layer is present
        accuracy = None
        if is_evaluate and gt_labels is not None:
            gt_labels = np.array(gt_labels)
            correct = np.sum(predictions == gt_labels)
            accuracy = correct / len(gt_labels)
            self.logger.info(
                "Classification accuracy: %.2f%% (%d/%d)",
                accuracy * 100, correct, len(gt_labels)
            )
        
        return predictions, accuracy
    
    def compute_cmc(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor,
        query_labels: np.ndarray,
        gallery_labels: np.ndarray,
        query_dates: Optional[np.ndarray] = None,
        gallery_dates: Optional[np.ndarray] = None,
        query_paths: Optional[np.ndarray] = None,
        gallery_paths: Optional[np.ndarray] = None,
        max_rank: int = 50,
        filter_date: bool = False,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Compute CMC (Cumulative Matching Characteristics) curve and mAP.
        
        Args:
            query_features: Query feature embeddings (N_q, D)
            gallery_features: Gallery feature embeddings (N_g, D)
            query_labels: Query identity labels (N_q,)
            gallery_labels: Gallery identity labels (N_g,)
            query_dates: Query dates for filtering (optional)
            gallery_dates: Gallery dates for filtering (optional)
            query_paths: Query image paths for duplicate filtering (optional)
            gallery_paths: Gallery image paths for duplicate filtering (optional)
            max_rank: Maximum rank for CMC curve
            filter_date: Whether to filter same-date samples
            
        Returns:
            cmc: CMC curve (max_rank,)
            mAP: Mean Average Precision
            all_AP: List of Average Precision for each query
        """
        # Normalize features
        query_features = torch.nn.functional.normalize(query_features, dim=1)
        gallery_features = torch.nn.functional.normalize(gallery_features, dim=1)
        
        # Compute distance matrix (cosine distance)
        distmat = compute_cosine_distance(query_features, gallery_features, batch_size=1000)
        
        # Prepare dates and paths
        if query_dates is None:
            query_dates = np.zeros(len(query_labels), dtype=str)
        if gallery_dates is None:
            gallery_dates = np.zeros(len(gallery_labels), dtype=str)
        
        # # Evaluate using GPU-accelerated function
        cmc, mAP, all_AP, sorted_indices, valid_indices = eval_func_gpu(
            distmat=distmat,
            q_pids=query_labels,
            g_pids=gallery_labels,
            q_dates=query_dates,
            g_dates=gallery_dates,
            q_paths=query_paths,
            g_paths=gallery_paths,
            max_rank=max_rank,
            device=str(self.device),
            mAP_for_max_rank=False,
            filter_date=filter_date,
            batch_size=100,
        )
        
        # Compute per-class metrics
        unique_labels = np.unique(query_labels)
        per_class_metrics = {}
        
        for label_id in unique_labels:
            # Find queries with this label
            label_mask = query_labels == label_id
            label_indices = np.where(label_mask)[0]
            
            if len(label_indices) == 0:
                continue
            
            # Get APs for this class
            class_APs = [all_AP[i] for i in label_indices if not np.isnan(all_AP[i])]
            
            if len(class_APs) > 0:
                class_mAP = np.mean(class_APs)
                
                # Compute per-class CMC
                # Extract the sorted indices for queries of this class
                class_cmc_list = []
                for q_idx in label_indices:
                    if q_idx < len(valid_indices) and len(valid_indices[q_idx]) > 0:
                        # Get gallery labels for this query's ranking
                        valid_gallery_indices = valid_indices[q_idx]
                        if len(valid_gallery_indices) > 0:
                            ranked_gallery_labels = gallery_labels[valid_gallery_indices[:max_rank]]
                            # Check if correct label appears in ranking
                            matches = (ranked_gallery_labels == label_id).astype(float)
                            if np.sum(matches) > 0:
                                cmc_curve = np.maximum.accumulate(matches)
                                class_cmc_list.append(cmc_curve)
                
                if len(class_cmc_list) > 0:
                    # Pad CMCs to max_rank
                    padded_cmcs = []
                    for cmc_curve in class_cmc_list:
                        if len(cmc_curve) < max_rank:
                            padded = np.pad(cmc_curve, (0, max_rank - len(cmc_curve)), 
                                          mode='edge')
                        else:
                            padded = cmc_curve[:max_rank]
                        padded_cmcs.append(padded)
                    
                    class_cmc = np.mean(padded_cmcs, axis=0)
                    
                    per_class_metrics[int(label_id)] = {
                        'mAP': class_mAP,
                        'CMC': class_cmc,
                        'Rank-1': class_cmc[0] if len(class_cmc) > 0 else 0,
                        'Rank-5': class_cmc[4] if len(class_cmc) > 4 else 0,
                        'Rank-10': class_cmc[9] if len(class_cmc) > 9 else 0,
                        'num_queries': len(label_indices),
                        'num_valid': len(class_APs),
                    }
                    
                    self.logger.info(
                        "Class %d - mAP: %.2f%%, Rank-1: %.2f%%, Rank-5: %.2f%%, Rank-10: %.2f%% (queries: %d)",
                        label_id,
                        class_mAP * 100,
                        per_class_metrics[int(label_id)]['Rank-1'] * 100,
                        per_class_metrics[int(label_id)]['Rank-5'] * 100,
                        per_class_metrics[int(label_id)]['Rank-10'] * 100,
                        len(label_indices),
                    )
        
        # Store per-class metrics for access
        self.per_class_metrics = per_class_metrics
        
        self.logger.info(
            "Overall CMC Evaluation - mAP: %.2f%%, Rank-1: %.2f%%, Rank-5: %.2f%%, Rank-10: %.2f%%",
            mAP * 100,
            cmc[0] * 100 if len(cmc) > 0 else 0,
            cmc[4] * 100 if len(cmc) > 4 else 0,
            cmc[9] * 100 if len(cmc) > 9 else 0,
        )
        
        return cmc, mAP, all_AP