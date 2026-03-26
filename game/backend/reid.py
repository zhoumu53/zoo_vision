"""
ReID feature management: load gallery, FAISS HNSW index, match features.

Uses the project's ReIDInference class for real feature extraction.

Matching pipeline:
  1. FAISS HNSW coarse search retrieves candidates from 74k+ gallery images
  2. Exact cosine similarity reranks candidates
  3. Top-k voting: the top VOTE_K nearest images vote for an elephant
  4. Match level is determined by vote ratio + mean similarity of voters:
       "same"    — dominant vote share + high mean sim
       "similar" — moderate vote share or moderate mean sim
       "unknown" — scattered votes or low similarity
"""

import logging
import sys
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch

from config import (
    GALAXY_POSITION_SCALE,
    VOTE_K,
    SAME_VOTE_RATIO,
    SAME_MEAN_SIM,
    SIMILAR_VOTE_RATIO,
    SIMILAR_MEAN_SIM,
    TOP_K_MATCHES,
    REID_CONFIG_PATH,
    REID_CHECKPOINT_PATH,
    REID_NUM_CLASSES,
    ELEPHANT_INFO,
)

# Add project paths so we can import ReIDInference
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POSE_REID_ROOT = PROJECT_ROOT / "training" / "PoseGuidedReID"
for p in (PROJECT_ROOT, POSE_REID_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

logger = logging.getLogger(__name__)

# FAISS HNSW parameters
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 256
COARSE_TOP_K = 5000


class GalleryManager:
    """Manages the gallery of elephant features with FAISS HNSW index."""

    def __init__(self, npz_path: Path):
        self.npz_path = npz_path
        self.features: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.label_ids: np.ndarray | None = None
        self.paths: np.ndarray | None = None
        self.centroids: dict[str, np.ndarray] = {}
        self.pca_params: dict | None = None
        self.elephant_positions: dict[str, tuple[float, float, float]] = {}
        self._index: faiss.Index | None = None
        self._unique_labels: list[str] = []
        self._label_to_int: dict[str, int] = {}

    def load(self) -> None:
        """Load gallery features from NPZ and build FAISS HNSW index."""
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Gallery NPZ not found: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)
        self.features = data["feature"].astype(np.float32)
        self.labels = data["label"]
        self.paths = data.get("path", np.array([""] * len(self.labels)))

        norms = np.linalg.norm(self.features, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        self.features = self.features / norms

        self._unique_labels = sorted(set(self.labels))
        self._label_to_int = {l: i for i, l in enumerate(self._unique_labels)}
        self.label_ids = np.array([self._label_to_int[l] for l in self.labels], dtype=np.int64)

        logger.info("Loaded %d features (%d dim) from %s",
                     len(self.features), self.features.shape[1], self.npz_path)

        self._build_faiss_index()
        self._compute_centroids()
        self._compute_pca_positions()

    def _build_faiss_index(self) -> None:
        t0 = time.perf_counter()
        dim = self.features.shape[1]
        self._index = faiss.IndexHNSWFlat(dim, HNSW_M)
        self._index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        self._index.hnsw.efSearch = HNSW_EF_SEARCH
        self._index.add(self.features)
        logger.info("FAISS HNSW index built: %d vectors, %.0f ms",
                     self.features.shape[0], (time.perf_counter() - t0) * 1000)

    def _compute_centroids(self) -> None:
        self.centroids = {}
        for label in self._unique_labels:
            mask = self.labels == label
            centroid = self.features[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-9:
                centroid = centroid / norm
            self.centroids[label] = centroid
        logger.info("Computed centroids for %d elephants", len(self.centroids))

    def _compute_pca_positions(self) -> None:
        labels = list(self.centroids.keys())
        if len(labels) < 3:
            for i, label in enumerate(labels):
                angle = i * 2.356
                self.elephant_positions[label] = (
                    float(np.cos(angle) * 15),
                    float(np.sin(angle) * 15),
                    float(i * 5 - len(labels) * 2.5),
                )
            self.pca_params = None
            return

        centroids_arr = np.stack([self.centroids[l] for l in labels])
        mean = centroids_arr.mean(axis=0, keepdims=True)
        centered = centroids_arr - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        coords_3d = U[:, :3] * S[:3]
        max_range = float(np.abs(coords_3d).max())
        if max_range > 0:
            coords_3d = coords_3d / max_range * GALAXY_POSITION_SCALE

        self.pca_params = {"mean": mean, "Vt": Vt, "max_range": max_range}
        for i, label in enumerate(labels):
            self.elephant_positions[label] = (
                float(coords_3d[i, 0]),
                float(coords_3d[i, 1]),
                float(coords_3d[i, 2]),
            )
        logger.info("PCA positions computed for %d elephants", len(labels))

    def project_to_3d(self, feature: np.ndarray) -> tuple[float, float, float] | None:
        if self.pca_params is None:
            return None
        mean = self.pca_params["mean"]
        Vt = self.pca_params["Vt"]
        max_range = self.pca_params["max_range"]
        centered = feature.reshape(1, -1) - mean
        coords = centered @ Vt[:3].T
        if max_range > 0:
            coords = coords / max_range * GALAXY_POSITION_SCALE
        return (float(coords[0, 0]), float(coords[0, 1]), float(coords[0, 2]))

    def find_nearest(
        self, feature: np.ndarray, top_k: int = TOP_K_MATCHES
    ) -> list[dict]:
        """Find nearest elephants using FAISS HNSW + top-k voting.

        1. HNSW retrieves COARSE_TOP_K candidates.
        2. Exact cosine similarity reranks them.
        3. Top VOTE_K images vote: count per elephant + mean sim per elephant.
        4. Winner = most votes. Match level from vote ratio + mean sim.
        """
        t0 = time.perf_counter()

        # L2-normalize query
        feature = feature.astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm > 1e-9:
            feature = feature / norm

        # --- HNSW coarse search ---
        distances, indices = self._index.search(feature.reshape(1, -1), COARSE_TOP_K)
        candidate_idx = indices[0]
        candidate_idx = candidate_idx[candidate_idx >= 0]

        # --- Exact cosine rerank ---
        exact_sims = (self.features[candidate_idx] @ feature).astype(np.float64)
        rank_order = np.argsort(-exact_sims)

        # --- Top-k voting ---
        vote_k = min(VOTE_K, len(rank_order))
        top_k_idx = candidate_idx[rank_order[:vote_k]]
        top_k_sims = exact_sims[rank_order[:vote_k]]
        top_k_labels = self.labels[top_k_idx]

        # Per-elephant: vote count + mean similarity of their voters
        vote_counts: Counter = Counter(top_k_labels)
        elephant_mean_sim: dict[str, float] = {}
        elephant_max_sim: dict[str, float] = {}
        for label in vote_counts:
            mask = top_k_labels == label
            sims = top_k_sims[mask]
            elephant_mean_sim[label] = float(np.mean(sims))
            elephant_max_sim[label] = float(np.max(sims))

        # Centroid fallback for elephants with zero votes
        for label in self._unique_labels:
            if label not in vote_counts:
                vote_counts[label] = 0
                csim = float(self.centroids[label] @ feature)
                elephant_mean_sim[label] = csim
                elephant_max_sim[label] = csim

        # Rank by vote count first, then by mean similarity as tiebreaker
        sorted_labels = sorted(
            self._unique_labels,
            key=lambda l: (vote_counts[l], elephant_mean_sim[l]),
            reverse=True,
        )

        # Winner stats
        winner = sorted_labels[0]
        winner_votes = vote_counts[winner]
        winner_ratio = winner_votes / vote_k if vote_k > 0 else 0.0
        winner_mean = elephant_mean_sim[winner]
        winner_max = elephant_max_sim[winner]
        runner_up = sorted_labels[1] if len(sorted_labels) > 1 else ""
        runner_votes = vote_counts.get(runner_up, 0)
        runner_ratio = runner_votes / vote_k if vote_k > 0 else 0.0

        query_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "find_nearest: %.1f ms | #1 %s votes=%d/%d (%.0f%%) mean=%.3f max=%.3f | "
            "#2 %s votes=%d (%.0f%%)",
            query_ms, winner, winner_votes, vote_k, winner_ratio * 100,
            winner_mean, winner_max,
            runner_up, runner_votes, runner_ratio * 100,
        )

        # --- Build results ---
        results = []
        for i, label in enumerate(sorted_labels[:top_k]):
            v_count = vote_counts[label]
            v_ratio = v_count / vote_k if vote_k > 0 else 0.0
            mean_sim = elephant_mean_sim[label]
            max_sim = elephant_max_sim[label]
            cosine_distance = round(1.0 - max_sim, 4)

            # Match level: top-1 gets full classification, others simple
            if i == 0:
                match_level = self._classify_topk(v_ratio, mean_sim)
            else:
                # Lower-ranked: use their own vote ratio + mean sim
                match_level = self._classify_topk(v_ratio, mean_sim)

            pos = self.elephant_positions.get(label)
            position = {"x": pos[0], "y": pos[1], "z": pos[2]} if pos else None
            image_count = int(np.sum(self.labels == label))
            info = ELEPHANT_INFO.get(label, {})

            results.append({
                "elephant_label": label,
                "similarity": max_sim,
                "mean_similarity": round(mean_sim, 4),
                "cosine_distance": cosine_distance,
                "match_level": match_level,
                "vote_count": v_count,
                "vote_ratio": round(v_ratio, 3),
                "margin": round(winner_ratio - runner_ratio, 3) if i == 0 else None,
                "image_count": image_count,
                "sample_crop_path": info.get("profile"),
                "position": position,
            })

        return results

    @staticmethod
    def _classify_topk(vote_ratio: float, mean_sim: float) -> str:
        """Classify match level from vote ratio and mean similarity.

        "same":    dominant vote share (>=60%) AND high mean sim (>=0.90)
        "similar": moderate vote share (>=30%) AND moderate mean sim (>=0.70)
        "unknown": everything else
        """
        if vote_ratio >= SAME_VOTE_RATIO and mean_sim >= SAME_MEAN_SIM:
            return "same"
        if vote_ratio >= SIMILAR_VOTE_RATIO and mean_sim >= SIMILAR_MEAN_SIM:
            return "similar"
        return "unknown"

    def get_elephants(self) -> list[dict]:
        elephants = []
        for label in sorted(self.centroids.keys()):
            mask = self.labels == label
            image_count = int(np.sum(mask))
            pos = self.elephant_positions.get(label)
            info = ELEPHANT_INFO.get(label, {})
            elephants.append({
                "elephant_label": label,
                "image_count": image_count,
                "sample_crop_path": info.get("profile"),
                "x": pos[0] if pos else None,
                "y": pos[1] if pos else None,
                "z": pos[2] if pos else None,
            })
        return elephants


class ReIDExtractor:
    """Wraps the project's ReIDInference for single-image feature extraction."""

    def __init__(self):
        self._model = None
        self._transform = None

    def _load(self):
        if self._model is not None:
            return
        from post_processing.core.reid_inference import ReIDInference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading ReID model on %s ...", device)
        self._model = ReIDInference(
            config_path=str(REID_CONFIG_PATH),
            checkpoint_path=str(REID_CHECKPOINT_PATH),
            device=device,
            num_classes=REID_NUM_CLASSES,
            mode="feature",
        )
        self._transform = self._model.transform
        logger.info("ReID model loaded, feature_dim=%d", self._model.feature_dim)

    def extract(self, bgr_image: np.ndarray) -> np.ndarray:
        self._load()
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self._transform(pil_img).unsqueeze(0).to(self._model.device)
        feat = self._model.extract_features(tensor)
        return feat.cpu().numpy().flatten()
