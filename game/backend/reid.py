"""
ReID feature management: load gallery, compute PCA, match features.

Adapted from:
  - zoo_vision/post_processing/core/reid_inference.py (feature extraction)
  - brown-bear-server/apps/api/app/api/galaxy.py (PCA, matching)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import (
    GALAXY_POSITION_SCALE,
    NEW_ELEPHANT_SIMILARITY_THRESHOLD,
    TOP_K_MATCHES,
)

logger = logging.getLogger(__name__)


class GalleryManager:
    """Manages the gallery of elephant features and provides matching."""

    def __init__(self, npz_path: Path):
        self.npz_path = npz_path
        self.features: np.ndarray | None = None  # (N, D)
        self.labels: np.ndarray | None = None  # (N,) elephant folder names
        self.paths: np.ndarray | None = None  # (N,) image paths
        self.centroids: dict[str, np.ndarray] = {}  # {label: (D,)}
        self.pca_params: dict | None = None
        self.elephant_positions: dict[str, tuple[float, float, float]] = {}

    def load(self) -> None:
        """Load gallery features from NPZ file."""
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Gallery NPZ not found: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)
        self.features = data["feature"].astype(np.float32)
        self.labels = data["label"]
        self.paths = data.get("path", np.array([""]*len(self.labels)))

        logger.info("Loaded %d features from %s", len(self.features), self.npz_path)

        self._compute_centroids()
        self._compute_pca_positions()

    def _compute_centroids(self) -> None:
        """Compute L2-normalized mean centroid per elephant."""
        unique_labels = np.unique(self.labels)
        self.centroids = {}

        for label in unique_labels:
            mask = self.labels == label
            feats = self.features[mask]
            centroid = feats.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-9:
                centroid = centroid / norm
            self.centroids[str(label)] = centroid

        logger.info("Computed centroids for %d elephants", len(self.centroids))

    def _compute_pca_positions(self) -> None:
        """Compute 3D PCA positions from centroids (SVD projection)."""
        labels = list(self.centroids.keys())
        if len(labels) < 3:
            # Spiral fallback for < 3 elephants
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
        """Project a feature vector into the 3D PCA space."""
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
        """Find nearest elephants to a feature vector using cosine similarity."""
        centroid_labels = list(self.centroids.keys())
        centroid_matrix = np.stack([self.centroids[l] for l in centroid_labels])

        # L2-normalize the query feature
        norm = np.linalg.norm(feature)
        if norm > 1e-9:
            feature = feature / norm

        # Cosine similarity (dot product of L2-normalized vectors)
        similarities = centroid_matrix @ feature
        sorted_indices = np.argsort(-similarities)

        results = []
        for idx in sorted_indices[:top_k]:
            label = centroid_labels[idx]
            sim = float(similarities[idx])
            pos = self.elephant_positions.get(label)
            position = {"x": pos[0], "y": pos[1], "z": pos[2]} if pos else None

            # Count images for this elephant
            image_count = int(np.sum(self.labels == label))

            # Get a sample image path
            mask = self.labels == label
            sample_paths = self.paths[mask]
            sample_path = str(sample_paths[0]) if len(sample_paths) > 0 else None

            results.append({
                "elephant_label": label,
                "similarity": sim,
                "image_count": image_count,
                "sample_crop_path": sample_path,
                "position": position,
                "possibly_new": sim < NEW_ELEPHANT_SIMILARITY_THRESHOLD,
            })

        return results

    def get_elephants(self) -> list[dict]:
        """Get all elephants with positions and counts."""
        elephants = []
        for label in sorted(self.centroids.keys()):
            mask = self.labels == label
            image_count = int(np.sum(mask))
            sample_paths = self.paths[mask]
            sample_path = str(sample_paths[0]) if len(sample_paths) > 0 else None
            pos = self.elephant_positions.get(label)

            elephants.append({
                "elephant_label": label,
                "image_count": image_count,
                "sample_crop_path": sample_path,
                "x": pos[0] if pos else None,
                "y": pos[1] if pos else None,
                "z": pos[2] if pos else None,
            })

        return elephants
