"""
Cluster track embeddings and visualize results.

Each track_dir is expected to contain pairs of:
  - <track_id>.mkv (track clip)
  - <track_id>.npz (ReID features saved by run_reid_feature_extraction)
The NPZ should have keys: features (N,D), frame_ids, avg_embedding (1,D), optional metadata.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from post_processing.tools.videoloader import VideoLoader

logger = logging.getLogger(__name__)


def list_npz_files(track_dir: Path) -> List[Path]:
    return sorted(track_dir.glob("*.npz"))


def load_embedding(npz_path: Path) -> Tuple[np.ndarray, Path]:
    """Load avg embedding from NPZ; fall back to mean over features."""
    data = np.load(npz_path, allow_pickle=True)
    if "avg_embedding" in data:
        emb = data["avg_embedding"]
    else:
        feats = data["features"]
        emb = feats.mean(axis=0, keepdims=True)
    return emb.astype(np.float32), npz_path.with_suffix(".mkv")


def load_first_frame(video_path: Path) -> np.ndarray:
    loader = VideoLoader(str(video_path), verbose=False)
    if not loader.ok() or len(loader) == 0:
        raise RuntimeError(f"Cannot read video: {video_path}")
    frame = loader[0]
    return frame


def load_dataset(track_dir: Path) -> Tuple[np.ndarray, List[str], List[np.ndarray]]:
    """Return embeddings, labels (stem), and thumbnail frames."""
    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    thumbs: List[np.ndarray] = []

    for npz_path in list_npz_files(track_dir):
        emb, video_path = load_embedding(npz_path)
        try:
            thumb = load_first_frame(video_path)
        except Exception as exc:  # pragma: no cover - best effort thumbnail
            logger.warning("Failed to load thumbnail for %s: %s", video_path, exc)
            thumb = np.zeros((64, 64, 3), dtype=np.uint8)

        embeddings.append(emb.reshape(1, -1))
        labels.append(npz_path.stem)
        thumbs.append(thumb)

    if not embeddings:
        raise RuntimeError(f"No NPZ files found in {track_dir}")

    return np.vstack(embeddings), labels, thumbs


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Run k-means clustering and return labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return model.fit_predict(embeddings)


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2D for visualization using PCA."""
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)


def plot_clusters(
    coords_2d: np.ndarray,
    labels: List[str],
    clusters: np.ndarray,
    thumbs: List[np.ndarray],
    save_path: Path,
) -> Path:
    """Scatter plot with image thumbnails for each point."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=clusters, cmap="tab20", alpha=0.4, s=30)
    ax.set_title("Track Embedding Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    for (x, y), thumb, lbl in zip(coords_2d, thumbs, labels):
        imagebox = OffsetImage(thumb, zoom=0.15)
        ab = AnnotationBbox(imagebox, (x, y), frameon=True, bboxprops=dict(edgecolor="gray", alpha=0.6))
        ax.add_artist(ab)
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Cluster ID")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info("Saved cluster plot to %s", save_path)
    return save_path


def run_clustering(track_dir: Path, n_clusters: int = 6, output: Path | None = None) -> Path:
    embeddings, labels, thumbs = load_dataset(track_dir)
    clusters = cluster_embeddings(embeddings, n_clusters=n_clusters)
    coords = reduce_to_2d(embeddings)
    save_path = output or (track_dir / "cluster_plot.png")
    return plot_clusters(coords, labels, clusters, thumbs, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster track embeddings and visualize results.")
    parser.add_argument("--track-dir", type=Path, required=True, help="Directory containing track .npz and .mkv files.")
    parser.add_argument("--clusters", type=int, default=6, help="Number of clusters.")
    parser.add_argument("--output", type=Path, help="Optional output path for the plot (png).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_clustering(track_dir=args.track_dir, n_clusters=args.clusters, output=args.output)


if __name__ == "__main__":
    main()
