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
import sys
import os
from pathlib import Path
from typing import List, Tuple
import base64
import io

# Keep native deps tame before heavy imports
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from PIL import Image

# Add project root for absolute imports
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from post_processing.core.file_manager import list_npz_files

logger = logging.getLogger(__name__)



def load_dataset(track_dir: Path, last_k: int = 1) -> Tuple[np.ndarray, List[str], List[np.ndarray], List[Path]]:
    """Return embeddings, labels (stem), thumbnail frames, and video paths.

    last_k: number of trailing frames per track to include (1 means only last frame).
    """
    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    thumbs: List[np.ndarray] = []
    videos: List[Path] = []

    for npz_path in list_npz_files(track_dir):
        feats, frame_ids, video_path = load_embedding(npz_path)
        k = min(last_k, len(feats))
        start = len(feats) - k
        for idx in range(start, len(feats)):
            fid = int(frame_ids[idx]) if idx < len(frame_ids) else idx
            label = f"{npz_path.stem}_f{fid}"
            try:
                thumb = load_frame_by_index(video_path, fid)
            except Exception as exc:  # pragma: no cover - best effort thumbnail
                logger.warning("Failed to load thumbnail for %s (frame %s): %s", video_path, fid, exc)
                thumb = np.zeros((64, 64, 3), dtype=np.uint8)

            embeddings.append(feats[idx].reshape(1, -1))
            labels.append(label)
            thumbs.append(thumb)
            videos.append(video_path)

    if not embeddings:
        raise RuntimeError(f"No NPZ files found in {track_dir}")
    
    ### print len
    print("Number of embeddings loaded: ", len(embeddings))
    print("Number of labels loaded: ", len(labels))
    print("Number of thumbs loaded: ", len(thumbs))
    print("Number of videos loaded: ", len(videos))

    return np.vstack(embeddings), labels, thumbs, videos


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


def _img_to_base64(img: np.ndarray, max_side: int = 256) -> str:
    """Convert RGB numpy image to base64 PNG, resizing to a manageable thumbnail."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w)) if max(h, w) > 0 else 1.0
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = np.array(Image.fromarray(img).resize((new_w, new_h)))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_clusters_plotly(
    coords_2d: np.ndarray,
    labels: List[str],
    clusters: np.ndarray,
    thumbs: List[np.ndarray],
    video_paths: List[Path],
    save_path: Path,
) -> Path:
    """
    Interactive scatter with thumbnails in hover (as inline base64 images).
    """
    save_path = save_path.with_suffix(".html")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    images_b64 = [_img_to_base64(t) for t in thumbs]
    hover_text = []
    for lbl, vid, img64 in zip(labels, video_paths, images_b64):
        hover_text.append(
            f"<b>{lbl}</b><br>{vid}<br><img src='data:image/png;base64,{img64}' "
            f"style='width:128px;height:auto;'>"
        )

    fig = go.Figure(
        data=go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode="markers",
            marker=dict(
                color=clusters,
                colorscale="Rainbow",
                size=10,
                opacity=0.85,
                showscale=True,
                colorbar=dict(title="Cluster ID"),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Track Embedding Clusters (interactive)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
    )
    fig.write_html(save_path)
    logger.info("Saved interactive cluster plot to %s", save_path)
    return save_path


def run_clustering(track_dir: Path, n_clusters: int = 6, output: Path | None = None, last_k: int = 5) -> Path:
    embeddings, labels, thumbs, videos = load_dataset(track_dir, last_k=last_k)

    print("len(embeddings): ", len(embeddings))
    clusters = cluster_embeddings(embeddings, n_clusters=n_clusters)
    coords = reduce_to_2d(embeddings)

    # Save both HTML (interactive) and PNG (static) when output not specified
    if output:
        if output.suffix.lower() == ".html":
            plot_clusters_plotly(coords, labels, clusters, thumbs, videos, output)
        else:
            plot_clusters(coords, labels, clusters, thumbs, output)
        return output

    html_path = track_dir / "cluster_plot.html"
    png_path = track_dir / "cluster_plot.png"
    plot_clusters_plotly(coords, labels, clusters, thumbs, videos, html_path)
    plot_clusters(coords, labels, clusters, thumbs, png_path)
    return html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster track embeddings and visualize results.")
    parser.add_argument("--track-dir", type=Path, default='/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20250318/20250318_15/ZAG-ELP-CAM-019-20250318-155551-1742309751619-7/tracks', help="Directory containing track .npz and .mkv files.")
    parser.add_argument("--clusters", type=int, default=2, help="Number of clusters.")
    parser.add_argument("--last-k", type=int, default=1, help="Number of last frames per track to include (e.g., 10).")
    parser.add_argument("--output", type=Path, help="Optional output path for the plot (png/html).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_clustering(track_dir=args.track_dir, n_clusters=args.clusters, output=args.output, last_k=args.last_k)


if __name__ == "__main__":
    main()