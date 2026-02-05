import argparse
import logging
from pathlib import Path
import sys
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import os

# Add project paths
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
POSE_REID_ROOT = PROJECT_ROOT / "training" / "PoseGuidedReID"

for path in (PROJECT_ROOT, POSE_REID_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from post_processing.core.reid_inference import ReIDInference, match_to_gallery
from post_processing.tools.videoloader import VideoLoader

logger = logging.getLogger(__name__)


def load_reid(
    config_path: Path,
    checkpoint_path: Path,
    device: str = "cpu",
    mode: str = "feature",
) -> ReIDInference:
    """Initialize ReIDInference with provided config and checkpoint."""
    return ReIDInference(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=device,
        mode=mode,
    )


def load_npz_files(npz_path: Path) -> dict:
    """Load NPZ file and return its contents as a dictionary."""
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}

def _prep_batch(frames: List[np.ndarray], transform, device: torch.device) -> torch.Tensor:
    """Convert a list of RGB numpy frames to a batch tensor."""
    pil_frames = [Image.fromarray(f) for f in frames]
    tensor_frames = [transform(img) for img in pil_frames]
    batch = torch.stack(tensor_frames).to(device)
    return batch


def extract_reid_features_from_video(
    video_path: Path,
    reid_model: ReIDInference,
    batch_size: int = 32,
    skip_frames: bool = True,
    fps : int = 25,
    K = 25,
    frame_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    Extract ReID features for every frame in a video.

    Returns:
        features: (N, D) numpy array
        indices: list of frame indices corresponding to each feature
    """
    loader = VideoLoader(str(video_path), verbose=True)
    if not loader.ok():
        raise RuntimeError(f"Could not open video: {video_path}")

    feats: List[torch.Tensor] = []
    device = reid_model.device
    transform = reid_model.transform

    batch_frames: List[np.ndarray] = []
    batch_indices: List[int] = []

    ### if the video is too long, skip frames  -- save head / tail
    total_frames = len(loader)
    #### sample frames if frame_ids is None
    if frame_ids is None:
        indices = list(range(total_frames))
    else:
        indices = frame_ids

    # Keep ~1 frame every 4 second (fps=25)
    if len(indices) > 1000 and skip_frames:
        step = max(1, int(fps) * 4)
        logger.info(
            f"Video has {total_frames} frames. Sampling 0.25 fps with step={step} frames "
            f"(~{total_frames/step:.1f} samples) + head/tail anchors."
        )
        indices = list(range(0, total_frames, step))
        
    # Always include K head/tail frames (guard for short videos) -- for stitching continuity
    head = list(range(0, min(K, total_frames)))
    tail_start = max(0, total_frames - K)
    tail = list(range(tail_start, total_frames))
    indices = sorted(set(indices + head + tail))

    for idx, frame in enumerate(loader):
        if idx not in indices:
            continue
        batch_frames.append(frame)

        if len(batch_frames) == batch_size:
            batch = _prep_batch(batch_frames, transform, device)
            feat = reid_model.extract_features(batch)
            feats.append(feat.cpu())
            batch_frames.clear()

    if batch_frames:
        batch = _prep_batch(batch_frames, transform, device)
        feat = reid_model.extract_features(batch)
        feats.append(feat.cpu())

    if not feats:
        return np.empty((0, reid_model.feature_dim), dtype=np.float32), [], np.empty((0, reid_model.feature_dim), dtype=np.float32)

    features = torch.cat(feats, dim=0)

    # average embedding
    avg_embedding = torch.mean(features, dim=0, keepdim=True)

    return features.numpy(), indices, avg_embedding.numpy()


def save_features_npz(
    features: np.ndarray,
    frame_ids: List[int],
    avg_embedding: np.ndarray,
    matched_labels: List[str] = [],
    avg_matched_labels: List[str] = [],
    voted_labels: List[str] = [],
    save_path: Path = None,
    metadata: dict | None = None,
) -> None:
    """
    Save ReID features and associated data to a compressed NPZ file.
    Args:
        features: (N, D) numpy array of ReID features.
        frame_ids: List of frame indices corresponding to each feature.
        avg_embedding: (1, D) numpy array of average embedding.
        matched_labels: List of matched labels per frame.
        avg_matched_labels: List of matched labels for average embedding.
        voted_labels: List of voted labels per frame.
        save_path: Path to save the NPZ file.
        metadata: Optional dictionary of metadata to include.
    """
    save_path = save_path.with_suffix(".npz")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "features": features,
        "frame_ids": np.array(frame_ids, dtype=np.int32),
        "avg_embedding": avg_embedding,
        "matched_labels": matched_labels,
        "avg_matched_labels": avg_matched_labels,
        "voted_labels": voted_labels,
    }
    if metadata:
        payload["metadata"] = metadata
    np.savez_compressed(save_path, **payload)
    logger.info("Saved %d features to %s", len(frame_ids), save_path)


def vote_matched_labels(matched_labels: List[List[str]]) -> List[str]:
    """Given matched labels per frame, return the most common label."""
    voted_labels = []
    for labels in matched_labels:
        if labels:
            most_common = labels[0]
            voted_labels.append(most_common)
        else:
            voted_labels.append("unknown")
    return voted_labels


def id_feature_extraction(
    video_path: Path,
    reid_model: ReIDInference,
    gallery_features: np.ndarray,
    gallery_labels: List[str],
    device: str = "cpu",
    batch_size: int = 32,
    frame_ids: Optional[List[int]] = None,
) -> str:
    """High-level helper: load model, extract features, and save alongside the video."""
    
    matched_labels=[]
    avg_matched_labels=[]
    voted_label = 'invalid'

    save_path = video_path.with_suffix(".npz")

    ### TODO -- voting with good frames - but keep features
    features, frame_ids, avg_embedding = extract_reid_features_from_video(
            video_path=video_path,
            reid_model=reid_model,
            batch_size=batch_size,
            frame_ids=frame_ids,
        )

    matched_labels = match_to_gallery(features, gallery_features, gallery_labels=gallery_labels)[-1]
    avg_matched_labels = match_to_gallery(avg_embedding, gallery_features, gallery_labels=gallery_labels)[-1]
    
    voted_labels = vote_matched_labels(matched_labels)
    voted_label = max(set(voted_labels), key=voted_labels.count)

    # print("---matched_labels:", matched_labels)
    # print("---avg_matched_labels:", avg_matched_labels)
    # print("---voted_label:", voted_label)

    ### if no frame_ids, return 'invalid' label, but still extract features -> for stitching
    if len(frame_ids) == 0:
        voted_label = 'invalid'
        
    save_features_npz(
        features=features,
        frame_ids=frame_ids,
        avg_embedding=avg_embedding,
        matched_labels=matched_labels,
        avg_matched_labels=avg_matched_labels,
        voted_labels=voted_label,
        save_path=save_path,
        metadata={
            "video": str(video_path),
            "device": device,
        },
    )

    return voted_label

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ReID features for a video/track and save to NPZ.")
    parser.add_argument("--video", type=Path, default='/media/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20251129/20251129_01/ZAG-ELP-CAM-016-20251129-011949-1764375589549-7/tracks', help="Path or directory to video or track clip (e.g., .mkv).")
    parser.add_argument("--config", type=Path, default='/media/mu/zoo_vision/training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml', help="ReID config file path.")
    parser.add_argument("--checkpoint", type=Path, default='/media/ElephantsWD/reid_models/logs/swin_adamw_lr0003_bs64_softmax_triplet/net_best.pth', help="ReID checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., cuda:0 or cpu).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--gallery-path", type=Path, default=None, help="Optional gallery features NPZ path for matching.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ### check if arg.video exists, if not, raise error
    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")

    reid = load_reid(config_path=args.config, checkpoint_path=args.checkpoint, device=args.device, mode="feature")

    gallery_path = args.gallery_path
    if gallery_path is None: 
        gallery_path = args.checkpoint.parent / "pred_features" / "train_iid" / "pytorch_result_e.npz"
        print(f"No gallery path provided. Using default gallery path: {gallery_path}")

    if args.video.is_dir():
        video_files = sorted(args.video.glob("*.mkv"))
        if not video_files:
            raise FileNotFoundError(f"No .mkv files found in directory: {args.video}")
        for video_file in tqdm(video_files, desc="Processing videos"):
            voted_identity_label = id_feature_extraction(
                video_path=video_file,
                reid_model=reid,
                device=args.device,
                batch_size=args.batch_size,
                gallery_path=gallery_path,
            )
            print(f"Saved features to {video_file.with_suffix('.npz')}, voted label: {voted_identity_label}")
    else:
        
        voted_identity_label = id_feature_extraction(
            video_path=args.video,
            reid_model=reid,
            device=args.device,
            batch_size=args.batch_size,
            gallery_path=args.gallery_path,
        )
        print(f"Saved features to {args.video.with_suffix('.npz')}, voted label: {voted_identity_label}")


if __name__ == "__main__":
    main()