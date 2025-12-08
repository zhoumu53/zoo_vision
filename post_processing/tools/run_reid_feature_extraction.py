import argparse
import logging
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project paths
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
POSE_REID_ROOT = PROJECT_ROOT / "training" / "PoseGuidedReID"

for path in (PROJECT_ROOT, POSE_REID_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from post_processing.core.reid_inference import ReIDInference
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
) -> Tuple[np.ndarray, List[int], np.ndarray:]:
    """
    Extract ReID features for every frame in a video.

    Returns:
        features: (N, D) numpy array
        frame_ids: list of frame indices corresponding to each feature
    """
    loader = VideoLoader(str(video_path), verbose=True)
    if not loader.ok():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_ids: List[int] = []
    feats: List[torch.Tensor] = []
    device = reid_model.device
    transform = reid_model.transform

    batch_frames: List[np.ndarray] = []
    batch_indices: List[int] = []

    for idx, frame in enumerate(loader):
        batch_frames.append(frame)
        batch_indices.append(idx)

        if len(batch_frames) == batch_size:
            batch = _prep_batch(batch_frames, transform, device)
            feat = reid_model.extract_features(batch)
            feats.append(feat.cpu())
            frame_ids.extend(batch_indices)
            batch_frames.clear()
            batch_indices.clear()

    if batch_frames:
        batch = _prep_batch(batch_frames, transform, device)
        feat = reid_model.extract_features(batch)
        feats.append(feat.cpu())
        frame_ids.extend(batch_indices)

    if not feats:
        return np.empty((0, reid_model.feature_dim), dtype=np.float32), []

    features = torch.cat(feats, dim=0)

    # average embedding
    avg_embedding = torch.mean(features, dim=0, keepdim=True)

    return features.numpy(), frame_ids, avg_embedding.numpy()


def save_features_npz(
    features: np.ndarray,
    frame_ids: List[int],
    avg_embedding: np.ndarray,
    save_path: Path,
    metadata: dict | None = None,
) -> None:
    """Save features and frame ids to compressed NPZ."""
    save_path = save_path.with_suffix(".npz")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "features": features,
        "frame_ids": np.array(frame_ids, dtype=np.int32),
        "avg_embedding": avg_embedding
    }
    if metadata:
        payload["metadata"] = metadata
    np.savez_compressed(save_path, **payload)
    logger.info("Saved %d features to %s", len(frame_ids), save_path)


def run_feature_extraction(
    video_path: Path,
    reid_model: ReIDInference,
    device: str = "cpu",
    batch_size: int = 32
) -> Path:
    """High-level helper: load model, extract features, and save alongside the video."""
    features, frame_ids, avg_embedding = extract_reid_features_from_video(
        video_path=video_path,
        reid_model=reid_model,
        batch_size=batch_size
    )
    save_path = video_path.with_suffix(".npz")
    save_features_npz(
        features=features,
        frame_ids=frame_ids,
        avg_embedding=avg_embedding,
        save_path=save_path,
        metadata={
            "video": str(video_path),
            "device": device,
        },
    )
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ReID features for a video/track and save to NPZ.")
    parser.add_argument("--video", type=Path, default='/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20250318/20250318_15/ZAG-ELP-CAM-016-20250318-155821-1742309901684-7/tracks', help="Path or directory to video or track clip (e.g., .mkv).")
    parser.add_argument("--config", type=Path, default='/media/mu/zoo_vision/training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml', help="ReID config file path.")
    parser.add_argument("--checkpoint", type=Path, default='/media/dherrera/ElephantsWD/reid_models/logs/swin_adamw_lr0003_bs64_softmax_triplet/net_best.pth', help="ReID checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., cuda:0 or cpu).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ### check if arg.video exists, if not, raise error
    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")

    reid = load_reid(config_path=args.config, checkpoint_path=args.checkpoint, device=args.device, mode="feature")

    if args.video.is_dir():
        video_files = sorted(args.video.glob("*.mkv"))
        if not video_files:
            raise FileNotFoundError(f"No .mkv files found in directory: {args.video}")
        for video_file in tqdm(video_files, desc="Processing videos"):
            saved = run_feature_extraction(
                video_path=video_file,
                reid=reid,
                device=args.device,
                batch_size=args.batch_size,
            )
            print(f"Saved features to {saved}")
    else:
        
        saved = run_feature_extraction(
            video_path=args.video,
            reid=reid,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"Saved features to {saved}")


if __name__ == "__main__":
    main()