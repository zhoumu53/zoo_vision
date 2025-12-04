#!/usr/bin/env python3
"""
Test script for BehaviorInference class.
Tests single image and video (MP4) inference with pre-cropped images.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from post_processing.core.behavior_inference import BehaviorInference


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_single_image(
    model_path: str,
    image_path: str,
    device: str = "cpu",
):
    """Test inference on a single image."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TEST 1: Single Image Inference")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    behavior = BehaviorInference(
        model_path=model_path,
        device=device,
    )
    
    # Load image
    logger.info(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image from {image_path}")
        return False
    
    logger.info(f"Image shape: {image.shape}")
    
    # Test predict_single
    logger.info("Running predict_single()...")
    label, confidence = behavior.predict_single(image)
    
    logger.info(f"✓ Prediction: {label} (confidence: {confidence:.4f})")
    logger.info("")
    
    return True


def test_multiple_images(
    model_path: str,
    image_paths: list[str],
    device: str = "cpu",
    batch_size: int = 8,
):
    """Test batch inference on multiple images."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TEST 2: Batch Image Inference")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    behavior = BehaviorInference(
        model_path=model_path,
        device=device,
    )
    
    # Load images
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            logger.info(f"Loaded: {img_path} {img.shape}")
        else:
            logger.warning(f"Failed to load: {img_path}")
    
    if not images:
        logger.error("No images loaded")
        return False
    
    logger.info(f"Total images loaded: {len(images)}")
    
    # Test predict
    logger.info(f"Running predict() with batch_size={batch_size}...")
    predictions = behavior.predict(images, batch_size=batch_size)
    
    logger.info(f"✓ Got {len(predictions)} predictions")
    for i, (label, conf) in enumerate(predictions):
        logger.info(f"  Image {i}: {label} (conf: {conf:.4f})")
    
    # Test get_class_distribution
    distribution = behavior.get_class_distribution(predictions)
    logger.info(f"\nClass distribution: {distribution}")
    logger.info("")
    
    return True


def test_video(
    model_path: str,
    video_path: str,
    device: str = "cpu",
    batch_size: int = 16,
    n_frames: int | None = None,
    max_frames: int = 100,
):
    """Test inference on video frames (simulating a track)."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TEST 3: Video/Track Inference")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    behavior = BehaviorInference(
        model_path=model_path,
        device=device,
    )
    
    # Open video
    logger.info(f"Opening video from {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return False
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video info: {total_video_frames} frames @ {fps:.2f} fps")
    
    # Read frames (assume they're already cropped to object of interest)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        if frame_count >= max_frames:
            logger.info(f"Reached max_frames limit ({max_frames})")
            break
    
    cap.release()
    logger.info(f"Loaded {len(frames)} frames from video")
    
    if not frames:
        logger.error("No frames loaded from video")
        return False
    
    # Test predict_tracks
    logger.info(f"Running predict_tracks() with n_frames={n_frames}, batch_size={batch_size}...")
    final_label, avg_conf, vote_dist = behavior.predict_tracks(
        frames=frames,
        batch_size=batch_size,
        n_frames=n_frames,
    )
    
    logger.info(f"✓ Track prediction complete")
    logger.info(f"  Final label: {final_label}")
    logger.info(f"  Average confidence: {avg_conf:.4f}")
    logger.info(f"  Vote distribution: {vote_dist}")
    
    # Show vote percentages
    total_votes = sum(vote_dist.values())
    logger.info(f"\nVote percentages:")
    for label, count in sorted(vote_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = 100.0 * count / total_votes
        logger.info(f"  {label}: {count}/{total_votes} ({percentage:.1f}%)")
    
    logger.info("")
    return True


def test_video_sampling(
    model_path: str,
    video_path: str,
    device: str = "cpu",
    max_frames: int = 100,
):
    """Test different sampling strategies on the same video."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TEST 4: Video Sampling Comparison")
    logger.info("=" * 80)
    
    # Load model
    behavior = BehaviorInference(
        model_path=model_path,
        device=device,
    )
    
    # Load frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        logger.error("No frames loaded")
        return False
    
    logger.info(f"Loaded {len(frames)} frames")
    
    # Test different sampling rates
    sampling_configs = [
        (None, "All frames"),
        (10, "Sample 10 frames"),
        (20, "Sample 20 frames"),
        (50, "Sample 50 frames"),
    ]
    
    results = []
    for n_frames, description in sampling_configs:
        if n_frames is not None and n_frames > len(frames):
            continue
        
        label, conf, votes = behavior.predict_tracks(
            frames=frames,
            n_frames=n_frames,
            batch_size=16,
        )
        
        results.append({
            'description': description,
            'n_frames': n_frames if n_frames else len(frames),
            'label': label,
            'confidence': conf,
            'votes': votes,
        })
        
        logger.info(f"{description:20s} -> {label:15s} (conf={conf:.4f}, votes={votes})")
    
    logger.info("")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test BehaviorInference with single image or video"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to behavior model (directory, config.json, or .ptc file)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image for testing"
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="Paths to multiple images for batch testing"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file (MP4, AVI, etc.)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Number of frames to sample from video (None = all frames)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames to read from video"
    )
    parser.add_argument(
        "--test-sampling",
        action="store_true",
        help="Run sampling comparison test"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Check inputs
    if not any([args.image, args.images, args.video]):
        logger.error("Must provide --image, --images, or --video")
        parser.print_help()
        return 1
    
    success = True
    
    # Test 1: Single image
    if args.image:
        success &= test_single_image(
            model_path=args.model,
            image_path=args.image,
            device=args.device,
        )
    
    # Test 2: Multiple images
    if args.images:
        success &= test_multiple_images(
            model_path=args.model,
            image_paths=args.images,
            device=args.device,
            batch_size=args.batch_size,
        )
    
    # Test 3: Video
    if args.video:
        success &= test_video(
            model_path=args.model,
            video_path=args.video,
            device=args.device,
            batch_size=args.batch_size,
            n_frames=args.n_frames,
            max_frames=args.max_frames,
        )
        
        # Test 4: Sampling comparison
        if args.test_sampling:
            success &= test_video_sampling(
                model_path=args.model,
                video_path=args.video,
                device=args.device,
                max_frames=args.max_frames,
            )
    
    if success:
        logger.info("=" * 80)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("=" * 80)
        logger.error("✗ SOME TESTS FAILED")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
