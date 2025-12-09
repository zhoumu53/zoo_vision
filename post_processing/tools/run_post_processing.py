#!/usr/bin/env python3
"""
Post-Processing CLI

Command-line interface for running post-processing (Stage 3).
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

# Add project paths
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
POSE_REID_ROOT = PROJECT_ROOT / "training" / "PoseGuidedReID"

for path in (PROJECT_ROOT, POSE_REID_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from post_processing.tools.run_reid_feature_extraction import run_feature_extraction, load_reid
from post_processing.tools.videoloader import VideoLoader

from post_processing.core.file_manager import (
    list_track_files_all_cams,
)



def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("post_processing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-processing: Clean and smooth tracking data"
    )
    
    # Required arguments
    parser.add_argument("--date", default='20250318', help="Input JSONL file (stitched)")
    parser.add_argument("--record_root", type=Path, default='/media/dherrera/ElephantsWD/elephants/test', help="Root directory for records.")
    parser.add_argument("--config", type=Path, default='/media/mu/zoo_vision/training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml', help="ReID config file path.")
    parser.add_argument("--checkpoint", type=Path, default='/media/dherrera/ElephantsWD/reid_models/logs/swin_adamw_lr0003_bs64_softmax_triplet/net_best.pth', help="ReID checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., cuda:0 or cpu).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files.")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ## vis_cluster
    parser.add_argument("--vis_cluster", action="store_true", help="Visualize clusters after reID feature extraction.")
    parser.set_defaults(vis_cluster=True)

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    
    # Load data
    logger.info("Loading data from day %s", args.date)

    ### Load file manager and get input file path
    camera_ids=['016', '017', '018', '019']
    all_track_files = list_track_files_all_cams(
        record_root=args.record_root,
        camera_ids=camera_ids,
        date=args.date,
        logger=logger,
    )

    full_track_videos = []
    full_track_csv = []
    for cam_id in camera_ids:
        full_track_videos.extend(all_track_files[cam_id][1])
        full_track_csv.extend(all_track_files[cam_id][0])

    logger.info("Found %d track files for post-processing.", len(full_track_videos))
    reid_model = load_reid(config_path=args.config, checkpoint_path=args.checkpoint, device=args.device, mode="feature")

    for video_file in tqdm(full_track_videos, desc="Processing videos"):
        # skip if processed
        # npz_path = video_file.with_suffix(".npz")
        # if npz_path.exists() and not args.overwrite:
        #     logger.info("Skipping already processed video: %s", video_file)
        #     continue

        saved = run_feature_extraction(
            video_path=video_file,
            reid_model=reid_model,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"Saved features to {saved}")


    
    # Visualize clusters if needed
    if args.vis_cluster:
        logger.info("Visualizing clusters for extracted features.")

    
    logger.info("Post-processing complete: %s", args.output)



    
    return 0


if __name__ == "__main__":
    sys.exit(main())
