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

from post_processing.tools.run_reid_feature_extraction import run_feature_extraction, load_reid, load_npz_files
from post_processing.tools.videoloader import VideoLoader

from post_processing.core.file_manager import (
    list_track_files_all_cams,
    get_track_dir,
)

from post_processing.core.tracklet_manager import TrackletManager
from post_processing.core.behavior_inference import BehaviorInference


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("post_processing")


def run_behavior_on_track(
    video_path: Path,
    behavior_model: BehaviorInference,
    batch_size: int = 64,
) -> list[tuple[str, float]]:
    """Run frame-by-frame behavior classification on a track clip."""
    loader = VideoLoader(str(video_path), verbose=False)
    if not loader.ok():
        raise RuntimeError(f"Could not open video for behavior inference: {video_path}")

    results: list[tuple[str, float]] = []
    batch: list = []

    for frame in loader:
        batch.append(frame)
        if len(batch) >= batch_size:
            preds = behavior_model.predict(batch, batch_size=batch_size)
            results.extend(preds)
            batch.clear()

    if batch:
        preds = behavior_model.predict(batch, batch_size=batch_size)
        results.extend(preds)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-processing: Clean and smooth tracking data"
    )
    
    # Required arguments
    parser.add_argument("--date", default='20251129', help="Input JSONL file (stitched)")
    parser.add_argument("--record_root", type=Path, default='/media/dherrera/ElephantsWD/elephants/test', help="Root directory for records.")
    parser.add_argument("--config", type=Path, default='/media/mu/zoo_vision/training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml', help="ReID config file path.")
    parser.add_argument("--checkpoint", type=Path, default='/media/dherrera/ElephantsWD/reid_models/logs/swin_adamw_lr0003_bs64_softmax_triplet/net_best.pth', help="ReID checkpoint path.")
    parser.add_argument("--start-time", type=str, default="15:30:00", help="Start time for processing (HH:MM:SS)." \
    " Default is 15:30:00 to cover night time from PM to AM.    ")
    parser.add_argument("--end-time", type=str, default="08:30:00", help="End time for processing (HH:MM:SS)." \
    " Default is 08:30:00 next day to cover night time from PM to AM.    ")
    ### behavior model
    parser.add_argument("--behavior-model", type=Path, default="/media/mu/zoo_vision/models/sleep/vit/v2_no_validation/config.ptc", help="Behavior classification model path (optional).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., cuda:0 or cpu).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files.")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    
    # Load data
    logger.info("Loading data from day %s", args.date)

    ### TODO -- load data from PM to AM for one night (15:30 to 08:30 next day)

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

    behavior_model = None
    if args.behavior_model is not None:
        logger.info("Loading behavior model from %s", args.behavior_model)
        behavior_model = BehaviorInference(model_path=str(args.behavior_model), device=args.device, logger=logger)


    tracklet_id2identity: dict[str, str] = {}


    for track_file in tqdm(full_track_videos, desc="Processing videos"):

        track_filename = track_file.name
        logger.info("Processing track video: %s", track_filename)

        csv_path = track_file.with_suffix(".csv")

        if not csv_path.exists():
            logger.warning("Corresponding CSV file not found for video: %s", track_file)
            continue

        df_tracks = pd.read_csv(csv_path)

        # ### run behavior classification  -> save to csv (frame level)
        if behavior_model is not None:
            behavior_preds = run_behavior_on_track(
                video_path=track_file,
                behavior_model=behavior_model,
                batch_size=args.batch_size,
            )
            if len(behavior_preds) != len(df_tracks):
                logger.warning(
                    "Behavior predictions length %d does not match CSV rows %d for %s",
                    len(behavior_preds),
                    len(df_tracks),
                    track_file,
                )

            df_tracks['behavior_label'] = [pred[0] for pred in behavior_preds]
            df_tracks['behavior_conf'] = [pred[1] for pred in behavior_preds]
            # Save updated CSV
            df_tracks.to_csv(csv_path, index=False)
            logger.info("Saved behavior predictions to %s", csv_path)


        # ### ReID feature extraction - ReID -> save npz
        # skip if processed
        npz_path = track_file.with_suffix(".npz")
        if npz_path.exists() and not args.overwrite:
            logger.info("Skipping already processed video: %s", track_file)
            
            voted_identity_label = load_npz_files(npz_path).get('voted_labels', 'unknown')
            continue
        else:
            voted_identity_label = run_feature_extraction(
                video_path=track_file,
                reid_model=reid_model,
                device=args.device,
                batch_size=args.batch_size,
            )
        tracklet_id2identity[track_filename] = voted_identity_label
        print(f"Tracklet {track_filename} assigned identity: {voted_identity_label}")
    
        

    ########################## Stitching within each camera ##########################
    logger.info("Starting tracklet stitching within each camera.")
    ## TODO: change timestamp 
    start_time = "15:30:00"
    end_time = "23:59:59"

    for camera_id in camera_ids:
        track_dir = get_track_dir(
            record_root=args.record_root,
            cam_id=camera_id,
            date=args.date,
        )
    
        tracklet_manager = TrackletManager(
            track_dir=track_dir,
            camera_id=camera_id,
            num_identities=2,   ## TODO: set dynamically? with human prior?
            logger=logger,
            start_time=start_time,
            end_time=end_time,
        )
        tracklet_manager.load_tracklets_for_camera()
        print(f"Loaded {len(tracklet_manager.tracklets)} tracklets for camera {tracklet_manager.camera_id}")

        # Test the new bidirectional gallery-based stitching
        print("\n" + "="*80)
        print("Testing BIDIRECTIONAL GALLERY-based stitching (recommended for elephants)")
        print("="*80)
        tracklet_manager.stitch_tracklets_bidirectional(
            max_gap_frames=600,
            local_sim_th=0.5,
            gallery_sim_th=0.45,
            head_k=5,
            tail_k=5,
            gallery_k=10,
            w_local=0.6,
            w_gallery=0.4,
        )
        tracklet_manager.save_stitched_tracklets()

        # Visualize results
        from post_processing.tools.visualization import visualize_stitched_tracks_pairs, plot_stitched_ids_on_original_frames
        visualize_stitched_tracks_pairs(
            tracklet_manager.tracklets,
            output_dir=Path("/media/mu/zoo_vision/post_processing/scrips/out"),
            camera_id=camera_id,
            head_k=3,
            tail_k=3,
            max_chains=None,
            max_tracklets_per_chain=None,
            cell_h=256,
            cell_w=256,
            logger_=logger,
        )
        
        from tests.validate_stitching_timeline import validate_stitched_timelines
        validate_stitched_timelines(
            tracklet_manager.tracklets,
            camera_id=camera_id,
            logger_=logger,
        )

        print("validate_stitched_timelines", validate_stitched_timelines)

        ### TODO: visualization - validate_stitched_ids


    
    ### TODO - cross-camera calibration and stitching
    


if __name__ == "__main__":
    sys.exit(main())
