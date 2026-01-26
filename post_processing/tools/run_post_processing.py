#!/usr/bin/env python3
"""
Post-Processing CLI

Command-line interface for running post-processing (Stage 3).
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

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

from post_processing.tools.run_reid_feature_extraction import (
    load_reid,
)
from post_processing.tools.videoloader import VideoLoader
from post_processing.core.id_matching_cross_cam import cross_camera_id_matching

from post_processing.core.file_manager import (
    list_track_files_all_cams,
    get_track_dir,
)

from post_processing.analysis.analyze_lateral_sleeping import (
    load_valid_tracks,
)

from post_processing.core.tracklet_manager import TrackletManager, track_csv2identity
from post_processing.tools.utils import setup_logger, load_gallery_features
from datetime import datetime, timedelta

def update_csv_from_df(df: pd.DataFrame) -> None:
    
    track_files = df['track_filename'].unique().tolist()
    
    for track_file in track_files:
        _df = df[ df['track_filename'] == track_file ]
        df_original = pd.read_csv(track_file)
        original_columns = df_original.columns.tolist()
        _df = _df[original_columns]
        _df.to_csv(track_file, index=False)


def run_stitching(
    track_dirs: list[Path],
    camera_id: str,
    start_time: str,
    end_time: str,
    gallery_features,
    gallery_labels,
    logger: logging.Logger,
    height: int = 1080,
    width: int = 1920,
    num_identities: int = 2,
    output_dir: Path = Path("/media/ElephantsWD/elephants/xmas/demo")
):
    """Stitch tracklets for a single camera/time window and assign IDs."""
    logger.info(
        "Stitching tracklets for camera %s between %s and %s",
        camera_id,
        start_time,
        end_time,
    )
    
    # TODO - check this
    # if known_labels is not None:
    #     logger.info("Using known labels for gallery filtering: %s", known_labels)
    #     valid_indices = (
    #         [i for i, label in enumerate(gallery_labels) if label in known_labels]
    #         if known_labels is not None
    #         else list(range(len(gallery_labels)))
    #     )
    #     g_feats = gallery_features[valid_indices]
    #     g_labels = [gallery_labels[i] for i in valid_indices]

    tracklet_manager = TrackletManager(
        track_dirs=track_dirs,
        camera_id=camera_id,
        num_identities=num_identities,  # TODO: set dynamically? with human prior?
        logger=logger,
        start_time=start_time,
        end_time=end_time,
        height=height,
        width=width,
    )
    
    tracklet_manager.load_tracklets_for_camera(track_dirs=track_dirs, camera_id=camera_id)
        
    print(f"Loaded {len(tracklet_manager.tracklets)} tracklets for camera {tracklet_manager.camera_id}")

    print("\n" + "=" * 80)
    print("Testing BIDIRECTIONAL GALLERY-based stitching")
    print("=" * 80)
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
    
    tracklet_manager.get_stitched_tracklets(
                                             gallery_features=gallery_features,
                                             gallery_labels=gallery_labels
                                             )
    
    tracklet_manager.save_stitched_tracklets(
        tracklet_manager.final_stitched_map,
        output_dir= output_dir
    )

    csv2identity = track_csv2identity(tracklet_manager.final_stitched_map)

    ## save identity label to csv
    for track_csv, identity_label in csv2identity.items():
        df = pd.read_csv(track_csv)
        df['identity_label'] = identity_label
        df.to_csv(track_csv, index=False)
        logger.info("Saved identity label %s to %s", identity_label, track_csv)

    return tracklet_manager, tracklet_manager.final_stitched_map

def id_matching():
    """Placeholder for id matching."""
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-processing: Clean and smooth tracking data"
    )
    
    # Required arguments
    parser.add_argument("--date", default='20251129', help="Input JSONL file (stitched)")
    parser.add_argument("--record-root", type=Path, default='/media/ElephantsWD/elephants/test', help="Root directory for records.")
    parser.add_argument("--config", type=Path, default='/media/mu/zoo_vision/training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml', help="ReID config file path.")
    parser.add_argument("--checkpoint", type=Path, default='/media/ElephantsWD/reid_models/logs/swin_adamw_lr0003_bs64_softmax_triplet/net_best.pth', help="ReID checkpoint path.")
    parser.add_argument("--start-time", type=str, default="15:30:00", help="Start time for processing (HH:MM:SS)." \
    " Default is 15:30:00 to cover night time from PM to AM.")
    parser.add_argument("--end-time", type=str, default="08:00:00", help="End time for processing (HH:MM:SS). Default is 08:00:00 next day to cover night time from PM to AM.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., cuda:0 or cpu).")
    parser.add_argument("--gallery-path", type=Path, default=None, help="Path to gallery features NPZ file.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--height", type=int, default=1080, help="Height of the video frames.")
    parser.add_argument("--width", type=int, default=1920, help="Width of the video frames.")
    parser.add_argument("--output_dir", type=str, default='/media/ElephantsWD/elephants/xmas/demo', 
                        help="Output directory for post-processed results.")

    return parser.parse_args()



def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    
    # Load data
    logger.info("Loading data from day %s", args.date)
    
    # Load file manager and get input file path
    camera_ids = ["016", "017", "018", "019"]
    # camera_ids = [ "017", "018"] # TESTING
    camera_ids = [ "016", "019"] # TESTING
    output_dir= Path(args.output_dir)

    reid_model = load_reid(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        mode="feature",
    )

    ### TODO -- load gallery features from past days
    gallery_features, gallery_labels = load_gallery_features(
        reid_model=reid_model,
        checkpoint_path=args.checkpoint,
        provided_path=args.gallery_path,
        logger=logger,
    )

    date = args.date
    ### formatting - if '2023-11-29' -> '20231129'
    if '-' in date:
        date = date.replace('-', '')
    dates = [date]
    
    start_time = args.start_time
    end_time = args.end_time
    if end_time < start_time:
        # end time is on the next day
        date_obj = datetime.strptime(date, "%Y%m%d")
        next_date_obj = date_obj + timedelta(days=1)
        date = next_date_obj.strftime("%Y%m%d")
        dates.append(date)
            
    start_datetime = f"{dates[0]} {start_time}"
    end_datetime = f"{dates[-1]} {end_time}"
    
    # stitching per camera

    final_camera_stitched_maps = {}

    for camera_id in camera_ids:
        
        num_identities = 2  # TODO - set dynamically based on prior knowledge
                    
        track_dirs = [get_track_dir(
            record_root=args.record_root,
            cam_id=camera_id,
            date=date,
        ) for date in dates]
        
        ## TODO - UPDATE THIS
        if not all([track_dir.exists() for track_dir in track_dirs]):
            logger.warning("Track directories for camera %s on dates %s do not all exist. Skipping.", camera_id, dates)
            continue
        
        # print(f"Processing camera {camera_id} from {start_datetime} to {end_datetime}")
    
        tracklet_manager, final_stitched_tracklets = run_stitching(
            track_dirs=track_dirs,
            camera_id=camera_id,
            start_time=start_datetime,
            end_time=end_datetime,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
            height=args.height,
            width=args.width,
            num_identities=num_identities,
            logger=logger,
            output_dir= Path(output_dir)
        )

        final_camera_stitched_maps[camera_id] = (tracklet_manager, final_stitched_tracklets)


    ### TODO - cross-camera calibration and stitching - OPTIMIZE THE CODE

    cam_pairs = [
        ("016", "019"),
        # ("018", "017"),
    ]

    print("\n" + "=" * 80 + "\n")
    print("Running CROSS-CAMERA ID MATCHING")
    print("\n" + "=" * 80 + "\n")

    for cam1_id, cam2_id in cam_pairs:
        updated_cam2_data = cross_camera_id_matching(final_camera_stitched_maps[cam1_id][1],
                                                     final_camera_stitched_maps[cam2_id][1],
                                                     window_hours=1,
                                                     time_window_seconds=0.5,
                                                     distance_threshold=2.0,
                                                     downsample_seconds=1.0)
        
        ### TODO -- clear the code
        tracklet_manager, _ = final_camera_stitched_maps[cam2_id]
        # tracklet_manager.save_stitched_tracklets(
        #     updated_cam2_data,
        #     output_dir= output_dir
        # )
        
        csv2identity = track_csv2identity(updated_cam2_data)
        ## save identity label to csv
        for track_csv, identity_label in csv2identity.items():
            df = pd.read_csv(track_csv)
            df['identity_label'] = identity_label
            df.to_csv(track_csv, index=False)
            logger.info("Saved identity label %s to %s", identity_label, track_csv)
        
        ### print out the original 
        # print("updated_cam2_data", updated_cam2_data)
    
    ### TODO
    # print("\n" + "=" * 80 + "\n")
    # print("Running CROSS-CAMERA BEHAVIOR MATCHING")
    # print("\n" + "=" * 80 + "\n")
    
    # for cam1_id, cam2_id in cam_pairs:
    #     # cross-camera behavior matching. -- update behavior labels based on two cameras
        
    #     df_results = load_valid_tracks(
    #         record_root=args.record_root,
    #         start_datetime=start_datetime,
    #         end_datetime=end_datetime,
    #         camera_ids=(cam1_id, cam2_id),
    #     )
    #     # TODO - smooth IDs

    #     df_results = smooth_behavior_cross_cameras(
    #         df_results)
        
    #     # update csv
    #     update_csv_from_df(df_results)




if __name__ == "__main__":
    sys.exit(main())
