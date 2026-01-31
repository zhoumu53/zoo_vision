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
    get_track_dir,
)

from post_processing.tools.utils import normalize_time_string


from post_processing.core.tracklet_manager import TrackletManager, track_csv2identity
from post_processing.tools.utils import setup_logger, load_gallery_features
from datetime import datetime, timedelta
from post_processing.core.temporal_smooth import smooth_behavior_cross_cameras

def update_csv_from_df(df: pd.DataFrame) -> None:
    
    track_files = df['track_csv_path'].unique().tolist()
    
    for track_file in track_files:
        _df = df[ df['track_csv_path'] == track_file ]
        df_original = pd.read_csv(track_file)
        original_columns = df_original.columns.tolist()
        _df = _df[original_columns]
        _df.to_csv(track_file, index=False)


def get_stitched_data(
    track_dirs: list[Path],
    tracklet_manager: TrackletManager,
    camera_id: str,
    output_dir: Path = Path("/media/ElephantsWD/elephants/xmas/demo"),
    run_stitching:  bool = True,
):
    """Stitch tracklets for a single camera/time window and assign IDs."""

    if not run_stitching:
        return  tracklet_manager.load_stitched_tracklets_from_dir(
            dir= output_dir
        )
    

    tracklet_manager.load_tracklets_for_camera(track_dirs=track_dirs, 
                                               camera_id=camera_id)
        
    print(f"Loaded {len(tracklet_manager.tracklets)} tracklets for camera {tracklet_manager.camera_id}")
    
    print("\n" + "=" * 80)
    print("Testing BIDIRECTIONAL GALLERY-based stitching")
    print("=" * 80)
    tracklet_manager.stitch_tracklets_bidirectional(
        max_gap_frames=600,
        local_sim_th=0.5,
        gallery_sim_th=0.45,
        head_k=25,
        tail_k= 25,
        gallery_k=10,
        w_local=0.6,
        w_gallery=0.4,
    )
    
    tracklet_manager.get_stitched_tracklets()
    
    save_path = tracklet_manager.save_stitched_tracklets(
        tracklet_manager.tracklets,
        output_dir= output_dir
    )

    return tracklet_manager.tracklets, save_path



def merge_csv_tracklets(record_root: Path, 
                        start_datetime: pd.Timestamp ,  ## date time
                        end_datetime: pd.Timestamp,
                        camera_ids: list[str]) -> pd.DataFrame:

    from post_processing.tools.utils import (load_valid_tracks, 
                                                load_identity_labels_from_json)
    
    df_behavior = load_valid_tracks(
        record_root=record_root,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        camera_ids=camera_ids,
    )

    df_label_predictions = load_identity_labels_from_json(
        record_root=Path(record_root),
        camera_ids=camera_ids,
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    # merge identity labels into df_behavior, on track_filename
    df_results = df_behavior.merge(
        df_label_predictions[['track_filename', 'stitched_label', 'voted_track_label', 'smoothed_label', 'identity_label']],
        on='track_filename',
        how='left'
    )
    
    return df_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-processing: Clean and smooth tracking data"
    )
    
    # Required arguments
    parser.add_argument("--date", default='20251129', help="Input JSONL file (stitched)")
    parser.add_argument("--record-root", type=Path, default='/media/ElephantsWD/elephants/test', help="Root directory for records.")
    parser.add_argument("--config", type=Path, default='/media/mu/zoo_vision/training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml', help="ReID config file path.")
    parser.add_argument("--checkpoint", type=Path, default='/media/ElephantsWD/elephants/reid_models/swin_adamw_lr0003_bs128_softmax_triplet_Fulldata/net_best.pth', help="ReID checkpoint path.")
    parser.add_argument(
        "--start_timestamp", type=str, help="Start timestamp for processing, e.g., '180000'", default='180000'
    )
    parser.add_argument(
        "--end_timestamp", type=str, help="End timestamp for processing, e.g., '080000'", default='080000'
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., cuda:0 or cpu).")
    parser.add_argument("--gallery-path", type=Path, default=None, help="Path to gallery features NPZ file.")
    parser.add_argument("--camera-ids", type=str, nargs="+", default=["016", "017", "018", "019"], help="List of camera IDs to process.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--height", type=int, default=1080, help="Height of the video frames.")
    parser.add_argument("--width", type=int, default=1920, help="Width of the video frames.")
    parser.add_argument("--cam1619-individuals", type=list, default=[], nargs="*", help="List of known individual IDs for the room.")
    parser.add_argument("--cam1718-individuals", type=list, default=[], nargs="*", help="List of known individual IDs for the room.")
    parser.add_argument("--output_dir", type=str, default='/media/ElephantsWD/elephants/xmas/demo', 
                        help="Output directory for post-processed results.")
    parser.add_argument("--run-stitching", action="store_true", help="Whether to run tracklet stitching.")
    parser.set_defaults(run_stitching=False)
    
    parser.add_argument("--cross-camera-matching", action="store_true", help="Whether to run cross-camera ID matching.")
    parser.set_defaults(cross_camera_matching=True)

    return parser.parse_args()



def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    
    # Load data
    logger.info("Loading data from day %s", args.date)
    
    # Load file manager and get input file path
    camera_ids = args.camera_ids
    output_dir= Path(args.output_dir)
    cam1619_individuals = args.cam1619_individuals
    cam1718_individuals = args.cam1718_individuals

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
    

    # Normalize and convert timestamp strings to pd.Timestamp
    start_timestamp = normalize_time_string(args.start_timestamp)
    end_timestamp = normalize_time_string(args.end_timestamp)
    
    start_timestamp = pd.to_datetime(start_timestamp, format="%H%M%S")
    end_timestamp = pd.to_datetime(end_timestamp, format="%H%M%S")
    

    if end_timestamp < start_timestamp:
        # end time is on the next day
        date_obj = datetime.strptime(date, "%Y%m%d")
        next_date_obj = date_obj + timedelta(days=1)
        date = next_date_obj.strftime("%Y%m%d")
        dates.append(date)
            

    start_datetime = f"{dates[0]} {start_timestamp.strftime('%H:%M:%S')}"
    end_datetime = f"{dates[-1]} {end_timestamp.strftime('%H:%M:%S')}"

    # stitching per camera

    final_camera_tracklets = {}

    for camera_id in camera_ids:

        known_individuals = None
        if camera_id in ['016', '019']:
            known_individuals = cam1619_individuals
        elif camera_id in ['017', '018']:
            known_individuals = cam1718_individuals

        print("\n" + "=" * 80)
        print("record root:", args.record_root)
                            
        track_dirs = [get_track_dir(
            record_root=args.record_root,
            cam_id=camera_id,
            date=date,
        ) for date in dates]
        
        if not all([track_dir.exists() for track_dir in track_dirs]):
            logger.warning("Track directories for camera %s on dates %s do not all exist. Skipping.", camera_id, dates)
            continue
        
        tracklet_manager = TrackletManager(
            track_dirs=track_dirs,
            camera_id=camera_id,
            num_identities=len(known_individuals) if known_individuals else 2,  # max identities
            logger=logger,
            start_time=start_datetime,
            end_time=end_datetime,
            height=args.height,
            width=args.width,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
            known_labels=known_individuals,
        )
        
        tracklet_results, save_path = get_stitched_data(
                track_dirs=track_dirs,
                tracklet_manager=tracklet_manager,
                camera_id=camera_id,
                output_dir= Path(output_dir),
                run_stitching=args.run_stitching,
            )

        final_camera_tracklets[camera_id] = (tracklet_results, save_path)


    ### TODO - cross-camera calibration and stitching - OPTIMIZE THE CODE - not ready yet

    
    ## TODO - optimize the cross-camera id matching code
    cam_pairs = [
        ['016', '019'],
        ['018', '017']]
    
    print("\n" + "=" * 80)
    print("Starting CROSS-CAMERA ID MATCHING")
    print("=" * 80)
    for camera_ids in cam_pairs:
        # cross-camera behavior matching. -- update behavior labels based on two cameras

        try:
            df_results = merge_csv_tracklets(
                record_root= args.record_root,
                start_datetime= pd.Timestamp(start_datetime),
                end_datetime= pd.Timestamp(end_datetime),
                camera_ids= camera_ids
            )


            df_results = smooth_behavior_cross_cameras(df_results, id_col='voted_track_label',)
            
            # update csv
            update_csv_from_df(df_results)
        except Exception as e:
            logger.error("Error during cross-camera ID matching for cameras %s: %s", camera_ids, str(e))
            continue




if __name__ == "__main__":
    sys.exit(main())
