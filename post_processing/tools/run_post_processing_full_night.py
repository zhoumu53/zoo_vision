#!/usr/bin/env python3
"""
Post-Processing CLI

Command-line interface for running post-processing (Stage 3).
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional



# Add project paths
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

for path in (PROJECT_ROOT,):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
        
import pandas as pd
import numpy as np
from post_processing.utils import *
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

from post_processing.core.file_manager import (
    get_track_dir,
)

from post_processing.tools.utils import (
    normalize_time_string,
    load_tracklet_json_for_camera,
    update_tracklet_json_identity_labels,
)


from post_processing.core.tracklet_manager import TrackletManager, track_csv2identity
from post_processing.tools.utils import setup_logger, load_gallery_features
from datetime import datetime, timedelta
from post_processing.core.temporal_smooth import *
from post_processing.core.cross_cam_matching import (
    run_cross_camera_matching_v2,
    summarize_cross_cam_match,
)

def update_csv_from_df(df: pd.DataFrame) -> None:
    
    track_files = df['track_csv_path'].unique().tolist()
    
    for track_file in track_files:
        _df = df[ df['track_csv_path'] == track_file ]
        behavior_csv = track_file.replace('.csv', '_behavior.csv')
        df_behavior = pd.read_csv(behavior_csv)
        original_columns = df_behavior.columns.tolist()
        original_columns.append('timestamp') if 'timestamp' not in original_columns else original_columns
        _df = _df[original_columns]
        _df.to_csv(behavior_csv.replace('.csv', '_smoothed.csv'), index=False)
    print("Behavior labels are smoothed.")


def get_stitched_data(
    track_dirs: list[Path],
    tracklet_manager: TrackletManager,
    camera_id: str,
    output_dir: Path = Path("/media/ElephantsWD/elephants/xmas/demo"),
    run_stitching:  bool = True,
):
    """Stitch tracklets for a single camera/time window and assign IDs."""

    if not run_stitching:
        return  load_stitched_tracklets_from_dir(
            dir= output_dir,
            start_time=tracklet_manager.start_time,
            end_time=tracklet_manager.end_time,
            camera_id=camera_id,
            logger=tracklet_manager.logger,
        )
    

    tracklet_manager.load_tracklets_for_camera(track_dirs=track_dirs, 
                                               camera_id=camera_id,)

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
    
    # tracklet_manager.assign_stitched_label()
    
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
    print("df_label_predictions", df_label_predictions.columns.tolist())
    # merge identity labels into df_behavior, on track_filename
    merge_cols = ['track_filename', 'stitched_id', 'stitched_label', 'voted_track_label', 'smoothed_label', 'identity_label']
    available_cols = [c for c in merge_cols if c in df_label_predictions.columns]
    df_results = df_behavior.merge(
        df_label_predictions[available_cols],
        on='track_filename',
        how='left'
    )
    # Ensure stitched_id exists — synthesize from track_filename if missing
    if 'stitched_id' not in df_results.columns:
        # Assign a unique stitched_id per (camera_id, track_filename)
        df_results['stitched_id'] = df_results.groupby(
            ['camera_id', 'track_filename']
        ).ngroup()
    print("df_results", df_results.columns.tolist())
    
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
    
    parser.add_argument("--cross-camera-matching", action="store_true", help="Whether to run cross-camera ID matching.")
    # parser.set_defaults(cross_camera_matching=True)

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
    
    

    reid_model = load_reid(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        mode="feature",
    )

    gallery_features, gallery_labels = load_gallery_features(
        reid_model=reid_model,
        checkpoint_path=args.checkpoint,
        provided_path=args.gallery_path,
        logger=logger,
    )

    final_camera_tracklets = {}
    vote_known_individuals_dict = {}
    
    for camera_id in camera_ids:

        known_individuals = None
        if camera_id in ['016', '019']:
            known_individuals = cam1619_individuals
        elif camera_id in ['017', '018']:
            known_individuals = cam1718_individuals
        
        ### TODO - check cam1619_individuals, cam1718_individuals, why [[]] appears?
        known_individuals = None if len(known_individuals) == 0 or known_individuals==[[]] else known_individuals
        ### if known_individuals is empty --> 1) check from database, 2) check from data between 15h30-17h30 (daytime - better visibility - better ID)
        if known_individuals is None:
            track_dir = get_track_dir(
                record_root=args.record_root,
                cam_id=camera_id,
                date=dates[0],   ## from the first day of the night
            )
            known_individuals = vote_known_individuals(
                track_dir=track_dir,
                date=dates[0],
                start_time='153000',
                end_time='173000',
                gallery_features=gallery_features,
                gallery_labels=gallery_labels
            )
            vote_known_individuals_dict[camera_id] = known_individuals
        # print(f"Known individuals for camera {camera_id}: {known_individuals}")

    # Aggregate votes across camera pairs considering social groups
    camera_individuals = assign_known_individuals_to_cameras(
        vote_results_dict=vote_known_individuals_dict,
        camera_pairs={
            "ohne": ["016", "019"],
            "mit": ["017", "018"],
        }
    )
    print("Final known individuals assigned to cameras:", camera_individuals)
    
    if args.run_stitching:

        for camera_id in camera_ids:
            known_individuals = camera_individuals.get(camera_id, None)
            print(f"Known individuals for camera {camera_id}: {known_individuals}")
                                
            track_dirs = [get_track_dir(
                record_root=args.record_root,
                cam_id=camera_id,
                date=date,
            ) for date in dates]
            
            if not all([track_dir.exists() for track_dir in track_dirs]):
                logger.warning("Track directories for camera %s on dates %s do not all exist.", camera_id, dates)
                ### TODO -- handle missing directories better
                # continue
            
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
            
            if args.gallery_path is not None:
                merged_gallery_npz_path = args.gallery_path.replace('train_iid', 'merged_train_iid')
            else:
                merged_gallery_npz_path = (
                    args.checkpoint.parent / "pred_features" / "merged_train_iid" / "pytorch_result_e.npz"
                )
            
            tracklet_results, save_path = get_stitched_data(
                    track_dirs=track_dirs,
                    tracklet_manager=tracklet_manager,
                    camera_id=camera_id,
                    output_dir= Path(output_dir),
                    run_stitching=args.run_stitching,
                )

            final_camera_tracklets[camera_id] = (tracklet_results, save_path)


    if args.cross_camera_matching:
        ## Track-level cross-camera ID matching (v2)
        ## Works directly with track CSVs (timestamp, world_x, world_y)
        ## No dependency on stitched_id or voted_track_label
        cam_pairs = [
            ['016', '019'],
            ['018', '017']
        ]

        for camera_ids in cam_pairs:
            try:
                known_individuals = camera_individuals.get(camera_ids[0], None)
                if not known_individuals:
                    known_individuals = None

                # Resolve gallery path for cross-camera ReID voting
                gallery_npz_path = args.gallery_path
                if gallery_npz_path is None:
                    gallery_npz_path = (
                        args.checkpoint.parent / "pred_features" / "train_iid" / "pytorch_result_e.npz"
                    )

                # ── Cross-camera track-level matching ──
                track_to_xcid, summary_df, bout_summary_df = run_cross_camera_matching_v2(
                    record_root=args.record_root,
                    camera_ids=camera_ids,
                    start_datetime=pd.Timestamp(start_datetime),
                    end_datetime=pd.Timestamp(end_datetime),
                    known_individuals=known_individuals,
                    gallery_path=gallery_npz_path,
                    logger=logger,
                )

                # save bout
                bout_summary_out = output_dir / 'night_bout_summary' / f'{dates[0]}' / f"{'_'.join(known_individuals) if known_individuals else 'unknown'}_{'_'.join(camera_ids)}.csv"
                bout_summary_out.parent.mkdir(parents=True, exist_ok=True)
                if 'behavior_label_raw' in bout_summary_df.columns:
                    # Filter out invalid behavior labels before saving
                    bout_summary_df = bout_summary_df[bout_summary_df['behavior_label_raw'] != '00_invalid']
                elif 'behavior_label_stage1' in bout_summary_df.columns:
                    bout_summary_df = bout_summary_df[bout_summary_df['behavior_label_stage1'] != '00_invalid']
                elif 'behavior_label_old' in bout_summary_df.columns:
                    bout_summary_df = bout_summary_df[bout_summary_df['behavior_label_old'] != '00_invalid']
                bout_summary_df.to_csv(bout_summary_out, index=False)
                logger.info("Bout summary saved to %s", bout_summary_out)

            except Exception as e:
                logger.error("Error during cross-camera ID matching for cameras %s: %s", camera_ids, str(e))
                import traceback
                traceback.print_exc()
                continue




if __name__ == "__main__":
    sys.exit(main())
