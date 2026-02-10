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
from datetime import datetime
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
    id_feature_extraction,
    load_reid,
    load_npz_files,
)

from post_processing.core.file_manager import (
    list_track_files,
    get_track_dir,
    get_track_files_by_timestamp,
)

from post_processing.core.behavior_inference import BehaviorInference

from post_processing.tools.utils import (
    setup_logger, 
    load_behavior_model, 
    run_behavior_on_track, 
    load_gallery_features, 
)
from post_processing.core.temporal_smooth import behavior_label_smooth



def process_behavior_for_track(
    track_video_file: Path,
    behavior_model: BehaviorInference | None,
    batch_size: int,
    logger: logging.Logger,
    overwrite_behavior: bool = True,
    out_csv_path: Path | None = None,
    sample_rate: float = 1.0,
) -> str | None:
    """Run behavior + ReID on a single track video."""
    track_video_filename = track_video_file.name
    logger.info("Processing track video: %s", track_video_filename)

    csv_path = track_video_file.with_suffix(".csv")
    if not csv_path.exists():
        logger.warning("Corresponding CSV file not found for video: %s", track_video_file)
        return None
    if not track_video_file.exists():
        logger.warning("Track video file not found: %s", track_video_file)
        return None

    df_tracks = pd.read_csv(csv_path)
    if df_tracks.empty:
        logger.warning("Empty CSV file for track: %s", csv_path)
        return None
    ### only read columns needed
    columns_drop = ['behavior_label','behavior_conf','identity_label','quality_label','quality_conf']
    # remove columns if exist -- from old processing
    for col in columns_drop:
        if col in df_tracks.columns:
            df_tracks = df_tracks.drop(columns=[col])

    frame_indices = []
        
    # Behavior classification, saved to CSV (frame level)
    out_csv_path = csv_path.with_name(csv_path.stem + '_behavior.csv') if out_csv_path is None else out_csv_path
    if behavior_model is not None and out_csv_path is not None:
        ####
            # if behavior column already exists and not overwrite
        if Path(out_csv_path).exists() and not overwrite_behavior:
            logger.info("Behavior CSV already exists and overwrite is False: %s", out_csv_path)
            # load csv - get good quality frame indices
            df_existing = pd.read_csv(out_csv_path)
            
            ### check if '_raw' column exists - if yes, use 'behavior_label' for behavior_label, if not, use 'behavior_label_raw' for behavior_label
            if 'behavior_label_raw' not in df_existing.columns:
                ## temporal smooth -- for behavior labels 
                df_track_smoothed = behavior_label_smooth(
                    df_existing,
                    beh_col='behavior_label',
                    out_col = "behavior_label_smoothed",
                )
                ## rename behavior_label_smoothed to behavior_label, behavior_label to behavior_label_raw
                df_track_smoothed = df_track_smoothed.rename(
                    columns={
                        'behavior_label': 'behavior_label_raw',
                        'behavior_label_smoothed': 'behavior_label',
                    }
                )
                ### save back to csv
                df_track_smoothed.to_csv(out_csv_path, index=False)
            
            ### sample from good quality frames & high confidence - behavior_conf >= 0.7
            ### TODO -- ACTIVE REID because currently in gallery-set we don't have enough sleeping frames for reid matching
            good_indices = df_existing.index[
                    # (df_existing["behavior_label"] == '01_standing' ) &
                    (df_existing["quality_label"] == 'good') &
                    (df_existing["behavior_conf"].astype(float) >= 0.7)
                ].tolist()
            frame_indices = good_indices

            return frame_indices
            

        try:
            ### only keep column ['timestamp'] initially -- timestamp mismatch issue fixed -- so copy full df_tracks
            df_tracks_behavior = df_tracks[['timestamp']].copy()
            behavior_preds = run_behavior_on_track(
                video_path=track_video_file,
                behavior_model=behavior_model,
                batch_size=batch_size,
            )
            if len(behavior_preds) != len(df_tracks):                
                logger.warning(
                    "Behavior predictions length %d does not match CSV rows %d for %s, skipping behavior processing.",
                    len(behavior_preds),
                    len(df_tracks),
                    track_video_filename,
                )
            frame_ids = []
            beh_preds = []
            qua_preds = []
            beh_confs = []
            qua_confs = []

            for idx, pred in enumerate(behavior_preds):
                behavior_label = pred[0]
                behavior_conf = pred[1]
                
                # DEFAULT MODEL - 2 HEADS - behavior + quality
                quality_label = behavior_label.split('_')[-1]
                behavior_label = behavior_label.replace(f"_{quality_label}", "")
                behavior_conf, quality_conf = behavior_conf.split('_')

                qua_preds.append(quality_label)
                qua_confs.append(quality_conf)
                beh_preds.append(behavior_label)
                beh_confs.append(behavior_conf)

            df_tracks_behavior["behavior_label"] = beh_preds
            df_tracks_behavior["behavior_conf"] = beh_confs
            df_tracks_behavior["quality_label"] = qua_preds
            df_tracks_behavior["quality_conf"] = qua_confs

            ### if beh_conf < 0.7 -> set to 'invalid'
            df_tracks_behavior["behavior_label"] = df_tracks_behavior.apply(
                lambda row: row["behavior_label"] if float(row["behavior_conf"]) >= 0.7 else "00_invalid",
                axis=1,
            )
            good_indices = df_tracks_behavior.index[
                    # (df_tracks_behavior["behavior_label"] == '01_standing' ) &
                    (df_tracks_behavior["quality_label"] == 'good') &
                    (df_tracks_behavior["behavior_conf"].astype(float) >= 0.7)
                ].tolist()
            frame_indices = good_indices
            
            ## temporal smooth -- for behavior labels 
            df_track_smoothed = behavior_label_smooth(
                df_tracks_behavior,
                beh_col='behavior_label',
                out_col = "behavior_label_smoothed",
            )
            ## rename behavior_label_smoothed to behavior_label, behavior_label to behavior_label_raw
            df_track_smoothed = df_track_smoothed.rename(
                columns={
                    'behavior_label': 'behavior_label_raw',
                    'behavior_label_smoothed': 'behavior_label',
                }
            )
            df_tracks_behavior = df_track_smoothed

            df_tracks_behavior.to_csv(out_csv_path, index=False)
            logger.info("Saved behavior predictions to %s", out_csv_path)
        except Exception as exc:
            logger.error("Error running behavior model on %s: %s", track_video_filename, exc)
            return frame_indices
        
    return frame_indices



def process_reid_for_track(
    track_video_file: Path,
    reid_model,
    gallery_features,
    gallery_labels,
    batch_size: int,
    device: str,
    logger: logging.Logger,
    overwrite_reid: bool = True,
    frame_indices: list[int] = [],
) -> str | None:
    """Run ReID feature extraction on a single track video."""

    # ReID feature extraction -> save npz; skip if processed
    npz_path = track_video_file.with_suffix(".npz")
    track_video_filename = track_video_file.name
    logger.info("Processing ReID for track video: %s", track_video_filename)
    try:
        if npz_path.exists() and not overwrite_reid:
            voted_identity_label = load_npz_files(npz_path).get("voted_labels", "unknown")
            logger.info(
                "NPZ already exists for %s, loaded voted label: %s",
                track_video_filename,
                voted_identity_label,
            )
            return voted_identity_label

        voted_identity_label = id_feature_extraction(
            video_path=track_video_file,
            reid_model=reid_model,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
            device=device,
            batch_size=batch_size,
            frame_ids=list(frame_indices)
        )
        logger.info(f"Voted ID for {track_video_filename}: {voted_identity_label}")
        return voted_identity_label
    except Exception as exc:  # keep processing other videos
        logger.error("Error processing ReID for %s: %s", track_video_filename, exc)
        return None



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-processing: Extract features for single camera"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "core" / "config" / "configs.yaml",
        help="Path to pipeline configuration YAML file",
    )
    parser.add_argument("--date", required=True, help="Date to process (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--run-night", action="store_true", help="Whether to run night videos")
    # parser.set_defaults(run_night=False)
    parser.add_argument("--cam-id", type=str, default=None, help="Camera ID to process (overrides config)")
    parser.add_argument("--record-root", type=Path, default=None, help="Root directory for records (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for inference (overrides config)")
    parser.add_argument("--sample-rate", type=float, default=None, help="Sample rate for frame selection in ReID (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (overrides config)")
    parser.add_argument("--gallery-path", type=Path, default=None, help="Path to gallery features NPZ file (overrides config)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--override", "-o",
        type=str,
        nargs="+",
        default=[],
        help="Override any config value using dot notation (e.g., processing.overwrite_reid=true)",
    )

    return parser.parse_args()



def main():
    starttime = datetime.now()
    args = parse_args()
    
    from post_processing.core.config.config_loader import load_config, update_config_from_args
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config from {args.config}: {e}")
        return 1
    
    logger = setup_logger(args.log_level)
    logger.info("Config loaded from: %s", args.config)
    logger.info("Processing date: %s", args.date)
    
    # Apply command-line overrides to config
    update_config_from_args(
        config,
        record_root=args.record_root,
        # sample_rate=args.sample_rate,
        camera_ids=None,
    )
    
    # Apply device override if provided
    if args.device is not None:
        config.processing.device = args.device
        logger.info("Override device: %s", config.processing.device)
    
    # Apply batch_size override if provided
    if args.batch_size is not None:
        config.processing.batch_size = args.batch_size
        logger.info("Override batch_size: %d", config.processing.batch_size)
    
    # Apply gallery_path override if provided
    if args.gallery_path is not None:
        config.models.reid_gallery_path = args.gallery_path
        logger.info("Override gallery_path: %s", config.models.reid_gallery_path)
    
    # Apply generic overrides from --override flag
    if args.override:
        override_dict = {}
        for item in args.override:
            if '=' in item:
                key, value = item.split('=', 1)
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
                override_dict[key] = value
            else:
                logger.warning("Invalid override format (missing '='): %s", item)
        
        if override_dict:
            logger.info("Applying %d generic overrides...", len(override_dict))
            config.update_from_dict(override_dict)
            for key, value in override_dict.items():
                logger.info("Override %s: %s", key, value)
    
    # Get camera_id from args or use first camera from config
    cam_id = args.cam_id if args.cam_id else config.cameras.ids[0]
    logger.info("Processing camera: %s", cam_id)
        
    # if run-night --> process data from date (1500 to 0800 next day)
    if args.run_night:        
        start_dt = pd.Timestamp(args.date + " 15:00:00")
        next_day = (pd.Timestamp(args.date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end_dt = pd.Timestamp(next_day + " 08:00:00")
        try:
            full_track_files = get_track_files_by_timestamp(
                record_root=args.record_root,
                camera_id=cam_id,
                start_dt=start_dt,
                end_dt=end_dt,
                logger=logger,
            )
        except Exception as exc:
            logger.error("Failed to get track files for night processing: %s", exc)
            return 1
    else:
        full_track_files = list_track_files(
            get_track_dir(args.record_root, cam_id, args.date),
        )

    if not full_track_files or len(full_track_files) == 0:
        logger.warning("No track files found for night processing for camera %s.", cam_id)
        logger.warning("Please ensure that tracks are available in record_root: %s", get_track_dir(args.record_root, cam_id, args.date))
        logger.info("Exiting post-processing.")
        return 1
    else:
        logger.info("Found %d track files for night processing.", len(full_track_files))

    reid_model = load_reid(
        config_path=config.models.reid_config,
        checkpoint_path=config.models.reid_checkpoint,
        device=config.processing.device,
        mode="feature",
    )

    gallery_features, gallery_labels = load_gallery_features(
        reid_model=reid_model,
        checkpoint_path=config.models.reid_checkpoint,
        provided_path=config.models.reid_gallery_path,
        logger=logger,
    )

    behavior_model = load_behavior_model(
        config.models.behavior_model_path,
        config.processing.device,
        logger,
    )
    
    for track_file in tqdm(full_track_files, desc="Processing track files"):
        if '_behavior' in str(track_file):
            continue
        
        ### skip - not full track files
        if 'part_' in str(track_file):
            continue

        track_video_file = track_file.with_suffix('.mkv')
        if not track_video_file.exists():
            track_video_file = track_file.with_suffix('.mp4')
        if not track_video_file.exists():
            logger.warning("No corresponding video file found for track: %s", track_file)
            continue

        frame_indices = process_behavior_for_track(
            track_video_file=track_video_file,
            behavior_model=behavior_model,
            batch_size=config.processing.batch_size,
            overwrite_behavior=config.processing.overwrite_behavior,
            logger=logger,
            sample_rate=config.processing.sample_rate,
        )
        
        ### if frame_indices is None, skip reid processing
        if frame_indices is None:  ### empty tracks (but if frame_indices is empty list, still process -- for stitching)
            continue

        process_reid_for_track(
            track_video_file=track_video_file,
            reid_model=reid_model,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
            batch_size=config.processing.batch_size,
            device=config.processing.device,
            overwrite_reid=config.processing.overwrite_reid,
            logger=logger,
            frame_indices=frame_indices,   ## good frame indices from behavior
        )
  
    endtime = datetime.now()
    logger.info("Post-processing completed in %s", str(endtime - starttime))


if __name__ == "__main__":
    sys.exit(main())
