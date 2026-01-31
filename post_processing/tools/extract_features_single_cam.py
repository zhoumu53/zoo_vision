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
    list_track_files_all_cams,
    get_track_dir,
)

from post_processing.core.behavior_inference import BehaviorInference

from post_processing.tools.utils import (
    setup_logger, 
    load_behavior_model, 
    run_behavior_on_track, 
    load_gallery_features, 
    get_good_frame_indices,
    select_frame_indices_by_second_best_quality
)
from post_processing.core.temporal_smooth import behavior_label_smooth



def process_track_video(
    track_video_file: Path,
    behavior_model: BehaviorInference | None,
    reid_model,
    gallery_features,
    gallery_labels,
    batch_size: int,
    device: str,
    logger: logging.Logger,
    overwrite_behavior: bool = True,
    overwrite_reid: bool = True,
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
    ### only read columns needed
    columns_drop = ['behavior_label','behavior_conf','identity_label','quality_label','quality_conf']
    # remove columns if exist
    for col in columns_drop:
        if col in df_tracks.columns:
            df_tracks = df_tracks.drop(columns=[col])

    frame_indices = []  ###TODO #get_good_frame_indices(df_tracks)
        
    # Behavior classification, saved to CSV (frame level)
    out_csv_path = csv_path if out_csv_path is None else out_csv_path
    if behavior_model is not None and out_csv_path is not None:
        ####

            # if behavior column already exists and not overwrite
        if Path(out_csv_path).exists() and not overwrite_behavior:
            logger.info("Behavior CSV already exists and overwrite is False: %s", out_csv_path)
            return None

        try:
            behavior_preds = run_behavior_on_track(
                video_path=track_video_file,
                behavior_model=behavior_model,
                batch_size=batch_size,
            )
            if len(behavior_preds) != len(df_tracks):
                logger.warning(
                    "Behavior predictions length %d does not match CSV rows %d for %s",
                    len(behavior_preds),
                    len(df_tracks),
                    track_video_filename,
                )
                return None
            
            beh_preds = []
            qua_preds = []
            beh_confs = []
            qua_confs = []

            for idx, pred in enumerate(behavior_preds):
                behavior_label = pred[0]
                behavior_conf = pred[1]
                
                ### check if pred from 2 heads model
                if 'good' in behavior_label or 'bad' in behavior_label:
                    quality_label = behavior_label.split('_')[-1]
                    behavior_label = behavior_label.replace(f"_{quality_label}", "")
                    behavior_conf, quality_conf = behavior_conf.split('_')

                else:
                    quality_label = None
                    quality_conf = None
                
                qua_preds.append(quality_label)
                qua_confs.append(quality_conf)
                beh_preds.append(behavior_label)
                beh_confs.append(behavior_conf)

            df_tracks["behavior_label"] = beh_preds
            df_tracks["behavior_conf"] = beh_confs

            if len(qua_preds) > 0 and len(qua_confs) > 0:
                df_tracks["quality_label"] = qua_preds
                df_tracks["quality_conf"] = qua_confs
                ### TODO - update the sample logic to use quality labels
                if sample_rate == 1:
                    good_quality_indices = df_tracks[ df_tracks["quality_label"] == 'good' ].index
                else:
                    # now best quality frames - top_k
                    good_quality_indices = set(
                        select_frame_indices_by_second_best_quality(df_tracks, top_k=2)
                    )
                if len(good_quality_indices) == 0:
                    frame_indices = set()
                else:
                    frame_indices = set(frame_indices).intersection(set(good_quality_indices))

            if out_csv_path is None:
                out_csv_path = csv_path
            df_tracks.to_csv(out_csv_path, index=False)
            logger.info("Saved behavior predictions to %s", out_csv_path)
        except Exception as exc:
            logger.error("Error running behavior model on %s: %s", track_video_filename, exc)
            return None



    # ReID feature extraction -> save npz; skip if processed
    npz_path = track_video_file.with_suffix(".npz")
    try:
        if npz_path.exists() and not overwrite_reid:
            voted_identity_label = load_npz_files(npz_path).get("voted_labels", "unknown")
            logger.info(
                "NPZ already exists for %s, loaded voted label: %s",
                track_video_filename,
                voted_identity_label,
            )
            return voted_identity_label
        
        # if sample_rate < 1.0:  
        #     frame_indices = sorted(list(frame_indices))
        #     # sample frame_indices according to sample_rate - uniformly
        #     _frame_indices = frame_indices[:: int(1/sample_rate)]
        #     frame_indices = set(_frame_indices)
        #     logger.info(f"Sampled {len(frame_indices)} frames for ReID from {len(df_tracks)} total frames.")

        ## only do ReID on selected frames
        if len(frame_indices) == 0:
            logger.warning("No good frames selected for ReID in %s", track_video_filename)
            return None

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
    
    camera_ids = [cam_id]
    
    all_track_files = list_track_files_all_cams(
        record_root=config.data.record_root,
        camera_ids=camera_ids,
        date=args.date,
        logger=logger,
    )

    full_track_files = [tf for cam_id in camera_ids for tf in all_track_files[cam_id]]
    logger.info("Found %d track files for post-processing.", len(full_track_files))

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
        if 'id_behavior' in str(track_file):
            continue

        track_video_file = track_file.with_suffix('.mkv')
        if not track_video_file.exists():
            track_video_file = track_file.with_suffix('.mp4')
        if not track_video_file.exists():
            logger.warning("No corresponding video file found for track: %s", track_file)
            continue

        process_track_video(
            track_video_file=track_video_file,
            behavior_model=behavior_model,
            reid_model=reid_model,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
            batch_size=config.processing.batch_size,
            device=config.processing.device,
            overwrite_behavior=config.processing.overwrite_behavior,
            overwrite_reid=config.processing.overwrite_reid,
            logger=logger,
            sample_rate=config.processing.sample_rate,
        )
    endtime = datetime.now()
    logger.info("Post-processing completed in %s", str(endtime - starttime))


if __name__ == "__main__":
    sys.exit(main())
