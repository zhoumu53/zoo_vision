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
    get_good_frame_indices
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

    frame_indices = get_good_frame_indices(df_tracks)
        
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
                good_quality_indices = df_tracks[ df_tracks["quality_label"] == 'good' ].index
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
        
        if sample_rate < 1.0:
            frame_indices = sorted(list(frame_indices))
            # sample frame_indices according to sample_rate - uniformly
            _frame_indices = frame_indices[:: int(1/sample_rate)]
            frame_indices = set(_frame_indices)
            logger.info(f"Sampled {len(frame_indices)} frames for ReID from {len(df_tracks)} total frames.")

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
        description="Post-processing: Clean and smooth tracking data"
    )
    
    parser.add_argument("--date", default='20251129', help="Input JSONL file (stitched)")
    parser.add_argument("--record-root", type=Path, default='/media/ElephantsWD/elephants/test', help="Root directory for records.")
    parser.add_argument("--config", type=Path, default='/media/mu/zoo_vision/training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml', help="ReID config file path.")
    parser.add_argument("--checkpoint", type=Path, default='/media/ElephantsWD/reid_models/logs/swin_adamw_lr0003_bs64_softmax_triplet/net_best.pth', help="ReID checkpoint path.")
    parser.add_argument("--behavior-model", type=Path, default="/media/mu/zoo_vision/models/sleep/vit/v2_no_validation/config.ptc", help="Behavior classification model path (optional).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., cuda:0 or cpu).")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference.")
    parser.add_argument("--overwrite-behavior", action="store_true", help="Overwrite existing behavior processed files.")
    parser.add_argument("--overwrite-reid", action="store_true", help="Overwrite existing ReID processed files.")
    parser.add_argument("--gallery-path", type=Path, default=None, help="Path to gallery features NPZ file.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--height", type=int, default=1080, help="Height of the video frames.")
    parser.add_argument("--width", type=int, default=1920, help="Width of the video frames.")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Sample rate for frame selection in ReID (0 < rate <= 1).")
    parser.add_argument("--camera-ids", nargs='+', default=["016", "017", "018", "019"], help="List of camera IDs to process.")

    return parser.parse_args()



def main():
    starttime = datetime.now()
    args = parse_args()
    logger = setup_logger(args.log_level)
    
    logger.info("Loading data from day %s", args.date)
    camera_ids = args.camera_ids
    
    all_track_files = list_track_files_all_cams(
        record_root=args.record_root,
        camera_ids=camera_ids,
        date=args.date,
        logger=logger,
    )

    full_track_files = [tf for cam_id in camera_ids for tf in all_track_files[cam_id]]
    logger.info("Found %d track files for post-processing.", len(full_track_files))

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


    behavior_model = load_behavior_model(args.behavior_model, args.device, logger)
    for track_file in tqdm(full_track_files, desc="Processing track files"):
        if 'id_behavior' in str(track_file):
            continue

        track_video_file = track_file.with_suffix('.mkv')
        if not track_video_file.exists():
            track_video_file = track_file.with_suffix('.mp4')
        if not track_video_file.exists():
            logger.warning("No corresponding video file found for track: %s", track_file)
            continue

        #### TODO -- update this for live version
        process_track_video(
            track_video_file=track_video_file,
            behavior_model=behavior_model,
            reid_model=reid_model,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
            batch_size=args.batch_size,
            device=args.device,
            overwrite_behavior=args.overwrite_behavior,
            overwrite_reid=args.overwrite_reid,
            logger=logger,
            sample_rate=args.sample_rate,
        )
    endtime = datetime.now()
    logger.info("Post-processing completed in %s", str(endtime - starttime))


if __name__ == "__main__":
    sys.exit(main())
