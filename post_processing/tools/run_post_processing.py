#!/usr/bin/env python3
"""
Post-Processing CLI

Command-line interface for running post-processing (Stage 3).
"""
import argparse
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

from post_processing.tools.run_reid_feature_extraction import (
    run_feature_extraction,
    load_reid,
    load_npz_files,
)
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


def load_gallery_features(
    reid_model,
    checkpoint_path: Path,
    provided_path: Path | None,
    logger: logging.Logger,
):
    """Resolve gallery path and load gallery features."""
    gallery_path = provided_path
    if gallery_path is None:
        gallery_path = (
            checkpoint_path.parent / "pred_features" / "train_iid" / "pytorch_result_e.npz"
        )
        logger.info("No gallery path provided. Using default: %s", gallery_path)
    return reid_model.load_features(gallery_path)


def load_behavior_model(
    behavior_model_path: Path | None, device: str, logger: logging.Logger
) -> BehaviorInference | None:
    """Load behavior model if path is provided."""
    if behavior_model_path is None:
        return None
    logger.info("Loading behavior model from %s", behavior_model_path)
    return BehaviorInference(
        model_path=str(behavior_model_path),
        device=device,
        logger=logger,
    )


def process_track_video(
    track_file: Path,
    behavior_model: BehaviorInference | None,
    reid_model,
    gallery_features,
    gallery_labels,
    batch_size: int,
    device: str,
    overwrite: bool,
    logger: logging.Logger,
) -> str | None:
    """Run behavior + ReID on a single track video."""
    track_filename = track_file.name
    logger.info("Processing track video: %s", track_filename)

    csv_path = track_file.with_suffix(".csv")
    if not csv_path.exists():
        logger.warning("Corresponding CSV file not found for video: %s", track_file)
        return None

    df_tracks = pd.read_csv(csv_path)

    # Behavior classification, saved to CSV (frame level)
    if behavior_model is not None and "behavior_label" not in df_tracks.columns:
        behavior_preds = run_behavior_on_track(
            video_path=track_file,
            behavior_model=behavior_model,
            batch_size=batch_size,
        )
        if len(behavior_preds) != len(df_tracks):
            logger.warning(
                "Behavior predictions length %d does not match CSV rows %d for %s",
                len(behavior_preds),
                len(df_tracks),
                track_file,
            )
            return None

        df_tracks["behavior_label"] = [pred[0] for pred in behavior_preds]
        df_tracks["behavior_conf"] = [pred[1] for pred in behavior_preds]
        df_tracks.to_csv(csv_path, index=False)
        logger.info("Saved behavior predictions to %s", csv_path)

    # ReID feature extraction -> save npz; skip if processed
    npz_path = track_file.with_suffix(".npz")
    try:
        if npz_path.exists() and not overwrite:
            voted_identity_label = load_npz_files(npz_path).get("voted_labels", "unknown")
            logger.info(
                "NPZ already exists for %s, loaded voted label: %s",
                track_filename,
                voted_identity_label,
            )
            return voted_identity_label

        voted_identity_label = run_feature_extraction(
            video_path=track_file,
            reid_model=reid_model,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
            device=device,
            batch_size=batch_size,
        )
        return voted_identity_label
    except Exception as exc:  # keep processing other videos
        logger.error("Error processing ReID for %s: %s", track_filename, exc)
        return None


def run_stitching(
    track_dir: Path,
    camera_id: str,
    start_time: str,
    end_time: str,
    gallery_features,
    gallery_labels,
    reid_model,
    logger: logging.Logger,
):
    """Stitch tracklets for a single camera/time window and assign IDs."""
    logger.info(
        "Stitching tracklets for camera %s between %s and %s",
        camera_id,
        start_time,
        end_time,
    )
    known_labels = ["Indi", "Chandra"]  # TODO: make configurable per camera
    valid_indices = (
        [i for i, label in enumerate(gallery_labels) if label in known_labels]
        if known_labels is not None
        else list(range(len(gallery_labels)))
    )
    g_feats = gallery_features[valid_indices]
    g_labels = [gallery_labels[i] for i in valid_indices]

    tracklet_manager = TrackletManager(
        track_dir=track_dir,
        camera_id=camera_id,
        num_identities=2,  # TODO: set dynamically? with human prior?
        logger=logger,
        start_time=start_time,
        end_time=end_time,
    )
    tracklet_manager.load_tracklets_for_camera()
    print(f"Loaded {len(tracklet_manager.tracklets)} tracklets for camera {tracklet_manager.camera_id}")

    print("\n" + "=" * 80)
    print("Testing BIDIRECTIONAL GALLERY-based stitching (recommended for elephants)")
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

    # TODO - assign identity labels based on gallery matching , or matching to the features from past days
    # agg_features = tracklet_manager.aggregate_tracklet_features()
    # trackid2identity_label = {}
    # for tid in tracklet_manager.track_ids:
    #     features = agg_features[tid]
    #     matched_labels = reid_model.match_to_gallery(
    #         features, g_feats, gallery_labels=g_labels, top_k=1
    #     )[-1][0]
    #     print(f"Stitched Tracklet ID {tid} matched identity: {matched_labels}")
    #     trackid2identity_label[tid] = matched_labels

    tracklet_manager.save_stitched_tracklets(trackid2identity_label=trackid2identity_label)


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
    # parser.add_argument("--start-time", type=str, default="15:30:00", help="Start time for processing (HH:MM:SS)." \
    # " Default is 15:30:00 to cover night time from PM to AM.")
    # parser.add_argument("--end-time", type=str, default="23:59:59", help="End time for processing (HH:MM:SS)." \
    # " Default is 23:59:59 next day to cover night time from PM to AM.    ")
    ### behavior model
    parser.add_argument("--behavior-model", type=Path, default="/media/mu/zoo_vision/models/sleep/vit/v2_no_validation/config.ptc", help="Behavior classification model path (optional).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., cuda:0 or cpu).")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files.")
    parser.add_argument("--gallery-path", type=Path, default=None, help="Path to gallery features NPZ file.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    
    # Load data
    logger.info("Loading data from day %s", args.date)

    ### TODO -- load data from PM to AM for one night (15:30 to 08:30 next day)
    time_splits = [
        ("00:00:00", "07:30:00"),
        ("16:30:00", "23:59:59"),
    ]
    
    # Load file manager and get input file path
    camera_ids = ["016", "017", "018", "019"]

    ### TODO: list all track files for 1 night (e.g. from 15:30 to next day 08:30)
    all_track_files = list_track_files_all_cams(
        record_root=args.record_root,
        camera_ids=camera_ids,
        date=args.date,
        logger=logger,
    )

    full_track_videos = [vf for cam_id in camera_ids for vf in all_track_files[cam_id][1]]
    logger.info("Found %d track files for post-processing.", len(full_track_videos))

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

    behavior_model = load_behavior_model(args.behavior_model, args.device, logger)

    for track_file in tqdm(full_track_videos, desc="Processing videos"):
        process_track_video(
            track_file=track_file,
            behavior_model=behavior_model,
            reid_model=reid_model,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
            batch_size=args.batch_size,
            device=args.device,
            overwrite=args.overwrite,
            logger=logger,
        )


    # stitching per camera
    for camera_id in camera_ids:
        track_dir = get_track_dir(
            record_root=args.record_root,
            cam_id=camera_id,
            date=args.date,
        )
        for start_time, end_time in time_splits:
            run_stitching(
                track_dir=track_dir,
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                gallery_features=gallery_features,
                gallery_labels=gallery_labels,
                reid_model=reid_model,
                logger=logger,
            )

    ### TODO - cross-camera calibration and stitching
    


if __name__ == "__main__":
    sys.exit(main())
