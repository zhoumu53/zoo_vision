import logging
from pathlib import Path
from post_processing.tools.videoloader import VideoLoader
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
    
def compute_box_wh_ratio(bbox_right, bbox_left, bbox_bottom, bbox_top) -> float:
    """Compute width/height ratio of the bounding box at a given index in the dataframe."""
    width = bbox_right - bbox_left
    height = bbox_bottom - bbox_top
    if height == 0 or width == 0:
        return 0.0  
    wh_ratio = width / height
    return wh_ratio
    
    
def filter_by_box_quality(df_tracks,
                           bbox_ratio_lower: float = 1/3,
                           bbox_ratio_upper: float = 3.0
                           ) -> list[int]:
    ### Identify good frame indices based on box quality 
    
    ## box (w/h) ratio threshold: 1/3 to 3
    wh_ratios = df_tracks.apply(
        lambda row: compute_box_wh_ratio(row['bbox_right'], 
                                         row['bbox_left'], 
                                         row['bbox_bottom'], 
                                         row['bbox_top']),
        axis=1
    )
    good_quality_mask = (wh_ratios >= bbox_ratio_lower) & (wh_ratios <= bbox_ratio_upper)
    
    return good_quality_mask


def get_good_frame_indices(df_tracks, ) -> list[int]:
    """Get frame indices with good quality boxes."""
    good_quality_mask = filter_by_box_quality(df_tracks)
    good_frame_indices = df_tracks.index[good_quality_mask].tolist()
    return good_frame_indices
