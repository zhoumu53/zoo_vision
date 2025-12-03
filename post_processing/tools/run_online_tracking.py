#!/usr/bin/env python3
"""
Online Tracking
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.online_tracker import OnlineTracker


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("online_tracking")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online tracking: YOLO + ByteTrack only"
    )
    
    # Required arguments
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--yolo-model", required=True, help="YOLO model path")
    parser.add_argument("--class-names", required=True, help="Class names file")
    
    # Device arguments
    parser.add_argument("--yolo-device", default="cuda", help="YOLO device (cuda/cpu)")
    
    # Detection parameters
    parser.add_argument("--conf-thres", type=float, default=0.65, help="YOLO confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="YOLO IOU threshold")
    parser.add_argument("--max-dets", type=int, default=50, help="Max detections per frame")
    
    # Processing parameters
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--resize-width", type=int, default=None, help="Resize width")
    
    # Other parameters
    parser.add_argument("--no-skip-existing", action="store_true", help="Reprocess even if output exists")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_level)
    
    # Create tracker
    tracker = OnlineTracker(
        video_path=args.video,
        output_dir=Path(args.output),
        yolo_model_path=args.yolo_model,
        class_names_path=args.class_names,
        yolo_device=args.yolo_device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_dets=args.max_dets,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        resize_width=args.resize_width,
        logger=logger,
    )
    
    # Run tracking
    print("Running online tracking...")
    output_path = tracker.run(skip_if_exists=not args.no_skip_existing)
    logger.info("Online tracking complete: %s", output_path)
    return 0


if __name__ == "__main__":
    main()
