"""
Video visualization script for Elephant ReID.

The script loads:
  1. A video file
  2. A YOLO segmentation/detection model (Ultralytics)
  3. A pretrained PoseGuidedReID model checkpoint
  4. A gallery features npz file generated via PoseGuidedReID inference

Pipeline:
  Video frame -> YOLO detection -> crop patches -> ReID feature extraction -> cosine similarity
  -> draw annotations -> save annotated video.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from decord import VideoReader, cpu, gpu
from utils import *


try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - guard for missing dependency
    raise ImportError(
        "Ultralytics is required for visualization. Install via `pip install ultralytics`."
    ) from exc

# Allow importing PoseGuidedReID modules when running from repo root
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Elephant ReID on a video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Path to save the annotated video.")
    parser.add_argument(
        "--yolo-model",
        required=True,
        help="Path to YOLO model weights (Ultralytics .pt / .onnx / TorchScript).",
    )
    parser.add_argument(
        "--class-names",
        required=True,
        help="Text file with YOLO class names (one per line).",
    )
    parser.add_argument(
        "--reid-config",
        required=True,
        help="Path to PoseGuidedReID config (.yml).",
    )
    parser.add_argument(
        "--reid-checkpoint",
        required=True,
        help="PoseGuidedReID checkpoint (.pth).",
    )
    parser.add_argument(
        "--gallery",
        required=True,
        help="Gallery features npz (generated via PoseGuidedReID inference).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for ReID model inference.",
    )
    parser.add_argument(
        "--gallery-device",
        default="cpu",
        help="Device to keep gallery embeddings on (cpu or cuda).",
    )
    parser.add_argument("--conf-thres", type=float, default=0.4, help="YOLO confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="YOLO IOU threshold.")
    parser.add_argument("--max-dets", type=int, default=50, help="Max detections per frame.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K ReID matches to visualize.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every N-th frame (>=1). Use >1 to speed up.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on processed frames (for quick debugging).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optionally resize video frames to this width while preserving aspect ratio.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity.",
    )
    return parser.parse_args()


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("visualize")


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    device = torch.device(args.device)
    gallery_device = torch.device(args.gallery_device)
    logger.info("Using device=%s, gallery_device=%s", device, gallery_device)

    class_names = load_class_names(args.class_names)
    yolo_model = YOLO(args.yolo_model)
    logger.info("Loaded YOLO model from %s", args.yolo_model)

    gallery = load_gallery_database(args.gallery, gallery_device)
    num_classes = len(set(gallery.ids.tolist()))
    logger.info("Loaded gallery embeddings: %s entries", len(gallery.labels))

    reid_model, transform = build_reid_model(
        args.reid_config,
        args.reid_checkpoint,
        num_classes=num_classes,
        device=device,
        logger=logger,
    )
    logger.info("Loaded ReID checkpoint from %s", args.reid_checkpoint)

    # --- decord video reader ---
    try:
        vr = VideoReader(args.video, ctx=cpu(0))  # or gpu(0) if you want GPU decode
    except Exception as e:
        raise FileNotFoundError(f"Unable to open video with decord: {args.video} ({e})")

    total_frames = len(vr)
    logger.info("Processing video (decord): %s (%d frames)", args.video, total_frames)

    # fps / size from decord
    try:
        fps = float(vr.get_avg_fps())
        if fps <= 0:
            fps = 30.0
    except Exception:
        fps = 30.0

    first_frame = vr[0].asnumpy()
    height, width = first_frame.shape[:2]
    if args.resize_width:
        writer_width = args.resize_width
        writer_height = int(height * (args.resize_width / width))
    else:
        writer_width, writer_height = width, height

    writer_size = (writer_width, writer_height)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        writer_size,
    )

    processed = 0
    bad_frame_count = 0

    if args.max_frames is not None:
        max_frame_idx = min(total_frames, args.max_frames * args.frame_skip)
    else:
        max_frame_idx = total_frames

    with torch.no_grad():
        with tqdm(total=total_frames if total_frames > 0 else None, desc="Frames") as pbar:
            for frame_idx in range(0, max_frame_idx):
                pbar.update(1)
                if args.frame_skip > 1 and frame_idx % args.frame_skip != 0:
                    continue

                if args.max_frames is not None and processed >= args.max_frames:
                    break

                try:
                    frame = vr[frame_idx].asnumpy()
                except Exception as e:
                    bad_frame_count += 1
                    if bad_frame_count <= 5 or bad_frame_count % 100 == 0:
                        logger.warning(
                            "Skipping damaged frame %d via decord (total skipped: %d) - %s",
                            frame_idx, bad_frame_count, e
                        )
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = maybe_resize(frame, args.resize_width)

                boxes, det_scores, cls_ids = run_yolo(
                    yolo_model,
                    frame,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    max_dets=args.max_dets,
                )
                if boxes.size == 0:
                    writer.write(frame)
                    processed += 1
                    continue

                tensors, kept_boxes, kept_indices = preprocess_patches(frame, boxes, transform)
                if not tensors:
                    writer.write(frame)
                    processed += 1
                    continue

                batch = torch.stack(tensors).to(device)
                feats = extract_features(reid_model, batch)
                feats = feats.to(gallery.features.device)
                scores_mat, indices_mat = match_gallery(feats, gallery, args.top_k)

                detections: List[DetectionResult] = []
                for idx, bbox in enumerate(kept_boxes):
                    det_idx = kept_indices[idx]
                    cls_id = cls_ids[det_idx] if det_idx < len(cls_ids) else -1
                    cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"id_{cls_id}"
                    det_score = float(det_scores[det_idx]) if det_idx < len(det_scores) else 0.0
                    match_scores = scores_mat[idx].tolist()
                    match_indices = indices_mat[idx].tolist()
                    matches = [
                        (str(gallery.labels[m_idx]), float(match_scores[j]))
                        for j, m_idx in enumerate(match_indices)
                    ]
                    detections.append(
                        DetectionResult(
                            bbox=bbox,
                            score=det_score,
                            cls_id=cls_id,
                            cls_name=cls_name,
                            matches=matches,
                            track_id=None,
                            display_track_id=None,
                            identity_label=None,
                            identity_score=None,
                            predictions=None,
                        )
                    )

                annotated = annotate_frame(frame.copy(), detections, gallery)
                writer.write(annotated)
                processed += 1

    writer.release()
    logger.info("Visualization saved to %s (processed %d frames)", args.output, processed)


if __name__ == "__main__":
    main()
