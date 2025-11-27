"""
Baseline visualization script: run YOLO + ByteTrack only (no ReID stitching).
Used to compare against improved ReID-enabled pipelines.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

from utils import DetectionResult, load_class_names, maybe_resize, run_yolo_byteTrack

# --------- 颜色映射：为每个 stitched ID 给一个稳定的颜色 ---------
TRACK_COLORS: Dict[int, Tuple[int, int, int]] = {}


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """给每个 display_track_id 分配一个稳定的随机颜色。"""
    if track_id not in TRACK_COLORS:
        rng = np.random.RandomState(track_id & 0xFFFF)
        color = tuple(int(x) for x in rng.randint(50, 255, size=3))
        TRACK_COLORS[track_id] = color
    return TRACK_COLORS[track_id]


# --------- CLI & logger ---------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight online ReID within a single video.")
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
        "--yolo-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for YOLO inference (cuda / cuda:0 / cpu).",
    )
    parser.add_argument("--conf-thres", type=float, default=0.4, help="YOLO confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="YOLO IOU threshold.")
    parser.add_argument("--max-dets", type=int, default=50, help="Max detections per frame.")
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
    # 可选：保存部分 JPG 方便你检查
    parser.add_argument(
        "--save-jpg",
        action="store_true",
        help="Save some annotated frames as JPGs for inspection.",
    )
    parser.add_argument(
        "--jpg-interval",
        type=int,
        default=10,
        help="Save a JPG every N processed frames.",
    )
    parser.add_argument(
        "--jpg-max-count",
        type=int,
        default=2000,
        help="Maximum number of JPG frames to save.",
    )
    return parser.parse_args()


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("visualize_light")


# --------- 可视化：只画 bbox + stitched ID ---------

def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[DetectionResult],
) -> np.ndarray:
    for det in detections:
        disp_id = det.display_track_id
        raw_id = det.track_id
        color = get_track_color(disp_id if disp_id is not None else raw_id)

        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # 文本：显示合并后的 ID 和 原始 tracker ID
        text = f"ID {disp_id} (trk {raw_id}) det {det.score:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


# --------- 主流程：YOLO + ByteTrack ---------

def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    # 1) 模型加载
    from ultralytics import YOLO

    class_names = load_class_names(args.class_names)
    yolo_model = YOLO(args.yolo_model)
    if args.yolo_device:
        yolo_model.to(args.yolo_device)
    logger.info("Loaded YOLO model from %s on %s", args.yolo_model, args.yolo_device)

    # 2) 视频读取（decord）
    try:
        vr = VideoReader(args.video, ctx=cpu(0))
    except Exception as e:
        raise FileNotFoundError(f"Unable to open video with decord: {args.video} ({e})")

    total_frames = len(vr)
    logger.info("Processing video (decord): %s (%d frames)", args.video, total_frames)

    try:
        fps = float(vr.get_avg_fps())
        if fps <= 0:
            fps = 30.0
    except Exception:
        fps = 30.0

    # 先读一帧确定尺寸
    first_frame = vr[0].asnumpy()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    first_frame = maybe_resize(first_frame, args.resize_width)
    height, width = first_frame.shape[:2]

    videoname = Path(args.video).stem + "_tracks.mp4"
    # create output directory if not exists 
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / videoname
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    # JPG 保存目录
    jpg_dir = output_path.with_suffix("")
    jpg_saved = 0
    if args.save_jpg:
        if jpg_dir.exists():
            import shutil
            shutil.rmtree(jpg_dir)
        jpg_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    bad_frame_count = 0

    # 计算将要处理的 frame indices
    if args.max_frames is not None:
        max_frame_idx = min(total_frames, args.max_frames * args.frame_skip)
    else:
        max_frame_idx = total_frames
    frame_indices = list(range(0, max_frame_idx, max(args.frame_skip, 1)))

    logger.info(
        "Will process %d / %d frames (frame_skip=%d, max_frames=%s)",
        len(frame_indices),
        total_frames,
        args.frame_skip,
        args.max_frames,
    )

    with torch.no_grad():
        with tqdm(total=len(frame_indices), desc="Frames") as pbar:
            for frame_idx in frame_indices:
                pbar.update(1)

                # 读帧
                try:
                    frame_np = vr[frame_idx].asnumpy()
                except Exception as e:
                    bad_frame_count += 1
                    if bad_frame_count <= 5 or bad_frame_count % 100 == 0:
                        logger.warning(
                            "Skipping damaged frame %d via decord (total skipped: %d) - %s",
                            frame_idx,
                            bad_frame_count,
                            e,
                        )
                    continue

                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                frame_bgr = maybe_resize(frame_bgr, args.resize_width)

                # YOLO + ByteTrack
                boxes, det_scores, cls_ids, track_ids = run_yolo_byteTrack(
                    yolo_model,
                    frame_bgr,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    max_dets=args.max_dets,
                    tracker_cfg="bytetrack.yaml",
                    device=args.yolo_device,
                )

                detections: List[DetectionResult] = []

                if boxes.size != 0:
                    for det_idx, bbox in enumerate(boxes):
                        raw_track_id = (
                            int(track_ids[det_idx])
                            if det_idx < len(track_ids)
                            else -1
                        )

                        cls_id = cls_ids[det_idx] if det_idx < len(cls_ids) else -1
                        cls_name = (
                            class_names[cls_id]
                            if 0 <= cls_id < len(class_names)
                            else f"id_{cls_id}"
                        )
                        det_score = (
                            float(det_scores[det_idx])
                            if det_idx < len(det_scores)
                            else 0.0
                        )

                        x1, y1, x2, y2 = (int(v) for v in bbox)

                        detections.append(
                            DetectionResult(
                                bbox=(x1, y1, x2, y2),
                                score=det_score,
                                cls_id=cls_id,
                                cls_name=cls_name,
                                track_id=raw_track_id,
                                display_track_id=raw_track_id,
                                identity_label=None,
                                identity_score=None,
                                matches=[],
                                predictions=[],
                            )
                        )

                annotated = annotate_frame(frame_bgr.copy(), detections)
                writer.write(annotated)
                processed += 1

                # 可选：保存部分 JPG 检查
                if args.save_jpg and jpg_saved < args.jpg_max_count:
                    if processed % args.jpg_interval == 0:
                        jpg_path = jpg_dir / f"frame_{processed:06d}.jpg"
                        cv2.imwrite(str(jpg_path), annotated)
                        jpg_saved += 1

    writer.release()
    logger.info(
        "Visualization saved to %s (processed %d frames)",
        output_path,
        processed,
    )


if __name__ == "__main__":
    main()
