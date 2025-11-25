"""
Lightweight online ReID stitching within a single video.

Key idea:
- Use YOLO + ByteTrack for tracking as usual.
- Maintain a per-video appearance prototype for each stitched track ID.
- Only run ReID when a *new raw track_id* appears (potential ID switch).
- For old raw track IDs, just reuse the existing stitched ID (no ReID).

This avoids:
- Large external gallery npz.
- Huge similarity matrices.
- Running ReID for every detection in every frame.

Result:
- display_track_id is longer and more stable than raw tracker IDs.
- Resource usage is much lighter, reducing OOM and GPU/desktop crashes.
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
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm

# 这里假设你的 utils.py 里定义了这些函数/类
from utils import (
    DetectionResult,
    build_reid_model,
    load_class_names,
    maybe_resize,
    run_yolo_byteTrack,
    preprocess_patches,
    extract_features,
)

# ------------------ 轨迹颜色：给每个 stitched ID 稳定一个颜色 ------------------

TRACK_COLORS: Dict[int, Tuple[int, int, int]] = {}


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """给每个 display_track_id 分配一个稳定的随机颜色。"""
    if track_id not in TRACK_COLORS:
        rng = np.random.RandomState(track_id & 0xFFFF)
        color = tuple(int(x) for x in rng.randint(50, 255, size=3))
        TRACK_COLORS[track_id] = color
    return TRACK_COLORS[track_id]


# ------------------ 轨迹外观原型（本视频内部） ------------------


class TrackPrototype:
    __slots__ = ("feat", "count", "last_frame", "last_center")

    def __init__(self, feat: torch.Tensor, frame_idx: int, center: Tuple[float, float]):
        """
        feat: 1D tensor (embedding), stored on CPU.
        """
        self.feat = feat.clone().detach().cpu()
        self.count = 1
        self.last_frame = frame_idx
        self.last_center = center

    def update(self, feat: torch.Tensor, frame_idx: int, center: Tuple[float, float]):
        feat = feat.clone().detach().cpu()
        # 简单的 running average
        self.feat = (self.feat * self.count + feat) / (self.count + 1)
        self.count += 1
        self.last_frame = frame_idx
        self.last_center = center


# ------------------ CLI & logger ------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO+ByteTrack with lightweight online ReID stitching (within one video)."
    )
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
        "--yolo-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for YOLO inference (cuda / cuda:0 / cpu).",
    )
    parser.add_argument(
        "--device",
        default="cpu",  # ⚠ 默认用 CPU 做 ReID，尽量减少对 GPU / 桌面的压力
        help="Device for ReID model inference (cpu / cuda / cuda:0).",
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
        "--reid-sim-thres",
        type=float,
        default=0.7,
        help="Cosine similarity threshold to stitch a new track into an existing one.",
    )
    parser.add_argument(
        "--reid-max-gap-frames",
        type=int,
        default=300,
        help="Maximum frame gap allowed when stitching tracks (e.g. 300 @30fps ~10s).",
    )
    parser.add_argument(
        "--reid-interval",
        type=int,
        default=1,
        help="Run ReID only every N processed frames (>=1). "
             "This controls how often *new* raw IDs are stitched.",
    )
    parser.add_argument(
        "--max-new-reid-per-frame",
        type=int,
        default=5,
        help="Maximum number of *new* raw track IDs to run ReID for per processed frame.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity.",
    )
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
    return logging.getLogger("visualize_stitch")


# ------------------ 可视化：画 bbox + stitched ID ------------------


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[DetectionResult],
) -> np.ndarray:
    for det in detections:
        raw_id = det.track_id
        disp_id = det.display_track_id if det.display_track_id is not None else raw_id
        color = get_track_color(disp_id if disp_id is not None else -1)

        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

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


# ------------------ 主流程：YOLO + ByteTrack + 轻量 ReID stitching ------------------


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    device = torch.device(args.device)
    logger.info("Using device for ReID = %s, YOLO device = %s", device, args.yolo_device)

    # 1) 加载模型
    from ultralytics import YOLO

    class_names = load_class_names(args.class_names)
    yolo_model = YOLO(args.yolo_model)
    if args.yolo_device:
        yolo_model.to(args.yolo_device)
    logger.info("Loaded YOLO model from %s on %s", args.yolo_model, args.yolo_device)

    # ReID 这里只当特征提取器用，不需要 num_classes
    reid_model, transform = build_reid_model(
        args.reid_config,
        args.reid_checkpoint,
        num_classes=5,
        device=device,
        logger=logger,
    )
    logger.info("Loaded ReID checkpoint from %s", args.reid_checkpoint)

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
        import shutil

        if jpg_dir.exists():
            shutil.rmtree(jpg_dir)
        jpg_dir.mkdir(parents=True, exist_ok=True)

    # 3) 在线 ReID stitching 状态
    track_prototypes: Dict[int, TrackPrototype] = {}  # stitched_id -> prototype
    track_alias: Dict[int, int] = {}  # raw track_id -> stitched_id

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
                    # 预处理 patch（这一步只做一次，后面 ReID 只用其中一部分）
                    tensors, kept_boxes, kept_indices = preprocess_patches(
                        frame_bgr, boxes, transform
                    )

                    # ------------------ 只对“新 raw_track_id” 准备 ReID ------------------
                    reid_tensors: List[torch.Tensor] = []
                    reid_meta: List[Tuple[int, int, Tuple[float, float], int]] = []
                    new_count = 0

                    for idx, bbox in enumerate(kept_boxes):
                        det_idx = kept_indices[idx]
                        raw_track_id = (
                            int(track_ids[det_idx])
                            if det_idx < len(track_ids)
                            else -1
                        )
                        if raw_track_id == -1:
                            continue
                        # 已经有 alias 的老轨迹：不需要再跑 ReID
                        if raw_track_id in track_alias:
                            continue

                        # 限制每帧最多跑多少个新 ID
                        if new_count >= args.max_new_reid_per_frame:
                            break

                        x1, y1, x2, y2 = bbox
                        cx = 0.5 * (x1 + x2)
                        cy = 0.5 * (y1 + y2)
                        center = (float(cx), float(cy))

                        reid_tensors.append(tensors[idx])
                        reid_meta.append((idx, raw_track_id, center, frame_idx))
                        new_count += 1

                    # ------------------ 只在满足 interval 时，对这些新 ID 跑 ReID ------------------
                    feats_for_new: Dict[int, torch.Tensor] = {}  # raw_track_id -> feat (CPU)
                    run_reid_this_frame = (
                        processed % max(args.reid_interval, 1) == 0
                    )

                    if reid_tensors and run_reid_this_frame:
                        batch = torch.stack(reid_tensors).to(
                            device, non_blocking=True
                        )
                        feats_batch = extract_features(reid_model, batch)  # [M, D]
                        feats_batch = F.normalize(feats_batch, dim=1)
                        feats_batch = feats_batch.cpu()

                        for feat_vec, (idx_in_kept, raw_id, center, fidx) in zip(
                            feats_batch, reid_meta
                        ):
                            feats_for_new[raw_id] = feat_vec

                    # ------------------ 构建 DetectionResult，并在这里做 stitching ------------------
                    for idx, bbox in enumerate(kept_boxes):
                        det_idx = kept_indices[idx]
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

                        x1, y1, x2, y2 = bbox
                        cx = 0.5 * (x1 + x2)
                        cy = 0.5 * (y1 + y2)
                        center = (float(cx), float(cy))

                        display_track_id = raw_track_id

                        # 情况 0：没有 tracker ID，直接用 raw_track_id（一般是 -1）
                        if raw_track_id == -1:
                            pass

                        # 情况 1：老轨迹，之前已经 stitch 过
                        elif raw_track_id in track_alias:
                            display_track_id = track_alias[raw_track_id]
                            # （可选）这里可以按一定间隔更新 prototype，目前先省略

                        # 情况 2：这是一个“新 raw ID”，并且这一帧对它跑了 ReID
                        elif raw_track_id in feats_for_new:
                            feat_vec = feats_for_new[raw_track_id]  # CPU 上的 1D 向量
                            best_id = None
                            best_sim = -1.0

                            # 在已有 stitched 轨迹 prototype 中找最像的
                            for cand_id, proto in track_prototypes.items():
                                # 时间约束：太久没出现的轨迹可以跳过
                                if (
                                    frame_idx - proto.last_frame
                                    > args.reid_max_gap_frames
                                ):
                                    continue

                                # 计算 cosine similarity
                                proto_feat = F.normalize(
                                    proto.feat.unsqueeze(0), dim=1
                                )[0]
                                sim = float(torch.dot(feat_vec, proto_feat).item())
                                if sim > best_sim:
                                    best_sim = sim
                                    best_id = cand_id

                            if best_id is not None and best_sim >= args.reid_sim_thres:
                                # 合并到已有 stitched 轨迹
                                stitched_id = best_id
                                track_alias[raw_track_id] = stitched_id
                                track_prototypes[stitched_id].update(
                                    feat_vec, frame_idx, center
                                )
                                display_track_id = stitched_id
                            else:
                                # 创建新的 stitched 轨迹
                                stitched_id = raw_track_id
                                track_alias[raw_track_id] = stitched_id
                                track_prototypes[stitched_id] = TrackPrototype(
                                    feat_vec, frame_idx, center
                                )
                                display_track_id = stitched_id

                        # 如果这个 raw ID 还没跑过 ReID（因为 interval 或 max_new 限制），就先用 raw_track_id
                        detections.append(
                            DetectionResult(
                                bbox=bbox,
                                score=det_score,
                                cls_id=cls_id,
                                cls_name=cls_name,
                                track_id=raw_track_id,
                                display_track_id=display_track_id,
                                identity_label=None,  # 不使用外部 identity
                                identity_score=None,
                                matches=[],  # 不显示 top-k gallery
                                predictions=[],
                            )
                        )

                    # 4) 画框 & 写视频
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
        args.output,
        processed,
    )


if __name__ == "__main__":
    main()
