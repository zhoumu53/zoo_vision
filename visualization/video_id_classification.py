"""
Video visualization script for Elephant identity classification.

The script loads:
  1. A video file
  2. A YOLO segmentation/detection model (Ultralytics)
  3. A pretrained identity classification model checkpoint (e.g., zoo_id_gru)

Pipeline:
  Video frame -> YOLO detection -> crop patches -> Identity classifier -> draw annotations -> save annotated video.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu, gpu
from utils import *

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - guard for missing dependency
    raise ImportError(
        "Ultralytics is required for visualization. Install via `pip install ultralytics`."
    ) from exc


from training.identity.model import get_model  # noqa: E402


def build_default_identity_names(num_classes: int) -> List[str]:
    names = list(DEFAULT_IDENTITY_NAMES)
    if num_classes <= len(names):
        return names[:num_classes]
    extra = [f"id_{idx+1}" for idx in range(len(names), num_classes)]
    return names + extra



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
        "--id-checkpoint",
        required=True,
        help="Path to identity classifier checkpoint (.pth).",
    )
    parser.add_argument(
        "--id-model-name",
        default=None,
        help="Model architecture name (e.g., zoo_id_gru, densenet121). "
        "If omitted, inferred from checkpoint args.",
    )
    parser.add_argument(
        "--id-input-size",
        type=int,
        default=224,
        help="Input resolution for identity classifier (square).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference.",
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


def build_identity_model(
    checkpoint_path: str,
    model_name: str | None,
    device: torch.device,
    input_size: int,
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, T.Compose, bool]:
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    suffix = checkpoint_path.suffix.lower()

    if suffix == ".ptc":
        logger.info("Loading TorchScript identity model from %s", checkpoint_path)
        model = torch.jit.load(checkpoint_path, map_location=device)
        model.eval()
        model.to(device)
        transform = T.Compose(
            [
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return model, transform, False

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    checkpoint_args = checkpoint.get("args")

    if model_name is None:
        if isinstance(checkpoint_args, dict):
            model_name = checkpoint_args.get("model")
        elif checkpoint_args is not None:
            model_name = getattr(checkpoint_args, "model", None)

    if model_name is None:
        if any("gru" in k for k in state_dict.keys()):
            model_name = "zoo_id_gru"
        else:
            model_name = "densenet121"
        logger.info("Inferred identity model type: %s", model_name)

    classifier_weight_key = None
    for key in state_dict.keys():
        if key.endswith("classifier.weight"):
            classifier_weight_key = key
            break
    if classifier_weight_key is None:
        raise RuntimeError("Checkpoint does not contain classifier weights.")

    num_classes = state_dict[classifier_weight_key].shape[0]
    model = get_model(model_name, num_classes=num_classes, weights=None)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    transform = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    is_temporal = hasattr(model, "gru")
    return model, transform, is_temporal


def identity_forward(
    model: torch.nn.Module,
    batch: torch.Tensor,
    is_temporal: bool,
) -> torch.Tensor:
    if is_temporal:
        outputs = model(batch.unsqueeze(1))
        logits = outputs["logits"]
        if logits.dim() == 3:
            logits = logits.squeeze(1)
    else:
        outputs = model(batch)
        if isinstance(outputs, dict):
            if "logits" not in outputs:
                raise ValueError("Model output dictionary does not contain 'logits'")
            logits = outputs["logits"]
        else:
            logits = outputs
    return logits


def infer_classifier_dim(
    model: torch.nn.Module,
    input_size: int,
    device: torch.device,
    is_temporal: bool,
) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_size, input_size, device=device)
        logits = identity_forward(model, dummy, is_temporal)
        if logits.dim() == 1:
            return logits.numel()
        return logits.shape[-1]


def run_yolo(
    model: YOLO,
    frame: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    max_dets: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run YOLO inference on a frame and return xyxy boxes, scores, class ids."""
    results = model.predict(
        source=frame,
        conf=conf_thres,
        iou=iou_thres,
        retina_masks=False,
        verbose=False,
        max_det=max_dets,
    )
    if not results:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    res = results[0]
    if res.boxes is None or res.boxes.data.shape[0] == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    return boxes, scores, cls_ids


def preprocess_patches(
    frame: np.ndarray,
    boxes: np.ndarray,
    transform: T.Compose,
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]], List[int]]:
    patches = []
    kept_boxes = []
    kept_indices: List[int] = []
    h, w = frame.shape[:2]

    for det_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = transform(pil_img)
        patches.append(tensor)
        kept_boxes.append((x1, y1, x2, y2))
        kept_indices.append(det_idx)
    return patches, kept_boxes, kept_indices


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[DetectionResult],
    label_colors: Dict[str, Tuple[int, int, int]],
) -> np.ndarray:
    for det in detections:
        color = label_colors.get(det.predictions[0][0], (0, 255, 0)) if det.predictions else (0, 255, 0)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        label = det.predictions[0][0] if det.predictions else "Unknown"
        score = det.predictions[0][1] if det.predictions else 0.0
        text = f"{label} ({score:.2f}) | det {det.score:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        for idx, (match_label, match_score) in enumerate(det.predictions[1:], start=1):
            caption = f"{idx+1}. {match_label} {match_score:.2f}"
            cv2.putText(
                frame,
                caption,
                (x1, y1 + 20 + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return frame


def maybe_resize(frame: np.ndarray, target_width: int | None) -> np.ndarray:
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / float(w)
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    device = torch.device(args.device)
    logger.info("Using device=%s", device)

    class_names = load_class_names(args.class_names)
    yolo_model = YOLO(args.yolo_model)
    logger.info("Loaded YOLO model from %s", args.yolo_model)

    classifier, transform, is_temporal = build_identity_model(
        checkpoint_path=args.id_checkpoint,
        model_name=args.id_model_name,
        device=device,
        logger=logger,
        input_size=args.id_input_size,
    )
    logger.info("Loaded identity classifier from %s", args.id_checkpoint)
    classifier_dim = infer_classifier_dim(classifier, args.id_input_size, device, is_temporal)
    id_class_names = build_default_identity_names(classifier_dim)
    if classifier_dim > len(DEFAULT_IDENTITY_NAMES):
        logger.warning(
            "Classifier outputs %d classes, more than default %d. "
            "Auto-generated placeholder names for the remaining classes.",
            classifier_dim,
            len(DEFAULT_IDENTITY_NAMES),
        )
    label_to_color = build_label_color_map(id_class_names)

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
                logits = identity_forward(classifier, batch, is_temporal)
                probs = torch.softmax(logits, dim=1)
                top_k = min(args.top_k, probs.shape[1])
                top_scores, top_indices = torch.topk(probs, k=top_k, dim=1)

                detections: List[DetectionResult] = []
                for idx, bbox in enumerate(kept_boxes):
                    det_idx = kept_indices[idx]
                    cls_id = cls_ids[det_idx] if det_idx < len(cls_ids) else -1
                    cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"id_{cls_id}"
                    det_score = float(det_scores[det_idx]) if det_idx < len(det_scores) else 0.0
                    preds = []
                    for rank in range(top_scores.shape[1]):
                        class_idx = int(top_indices[idx, rank].item())
                        prob = float(top_scores[idx, rank].item())
                        label_name = id_class_names[class_idx] if class_idx < len(id_class_names) else f"id_{class_idx}"
                        preds.append((label_name, prob))
                    detections.append(
                        DetectionResult(
                            bbox=bbox,
                            score=det_score,
                            cls_id=cls_id,
                            cls_name=cls_name,
                            predictions=preds,
                            track_id=None,
                            display_track_id=None,
                            identity_label=None,
                            identity_score=None,
                            matches=[],
                        )
                    )

                annotated = annotate_frame(frame.copy(), detections, label_to_color)
                writer.write(annotated)
                processed += 1

    writer.release()
    logger.info("Visualization saved to %s (processed %d frames)", args.output, processed)


if __name__ == "__main__":
    main()
