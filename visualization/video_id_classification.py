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
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the annotated video will be written.",
    )
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
        "--yolo-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for YOLO inference (cuda / cuda:0 / cpu).",
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


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    device = torch.device(args.device)
    logger.info("Using device=%s", device)

    class_names = load_class_names(args.class_names)
    yolo_model = YOLO(args.yolo_model)
    if args.yolo_device:
        yolo_model.to(args.yolo_device)
    logger.info("Loaded YOLO model from %s on %s", args.yolo_model, args.yolo_device)

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
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    first_frame = maybe_resize(first_frame, args.resize_width)
    writer_height, writer_width = first_frame.shape[:2]

    videoname = Path(args.video).stem + "_idcls.mp4"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / videoname
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (writer_width, writer_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    jpg_dir = output_path.with_suffix("")
    jpg_saved = 0
    if args.save_jpg:
        import shutil

        if jpg_dir.exists():
            shutil.rmtree(jpg_dir)
        jpg_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    bad_frame_count = 0

    if args.max_frames is not None:
        max_frame_idx = min(total_frames, args.max_frames * max(args.frame_skip, 1))
    else:
        max_frame_idx = total_frames

    frame_indices = list(range(0, max_frame_idx, max(args.frame_skip, 1)))

    with torch.no_grad():
        with tqdm(total=len(frame_indices), desc="Frames") as pbar:
            for frame_idx in frame_indices:
                pbar.update(1)

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

                boxes, det_scores, cls_ids = run_yolo(
                    yolo_model,
                    frame_bgr,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    max_dets=args.max_dets,
                    device=args.yolo_device,
                )

                detections: List[DetectionResult] = []

                if boxes.size != 0:
                    tensors, kept_boxes, kept_indices = preprocess_patches(
                        frame_bgr, boxes, transform
                    )
                    if tensors:
                        batch = torch.stack(tensors).to(device)
                        logits = identity_forward(classifier, batch, is_temporal)
                        probs = torch.softmax(logits, dim=1)
                        top_k = min(args.top_k, probs.shape[1])
                        top_scores, top_indices = torch.topk(probs, k=top_k, dim=1)

                        for idx, bbox in enumerate(kept_boxes):
                            det_idx = kept_indices[idx]
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
                            preds = []
                            for rank in range(top_scores.shape[1]):
                                class_idx = int(top_indices[idx, rank].item())
                                prob = float(top_scores[idx, rank].item())
                                label_name = (
                                    id_class_names[class_idx]
                                    if class_idx < len(id_class_names)
                                    else f"id_{class_idx}"
                                )
                                preds.append((label_name, prob))
                            x1, y1, x2, y2 = (int(v) for v in bbox)
                            detections.append(
                                DetectionResult(
                                    bbox=(x1, y1, x2, y2),
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

                    annotated = annotate_frame(frame_bgr.copy(), detections, label_to_color)
                    writer.write(annotated)
                    processed += 1
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
