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

from project.config import cfg as base_cfg  # noqa: E402
from project.datasets.make_dataloader import get_transforms  # noqa: E402
from project.models import make_model  # noqa: E402
from project.utils.tools import load_model  # noqa: E402


@dataclass
class GalleryDB:
    """Holds gallery features and metadata."""

    features: torch.Tensor
    labels: np.ndarray
    ids: np.ndarray
    paths: np.ndarray
    label_to_color: Dict[str, Tuple[int, int, int]]


@dataclass
class DetectionResult:
    """Structure storing a single detection + reid match information."""

    bbox: Tuple[int, int, int, int]
    score: float
    cls_id: int
    cls_name: str
    matches: List[Tuple[str, float]]


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


def load_class_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {path}")
    return names


def hash_color(label: str) -> Tuple[int, int, int]:
    palette = [
        (255, 99, 71),
        (50, 205, 50),
        (65, 105, 225),
        (255, 215, 0),
        (0, 128, 255),
        (154, 205, 50),
    ]
    idx = hash(label) % len(palette)
    r, g, b = palette[idx]
    return int(b), int(g), int(r)


def load_gallery_database(path: str, device: torch.device) -> GalleryDB:
    npz = np.load(path, allow_pickle=True)
    required_keys = {"feature", "label", "id", "path"}
    if not required_keys.issubset(set(npz.files)):
        raise KeyError(f"Gallery file missing keys: expected {required_keys}, found {set(npz.files)}")

    features = torch.from_numpy(npz["feature"]).float()
    features = torch.nn.functional.normalize(features, dim=1)
    features = features.to(device)

    labels = npz["label"]
    ids = npz["id"]
    paths = npz["path"]
    unique_labels = sorted(set(labels.tolist()))
    label_to_color = {label: hash_color(label) for label in unique_labels}
    return GalleryDB(
        features=features,
        labels=labels,
        ids=ids,
        paths=paths,
        label_to_color=label_to_color,
    )


def build_reid_model(
    config_path: str,
    checkpoint_path: str,
    num_classes: int,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, T.Compose]:
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(config_path)
    cfg.TEST.WEIGHT = checkpoint_path
    cfg.freeze()

    model = make_model(
        cfg,
        num_classes=num_classes,
        logger=logger,
        return_feature=True,
        device=device,
        camera_num=0,
        view_num=0,
    )
    model = load_model(
        model,
        checkpoint_path,
        logger=logger,
        remove_fc=True,
        local_rank=0,
        is_swin=("swin" in cfg.MODEL.TYPE),
    )
    model.eval()
    model.to(device)
    preprocess = get_transforms(cfg, is_train=False)
    return model, preprocess


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


def extract_features(
    model: torch.nn.Module,
    batch: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(batch)

    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 4:
            feat = outputs[2]
        elif len(outputs) > 1:
            feat = outputs[1]
        else:
            feat = outputs[0]
    else:
        feat = outputs
    feat = torch.nn.functional.normalize(feat, dim=1)
    return feat


def match_gallery(
    query_feats: torch.Tensor,
    gallery: GalleryDB,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sims = query_feats @ gallery.features.t()
    k = min(top_k, sims.shape[1])
    scores, indices = torch.topk(sims, k=k, dim=1)
    return scores.cpu(), indices.cpu()


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[DetectionResult],
    gallery: GalleryDB,
) -> np.ndarray:
    for det in detections:
        color = gallery.label_to_color.get(det.matches[0][0], (0, 255, 0)) if det.matches else (0, 255, 0)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        label = det.matches[0][0] if det.matches else "Unknown"
        score = det.matches[0][1] if det.matches else 0.0
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
        for idx, (match_label, match_score) in enumerate(det.matches[1:], start=1):
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

    # 计算要处理的最大帧数
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
                        )
                    )

                annotated = annotate_frame(frame.copy(), detections, gallery)
                writer.write(annotated)
                processed += 1

    writer.release()
    logger.info("Visualization saved to %s (processed %d frames)", args.output, processed)


if __name__ == "__main__":
    main()
