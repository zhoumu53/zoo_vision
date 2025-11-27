from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timedelta

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from utils import (
    DetectionResult,
    build_reid_model,
    load_class_names,
    maybe_resize,
    run_yolo_byteTrack,
    preprocess_patches,
    extract_features,
)
from visualization.tracklet_extraction import extract_tracklets


# ------------------ Stable visual IDs & colors ------------------

CANONICAL_TO_VISUAL: Dict[int, int] = {}
NEXT_VISUAL_ID: int = 1
TRACK_COLORS: Dict[int, Tuple[int, int, int]] = {}


def get_or_create_visual_id(canonical_id: int | None) -> int:
    """
    Map stitched (canonical) ID to a dense visual ID: 1..N.

    - Only IDs that actually show up in detections get a visual ID.
    - If canonical_id is -1 or None, return -1.
    """
    global NEXT_VISUAL_ID

    if canonical_id is None or canonical_id == -1:
        return -1

    if canonical_id not in CANONICAL_TO_VISUAL:
        CANONICAL_TO_VISUAL[canonical_id] = NEXT_VISUAL_ID
        NEXT_VISUAL_ID += 1

    return CANONICAL_TO_VISUAL[canonical_id]


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Assign a stable pseudo-random color to each display_track_id."""
    if track_id not in TRACK_COLORS:
        rng = np.random.RandomState(track_id & 0xFFFF)
        color = tuple(int(x) for x in rng.randint(50, 255, size=3))
        TRACK_COLORS[track_id] = color
    return TRACK_COLORS[track_id]


# ------------------ Per-video feature hub (memory-optimized) ------------------

class TrackFeatureHub:
    """
    Per-video feature gallery.

    Stores sparse snapshots:
      - feats:  [N, D] float32
      - ids:    [N]   int64  (canonical track IDs)
      - frames: [N]   int64

    This version is memory-friendlier:
      - Only snapshot canonical prototypes every K seconds (configurable).
      - Keeps Python lists as the source of truth.
      - Uses cached NumPy arrays to avoid re-stacking on every match.
      - Supports pruning old entries by frame gap.
    """

    def __init__(self, feat_dim: int):
        self.feats: List[np.ndarray] = []
        self.ids: List[int] = []
        self.frames: List[int] = []
        self.feat_dim = feat_dim

        # Cached numpy arrays
        self._feats_np: Optional[np.ndarray] = None
        self._ids_np: Optional[np.ndarray] = None
        self._frames_np: Optional[np.ndarray] = None
        self._dirty: bool = False

    def add(self, track_id: int, feat: torch.Tensor, frame_idx: int) -> None:
        """
        Add a single feature snapshot (already on CPU & typically normalized).
        track_id is the canonical track id here.
        """
        f = feat.detach().cpu().float().numpy().reshape(-1)
        if f.shape[0] != self.feat_dim:
            raise ValueError(
                f"Feature dim mismatch: got {f.shape[0]}, expected {self.feat_dim}"
            )
        self.feats.append(f)
        self.ids.append(int(track_id))
        self.frames.append(int(frame_idx))
        self._dirty = True

    def _refresh_cache(self) -> None:
        """Rebuild cached NumPy arrays if dirty."""
        if not self._dirty:
            return
        if not self.feats:
            self._feats_np = np.zeros((0, self.feat_dim), dtype=np.float32)
            self._ids_np = np.zeros((0,), dtype=np.int64)
            self._frames_np = np.zeros((0,), dtype=np.int64)
        else:
            self._feats_np = np.stack(self.feats, axis=0).astype(np.float32)
            self._ids_np = np.array(self.ids, dtype=np.int64)
            self._frames_np = np.array(self.frames, dtype=np.int64)
        self._dirty = False

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._refresh_cache()
        return (
            self._feats_np,
            self._ids_np,
            self._frames_np,
        )

    def save(self, path: Path) -> None:
        feats, ids, frames = self.as_arrays()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feats=feats,
            ids=ids,
            frames=frames,
        )

    def prune(self, current_frame: int, max_gap_frames: int) -> None:
        """
        Drop features whose frame index is older than current_frame - max_gap_frames.

        This bounds memory over long videos: only recent history is kept.
        """
        if not self.feats:
            return

        feats_np, ids_np, frames_np = self.as_arrays()
        valid_mask = (current_frame - frames_np) <= max_gap_frames
        if valid_mask.all():
            return

        # Rebuild Python lists with only valid entries
        self.feats = [f for f, v in zip(self.feats, valid_mask) if v]
        self.ids = [i for i, v in zip(self.ids, valid_mask) if v]
        self.frames = [fr for fr, v in zip(self.frames, valid_mask) if v]
        self._dirty = True

    def match_best(
        self,
        feat: torch.Tensor,
        top_k: int = 1,
        chunk_size: int = 4096,
    ) -> Tuple[int | None, float]:
        """
        Match a single feature against all history in the hub.

        Returns:
          best_id, best_sim
        where best_id is the canonical track id stored in the hub.
        If hub empty, returns (None, -1.0).
        Assumes all feats are L2-normalized; feat should also be normalized.
        """
        feats, ids, frames = self.as_arrays()
        if feats.shape[0] == 0:
            return None, -1.0

        q = feat.detach().cpu().float().numpy().reshape(-1)
        best_sim = -1.0
        best_id: int | None = None

        N = feats.shape[0]
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = feats[start:end]  # [B, D]
            sims = np.dot(chunk, q)  # cosine similarity
            idx = int(np.argmax(sims))
            sim = float(sims[idx])
            if sim > best_sim:
                best_sim = sim
                best_id = int(ids[start + idx])

        return best_id, best_sim


class TrackPrototype:
    """
    Per-stitched (canonical) track running prototype.

    Stores:
      - feat: averaged embedding on CPU
      - count: how many updates
      - last_frame: last frame index where it was seen
      - last_center: last (cx, cy) center in image
    """
    __slots__ = ("feat", "count", "last_frame", "last_center")

    def __init__(self, feat: torch.Tensor, frame_idx: int, center: Tuple[float, float]):
        feat = feat.clone().detach().cpu()
        self.feat = feat
        self.count = 1
        self.last_frame = frame_idx
        self.last_center = center

    def update(self, feat: torch.Tensor, frame_idx: int, center: Tuple[float, float]):
        feat = feat.clone().detach().cpu()
        self.feat = (self.feat * self.count + feat) / (self.count + 1)
        self.count += 1
        self.last_frame = frame_idx
        self.last_center = center


# ------------------ Behaviour classifier helpers ------------------

def resolve_behavior_model_dir(path_str: str, logger: logging.Logger) -> Path:
    """
    Accepts either a HuggingFace directory, a config.json file, or a TorchScript file
    that lives alongside HuggingFace assets. Returns the directory that AutoModel can load.
    """
    path = Path(path_str).expanduser()
    if path.is_dir():
        return path
    if path.is_file():
        if path.name == "config.json":
            return path.parent
        if path.suffix == ".ptc":
            parent = path.parent
            config_path = parent / "config.json"
            if config_path.exists():
                logger.info(
                    "Resolved TorchScript checkpoint %s to HuggingFace directory %s for behaviour inference.",
                    path,
                    parent,
                )
                return parent
    raise FileNotFoundError(
        f"Unable to resolve behaviour model from {path_str}. "
        f"Provide a directory or config.json exported by HuggingFace."
    )


def load_behavior_model(
    path_str: str,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[AutoModelForImageClassification, AutoImageProcessor, Dict[int, str]]:
    model_dir = resolve_behavior_model_dir(path_str, logger)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    logger.info(
        "Loaded behaviour classifier from %s (%d classes) on %s",
        model_dir,
        len(id2label),
        device,
    )
    return model, processor, id2label


def parse_video_start_datetime(video_path: str) -> Optional[datetime]:
    """
    Try to parse start datetime from a video filename like:
    ZAG-ELP-CAM-016-20240905-224718-XXXX.mp4 -> 2024-09-05 22:47:18
    """
    stem = Path(video_path).stem
    parts = stem.split("-")
    if len(parts) >= 6:
        date_str = parts[4]
        time_str = parts[5]
        try:
            return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        except ValueError:
            return None
    return None


def prepare_behavior_crops(
    frame_bgr: np.ndarray,
    detections: Sequence[DetectionResult],
    context: float,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Crop detection boxes with a bit of context and convert to RGB uint8 arrays for the classifier.
    """
    h, w = frame_bgr.shape[:2]
    context = max(context, 1.0)
    crops: List[np.ndarray] = []
    det_indices: List[int] = []

    for idx, det in enumerate(detections):
        if det.bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half_w = 0.5 * width * context
        half_h = 0.5 * height * context
        new_x1 = max(0, int(np.floor(cx - half_w)))
        new_y1 = max(0, int(np.floor(cy - half_h)))
        new_x2 = min(w, int(np.ceil(cx + half_w)))
        new_y2 = min(h, int(np.ceil(cy + half_h)))
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            continue
        crop = frame_bgr[new_y1:new_y2, new_x1:new_x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(crop_rgb)
        det_indices.append(idx)
    return crops, det_indices


def run_behavior_inference(
    model: AutoModelForImageClassification,
    processor: AutoImageProcessor,
    device: torch.device,
    crops: Sequence[np.ndarray],
    id2label: Dict[int, str],
    batch_size: int,
) -> List[Tuple[str, float]]:
    predictions: List[Tuple[str, float]] = []
    effective_batch = max(batch_size, 1)
    for start in range(0, len(crops), effective_batch):
        batch = list(crops[start : start + effective_batch])
        if not batch:
            continue
        encoded = processor(images=batch, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            max_probs, max_ids = torch.max(probs, dim=-1)
        for prob, idx in zip(max_probs, max_ids):
            label = id2label.get(int(idx.item()), str(int(idx.item())))
            predictions.append((label, float(prob.item())))
    return predictions


# ------------------ CLI & logger ------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO+ByteTrack with lightweight online ReID stitching (within one video) + behaviour labels."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Directory to save outputs (video, jsonl, hub, etc.).")
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
        default="cpu",
        help="Device for ReID model inference (cpu / cuda / cuda:0).",
    )
    parser.add_argument("--conf-thres", type=float, default=0.65, help="YOLO confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.65, help="YOLO IOU threshold.")
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
        default=3000,
        help="Maximum frame gap allowed when stitching tracks (e.g. 300 @30fps ~10s). Also used to prune hub.",
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
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Directory to dump annotated frames (default: <output>/<video_name>_.../frames).",
    )
    parser.add_argument(
        "--tracks-json",
        default=None,
        help="Optional JSONL file to record per-frame tracks for downstream analysis / Streamlit.",
    )
    parser.add_argument(
        "--online-reid-from-hub",
        action="store_true",
        help="If set, match new tracks against the history stored in the feature hub "
             "in addition to in-memory prototypes.",
    )
    parser.add_argument(
        "--hub-chunk-size",
        type=int,
        default=4096,
        help="Chunk size along gallery dimension when matching against the hub.",
    )
    parser.add_argument(
        "--hub-snapshot-interval-seconds",
        type=float,
        default=5.0,
        help="Interval (in seconds) between feature snapshots per canonical track into the hub. "
             "Larger value => smaller hub => less CPU memory.",
    )
    parser.add_argument(
        "--hub-prune-interval-frames",
        type=int,
        default=10000,
        help="How often (in frames) to prune old entries from the hub by reid-max-gap-frames.",
    )
    parser.add_argument(
        "--no-new-stitching",
        action="store_true",
        help="Disable lightweight ReID stitching and keep raw YOLO+ByteTrack track IDs only.",
    )
    parser.add_argument(
        "--behavior-model",
        required=True,
        help="Path to the behaviour classifier directory (HuggingFace) or its config.json / config.ptc file.",
    )
    parser.add_argument(
        "--behavior-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for behaviour classifier inference (cpu / cuda / cuda:0).",
    )
    parser.add_argument(
        "--behavior-batch-size",
        type=int,
        default=16,
        help="Batch size for behaviour inference.",
    )
    parser.add_argument(
        "--behavior-context",
        type=float,
        default=1.1,
        help="Context scale applied around each bbox before cropping for behaviour classification.",
    )
    parser.add_argument(
        "--save-tracklets",
        action="store_true",
        help="Save tracklet images for each tracked object.",
    )
    parser.set_defaults(save_tracklets=True)

    return parser.parse_args(args=argv)


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("visualize_stitch")


# ------------------ Visualization ------------------

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

        behaviour_text = ""
        if det.predictions:
            top_label, top_prob = det.predictions[0]
            # top_label e.g. "00_sleeping" -> display "sleeping"
            behaviour_text = f" | {top_label[3:]} ({top_prob:.2f})"
        score_text = f"{det.score:.2f}" if det.score is not None else "nan"
        text = f"ID {disp_id}{behaviour_text} | det {score_text}"
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


def _ensure_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _ensure_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ensure_json_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class TrackJSONLogger:
    """Write per-frame tracking results to a JSONL file for labeling/analysis."""

    def __init__(
        self,
        path: Path,
        video: str,
        fps: float,
        width: int,
        height: int,
        class_names: List[str],
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w", encoding="utf-8")
        meta = {
            "video": video,
            "fps": fps,
            "width": width,
            "height": height,
            "classes": class_names,
        }
        self.fp.write(json.dumps({"meta": _ensure_json_serializable(meta)}) + "\n")

    def log_frame(
        self,
        frame_idx: int,
        tracks: List[Dict],
        frame_timestamp: str | None = None,
    ) -> None:
        payload = {
            "frame_idx": int(frame_idx),
            "tracks": [_ensure_json_serializable(track) for track in tracks],
        }
        if frame_timestamp is not None:
            payload["timestamp"] = str(frame_timestamp)
        self.fp.write(json.dumps({"results": payload}) + "\n")

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass


# ------------------ Main pipeline: YOLO + ByteTrack + lightweight ReID stitching ------------------

def run_tracking(
    args: argparse.Namespace,
    frame_callback: Callable[[int, np.ndarray, List[Dict[str, Any]]], None] | None = None,
) -> None:
    logger = setup_logger(args.log_level)

    device = torch.device(args.device)
    logger.info("Using device for ReID = %s, YOLO device = %s", device, args.yolo_device)

    from ultralytics import YOLO

    # Load YOLO
    class_names = load_class_names(args.class_names)
    yolo_model = YOLO(args.yolo_model)
    if args.yolo_device:
        yolo_model.to(args.yolo_device)
    logger.info("Loaded YOLO model from %s on %s", args.yolo_model, args.yolo_device)

    # Load behaviour classifier
    behaviour_device = torch.device(args.behavior_device)
    behaviour_model, behaviour_processor, behaviour_id2label = load_behavior_model(
        args.behavior_model, behaviour_device, logger
    )

    # ReID model & feature hub
    reid_model = None
    transform = None
    feature_hub: TrackFeatureHub | None = None

    if args.no_new_stitching:
        logger.info("no-new-stitching enabled: skipping ReID model and stitching logic.")
    else:
        # Build ReID model once, used only as feature extractor
        reid_model, transform = build_reid_model(
            args.reid_config,
            args.reid_checkpoint,
            num_classes=5,
            device=device,
            logger=logger,
        )
        logger.info("Loaded ReID checkpoint from %s", args.reid_checkpoint)

        reid_model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            dummy_feat = extract_features(reid_model, dummy)
            feat_dim = int(dummy_feat.shape[1])

        # Only allocate a feature hub if we actually want hub-based matching
        if args.online_reid_from_hub:
            feature_hub = TrackFeatureHub(feat_dim=feat_dim)
            logger.info("Online ReID from hub is enabled (feat_dim=%d)", feat_dim)
        else:
            logger.info("online-reid-from-hub is disabled: feature hub will not be used.")

    # Open video with decord
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

    video_start_dt = parse_video_start_datetime(args.video)

    # Get first frame for size & writer
    first_frame = vr[0].asnumpy()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    first_frame = maybe_resize(first_frame, args.resize_width)
    height, width = first_frame.shape[:2]

    # Prepare output dirs
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    subdir_name = f"{Path(args.video).stem}_max{args.max_frames if args.max_frames else 'all'}"
    output_dir = output_root / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    video_output_path = output_dir / f"{Path(args.video).stem}_tracks.mp4"
    writer = cv2.VideoWriter(
        str(video_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_output_path}")

    # JPG frames dir
    jpg_dir = output_dir / "frames"
    if args.frames_dir:
        jpg_dir = Path(args.frames_dir)
    jpg_saved = 0
    if args.save_jpg or frame_callback is not None:
        import shutil
        if jpg_dir.exists():
            shutil.rmtree(jpg_dir)
        jpg_dir.mkdir(parents=True, exist_ok=True)

    # Online stitching state
    track_prototypes: Dict[int, TrackPrototype] = {}   # canonical_id -> prototype
    track_alias: Dict[int, int] = {}                  # raw_track_id -> canonical_id
    last_hub_update_frame: Dict[int, int] = {}        # canonical_id -> last frame snapshot into hub

    processed = 0
    rendered_frames = 0
    bad_frame_count = 0

    # Frame count to process
    if args.max_frames is not None:
        max_frame_idx = min(total_frames, args.max_frames)
    else:
        max_frame_idx = total_frames

    logger.info(
        "Will process %d / %d frames (visualization interval=%d, max_frames=%s)",
        max_frame_idx,
        total_frames,
        args.frame_skip,
        args.max_frames,
    )

    # JSON logger
    tracks_json_path = Path(args.tracks_json) if args.tracks_json else video_output_path.with_suffix(".jsonl")
    track_logger: TrackJSONLogger | None = None
    try:
        track_logger = TrackJSONLogger(
            path=tracks_json_path,
            video=args.video,
            fps=fps,
            width=width,
            height=height,
            class_names=class_names,
        )
        logger.info("Per-frame tracks will be logged to %s", tracks_json_path)
    except Exception as exc:
        logger.warning("Failed to open track logger at %s: %s", tracks_json_path, exc)
        track_logger = None

    # Convert snapshot interval (seconds) -> frames
    hub_snapshot_interval_frames = max(1, int(args.hub_snapshot_interval_seconds * fps))

    with torch.no_grad():
        with tqdm(total=max_frame_idx, desc="Frames") as pbar:
            for frame_idx in range(0, max_frame_idx):
                pbar.update(1)
                render_frame = (frame_idx % max(args.frame_skip, 1) == 0)

                # Periodic pruning of the feature hub (if any)
                if (
                    feature_hub is not None
                    and args.reid_max_gap_frames > 0
                    and frame_idx > 0
                    and frame_idx % max(args.hub_prune_interval_frames, 1) == 0
                ):
                    ### TODO -- prune by similarity
                    feature_hub.prune(current_frame=frame_idx, max_gap_frames=args.reid_max_gap_frames)

                # Read frame via decord
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
                timestamp_s = frame_idx / max(fps, 1e-6)

                if video_start_dt is not None:
                    frame_dt = video_start_dt + timedelta(seconds=timestamp_s)
                else:
                    frame_dt = datetime.now()
                frame_timestamp_str = frame_dt.strftime("%Y%m%d_%H%M%S")
                frame_name = f"{Path(args.video).stem}_{frame_timestamp_str}_{frame_idx:06d}.jpg"

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
                tracks_for_json: List[Dict[str, Any]] = []

                if boxes.size != 0:
                    # ------------------ Case 1: no stitching, raw YOLO+ByteTrack IDs only ------------------
                    if args.no_new_stitching:
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
                            canonical_id = raw_track_id

                            detections.append(
                                DetectionResult(
                                    bbox=tuple(int(v) for v in bbox),
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
                            tracks_for_json.append(
                                {
                                    "raw_track_id": raw_track_id,
                                    "canonical_track_id": canonical_id,
                                    "display_track_id": raw_track_id,
                                    "frame_name": frame_name,
                                    "bbox": [int(v) for v in bbox],
                                    "score": det_score,
                                    "cls_id": cls_id,
                                    "cls_name": cls_name,
                                    "behavior": None,
                                }
                            )

                    # ------------------ Case 2: lightweight stitching with ReID features ------------------
                    else:
                        if reid_model is None or transform is None:
                            raise RuntimeError("ReID model/transform missing while stitching is enabled.")

                        # Preprocess patches once
                        tensors, kept_boxes, kept_indices = preprocess_patches(
                            frame_bgr, boxes, transform
                        )

                        # Prepare new raw IDs for ReID
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
                            if raw_track_id in track_alias:
                                # already stitched; no need to run ReID again
                                continue

                            if new_count >= args.max_new_reid_per_frame:
                                break

                            x1, y1, x2, y2 = bbox
                            cx = 0.5 * (x1 + x2)
                            cy = 0.5 * (y1 + y2)
                            center = (float(cx), float(cy))

                            reid_tensors.append(tensors[idx])
                            reid_meta.append((idx, raw_track_id, center, frame_idx))
                            new_count += 1

                        # Only run ReID on some frames (interval)
                        feats_for_new: Dict[int, torch.Tensor] = {}  # raw_track_id -> feat (CPU)
                        run_reid_this_frame = processed % max(args.reid_interval, 1) == 0

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

                        # Build DetectionResult + stitching
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
                            canonical_id = raw_track_id

                            # Case 0: no tracker ID
                            if raw_track_id == -1:
                                pass

                            # Case 1: existing raw track alias
                            elif raw_track_id in track_alias:
                                canonical_id = track_alias[raw_track_id]
                                visual_id = get_or_create_visual_id(canonical_id)
                                display_track_id = visual_id

                            # Case 2: new raw ID with freshly computed ReID feature
                            elif raw_track_id in feats_for_new:
                                feat_vec = feats_for_new[raw_track_id]  # CPU, normalized
                                best_id = None
                                best_sim = -1.0

                                # Mode A: match against hub history (if enabled)
                                if args.online_reid_from_hub and feature_hub is not None:
                                    hub_best_id, hub_best_sim = feature_hub.match_best(
                                        feat_vec,
                                        top_k=1,
                                        chunk_size=args.hub_chunk_size,
                                    )
                                    best_id = hub_best_id
                                    best_sim = hub_best_sim

                                # Mode B: match against prototypes
                                if best_id is None or best_sim < args.reid_sim_thres:
                                    for cand_id, proto in track_prototypes.items():
                                        if (
                                            frame_idx - proto.last_frame
                                            > args.reid_max_gap_frames
                                        ):
                                            continue
                                        proto_norm = F.normalize(
                                            proto.feat.unsqueeze(0), dim=1
                                        )[0]
                                        sim = float(torch.dot(feat_vec, proto_norm).item())
                                        if sim > best_sim:
                                            best_sim = sim
                                            best_id = cand_id

                                # Decide stitching
                                if best_id is not None and best_sim >= args.reid_sim_thres:
                                    canonical_id = best_id
                                    track_alias[raw_track_id] = canonical_id
                                    if canonical_id not in track_prototypes:
                                        track_prototypes[canonical_id] = TrackPrototype(
                                            feat_vec, frame_idx, center
                                        )
                                    else:
                                        track_prototypes[canonical_id].update(
                                            feat_vec, frame_idx, center
                                        )
                                else:
                                    # New stitched track
                                    canonical_id = raw_track_id
                                    track_alias[raw_track_id] = canonical_id
                                    track_prototypes[canonical_id] = TrackPrototype(
                                        feat_vec, frame_idx, center
                                    )

                                visual_id = get_or_create_visual_id(canonical_id)
                                display_track_id = visual_id

                                # Sparse feature hub snapshot: one snapshot per canonical track every K seconds
                                if args.online_reid_from_hub and feature_hub is not None:
                                    proto = track_prototypes[canonical_id]
                                    last_snapshot_frame = last_hub_update_frame.get(
                                        canonical_id, -10**9
                                    )
                                    if frame_idx - last_snapshot_frame >= hub_snapshot_interval_frames:
                                        proto_feat_norm = F.normalize(
                                            proto.feat.unsqueeze(0), dim=1
                                        )[0]
                                        feature_hub.add(
                                            track_id=canonical_id,
                                            feat=proto_feat_norm,
                                            frame_idx=frame_idx,
                                        )
                                        last_hub_update_frame[canonical_id] = frame_idx

                            # Build detection object
                            detections.append(
                                DetectionResult(
                                    bbox=tuple(int(v) for v in bbox),
                                    score=det_score,
                                    cls_id=cls_id,
                                    cls_name=cls_name,
                                    track_id=raw_track_id,
                                    display_track_id=display_track_id,
                                    identity_label=None,
                                    identity_score=None,
                                    matches=[],
                                    predictions=[],
                                )
                            )
                            tracks_for_json.append(
                                {
                                    "raw_track_id": raw_track_id,
                                    "canonical_track_id": canonical_id,
                                    "display_track_id": display_track_id,
                                    "frame_name": frame_name,
                                    "bbox": [int(v) for v in bbox],
                                    "score": det_score,
                                    "cls_id": cls_id,
                                    "cls_name": cls_name,
                                    "behavior": None,
                                }
                            )

                # Behaviour classification
                if detections:
                    behaviour_crops, behaviour_indices = prepare_behavior_crops(
                        frame_bgr,
                        detections,
                        context=args.behavior_context,
                    )
                    if behaviour_crops:
                        behaviour_preds = run_behavior_inference(
                            model=behaviour_model,
                            processor=behaviour_processor,
                            device=behaviour_device,
                            crops=behaviour_crops,
                            id2label=behaviour_id2label,
                            batch_size=args.behavior_batch_size,
                        )
                        for det_idx, pred in zip(behaviour_indices, behaviour_preds):
                            detections[det_idx].predictions = [pred]
                            if 0 <= det_idx < len(tracks_for_json):
                                tracks_for_json[det_idx]["behavior"] = {
                                    "label": pred[0][3:],
                                    "id": pred[0][:2],
                                    "prob": np.round(float(pred[1]), 3),
                                }

                # JSONL logging
                if track_logger is not None:
                    track_logger.log_frame(
                        frame_idx=frame_idx,
                        tracks=tracks_for_json,
                        frame_timestamp=frame_timestamp_str,
                    )

                # Visualization
                if render_frame:
                    annotated = annotate_frame(frame_bgr.copy(), detections)
                    writer.write(annotated)
                    rendered_frames += 1

                    if frame_callback is not None:
                        try:
                            frame_callback(frame_idx, annotated, tracks_for_json)
                        except Exception as exc:
                            logger.warning("Frame callback failed at frame %d: %s", frame_idx, exc)

                    if (args.save_jpg or frame_callback is not None) and jpg_saved < args.jpg_max_count:
                        if rendered_frames % args.jpg_interval == 0:
                            jpg_path = jpg_dir / frame_name
                            cv2.imwrite(str(jpg_path), annotated)
                            jpg_saved += 1

                processed += 1

    writer.release()
    if track_logger is not None:
        track_logger.close()
    if feature_hub is not None:
        hub_path = video_output_path.with_suffix(".npz")
        feature_hub.save(hub_path)
        logger.info("Track feature hub saved to %s", hub_path)

    logger.info(
        "Visualization saved to %s (processed %d frames)",
        video_output_path,
        processed,
    )

    # if args.save_tracklets:
    #     tracklets_output_dir = output_dir / "tracklets"
    #     extract_tracklets(
    #         tracks_json_path=tracks_json_path,
    #         video_path=args.video,
    #         output_dir=tracklets_output_dir,
    #     )
    #     logger.info("Tracklets saved to %s", tracklets_output_dir)


def main() -> None:
    run_tracking(parse_args())


if __name__ == "__main__":
    main()