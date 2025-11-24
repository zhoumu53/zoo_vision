import logging
from typing import Dict, List, Sequence, Tuple

import sys
import torch
import torchvision.transforms as T
from PIL import Image

import cv2
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from project.config import cfg as base_cfg  # noqa: E402
from project.datasets.make_dataloader import get_transforms  # noqa: E402
from project.models import make_model  # noqa: E402
from project.utils.tools import load_model  # noqa: E402

from pathlib import Path

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


GT_IMAGES_DIR = '/media/dherrera/ElephantsWD/elephants/reid_gt_cleaned_data'


COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (231, 76, 60),
    (46, 205, 113),
    (52, 152, 219),
    (155, 89, 182),
    (241, 196, 15),
]


ID2NAMES = {
    '01': 'Chandra',
    '02': 'Indi',
    '03': 'Fahra',
    '04': 'Panang',
    '05': 'Thai',
    '06': 'Zali',
}


DEFAULT_IDENTITY_NAMES = [
    "01_Chandra",
    "02_Indi",
    "03_Fahra",
    "04_Panang",
    "05_Thai",
    "06_Zali",
]


CAMERA_PARIS = {
    '016': '019',
    '019': '016',
    '017': '018',
    '018': '017',
}


def load_data(csv_file='/media/dherrera/ElephantsWD/elephants/reid_gt_cleaned_data/train_val_split/full_reid_data.csv'):
    df = pd.read_csv(csv_file)
    return df

def time2ampm(time_str: str) -> str:
    """
    Convert time in 'HHMMSS' format to 'AM/PM' format.
    e.g. '063510' -> 'AM'
    """
    hour = int(time_str[0:2])
    ampm = 'AM' if hour < 12 else 'PM'
    return ampm


def extract_metadata_from_video_path(videopath) -> Tuple[str, str, str, str]:
    """
    e.g. '/home/mu/Desktop/gt_videos/train/ZAG-ELP-CAM-018-20250830-025815-1756515495749-7.mp4' -> ('018', '20250830', '025815')
    """
    parts = videopath.split('/')
    
    filename = parts[-1]
    
    print("Filename extracted:", filename)
    camera_id = filename.split('-')[3]
    date = filename.split('-')[4]
    time = filename.split('-')[5]
    ampm = time2ampm(time)
    return camera_id, date, time, ampm


def extract_other_cameras(camera_id, date, time, ampm, raw_video_dir='/mnt/camera_nas') -> str | None:
    """
    Match the given time string to find the corresponding videos from other cameras.
    Arguments:
        raw_video_dir: str, base directory where raw videos are stored
        camera_id: str, camera ID extracted from the filename
        date: str, date extracted from the filename in 'YYYYMMDD' format
        time: str, time extracted from the filename in 'HHMMSS' format
        ampm: str, 'AM' or 'PM' based on the time
        find_opposite: bool, if True, find video from opposite camera (016<->019, 017<->018)
    Returns:
        matched_video_file: str or None, path to the matched video file or None if not found
    """

    video_dir = f'{raw_video_dir}/ZAG-ELP-CAM-{camera_id}/{date}{ampm}'

    video_files = [f for f in os.listdir(video_dir) if f.startswith(f'ZAG-ELP-CAM-{camera_id}-{date}')]


    is_matched = False
    matched_video_file = None
    for video_file in video_files:
        video_starting_time = video_file.split('-')[5]  # Extract time part from filename
        
        video_start_hour = int(video_starting_time[0:2])
        video_start_minute = int(video_starting_time[2:4])
        video_start_second = int(video_starting_time[4:6])
        given_hour = int(time[0:2])
        given_minute = int(time[2:4])
        given_second = int(time[4:6])
        time_diff = (given_hour - video_start_hour) * 3600 + (given_minute - video_start_minute) * 60 + (given_second - video_start_second)
        if -1800 <= time_diff <= 1800:  # within 30 minutes
            is_matched = True
            matched_video_file = os.path.join(video_dir, video_file)
            break

    return matched_video_file if is_matched else None


@dataclass
class DetectionResult:
    """Structure storing a single detection + reid match information."""

    bbox: Tuple[int, int, int, int] # x1, y1, x2, y2
    score: float | None
    cls_id: int | None
    cls_name: str | None
    track_id: int | None
    display_track_id: int | None
    identity_label: str | None
    identity_score: float | None
    matches: List[Tuple[str, float]] | None
    predictions: List[Tuple[str, float]] | None


@dataclass
class GalleryDB:
    """Holds gallery features and metadata."""

    features: torch.Tensor
    labels: np.ndarray
    ids: np.ndarray
    paths: np.ndarray
    label_to_color: Dict[str, Tuple[int, int, int]]


def load_class_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {path}")
    return names


def bgr_from_palette_index(idx: int) -> Tuple[int, int, int]:
    r, g, b = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    return int(b), int(g), int(r)


def build_label_color_map(labels: Sequence[str]) -> Dict[str, Tuple[int, int, int]]:
    """Assigns deterministic colors per label to avoid collisions."""
    mapping: Dict[str, Tuple[int, int, int]] = {}
    for idx, label in enumerate(sorted(set(labels))):
        mapping[label] = bgr_from_palette_index(idx)
    return mapping

def resolve_checkpoint_path(path_str: str) -> Path:
    """Resolve checkpoint path allowing relative references to repo root."""
    candidates = []
    input_path = Path(path_str)
    if input_path.is_absolute():
        candidates.append(input_path)
    else:
        candidates.extend(
            [
                (Path.cwd() / input_path),
                (PROJECT_ROOT / input_path),
                (THIS_DIR / input_path),
            ]
        )
    seen = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Identity checkpoint not found at {path_str}")


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
    unique_labels = labels.tolist()
    label_to_color = build_label_color_map(unique_labels)
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
    device: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run YOLO inference on a frame and return xyxy boxes, scores, class ids."""
    results = model.predict(
        source=frame,
        conf=conf_thres,
        iou=iou_thres,
        retina_masks=False,
        verbose=False,
        max_det=max_dets,
        device=device,
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


def run_yolo_byteTrack(
    model: YOLO,
    frame: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    max_dets: int,
    tracker_cfg: str | None,
    device: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run YOLO detection + ByteTrack, returning boxes, scores, classes, and track IDs."""
    results = model.track(
        source=frame,
        conf=conf_thres,
        iou=iou_thres,
        retina_masks=False,
        verbose=False,
        max_det=max_dets,
        tracker=tracker_cfg,
        persist=True,
        device=device,
    )
    if not results:
        return (
            np.empty((0, 4)),
            np.empty((0,)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )
    res = results[0]
    if res.boxes is None or res.boxes.data.shape[0] == 0:
        return (
            np.empty((0, 4)),
            np.empty((0,)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )

    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    track_ids = (
        res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.full(len(boxes), -1, dtype=int)
    )
    return boxes, scores, cls_ids, track_ids



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
