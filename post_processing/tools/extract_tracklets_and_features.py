from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch

from post_processing.tools.extract_tracklets import export_tracklets
from post_processing.tools.run_reid_feature_extraction import load_reid, run_feature_extraction

logger = logging.getLogger(__name__)

CAMERA_TOKEN_RE = re.compile(r"([A-Za-z0-9]+-[A-Za-z0-9]+-CAM-\d{3})", re.IGNORECASE)
CAMERA_WITH_DATE_RE = re.compile(
    r"(?P<prefix>[A-Za-z0-9]+-[A-Za-z0-9]+-CAM-(?P<cam>\d{3}))-(?P<date>\d{8})(?:-(?P<time>\d{6}))?",
    re.IGNORECASE,
)
DATE_TOKEN_RE = re.compile(r"(20\d{6})")


def _format_date(raw: str | None) -> str:
    if not raw:
        return "unknown_date"
    raw = raw.strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw


def _date_from_timestamp(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        cleaned = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).date().isoformat()
    except Exception:
        return None


def _read_first_timestamp(jsonl_path: Path) -> str | None:
    try:
        with jsonl_path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ts = obj.get("timestamp")
                if not ts:
                    ts = obj.get("results", {}).get("timestamp")
                if ts:
                    return ts
    except Exception as err:  # pragma: no cover - safety net for bad inputs
        logger.warning("Could not read timestamps from %s: %s", jsonl_path, err)
    return None


def _infer_camera(jsonl_path: Path) -> str:
    match = CAMERA_TOKEN_RE.search(jsonl_path.name)
    if not match:
        for part in jsonl_path.parts:
            found = CAMERA_TOKEN_RE.search(part)
            if found:
                match = found
                break
    camera_token = match.group(1) if match else "unknown_cam"
    return camera_token.replace("-", "_").lower()


def _infer_date(jsonl_path: Path) -> str:
    match = CAMERA_WITH_DATE_RE.search(jsonl_path.name)
    if match:
        return _format_date(match.group("date"))

    for part in jsonl_path.parts:
        date_match = DATE_TOKEN_RE.fullmatch(part)
        if date_match:
            return _format_date(date_match.group(1))

    ts = _read_first_timestamp(jsonl_path)
    parsed = _date_from_timestamp(ts)
    return parsed or _format_date(None)


def _collect_jsonls_for_date(date_str: str, search_root: Path) -> List[Path]:
    """Find all JSONL files for a given date (yyyymmdd) under search_root."""
    found: List[Path] = []
    pattern = f"*{date_str}*tracks.jsonl"
    if not search_root.exists():
        logger.warning("Search root does not exist: %s", search_root)
        return found
    for path in search_root.rglob(pattern):
        if path.is_file():
            found.append(path)
    found.sort()
    return found


def _process_jsonl(
    jsonl_path: Path,
    output_root: Path,
    reid_model,
    gallery_path: Path,
    output_size: int,
    bbox_format: str,
    device: str,
    batch_size: int,
) -> Tuple[Path, List[Path]]:
    camera_slug = _infer_camera(jsonl_path)
    date_str = _infer_date(jsonl_path)

    session_dir = jsonl_path.stem
    tracks_output_dir = output_root / camera_slug / date_str
    logger.info("Processing %s -> %s", jsonl_path, tracks_output_dir)

    tracks_dir = export_tracklets(
        jsonl_path=jsonl_path,
        output_size=output_size,
        bbox_format=bbox_format,
        output_dir=tracks_output_dir,
    )

    video_files = sorted(tracks_dir.glob("*.mkv"))
    if not video_files:
        logger.warning("No .mkv tracks found in %s", tracks_dir)
        return tracks_dir, []

    saved_paths: List[Path] = []
    for video_file in video_files:
        saved = run_feature_extraction(
            video_path=video_file,
            reid_model=reid_model,
            device=device,
            batch_size=batch_size,
            gallery_path=gallery_path,
        )
        saved_paths.append(saved)
    return tracks_dir, saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract tracklets from tracking JSONL files and run ReID feature extraction.")
    parser.add_argument("--date", default='20251129', type=str, help="Date string in yyyymmdd format used to locate JSONL files.")
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("/media/dherrera/ElephantsWD/tracking_results"),
        help="Root directory to search for tracking JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/media/dherrera/ElephantsWD/elephants/test/tracks"),
        help="Root directory to store extracted tracks and features (organized by camera/date).",
    )
    parser.add_argument("--output-size", type=int, default=512, help="Square crop/resize size.")
    parser.add_argument("--bbox-format", type=str, default="auto", choices=["auto", "xyxy", "xywh"], help="BBox format in JSON.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on (e.g., cuda:0 or cpu).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/media/mu/zoo_vision/training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml"),
        help="ReID config file path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/media/dherrera/ElephantsWD/reid_models/logs/swin_adamw_lr0003_bs64_softmax_triplet/net_best.pth"),
        help="ReID checkpoint path.",
    )
    parser.add_argument(
        "--gallery-path",
        type=Path,
        default=None,
        help="Optional gallery features NPZ path for matching. Defaults to checkpoint.parent/pred_features/train_iid/pytorch_result_e.npz",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not re.fullmatch(r"20\d{6}", args.date):
        raise ValueError(f"Date must be in yyyymmdd format (e.g., 20251129). Got: {args.date}")

    jsonl_inputs = _collect_jsonls_for_date(args.date, args.search_root)
    if not jsonl_inputs:
        raise FileNotFoundError(f"No JSONL files found for date {args.date} under {args.search_root}")

    if args.device != "cpu" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    config = args.config
    checkpoint = args.checkpoint

    gallery_path = args.gallery_path
    if gallery_path is None:
        gallery_path = Path(checkpoint).parent / "pred_features" / "train_iid" / "pytorch_result_e.npz"
        logger.info("No gallery path provided. Using default gallery path: %s", gallery_path)

    try:
        reid_model = load_reid(config_path=config, checkpoint_path=checkpoint, device=args.device, mode="feature")
    except RuntimeError as err:
        logger.error("Failed to load ReID model on %s: %s", args.device, err)
        if args.device != "cpu":
            logger.info("Retrying model load on CPU to avoid CUDA-related errors.")
            reid_model = load_reid(config_path=config, checkpoint_path=checkpoint, device="cpu", mode="feature")
            args.device = "cpu"
        else:
            raise

    for jsonl_path in jsonl_inputs:
        print("Processing file:", jsonl_path)
        tracks_dir, feature_paths = _process_jsonl(
            jsonl_path=jsonl_path,
            output_root=args.output_root,
            reid_model=reid_model,
            gallery_path=gallery_path,
            output_size=args.output_size,
            bbox_format=args.bbox_format,
            device=args.device,
            batch_size=args.batch_size,
        )
        logger.info("Finished %s -> %s (%d feature files)", jsonl_path.name, tracks_dir, len(feature_paths))


if __name__ == "__main__":
    main()
