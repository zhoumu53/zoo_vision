#!/usr/bin/env python3
"""
visualize_overlay_on_raw_videos_sparse_v2.py

Overlay stitched tracklet detections onto raw video frames (sparse, fast).

Updates requested:
- Accept multiple JSON files: --stitched_json can be repeated.
- Output directory grouped by (date, camera_id):
    out_dir_root/{YYYYMMDD}-{cam_id}-{start_time}-{end_time}/
  where start_time/end_time are extracted from JSON start_timestamp/end_timestamp (min/max).
- Frame filenames use timestamps (plus frame idx to avoid collisions):
    ts_{timestamp_str}_f{raw_frame_idx:08d}.png
- No separation by raw video name: all results for same day+camera go into the same folder.

Other behavior:
- Only render frames with detections.
- If multiple dets share same coarse timestamp for the same track, keep only 1 per (raw_track_id, timestamp_str).
- Find raw video via _find_raw_video(camera_id, row_ts_dt, raw_root).
- Map row timestamp to raw frame index using video start datetime + fps.

Usage:
  python visualize_overlay_on_raw_videos_sparse_v2.py \
    --stitched_json a.json --stitched_json b.json \
    --raw_root /mnt/camera_nas \
    --out_dir_root /path/to/out \
    --camera_id 016 \
    --output_mkv --save_frames
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("vis_overlay_raw_sparse_v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# -----------------------------
# Raw video lookup (provided)
# -----------------------------
@dataclass(frozen=True)
class Pt:
    stitched_id: int
    t: datetime
    x: float
    y: float
    video_path: Path
    frame_idx: int

def _parse_video_start_from_name(path: Path) -> Optional[datetime]:
    """Parse datetime from raw video filename like ZAG-ELP-CAM-016-20251129-001949-....mp4."""
    name = path.name
    m = re.search(r"CAM-(?P<cam>\d{3})-(?P<date>\d{8})-(?P<hms>\d{6})", name)
    if not m:
        return None
    dt_str = f"{m.group('date')}{m.group('hms')}"
    try:
        return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    except Exception:
        return None


def _find_raw_video(
    camera_id: str,
    timestamp: Optional[datetime],
    root: Path = Path("/mnt/camera_nas"),
) -> Optional[Path]:
    """
    Find the closest raw video under /mnt/camera_nas/ZAG-ELP-CAM-{cam}/{date}{AM|PM}/.
    Chooses the video whose parsed start time is closest to the given timestamp.
    """
    if timestamp is None:
        return None

    date_str = timestamp.strftime("%Y%m%d")
    ampm = "AM" if timestamp.hour < 12 else "PM"
    base_dir = root / f"ZAG-ELP-CAM-{camera_id}" / f"{date_str}{ampm}"

    if not base_dir.exists():
        return None

    candidates = list(base_dir.glob("*.mp4"))
    if not candidates:
        candidates = list(base_dir.rglob("*.mp4"))
    if not candidates:
        return None

    best_path = None
    best_delta = None
    for p in candidates:
        start_dt = _parse_video_start_from_name(p)
        if start_dt is None:
            continue
        delta = abs((start_dt - timestamp).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_path = p

    return best_path or candidates[0]


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class SegmentMeta:
    stitched_id: int
    raw_track_id: str
    camera_id: str
    track_csv_path: Path
    feature_path: Path
    track_video_path: Optional[Path]
    start_timestamp: Optional[str]
    end_timestamp: Optional[str]


@dataclass(frozen=True)
class Det:
    stitched_id: int
    raw_track_id: str
    camera_id: str

    timestamp_str: str
    timestamp_dt: Optional[datetime]

    bbox: Tuple[int, int, int, int]  # left, top, right, bottom
    score: Optional[float]
    behavior_label: Optional[str]
    behavior_conf: Optional[float]

    matched_label: Optional[str]
    avg_common_label: Optional[str]
    voted_label: Optional[str]


# -----------------------------
# JSON loading (multi-file)
# -----------------------------

def load_one_stitched_json(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items() if k.isdigit()}


def load_many_stitched_json(paths: List[Path]) -> Dict[int, List[Dict[str, Any]]]:
    merged: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for p in paths:
        d = load_one_stitched_json(p)
        for sid, segs in d.items():
            merged[int(sid)].extend(segs)
    return dict(merged)


# -----------------------------
# CSV / NPZ helpers
# -----------------------------

def load_track_table(path: Path) -> pd.DataFrame:
    """Auto-detect CSV/TSV delimiter."""
    if not path.exists():
        raise FileNotFoundError(f"Track CSV not found: {path}")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()

    if "\t" in header:
        sep = "\t"
    elif "," in header:
        sep = ","
    else:
        sep = None

    if sep is None:
        for trial in [",", "\t"]:
            try:
                df = pd.read_csv(path, sep=trial, engine="python")
                if "timestamp" in df.columns and "bbox_left" in df.columns:
                    sep = trial
                    break
            except Exception:
                pass

    if sep is None:
        df = pd.read_csv(path, sep=r"\s+|,|\t", engine="python")
    else:
        df = pd.read_csv(path, sep=sep, engine="python")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def pick_most_common_label(x: Any) -> Optional[str]:
    """For avg_matched_labels: pick most common element."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        vals = [str(v) for v in x.flatten().tolist()]
    elif isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        vals: List[str] = []
        for v in x:
            if isinstance(v, (list, tuple, np.ndarray)):
                vals.extend([str(z) for z in np.asarray(v).flatten().tolist()])
            else:
                vals.append(str(v))
        if not vals:
            return None
    else:
        return str(x)
    return Counter(vals).most_common(1)[0][0]


def pick_scalar_label(x: Any) -> Optional[str]:
    """For voted_labels: preserve original semantics."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return str(x.item())
        if x.size == 0:
            return None
        return str(x.flatten()[0])
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        return str(x[0])
    return str(x)


def load_npz_labels(npz_path: Path) -> Tuple[Dict[int, str], Optional[str], Optional[str]]:
    if not npz_path.exists():
        LOGGER.warning("NPZ not found: %s", npz_path)
        return {}, None, None

    npz = np.load(npz_path, allow_pickle=True)

    frame_ids = npz.get("frame_ids", None)
    matched_labels = npz.get("matched_labels", None)
    avg_matched_labels = npz.get("avg_matched_labels", None)
    voted_labels = npz.get("voted_labels", None)

    matched_by_fid: Dict[int, str] = {}
    if frame_ids is not None and matched_labels is not None:
        fids = np.asarray(frame_ids).astype(int).tolist()
        mls = list(matched_labels)
        if len(fids) == len(mls):
            matched_by_fid = {int(fid): str(lbl) for fid, lbl in zip(fids, mls)}
        else:
            LOGGER.warning("NPZ mismatch: %s frame_ids=%d matched_labels=%d", npz_path, len(fids), len(mls))

    avg_common = pick_most_common_label(avg_matched_labels)
    voted = pick_scalar_label(voted_labels)
    return matched_by_fid, avg_common, voted


# -----------------------------
# Timestamp parsing
# -----------------------------

_PAT_YYYYMMDD_HHMMSS = re.compile(r"^(\d{8})_(\d{6})$")  # 20251129_153128

def parse_row_timestamp(ts: str) -> Optional[datetime]:
    ts = str(ts)
    m = _PAT_YYYYMMDD_HHMMSS.match(ts)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except Exception:
            return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def parse_iso_dt(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


def fmt_date(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def fmt_time(dt: datetime) -> str:
    return dt.strftime("%H%M%S")


# -----------------------------
# Segment parsing + filters
# -----------------------------

def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_segments(
    stitched_json: Dict[int, List[Dict[str, Any]]],
    include_ids: Optional[List[int]],
    camera_id: Optional[str],
    path_contains: Optional[str],
) -> List[SegmentMeta]:
    stitched_ids = sorted(stitched_json.keys()) if include_ids is None else include_ids
    segments: List[SegmentMeta] = []

    for sid in stitched_ids:
        for seg in stitched_json.get(sid, []):
            cam = seg.get("camera_id", None)
            if cam is None:
                continue
            cam = str(cam)

            if camera_id is not None and cam != str(camera_id):
                continue

            csv_path = Path(seg["track_csv_path"])
            if path_contains is not None and path_contains not in str(csv_path):
                continue

            tvp = seg.get("track_video_path", None)
            tvp_path = Path(tvp) if tvp else None

            segments.append(
                SegmentMeta(
                    stitched_id=int(seg.get("stitched_id", sid)),
                    raw_track_id=str(seg["raw_track_id"]),
                    camera_id=cam,
                    track_csv_path=csv_path,
                    feature_path=Path(seg["feature_path"]),
                    track_video_path=tvp_path,
                    start_timestamp=str(seg.get("start_timestamp")) if seg.get("start_timestamp") is not None else None,
                    end_timestamp=str(seg.get("end_timestamp")) if seg.get("end_timestamp") is not None else None,
                )
            )

    return segments


# -----------------------------
# Output dir naming from JSON
# -----------------------------
import re
from pathlib import Path
from typing import Tuple

_JSON_NAME_PAT = re.compile(
    r"stitched_tracklets_cam(?P<cam>\d{3})_(?P<start>\d{6})_(?P<end>\d{6})\.json$",
    re.IGNORECASE,
)

_DATE_IN_STR = re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})")


def compute_outdir_suffix_from_json_filename_single(
    stitched_json_path: Path,
    path_contains: str | None = None,
) -> Tuple[str, str, str, str]:
    """
    Returns (date_yyyymmdd, cam_id, start_hhmmss, end_hhmmss)
    from a SINGLE json filename:
      stitched_tracklets_cam016_153000_235959.json
    """

    m = _JSON_NAME_PAT.match(stitched_json_path.name)
    if not m:
        raise ValueError(
            f"JSON filename does not match expected pattern: {stitched_json_path.name}"
        )

    cam = m.group("cam")
    start_hms = m.group("start")
    end_hms = m.group("end")

    # date from path_contains (preferred)
    date_yyyymmdd = "unknown_date"
    if path_contains:
        dm = _DATE_IN_STR.search(path_contains)
        if dm:
            date_yyyymmdd = f"{dm.group('y')}{dm.group('m')}{dm.group('d')}"

    # fallback: try extracting date from json path
    if date_yyyymmdd == "unknown_date":
        dm = _DATE_IN_STR.search(str(stitched_json_path))
        if dm:
            date_yyyymmdd = f"{dm.group('y')}{dm.group('m')}{dm.group('d')}"

    return date_yyyymmdd, cam, start_hms, end_hms


# -----------------------------
# Dedup (speed): one bbox per (raw_track_id, timestamp_str) per frame
# -----------------------------

def dedup_one_per_track_per_timestamp(dets: List[Det]) -> List[Det]:
    best: Dict[Tuple[str, str], Det] = {}
    for d in dets:
        key = (d.raw_track_id, d.timestamp_str)
        if key not in best:
            best[key] = d
            continue
        cur = best[key]
        if d.score is None:
            continue
        if cur.score is None or float(d.score) > float(cur.score):
            best[key] = d
    return list(best.values())


# -----------------------------
# Bucketing: video_path -> frame_idx_guess -> dets
# -----------------------------

def build_buckets_sparse(
    segments: List[SegmentMeta],
    raw_root: Path,
    max_video_delta_sec: float = 600.0,
    fps_guess: float = 25.0,
) -> Dict[Path, Dict[int, List[Det]]]:
    buckets: Dict[Path, Dict[int, List[Det]]] = defaultdict(lambda: defaultdict(list))
    start_cache: Dict[Path, Optional[datetime]] = {}

    for seg in segments:
        df = load_track_table(seg.track_csv_path)
        required = ["frame_id", "timestamp", "bbox_top", "bbox_left", "bbox_bottom", "bbox_right"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {seg.track_csv_path}. Detected: {list(df.columns)}")

        matched_by_fid, avg_common, voted = load_npz_labels(seg.feature_path)

        for _, row in df.iterrows():
            ts_str = str(row["timestamp"])
            ts_dt = parse_row_timestamp(ts_str)
            if ts_dt is None:
                continue

            # resolve raw video
            video_path = seg.track_video_path
            if video_path is None or str(video_path) == "" or not Path(video_path).exists():
                video_path = _find_raw_video(seg.camera_id, ts_dt, root=raw_root)
            if video_path is None:
                continue
            video_path = Path(video_path)

            if video_path not in start_cache:
                start_cache[video_path] = _parse_video_start_from_name(video_path)
            vstart = start_cache[video_path]
            if vstart is None:
                continue

            delta_sec = (ts_dt - vstart).total_seconds()
            if delta_sec < 0:
                continue
            if max_video_delta_sec > 0 and abs(delta_sec) > max_video_delta_sec:
                continue

            frame_idx_guess = int(round(delta_sec * fps_guess))
            if frame_idx_guess < 0:
                continue

            fid = int(row["frame_id"])
            matched = matched_by_fid.get(fid, None)

            score = row.get("score", None)
            score = None if (pd.isna(score) if score is not None else True) else float(score)

            b_lbl = row.get("behavior_label", None)
            b_lbl = None if (pd.isna(b_lbl) if b_lbl is not None else True) else str(b_lbl)

            b_conf = row.get("behavior_conf", None)
            b_conf = None if (pd.isna(b_conf) if b_conf is not None else True) else float(b_conf)

            det = Det(
                stitched_id=seg.stitched_id,
                raw_track_id=seg.raw_track_id,
                camera_id=seg.camera_id,
                timestamp_str=ts_str,
                timestamp_dt=ts_dt,
                bbox=(int(row["bbox_left"]), int(row["bbox_top"]), int(row["bbox_right"]), int(row["bbox_bottom"])),
                score=score,
                behavior_label=b_lbl,
                behavior_conf=b_conf,
                matched_label=matched,
                avg_common_label=avg_common,
                voted_label=voted,
            )
            buckets[video_path][frame_idx_guess].append(det)

    return buckets


def remap_buckets_to_true_fps(
    video_path: Path,
    per_frame_guess: Dict[int, List[Det]],
    fps_true: float,
) -> Dict[int, List[Det]]:
    vstart = _parse_video_start_from_name(video_path)
    if vstart is None:
        return per_frame_guess

    out: Dict[int, List[Det]] = defaultdict(list)
    for _, dets in per_frame_guess.items():
        for d in dets:
            if d.timestamp_dt is None:
                continue
            delta_sec = (d.timestamp_dt - vstart).total_seconds()
            idx = int(round(delta_sec * fps_true))
            if idx >= 0:
                out[idx].append(d)
    return out


# -----------------------------
# Video IO + drawing
# -----------------------------

def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def get_video_props(cap: cv2.VideoCapture) -> Tuple[int, int, float, int]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, n


def make_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec_trials = ["X264", "H264", "avc1", "mp4v", "MJPG"]
    for c in codec_trials:
        fourcc = cv2.VideoWriter_fourcc(*c)
        w = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if w.isOpened():
            LOGGER.info("VideoWriter opened: %s (fourcc=%s)", output_path, c)
            return w
    raise RuntimeError(f"Failed to open VideoWriter for {output_path}. Tried codecs: {codec_trials}")


def color_for_stitched_id(stitched_id: int) -> Tuple[int, int, int]:
    hue = (stitched_id * 0.61803398875) % 1.0
    h = int(hue * 179)
    s = 200
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def put_text(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int],
             scale: float = 0.6, thickness: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_dets(frame: np.ndarray, dets: List[Det], header: str = "") -> np.ndarray:
    dets = dedup_one_per_track_per_timestamp(dets)

    if header:
        put_text(frame, header, (10, 30), (255, 255, 255), scale=0.8, thickness=2)

    for d in dets:
        left, top, right, bottom = d.bbox
        col = color_for_stitched_id(d.stitched_id)
        cv2.rectangle(frame, (left, top), (right, bottom), col, 2)

        text_id = f"ID={d.stitched_id} ({d.raw_track_id})",
        # if d.matched_label is not None:
        #     lines.append(f"matched={d.matched_label}")
        text_label = ''
        if d.avg_common_label is not None and d.voted_label is not None:
            text_label = f"{d.avg_common_label[:1]}, {d.voted_label[:1]} "
        if d.behavior_label is not None:
            if d.behavior_conf is not None:
                text_label += f"{d.behavior_label.split('_')[1]}({d.behavior_conf:.2f})"
            else:
                text_label += f"{d.behavior_label}"

        x0 = max(0, left)
        y0 = max(20, top - 25)  
        put_text(frame, str(text_id), (x0, y0), col, scale=0.5, thickness=1)
        put_text(frame, str(text_label), (x0, y0 + 22), (255, 255, 255), scale=0.5, thickness=1)

    return frame


def build_pts_from_buckets(
    buckets: Dict[Path, Dict[int, List[Det]]]
) -> List[Pt]:
    pts: List[Pt] = []

    for video_path, per_frame in buckets.items():
        for frame_idx, dets in per_frame.items():
            dets = dedup_one_per_track_per_timestamp(dets)
            for d in dets:
                if d.timestamp_dt is None:
                    continue
                l, t, r, b = d.bbox
                cx = 0.5 * (l + r)
                cy = 0.5 * (t + b)
                pts.append(
                    Pt(
                        stitched_id=int(d.stitched_id),
                        t=d.timestamp_dt,
                        x=cx,
                        y=cy,
                        video_path=video_path,
                        frame_idx=int(frame_idx),
                    )
                )
    return pts
# -----------------------------
# Rendering (sparse frames only) + new naming
# -----------------------------

def sanitize_ts_for_filename(ts: str) -> str:
    s = str(ts)
    s = s.replace(":", "-")
    s = s.replace("/", "-")
    return re.sub(r"[^0-9A-Za-z_\-\.]", "_", s)


def render_one_video_sparse(
    video_path: Path,
    per_frame_guess: Dict[int, List[Det]],
    out_daycam_dir: Path,
    save_frames: bool,
    output_mkv: bool,
    fps_override: Optional[float],
    write_every: int,
) -> None:
    cap = open_video(video_path)
    w, h, fps, n = get_video_props(cap)
    if fps_override is not None:
        fps = float(fps_override)

    per_frame = remap_buckets_to_true_fps(video_path, per_frame_guess, fps_true=fps)
    keys = sorted(per_frame.keys())
    if not keys:
        cap.release()
        return

    if write_every <= 0:
        raise ValueError("--write_every must be >= 1")

    frames_dir = out_daycam_dir / "frames"
    videos_dir = out_daycam_dir / "videos"
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if output_mkv:
        videos_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if output_mkv:
        # one mkv per raw video (still useful), but placed under same day+cam folder
        writer = make_writer(videos_dir / f"{video_path.stem}.mkv", fps, w, h)

    LOGGER.info("Render sparse %s: det_frames=%d", video_path.name, len(keys))

    for idx, fid in enumerate(keys):
        if write_every > 1 and (idx % write_every != 0):
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok:
            continue

        dets = per_frame.get(fid, [])
        if not dets:
            continue

        # Build a representative timestamp for filename (dominant ts among dets)
        ts_list = [d.timestamp_str for d in dets if d.timestamp_str]
        dom_ts = Counter(ts_list).most_common(1)[0][0] if ts_list else "unknown_ts"
        dom_ts_safe = sanitize_ts_for_filename(dom_ts)

        header = f"{video_path.name} frame={fid} dets={len(dets)}"
        frame = draw_dets(frame, dets, header=header)

        if save_frames:
            # Requested: use timestamp in filename; also add fid to avoid overwriting
            out_path = frames_dir / f"ts_{dom_ts_safe}_f{fid:08d}.png"
            cv2.imwrite(str(out_path), frame)

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

def det_center(d: Det) -> Tuple[float, float]:
    l, t, r, b = d.bbox
    return (l + r) / 2.0, (t + b) / 2.0


def collect_points_from_buckets(
    buckets: Dict[Path, Dict[int, List[Det]]],
    raw_root: Path,
    camera_id: str,
    max_video_delta_sec: float = 600.0,
) -> Tuple[List[Tuple[int, datetime, float, float]], Optional[Tuple[int, int]]]:
    """
    Returns:
      points: list of (stitched_id, timestamp_dt, cx, cy)
      frame_size: (W,H) if we can infer from any video, else None
    """
    points: List[Tuple[int, datetime, float, float]] = []
    frame_size: Optional[Tuple[int, int]] = None

    for video_path, per_frame_guess in buckets.items():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened() and frame_size is None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (w, h)
        if cap.isOpened():
            cap.release()

        # We can remap accurately by reading fps once
        cap2 = cv2.VideoCapture(str(video_path))
        fps = cap2.get(cv2.CAP_PROP_FPS) if cap2.isOpened() else 25.0
        if cap2.isOpened():
            cap2.release()

        per_frame = remap_buckets_to_true_fps(Path(video_path), per_frame_guess, fps_true=float(fps))

        for fid, dets in per_frame.items():
            dets = dedup_one_per_track_per_timestamp(dets)  # same dedup
            for d in dets:
                if d.timestamp_dt is None:
                    continue
                cx, cy = det_center(d)
                points.append((d.stitched_id, d.timestamp_dt, cx, cy))

    return points, frame_size

def load_background_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame

def blend(img_a: np.ndarray, img_b: np.ndarray, alpha_b: float) -> np.ndarray:
    """Return (1-alpha_b)*img_a + alpha_b*img_b."""
    alpha_b = float(np.clip(alpha_b, 0.0, 1.0))
    if img_a.shape[:2] != img_b.shape[:2]:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img_a, 1.0 - alpha_b, img_b, alpha_b, 0.0)

def make_heatmap_gray_for_id(
    pts: List[Pt],
    stitched_id: int,
    frame_size: Tuple[int, int],
    bin_size: int = 25,
) -> np.ndarray:
    """Return single-channel heatmap float32 in [0,1] for a given id."""
    W, H = frame_size
    sub = [p for p in pts if p.stitched_id == stitched_id]
    if not sub:
        return np.zeros((H, W), dtype=np.float32)

    xs = np.array([p.x for p in sub], dtype=np.float32)
    ys = np.array([p.y for p in sub], dtype=np.float32)

    xbins = max(5, W // bin_size)
    ybins = max(5, H // bin_size)

    H2, _, _ = np.histogram2d(xs, ys, bins=(xbins, ybins), range=[[0, W], [0, H]])
    H2 = np.log1p(H2)

    hm = (H2 / (H2.max() + 1e-6)).astype(np.float32)  # [0,1]
    hm = cv2.resize(hm.T, (W, H), interpolation=cv2.INTER_LINEAR)  # transpose for correct orientation
    return hm


def colorize_gray_heatmap(gray01: np.ndarray, bgr: Tuple[int, int, int], strength: float = 1.0) -> np.ndarray:
    """
    gray01: HxW float32 in [0,1]
    returns HxWx3 uint8
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    g = np.clip(gray01, 0.0, 1.0) * strength
    out = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.float32)
    out[..., 0] = g * bgr[0]
    out[..., 1] = g * bgr[1]
    out[..., 2] = g * bgr[2]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_heatmaps(images: List[np.ndarray]) -> np.ndarray:
    """Add multiple heatmaps (uint8 BGR) with clipping."""
    if not images:
        return None
    acc = np.zeros_like(images[0], dtype=np.uint16)
    for im in images:
        acc += im.astype(np.uint16)
    return np.clip(acc, 0, 255).astype(np.uint8)



# def draw_trajectories_on(
#     canvas: np.ndarray,
#     pts: List[Pt],
#     min_points_per_id: int = 5,
#     draw_points_every: int = 10,
#     thickness: int = 2,
# ) -> np.ndarray:
#     """Draw per-ID polyline trajectories on a canvas (in-place copy)."""
#     img = canvas.copy()
#     by_id: Dict[int, List[Pt]] = defaultdict(list)
#     for p in pts:
#         by_id[int(p.stitched_id)].append(p)

#     for sid, plist in by_id.items():
#         plist.sort(key=lambda p: p.t)
#         if len(plist) < min_points_per_id:
#             continue

#         col = color_for_stitched_id(int(sid))
#         poly = np.array([(int(p.x), int(p.y)) for p in plist], dtype=np.int32)

#         cv2.polylines(img, [poly], isClosed=False, color=col, thickness=thickness)

#         if draw_points_every > 0:
#             for i in range(0, len(poly), draw_points_every):
#                 cv2.circle(img, tuple(poly[i]), 3, col, -1)

#         # label at start
#         cv2.putText(
#             img,
#             f"ID {sid}",
#             tuple(poly[0]),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             col,
#             2,
#             cv2.LINE_AA,
#         )

#     return img

def draw_trajectories_on(
    canvas: np.ndarray,
    pts: List[Pt],
    min_points_per_id: int = 5,
    draw_points_every: int = 10,
    thickness: int = 2,
    draw_arrows: bool = True,
    arrow_every: int = 25,          # draw an arrow every N points
    arrow_step: int = 5,            # arrow from i -> i+arrow_step
    arrow_tip_length: float = 0.35, # cv2.arrowedLine tip size
    arrow_thickness: int = 1,
    arrow_color: Tuple[int, int, int] = (0, 255, 0),  # GREEN

    min_arrow_px: float = 12.0,
    max_arrow_px: float = 300.0,
) -> np.ndarray:
    """
    Draw per-ID polyline trajectories and optional direction arrows on a canvas.
    """
    img = canvas.copy()
    by_id: Dict[int, List[Pt]] = defaultdict(list)
    for p in pts:
        by_id[int(p.stitched_id)].append(p)

    for sid, plist in by_id.items():
        plist.sort(key=lambda p: p.t)
        if len(plist) < min_points_per_id:
            continue

        col = color_for_stitched_id(int(sid))
        poly = np.array([(int(p.x), int(p.y)) for p in plist], dtype=np.int32)

        # trajectory polyline
        cv2.polylines(img, [poly], isClosed=False, color=col, thickness=thickness)

        # sparse points
        if draw_points_every and draw_points_every > 0:
            for i in range(0, len(poly), draw_points_every):
                cv2.circle(img, tuple(poly[i]), 3, col, -1)

        # label at start
        cv2.putText(
            img,
            f"ID {sid}",
            tuple(poly[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            col,
            2,
            cv2.LINE_AA,
        )

        # # direction arrows (small, green)
        # if draw_arrows and arrow_every > 0:
        #     n = len(poly)
        #     for i in range(0, n - arrow_step, arrow_every):
        #         p0 = poly[i]
        #         p1 = poly[i + arrow_step]

        #         dx = float(p1[0] - p0[0])
        #         dy = float(p1[1] - p0[1])
        #         dist = (dx * dx + dy * dy) ** 0.5

        #         if dist < min_arrow_px or dist > max_arrow_px:
        #             continue

        #         cv2.arrowedLine(
        #             img,
        #             tuple(p0),
        #             tuple(p1),
        #             arrow_color,                 # GREEN
        #             arrow_thickness,             # thinner
        #             tipLength=arrow_tip_length,  # smaller tip
        #             line_type=cv2.LINE_AA,
        #         )

    return img


def plotting(
    pts: List[Pt],
    out_dir,
    frame_size: Optional[Tuple[int, int]] = None,
    background_mode: str = "earliest",  # earliest|none
    bg_alpha: float = 0.35,
    heat_alpha: float = 0.65,
    bin_size: int = 25,
    per_id_strength: float = 1.0,
    min_points_per_id: int = 20,
    traj_min_points_per_id: int = 5,
    traj_draw_points_every: int = 10,
) -> None:
    """
    Saves:
      - out_dir/trajectory.png
      - out_dir/heatmap.png
      - out_dir/combined.png

    pts: list of Pt(stitched_id, t, x, y, video_path, frame_idx)
    frame_size: (W,H). If None, inferred from background frame if available.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pts:
        raise ValueError("plotting(): pts is empty")

    # Determine frame size and optional background frame
    bg = None
    pts_sorted = sorted(pts, key=lambda p: p.t)

    if background_mode == "earliest":
        bg = load_background_frame(pts_sorted[0].video_path, pts_sorted[0].frame_idx)

    if frame_size is None:
        if bg is None:
            raise ValueError("frame_size is None and background frame could not be loaded. Provide frame_size.")
        frame_size = (bg.shape[1], bg.shape[0])  # W,H

    W, H = frame_size

    # Base canvas (black)
    base = np.zeros((H, W, 3), dtype=np.uint8)

    # Blend background into base (optional)
    if bg is not None:
        base = blend(base, bg, bg_alpha)

    # ---- Heatmap (per-ID colored) ----
    ids = sorted({p.stitched_id for p in pts})
    colored_maps: List[np.ndarray] = []
    for sid in ids:
        if sum(1 for p in pts if p.stitched_id == sid) < min_points_per_id:
            continue
        gray = make_heatmap_gray_for_id(pts, sid, frame_size, bin_size=bin_size)
        col = color_for_stitched_id(int(sid))
        colored_maps.append(colorize_gray_heatmap(gray, col, strength=per_id_strength))

    if colored_maps:
        heatmap = add_heatmaps(colored_maps)
    else:
        heatmap = np.zeros((H, W, 3), dtype=np.uint8)

    cv2.imwrite(str(out_dir / "heatmap.png"), heatmap)

    # ---- Trajectory only (on background or black) ----
    traj_canvas = base.copy()  # base already has optional bg
    traj_img = draw_trajectories_on(
        traj_canvas,
        pts,
        min_points_per_id=traj_min_points_per_id,
        draw_points_every=traj_draw_points_every,
        thickness=2,
    )
    cv2.imwrite(str(out_dir / "trajectory.png"), traj_img)

    # ---- Combined: base + heatmap + trajectories ----
    combined = cv2.addWeighted(base, 1.0, heatmap, float(np.clip(heat_alpha, 0.0, 1.0)), 0.0)
    combined = draw_trajectories_on(
        combined,
        pts,
        min_points_per_id=traj_min_points_per_id,
        draw_points_every=traj_draw_points_every,
        thickness=2,
    )
    cv2.imwrite(str(out_dir / "combined.png"), combined)


def make_multi_id_heatmap(
    pts: List[Pt],
    frame_size: Tuple[int, int],
    bin_size: int = 25,
    per_id_strength: float = 1.0,
    min_points_per_id: int = 20,
) -> np.ndarray:
    ids = sorted({p.stitched_id for p in pts})
    colored_maps: List[np.ndarray] = []

    for sid in ids:
        if sum(1 for p in pts if p.stitched_id == sid) < min_points_per_id:
            continue
        gray = make_heatmap_gray_for_id(pts, sid, frame_size, bin_size=bin_size)
        col = color_for_stitched_id(sid)  # your existing function
        colored_maps.append(colorize_gray_heatmap(gray, col, strength=per_id_strength))

    if not colored_maps:
        W, H = frame_size
        return np.zeros((H, W, 3), dtype=np.uint8)

    return add_heatmaps(colored_maps)


def draw_trajectories(
    points: List[Tuple[int, datetime, float, float]],
    frame_size: Tuple[int, int],
    background: Optional[np.ndarray] = None,
    min_points_per_id: int = 5,
    draw_points_every: int = 10,
) -> np.ndarray:
    """
    Draw per-ID trajectories as polylines.
    """
    W, H = frame_size
    if background is None:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        canvas = background.copy()

    # group by id
    by_id: Dict[int, List[Tuple[datetime, float, float]]] = defaultdict(list)
    for sid, t, x, y in points:
        by_id[int(sid)].append((t, x, y))

    for sid, pts in by_id.items():
        pts.sort(key=lambda z: z[0])
        if len(pts) < min_points_per_id:
            continue

        col = color_for_stitched_id(sid)
        poly = np.array([(int(x), int(y)) for _, x, y in pts], dtype=np.int32)

        # polyline
        cv2.polylines(canvas, [poly], isClosed=False, color=col, thickness=2)

        # sparse points
        if draw_points_every > 0:
            for i in range(0, len(poly), draw_points_every):
                cv2.circle(canvas, tuple(poly[i]), 3, col, -1)

        # label at start
        cv2.putText(canvas, f"ID {sid}", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    return canvas


def overlay_heatmap(base: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(base, 1.0 - alpha, heatmap, alpha, 0.0)

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--stitched_json", type=Path, required=True, action="append",
                    help="Path to stitched json. Can be repeated.")
    ap.add_argument("--raw_root", type=Path, required=True, help="Root of camera NAS, e.g., /mnt/camera_nas")
    ap.add_argument("--out_dir_root", type=Path, required=True, help="Root output folder.")

    ap.add_argument("--include_ids", type=str, default=None, help="Comma-separated stitched_ids to include (default all).")
    ap.add_argument("--path_contains", type=str, default=None, help="Filter segments by substring in track_csv_path (e.g. date).")

    ap.add_argument("--output_mkv", action="store_true")
    ap.add_argument("--save_frames", action="store_true")

    ap.add_argument("--fps", type=float, default=None, help="Override FPS for output MKV (default raw fps).")
    ap.add_argument("--write_every", type=int, default=1, help="Render every N detected frames (default 1).")

    ap.add_argument("--max_video_delta_sec", type=float, default=600.0,
                    help="Reject row->video matches if |row_ts - video_start| > this (set 0 to disable).")

    args = ap.parse_args()

    stitched_paths = [Path(p) for p in args.stitched_json]
    stitched_json = load_many_stitched_json(stitched_paths)
    include_ids = parse_int_list(args.include_ids)

    # Build day+cam output dir name from JSON start/end timestamps
    date_str, cam_id, st_hms, et_hms = compute_outdir_suffix_from_json_filename_single(
        stitched_json_path=stitched_paths[0],
        path_contains=args.path_contains,
    )
    segments = parse_segments(
        stitched_json=stitched_json,
        include_ids=include_ids,
        camera_id=str(cam_id),
        path_contains=args.path_contains,
    )
    LOGGER.info("Segments collected: %d (cam=%s)", len(segments), cam_id)
    if not segments:
        raise SystemExit("No segments matched your filters.")

    out_daycam_dir = args.out_dir_root / f"{date_str}-{cam_id}-{st_hms}-{et_hms}"
    out_daycam_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output folder: %s", out_daycam_dir)

    buckets = build_buckets_sparse(
        segments=segments,
        raw_root=args.raw_root,
        max_video_delta_sec=float(args.max_video_delta_sec),
        fps_guess=25.0,
    )
    LOGGER.info("Raw videos matched: %d", len(buckets))
    if not buckets:
        raise SystemExit("No raw videos matched. Check --raw_root and timestamp parsing.")

    # validate buckets
    for vp in sorted(buckets.keys()):
        render_one_video_sparse(
            video_path=vp,
            per_frame_guess=buckets[vp],
            out_daycam_dir=out_daycam_dir,
            save_frames=bool(args.save_frames),
            output_mkv=bool(args.output_mkv),
            fps_override=args.fps,
            write_every=int(args.write_every),
        )
    
    pts = build_pts_from_buckets(buckets)

    plotting(
        pts=pts,
        out_dir=out_daycam_dir,     # same folder you already use
        frame_size=None,            # infer from earliest background frame
        bg_alpha=0.35,
        heat_alpha=0.65,
        bin_size=25,
    )

if __name__ == "__main__":
    main()
