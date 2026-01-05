# visualization.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# 调整为你项目里的正确导入
from post_processing.core.tracklet_manager import Tracklet
from post_processing.utils import load_embedding

logger = logging.getLogger(__name__)


def _ensure_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _extract_head_tail_frames_from_video(
    tracklet: Tracklet,
    head_k: int = 3,
    tail_k: int = 3,
    logger_: Optional[logging.Logger] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从 tracklet 对应的原始视频里抽取一个 head frame 和一个 tail frame（RGB）。

    返回:
        head_img_rgb, tail_img_rgb
    """
    log = logger_ or logger

    if tracklet.feature_path is None:
        print(f"[viz] Tracklet {tracklet.track_id} has no feature_path.")
        return None, None

    try:
        feats, frame_ids, video_path, _ = load_embedding(tracklet.feature_path)
    except Exception as e:
        log.exception(
            f"[viz] Failed to load embedding for tracklet {tracklet.track_id}: {e}"
        )
        return None, None

    frame_ids = np.asarray(frame_ids)
    if frame_ids.size == 0:
        print(f"[viz] Tracklet {tracklet.track_id} has empty frame_ids.")
        return None, None

    frame_ids_sorted = np.sort(frame_ids)

    head_k = min(head_k, frame_ids_sorted.size)
    tail_k = min(tail_k, frame_ids_sorted.size)

    head_frame_id = int(frame_ids_sorted[:head_k].mean())
    tail_frame_id = int(frame_ids_sorted[-tail_k:].mean())

    # 确定视频路径
    if video_path is None:
        video_path = tracklet.feature_path.with_suffix(".mkv")
    video_path = Path(video_path)

    if not video_path.exists():
        print(
            f"[viz] Video file {video_path} for tracklet {tracklet.track_id} does not exist."
        )
        return None, None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(
            f"[viz] Failed to open video {video_path} for tracklet {tracklet.track_id}."
        )
        return None, None

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 防止越界
    if head_frame_id >= n_frames or tail_frame_id >= n_frames:
        print(
            f"[viz] Frame id out of range for {video_path}, "
            f"head={head_frame_id}, tail={tail_frame_id}, total={n_frames}."
        )
        cap.release()
        return None, None

    def _read_frame_at(fid: int) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(
                f"[viz] Failed to read frame {fid} from {video_path} "
                f"for tracklet {tracklet.track_id}."
            )
            # cap.set(cv2.CAP_PROP_POS_FRAMES, fid+1)
            # ret, frame = cap.read()
            return None
        return _ensure_rgb(frame)
    

    ### try to read head and tail frames - if fail, read the nearest valid frame
    try:
        head_img = _read_frame_at(head_frame_id)
        tail_img = _read_frame_at(tail_frame_id)
    except Exception as e:
        print(
            f"[viz] Exception reading head/tail frames for tracklet {tracklet.track_id}: {e}"
        )
        # 读取失败，尝试附近帧
        head_img = _read_frame_at(head_frame_id+1)
        tail_img = _read_frame_at(tail_frame_id)

    cap.release()
    return head_img, tail_img


def _sort_chain(tracklets: List[Tracklet]) -> List[Tracklet]:
    """按时间排序 tracklets。优先 start_frame_id，其次 timestamp。"""

    def key_fn(t: Tracklet):
        if t.start_timestamp is not None:
            return (int(t.start_timestamp.timestamp()), 1)
        return (0, 2)

    return sorted(tracklets, key=key_fn)


def _resize_to_common(
    images: List[Optional[np.ndarray]],
    target_h: int = 256,
    target_w: int = 256,
) -> List[np.ndarray]:
    """
    把一组图像 resize 到统一大小（缺失的用灰底代替）。
    输入图像为 RGB。
    """
    out: List[np.ndarray] = []
    for img in images:
        if img is None:
            canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        else:
            h, w = img.shape[:2]
            # 保持比例缩放，再居中填充
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
            y0 = (target_h - new_h) // 2
            x0 = (target_w - new_w) // 2
            canvas[y0:y0 + new_h, x0:x0 + new_w, :] = resized
        out.append(canvas)
    return out


def visualize_stitched_tracks_opencv(
    tracklets: List[Tracklet],
    output_dir: Path,
    camera_id: Optional[str] = None,
    head_k: int = 3,
    tail_k: int = 3,
    max_chains: Optional[int] = None,
    max_tracklets_per_chain: Optional[int] = None,
    cell_h: int = 256,
    cell_w: int = 256,
    max_cols_per_row: int = 5,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    使用 OpenCV 可视化 stitched 轨迹：

    对每个 stitched_id 生成一张图：
      - 图片大小: 2 行 × N 列，每个 cell 大小 cell_h × cell_w
      - 第一行: 每个 tracklet 的 head frame
      - 第二行: 对应 tracklet 的 tail frame
      - 在每个 cell 上用 cv2.putText 打上 track_id（以及 row: head/tail）

    Args:
        tracklets: 已经包含 stitched_id 的 Tracklet 列表。
        output_dir: 输出目录。
        camera_id: 若不为 None，则只可视化该相机的 tracklets。
        head_k, tail_k: 用于计算 head/tail frame index 的 frame_ids 数量。
        max_chains: 限制最多可视化多少个 stitched_id。
        max_tracklets_per_chain: 限制每个 chain 中最多显示多少个 tracklet。
        cell_h, cell_w: 每个小图的大小。
    """
    log = logger_ or logger
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 过滤相机
    if camera_id is not None:
        filtered = [t for t in tracklets if t.camera_id == camera_id]
    else:
        filtered = list(tracklets)

    # 按 stitched_id 分组
    from collections import defaultdict

    chains: Dict[int, List[Tracklet]] = defaultdict(list)
    for t in filtered:
        if t.stitched_id is None:
            continue
        chains[int(t.stitched_id)].append(t)

    if not chains:
        print("visualize_stitched_tracks_opencv: no stitched_id found.")
        return

    print(
        f"[viz] Found {len(chains)} stitched chains "
        f"(camera={camera_id if camera_id is not None else 'ALL'})."
    )

    stitched_ids = sorted(chains.keys())
    if max_chains is not None:
        stitched_ids = stitched_ids[:max_chains]

    for sid in stitched_ids:
        chain_tracklets = _sort_chain(chains[sid])
        if max_tracklets_per_chain is not None:
            chain_tracklets = chain_tracklets[:max_tracklets_per_chain]

        n = len(chain_tracklets)
        if n == 0:
            continue

        cols = max(1, max_cols_per_row)
        rows = (n + cols - 1) // cols

        print(
            f"[viz] Visualizing stitched_id={sid} with {n} tracklets "
            f"(camera={camera_id if camera_id is not None else 'ALL'})."
        )

        head_imgs: List[Optional[np.ndarray]] = []
        tail_imgs: List[Optional[np.ndarray]] = []
        track_ids: List[str] = []

        for t in chain_tracklets:
            head_img, tail_img = _extract_head_tail_frames_from_video(
                t, head_k=head_k, tail_k=tail_k, logger_=log
            )
            head_imgs.append(head_img)
            tail_imgs.append(tail_img)
            track_ids.append(t.track_id)

        # 统一 resize
        head_resized = _resize_to_common(head_imgs, cell_h, cell_w)
        tail_resized = _resize_to_common(tail_imgs, cell_h, cell_w)

        # 拼 mosaic: 2 行 × cols 列, 多行折行
        mosaic_h = cell_h * 2 * rows
        mosaic_w = cell_w * cols
        mosaic = np.full((mosaic_h, mosaic_w, 3), 0, dtype=np.uint8)

        # 字体设置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        for col in range(n):
            row_idx = col // cols
            col_idx = col % cols

            # head row
            h_img = head_resized[col]
            t_img = tail_resized[col]

            x0 = col_idx * cell_w
            x1 = x0 + cell_w
            y_row = row_idx * 2 * cell_h

            # row 0: head
            mosaic[y_row : y_row + cell_h, x0:x1] = h_img
            # row 1: tail
            mosaic[y_row + cell_h : y_row + 2 * cell_h, x0:x1] = t_img

            # 写文字：track_id + row label
            text_id = track_ids[col]
            cv2.putText(
                mosaic,
                f"{text_id}",
                (x0 + 5, y_row + 15),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                mosaic,
                "head",
                (x0 + 5, y_row + cell_h - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                mosaic,
                "tail",
                (x0 + 5, y_row + cell_h + 15),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        out_name = (
            f"stitched_chain_{camera_id or 'ALL'}_{sid}_N{n}.png"
        )
        out_path = output_dir / out_name

        # 注意：OpenCV 需要 BGR, 我们的 mosaic 是 RGB，可直接保存也问题不大，
        # 如果你想严格，就转换回 BGR:
        mosaic_bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), mosaic_bgr)

        print(f"[viz] Saved stitched chain visualization to {out_path}")


def visualize_stitched_tracks_pairs(
    tracklets: List[Tracklet],
    output_dir: Path,
    camera_id: Optional[str] = None,
    head_k: int = 3,
    tail_k: int = 3,
    max_chains: Optional[int] = None,
    max_tracklets_per_chain: Optional[int] = None,
    cell_h: int = 256,
    cell_w: int = 256,
    max_cols_per_row: int = 5,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    Visualize stitched tracks as a 1D sequence of (head|tail) pairs.

    For each stitched_id we create a single-row mosaic:

        [ head(track_0) | tail(track_0) ]   [ head(track_1) | tail(track_1) ] ...

    where tracklets are sorted temporally.

    Args:
        tracklets: Tracklet list with stitched_id already set.
        output_dir: Where to save PNGs.
        camera_id: If set, only use tracklets from this camera.
        head_k, tail_k: How many frame_ids to average to choose head/tail frame index.
        max_chains: Max number of stitched_ids to visualize.
        max_tracklets_per_chain: Max number of tracklets per chain.
        cell_h, cell_w: Size of each head or tail cell. One pair is (cell_h x 2*cell_w).
    """
    log = logger_ or logger
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter by camera if needed
    if camera_id is not None:
        filtered = [t for t in tracklets if t.camera_id == camera_id]
    else:
        filtered = list(tracklets)

    from collections import defaultdict
    chains: Dict[int, List[Tracklet]] = defaultdict(list)
    for t in filtered:
        if t.stitched_id is None:
            continue
        chains[int(t.stitched_id)].append(t)

    if not chains:
        log.warning("visualize_stitched_tracks_pairs: no stitched_id found.")
        return

    print(
        f"[viz_pairs] Found {len(chains)} stitched chains "
        f"(camera={camera_id if camera_id is not None else 'ALL'})."
    )

    stitched_ids = sorted(chains.keys())
    if max_chains is not None:
        stitched_ids = stitched_ids[:max_chains]

    for sid in stitched_ids:
        chain_tracklets = _sort_chain(chains[sid])
        if max_tracklets_per_chain is not None:
            chain_tracklets = chain_tracklets[:max_tracklets_per_chain]

        n = len(chain_tracklets)
        if n == 0:
            continue

        cols = max(1, max_cols_per_row)
        rows = (n + cols - 1) // cols

        print(
            f"[viz_pairs] Visualizing stitched_id={sid} with {n} tracklets "
            f"(camera={camera_id if camera_id is not None else 'ALL'})."
        )

        head_imgs: List[Optional[np.ndarray]] = []
        tail_imgs: List[Optional[np.ndarray]] = []
        raw_track_ids: List[str] = []

        for t in chain_tracklets:
            head_img, tail_img = _extract_head_tail_frames_from_video(
                t, head_k=head_k, tail_k=tail_k, logger_=log
            )
            head_imgs.append(head_img)
            tail_imgs.append(tail_img)
            raw_track_ids.append(t.raw_track_id)

        # Resize all to common size
        head_resized = _resize_to_common(head_imgs, cell_h, cell_w)
        tail_resized = _resize_to_common(tail_imgs, cell_h, cell_w)

        # For each tracklet: pair_img = [head | tail]  (cell_h x (2*cell_w))
        pair_width = 2 * cell_w
        mosaic_h = cell_h * rows
        mosaic_w = pair_width * cols
        mosaic = np.full((mosaic_h, mosaic_w, 3), 0, dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        for col in range(n):
            h_img = head_resized[col]
            t_img = tail_resized[col]

            row_idx = col // cols
            col_idx = col % cols

            x0 = col_idx * pair_width
            x_mid = x0 + cell_w
            x1 = x0 + pair_width
            y0 = row_idx * cell_h
            y1 = y0 + cell_h

            # left half: head
            mosaic[y0:y1, x0:x_mid] = h_img
            # right half: tail
            mosaic[y0:y1, x_mid:x1] = t_img

            # track_id text centered above pair
            text = raw_track_ids[col]
            cv2.putText(
                mosaic,
                text,
                (x0 + 5, y0 + 15),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                mosaic,
                "H",
                (x0 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                mosaic,
                "T",
                (x_mid + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        out_name = f"stitched_chain_pairs_{camera_id or 'ALL'}_{sid}_N{n}.png"
        out_path = output_dir / out_name

        mosaic_bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), mosaic_bgr)
        print(f"[viz_pairs] Saved stitched chain (pairs) to {out_path}")





##################
def _extract_rep_frame(
    tracklet: Tracklet,
    head_k: int = 3,
    logger_: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """
    Extract a representative (head) frame for plotting.
    If anything goes wrong (video missing / decoding error), return None.
    """
    log = logger_ or logger

    if tracklet.feature_path is None:
        log.warning(f"[plot] Tracklet {tracklet.track_id} has no feature_path.")
        return None

    try:
        feats, frame_ids, video_path, _ = load_embedding(tracklet.feature_path)
    except Exception as e:
        log.exception(
            f"[plot] Failed to load embedding for tracklet {tracklet.track_id}: {e}"
        )
        return None

    frame_ids = np.asarray(frame_ids)
    if frame_ids.size == 0:
        log.warning(f"[plot] Tracklet {tracklet.track_id} has empty frame_ids.")
        return None

    frame_ids_sorted = np.sort(frame_ids)
    head_k = min(head_k, frame_ids_sorted.size)
    head_frame_id = int(frame_ids_sorted[:head_k].mean())

    # Determine video path
    if video_path is None:
        video_path = tracklet.feature_path.with_suffix(".mkv")
    video_path = Path(video_path)

    if not video_path.exists():
        log.warning(
            f"[plot] Video file {video_path} for tracklet {tracklet.track_id} does not exist."
        )
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(
            f"[plot] Failed to open video {video_path} for tracklet {tracklet.track_id}."
        )
        cap.release()
        return None

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if head_frame_id >= n_frames:
        log.warning(
            f"[plot] Frame id {head_frame_id} out of range for {video_path} "
            f"(total={n_frames}) for tracklet {tracklet.track_id}."
        )
        cap.release()
        return None

    def _read_frame_at(fid: int) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret or frame is None:
            log.warning(
                f"[plot] Failed to read frame {fid} from {video_path} "
                f"for tracklet {tracklet.track_id}."
            )
            return None
        return _ensure_rgb(frame)

    img = _read_frame_at(head_frame_id)
    cap.release()
    return img


import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

def _fmt_time(dt: Optional[datetime]) -> str:
    if dt is None:
        return "None"
    return dt.strftime("%H:%M:%S")


def _parse_video_start_from_name(path: Path) -> Optional[datetime]:
    """Parse datetime from raw video filename like ZAG-ELP-CAM-016-20251129-001949-....mp4."""
    name = path.name
    import re

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


def _extract_rep_frame_from_raw(
    tracklet: Tracklet,
    head_k: int = 3,
    logger_: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """
    Extract a representative frame from the raw camera video and draw the bbox,
    using the track CSV to locate frame_id and bbox.
    """
    log = logger_ or logger

    track_csv = None
    if tracklet.track_csv_path and Path(tracklet.track_csv_path).exists():
        track_csv = Path(tracklet.track_csv_path)
    elif tracklet.feature_path:
        candidate = Path(tracklet.feature_path).with_suffix(".csv")
        if candidate.exists():
            track_csv = candidate

    if track_csv is None or not track_csv.exists():
        log.warning(f"[plot] Missing track CSV for tracklet {tracklet.track_id}")
        return None

    try:
        df = pd.read_csv(track_csv)
    except Exception as err:
        log.warning(f"[plot] Failed to read {track_csv}: {err}")
        return None

    if df.empty:
        log.warning(f"[plot] Empty CSV for tracklet {tracklet.track_id}: {track_csv}")
        return None

    k = min(head_k, len(df))
    frame_ids = df["frame_id"].to_numpy()
    rep_frame_id = int(np.mean(frame_ids[:k]))

    # choose the row closest to rep_frame_id for bbox/timestamp
    idx = int(np.argmin(np.abs(frame_ids - rep_frame_id)))
    row = df.iloc[idx]

    bbox = [
        float(row["bbox_top"]),
        float(row["bbox_left"]),
        float(row["bbox_bottom"]),
        float(row["bbox_right"]),
    ]
    timestamp_str = row.get("timestamp", None)
    ts_dt = None
    if isinstance(timestamp_str, str):
        try:
            ts_dt = datetime.fromisoformat(timestamp_str)
        except Exception:
            ts_dt = tracklet.start_timestamp
    else:
        ts_dt = tracklet.start_timestamp

    video_path = tracklet.track_video_path
    if video_path is None:
        video_path = _find_raw_video(tracklet.camera_id, ts_dt)
        tracklet.track_video_path = video_path

    if video_path is None or not Path(video_path).exists():
        log.warning(f"[plot] No raw video found for tracklet {tracklet.track_id} (camera {tracklet.camera_id}).")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"[plot] Failed to open raw video {video_path} for tracklet {tracklet.track_id}.")
        return None

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if rep_frame_id >= n_frames:
        rep_frame_id = max(0, n_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, rep_frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        log.warning(
            f"[plot] Failed to read frame {rep_frame_id} from {video_path} for tracklet {tracklet.track_id}."
        )
        return None

    # Draw bbox and labels on BGR frame
    pt1 = (int(bbox[1]), int(bbox[0]))
    pt2 = (int(bbox[3]), int(bbox[2]))
    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
    sid_text = tracklet.stitched_id if tracklet.stitched_id is not None else "NA"
    label = f"SID {sid_text} | {tracklet.track_id}"
    cv2.putText(frame, label, (pt1[0], max(15, pt1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return _ensure_rgb(frame)


def plot_stitched_ids_on_original_frames(
    tracklets: List[Tracklet],
    camera_id: Optional[str] = None,
    max_ids: Optional[int] = None,
    max_tracklets_per_id: Optional[int] = None,
    head_k: int = 3,  # unused but kept for API compatibility
    cols: int = 5,  # unused but kept for API compatibility
    figsize_per_cell: float = 3.0,  # unused but kept for API compatibility
    output_dir: Path = Path("./stitched_overlays"),
    seconds_stride: float = 1.0,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    Render raw videos with stitched track bboxes overlaid, saving annotated frames as images.

    Args:
        tracklets: list of Tracklet with stitched_id already set.
        camera_id: if not None, only use this camera.
        max_ids: max number of stitched_ids to include.
        max_tracklets_per_id: cap number of tracklets per identity.
        output_dir: where to save annotated images.
        seconds_stride: sample one annotated frame every `seconds_stride` seconds (>=0.1).
    """
    log = logger_ or logger

    # Filter
    filtered: List[Tracklet] = []
    for t in tracklets:
        if t.stitched_id is None:
            continue
        if camera_id is not None and t.camera_id != camera_id:
            continue
        if t.invalid_flag:
            continue
        filtered.append(t)

    if not filtered:
        log.warning(
            "plot_stitched_ids_on_original_frames: no valid tracklets found "
            f"(camera={camera_id if camera_id else 'ALL'})."
        )
        return

    # Limit stitched IDs if requested
    by_sid: Dict[int, List[Tracklet]] = defaultdict(list)
    for t in filtered:
        by_sid[int(t.stitched_id)].append(t)

    stitched_ids = sorted(by_sid.keys())
    if max_ids is not None:
        stitched_ids = stitched_ids[:max_ids]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build overlays per raw video
    video_overlays: Dict[Path, Dict[int, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    color_map: Dict[str, Tuple[int, int, int]] = {}
    rng = np.random.default_rng(42)

    for sid in stitched_ids:
        chain = by_sid[sid]
        chain_sorted = sorted(chain, key=lambda t: t.start_timestamp or datetime.min)
        if max_tracklets_per_id is not None:
            chain_sorted = chain_sorted[:max_tracklets_per_id]

        for t in chain_sorted:
            track_csv = None
            if t.track_csv_path and Path(t.track_csv_path).exists():
                track_csv = Path(t.track_csv_path)
            elif t.feature_path:
                candidate = Path(t.feature_path).with_suffix(".csv")
                if candidate.exists():
                    track_csv = candidate
            if track_csv is None:
                log.warning(f"[plot] Missing track CSV for tracklet {t.track_id}")
                continue

            try:
                df = pd.read_csv(track_csv)
            except Exception as err:
                log.warning(f"[plot] Failed to read {track_csv}: {err}")
                continue
            if df.empty:
                continue

            # Find raw video for this tracklet using its start timestamp
            video_path = t.track_video_path
            if video_path is None:
                video_path = _find_raw_video(t.camera_id, t.start_timestamp)
                t.track_video_path = video_path
            if video_path is None or not Path(video_path).exists():
                log.warning(f"[plot] No raw video found for tracklet {t.track_id} (camera {t.camera_id}).")
                continue

            # Choose a color per stitched_id
            color_key = f"sid-{sid}"
            if color_key not in color_map:
                color_map[color_key] = tuple(int(x) for x in rng.integers(64, 255, size=3))
            color = color_map[color_key]

            for _, row in df.iterrows():
                fid = int(row["frame_id"])
                bbox = (
                    float(row["bbox_top"]),
                    float(row["bbox_left"]),
                    float(row["bbox_bottom"]),
                    float(row["bbox_right"]),
                )
                video_overlays[Path(video_path)][fid].append(
                    {
                        "bbox": bbox,
                        "sid": sid,
                        "tid": t.track_id,
                        "color": color,
                    }
                )

    # Group by stitched_id
    if not video_overlays:
        log.warning("plot_stitched_ids_on_original_frames: no overlays to draw.")
        return

    for video_path, overlay_map in video_overlays.items():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.warning(f"[plot] Failed to open video {video_path}")
            continue

        out_dir = output_dir / f"{video_path.stem}_frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_ids = sorted(overlay_map.keys())
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        stride = max(0.1, float(seconds_stride))
        saved = 0

        # bucket frames by stride seconds; pick frame with most overlays in that bucket
        bucket_best: Dict[int, int] = {}
        for fid in frame_ids:
            sec_bucket = int((fid / fps) / stride) if fps > 0 else fid
            best_fid = bucket_best.get(sec_bucket)
            if best_fid is None or len(overlay_map[fid]) > len(overlay_map[best_fid]):
                bucket_best[sec_bucket] = fid

        for sec_bucket, fid in sorted(bucket_best.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning(f"[plot] Failed to read frame {fid} from {video_path}")
                continue

            overlays = overlay_map.get(fid, [])
            for ov in overlays:
                bbox = ov["bbox"]
                sid = ov["sid"]
                tid = ov["tid"]
                color = ov["color"]
                pt1 = (int(bbox[1]), int(bbox[0]))
                pt2 = (int(bbox[3]), int(bbox[2]))
                cv2.rectangle(frame, pt1, pt2, color, 2)
                label = f"SID {sid} | {tid}"
                cv2.putText(frame, label, (pt1[0], max(15, pt1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out_path = out_dir / f"{video_path.stem}_frame_{fid:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        cap.release()
        log.info(f"[plot] Saved {saved} annotated frames to {out_dir}")
