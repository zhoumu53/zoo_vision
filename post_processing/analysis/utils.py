from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from matplotlib.ticker import MultipleLocator

try:
    from post_processing.analysis.stereotype_classifier.inference import (
        load_model_for_inference,
        predict_label_from_image,
    )
except Exception:
    load_model_for_inference = None
    predict_label_from_image = None

CAMERA_ROOM_PAIRS: dict[int, int] = {
    16: 19,
    19: 16,
    17: 18,
    18: 17,
}

_RAW_VIDEO_RE = re.compile(r"ZAG-ELP-CAM-(\d{3})-(\d{8})-(\d{6})-")


@dataclass(frozen=True)
class RawVideoMeta:
    path: Path
    camera_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    fps: float
    frame_count: int


def _extract_group_from_csv_stem(stem: str) -> list[str]:
    parts = [p for p in str(stem).split("_") if p]
    while parts and parts[-1].isdigit():
        parts.pop()
    return parts


def _extract_camera_ids_from_csv_stem(stem: str) -> list[int]:
    parts = [p for p in str(stem).split("_") if p]
    trailing_nums: list[int] = []
    i = len(parts) - 1
    while i >= 0 and parts[i].isdigit():
        trailing_nums.append(int(parts[i]))
        i -= 1
    if not trailing_nums:
        return []
    trailing_nums.reverse()
    if len(trailing_nums) >= 2:
        trailing_nums = trailing_nums[-2:]
    return sorted(set(trailing_nums))


def _normalized_group_key(names: Iterable[str]) -> tuple[str, ...]:
    out = []
    for n in names:
        s = str(n).strip()
        if s:
            out.append(s.lower())
    return tuple(sorted(set(out)))


def _date_to_dash(date: str) -> str:
    txt = str(date)
    if len(txt) == 8 and txt.isdigit():
        return f"{txt[:4]}-{txt[4:6]}-{txt[6:8]}"
    return txt


def _parse_ids_field(ids_value: object) -> list[str]:
    txt = str(ids_value).strip()
    if not txt:
        return []
    try:
        parsed = ast.literal_eval(txt)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    vals = re.findall(r"[A-Za-z]+", txt)
    return [v for v in vals if v]


def _lookup_other_group_from_paired_summary(
    date: str,
    current_group: list[str],
    paired_summary_csv: Path,
) -> str:
    if not paired_summary_csv.exists():
        return "unknown"
    try:
        df = pd.read_csv(paired_summary_csv)
    except Exception:
        return "unknown"
    if df.empty or "date" not in df.columns or "IDs" not in df.columns:
        return "unknown"

    date_dash = _date_to_dash(date)
    cur_key = _normalized_group_key(current_group)
    same_day = df[df["date"].astype(str).str.strip() == date_dash]
    if same_day.empty:
        return "unknown"
    for _, row in same_day.iterrows():
        ids = _parse_ids_field(row.get("IDs", ""))
        if not ids:
            continue
        if _normalized_group_key(ids) != cur_key:
            return ", ".join(ids)
    return "unknown"


def get_bout_csvs(
    output_dir: Path,
    dates: list[str],
    filename_keyword: str | None = None,
    strict: bool = True,
    paired_summary_csv: Path = Path("/media/mu/zoo_vision/data/semi_gts/paired_summary.csv"),
) -> list[tuple[str, Path, str, str, str]]:
    collected: list[tuple[str, Path, str, str, str]] = []
    for date in dates:
        date_dir = output_dir / date
        if not date_dir.exists():
            continue
        pattern = "*.csv" if not filename_keyword else f"*{filename_keyword}*.csv"
        matching_csvs = sorted(date_dir.glob(pattern))
        all_csvs = sorted(date_dir.glob("*.csv"))
        for csv_path in matching_csvs:
            current_group = _extract_group_from_csv_stem(csv_path.stem)
            current_group_txt = ", ".join(current_group) if current_group else "unknown"
            camera_ids = _extract_camera_ids_from_csv_stem(csv_path.stem)
            camera_ids_txt = ",".join(str(int(c)) for c in sorted(set(camera_ids)))
            cur_key = _normalized_group_key(current_group)

            other_group = "unknown"
            for sibling in all_csvs:
                if sibling == csv_path:
                    continue
                sibling_group = _extract_group_from_csv_stem(sibling.stem)
                if not sibling_group:
                    continue
                if _normalized_group_key(sibling_group) != cur_key:
                    other_group = ", ".join(sibling_group)
                    break

            if other_group == "unknown":
                other_group = _lookup_other_group_from_paired_summary(
                    date=date,
                    current_group=current_group,
                    paired_summary_csv=paired_summary_csv,
                )
            collected.append((date, csv_path, current_group_txt, other_group, camera_ids_txt))

    if not collected and strict:
        keyword_txt = f" containing '{filename_keyword}'" if filename_keyword else ""
        raise FileNotFoundError(
            f"No bout summary CSV files{keyword_txt} found in {output_dir} for dates: {dates}"
        )
    return collected


def _parse_camera_ids(camera_ids_value: object) -> list[int]:
    if camera_ids_value is None:
        return []
    txt = str(camera_ids_value).strip()
    if not txt:
        return []
    ids: list[int] = []
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            ids.append(int(p))
        except ValueError:
            continue
    return sorted(set(ids))


def _expand_room_pair_cameras(camera_ids: Iterable[int]) -> list[int]:
    out = set(int(c) for c in camera_ids)
    for cam in list(out):
        mate = CAMERA_ROOM_PAIRS.get(cam)
        if mate is not None:
            out.add(int(mate))
    return sorted(out)


def _parse_raw_video_start(path: Path) -> tuple[int, pd.Timestamp] | None:
    m = _RAW_VIDEO_RE.search(path.name)
    if m is None:
        return None
    cam = int(m.group(1))
    ts = pd.to_datetime(f"{m.group(2)} {m.group(3)}", format="%Y%m%d %H%M%S", errors="coerce")
    if pd.isna(ts):
        return None
    return cam, pd.Timestamp(ts)


def _video_duration_and_fps(path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0, 0.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0.0 or frame_count <= 0:
        return 0.0, fps, frame_count
    duration_sec = float(frame_count) / float(fps)
    return duration_sec, fps, frame_count


def _candidate_raw_video_paths(
    raw_root: Path,
    camera_id: int,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> list[Path]:
    cam_dir = raw_root / f"ZAG-ELP-CAM-{int(camera_id):03d}"
    if not cam_dir.exists():
        return []

    days = {
        (start_time - pd.Timedelta(days=1)).strftime("%Y%m%d"),
        start_time.strftime("%Y%m%d"),
        end_time.strftime("%Y%m%d"),
        (end_time + pd.Timedelta(days=1)).strftime("%Y%m%d"),
    }
    paths: list[Path] = []
    for day in sorted(days):
        pattern = f"{day}*/ZAG-ELP-CAM-{int(camera_id):03d}-{day}-*.mp4"
        paths.extend(sorted(cam_dir.glob(pattern)))
    return sorted(set(paths))


def find_overlapping_raw_videos(
    raw_root: Path,
    camera_id: int,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> list[RawVideoMeta]:
    metas: list[RawVideoMeta] = []
    for p in _candidate_raw_video_paths(raw_root, camera_id, start_time, end_time):
        parsed = _parse_raw_video_start(p)
        if parsed is None:
            continue
        cam_from_name, video_start = parsed
        if cam_from_name != int(camera_id):
            continue
        duration_sec, fps, frame_count = _video_duration_and_fps(p)
        if duration_sec <= 0.0:
            continue
        video_end = video_start + pd.Timedelta(seconds=duration_sec)
        if video_end <= start_time or video_start >= end_time:
            continue
        metas.append(
            RawVideoMeta(
                path=p,
                camera_id=int(camera_id),
                start_time=video_start,
                end_time=video_end,
                fps=fps,
                frame_count=frame_count,
            )
        )
    return sorted(metas, key=lambda x: x.start_time)


def _write_windowed_video_from_segments(
    segments: list[RawVideoMeta],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    out_path: Path,
    output_size: tuple[int, int] = (1060, 600),
    output_fps: float = 2.0,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        (int(output_size[0]), int(output_size[1])),
    )
    if not writer.isOpened():
        return 0

    total_written = 0
    for seg in segments:
        clip_start = max(start_time, seg.start_time)
        clip_end = min(end_time, seg.end_time)
        if clip_end <= clip_start:
            continue

        cap = cv2.VideoCapture(str(seg.path))
        if not cap.isOpened():
            continue

        fps_in = float(cap.get(cv2.CAP_PROP_FPS))
        if fps_in <= 0:
            cap.release()
            continue
        in_step = max(1, int(round(float(fps_in) / float(output_fps))))

        frame_start = int(max(0.0, (clip_start - seg.start_time).total_seconds()) * fps_in)
        frame_end = int(max(0.0, (clip_end - seg.start_time).total_seconds()) * fps_in)
        if frame_end <= frame_start:
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        next_sample = frame_start
        frame_idx = frame_start
        while frame_idx < frame_end:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= next_sample:
                resized = cv2.resize(frame, (int(output_size[0]), int(output_size[1])), interpolation=cv2.INTER_AREA)
                writer.write(resized)
                total_written += 1
                next_sample += in_step
            frame_idx += 1
        cap.release()

    writer.release()
    if total_written == 0 and out_path.exists():
        out_path.unlink(missing_ok=True)
    return total_written


def _select_camera_pair(camera_ids: list[int]) -> tuple[int, int] | None:
    if not camera_ids:
        return None
    cams = sorted(set(int(c) for c in camera_ids))
    for cam in cams:
        mate = CAMERA_ROOM_PAIRS.get(cam)
        if mate is not None and mate in cams:
            a, b = sorted((int(cam), int(mate)))
            return (a, b)
    if len(cams) >= 2:
        return (cams[0], cams[1])
    mate = CAMERA_ROOM_PAIRS.get(cams[0])
    if mate is not None:
        a, b = sorted((int(cams[0]), int(mate)))
        return (a, b)
    return None


def _close_reader_state(state: dict) -> None:
    cap = state.get("cap")
    if cap is not None:
        cap.release()
    state["cap"] = None
    state["seg_idx"] = -1


def _read_frame_at_timestamp(
    segments: list[RawVideoMeta],
    ts: pd.Timestamp,
    state: dict,
    out_size: tuple[int, int] | None = None,
) -> np.ndarray | None:
    if not segments:
        return None

    seg_idx = state.get("seg_idx", -1)
    active_seg = segments[seg_idx] if (0 <= seg_idx < len(segments)) else None
    if active_seg is None or not (active_seg.start_time <= ts < active_seg.end_time):
        matched_idx = -1
        for i, seg in enumerate(segments):
            if seg.start_time <= ts < seg.end_time:
                matched_idx = i
                break
        if matched_idx < 0:
            _close_reader_state(state)
            return None
        if matched_idx != seg_idx:
            _close_reader_state(state)
            cap = cv2.VideoCapture(str(segments[matched_idx].path))
            if not cap.isOpened():
                _close_reader_state(state)
                return None
            state["cap"] = cap
            state["seg_idx"] = matched_idx
        active_seg = segments[matched_idx]

    cap = state.get("cap")
    if cap is None:
        return None
    fps = float(active_seg.fps)
    if fps <= 0:
        return None
    frame_idx = int(max(0.0, (ts - active_seg.start_time).total_seconds()) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if out_size is None:
        return frame
    return cv2.resize(frame, (int(out_size[0]), int(out_size[1])), interpolation=cv2.INTER_AREA)


def _resize_keep_aspect_by_width(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame
    out_h = max(1, int(round(float(h) * float(target_width) / float(w))))
    return cv2.resize(frame, (int(target_width), int(out_h)), interpolation=cv2.INTER_AREA)


def _infer_resized_height_from_segments(
    segments: list[RawVideoMeta],
    ts: pd.Timestamp,
    target_width: int,
    default_height: int,
) -> int:
    state: dict = {"seg_idx": -1, "cap": None}
    frame = _read_frame_at_timestamp(segments, ts, state, out_size=None)
    _close_reader_state(state)
    if frame is None:
        return int(default_height)
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return int(default_height)
    return max(1, int(round(float(h) * float(target_width) / float(w))))


def _write_synced_pair_video(
    segments_a: list[RawVideoMeta],
    segments_b: list[RawVideoMeta],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    out_path: Path,
    output_size: tuple[int, int] = (1060, 600),
    output_fps: float = 2.0,
) -> int:
    target_width = int(output_size[0])
    default_height = int(output_size[1])
    height_a = _infer_resized_height_from_segments(segments_a, start_time, target_width, default_height)
    height_b = _infer_resized_height_from_segments(segments_b, start_time, target_width, default_height)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        (int(target_width), int(height_a + height_b)),
    )
    if not writer.isOpened():
        return 0

    black_a = np.zeros((int(height_a), int(target_width), 3), dtype=np.uint8)
    black_b = np.zeros((int(height_b), int(target_width), 3), dtype=np.uint8)
    state_a: dict = {"seg_idx": -1, "cap": None}
    state_b: dict = {"seg_idx": -1, "cap": None}
    step = pd.Timedelta(seconds=(1.0 / float(output_fps)))

    n_written = 0
    ts = pd.Timestamp(start_time)
    while ts < end_time:
        frame_a = _read_frame_at_timestamp(segments_a, ts, state_a, out_size=None)
        frame_b = _read_frame_at_timestamp(segments_b, ts, state_b, out_size=None)
        if frame_a is None and frame_b is None:
            ts = ts + step
            continue
        if frame_a is not None:
            frame_a = _resize_keep_aspect_by_width(frame_a, target_width)
            if frame_a.shape[0] != height_a:
                frame_a = cv2.resize(frame_a, (target_width, height_a), interpolation=cv2.INTER_AREA)
        else:
            frame_a = black_a
        if frame_b is not None:
            frame_b = _resize_keep_aspect_by_width(frame_b, target_width)
            if frame_b.shape[0] != height_b:
                frame_b = cv2.resize(frame_b, (target_width, height_b), interpolation=cv2.INTER_AREA)
        else:
            frame_b = black_b
        writer.write(cv2.vconcat([frame_a, frame_b]))
        n_written += 1
        ts = ts + step

    _close_reader_state(state_a)
    _close_reader_state(state_b)
    writer.release()
    if n_written == 0 and out_path.exists():
        out_path.unlink(missing_ok=True)
    return n_written


def export_stereotypy_event_videos_from_csv(
    stereotypy_csv: Path,
    out_root: Path,
    individual_group: str,
    raw_root: Path = Path("/mnt/camera_nas"),
    output_size: tuple[int, int] = (1060, 600),
    output_fps: float = 2.0,
) -> Path | None:
    if not stereotypy_csv.exists():
        return None

    df = pd.read_csv(stereotypy_csv)
    required = {"start_timestamp", "end_timestamp", "date", "camera_ids"}
    if not required.issubset(df.columns):
        return None
    if df.empty:
        return None

    df = df.copy()
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], errors="coerce")
    df["end_timestamp"] = pd.to_datetime(df["end_timestamp"], errors="coerce")
    df = df.dropna(subset=["start_timestamp", "end_timestamp"])
    if df.empty:
        return None

    out_video_dir = Path(out_root) / "videos"
    out_video_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing stereotypy events"):
        start_ts = pd.Timestamp(row["start_timestamp"])
        end_ts = pd.Timestamp(row["end_timestamp"])
        if end_ts <= start_ts:
            continue

        cams_observed = _parse_camera_ids(row.get("camera_ids", ""))
        cams_target = _expand_room_pair_cameras(cams_observed)
        cam_pair = _select_camera_pair(cams_target)
        if cam_pair is None:
            continue
        cam_a, cam_b = cam_pair

        cams_tag = f"{cam_a:03d}-{cam_b:03d}"
        window_tag = f"{start_ts:%Y%m%d_%H%M%S}_{end_ts:%Y%m%d_%H%M%S}"
        segments_a = find_overlapping_raw_videos(
            raw_root=Path(raw_root),
            camera_id=int(cam_a),
            start_time=start_ts,
            end_time=end_ts,
        )
        segments_b = find_overlapping_raw_videos(
            raw_root=Path(raw_root),
            camera_id=int(cam_b),
            start_time=start_ts,
            end_time=end_ts,
        )
        if not segments_a and not segments_b:
            continue

        out_name = f"{individual_group}_cams{cams_tag}_sterotype_{window_tag}.mp4"
        out_path = out_video_dir / out_name
        n_written = _write_synced_pair_video(
            segments_a=segments_a,
            segments_b=segments_b,
            start_time=start_ts,
            end_time=end_ts,
            out_path=out_path,
            output_size=output_size,
            output_fps=output_fps,
        )
        if n_written <= 0:
            continue

        manifest_rows.append({
            "individual_group": str(individual_group),
            "date": str(row["date"]),
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "camera_ids_from_event": ",".join(str(c) for c in cams_observed),
            "camera_pair_target": f"{cam_a},{cam_b}",
            "n_source_segments_cam_a": int(len(segments_a)),
            "n_source_segments_cam_b": int(len(segments_b)),
            "n_output_frames": int(n_written),
            "output_video": str(out_path),
            "source_videos_cam_a": ";".join(str(s.path) for s in segments_a),
            "source_videos_cam_b": ";".join(str(s.path) for s in segments_b),
        })

    if not manifest_rows:
        return None

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["date", "start_timestamp", "camera_pair_target"]
    ).reset_index(drop=True)
    manifest_csv = out_video_dir / f"{individual_group}_stereotypy_video_manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    return manifest_csv


def build_ethogram_dataframe(
    identity_id: str,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    required = {"start_time", "end_time", "behavior_label"}
    if not required.issubset(segments.columns):
        missing = sorted(required.difference(segments.columns))
        raise ValueError(f"segments missing required columns: {missing}")
    return pd.DataFrame(
        {
            "identity_id": str(identity_id),
            "start_dt": segments["start_time"],
            "end_dt": segments["end_time"],
            "label": segments["behavior_label"],
        }
    )


def save_ethogram_csv(
    identity_id: str,
    segments: pd.DataFrame,
    out_csv: Path,
) -> pd.DataFrame:
    ethogram = build_ethogram_dataframe(identity_id=identity_id, segments=segments)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ethogram.to_csv(out_csv, index=False)
    return ethogram


def analyze_ethogram_and_plot_activity_budget(
    ethogram_csv: Path,
    out_plot: Path,
    date: str,
    camera_ids: Iterable[int] | str | None = None,
    other_group: str | None = None,
    title_prefix: str = "Activity budget for",
    label_display_map: dict[str, str] | None = None,
    label_color_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not ethogram_csv.exists():
        raise FileNotFoundError(f"Ethogram CSV not found: {ethogram_csv}")

    eth = pd.read_csv(ethogram_csv)
    required = {"start_dt", "end_dt", "label"}
    if not required.issubset(eth.columns):
        missing = sorted(required.difference(eth.columns))
        raise ValueError(f"ethogram missing required columns: {missing}")

    eth = eth.copy()
    eth["start_dt"] = pd.to_datetime(eth["start_dt"], errors="coerce")
    eth["end_dt"] = pd.to_datetime(eth["end_dt"], errors="coerce")
    eth = eth.dropna(subset=["start_dt", "end_dt", "label"])
    if eth.empty:
        budget = pd.DataFrame(columns=["label", "duration_min", "percentage"])
    else:
        eth["duration_min"] = (eth["end_dt"] - eth["start_dt"]).dt.total_seconds() / 60.0
        eth = eth[eth["duration_min"] > 0]
        if eth.empty:
            budget = pd.DataFrame(columns=["label", "duration_min", "percentage"])
        else:
            budget = (
                eth.groupby("label", as_index=False)["duration_min"]
                .sum()
                .sort_values("duration_min", ascending=False)
                .reset_index(drop=True)
            )
            total_min = float(budget["duration_min"].sum())
            budget["percentage"] = (
                budget["duration_min"] / total_min * 100.0 if total_min > 0 else 0.0
            )

    if camera_ids is None:
        camera_txt = "unknown"
    elif isinstance(camera_ids, str):
        camera_txt = camera_ids if camera_ids.strip() else "unknown"
    else:
        camera_txt = ",".join(str(int(c)) for c in sorted(set(int(c) for c in camera_ids)))
        if not camera_txt:
            camera_txt = "unknown"

    other_group_txt = str(other_group).strip() if other_group is not None else ""
    if not other_group_txt:
        other_group_txt = "unknown"
    total_hours = budget["duration_min"].sum() / 60.0
    title = (
        f"{title_prefix} {date}, \n camera ids {camera_txt}\n"
        f"total {total_hours:.1f} hours\n"
        f"other group {other_group_txt}\n"
    )
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    if budget.empty:
        ax.text(0.5, 0.5, "No valid observations", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        labels = []
        durations = []
        duration_labels= []
        for _, row in budget.iterrows():
            raw_label = str(row["label"])
            display_label = (
                label_display_map.get(raw_label, raw_label)
                if label_display_map is not None
                else raw_label
            )
            durations.append(float(row["duration_min"]))
            duration_labels.append(f"{row['duration_min']:.1f} min")
            labels.append(display_label)
        if label_color_map is not None:
            colors = [
                label_color_map.get(str(row["label"]), "#9E9E9E")
                for _, row in budget.iterrows()
            ]
        else:
            colors = plt.cm.rainbow(np.linspace(0.0, 1.0, len(budget)))
        wedges, _, _ = ax.pie(
            budget["duration_min"].to_numpy(dtype=float),
            labels=duration_labels,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            labeldistance=1.12,
            pctdistance=0.72,
            colors=colors,
        )
        ax.legend(
            wedges,
            labels,
            title="Labels",
            loc="upper right",
            bbox_to_anchor=(0.01, 0.99),
            frameon=False,
        )
        ax.axis("equal")
    ax.set_title(title)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_plot, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return budget


_STEREOTYPY_MODEL_CHECKPOINT = Path(
    "/media/mu/zoo_vision/post_processing/analysis/stereotype_classifier/model.pt"
)
_STEREOTYPY_INFER_BUNDLE = None
_TRAJ_STANDING_WALKING_LABELS = ["01_standing", "walking"]
_TRAJ_CMAP = {
    "01_standing": "Spectral",
    "walking": "cool",
}


def _get_stereotypy_inference_bundle():
    global _STEREOTYPY_INFER_BUNDLE
    if _STEREOTYPY_INFER_BUNDLE is not None:
        return _STEREOTYPY_INFER_BUNDLE
    if load_model_for_inference is None:
        print("Stereotypy inference import unavailable; falling back to label='no'.")
        return None
    if not _STEREOTYPY_MODEL_CHECKPOINT.exists():
        print(f"Stereotypy checkpoint not found: {_STEREOTYPY_MODEL_CHECKPOINT}; falling back to label='no'.")
        return None
    try:
        _STEREOTYPY_INFER_BUNDLE = load_model_for_inference(_STEREOTYPY_MODEL_CHECKPOINT)
    except Exception as exc:
        print(f"Failed to load stereotypy model: {exc}; falling back to label='no'.")
        _STEREOTYPY_INFER_BUNDLE = None
    return _STEREOTYPY_INFER_BUNDLE


def _resolve_hourly_traj_behaviors(behaviors: Iterable[str] | None) -> list[str]:
    if behaviors is None:
        return list(_TRAJ_STANDING_WALKING_LABELS)
    resolved: list[str] = []
    for item in behaviors:
        key = str(item).strip()
        if key in _TRAJ_STANDING_WALKING_LABELS and key not in resolved:
            resolved.append(key)
    if not resolved:
        return list(_TRAJ_STANDING_WALKING_LABELS)
    return resolved


def _camera_id_to_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    txt = str(value).strip()
    if not txt:
        return None
    m = re.search(r"(\d+)$", txt)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _get_camera_pairs(points: pd.DataFrame) -> pd.Series:
    if points is None or points.empty or "camera_id" not in points.columns:
        return pd.Series(dtype=float)
    return points["camera_id"].map(_camera_id_to_int)


def _timestamp_snaps_from_points(
    points: pd.DataFrame,
    gap_seconds: float = 3.0,
    min_duration_seconds: float = 1.0,
    merge_gap_seconds: float = 0.0,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if points is None or points.empty or "timestamp" not in points.columns:
        return []
    ts = pd.to_datetime(points["timestamp"], errors="coerce").dropna().sort_values().drop_duplicates()
    if ts.empty:
        return []

    gap = pd.Timedelta(seconds=float(gap_seconds))
    min_dur = pd.Timedelta(seconds=float(min_duration_seconds))
    snaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    seg_start = pd.Timestamp(ts.iloc[0])
    prev = pd.Timestamp(ts.iloc[0])
    for i in range(1, len(ts)):
        cur = pd.Timestamp(ts.iloc[i])
        if (cur - prev) > gap:
            seg_end = prev
            if seg_end <= seg_start:
                seg_end = seg_start + min_dur
            snaps.append((seg_start, seg_end))
            seg_start = cur
        prev = cur

    seg_end = prev
    if seg_end <= seg_start:
        seg_end = seg_start + min_dur
    snaps.append((seg_start, seg_end))
    if merge_gap_seconds <= 0 or len(snaps) <= 1:
        return snaps

    merge_gap = pd.Timedelta(seconds=float(merge_gap_seconds))
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_start, cur_end = snaps[0]
    for nxt_start, nxt_end in snaps[1:]:
        if (nxt_start - cur_end) <= merge_gap:
            if nxt_end > cur_end:
                cur_end = nxt_end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = nxt_start, nxt_end
    merged.append((cur_start, cur_end))
    return merged


def _plot_world_heatmap_standing_walking_bin(
    df_traj: pd.DataFrame,
    out_dir: Path,
    title: str,
    night_start: pd.Timestamp,
    night_end: pd.Timestamp,
    bin_hours: float,
    behaviors: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bin_hours <= 0:
        raise ValueError("bin_hours must be positive")

    behaviors_to_show = _resolve_hourly_traj_behaviors(behaviors)
    d = df_traj.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp", "world_x", "world_y"])
    d = d[d["behavior_label"].isin(behaviors_to_show)]
    d = d[(d["timestamp"] >= night_start) & (d["timestamp"] <= night_end)]
    stereotypy_cols = [
        "start_timestamp",
        "end_timestamp",
        "date",
        "camera_ids",
        "real_camera_ids",
        "track_filenames",
        "stereotypy_score",
        "prediction_label",
    ]
    debug_cols = [
        "start_timestamp",
        "end_timestamp",
        "behavior_label",
        "camera_ids",
        "n_points",
        "plot_path",
        "prediction_label",
        "stereotypy_score",
        "is_stereotypy",
    ]
    if d.empty:
        return pd.DataFrame(columns=stereotypy_cols), pd.DataFrame(columns=debug_cols)

    x_min, x_max = float(d["world_x"].min()), float(d["world_x"].max())
    y_min, y_max = float(d["world_y"].min()), float(d["world_y"].max())

    def _interval_step(vmin: float, vmax: float) -> int:
        span = max(0.0, float(vmax) - float(vmin))
        if span <= 0.0:
            return 5
        step = int(round(span / 8.0))
        return int(min(15, max(5, step)))

    bin_delta = pd.Timedelta(hours=float(bin_hours))
    time_edges = pd.date_range(start=night_start, end=night_end, freq=bin_delta)
    if len(time_edges) == 0 or time_edges[-1] < night_end:
        time_edges = time_edges.append(pd.DatetimeIndex([night_end]))
    if len(time_edges) < 2:
        time_edges = pd.DatetimeIndex([night_start, night_end])

    n_bins = len(time_edges) - 1
    out_dir.mkdir(parents=True, exist_ok=True)
    infer_bundle = _get_stereotypy_inference_bundle()

    stereotypy_events: list[dict] = []
    debug_rows: list[dict] = []
    for r in range(n_bins):
        b_start = time_edges[r]
        b_end = time_edges[r + 1]
        in_bin = (d["timestamp"] >= b_start) & (d["timestamp"] < b_end if r < (n_bins - 1) else d["timestamp"] <= b_end)
        d_bin = d[in_bin]
        if d_bin.empty:
            continue

        for beh in behaviors_to_show:
            g = d_bin[d_bin["behavior_label"] == beh]
            if g.empty:
                continue

            cam_ids_series = _get_camera_pairs(g)
            cam_ids_list: list[int] = []
            if not cam_ids_series.empty:
                cam_ids_list = sorted(pd.Series(cam_ids_series).dropna().astype(int).unique().tolist())
            cam_ids_str = ",".join(str(cid) for cid in cam_ids_list)

            slot_png = out_dir / f"{title}_{beh}_{b_start:%Y%m%d_%H%M%S}_{b_end:%Y%m%d_%H%M%S}.png"
            fig_single, ax_single = plt.subplots(figsize=(6, 6), dpi=180, facecolor="black")
            ax_single.set_facecolor("black")
            ax_single.hexbin(
                g["world_x"].to_numpy(),
                g["world_y"].to_numpy(),
                gridsize=120,
                bins="log",
                cmap=_TRAJ_CMAP.get(beh, "Greys"),
                mincnt=1,
                alpha=0.90,
            )
            ax_single.set_xlim(x_min, x_max)
            ax_single.set_ylim(y_min, y_max)
            ax_single.set_aspect("equal", adjustable="box")
            ax_single.xaxis.set_major_locator(MultipleLocator(_interval_step(x_min, x_max)))
            ax_single.yaxis.set_major_locator(MultipleLocator(_interval_step(y_min, y_max)))
            ax_single.set_title("")
            ax_single.set_xlabel("")
            ax_single.set_ylabel("")
            ax_single.tick_params(colors="white")
            ax_single.set_xticks([])
            ax_single.set_yticks([])
            for spine in ax_single.spines.values():
                spine.set_color("white")
            ax_single.grid(False)
            fig_single.tight_layout()
            fig_single.savefig(slot_png, facecolor=fig_single.get_facecolor(), bbox_inches="tight")
            plt.close(fig_single)

            prediction_label = "no"
            if infer_bundle is not None and predict_label_from_image is not None:
                try:
                    prediction_label = str(
                        predict_label_from_image(
                            slot_png,
                            infer_bundle,
                            input_is_bgr=False,
                        )
                    )
                except Exception as exc:
                    print(f"Inference failed for {slot_png.name}: {exc}. Using label='no'.")
                    prediction_label = "no"
            is_stereotypy = prediction_label == "yes"

            debug_rows.append(
                {
                    "start_timestamp": pd.Timestamp(b_start),
                    "end_timestamp": pd.Timestamp(b_end),
                    "behavior_label": str(beh),
                    "camera_ids": cam_ids_str,
                    "n_points": int(len(g)),
                    "plot_path": str(slot_png),
                    "prediction_label": prediction_label,
                    "stereotypy_score": 1.0 if is_stereotypy else 0.0,
                    "is_stereotypy": bool(is_stereotypy),
                }
            )

            if is_stereotypy and beh == "walking":
                snaps = _timestamp_snaps_from_points(
                    g,
                    gap_seconds=3.0,
                    min_duration_seconds=1.0,
                    merge_gap_seconds=60.0,
                )
                if not snaps:
                    snaps = [(pd.Timestamp(b_start), pd.Timestamp(b_end))]
                for snap_start, snap_end in snaps:
                    g_snap = g[
                        (g["timestamp"] >= snap_start) & (g["timestamp"] <= snap_end)
                    ].copy()
                    cam_ids: list[str] = []
                    if "camera_id" in g_snap.columns:
                        cam_ids = sorted(pd.Series(g_snap["camera_id"]).dropna().astype(str).unique().tolist())
                    track_tags: list[str] = []
                    if "track_filename" in g_snap.columns and "camera_id" in g_snap.columns:
                        track_pairs = (
                            g_snap[["track_filename", "camera_id"]]
                            .dropna()
                            .astype(str)
                            .drop_duplicates()
                        )
                        track_tags = sorted(
                            f"{row1['track_filename']}_{row1['camera_id']}"
                            for _, row1 in track_pairs.iterrows()
                        )
                    stereotypy_events.append(
                        {
                            "start_timestamp": pd.Timestamp(snap_start),
                            "end_timestamp": pd.Timestamp(snap_end),
                            "date": pd.Timestamp(snap_start).strftime("%Y%m%d"),
                            "camera_ids": ",".join(cam_ids),
                            "real_camera_ids": ",".join(cam_ids),
                            "track_filenames": ",".join(track_tags),
                            "stereotypy_score": 1.0,
                            "prediction_label": prediction_label,
                        }
                    )

    if not debug_rows:
        return pd.DataFrame(columns=stereotypy_cols), pd.DataFrame(columns=debug_cols)

    debug_df = pd.DataFrame(debug_rows)
    for col in debug_cols:
        if col not in debug_df.columns:
            debug_df[col] = np.nan
    debug_df = debug_df[debug_cols].copy()

    if not stereotypy_events:
        return pd.DataFrame(columns=stereotypy_cols), debug_df

    out_flags = pd.DataFrame(stereotypy_events)
    out_flags = out_flags.drop_duplicates(
        subset=[
            "start_timestamp",
            "end_timestamp",
            "date",
            "camera_ids",
            "real_camera_ids",
            "track_filenames",
            "prediction_label",
        ]
    )
    out_flags = out_flags.sort_values(["start_timestamp", "end_timestamp"]).reset_index(drop=True)
    for col in stereotypy_cols:
        if col not in out_flags.columns:
            out_flags[col] = np.nan
    out_flags = out_flags[stereotypy_cols].copy()
    return out_flags, debug_df
