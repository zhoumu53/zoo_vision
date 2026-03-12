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
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import Patch
import json

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


CAMERA_TO_ROOM_LABEL: dict[int, str] = {
    16: "Enclosure1 (w/o pool)",
    19: "Enclosure1 (w/o pool)",
    17: "Enclosure2 (w. pool)",
    18: "Enclosure2 (w. pool)",
}

_RAW_VIDEO_RE = re.compile(r"ZAG-ELP-CAM-(\d{3})-(\d{8})-(\d{6})-")


VALID_LABELS = {
    "01_standing",
    "02_sleeping_left",
    "03_sleeping_right",
    "walking",
    "stereotypy",
}

LABEL_ORDER = [
    "01_standing",
    "walking",
    "02_sleeping_left",
    "03_sleeping_right",
    "stereotypy",
    "no_observation",
]

LABEL_DISPLAY = {
    "01_standing": "standing",
    "02_sleeping_left": "lateral recumbancy (left)",
    "03_sleeping_right": "lateral recumbancy (right)",
    "no_observation": "outside / no observation",
    "walking": "locomotion",
    "stereotypy": "route tracing",
}


LABEL_COLORS = {
    "02_sleeping_left": "#5FB13E",
    "03_sleeping_right": "#FF9A2A",
    "no_observation": "#F1F1F1",
    "stereotypy": "#FF2624",
    "01_standing": "#8a00ac",
    "walking": "#c372cb",
}

GT_LABEL_COLORS = {
    "02_sleeping_left": "#539B32",
    "03_sleeping_right": "#FFC300",
}

GT_LABEL_MAP = {
    "sleep_left": "02_sleeping_left",
    "sleeping_left": "02_sleeping_left",
    "left_sleep": "02_sleeping_left",
    "sleep_right": "03_sleeping_right",
    "sleeping_right": "03_sleeping_right",
    "right_sleep": "03_sleeping_right",
    "standing": "01_standing",
    "stand": "01_standing",
    "walking": "walking",
    "walk": "walking",
    "stereotypy": "stereotypy",
}

TRAJ_HEATMAP_LABELS = [
    "01_standing",
    "walking",
    "02_sleeping_left",
    "03_sleeping_right",
]

TRAJ_STANDING_WALKING_LABELS = [
    "01_standing",
    "walking",
]

LABEL_PRIORITY = {
    "no_observation": 0,
    "01_standing": 1,
    "02_sleeping_left": 2,
    "03_sleeping_right": 2,
    "walking": 3,
    "stereotypy": 4,
}

STEREOTYPY_CAMERA_IDS = {16, 19}   ### now the stereotypy only for 016-019, the trajs from 017-018 are not good enough for stereotypy analysis


STEREOTYPY_MODEL_CHECKPOINT = Path(
    "/media/mu/zoo_vision/post_processing/analysis/stereotype_classifier/model.pt"
)
_STEREOTYPY_BUNDLE: Any = None



def _trajectory_behaviour_cmap_map() -> dict[str, str]:
    return {
        "01_standing": "Spectral",
        "02_sleeping_left": "summer",
        "03_sleeping_right": "Wistia",
        "walking": "cool",
        "stereotypy": "cool",
    }


def _get_submap_background() -> tuple[np.ndarray, np.ndarray] | None:
    """Load submap image + world->submap transform using track_heatmap settings."""
    repo_root = Path(__file__).resolve().parents[2]
    floorplan_path = repo_root / "data" / "kkep_floorplan.png"
    config_path = repo_root / "data" / "config.json"
    if not floorplan_path.exists() or not config_path.exists():
        return None

    try:
        with config_path.open() as f:
            cfg = json.load(f)
        t_map_from_world2 = np.asarray(cfg["map"]["T_map_from_world2"], dtype=float)
        
    except Exception as exc:
        print(f"Warning: failed to load map background ({exc}); fallback to world_x/world_y heatmap.")
        return None
    im_map = cv2.imread(str(floorplan_path))
    if im_map.ndim == 3 and im_map.shape[2] > 3:
        im_map = im_map[:, :, :3]

    submap_x = 1450
    submap_y = 1300
    submap_w = 1250
    submap_h = 900
    submap_scale = 0.25

    im_sub = im_map[
        submap_y : (submap_y + submap_h),
        submap_x : (submap_x + submap_w) :,
    ]
    im_sub = cv2.resize(
        im_sub,
        dsize=None,
        fx=submap_scale,
        fy=submap_scale,
        interpolation=cv2.INTER_AREA,
    )
    cv2.cvtColor(im_sub, cv2.COLOR_BGR2RGB, im_sub)

    t_sub_from_world2 = (
        np.array(
            [
                [submap_scale, 0.0, -submap_scale * submap_x],
                [0.0, submap_scale, -submap_scale * submap_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        @ t_map_from_world2
    )
    return im_sub, t_sub_from_world2


def _world_to_submap_xy(
    world_x: np.ndarray,
    world_y: np.ndarray,
    t_sub_from_world2: np.ndarray,
    sub_w: int,
    sub_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(world_x) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    pts_h = np.column_stack((world_x.astype(float), world_y.astype(float), np.ones(len(world_x), dtype=float)))
    map_h = (t_sub_from_world2 @ pts_h.T).T
    u = map_h[:, 0]
    v = map_h[:, 1]
    valid = (u >= 0.0) & (u < float(sub_w)) & (v >= 0.0) & (v < float(sub_h))
    return u[valid], v[valid]


def plot_world_heatmap_by_behaviour_after_stereotypy(
    df_traj: pd.DataFrame,
    out_path: Path,
    title: str,
    night_start: pd.Timestamp | None = None,
    night_end: pd.Timestamp | None = None,
    df_stereotypy_windows: pd.DataFrame | None = None,
    separate: bool = True,
) -> None:
    """Plot world heatmap by behavior using final stereotypy windows."""

    d = df_traj.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp", "world_x", "world_y"])
    if d.empty:
        return

    if night_start is not None:
        d = d[d["timestamp"] >= pd.Timestamp(night_start)]
    if night_end is not None:
        d = d[d["timestamp"] <= pd.Timestamp(night_end)]
    if d.empty:
        return

    if df_stereotypy_windows is not None and not df_stereotypy_windows.empty:
        windows = df_stereotypy_windows.copy()
        windows["start_time"] = pd.to_datetime(windows["start_time"], errors="coerce")
        windows["end_time"] = pd.to_datetime(windows["end_time"], errors="coerce")
        windows = windows.dropna(subset=["start_time", "end_time"])
        if not windows.empty:
            for _, row in windows.iterrows():
                s = pd.Timestamp(row["start_time"])
                e = pd.Timestamp(row["end_time"])
                if s >= e:
                    continue
                in_window = (d["timestamp"] >= s) & (d["timestamp"] < e)
                d.loc[in_window & (d["behavior_label"] == "walking"), "behavior_label"] = "stereotypy"

    behaviors_order = [*TRAJ_HEATMAP_LABELS, "stereotypy"]
    d = d[d["behavior_label"].isin(behaviors_order)]
    if d.empty:
        return

    cmap_for = _trajectory_behaviour_cmap_map()
    bg_info = _get_submap_background()
    use_map_bg = bg_info is not None
    if use_map_bg:
        im_sub, t_sub_from_world2 = bg_info
        sub_h, sub_w = int(im_sub.shape[0]), int(im_sub.shape[1])
    else:
        x_min, x_max = float(d["world_x"].min()), float(d["world_x"].max())
        y_min, y_max = float(d["world_y"].min()), float(d["world_y"].max())

    def _render_labels(label_group: list[str], save_path: Path, panel_title: str) -> None:
        gg = d[d["behavior_label"].isin(label_group)].copy()
        if gg.empty:
            return
        fig, ax = plt.subplots(1, 1, figsize=(9, 7), dpi=300, facecolor="black")
        ax.set_facecolor("black")

        if use_map_bg:
            ax.imshow(im_sub, zorder=0)

        plotted_any = False
        for beh in label_group:
            g = gg[gg["behavior_label"] == beh]
            if g.empty:
                continue
            if use_map_bg:
                xs, ys = _world_to_submap_xy(
                    g["world_x"].to_numpy(dtype=float),
                    g["world_y"].to_numpy(dtype=float),
                    t_sub_from_world2,
                    sub_w,
                    sub_h,
                )
                if len(xs) == 0:
                    continue
                ax.hexbin(
                    xs,
                    ys,
                    gridsize=140,
                    bins="log",
                    cmap=cmap_for.get(str(beh), "Greys"),
                    mincnt=1,
                    alpha=0.78,
                    zorder=1,
                )
                plotted_any = True
            else:
                ax.hexbin(
                    g["world_x"].to_numpy(),
                    g["world_y"].to_numpy(),
                    gridsize=140,
                    bins="log",
                    cmap=cmap_for.get(str(beh), "Greys"),
                    mincnt=1,
                    alpha=0.85,
                )
                plotted_any = True

        if not plotted_any:
            plt.close(fig)
            return

        if use_map_bg:
            ax.set_xlim(0, sub_w)
            ax.set_ylim(sub_h, 0)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        else:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        ax.set_title(panel_title, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Legend only for merged sleeping panel (left vs right).
        if set(label_group) == {"02_sleeping_left", "03_sleeping_right"}:
            legend_handles = [
                Patch(facecolor=LABEL_COLORS["02_sleeping_left"], edgecolor="none", label=LABEL_DISPLAY["02_sleeping_left"]),
                Patch(facecolor=LABEL_COLORS["03_sleeping_right"], edgecolor="none", label=LABEL_DISPLAY["03_sleeping_right"]),
            ]
            ax.legend(handles=legend_handles, ncol=1, frameon=False, loc="upper right")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=300)
        plt.close(fig)

    if separate:
        stem = out_path.stem
        suffix = out_path.suffix if out_path.suffix else ".png"
        group_defs: list[tuple[str, list[str], str]] = [
            ("standing", ["01_standing"], LABEL_DISPLAY.get("01_standing", "standing")),
            ("walking", ["walking"], LABEL_DISPLAY.get("walking", "walking")),
            ("stereotypy", ["stereotypy"], LABEL_DISPLAY.get("stereotypy", "stereotypy")),
            (
                "sleeping",
                ["02_sleeping_left", "03_sleeping_right"],
                f"{LABEL_DISPLAY.get('02_sleeping_left', 'sleeping left')} + {LABEL_DISPLAY.get('03_sleeping_right', 'sleeping right')}",
            ),
        ]
        for tag, labels, label_txt in group_defs:
            if not d["behavior_label"].isin(labels).any():
                continue
            out_i = out_path.with_name(f"{stem}_{tag}{suffix}")
            if 'lateral' in label_txt:
                label_txt = "lateral recumbancy"
            _render_labels(labels, out_i, f"Trajectory ({label_txt})")
    else:
        _render_labels([b for b in behaviors_order if (d["behavior_label"] == b).any()], out_path, title)


def camera_ids_to_room_label_text(camera_ids: object) -> str:
    """Map camera id(s) to room label(s) for display text.

    If input already looks like room-label text (no pure numeric camera ids),
    it is returned unchanged to avoid double conversion.
    """
    if camera_ids is None:
        return "unknown"
    if isinstance(camera_ids, float) and pd.isna(camera_ids):
        return "unknown"

    cams: list[int] = []
    passthrough_txt = ""

    if isinstance(camera_ids, str):
        txt = camera_ids.strip()
        if not txt:
            return "unknown"
        passthrough_txt = txt
        parts = [p.strip() for p in re.split(r"[,;|/]+", txt) if p.strip()]
        for part in parts:
            for token in part.split():
                if token.isdigit():
                    cams.append(int(token))
    elif isinstance(camera_ids, Iterable):
        raw_items: list[str] = []
        for item in camera_ids:
            if item is None:
                continue
            item_txt = str(item).strip()
            if not item_txt:
                continue
            raw_items.append(item_txt)
            if item_txt.isdigit():
                cams.append(int(item_txt))
        passthrough_txt = ",".join(raw_items)
    else:
        txt = str(camera_ids).strip()
        if txt.isdigit():
            cams.append(int(txt))
        passthrough_txt = txt

    cams = sorted(set(cams))
    if not cams:
        return passthrough_txt if passthrough_txt else "unknown"

    room_labels: list[str] = []
    unknown_cam_labels: list[str] = []
    for cam in cams:
        room_label = CAMERA_TO_ROOM_LABEL.get(int(cam))
        if room_label:
            if room_label not in room_labels:
                room_labels.append(room_label)
        else:
            unknown_cam_labels.append(f"cam {int(cam):03d}")

    labels = room_labels + unknown_cam_labels
    return ", ".join(labels) if labels else "unknown"


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
            # Calculate percentage based on observed time only (exclude no_observation)
            observed_min = float(
                budget[budget["label"] != "no_observation"]["duration_min"].sum()
            )
            if observed_min > 0:
                budget["percentage"] = budget["duration_min"] / observed_min * 100.0
            else:
                budget["percentage"] = 0.0

    if camera_ids is None:
        camera_txt = "unknown"
    elif isinstance(camera_ids, str):
        camera_txt = camera_ids if camera_ids.strip() else "unknown"
    else:
        camera_txt = ",".join(str(int(c)) for c in sorted(set(int(c) for c in camera_ids)))
        if not camera_txt:
            camera_txt = "unknown"
    camera_txt = camera_ids_to_room_label_text(camera_txt)

    other_group_txt = str(other_group).strip() if other_group is not None else ""
    if not other_group_txt:
        other_group_txt = "unknown"
    total_hours = budget["duration_min"].sum() / 60.0
    title = (
        f"{title_prefix} {date}, \n room {camera_txt}\n"
        f"total {total_hours:.1f} hours\n"
        f"other group {other_group_txt}\n"
    )
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
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
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return budget


def plot_hourly_activity_budget_radial(
    ethograms_by_date: dict[str, pd.DataFrame],
    out_path: Path,
    title: str,
    label_order: list[str] | None = None,
    label_display_map: dict[str, str] | None = None,
    label_color_map: dict[str, str] | None = None,
    start_hour: int = 18,
    end_hour: int = 6,
) -> pd.DataFrame:
    """Plot average activity budget per hour (across dates) as a radial stacked bar chart.

    Returns a long-form dataframe with mean minutes per behavior per hour.
    """
    if not ethograms_by_date:
        raise ValueError("No ethogram data provided for radial activity budget plot")
    if start_hour == end_hour:
        raise ValueError("start_hour and end_hour must define a non-empty interval")

    labels = list(label_order) if label_order is not None else list(LABEL_ORDER)
    if "no_observation" in labels:
        labels = [l for l in labels if l != "no_observation"]
    if not labels:
        labels = [l for l in LABEL_ORDER if l != "no_observation"]

    def _build_hour_edges(base_date_txt: str) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
        base_date = pd.to_datetime(base_date_txt, format="%Y%m%d", errors="coerce")
        if pd.isna(base_date):
            raise ValueError(f"Invalid date string: {base_date_txt}")
        night_start = pd.Timestamp(base_date) + pd.Timedelta(hours=int(start_hour))
        if end_hour <= start_hour:
            n_hours = (24 - int(start_hour)) + int(end_hour)
            night_end = pd.Timestamp(base_date) + pd.Timedelta(days=1, hours=int(end_hour))
        else:
            n_hours = int(end_hour - start_hour)
            night_end = pd.Timestamp(base_date) + pd.Timedelta(hours=int(end_hour))
        edges = pd.date_range(start=night_start, end=night_end, freq="1h")
        if len(edges) != n_hours + 1:
            edges = pd.DatetimeIndex([night_start + pd.Timedelta(hours=h) for h in range(n_hours + 1)])
        return night_start, edges

    rows: list[dict] = []
    for date_txt, eth in ethograms_by_date.items():
        if eth is None or eth.empty:
            continue
        if not {"start_dt", "end_dt", "label"}.issubset(eth.columns):
            continue
        e = eth.copy()
        e["start_dt"] = pd.to_datetime(e["start_dt"], errors="coerce")
        e["end_dt"] = pd.to_datetime(e["end_dt"], errors="coerce")
        e = e.dropna(subset=["start_dt", "end_dt", "label"])
        if e.empty:
            continue

        night_start, hour_edges = _build_hour_edges(date_txt)
        n_hours = len(hour_edges) - 1
        for h_idx in range(n_hours):
            h_start = pd.Timestamp(hour_edges[h_idx])
            h_end = pd.Timestamp(hour_edges[h_idx + 1])
            for beh in labels:
                seg = e[e["label"].astype(str) == str(beh)]
                if seg.empty:
                    overlap_sec = 0.0
                else:
                    s = seg["start_dt"].clip(lower=h_start, upper=h_end)
                    t = seg["end_dt"].clip(lower=h_start, upper=h_end)
                    overlap = (t - s).dt.total_seconds()
                    overlap_sec = float(overlap[overlap > 0].sum())
                rows.append(
                    {
                        "date": str(date_txt),
                        "hour_idx": int(h_idx),
                        "hour_label": f"{h_start:%H}:00-{h_end:%H}:00",
                        "behavior_label": str(beh),
                        "minutes": float(overlap_sec / 60.0),
                    }
                )

    if not rows:
        raise ValueError("No valid hourly overlaps found in provided ethograms")

    df_hourly = pd.DataFrame(rows)
    agg = (
        df_hourly.groupby(["hour_idx", "hour_label", "behavior_label"], as_index=False)["minutes"]
        .mean()
        .sort_values(["hour_idx", "behavior_label"])
        .reset_index(drop=True)
    )

    hour_labels = agg[["hour_idx", "hour_label"]].drop_duplicates().sort_values("hour_idx")
    hours = hour_labels["hour_idx"].to_numpy(dtype=int)
    n_hours = len(hours)
    n_behaviors = len(labels)
    
    # Extract actual hour from hour_label (e.g., "18:00-19:00" -> 18)
    # Map to 12-hour clock positions: 18->6, 19->7, ..., 23->11, 0->12, 1->1, ..., 5->5
    hour_to_clock = {}
    hour_to_label = {}
    for _, row in hour_labels.iterrows():
        h_idx = int(row["hour_idx"])
        h_label = str(row["hour_label"])
        actual_hour = int(h_label.split(":")[0])  # Extract hour from "HH:00-HH:00"
        # Map to 12-hour clock: 0/24->0 (12 o'clock), 1->1, ..., 11->11, 12->0, 13->1, etc.
        clock_pos = actual_hour % 12
        hour_to_clock[h_idx] = clock_pos
        hour_to_label[h_idx] = h_label.split("-")[0]
    
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(12, 10), dpi=300)
    ax.set_theta_zero_location("N")  # theta=0 at North (12 o'clock)
    ax.set_theta_direction(-1)  # Clockwise direction

    # Width of each behavior wedge at each clock position
    clock_width = 2.0 * np.pi / 12  # 12 positions on clock
    behavior_width = clock_width / n_behaviors * 0.95  # 0.95 for small gaps
    
    behavior_present = []
    behavior_colors = {}  # Track first color used for each behavior for legend
    max_val = 0.0
    
    for hour_idx, h in enumerate(hours):
        # Get clock position (0-11) for this hour
        clock_pos = hour_to_clock[h]
        # Center wedges at exact clock position: 0:00 at theta=0 (top), 1:00 at next position clockwise, etc.
        hour_theta_center = clock_pos * clock_width
        hour_theta_start = hour_theta_center - (n_behaviors * behavior_width) / 2
        
        for beh_idx, beh in enumerate(labels):
            # Get value for this (hour, behavior) combination
            m = agg[(agg["hour_idx"] == int(h)) & (agg["behavior_label"] == str(beh))]
            val = float(m["minutes"].iloc[0]) if not m.empty else 0.0
            
            if val > 0 and beh not in behavior_present:
                behavior_present.append(beh)
            
            max_val = max(max_val, val)
            
            color = (
                label_color_map.get(beh, "#9E9E9E")
                if label_color_map is not None
                else LABEL_COLORS.get(beh, "#9E9E9E")
            )
            
            # Store color for legend (keep first encounter)
            if beh not in behavior_colors:
                behavior_colors[beh] = color
            
            # Calculate theta position for this behavior wedge within the hour
            wedge_theta = hour_theta_start + beh_idx * behavior_width
            
            # Draw individual wedge for this (hour, behavior)
            ax.bar(
                wedge_theta, 
                val, 
                width=behavior_width, 
                bottom=0,  # Each wedge starts from center
                color=color, 
                edgecolor="white",  # White separation between wedges
                linewidth=1.0,
                align='edge'  # Align bars to the edge
            )
    
    # Set radial limits and ticks
    # Radius unit: "Mean minutes / hour" - max at 30 minutes
    max_radius = 30.0
    ax.set_ylim(0, max_radius)
    ax.set_yticks([5, 10, 15, 20, 25, 30])
    ax.set_yticklabels(['5', '10', '15', '20', '25', '30'], fontsize=14)

    # Hour labels on x-axis (angular axis) - one label per clock position
    # Group hours by clock position (some positions may have multiple hours from our data)
    clock_labels = {}
    for h_idx in hours:
        clock_pos = hour_to_clock[h_idx]
        if clock_pos not in clock_labels:
            clock_labels[clock_pos] = hour_to_label[h_idx]
    
    # Place labels at exact clock positions: 0:00 at top (theta=0), others clockwise
    label_positions = sorted(clock_labels.keys())
    hour_theta_centers = [pos * clock_width for pos in label_positions]
    tick_labels = [clock_labels[pos][0:2] for pos in label_positions]
    ax.set_xticks(hour_theta_centers)
    ax.set_xticklabels(tick_labels, fontsize=14, fontweight="bold")
    
    # Title and axis labels
    ax.set_title(title, pad=28, fontsize=16, fontweight="bold")
    ax.set_ylabel("", labelpad=30, fontsize=14) 

    # Legend: "Behavior" title, upper right with explicit color handles
    if behavior_present:
        from matplotlib.patches import Patch
        
        legend_handles = []
        legend_labels = []
        # Use the original labels order (from the function parameter) for consistency
        for b in labels:
            if b in behavior_present:
                # Get the correct color for this behavior
                color = behavior_colors.get(b, "#9E9E9E")
                # Get the display label
                label = (
                    label_display_map.get(b, b) if label_display_map is not None 
                    else LABEL_DISPLAY.get(b, b)
                )
                legend_handles.append(Patch(facecolor=color, edgecolor='white', linewidth=1))
                legend_labels.append(label)
        
        ax.legend(
            legend_handles,
            legend_labels, 
            loc="upper right", 
            bbox_to_anchor=(1.25, 1.15), 
            frameon=True,
            title="Behavior",
            title_fontsize=14,
            fontsize=10,
            fancybox=False,
            framealpha=0.95
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return agg


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
            fig_single, ax_single = plt.subplots(figsize=(6, 6), dpi=300, facecolor="black")
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


def categorize_time_period(timestamp: pd.Timestamp) -> str:
    """Categorize timestamp into time period.
    
    Args:
        timestamp: Timestamp to categorize
        
    Returns:
        'early_night' (18:00-00:00), 'mid_night' (00:00-05:00), or 'early_morning' (05:00-08:00)
    """
    hour = timestamp.hour
    if 18 <= hour < 24:
        return 'early_night'
    elif 0 <= hour < 5:
        return 'mid_night'
    elif 5 <= hour < 8:
        return 'early_morning'
    else:
        # Outside normal observation window
        return 'outside_window'


def analyze_stereotypy_from_ethogram(
    ethogram_csv: Path,
    individual_info_csv: Path | None = None,
) -> pd.DataFrame | None:
    """Analyze stereotypy patterns from ethogram data.
    
    Args:
        ethogram_csv: Path to ethogram CSV with columns: start_dt, end_dt, label
        individual_info_csv: Optional path to individual_info.csv with companions and camera info
        
    Returns:
        DataFrame with stereotypy analysis per time period, or None if no valid data
    """
    if not ethogram_csv.exists():
        return None
    
    # Load ethogram
    ethogram = pd.read_csv(ethogram_csv)
    if ethogram.empty or not {'start_dt', 'end_dt', 'label'}.issubset(ethogram.columns):
        return None
    
    ethogram['start_dt'] = pd.to_datetime(ethogram['start_dt'], errors='coerce')
    ethogram['end_dt'] = pd.to_datetime(ethogram['end_dt'], errors='coerce')
    ethogram = ethogram.dropna(subset=['start_dt', 'end_dt', 'label'])
    
    if ethogram.empty:
        return None
    
    # Load individual info if available
    companions = 'unknown'
    camera_ids = 'unknown'
    if individual_info_csv and individual_info_csv.exists():
        info = pd.read_csv(individual_info_csv)
        if not info.empty:
            companions = info.iloc[0].get('companions', 'unknown')
            camera_ids = info.iloc[0].get('camera_ids', 'unknown')
            
    # Normalize companions
    if pd.isna(companions) or str(companions).strip() == "":
        companions = 'unknown'
    else:
        companions = normalize_companion_group(companions)
    
    # Calculate duration in seconds
    ethogram['duration_s'] = (ethogram['end_dt'] - ethogram['start_dt']).dt.total_seconds()
    
    # Filter out no_observation periods for analysis
    observed = ethogram[ethogram['label'] != 'no_observation'].copy()
    
    if observed.empty:
        return None
    
    # Categorize by time period
    observed['time_period'] = observed['start_dt'].apply(categorize_time_period)
    
    # Filter out entries outside observation window
    observed = observed[observed['time_period'] != 'outside_window']
    
    if observed.empty:
        return None
    
    # Analyze by time period
    results = []
    for period in ['early_night', 'mid_night', 'early_morning']:
        period_data = observed[observed['time_period'] == period]
        
        if period_data.empty:
            results.append({
                'time_period': period,
                'total_observed_duration_s': 0.0,
                'stereotypy_duration_s': 0.0,
                'stereotypy_percentage': 0.0,
                'n_stereotypy_bouts': 0,
                'companions': companions,
                'camera_ids': camera_ids,
            })
            continue
        
        total_duration = period_data['duration_s'].sum()
        stereotypy_data = period_data[period_data['label'] == 'stereotypy']
        stereotypy_duration = stereotypy_data['duration_s'].sum()
        n_stereotypy_bouts = len(stereotypy_data)
        
        stereotypy_pct = (stereotypy_duration / total_duration * 100) if total_duration > 0 else 0.0
        
        results.append({
            'time_period': period,
            'total_observed_duration_s': total_duration,
            'stereotypy_duration_s': stereotypy_duration,
            'stereotypy_percentage': stereotypy_pct,
            'n_stereotypy_bouts': n_stereotypy_bouts,
            'companions': companions,
            'camera_ids': camera_ids,
        })
    
    # Also add overall stats
    total_duration = observed['duration_s'].sum()
    stereotypy_data = observed[observed['label'] == 'stereotypy']
    stereotypy_duration = stereotypy_data['duration_s'].sum()
    n_stereotypy_bouts = len(stereotypy_data)
    stereotypy_pct = (stereotypy_duration / total_duration * 100) if total_duration > 0 else 0.0
    
    results.append({
        'time_period': 'overall',
        'total_observed_duration_s': total_duration,
        'stereotypy_duration_s': stereotypy_duration,
        'stereotypy_percentage': stereotypy_pct,
        'n_stereotypy_bouts': n_stereotypy_bouts,
        'companions': companions,
        'camera_ids': camera_ids,
    })
    
    return pd.DataFrame(results)


def analyze_sleeping_from_ethogram(
    ethogram_csv: Path,
    individual_info_csv: Path | None = None,
) -> pd.DataFrame | None:
    """Analyze sleeping patterns from ethogram data.
    
    Args:
        ethogram_csv: Path to ethogram CSV with columns: start_dt, end_dt, label
        individual_info_csv: Optional path to individual_info.csv with companions and camera info
        
    Returns:
        DataFrame with sleeping analysis per time period, or None if no valid data
    """
    if not ethogram_csv.exists():
        return None
    
    # Load ethogram
    ethogram = pd.read_csv(ethogram_csv)
    if ethogram.empty or not {'start_dt', 'end_dt', 'label'}.issubset(ethogram.columns):
        return None
    
    ethogram['start_dt'] = pd.to_datetime(ethogram['start_dt'], errors='coerce')
    ethogram['end_dt'] = pd.to_datetime(ethogram['end_dt'], errors='coerce')
    ethogram = ethogram.dropna(subset=['start_dt', 'end_dt', 'label'])
    
    if ethogram.empty:
        return None
    
    # Load individual info if available
    companions = 'unknown'
    camera_ids = 'unknown'
    if individual_info_csv and individual_info_csv.exists():
        info = pd.read_csv(individual_info_csv)
        if not info.empty:
            companions = info.iloc[0].get('companions', 'unknown')
            camera_ids = info.iloc[0].get('camera_ids', 'unknown')
            
    # Normalize companions
    if pd.isna(companions) or str(companions).strip() == "":
        companions = 'unknown'
    else:
        companions = normalize_companion_group(companions)
    
    # Calculate duration in seconds
    ethogram['duration_s'] = (ethogram['end_dt'] - ethogram['start_dt']).dt.total_seconds()
    
    # Filter out no_observation periods for analysis
    observed = ethogram[ethogram['label'] != 'no_observation'].copy()
    
    if observed.empty:
        return None
    
    # Categorize by time period
    observed['time_period'] = observed['start_dt'].apply(categorize_time_period)
    
    # Filter out entries outside observation window
    observed = observed[observed['time_period'] != 'outside_window']
    
    if observed.empty:
        return None
    
    # Check for intensive stereotypy (any single bout > 10 minutes)
    stereotypy_data = observed[observed['label'] == 'stereotypy']
    has_intensive_stereotypy = False
    if not stereotypy_data.empty:
        max_stereotypy_duration = stereotypy_data['duration_s'].max()
        has_intensive_stereotypy = max_stereotypy_duration > 600  # 10 minutes = 600 seconds
    
    intensive_stereotypy = 'yes' if has_intensive_stereotypy else 'no'
    
    # Define sleeping labels
    sleeping_labels = ['02_sleeping_left', '03_sleeping_right']
    
    # Analyze by time period
    results = []
    for period in ['early_night', 'mid_night', 'early_morning']:
        period_data = observed[observed['time_period'] == period]
        
        if period_data.empty:
            results.append({
                'time_period': period,
                'total_observed_duration_s': 0.0,
                'sleeping_duration_s': 0.0,
                'sleeping_percentage': 0.0,
                'n_sleeping_bouts': 0,
                'companions': companions,
                'camera_ids': camera_ids,
                'intensive_stereotypy': intensive_stereotypy,
            })
            continue
        
        total_duration = period_data['duration_s'].sum()
        sleeping_data = period_data[period_data['label'].isin(sleeping_labels)]
        sleeping_duration = sleeping_data['duration_s'].sum()
        n_sleeping_bouts = len(sleeping_data)
        
        sleeping_pct = (sleeping_duration / total_duration * 100) if total_duration > 0 else 0.0
        
        results.append({
            'time_period': period,
            'total_observed_duration_s': total_duration,
            'sleeping_duration_s': sleeping_duration,
            'sleeping_percentage': sleeping_pct,
            'n_sleeping_bouts': n_sleeping_bouts,
            'companions': companions,
            'camera_ids': camera_ids,
            'intensive_stereotypy': intensive_stereotypy,
        })
    
    # Also add overall stats
    total_duration = observed['duration_s'].sum()
    sleeping_data = observed[observed['label'].isin(sleeping_labels)]
    sleeping_duration = sleeping_data['duration_s'].sum()
    n_sleeping_bouts = len(sleeping_data)
    sleeping_pct = (sleeping_duration / total_duration * 100) if total_duration > 0 else 0.0
    
    results.append({
        'time_period': 'overall',
        'total_observed_duration_s': total_duration,
        'sleeping_duration_s': sleeping_duration,
        'sleeping_percentage': sleeping_pct,
        'n_sleeping_bouts': n_sleeping_bouts,
        'companions': companions,
        'camera_ids': camera_ids,
        'intensive_stereotypy': intensive_stereotypy,
    })
    
    return pd.DataFrame(results)


def aggregate_stereotypy_analysis_multi_dates(
    output_dir: Path,
    individual_name: str,
    dates: list[str],
) -> pd.DataFrame:
    """Aggregate stereotypy analysis across multiple dates.
    
    Args:
        output_dir: Root output directory containing date subdirectories
        individual_name: Name of individual (e.g., 'Thai')
        dates: List of date strings (YYYYMMDD format)
        
    Returns:
        DataFrame with aggregated analysis per date and time period
    """
    all_results = []
    
    for date in dates:
        date_dir = output_dir / date / individual_name
        ethogram_csv = date_dir / 'csvs' / 'ethogram.csv'
        individual_info_csv = date_dir / 'csvs' / 'individual_info.csv'
        
        if not ethogram_csv.exists():
            print(f"  Skipping {date}: ethogram not found")
            continue
        
        date_analysis = analyze_stereotypy_from_ethogram(
            ethogram_csv=ethogram_csv,
            individual_info_csv=individual_info_csv if individual_info_csv.exists() else None,
        )
        
        if date_analysis is not None:
            date_analysis.insert(0, 'date', date)
            all_results.append(date_analysis)
            print(f"  ✓ Processed {date}")
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.concat(all_results, ignore_index=True)


def plot_stereotypy_analysis(
    df_analysis: pd.DataFrame,
    out_dir: Path,
    individual_name: str,
) -> None:
    """Generate visualization plots for stereotypy analysis.
    
    Args:
        df_analysis: DataFrame with stereotypy analysis data
        out_dir: Output directory for plots
        individual_name: Name of individual
    """
    if df_analysis.empty:
        print("No data to plot")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Stereotypy percentage by time period (overall across dates)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Exclude 'overall' for time period comparison
    time_period_data = df_analysis[df_analysis['time_period'] != 'overall'].copy()
    
    if not time_period_data.empty:
        # Group by time period and calculate mean
        period_stats = time_period_data.groupby('time_period').agg({
            'stereotypy_percentage': ['mean', 'std'],
            'n_stereotypy_bouts': 'sum',
            'total_observed_duration_s': 'sum',
        }).reset_index()
        
        period_stats.columns = ['time_period', 'mean_pct', 'std_pct', 'total_bouts', 'total_duration']
        period_order = ['early_night', 'mid_night', 'early_morning']
        period_stats['time_period'] = pd.Categorical(
            period_stats['time_period'], 
            categories=period_order, 
            ordered=True
        )
        period_stats = period_stats.sort_values('time_period')
        
        colors=['#FF9830', '#5794F2', '#73BF69']
        ax.bar(
            period_stats['time_period'],
            period_stats['mean_pct'],
            yerr=period_stats['std_pct'],
            capsize=5,
            alpha=0.5,
            color=colors[:len(period_stats)]
        )
        
        # Overlay individual data points
        for i, period in enumerate(period_order):
            period_values = time_period_data[time_period_data['time_period'] == period]['stereotypy_percentage']
            if not period_values.empty:
                # Add small random jitter for visibility
                x_positions = np.random.normal(i, 0.04, size=len(period_values))
                color=colors[i % len(colors)]
                ax.scatter(x_positions, period_values, color=color, alpha=1, s=30, zorder=3, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Mean Stereotypy Percentage (%)', fontsize=12)
        ax.set_title(f'Route Tracing Behavior by Time Period - {individual_name}', fontsize=14)
        ax.set_xticklabels(['\n18:00-00:00', '\n00:00-05:00', '\n05:00-08:00'])
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(out_dir / f'{individual_name}_stereotypy_by_time_period.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_dir / f'{individual_name}_stereotypy_by_time_period.png'}")
    
    # 2. Stereotypy trends over dates
    fig, axes = plt.subplots(figsize=(8, 5), dpi=300)
    
    overall_data = df_analysis[df_analysis['time_period'] == 'overall'].copy()
    if not overall_data.empty:
        overall_data = overall_data.sort_values('date')
        
        # Stereotypy percentage over time -- no line connecting points, just markers to show variability across dates
        axes.scatter(
            range(len(overall_data)),
            overall_data['stereotypy_percentage'],
            marker='o',
            s=60,
            color='#F2495C',
            edgecolor='black',
            linewidth=0.5,
            zorder=3
        )
        axes.set_xlabel('Night (Date)', fontsize=11)
        axes.set_ylabel('Stereotypy Percentage (%)', fontsize=11)
        axes.set_title(f'Route Tracing Percentage Over Time - {individual_name}', fontsize=13)
        axes.set_ylim(0, 100)
        axes.grid(True, alpha=0.3)
        
        # Add date labels on x-axis
        # tick_positions = range(0, len(overall_data), max(1, len(overall_data) // 10))
        axes.set_xticks(range(len(overall_data)))
        axes.set_xticklabels([overall_data.iloc[i]['date'] for i in range(len(overall_data))], rotation=90, ha='right', fontsize=8)
        
        
        fig.tight_layout()
        fig.savefig(out_dir / f'{individual_name}_stereotypy_trends.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_dir / f'{individual_name}_stereotypy_trends.png'}")
    
    # 4. Stereotypy by camera location
    if 'camera_ids' in df_analysis.columns:
        overall_with_cameras = df_analysis[df_analysis['time_period'] == 'overall'].copy()
        if not overall_with_cameras.empty:
            camera_stats = overall_with_cameras.groupby('camera_ids').agg({
                'stereotypy_percentage': ['mean', 'std', 'count'],
                'n_stereotypy_bouts': 'sum',
            }).reset_index()
            camera_stats.columns = ['camera_ids', 'mean_pct', 'std_pct', 'n_dates', 'total_bouts']
            
            # Filter out camera groups with very few observations
            camera_stats = camera_stats[camera_stats['n_dates'] >= 2]
            
            if not camera_stats.empty:
                fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
                
                bar_colors = ['#FF9830', '#5794F2', '#73BF69', '#F2495C', '#B877D9', '#FFB357']
                ax.bar(
                    range(len(camera_stats)),
                    camera_stats['mean_pct'],
                    yerr=camera_stats['std_pct'],
                    capsize=5,
                    alpha=1,
                    color='gray',
                    edgecolor='black',
                    linewidth=1
                )
                
                # Overlay individual data points with different colors and markers for different companion groups
                if 'companions' in overall_with_cameras.columns:
                    # Get unique companion groups for marker and color assignment
                    unique_companions = sorted(overall_with_cameras['companions'].dropna().unique())
                    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
                    scatter_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E', '#16A085', '#D35400']
                    companion_to_marker = {comp: markers[idx % len(markers)] for idx, comp in enumerate(unique_companions)}
                    companion_to_color = {comp: scatter_colors[idx % len(scatter_colors)] for idx, comp in enumerate(unique_companions)}
                    
                    # Track which companions have been plotted for legend
                    plotted_companions = set()
                    
                    for i, camera_id in enumerate(camera_stats['camera_ids']):
                        camera_data = overall_with_cameras[overall_with_cameras['camera_ids'] == camera_id]
                        for companion in camera_data['companions'].dropna().unique():
                            companion_subset = camera_data[camera_data['companions'] == companion]
                            if not companion_subset.empty:
                                values = companion_subset['stereotypy_percentage']
                                x_positions = np.random.normal(i, 0.04, size=len(values))
                                marker = companion_to_marker.get(companion, 'o')
                                color = companion_to_color.get(companion, '#E74C3C')
                                # Only add label for first occurrence of each companion group
                                label = f'{companion}' if companion not in plotted_companions else ''
                                plotted_companions.add(companion)
                                ax.scatter(x_positions, values, color=color, 
                                         marker=marker, alpha=0.8, s=60, zorder=3, 
                                         edgecolor='black', linewidth=0.5, label=label)
                else:
                    # Fallback if no companion info
                    for i, camera_id in enumerate(camera_stats['camera_ids']):
                        camera_values = overall_with_cameras[overall_with_cameras['camera_ids'] == camera_id]['stereotypy_percentage']
                        if not camera_values.empty:
                            x_positions = np.random.normal(i, 0.04, size=len(camera_values))
                            ax.scatter(x_positions, camera_values, color='#E74C3C', 
                                     alpha=0.8, s=60, zorder=3, edgecolor='black', linewidth=0.5)
                
                ax.set_xlabel('Enclosure', fontsize=12)
                ax.set_ylabel('Mean Stereotypy Percentage (%)', fontsize=12)
                ax.set_title(f'Route Tracing Behavior by Enclosure Location - {individual_name}\n(Color + Marker = Social group)', fontsize=14)
                ax.set_xticks(range(len(camera_stats)))
                room_tick_labels = camera_stats["camera_ids"].map(camera_ids_to_room_label_text)
                ax.set_xticklabels(room_tick_labels.astype(str), rotation=0, ha='right')
                ax.set_ylim(0, 100)
                ax.grid(axis='y', alpha=0.3)
                
                # Add legend if we have companion markers, to right top inside the plot area
                if 'companions' in overall_with_cameras.columns and len(overall_with_cameras['companions'].dropna().unique()) > 1:
                    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, title='Social Groups', ncol=1)
                
                # Add count annotations
                for i, row in enumerate(camera_stats.itertuples()):
                    ax.text(
                        i,
                        row.mean_pct + (row.std_pct if not pd.isna(row.std_pct) else 0) + 1.0,
                        f'n={row.n_dates}',
                        ha='center',
                        fontsize=9
                    )
                
                fig.tight_layout()
                fig.savefig(out_dir / f'{individual_name}_stereotypy_by_camera.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {out_dir / f'{individual_name}_stereotypy_by_camera.png'}")


def statistical_analysis_stereotypy(
    df_analysis: pd.DataFrame,
    out_dir: Path,
    individual_name: str,
) -> dict:
    """Perform statistical analysis to identify factors influencing stereotypy behavior.
    
    Uses Kruskal-Wallis test (non-parametric) and effect size calculations to determine which factors
    (time_period, companions, room) have the strongest influence on stereotypy percentage.
    
    Args:
        df_analysis: DataFrame with stereotypy analysis data
        out_dir: Output directory for results
        individual_name: Name of individual
        
    Returns:
        Dictionary with statistical results
    """
    try:
        from scipy.stats import kruskal
    except ImportError:
        print("Warning: scipy not available, skipping statistical analysis")
        return {}
    
    # Filter to only overall data (not time_period breakdowns)
    overall_data = df_analysis[df_analysis['time_period'] == 'overall'].copy()
    
    if overall_data.empty or len(overall_data) < 3:
        print("Insufficient data for statistical analysis (need at least 3 observations)")
        return {}
    
    # Convert camera_ids to room labels
    if 'camera_ids' in overall_data.columns:
        overall_data['room'] = overall_data['camera_ids'].apply(camera_ids_to_room_label_text)
    
    results = {
        'n_observations': len(overall_data),
        'factors': {},
        'summary': [],
        'table_rows': []
    }
    
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append(f"STATISTICAL ANALYSIS - {individual_name}")
    report_lines.append(f"Route Tracing Behavior Across Different Factors")
    report_lines.append("="*80)
    report_lines.append(f"\nTotal observations: {len(overall_data)}")
    report_lines.append(f"Mean stereotypy percentage: {overall_data['stereotypy_percentage'].mean():.2f}% (SD: {overall_data['stereotypy_percentage'].std():.2f}%)")
    report_lines.append("")
    report_lines.append("METHOD: Kruskal-Wallis test (non-parametric)")
    report_lines.append("  - Appropriate for non-normal distributions or unequal variances")
    report_lines.append("  - Tests whether groups come from the same distribution")
    report_lines.append("")
    report_lines.append("-"*80)
    report_lines.append("FACTOR ANALYSIS")
    report_lines.append("-"*80)
    report_lines.append("")
    
    # Analyze each factor
    factors_to_test = []
    
    # 1. Time period analysis (using non-overall data)
    time_period_data = df_analysis[df_analysis['time_period'] != 'overall'].copy()
    if len(time_period_data['time_period'].unique()) >= 2:
        factors_to_test.append(('time_period', time_period_data, 'Time Period'))
    
    # 2. Companions analysis
    if 'companions' in overall_data.columns:
        companions_clean = overall_data[overall_data['companions'].notna()]
        if len(companions_clean['companions'].unique()) >= 2:
            factors_to_test.append(('companions', companions_clean, 'Social Group'))
    
    # 3. Enclosure analysis (converted from camera_ids)
    if 'room' in overall_data.columns:
        room_clean = overall_data[overall_data['room'].notna()]
        if len(room_clean['room'].unique()) >= 2:
            factors_to_test.append(('room', room_clean, 'Enclosure Location'))
    
    for factor_name, factor_data, factor_label in factors_to_test:
        report_lines.append(f"\n{factor_label.upper()}")
        report_lines.append("-" * 40)
        
        # Get groups
        groups = factor_data.groupby(factor_name)['stereotypy_percentage'].apply(list)
        group_names = list(groups.index)
        group_values = [np.array(g) for g in groups.values if len(g) > 0]
        
        if len(group_values) < 2:
            report_lines.append(f"  Skipped: Need at least 2 groups (found {len(group_values)})")
            continue
        
        # Report group sizes and means
        report_lines.append(f"  Groups: {len(group_names)}")
        for gname, gvals in zip(group_names, group_values):
            report_lines.append(f"    {gname}: n={len(gvals)}, mean={np.mean(gvals):.2f}%, SD={np.std(gvals):.2f}%")
        
        # Use Kruskal-Wallis test
        stat, p_value = kruskal(*group_values)
        test_name = "Kruskal-Wallis"
        results['factors'][factor_name] = {
            'test': 'kruskal_wallis',
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'groups': group_names
        }
        
        # Calculate effect size (eta-squared)
        total_mean = np.mean(np.concatenate(group_values))
        ss_between = sum(len(g) * (np.mean(g) - total_mean)**2 for g in group_values)
        ss_total = sum((v - total_mean)**2 for g in group_values for v in g)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        results['factors'][factor_name]['eta_squared'] = float(eta_squared)
        
        # Interpret effect size
        if eta_squared < 0.01:
            effect_interpretation = "negligible"
        elif eta_squared < 0.06:
            effect_interpretation = "small"
        elif eta_squared < 0.14:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        report_lines.append(f"\n  {test_name}: H={stat:.4f}, p={p_value:.4f}")
        report_lines.append(f"    → {'SIGNIFICANT' if p_value < 0.05 else 'Not significant'} (α=0.05)")
        report_lines.append(f"\n  Effect size (η²): {eta_squared:.4f} ({effect_interpretation})")
        
        results['summary'].append({
            'factor': factor_label,
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'eta_squared': float(eta_squared),
            'effect_size': effect_interpretation
        })
        
        # Add to table rows
        results['table_rows'].append({
            'Factor': factor_label,
            'H-statistic': f"{stat:.4f}",
            'p-value': f"{p_value:.4f}",
            'Significant': 'Yes' if p_value < 0.05 else 'No',
            'Effect Size (η²)': f"{eta_squared:.4f}",
            'Interpretation': effect_interpretation.capitalize()
        })
    
    # Generate summary table
    if results['table_rows']:
        report_lines.append("\n")
        report_lines.append("="*80)
        report_lines.append("SUMMARY TABLE")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Create formatted table
        table_df = pd.DataFrame(results['table_rows'])
        report_lines.append(table_df.to_string(index=False))
        report_lines.append("")
        report_lines.append("Note: Significance level α=0.05")
        
        # Save table as CSV
        table_csv = out_dir / f"{individual_name}_statistical_analysis_table.csv"
        table_df.to_csv(table_csv, index=False)
        print(f"  ✓ Saved table: {table_csv}")
    
    # Rank factors by effect size
    if results['summary']:
        report_lines.append("\n")
        report_lines.append("="*80)
        report_lines.append("RANKING OF FACTORS BY INFLUENCE")
        report_lines.append("="*80)
        report_lines.append("")
        
        ranked = sorted(results['summary'], key=lambda x: x['eta_squared'], reverse=True)
        for i, item in enumerate(ranked, 1):
            sig_marker = "***" if item['significant'] else ""
            report_lines.append(
                f"{i}. {item['factor']:20s} | η²={item['eta_squared']:.4f} ({item['effect_size']:10s}) | "
                f"p={item['p_value']:.4f} {sig_marker}"
            )
        
        report_lines.append("")
        report_lines.append("*** = statistically significant at α=0.05")
        report_lines.append("")
        report_lines.append("Effect Size Interpretation:")
        report_lines.append("  η² < 0.01: negligible effect")
        report_lines.append("  η² < 0.06: small effect")
        report_lines.append("  η² < 0.14: medium effect")
        report_lines.append("  η² ≥ 0.14: large effect")
        
        # Add narrative summary
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("SUMMARY")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Generate narrative summary
        sig_factors = [item for item in ranked if item['significant']]
        if sig_factors:
            report_lines.append(f"Analysis of {individual_name}'s route tracing behavior revealed ")
            report_lines.append(f"{len(sig_factors)} significant factor(s) influencing stereotypy levels:")
            report_lines.append("")
            for i, item in enumerate(sig_factors, 1):
                report_lines.append(
                    f"{i}. {item['factor']} showed a {item['effect_size']} effect (η²={item['eta_squared']:.4f}, p={item['p_value']:.4f}), "
                    f"indicating that {item['factor'].lower()} {'substantially influences' if item['eta_squared'] >= 0.14 else 'influences' if item['eta_squared'] >= 0.06 else 'modestly influences'} "
                    f"route tracing behavior."
                )
            
            # Identify the most influential factor
            most_influential = sig_factors[0]
            report_lines.append("")
            report_lines.append(f"The most influential factor is {most_influential['factor']} (η²={most_influential['eta_squared']:.4f}), ")
            report_lines.append(f"which explains approximately {most_influential['eta_squared']*100:.1f}% of the variance in stereotypy percentage.")
        else:
            report_lines.append(f"Analysis of {individual_name}'s route tracing behavior found no statistically ")
            report_lines.append("significant factors among those tested (time period, social group, room location). ")
            report_lines.append("This suggests that stereotypy levels may be relatively consistent across these conditions, ")
            report_lines.append("or that other unmeasured factors may be more influential.")
        
        # Add methodological note
        report_lines.append("")
        report_lines.append("Methodology: Kruskal-Wallis test (non-parametric alternative to one-way ANOVA) ")
        report_lines.append("was used due to potential non-normality or unequal variances in the data.")
    
    # Save report
    report_file = out_dir / f"{individual_name}_statistical_analysis.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n{'='*80}")
    print('\n'.join(report_lines))
    print(f"{'='*80}")
    print(f"\n✓ Statistical analysis saved to: {report_file}")
    
    return results


# Normalize companion groups (sort names alphabetically so "Panang, Fahra" == "Fahra, Panang")
def normalize_companion_group(companion_str):
    if pd.isna(companion_str) or str(companion_str).strip() == '':
        return 'unknown'
    # Split by comma, strip whitespace, sort alphabetically, rejoin
    names = [name.strip() for name in str(companion_str).split(',')]
    return ', '.join(sorted(names))


def statistical_analysis_sleeping(
    df_analysis: pd.DataFrame,
    out_dir: Path,
    individual_name: str,
) -> dict:
    """Perform statistical analysis to identify factors influencing sleeping duration.
    
    Uses Kruskal-Wallis test (non-parametric) and effect size calculations to determine which factors
    (time_period, companions, room, intensive_stereotypy) have the strongest influence on sleeping duration.
    
    Args:
        df_analysis: DataFrame with sleeping analysis data
        out_dir: Output directory for results
        individual_name: Name of individual
        
    Returns:
        Dictionary with statistical results
    """
    try:
        from scipy.stats import kruskal
    except ImportError:
        print("Warning: scipy not available, skipping statistical analysis")
        return {}
    
    # Filter to only overall data (not time_period breakdowns)
    overall_data = df_analysis[df_analysis['time_period'] == 'overall'].copy()
    
    if overall_data.empty or len(overall_data) < 3:
        print("Insufficient data for statistical analysis (need at least 3 observations)")
        return {}
    
    # Convert camera_ids to room labels
    if 'camera_ids' in overall_data.columns:
        overall_data['room'] = overall_data['camera_ids'].apply(camera_ids_to_room_label_text)
    
    results = {
        'n_observations': len(overall_data),
        'factors': {},
        'summary': [],
        'table_rows': []
    }
    
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append(f"STATISTICAL ANALYSIS - {individual_name}")
    report_lines.append(f"Sleeping Behavior Across Different Factors")
    report_lines.append("="*80)
    report_lines.append(f"\nTotal observations: {len(overall_data)}")
    report_lines.append(f"Mean sleeping duration: {overall_data['sleeping_duration_s'].mean() / 3600:.2f} hours (SD: {overall_data['sleeping_duration_s'].std() / 3600:.2f} hours)")
    report_lines.append(f"Mean sleeping percentage: {overall_data['sleeping_percentage'].mean():.2f}% (SD: {overall_data['sleeping_percentage'].std():.2f}%)")
    report_lines.append("")
    report_lines.append("METHOD: Kruskal-Wallis test (non-parametric)")
    report_lines.append("  - Appropriate for non-normal distributions or unequal variances")
    report_lines.append("  - Tests whether groups come from the same distribution")
    report_lines.append("")
    report_lines.append("-"*80)
    report_lines.append("FACTOR ANALYSIS")
    report_lines.append("-"*80)
    report_lines.append("")
    
    # Analyze each factor (use sleeping_duration_s as the dependent variable)
    factors_to_test = []
    
    # 1. Time period analysis (using non-overall data)
    time_period_data = df_analysis[df_analysis['time_period'] != 'overall'].copy()
    if len(time_period_data['time_period'].unique()) >= 2:
        factors_to_test.append(('time_period', time_period_data, 'Time Period'))
    
    # 2. Companions analysis
    if 'companions' in overall_data.columns:
        companions_clean = overall_data[overall_data['companions'].notna()]
        if len(companions_clean['companions'].unique()) >= 2:
            factors_to_test.append(('companions', companions_clean, 'Social Group'))
    
    # 3. Enclosure analysis (converted from camera_ids)
    if 'room' in overall_data.columns:
        room_clean = overall_data[overall_data['room'].notna()]
        if len(room_clean['room'].unique()) >= 2:
            factors_to_test.append(('room', room_clean, 'Enclosure Location'))
    
    # 4. Intensive stereotypy analysis (NEW)
    if 'intensive_stereotypy' in overall_data.columns:
        intensive_clean = overall_data[overall_data['intensive_stereotypy'].notna()]
        if len(intensive_clean['intensive_stereotypy'].unique()) >= 2:
            factors_to_test.append(('intensive_stereotypy', intensive_clean, 'Intensive Stereotypy (>10min)'))
    
    for factor_name, factor_data, factor_label in factors_to_test:
        report_lines.append(f"\n{factor_label.upper()}")
        report_lines.append("-" * 40)
        
        # Get groups (use sleeping_duration_s in hours for analysis)
        groups = factor_data.groupby(factor_name)['sleeping_duration_s'].apply(lambda x: [v / 3600 for v in x])
        group_names = list(groups.index)
        group_values = [np.array(g) for g in groups.values if len(g) > 0]
        
        if len(group_values) < 2:
            report_lines.append(f"  Skipped: Need at least 2 groups (found {len(group_values)})")
            continue
        
        # Report group sizes and means
        report_lines.append(f"  Groups: {len(group_names)}")
        for gname, gvals in zip(group_names, group_values):
            report_lines.append(f"    {gname}: n={len(gvals)}, mean={np.mean(gvals):.2f} hours, SD={np.std(gvals):.2f} hours")
        
        # Use Kruskal-Wallis test
        stat, p_value = kruskal(*group_values)
        test_name = "Kruskal-Wallis"
        results['factors'][factor_name] = {
            'test': 'kruskal_wallis',
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'groups': group_names
        }
        
        # Calculate effect size (eta-squared)
        total_mean = np.mean(np.concatenate(group_values))
        ss_between = sum(len(g) * (np.mean(g) - total_mean)**2 for g in group_values)
        ss_total = sum((v - total_mean)**2 for g in group_values for v in g)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        results['factors'][factor_name]['eta_squared'] = float(eta_squared)
        
        # Interpret effect size
        if eta_squared < 0.01:
            effect_interpretation = "negligible"
        elif eta_squared < 0.06:
            effect_interpretation = "small"
        elif eta_squared < 0.14:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        report_lines.append(f"\n  {test_name}: H={stat:.4f}, p={p_value:.4f}")
        report_lines.append(f"    → {'SIGNIFICANT' if p_value < 0.05 else 'Not significant'} (α=0.05)")
        report_lines.append(f"\n  Effect size (η²): {eta_squared:.4f} ({effect_interpretation})")
        
        results['summary'].append({
            'factor': factor_label,
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'eta_squared': float(eta_squared),
            'effect_size': effect_interpretation
        })
        
        # Add to table rows
        results['table_rows'].append({
            'Factor': factor_label,
            'H-statistic': f"{stat:.4f}",
            'p-value': f"{p_value:.4f}",
            'Significant': 'Yes' if p_value < 0.05 else 'No',
            'Effect Size (η²)': f"{eta_squared:.4f}",
            'Interpretation': effect_interpretation.capitalize()
        })
    
    # Generate summary table
    if results['table_rows']:
        report_lines.append("\n")
        report_lines.append("="*80)
        report_lines.append("SUMMARY TABLE")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Create formatted table
        table_df = pd.DataFrame(results['table_rows'])
        report_lines.append(table_df.to_string(index=False))
        report_lines.append("")
        report_lines.append("Note: Significance level α=0.05")
        
        # Save table as CSV
        table_csv = out_dir / f"{individual_name}_sleeping_statistical_analysis_table.csv"
        table_df.to_csv(table_csv, index=False)
        print(f"  ✓ Saved table: {table_csv}")
    
    # Rank factors by effect size
    if results['summary']:
        report_lines.append("\n")
        report_lines.append("="*80)
        report_lines.append("RANKING OF FACTORS BY INFLUENCE")
        report_lines.append("="*80)
        report_lines.append("")
        
        ranked = sorted(results['summary'], key=lambda x: x['eta_squared'], reverse=True)
        for i, item in enumerate(ranked, 1):
            sig_marker = "***" if item['significant'] else ""
            report_lines.append(
                f"{i}. {item['factor']:30s} | η²={item['eta_squared']:.4f} ({item['effect_size']:10s}) | "
                f"p={item['p_value']:.4f} {sig_marker}"
            )
        
        report_lines.append("")
        report_lines.append("*** = statistically significant at α=0.05")
        report_lines.append("")
        report_lines.append("Effect Size Interpretation:")
        report_lines.append("  η² < 0.01: negligible effect")
        report_lines.append("  η² < 0.06: small effect")
        report_lines.append("  η² < 0.14: medium effect")
        report_lines.append("  η² ≥ 0.14: large effect")
        
        # Add narrative summary
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("SUMMARY")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Generate narrative summary
        sig_factors = [item for item in ranked if item['significant']]
        if sig_factors:
            report_lines.append(f"Analysis of {individual_name}'s sleeping behavior revealed ")
            report_lines.append(f"{len(sig_factors)} significant factor(s) influencing sleep duration:")
            report_lines.append("")
            for i, item in enumerate(sig_factors, 1):
                report_lines.append(
                    f"{i}. {item['factor']} showed a {item['effect_size']} effect (η²={item['eta_squared']:.4f}, p={item['p_value']:.4f}), "
                    f"indicating that {item['factor'].lower()} {'substantially influences' if item['eta_squared'] >= 0.14 else 'influences' if item['eta_squared'] >= 0.06 else 'modestly influences'} "
                    f"sleeping duration."
                )
            
            # Identify the most influential factor
            most_influential = sig_factors[0]
            report_lines.append("")
            report_lines.append(f"The most influential factor is {most_influential['factor']} (η²={most_influential['eta_squared']:.4f}), ")
            report_lines.append(f"which explains approximately {most_influential['eta_squared']*100:.1f}% of the variance in sleeping duration.")
        else:
            report_lines.append(f"Analysis of {individual_name}'s sleeping behavior found no statistically ")
            report_lines.append("significant factors among those tested (time period, social group, room location, intensive stereotypy). ")
            report_lines.append("This suggests that sleeping duration may be relatively consistent across these conditions, ")
            report_lines.append("or that other unmeasured factors may be more influential.")
        
        # Add methodological note
        report_lines.append("")
        report_lines.append("Methodology: Kruskal-Wallis test (non-parametric alternative to one-way ANOVA) ")
        report_lines.append("was used due to potential non-normality or unequal variances in the data.")
    
    # Save report
    report_file = out_dir / f"{individual_name}_sleeping_statistical_analysis.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n{'='*80}")
    print('\n'.join(report_lines))
    print(f"{'='*80}")
    print(f"\n✓ Statistical analysis saved to: {report_file}")
    
    return results


def lme_analysis_sleeping(
    df_analysis: pd.DataFrame,
    out_dir: Path,
    individual_name: str,
) -> dict:
    """Perform regression analysis for sleeping duration (between-night factors).
    
    Uses ordinary least squares (OLS) regression to test between-night factors:
    - Social Group (companions)
    - Enclosure Location
    - Intensive Stereotypy (>10 min bouts)
    
    Generates both text report and markdown summary table.
    
    Args:
        df_analysis: DataFrame with sleeping analysis data
        out_dir: Output directory for results
        individual_name: Name of individual
        
    Returns:
        Dictionary with regression results and summary table
    """
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
        from statsmodels.formula.api import ols
        import statsmodels.api as sm
    except ImportError:
        print("Warning: statsmodels not available, skipping LME analysis")
        print("Install with: pip install statsmodels")
        return {}
    
    # Prepare data - use overall data (one per night) for between-night comparisons
    df_overall = df_analysis[df_analysis['time_period'] == 'overall'].copy()
    
    # Also prepare time-period data for within-night analysis
    df_no_overall = df_analysis[df_analysis['time_period'] != 'overall'].copy()
    
    if df_overall.empty or len(df_overall) < 3:
        print("Insufficient data for LME analysis (need at least 3 nights)")
        return {}
    
    # Convert sleeping duration to hours for interpretability
    df_overall['sleeping_hours'] = df_overall['sleeping_duration_s'] / 3600
    df_no_overall['sleeping_hours'] = df_no_overall['sleeping_duration_s'] / 3600
    
    # Convert camera_ids to room labels
    if 'camera_ids' in df_overall.columns:
        df_overall['room'] = df_overall['camera_ids'].apply(camera_ids_to_room_label_text)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append(f"LINEAR REGRESSION ANALYSIS - {individual_name}")
    report_lines.append(f"Sleeping Duration: Between-Night Factors")
    report_lines.append("="*80)
    report_lines.append(f"\nTotal nights: {len(df_overall)}")
    report_lines.append(f"Mean sleeping duration: {df_overall['sleeping_hours'].mean():.2f} hours (SD: {df_overall['sleeping_hours'].std():.2f} hours)")
    report_lines.append("")
    report_lines.append("APPROACH:")
    report_lines.append("  Using ordinary linear regression (OLS) to test between-night factors:")
    report_lines.append("  - Social Group (companions)")
    report_lines.append("  - Enclosure Location")
    report_lines.append("  - Intensive Stereotypy (>10 min bouts)")
    report_lines.append("")
    
    results = {
        'models': {},
        'best_model': None,
        'summary_table': [],
    }
    
    # Try different model configurations
    models_to_test = []
    
    # Model 1: Social Group only (OLS - between nights)
    if 'companions' in df_overall.columns and len(df_overall['companions'].unique()) >= 2:
        models_to_test.append({
            'name': 'Social Group',
            'formula': 'sleeping_hours ~ C(companions)',
            'data': df_overall,
            'model_type': 'ols',
            'factor': 'Social Group',
        })
    
    # Model 2: Enclosure only (OLS - between nights)
    if 'room' in df_overall.columns and len(df_overall['room'].unique()) >= 2:
        models_to_test.append({
            'name': 'Enclosure',
            'formula': 'sleeping_hours ~ C(room)',
            'data': df_overall,
            'model_type': 'ols',
            'factor': 'Enclosure Location',
        })
    
    # Model 3: Intensive Stereotypy only (OLS - between nights)
    if 'intensive_stereotypy' in df_overall.columns and len(df_overall['intensive_stereotypy'].unique()) >= 2:
        models_to_test.append({
            'name': 'Intensive Stereotypy',
            'formula': 'sleeping_hours ~ C(intensive_stereotypy)',
            'data': df_overall,
            'model_type': 'ols',
            'factor': 'Intensive Stereotypy',
        })
    
    # Model 4: Combined between-night factors (OLS)
    combined_factors = []
    if 'companions' in df_overall.columns and len(df_overall['companions'].unique()) >= 2:
        combined_factors.append('C(companions)')
    if 'room' in df_overall.columns and len(df_overall['room'].unique()) >= 2:
        combined_factors.append('C(room)')
    if 'intensive_stereotypy' in df_overall.columns and len(df_overall['intensive_stereotypy'].unique()) >= 2:
        combined_factors.append('C(intensive_stereotypy)')
    
    if len(combined_factors) >= 2 and len(df_overall) >= 10:
        models_to_test.append({
            'name': 'Combined Model',
            'formula': f"sleeping_hours ~ {' + '.join(combined_factors)}",
            'data': df_overall,
            'model_type': 'ols',
            'factor': 'Combined',
        })
    
    report_lines.append("-"*80)
    report_lines.append("MODEL COMPARISON")
    report_lines.append("-"*80)
    report_lines.append("")
    
    best_aic = float('inf')
    best_model_info = None
    
    for model_info in models_to_test:
        try:
            report_lines.append(f"\n{model_info['name'].upper()}")
            report_lines.append(f"Formula: {model_info['formula']}")
            report_lines.append("-" * 40)
            
            # Fit OLS model
            from statsmodels.formula.api import ols
            model = ols(model_info['formula'], data=model_info['data'])
            result = model.fit()
            
            # Store results
            results['models'][model_info['name']] = {
                'formula': model_info['formula'],
                'type': model_info['model_type'],
                'aic': float(result.aic),
                'bic': float(result.bic),
                'log_likelihood': float(result.llf),
                'rsquared': float(result.rsquared),
                'rsquared_adj': float(result.rsquared_adj),
                'f_pvalue': float(result.f_pvalue),
                'factor': model_info.get('factor', model_info['name']),
            }
            
            report_lines.append(f"  AIC: {result.aic:.2f}")
            report_lines.append(f"  BIC: {result.bic:.2f}")
            report_lines.append(f"  R-squared: {result.rsquared:.3f}")
            report_lines.append(f"  Adjusted R-squared: {result.rsquared_adj:.3f}")
            report_lines.append(f"  F-statistic p-value: {result.f_pvalue:.4f}")
            
            # Show parameters
            report_lines.append(f"\n  Coefficients:")
            for param, value in result.params.items():
                pval = result.pvalues[param]
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                stderr = result.bse[param]
                report_lines.append(f"    {param:45s}: β={value:7.3f}, SE={stderr:6.3f}, p={pval:.4f} {sig}")
            
            # Add to summary table (only single-factor models)
            if model_info['name'] != 'Combined Model':
                # Get the significant effect
                sig_params = []
                for param, value in result.params.items():
                    if param != 'Intercept' and result.pvalues[param] < 0.05:
                        sig_params.append({
                            'param': param,
                            'beta': value,
                            'pvalue': result.pvalues[param]
                        })
                
                results['summary_table'].append({
                    'Factor': model_info['factor'],
                    'R²': result.rsquared,
                    'Adj. R²': result.rsquared_adj,
                    'F p-value': result.f_pvalue,
                    'AIC': result.aic,
                    'Significant': 'Yes' if result.f_pvalue < 0.05 else 'No',
                    'Effect Size': 'Large' if result.rsquared >= 0.14 else 'Medium' if result.rsquared >= 0.06 else 'Small',
                })
            
        except Exception as e:
            report_lines.append(f"  ERROR: Could not fit model - {str(e)}")
            continue
    
    # Summary interpretation
    report_lines.append("\n")
    report_lines.append("="*80)
    report_lines.append("SUMMARY & INTERPRETATION")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Summarize significant effects from all models
    for model_name, model_data in results['models'].items():
        if model_name != 'Combined Model':
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  R²: {model_data['rsquared']:.3f} (explains {model_data['rsquared']*100:.1f}% of variance)")
            report_lines.append(f"  Significant: {'Yes (p < 0.05)' if model_data['f_pvalue'] < 0.05 else 'No'}")
    
    report_lines.append("")
    report_lines.append("-"*80)
    report_lines.append("KEY FINDINGS:")
    report_lines.append("-"*80)
    report_lines.append("")
    
    sig_factors = [item for item in results['summary_table'] if item['Significant'] == 'Yes']
    if sig_factors:
        sig_factors_sorted = sorted(sig_factors, key=lambda x: x['R²'], reverse=True)
        report_lines.append(f"Found {len(sig_factors)} significant factor(s) affecting sleeping duration:")
        for i, factor in enumerate(sig_factors_sorted, 1):
            report_lines.append(f"{i}. {factor['Factor']}: R²={factor['R²']:.3f} ({factor['Effect Size']} effect), p={factor['F p-value']:.4f}")
    else:
        report_lines.append("No significant factors found (all p >= 0.05)")
    
    # Create markdown table
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("SUMMARY TABLE (Markdown Format)")
    report_lines.append("="*80)
    report_lines.append("")
    
    md_lines = []
    md_lines.append(f"# Sleeping Duration Analysis - {individual_name}")
    md_lines.append("")
    md_lines.append(f"**Total Nights:** {len(df_overall)}")
    md_lines.append(f"**Mean Sleeping Duration:** {df_overall['sleeping_hours'].mean():.2f} hours (SD: {df_overall['sleeping_hours'].std():.2f} hours)")
    md_lines.append("")
    md_lines.append("## Factor Analysis Summary")
    md_lines.append("")
    
    if results['summary_table']:
        md_lines.append("| Factor | R² | Adj. R² | F p-value | AIC | Significant | Effect Size |")
        md_lines.append("|--------|-------|---------|-----------|-----|-------------|-------------|")
        
        for item in results['summary_table']:
            md_lines.append(
                f"| {item['Factor']} | {item['R²']:.3f} | {item['Adj. R²']:.3f} | "
                f"{item['F p-value']:.4f} | {item['AIC']:.2f} | "
                f"{'✓' if item['Significant'] == 'Yes' else '✗'} | {item['Effect Size']} |"
            )
        
        md_lines.append("")
        md_lines.append("**Legend:**")
        md_lines.append("- R²: Proportion of variance explained")
        md_lines.append("- Adj. R²: Adjusted R² (accounts for number of predictors)")
        md_lines.append("- F p-value: Overall model significance")
        md_lines.append("- AIC: Akaike Information Criterion (lower is better)")
        md_lines.append("- Effect Size: Small (R²<0.06), Medium (0.06≤R²<0.14), Large (R²≥0.14)")
        md_lines.append("")
        
        # Add interpretation
        md_lines.append("## Interpretation")
        md_lines.append("")
        sig_factors = [item for item in results['summary_table'] if item['Significant'] == 'Yes']
        if sig_factors:
            sig_factors_sorted = sorted(sig_factors, key=lambda x: x['R²'], reverse=True)
            md_lines.append(f"**{len(sig_factors)} significant factor(s)** were identified:")
            md_lines.append("")
            for i, factor in enumerate(sig_factors_sorted, 1):
                md_lines.append(f"{i}. **{factor['Factor']}** (R²={factor['R²']:.3f}, p={factor['F p-value']:.4f})")
                md_lines.append(f"   - Explains {factor['R²']*100:.1f}% of variance in sleeping duration")
                md_lines.append(f"   - Effect size: {factor['Effect Size']}")
                md_lines.append("")
            
            most_influential = sig_factors_sorted[0]
            md_lines.append(f"**Most influential factor:** {most_influential['Factor']} explains {most_influential['R²']*100:.1f}% of the variance in sleeping duration.")
        else:
            md_lines.append("No statistically significant factors were identified (all p ≥ 0.05).")
            md_lines.append("")
            md_lines.append("This suggests that sleeping duration is relatively consistent across the tested conditions,")
            md_lines.append("or that other unmeasured factors may be more influential.")
    else:
        md_lines.append("No factors available for analysis.")
    
    # Add markdown table to text report
    for line in md_lines:
        report_lines.append(line)
    
    # Save text report
    report_file = out_dir / f"{individual_name}_sleeping_regression_analysis.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Save markdown report
    md_file = out_dir / f"{individual_name}_sleeping_regression_summary.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n{'='*80}")
    print('\n'.join(report_lines))
    print(f"{'='*80}")
    print(f"\n✓ Regression analysis saved to: {report_file}")
    print(f"✓ Markdown summary saved to: {md_file}")
    
    return results


def plot_sleeping_analysis(
    df_analysis: pd.DataFrame,
    out_dir: Path,
    individual_name: str,
) -> None:
    """Generate visualization plots for sleeping behavior analysis.
    
    Args:
        df_analysis: DataFrame with sleeping analysis data
        out_dir: Output directory for plots
        individual_name: Name of individual
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating sleeping analysis plots...")
    
    # 1. Sleeping duration by time period (box plot with individual data points)
    time_period_data = df_analysis[df_analysis['time_period'] != 'overall'].copy()
    if not time_period_data.empty:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        
        periods = ['early_night', 'mid_night', 'early_morning']
        period_labels = ['Early Night\n18:00-00:00', 'Mid Night\n00:00-05:00', 'Early Morning\n05:00-08:00']
        colors = ['#5FB13E', '#FF9A2A', '#8a00ac']
        
        box_data = []
        for period in periods:
            period_values = time_period_data[time_period_data['time_period'] == period]['sleeping_duration_s'] / 3600  # Convert to hours
            box_data.append(period_values)
        
        # Create box plot
        bp = ax.boxplot(box_data, positions=range(len(periods)), widths=0.5, 
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='lightgray', alpha=0.7),
                        medianprops=dict(color='black', linewidth=2))
        
        # Overlay individual data points with jitter
        for i, period in enumerate(periods):
            period_values = time_period_data[time_period_data['time_period'] == period]['sleeping_duration_s'] / 3600
            if not period_values.empty:
                # Add small random jitter for visibility
                x_positions = np.random.normal(i, 0.04, size=len(period_values))
                color = colors[i % len(colors)]
                ax.scatter(x_positions, period_values, color=color, alpha=1, s=30, zorder=3, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Sleeping Duration (hours)', fontsize=12)
        ax.set_title(f'Sleeping Duration by Time Period - {individual_name}', fontsize=14)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(period_labels)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(out_dir / f'{individual_name}_sleeping_by_time_period.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_dir / f'{individual_name}_sleeping_by_time_period.png'}")
    
    # 2. Sleeping duration trends over dates
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    overall_data = df_analysis[df_analysis['time_period'] == 'overall'].copy()
    if not overall_data.empty:
        overall_data = overall_data.sort_values('date')
        
        # Sleeping duration over time -- no line connecting points, just markers to show variability across dates
        ax.scatter(
            range(len(overall_data)),
            overall_data['sleeping_duration_s'] / 3600,  # Convert to hours
            marker='o',
            s=60,
            color='#5FB13E',
            edgecolor='black',
            linewidth=0.5,
            zorder=3
        )
        ax.set_xlabel('Night (Date)', fontsize=11)
        ax.set_ylabel('Total Sleeping Duration (hours)', fontsize=11)
        ax.set_title(f'Sleeping Duration Over Time - {individual_name}', fontsize=13)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        
        # Add date labels on x-axis
        ax.set_xticks(range(len(overall_data)))
        ax.set_xticklabels([overall_data.iloc[i]['date'] for i in range(len(overall_data))], rotation=90, ha='right', fontsize=8)
        
        fig.tight_layout()
        fig.savefig(out_dir / f'{individual_name}_sleeping_trends.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_dir / f'{individual_name}_sleeping_trends.png'}")
    
    # 3. Sleeping duration by intensive stereotypy
    if 'intensive_stereotypy' in df_analysis.columns:
        overall_data = df_analysis[df_analysis['time_period'] == 'overall'].copy()
        if not overall_data.empty and len(overall_data['intensive_stereotypy'].unique()) >= 2:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
            
            intensive_stats = overall_data.groupby('intensive_stereotypy').agg({
                'sleeping_duration_s': ['mean', 'std', 'count'],
            }).reset_index()
            intensive_stats.columns = ['intensive_stereotypy', 'mean_duration', 'std_duration', 'n_dates']
            
            # Create bar plot
            bar_colors = ['#73BF69', '#FF2624']  # Green for no, red for yes
            color_map = {'no': '#73BF69', 'yes': '#FF2624'}
            colors = [color_map.get(val, '#73BF69') for val in intensive_stats['intensive_stereotypy']]
            
            ax.bar(
                range(len(intensive_stats)),
                intensive_stats['mean_duration'] / 3600,  # Convert to hours
                yerr=intensive_stats['std_duration'] / 3600,
                capsize=5,
                alpha=1,
                color=colors,
                edgecolor='black',
                linewidth=1
            )
            
            # Overlay individual data points
            for i, intensive_val in enumerate(intensive_stats['intensive_stereotypy']):
                values = overall_data[overall_data['intensive_stereotypy'] == intensive_val]['sleeping_duration_s'] / 3600
                if not values.empty:
                    x_positions = np.random.normal(i, 0.04, size=len(values))
                    ax.scatter(x_positions, values, color='white', 
                             alpha=0.8, s=50, zorder=3, edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Intensive Stereotypy (>10 min bouts)', fontsize=12)
            ax.set_ylabel('Mean Sleeping Duration (hours)', fontsize=12)
            ax.set_title(f'Sleeping Duration by Intensive Stereotypy - {individual_name}', fontsize=13)
            ax.set_xticks(range(len(intensive_stats)))
            ax.set_xticklabels(['No', 'Yes'])
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=0.3)
            
            # Add count annotations
            for i, row in enumerate(intensive_stats.itertuples()):
                y_pos = (row.mean_duration + (row.std_duration if not pd.isna(row.std_duration) else 0)) / 3600
                ax.text(i, y_pos + 0.1, f'n={row.n_dates}', ha='center', fontsize=9)
            
            fig.tight_layout()
            fig.savefig(out_dir / f'{individual_name}_sleeping_by_intensive_stereotypy.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_dir / f'{individual_name}_sleeping_by_intensive_stereotypy.png'}")
    
    # 4. Sleeping duration by social group (companions)
    if 'companions' in df_analysis.columns:
        overall_data = df_analysis[df_analysis['time_period'] == 'overall'].copy()
        if not overall_data.empty:
            companion_stats = overall_data.groupby('companions').agg({
                'sleeping_duration_s': ['mean', 'std', 'count'],
            }).reset_index()
            companion_stats.columns = ['companions', 'mean_duration', 'std_duration', 'n_dates']
            
            # Filter out companion groups with very few observations
            companion_stats = companion_stats[companion_stats['n_dates'] >= 2]
            
            if not companion_stats.empty:
                fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
                
                ax.bar(
                    range(len(companion_stats)),
                    companion_stats['mean_duration'] / 3600,  # Convert to hours
                    yerr=companion_stats['std_duration'] / 3600,
                    capsize=5,
                    alpha=1,
                    color='#5FB13E',
                    edgecolor='black',
                    linewidth=1
                )
                
                # Overlay individual data points
                for i, companion in enumerate(companion_stats['companions']):
                    values = overall_data[overall_data['companions'] == companion]['sleeping_duration_s'] / 3600
                    if not values.empty:
                        x_positions = np.random.normal(i, 0.04, size=len(values))
                        ax.scatter(x_positions, values, color='white', 
                                 alpha=0.8, s=50, zorder=3, edgecolor='black', linewidth=1)
                
                ax.set_xlabel('Social Group', fontsize=12)
                ax.set_ylabel('Mean Sleeping Duration (hours)', fontsize=12)
                ax.set_title(f'Sleeping Duration by Social Group - {individual_name}', fontsize=13)
                ax.set_xticks(range(len(companion_stats)))
                ax.set_xticklabels(companion_stats['companions'], rotation=45, ha='right')
                ax.set_ylim(bottom=0)
                ax.grid(axis='y', alpha=0.3)
                
                # Add count annotations
                for i, row in enumerate(companion_stats.itertuples()):
                    y_pos = (row.mean_duration + (row.std_duration if not pd.isna(row.std_duration) else 0)) / 3600
                    ax.text(i, y_pos + 0.1, f'n={row.n_dates}', ha='center', fontsize=9)
                
                fig.tight_layout()
                fig.savefig(out_dir / f'{individual_name}_sleeping_by_companions.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {out_dir / f'{individual_name}_sleeping_by_companions.png'}")

    
def plot_stereotypy_factor_timelines(
    df_analysis: pd.DataFrame,
    out_dir: Path,
    individual_name: str,
) -> None:
    """Plot stereotypy percentage over time with different markers for factor combinations.
    
    Creates 3 subplots (one per time period) showing percentage trends over dates,
    with different markers/colors representing different factor combinations (camera + companions).
    This allows visual isolation of different factor influences.
    
    Args:
        df_analysis: DataFrame with stereotypy analysis data
        out_dir: Output directory for plots
        individual_name: Name of individual
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out overall data, keep only time period breakdowns
    df_plot = df_analysis[df_analysis['time_period'].isin(['early_night', 'mid_night', 'early_morning'])].copy()
    
    if df_plot.empty:
        print("  Skipping factor timeline plot: No time period data available")
        return
    
    # Convert date to datetime for sorting
    df_plot['date_dt'] = pd.to_datetime(df_plot['date'], format='%Y%m%d', errors='coerce')
    df_plot = df_plot.dropna(subset=['date_dt'])
    df_plot = df_plot.sort_values('date_dt')
        
    # Get unique cameras and normalized companions for marker/color assignment
    unique_cameras = sorted(df_plot['camera_ids'].dropna().unique())
    unique_companions_normalized = sorted(df_plot['companions'].unique())
    
    # Define marker styles (one per camera)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    camera_to_marker = {cam: markers[i % len(markers)] for i, cam in enumerate(unique_cameras)}
    
    # Define colors (one per normalized companion group)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    companion_to_color = {comp: colors[i % len(colors)] for i, comp in enumerate(unique_companions_normalized)}
    
    # Create factor combination identifier and assign styles
    df_plot['camera_ids_str'] = df_plot['camera_ids'].fillna('unknown')
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    time_periods = ['early_night', 'mid_night', 'early_morning']
    time_labels = {
        'early_night': '18:00-00:00',
        'mid_night': '00:00-05:00',
        'early_morning': '05:00-08:00'
    }
    
    for ax_idx, time_period in enumerate(time_periods):
        ax = axes[ax_idx]
        df_period = df_plot[df_plot['time_period'] == time_period].copy()
        
        if df_period.empty:
            ax.text(0.5, 0.5, f'No data for {time_labels[time_period]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(time_labels[time_period], fontsize=12, fontweight='bold')
            continue
        
        # Create date index mapping for this time period (equal spacing)
        unique_dates = sorted(df_period['date'].unique())
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        df_period['x_pos'] = df_period['date'].map(date_to_idx)
        
        # Plot each camera + companion combination
        plotted_labels = set()  # Track labels to avoid duplicates in legend
        for camera in unique_cameras:
            for companion_norm in unique_companions_normalized:
                df_combo = df_period[
                    (df_period['camera_ids_str'] == camera) & 
                    (df_period['companions'] == companion_norm)
                ]
                if not df_combo.empty:
                    marker = camera_to_marker.get(camera, 'o')
                    color = companion_to_color.get(companion_norm, 'gray')
                    # Use normalized companion name in label for consistency
                    room_label = camera_ids_to_room_label_text(camera)
                    label = f"{room_label} + {companion_norm}"
                    
                    # Use x_pos (integer index) instead of date_dt
                    ax.plot(df_combo['x_pos'], df_combo['stereotypy_percentage'],
                           marker=marker, 
                           color=color,
                           linestyle='',
                           linewidth=1.5,
                           markersize=8,
                           label=label,
                           alpha=0.7)
                    plotted_labels.add(label)
        
        # Format axes
        ax.set_ylabel('Stereotypy %', fontsize=11, fontweight='bold')
        ax.set_title(time_labels[time_period], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)
        
        # Set x-axis with all date labels
        ax.set_xlabel('Night (date)', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(unique_dates)))
        ax.set_xticklabels(unique_dates, rotation=90, ha='right')
        ax.set_xlim(-0.5, len(unique_dates) - 0.5)
        
        # Add legend for each subplot (only if there are multiple combinations)
        if len(plotted_labels) > 1 and not df_period.empty:
            ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
    
    # Overall title with explanation
    title_text = f'{individual_name} - Route Tracing Timeline by Factor Combinations\n'
    title_text += '(Marker shape = Enclosure, Color = Social Group)'
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.995)
    
    fig.tight_layout()
    
    # Save figure
    output_file = out_dir / f'{individual_name}_stereotypy_factor_timeline.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_file}")
