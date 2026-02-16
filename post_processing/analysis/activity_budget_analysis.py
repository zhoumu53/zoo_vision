import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


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
    "outside",
]

LABEL_DISPLAY = {
    "01_standing": "standing",
    "02_sleeping_left": "sleeping left",
    "03_sleeping_right": "sleeping right",
    "outside": "outside / no observation",
    "walking": "walking",
    "stereotypy": "stereotypy",
}

LABEL_COLORS = {
    "02_sleeping_left": "#5FB13E",
    "03_sleeping_right": "#FF9A2A",
    "outside": "#F1F1F1",
    "stereotypy": "#FF2624",
    "01_standing": "#8a00ac",
    "walking": "#c372cb",
}

GT_LABEL_COLORS = {
    "02_sleeping_left": "#E3E902",   ## lighter green
    "03_sleeping_right": "#FFEC17",  ## lighter yellow
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

LABEL_PRIORITY = {
    "outside": 0,
    "01_standing": 2,
    "02_sleeping_left": 1,
    "03_sleeping_right": 1,
    "walking": 3,
    "stereotypy": 4,
}


def normalize_date(date_str: str) -> str:
    return date_str.replace("-", "")


def get_available_dates(output_dir: Path) -> list[str]:
    return sorted(
        p.name for p in output_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8
    )


def get_bout_csvs(
    output_dir: Path,
    dates: list[str],
    filename_keyword: str | None = None,
    strict: bool = True,
) -> list[tuple[str, Path]]:
    collected: list[tuple[str, Path]] = []
    for date in dates:
        date_dir = output_dir / date
        if not date_dir.exists():
            continue
        pattern = "*.csv" if not filename_keyword else f"*{filename_keyword}*.csv"
        for csv_path in sorted(date_dir.glob(pattern)):
            collected.append((date, csv_path))

    if not collected and strict:
        keyword_txt = f" containing '{filename_keyword}'" if filename_keyword else ""
        raise FileNotFoundError(
            f"No bout summary CSV files{keyword_txt} found in {output_dir} for dates: {dates}"
        )
    return collected


def _behavior_label_col(df: pd.DataFrame) -> str:
    if "behavior_label" in df.columns:
        return "behavior_label"
    if "behavior_label_raw" in df.columns:
        return "behavior_label_raw"
    if "behavior_label_stage1" in df.columns:
        return "behavior_label_stage1"
    if "behavior_label_old" in df.columns:
        return "behavior_label_old"
    raise ValueError("No behavior label column found in CSV")


def _night_window(date_str: str, max_ts: pd.Timestamp | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    date = pd.to_datetime(date_str, format="%Y%m%d")
    start = date + pd.Timedelta(hours=18)
    end_7 = date + pd.Timedelta(days=1, hours=7)
    end_8 = date + pd.Timedelta(days=1, hours=8)

    if max_ts is not None and max_ts >= end_8:
        return start, end_8
    return start, end_7


def _assign_bouts_to_timeline(
    bouts: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    total_seconds = int((end - start).total_seconds())
    if total_seconds <= 0:
        raise ValueError("Invalid night window; end must be after start")

    label_codes = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    code_to_label = {i: lbl for lbl, i in label_codes.items()}

    labels = np.full(total_seconds, label_codes["outside"], dtype=np.int16)
    priorities = np.full(total_seconds, LABEL_PRIORITY["outside"], dtype=np.int16)

    if bouts.empty:
        return pd.DataFrame({
            "behavior_label": ["outside"],
            "duration_sec": [total_seconds],
        })

    bouts = bouts.sort_values(by=["start_time", "end_time"]).reset_index(drop=True)

    for _, row in bouts.iterrows():
        label = row["behavior_label"]
        if label not in VALID_LABELS:
            continue
        start_ts = row["start_time"]
        end_ts = row["end_time"]
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue

        s = max(start, start_ts)
        e = min(end, end_ts)
        if s >= e:
            continue

        s_idx = int((s - start).total_seconds())
        e_idx = int((e - start).total_seconds())
        pr = LABEL_PRIORITY[label]

        mask = priorities[s_idx:e_idx] < pr
        if not np.any(mask):
            continue

        labels[s_idx:e_idx][mask] = label_codes[label]
        priorities[s_idx:e_idx][mask] = pr

    counts = np.bincount(labels, minlength=len(LABEL_ORDER))
    data = []
    for code, secs in enumerate(counts):
        label = code_to_label.get(code, "outside")
        data.append({
            "behavior_label": label,
            "duration_sec": int(secs),
        })
    return pd.DataFrame(data)


def _build_timeline_labels(
    bouts: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    total_seconds = int((end - start).total_seconds())
    if total_seconds <= 0:
        raise ValueError("Invalid night window; end must be after start")

    label_codes = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    labels = np.full(total_seconds, label_codes["outside"], dtype=np.int16)
    priorities = np.full(total_seconds, LABEL_PRIORITY["outside"], dtype=np.int16)

    if bouts.empty:
        return labels

    bouts = bouts.sort_values(by=["start_time", "end_time"]).reset_index(drop=True)

    for _, row in bouts.iterrows():
        label = row["behavior_label"]
        if label not in VALID_LABELS:
            continue
        start_ts = row["start_time"]
        end_ts = row["end_time"]
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue

        s = max(start, start_ts)
        e = min(end, end_ts)
        if s >= e:
            continue

        s_idx = int((s - start).total_seconds())
        e_idx = int((e - start).total_seconds())
        pr = LABEL_PRIORITY[label]

        mask = priorities[s_idx:e_idx] < pr
        if not np.any(mask):
            continue

        labels[s_idx:e_idx][mask] = label_codes[label]
        priorities[s_idx:e_idx][mask] = pr

    return labels


def _summarize_labels(labels: np.ndarray) -> pd.DataFrame:
    code_to_label = {i: lbl for i, lbl in enumerate(LABEL_ORDER)}
    counts = np.bincount(labels, minlength=len(LABEL_ORDER))
    data = []
    for code, secs in enumerate(counts):
        label = code_to_label.get(code, "outside")
        data.append({
            "behavior_label": label,
            "duration_sec": int(secs),
        })
    return pd.DataFrame(data)


def _aggregate_timeline(
    labels: np.ndarray,
    start: pd.Timestamp,
    bin_minutes: int,
) -> pd.DataFrame:
    if bin_minutes <= 0:
        raise ValueError("bin_minutes must be positive")
    bin_seconds = bin_minutes * 60
    total_seconds = labels.shape[0]
    n_bins = int(np.ceil(total_seconds / bin_seconds))

    bin_idx = np.arange(total_seconds) // bin_seconds
    df = pd.DataFrame({
        "bin": bin_idx,
        "label_code": labels,
    })
    counts = pd.crosstab(df["bin"], df["label_code"])

    for code in range(len(LABEL_ORDER)):
        if code not in counts.columns:
            counts[code] = 0
    counts = counts[sorted(counts.columns)]

    bins = pd.date_range(start=start, periods=n_bins, freq=f"{bin_minutes}min")
    out = pd.DataFrame({
        "bin_start": bins,
        "bin_end": bins + pd.Timedelta(minutes=bin_minutes),
    })

    for code, lbl in enumerate(LABEL_ORDER):
        out[lbl] = counts[code].values.astype(int)

    return out


def _labels_to_segments(
    labels: np.ndarray,
    start: pd.Timestamp,
) -> pd.DataFrame:
    if labels.size == 0:
        return pd.DataFrame(columns=["start_time", "end_time", "behavior_label", "duration_sec"])

    change_idx = np.flatnonzero(np.diff(labels) != 0) + 1
    seg_starts = np.concatenate(([0], change_idx))
    seg_ends = np.concatenate((change_idx, [labels.size]))

    code_to_label = {i: lbl for i, lbl in enumerate(LABEL_ORDER)}
    rows = []
    for s_idx, e_idx in zip(seg_starts, seg_ends):
        rows.append({
            "start_time": start + pd.Timedelta(seconds=int(s_idx)),
            "end_time": start + pd.Timedelta(seconds=int(e_idx)),
            "behavior_label": code_to_label[int(labels[s_idx])],
            "duration_sec": int(e_idx - s_idx),
        })
    return pd.DataFrame(rows)


def _plot_activity_budget(df_budget: pd.DataFrame, out_path: Path, title: str) -> None:
    df_budget = df_budget.copy()
    df_budget["display_label"] = df_budget["behavior_label"].map(LABEL_DISPLAY)
    df_budget["duration_hr"] = df_budget["duration_sec"] / 3600.0

    order = [LABEL_DISPLAY[lbl] for lbl in LABEL_ORDER]
    df_budget["display_label"] = pd.Categorical(df_budget["display_label"], categories=order, ordered=True)
    df_budget = df_budget.sort_values("display_label")

    plt.figure(figsize=(8, 4))
    plt.bar(df_budget["display_label"], df_budget["duration_hr"], color="#2E6F8E")
    plt.ylabel("Hours")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_activity_timeline(
    df_segments: pd.DataFrame,
    out_path: Path,
    title: str,
    df_gt_segments: pd.DataFrame | None = None,
) -> None:
    df = df_segments.copy()
    if df.empty:
        raise ValueError("No timeline segments to plot")

    fig, ax = plt.subplots(figsize=(13, 2.8))

    for label in LABEL_ORDER:
        dfl = df[df["behavior_label"] == label]
        if dfl.empty:
            continue
        starts = mdates.date2num(dfl["start_time"])
        widths = mdates.date2num(dfl["end_time"]) - starts
        ax.bar(
            starts,
            np.ones(len(dfl)),
            width=widths,
            bottom=0.0,
            align="edge",
            label="_nolegend_",
            color=LABEL_COLORS[label],
            # edgecolor="white",
            linewidth=0.4,
        )

    gt_present_labels: list[str] = []
    if df_gt_segments is not None and not df_gt_segments.empty:
        gt_df = df_gt_segments.copy().sort_values(["start_time", "end_time"])
        # Overlay GT directly on top of the behavior bar.
        y_gt = 0.50
        for gt_label, dfg in gt_df.groupby("behavior_label"):
            color = GT_LABEL_COLORS.get(gt_label)
            if color is None:
                continue
            gt_present_labels.append(gt_label)
            starts = mdates.date2num(dfg["start_time"])
            ends = mdates.date2num(dfg["end_time"])
            ax.hlines(
                y=y_gt,
                xmin=starts,
                xmax=ends,
                colors=color,
                linewidth=3.0,
                zorder=5,
            )
            centers = (starts + ends) / 2.0
            ax.scatter(
                centers,
                np.full(len(centers), y_gt),
                marker="*",
                s=90,
                c=color,
                edgecolors="black",
                linewidths=0.4,
                zorder=6,
            )

    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(mdates.date2num(df["start_time"].min()), mdates.date2num(df["end_time"].max()))
    ax.set_ylim(0, 1.0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    legend_handles = [
        Patch(facecolor=LABEL_COLORS[label], edgecolor="none", label=LABEL_DISPLAY[label])
        for label in LABEL_ORDER
    ]
    legend_handles.extend([
        Line2D(
            [0],
            [0],
            color=GT_LABEL_COLORS[label],
            lw=3.0,
            marker="*",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=f"GT {LABEL_DISPLAY[label]}",
        )
        for label in sorted(set(gt_present_labels))
    ])
    ax.legend(
        handles=legend_handles,
        ncol=1,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        title="Behavior",
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_activity_timeline_multi_night(
    df_segments_all: pd.DataFrame,
    out_path: Path,
    title: str,
    df_gt_segments_all: pd.DataFrame | None = None,
) -> None:
    df = df_segments_all.copy()
    gt_df = pd.DataFrame() if df_gt_segments_all is None else df_gt_segments_all.copy()
    if df.empty and gt_df.empty:
        raise ValueError("No multi-night timeline/GT segments to plot")

    dates_union = set()
    if not df.empty:
        dates_union.update(df["date"].astype(str).unique())
    if not gt_df.empty:
        dates_union.update(gt_df["date"].astype(str).unique())
    dates_sorted = sorted(dates_union)
    date_to_y = {d: i for i, d in enumerate(dates_sorted)}
    y_labels = [pd.to_datetime(d, format="%Y%m%d").strftime("%Y-%m-%d") for d in dates_sorted]

    anchor = pd.Timestamp("2000-01-01 18:00:00")
    if not df.empty:
        start_nums = mdates.date2num(anchor + pd.to_timedelta(df["offset_start_sec"], unit="s"))
        width_days = (df["offset_end_sec"] - df["offset_start_sec"]).to_numpy(dtype=float) / 86400.0
    else:
        start_nums = np.array([], dtype=float)
        width_days = np.array([], dtype=float)

    # Full-page friendly layout for many nights.
    fig_h = max(11.0, 0.4 * len(dates_sorted) + 1.5)
    fig, ax = plt.subplots(figsize=(17, fig_h))

    if not df.empty:
        for label in LABEL_ORDER:
            dfl = df[df["behavior_label"] == label]
            if dfl.empty:
                continue
            idx = dfl.index.to_numpy()
            ax.barh(
                [date_to_y[d] for d in dfl["date"].astype(str)],
                width_days[idx],
                left=start_nums[idx],
                height=0.56,
                color=LABEL_COLORS[label],
                # edgecolor="white",
                linewidth=0.3,
                label="_nolegend_",
            )

    gt_present_labels: list[str] = []
    if not gt_df.empty:
        gt_start_nums = mdates.date2num(anchor + pd.to_timedelta(gt_df["offset_start_sec"], unit="s"))
        gt_end_nums = mdates.date2num(anchor + pd.to_timedelta(gt_df["offset_end_sec"], unit="s"))
        for gt_label, dfg in gt_df.groupby("behavior_label"):
            color = GT_LABEL_COLORS.get(gt_label)
            if color is None:
                continue
            gt_present_labels.append(gt_label)
            idx = dfg.index.to_numpy()
            ys = [date_to_y[d] for d in dfg["date"].astype(str)]
            ax.hlines(
                y=ys,
                xmin=gt_start_nums[idx],
                xmax=gt_end_nums[idx],
                colors=color,
                linewidth=2.6,
                zorder=5,
            )
            centers = (gt_start_nums[idx] + gt_end_nums[idx]) / 2.0
            ax.scatter(
                centers,
                ys,
                marker="*",
                s=70,
                c=color,
                edgecolors="black",
                linewidths=0.4,
                zorder=6,
            )

    x0 = mdates.date2num(anchor)
    max_end_offset = 0.0
    if not df.empty:
        max_end_offset = max(max_end_offset, float(df["offset_end_sec"].max()))
    if not gt_df.empty:
        max_end_offset = max(max_end_offset, float(gt_df["offset_end_sec"].max()))
    x1 = mdates.date2num(anchor + pd.Timedelta(seconds=max_end_offset))
    ax.set_xlim(x0, x1)
    ax.set_title(title)
    ax.set_ylabel("Night")
    ax.set_yticks(range(len(dates_sorted)))
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=LABEL_COLORS[label], edgecolor="none", label=LABEL_DISPLAY[label])
        for label in LABEL_ORDER
    ]
    legend_handles.extend([
        Line2D(
            [0],
            [0],
            color=GT_LABEL_COLORS[label],
            lw=3.0,
            marker="*",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=f"GT {LABEL_DISPLAY[label]}",
        )
        for label in sorted(set(gt_present_labels))
    ])
    ax.legend(
        handles=legend_handles,
        ncol=1,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        title="Behavior",
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _normalize_track_filename(track_filename: str) -> str:
    name = str(track_filename).strip()
    if name.endswith(".csv"):
        return name
    return f"{name}.csv"


def _resolve_track_csv_path(
    track_dir: Path,
    cam_id: str,
    bout_start: pd.Timestamp,
    track_filename: str,
) -> Path | None:
    date_dir = bout_start.strftime("%Y-%m-%d")
    fname = _normalize_track_filename(track_filename)
    cam_txt = str(cam_id).strip()

    preferred = [
        track_dir / f"zag_elp_cam_{cam_txt}" / date_dir / fname,
    ]
    if cam_txt.isdigit():
        preferred.append(track_dir / f"zag_elp_cam_{int(cam_txt):03d}" / date_dir / fname)

    for p in preferred:
        if p.exists():
            return p

    fallback = list(track_dir.glob(f"**/{date_dir}/{fname}"))
    if not fallback:
        return None
    if cam_txt:
        cam_hits = [p for p in fallback if f"cam_{cam_txt}" in p.as_posix()]
        if cam_hits:
            return cam_hits[0]
    return fallback[0]


def _read_track_points(
    track_csv: Path,
    cache: dict[Path, pd.DataFrame],
) -> pd.DataFrame:
    if track_csv in cache:
        return cache[track_csv]

    try:
        df = pd.read_csv(track_csv)
    except pd.errors.EmptyDataError:
        cache[track_csv] = pd.DataFrame(columns=["timestamp", "world_x", "world_y"])
        print(f"Warning: skipping empty track CSV: {track_csv}")
        return cache[track_csv]
    except pd.errors.ParserError as exc:
        cache[track_csv] = pd.DataFrame(columns=["timestamp", "world_x", "world_y"])
        print(f"Warning: skipping unparsable track CSV: {track_csv} ({exc})")
        return cache[track_csv]
    required = {"timestamp", "world_x", "world_y"}
    if not required.issubset(df.columns):
        cache[track_csv] = pd.DataFrame(columns=["timestamp", "world_x", "world_y"])
        return cache[track_csv]

    pts = df[["timestamp", "world_x", "world_y"]].copy()
    pts["timestamp"] = pd.to_datetime(pts["timestamp"], errors="coerce")
    pts = pts.dropna(subset=["timestamp", "world_x", "world_y"]).sort_values("timestamp")
    cache[track_csv] = pts
    return pts


def _split_standing_into_standing_walking(
    bouts: pd.DataFrame,
    track_dir: Path | None,
    standing_merge_gap_sec: float,
    walking_bin_minutes: float,
    walking_bin_distance_threshold: float,
    movement_step_clip: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bouts.empty:
        return bouts, pd.DataFrame()

    non_standing = bouts[bouts["behavior_label"] != "01_standing"].copy()
    standing = bouts[bouts["behavior_label"] == "01_standing"].copy()
    if standing.empty:
        out0 = bouts.sort_values(["start_time", "end_time"]).reset_index(drop=True)
        diag0 = pd.DataFrame(
            columns=[
                "start_time",
                "end_time",
                "duration_sec",
                "window_start_time",
                "window_end_time",
                "n_source_bouts",
                "n_track_points",
                "bin_distance",
                "walking_bin_minutes",
                "walking_bin_distance_threshold",
                "movement_step_clip",
                "predicted_label",
            ]
        )
        return out0, diag0

    standing = standing.sort_values(["start_time", "end_time"]).reset_index(drop=True)
    gap = pd.Timedelta(seconds=float(standing_merge_gap_sec))
    merged_windows: list[dict] = []

    curr_start = standing.loc[0, "start_time"]
    curr_end = standing.loc[0, "end_time"]
    curr_rows = [0]
    for i in range(1, len(standing)):
        s = standing.loc[i, "start_time"]
        e = standing.loc[i, "end_time"]
        if s <= curr_end + gap:
            if e > curr_end:
                curr_end = e
            curr_rows.append(i)
            continue
        merged_windows.append({"start_time": curr_start, "end_time": curr_end, "rows": curr_rows})
        curr_start, curr_end, curr_rows = s, e, [i]
    merged_windows.append({"start_time": curr_start, "end_time": curr_end, "rows": curr_rows})

    track_cache: dict[Path, pd.DataFrame] = {}
    new_rows: list[dict] = []
    diagnostics: list[dict] = []
    for window in merged_windows:
        w_start = window["start_time"]
        w_end = window["end_time"]
        if pd.isna(w_start) or pd.isna(w_end) or w_start >= w_end:
            continue

        parts = standing.loc[window["rows"]]
        collected_pts: list[pd.DataFrame] = []
        for _, row in parts.iterrows():
            track_filename = row.get("track_filename")
            cam_id = row.get("cam_id")
            if pd.isna(track_filename) or pd.isna(cam_id):
                continue
            row_track_dir = row.get("track_root")
            resolved_track_dir = Path(str(row_track_dir)) if pd.notna(row_track_dir) else track_dir
            if resolved_track_dir is None:
                continue
            track_csv = _resolve_track_csv_path(resolved_track_dir, str(cam_id), row["start_time"], str(track_filename))
            if track_csv is None:
                continue

            pts = _read_track_points(track_csv, track_cache)
            if pts.empty:
                continue
            wpts = pts[(pts["timestamp"] >= w_start) & (pts["timestamp"] <= w_end)]
            if not wpts.empty:
                collected_pts.append(wpts[["timestamp", "world_x", "world_y"]])

        duration_sec = float((w_end - w_start).total_seconds())
        all_pts = pd.DataFrame(columns=["timestamp", "world_x", "world_y"])
        if collected_pts:
            all_pts = pd.concat(collected_pts, ignore_index=True).sort_values("timestamp")
            all_pts = all_pts.groupby("timestamp", as_index=False)[["world_x", "world_y"]].median().set_index("timestamp")
            # Smooth to reduce frame-level jitter that inflates path length.
            all_pts = all_pts.resample("1s").median().interpolate(limit_direction="both").dropna().reset_index()

        bin_seconds = max(1, int(round(float(walking_bin_minutes) * 60.0)))
        n_bins = max(1, int(np.ceil(duration_sec / bin_seconds)))
        bin_rows: list[dict] = []
        for b in range(n_bins):
            b_start = w_start + pd.Timedelta(seconds=b * bin_seconds)
            b_end = min(w_end, w_start + pd.Timedelta(seconds=(b + 1) * bin_seconds))
            if b_start >= b_end:
                continue

            bpts = all_pts[(all_pts["timestamp"] >= b_start) & (all_pts["timestamp"] <= b_end)]
            bin_duration_sec = float((b_end - b_start).total_seconds())
            bin_distance = 0.0
            if len(bpts) >= 2:
                step = np.sqrt(bpts["world_x"].diff().pow(2) + bpts["world_y"].diff().pow(2))
                bin_distance = float(step.clip(upper=float(movement_step_clip)).iloc[1:].sum())

            # classify walking on full bins only, based on bin distance - threshold observed from raw video
            is_full_bin = abs(bin_duration_sec - float(bin_seconds)) <= 1.0
            # pred_label = "walking" if (is_full_bin and bin_distance > float(walking_bin_distance_threshold)) else "01_standing"
            ### for not full_bin (it still may walking but distance is underestimated due to short bin) -- count the speed from bin seconds, and bin threshold
            if not is_full_bin and bin_duration_sec > 0:
                speed = bin_distance / bin_duration_sec
                pred_label = "walking" if speed > (float(walking_bin_distance_threshold) / float(bin_seconds)) else "01_standing"
            else:
                pred_label = "walking" if bin_distance > float(walking_bin_distance_threshold) else "01_standing"
            
            
            bin_rows.append({
                "start_time": b_start,
                "end_time": b_end,
                "behavior_label": pred_label,
            })
            diagnostics.append({
                "start_time": b_start,
                "end_time": b_end,
                "duration_sec": bin_duration_sec,
                "window_start_time": w_start,
                "window_end_time": w_end,
                "n_source_bouts": len(parts),
                "n_track_points": int(len(bpts)),
                "bin_distance": bin_distance,
                "walking_bin_minutes": float(walking_bin_minutes),
                "walking_bin_distance_threshold": float(walking_bin_distance_threshold),
                "movement_step_clip": float(movement_step_clip),
                "predicted_label": pred_label,
            })

        if not bin_rows:
            continue

        cur = bin_rows[0].copy()
        for row in bin_rows[1:]:
            if row["behavior_label"] == cur["behavior_label"] and row["start_time"] <= cur["end_time"]:
                cur["end_time"] = row["end_time"]
            else:
                new_rows.append(cur)
                cur = row.copy()
        new_rows.append(cur)

    out = pd.concat([non_standing[["start_time", "end_time", "behavior_label"]], pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values(["start_time", "end_time"]).reset_index(drop=True)
    diag = pd.DataFrame(diagnostics)
    return out, diag


def _load_bouts_for_date(
    csv_sources: Iterable[tuple[Path, Path]],
    individual_label: str,
    standing_merge_gap_sec: float,
    walking_bin_minutes: float,
    walking_bin_distance_threshold: float,
    movement_step_clip: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for csv_path, track_dir in csv_sources:
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: skipping empty CSV with no header/data: {csv_path}")
            continue
        except pd.errors.ParserError as exc:
            print(f"Warning: skipping unparsable CSV: {csv_path} ({exc})")
            continue
        label_col = _behavior_label_col(df)

        if "start_time" not in df.columns or "end_time" not in df.columns:
            raise ValueError(f"Missing start_time/end_time in {csv_path}")
        if "identity_label" not in df.columns:
            raise ValueError(f"Missing identity_label in {csv_path}")
        if "cam_id" not in df.columns or "track_filename" not in df.columns:
            raise ValueError(f"Missing cam_id/track_filename in {csv_path}")

        df = df.copy()
        df["behavior_label"] = df[label_col].astype(str)
        df = df[df["behavior_label"].isin(VALID_LABELS)]
        df = df[df["identity_label"].astype(str) == individual_label]
        if df.empty:
            continue

        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
        df = df.dropna(subset=["start_time", "end_time"])
        if df.empty:
            continue

        df["track_root"] = str(track_dir)
        frames.append(df[["start_time", "end_time", "behavior_label", "cam_id", "track_filename", "track_root"]])

    if not frames:
        return (
            pd.DataFrame(columns=["start_time", "end_time", "behavior_label"]),
            pd.DataFrame(),
        )
    merged = pd.concat(frames, ignore_index=True)
    return _split_standing_into_standing_walking(
        bouts=merged,
        track_dir=None,
        standing_merge_gap_sec=standing_merge_gap_sec,
        walking_bin_minutes=walking_bin_minutes,
        walking_bin_distance_threshold=walking_bin_distance_threshold,
        movement_step_clip=movement_step_clip,
    )


def _resolve_record_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    if args.record_roots:
        roots.extend(Path(p) for p in args.record_roots)
    elif args.record_root:
        roots.append(Path(args.record_root))
    if not roots:
        raise ValueError("Provide --record_root or --record_roots")
    return roots


def _build_source_configs(
    record_roots: list[Path],
    output_dir_override: str | None,
) -> list[tuple[Path, Path]]:
    if output_dir_override and len(record_roots) > 1:
        raise ValueError("--output_dir override supports only one record root")

    configs: list[tuple[Path, Path]] = []
    for i, root in enumerate(record_roots):
        output_dir = Path(output_dir_override) if (output_dir_override and i == 0) else (root / "demo" / "night_bout_summary")
        track_dir = root / "tracks"
        configs.append((output_dir, track_dir))
    return configs


def _per_date_output_stem(
    csv_sources: Iterable[tuple[Path, Path]],
    fallback: str,
) -> str:
    stems = sorted({csv_path.stem for csv_path, _ in csv_sources})
    if len(stems) == 1:
        return stems[0]
    return fallback


def _remove_date_suffix(stem: str, date: str) -> str:
    suffixes = (f"_{date}", f"_{pd.to_datetime(date, format='%Y%m%d'):%Y-%m-%d}")
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _map_gt_label(gt_label: str) -> str | None:
    key = str(gt_label).strip().lower().replace(" ", "_")
    mapped = GT_LABEL_MAP.get(key)
    if mapped in VALID_LABELS:
        return mapped
    return None


def _parse_gt_clock_timestamp(date: str, value: str) -> pd.Timestamp | None:
    txt = str(value).strip()
    if not txt:
        return None

    full_ts = pd.to_datetime(txt, errors="coerce")
    if pd.notna(full_ts) and ("-" in txt or "T" in txt):
        return pd.Timestamp(full_ts)

    parsed_time = pd.to_datetime(txt, format="%H:%M:%S", errors="coerce")
    if pd.isna(parsed_time):
        parsed_time = pd.to_datetime(txt, format="%H:%M", errors="coerce")
    if pd.isna(parsed_time):
        return None

    base = pd.to_datetime(date, format="%Y%m%d")
    ts = pd.Timestamp.combine(base.date(), parsed_time.time())
    if ts.hour < 18:
        ts = ts + pd.Timedelta(days=1)
    return ts


def _ensure_and_load_gt_segments(
    gt_csv: Path,
    date: str,
    individual_label: str,
    night_start: pd.Timestamp,
    night_end: pd.Timestamp,
) -> tuple[pd.DataFrame, bool]:
    required_cols = ["id", "gt", "start_timestamp", "end_timestamp"]
    created = False
    if not gt_csv.exists():
        gt_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=required_cols).to_csv(gt_csv, index=False)
        created = True

    try:
        gt_df = pd.read_csv(gt_csv)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["start_time", "end_time", "behavior_label"]), created
    except pd.errors.ParserError as exc:
        print(f"Warning: skipping unparsable GT CSV: {gt_csv} ({exc})")
        return pd.DataFrame(columns=["start_time", "end_time", "behavior_label"]), created

    if gt_df.empty:
        return pd.DataFrame(columns=["start_time", "end_time", "behavior_label"]), created
    if not set(required_cols).issubset(gt_df.columns):
        print(f"Warning: GT CSV missing required columns {required_cols}: {gt_csv}")
        return pd.DataFrame(columns=["start_time", "end_time", "behavior_label"]), created

    gt_df = gt_df.copy()
    gt_df["id"] = gt_df["id"].astype(str).str.strip()
    gt_df = gt_df[gt_df["id"].str.lower() == individual_label.lower()]
    if gt_df.empty:
        return pd.DataFrame(columns=["start_time", "end_time", "behavior_label"]), created

    rows: list[dict] = []
    for _, row in gt_df.iterrows():
        mapped = _map_gt_label(str(row["gt"]))
        if mapped is None:
            continue
        s = _parse_gt_clock_timestamp(date, str(row["start_timestamp"]))
        e = _parse_gt_clock_timestamp(date, str(row["end_timestamp"]))
        if s is None or e is None:
            continue
        s = max(s, night_start)
        e = min(e, night_end)
        if s >= e:
            continue
        rows.append({"start_time": s, "end_time": e, "behavior_label": mapped})

    if not rows:
        return pd.DataFrame(columns=["start_time", "end_time", "behavior_label"]), created
    out = pd.DataFrame(rows).sort_values(["start_time", "end_time"]).reset_index(drop=True)
    return out, created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute activity budget per night from bout summary CSVs")
    parser.add_argument(
        "--record_root",
        type=str,
        default="/media/ElephantsWD/elephants/test_dan/results",
        help="Root directory of one record.",
    )
    parser.add_argument(
        "--record_roots",
        nargs="+",
        default=None,
        help="Optional list of record roots. When provided, all roots are merged in one analysis.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional direct path to one night_bout_summary directory. Overrides --record_root for single-root runs.",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        help="Dates to process, e.g. 2026-02-04 2026-02-05. If omitted, process all available dates.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/media/mu/zoo_vision/post_processing/analysis/activity_budget",
        help="Output root directory for activity budget CSVs and plots",
    )
    parser.add_argument(
        "--individual_group",
        type=str,
        default="Thai",
        help="Optional keyword to filter bout summary CSV files. Default is 'Thai'.",
    )
    parser.add_argument(
        "--bin_minutes",
        type=int,
        default=10,
        help="Bin size in minutes for the ethogram-style stacked bar plot.",
    )
    parser.add_argument(
        "--standing_merge_gap_sec",
        type=float,
        default=1.0,
        help="Merge adjacent standing bouts when gap is <= this many seconds.",
    )
    parser.add_argument(
        "--walking_bin_minutes",
        type=float,
        default=2.0,
        help="Time bin size (minutes) for standing-vs-walking split.",
    )
    parser.add_argument(
        "--walking_bin_distance_threshold",
        type=float,
        default=15.0,
        help="Mark a bin as walking when movement distance in that bin exceeds this threshold (world units).",
    )
    parser.add_argument(
        "--movement_step_clip",
        type=float,
        default=2.0,
        help="Maximum per-second movement step (world units) used in movement integration; larger jumps are clipped.",
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        default="/media/mu/zoo_vision/data/GT_id_behavior/behavior_GTs",
        help="Root directory for GT CSV files. Files are read/written under <gt_root>/<individual_group>/.",
    )
    return parser.parse_args()


def run_analysis(args: argparse.Namespace) -> None:
    record_roots = _resolve_record_roots(args)
    source_configs = _build_source_configs(record_roots, args.output_dir)

    for output_dir, _ in source_configs:
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

    if args.dates:
        dates = [normalize_date(d) for d in args.dates]
    else:
        all_dates = set()
        for output_dir, _ in source_configs:
            all_dates.update(get_available_dates(output_dir))
        dates = sorted(all_dates)
    if not dates:
        raise FileNotFoundError("No date folders found in the selected input roots")

    filename_keyword = args.individual_group if args.individual_group else None
    out_root = Path(args.out_root)
    gt_dir = Path(args.gt_root) / args.individual_group

    by_date: dict[str, list[tuple[Path, Path]]] = {}
    for output_dir, track_dir in source_configs:
        bouts = get_bout_csvs(
            output_dir=output_dir,
            dates=dates,
            filename_keyword=filename_keyword,
            strict=False,
        )
        for date, csv_path in bouts:
            by_date.setdefault(date, []).append((csv_path, track_dir))

    if not by_date:
        raise FileNotFoundError("No bout summary CSV files found for selected roots/dates")

    all_night_segments: list[pd.DataFrame] = []
    all_night_gt_segments: list[pd.DataFrame] = []
    all_budgets: list[pd.DataFrame] = []

    for date, csv_sources in sorted(by_date.items()):
        output_stem = _per_date_output_stem(csv_sources, args.individual_group)
        gt_stem = _remove_date_suffix(output_stem, date)
        bouts, standing_diag = _load_bouts_for_date(
            csv_sources=csv_sources,
            individual_label=args.individual_group,
            standing_merge_gap_sec=args.standing_merge_gap_sec,
            walking_bin_minutes=args.walking_bin_minutes,
            walking_bin_distance_threshold=args.walking_bin_distance_threshold,
            movement_step_clip=args.movement_step_clip,
        )
        max_ts = None
        if not bouts.empty:
            max_ts = max(bouts["end_time"].max(), bouts["start_time"].max())

        night_start, night_end = _night_window(date, max_ts)
        gt_csv = gt_dir / f"{gt_stem}_{date}.csv"
        gt_segments, gt_created = _ensure_and_load_gt_segments(
            gt_csv=gt_csv,
            date=date,
            individual_label=args.individual_group,
            night_start=night_start,
            night_end=night_end,
        )
        if not gt_segments.empty:
            gt_segments_plot = gt_segments.copy()
            gt_segments_plot["date"] = date
            gt_segments_plot["offset_start_sec"] = (gt_segments_plot["start_time"] - night_start).dt.total_seconds()
            gt_segments_plot["offset_end_sec"] = (gt_segments_plot["end_time"] - night_start).dt.total_seconds()
            all_night_gt_segments.append(gt_segments_plot)
        labels = _build_timeline_labels(bouts, night_start, night_end)
        budget = _summarize_labels(labels)
        budget["duration_min"] = budget["duration_sec"] / 60.0
        budget["duration_hr"] = budget["duration_sec"] / 3600.0
        budget["percent"] = budget["duration_sec"] / budget["duration_sec"].sum() * 100.0
        budget["night_start"] = night_start
        budget["night_end"] = night_end
        budget["date"] = date
        all_budgets.append(budget)

        out_dir = out_root / date
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{output_stem}_activity_budget.csv"
        budget.to_csv(out_csv, index=False)

        timeline = _aggregate_timeline(labels, night_start, args.bin_minutes)
        timeline["date"] = date
        timeline_csv = out_dir / f"{output_stem}_activity_timeline_{args.bin_minutes}min.csv"
        timeline.to_csv(timeline_csv, index=False)
        standing_diag_csv = out_dir / f"{output_stem}_standing_walking_diagnostics.csv"
        standing_diag.to_csv(standing_diag_csv, index=False)
        segments = _labels_to_segments(labels, night_start)
        segments["date"] = date
        segments["offset_start_sec"] = (segments["start_time"] - night_start).dt.total_seconds()
        segments["offset_end_sec"] = (segments["end_time"] - night_start).dt.total_seconds()
        # Keep y-axis dates in the all-night plot only for nights with actual bouts.
        if not bouts.empty:
            all_night_segments.append(segments)

        plot_path = out_dir / f"{output_stem}_activity_budget.png"
        title = f"{output_stem} Activity Budget {date} ({night_start:%H:%M}–{night_end:%H:%M})"
        _plot_activity_budget(budget, plot_path, title)

        ethogram_path = out_dir / f"{output_stem}_activity_ethogram_{args.bin_minutes}min.png"
        ethogram_title = f"{output_stem} Activity Ethogram {date} ({night_start:%H:%M}–{night_end:%H:%M})"
        _plot_activity_timeline(segments, ethogram_path, ethogram_title, df_gt_segments=gt_segments)

        print(f"Saved: {out_csv}")
        print(f"Saved: {timeline_csv}")
        print(f"Saved: {standing_diag_csv}")
        print(f"Saved: {plot_path}")
        print(f"Saved: {ethogram_path}")
        if gt_created:
            print(f"Created empty GT template: {gt_csv}")

    if all_night_segments or all_night_gt_segments:
        all_segments = pd.concat(all_night_segments, ignore_index=True) if all_night_segments else pd.DataFrame()
        if not all_segments.empty:
            all_segments_csv = out_root / f"{args.individual_group}_activity_segments_all_nights.csv"
            all_segments.to_csv(all_segments_csv, index=False)
            print(f"Saved: {all_segments_csv}")

        all_ethogram_path = out_root / f"{args.individual_group}_activity_ethogram_all_nights.png"
        all_ethogram_title = f"{args.individual_group} Activity Ethogram Across Nights"
        all_gt_segments = pd.concat(all_night_gt_segments, ignore_index=True) if all_night_gt_segments else pd.DataFrame()
        _plot_activity_timeline_multi_night(
            all_segments,
            all_ethogram_path,
            all_ethogram_title,
            df_gt_segments_all=all_gt_segments,
        )
        print(f"Saved: {all_ethogram_path}")

    if all_budgets:
        merged_budget = pd.concat(all_budgets, ignore_index=True)
        merged_budget = merged_budget.groupby("behavior_label", as_index=False)["duration_sec"].sum()
        merged_budget["duration_min"] = merged_budget["duration_sec"] / 60.0
        merged_budget["duration_hr"] = merged_budget["duration_sec"] / 3600.0
        merged_budget["percent"] = merged_budget["duration_sec"] / merged_budget["duration_sec"].sum() * 100.0

        merged_budget_csv = out_root / f"{args.individual_group}_activity_budget_all_nights.csv"
        merged_budget_png = out_root / f"{args.individual_group}_activity_budget_all_nights.png"
        merged_budget.to_csv(merged_budget_csv, index=False)
        _plot_activity_budget(
            merged_budget,
            merged_budget_png,
            f"{args.individual_group} Activity Budget Across All Nights",
        )
        print(f"Saved: {merged_budget_csv}")
        print(f"Saved: {merged_budget_png}")


if __name__ == "__main__":
    run_analysis(parse_args())
