import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
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
    "outside": 0,
    "01_standing": 1,
    "02_sleeping_left": 2,
    "03_sleeping_right": 2,
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
            starts = np.asarray(mdates.date2num(dfg["start_time"]), dtype=float)
            ends = np.asarray(mdates.date2num(dfg["end_time"]), dtype=float)
            ys = np.full(starts.shape[0], y_gt, dtype=float)
            ax.hlines(
                y=ys,
                xmin=starts,
                xmax=ends,
                colors=color,
                linewidth=1.0,
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


def _read_behavior_quality_points(
    track_csv: Path,
    cache: dict[Path, pd.DataFrame],
) -> pd.DataFrame:
    if track_csv in cache:
        return cache[track_csv]

    beh_csv = track_csv.with_name(f"{track_csv.stem}_behavior.csv")
    if not beh_csv.exists():
        cache[track_csv] = pd.DataFrame(columns=["timestamp", "is_bad_quality"])
        return cache[track_csv]

    try:
        df = pd.read_csv(beh_csv)
    except pd.errors.EmptyDataError:
        cache[track_csv] = pd.DataFrame(columns=["timestamp", "is_bad_quality"])
        return cache[track_csv]
    except pd.errors.ParserError as exc:
        print(f"Warning: skipping unparsable behavior CSV: {beh_csv} ({exc})")
        cache[track_csv] = pd.DataFrame(columns=["timestamp", "is_bad_quality"])
        return cache[track_csv]

    if "timestamp" not in df.columns:
        cache[track_csv] = pd.DataFrame(columns=["timestamp", "is_bad_quality"])
        return cache[track_csv]

    if {"quality_label", "behavior_conf"}.issubset(df.columns):
        quality_norm = df["quality_label"].astype(str).str.strip().str.lower()
        n_total = len(df)
        n_bad = int(quality_norm.eq("bad").sum())
        bad_ratio = (float(n_bad) / float(n_total)) if n_total > 0 else 0.0

        conf = pd.to_numeric(df["behavior_conf"], errors="coerce")
        avg_conf = float(conf.mean()) if conf.notna().any() else float("nan")
        conf_too_low = (pd.notna(avg_conf) and avg_conf < 0.7) or pd.isna(avg_conf)
        if bad_ratio >= 0.8 or conf_too_low:
            print(
                f"Skipping whole track due to poor quality: {beh_csv} "
                f"(bad_ratio={bad_ratio:.3f}, avg_behavior_conf={avg_conf})"
            )
            cache[track_csv] = pd.DataFrame(columns=["timestamp", "is_bad_quality"])
            return cache[track_csv]

    ts_raw = df["timestamp"].astype(str).str.strip()
    ts = pd.to_datetime(ts_raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    missing = ts.isna()
    if missing.any():
        ts_sub = pd.to_datetime(ts_raw[missing], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        ts.loc[missing] = ts_sub
    out = pd.DataFrame({"timestamp": ts})
    if "quality_label" in df.columns:
        out["is_bad_quality"] = df["quality_label"].astype(str).str.strip().str.lower().eq("bad")
    else:
        out["is_bad_quality"] = False
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    cache[track_csv] = out
    return out


def _filter_track_points_by_behavior_quality(
    pts: pd.DataFrame,
    track_csv: Path,
    quality_cache: dict[Path, pd.DataFrame],
) -> pd.DataFrame:
    if pts.empty:
        return pts
    quality = _read_behavior_quality_points(track_csv, quality_cache)
    if quality.empty:
        # Existing behavior CSV + empty quality output means whole-track rejection.
        beh_csv = track_csv.with_name(f"{track_csv.stem}_behavior.csv")
        if beh_csv.exists():
            return pts.iloc[0:0][["timestamp", "world_x", "world_y"]]
        return pts
    if "is_bad_quality" not in quality.columns:
        return pts

    merged = pts.merge(
        quality[["timestamp", "is_bad_quality"]],
        on="timestamp",
        how="left",
    )
    is_bad = merged["is_bad_quality"].astype("boolean").fillna(False)
    keep = ~is_bad.to_numpy(dtype=bool)
    filtered = merged.loc[keep, ["timestamp", "world_x", "world_y"]]
    return filtered


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    frames: list[pd.DataFrame] = []
    has_nonempty_source_csv = False
    for csv_path, track_dir in csv_sources:
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: skipping empty CSV with no header/data: {csv_path}")
            continue
        except pd.errors.ParserError as exc:
            print(f"Warning: skipping unparsable CSV: {csv_path} ({exc})")
            continue
        if df.empty:
            print(f"Warning: skipping empty CSV with header but no rows: {csv_path}")
            continue
        has_nonempty_source_csv = True
        label_col = _behavior_label_col(df)

        if "start_time" not in df.columns or "end_time" not in df.columns:
            raise ValueError(f"Missing start_time/end_time in {csv_path}")
        if "identity_label" not in df.columns:
            raise ValueError(f"Missing identity_label in {csv_path}")
        if "cam_id" not in df.columns or "track_filename" not in df.columns:
            raise ValueError(f"Missing cam_id/track_filename in {csv_path}")
        
        ### DO FILTERING -- ONLY KEEP THE TRACK FILES WITH VALID TRACK FRAMES
        df['date'] = pd.to_datetime(df['start_time'], errors='coerce').dt.strftime('%Y-%m-%d')
        unique_tracks = df[["date", "cam_id", "track_filename"]].drop_duplicates()
        valid_track_rows = []
        for _, track_row in unique_tracks.iterrows():
            beh_csv_path = track_dir / f"zag_elp_cam_0{str(track_row['cam_id']).strip()}" / track_row["date"] / f"{track_row['track_filename']}_behavior.csv"
            if not beh_csv_path.exists():
                continue
            df_beh = pd.read_csv(beh_csv_path)
            if df_beh.empty:
                continue
            # if 80% 'bad' quality frames (quality_label == 'bad') in df_beh, or avg behavior_conf < 0.7, then skip this track file
            if "quality_label" in df_beh.columns and "behavior_conf" in df_beh.columns:
                n_bad = (df_beh["quality_label"] == "bad").sum()
                n_total = len(df_beh)
                if n_total > 0 and (n_bad / n_total) >= 0.8:
                    print(f"Skipping track {beh_csv_path} due to high proportion of bad quality frames ({n_bad}/{n_total})")
                    continue
                
                avg_conf = df_beh["behavior_conf"].mean()
                if avg_conf < 0.7:
                    print(f"Skipping track {beh_csv_path} due to low behavior confidence ({avg_conf})")
                    continue
            
            valid_track_rows.append(track_row)

        df = df.copy()
        # keep items with valid track files only
        if valid_track_rows:
            valid_track_filenames = set([row["track_filename"] for row in valid_track_rows])
            df = df[df["track_filename"].isin(valid_track_filenames)]

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
            pd.DataFrame(columns=["start_time", "end_time", "behavior_label", "cam_id", "track_filename", "track_root"]),
            has_nonempty_source_csv,
        )
    merged = pd.concat(frames, ignore_index=True)
    split_bouts, diagnostics = _split_standing_into_standing_walking(
        bouts=merged,
        track_dir=None,
        standing_merge_gap_sec=standing_merge_gap_sec,
        walking_bin_minutes=walking_bin_minutes,
        walking_bin_distance_threshold=walking_bin_distance_threshold,
        movement_step_clip=movement_step_clip,
    )
    return split_bouts, diagnostics, merged, has_nonempty_source_csv


def _build_identity_trajectory_points(
    source_bouts: pd.DataFrame,
    refined_bouts: pd.DataFrame,
) -> pd.DataFrame:
    if source_bouts.empty:
        return pd.DataFrame(
            columns=["timestamp", "world_x", "world_y", "behavior_label", "camera_id", "label_priority"]
        )

    track_cache: dict[Path, pd.DataFrame] = {}
    quality_cache: dict[Path, pd.DataFrame] = {}
    rows: list[pd.DataFrame] = []
    for _, row in source_bouts.iterrows():
        track_filename = row.get("track_filename")
        cam_id = row.get("cam_id")
        track_root = row.get("track_root")
        if pd.isna(track_filename) or pd.isna(cam_id) or pd.isna(track_root):
            continue

        track_csv = _resolve_track_csv_path(Path(str(track_root)), str(cam_id), row["start_time"], str(track_filename))
        if track_csv is None:
            continue

        pts = _read_track_points(track_csv, track_cache)
        if pts.empty:
            continue
        pts = _filter_track_points_by_behavior_quality(pts, track_csv, quality_cache)
        if pts.empty:
            continue

        seg = pts[(pts["timestamp"] >= row["start_time"]) & (pts["timestamp"] <= row["end_time"])].copy()
        if seg.empty:
            continue
        seg["camera_id"] = str(cam_id)
        rows.append(seg)

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "world_x", "world_y", "behavior_label", "camera_id", "label_priority"]
        )

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["timestamp"])
    out = out.drop_duplicates(subset=["timestamp", "camera_id"], keep="last")

    # Re-label trajectory points using refined bouts (standing split into standing/walking).
    out["behavior_label"] = "outside"
    out["label_priority"] = int(LABEL_PRIORITY["outside"])
    if not refined_bouts.empty:
        refined = refined_bouts.sort_values(["start_time", "end_time"]).reset_index(drop=True)
        for _, bout in refined.iterrows():
            label = str(bout["behavior_label"])
            if label not in VALID_LABELS:
                continue
            s = bout["start_time"]
            e = bout["end_time"]
            if pd.isna(s) or pd.isna(e) or s >= e:
                continue
            pr = int(LABEL_PRIORITY.get(label, 0))
            mask = (out["timestamp"] >= s) & (out["timestamp"] <= e) & (out["label_priority"] < pr)
            if not mask.any():
                continue
            out.loc[mask, "behavior_label"] = label
            out.loc[mask, "label_priority"] = pr

    out = out[out["behavior_label"].isin(VALID_LABELS)]
    out = out.drop(columns=["label_priority"])
    return out.reset_index(drop=True)


def _trajectory_behaviour_cmap_map() -> dict[str, str]:
    return {
        "01_standing": "Spectral",
        "02_sleeping_left": "magma",
        "03_sleeping_right": "plasma",
        "walking": "cool",
        "stereotypy": "inferno",
    }


def _normalize_behavior_label_name(label: str) -> str | None:
    key = str(label).strip().lower().replace(" ", "_")
    if key in VALID_LABELS:
        return key
    mapped = GT_LABEL_MAP.get(key)
    if mapped in VALID_LABELS:
        return mapped
    return None


def _resolve_hourly_traj_behaviors(behaviors: Iterable[str] | None) -> list[str]:
    if behaviors is None:
        return list(TRAJ_STANDING_WALKING_LABELS)

    resolved: list[str] = []
    for item in behaviors:
        mapped = _normalize_behavior_label_name(str(item))
        if mapped is None:
            print(f"Warning: unknown behavior for hourly trajectory heatmap, skipped: {item}")
            continue
        if mapped not in resolved:
            resolved.append(mapped)

    if not resolved:
        print("Warning: no valid hourly trajectory behaviors provided; using defaults: standing, walking")
        return list(TRAJ_STANDING_WALKING_LABELS)
    return resolved


def _plot_world_heatmap_by_behaviour(
    df_traj: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    d = df_traj.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp", "world_x", "world_y"])
    d = d[d["behavior_label"].isin(TRAJ_HEATMAP_LABELS)]
    if d.empty:
        return

    cmap_for = _trajectory_behaviour_cmap_map()
    behaviors_to_show = [lbl for lbl in TRAJ_HEATMAP_LABELS if (d["behavior_label"] == lbl).any()]
    n = len(behaviors_to_show)
    if n == 0:
        return

    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, max(4, 3.8 * nrows)),
        dpi=180,
        facecolor="black",
    )
    axes_arr = np.array(axes, dtype=object).reshape(-1)
    x_min, x_max = float(d["world_x"].min()), float(d["world_x"].max())
    y_min, y_max = float(d["world_y"].min()), float(d["world_y"].max())

    for i, beh in enumerate(behaviors_to_show):
        ax = axes_arr[i]
        ax.set_facecolor("black")
        g = d[d["behavior_label"] == beh]
        hb = ax.hexbin(
            g["world_x"].to_numpy(),
            g["world_y"].to_numpy(),
            gridsize=140,
            bins="log",
            cmap=cmap_for.get(str(beh), "Greys"),
            mincnt=1,
            alpha=0.85,
        )
        ax.set_title(LABEL_DISPLAY.get(beh, beh), color="white")
        ax.set_xlabel("world_x", color="white")
        ax.set_ylabel("world_y", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(False)

        cbar = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Density (log scale)", color="white", rotation=270, labelpad=16)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.outline.set_edgecolor("white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(title, color="white")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _plot_world_time_ordered_trajectory(
    df_traj: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    d = df_traj.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp", "world_x", "world_y"]).sort_values("timestamp")
    if d.empty:
        return

    t = d["timestamp"]
    dt = (t - t.min()).dt.total_seconds().to_numpy()
    dt_range = max(float(dt.max()), 1e-9)
    t_norm = dt / dt_range

    x = d["world_x"].to_numpy(dtype=float)
    y = d["world_y"].to_numpy(dtype=float)

    breaks = np.zeros(len(x), dtype=bool)
    if len(x) >= 2:
        gaps = np.diff(dt)
        jumps = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        breaks[1:] = (gaps > 10.0) | (jumps > 20.0)

    fig = plt.figure(figsize=(10, 7), dpi=180, facecolor="black")
    ax = fig.add_subplot(111, facecolor="black")

    ax.hexbin(
        x,
        y,
        gridsize=140,
        bins="log",
        cmap="Greys",
        mincnt=1,
        alpha=0.30,
    )

    sc = None
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or breaks[i]:
            xs = x[start:i]
            ys = y[start:i]
            ts = t_norm[start:i]
            if len(xs) >= 2:
                ax.plot(xs, ys, linewidth=2.0, alpha=0.9)
            sc = ax.scatter(xs, ys, c=ts, s=10.0, cmap="turbo", alpha=0.95, linewidths=0)
            start = i

    ax.set_title(title, color="white")
    ax.set_xlabel("world_x", color="white")
    ax.set_ylabel("world_y", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Time (normalized)", color="white", rotation=270, labelpad=20)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.outline.set_edgecolor("white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _plot_world_heatmap_standing_walking_by_hour(
    df_traj: pd.DataFrame,
    out_path: Path,
    title: str,
    night_start: pd.Timestamp,
    night_end: pd.Timestamp,
    bin_hours: float,
    behaviors: Iterable[str] | None = None,
) -> None:
    if bin_hours <= 0:
        raise ValueError("bin_hours must be positive")

    behaviors_to_show = _resolve_hourly_traj_behaviors(behaviors)
    d = df_traj.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp", "world_x", "world_y"])
    d = d[d["behavior_label"].isin(behaviors_to_show)]
    d = d[(d["timestamp"] >= night_start) & (d["timestamp"] <= night_end)]
    if d.empty:
        return

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
    cmap_for = _trajectory_behaviour_cmap_map()
    per_behavior_root = out_path.parent / out_path.stem
    valid_time_bins: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []
    for r in range(n_bins):
        b_start = time_edges[r]
        b_end = time_edges[r + 1]
        in_bin = (d["timestamp"] >= b_start) & (d["timestamp"] < b_end if r < (n_bins - 1) else d["timestamp"] <= b_end)
        d_bin = d[in_bin]
        if d_bin.empty:
            continue
        valid_time_bins.append((b_start, b_end, d_bin))

        for beh in behaviors_to_show:
            g = d_bin[d_bin["behavior_label"] == beh]
            if g.empty:
                continue

            # Save one PNG per behavior per hour only when there are valid points.
            time_label = f"{b_start:%H:%M}-{b_end:%H:%M}"
            beh_dir = per_behavior_root / str(beh)
            beh_dir.mkdir(parents=True, exist_ok=True)
            slot_png = beh_dir / f"{out_path.stem}_{beh}_{b_start:%Y%m%d_%H%M%S}_{b_end:%Y%m%d_%H%M%S}.png"
            fig_single, ax_single = plt.subplots(figsize=(6, 6), dpi=180, facecolor="black")
            ax_single.set_facecolor("black")
            hb_single = ax_single.hexbin(
                g["world_x"].to_numpy(),
                g["world_y"].to_numpy(),
                gridsize=120,
                bins="log",
                cmap=cmap_for.get(beh, "Greys"),
                mincnt=1,
                alpha=0.90,
            )
            ax_single.set_xlim(x_min, x_max)
            ax_single.set_ylim(y_min, y_max)
            ax_single.set_aspect("equal", adjustable="box")
            ax_single.xaxis.set_major_locator(MultipleLocator(_interval_step(x_min, x_max)))
            ax_single.yaxis.set_major_locator(MultipleLocator(_interval_step(y_min, y_max)))
            ax_single.set_title(f"{LABEL_DISPLAY.get(beh, beh)} | {time_label}", color="white", fontsize=10)
            ax_single.set_xlabel("world_x", color="white")
            ax_single.set_ylabel("world_y", color="white")
            ax_single.tick_params(colors="white")
            for spine in ax_single.spines.values():
                spine.set_color("white")
            ax_single.grid(False)
            cbar_single = fig_single.colorbar(hb_single, ax=ax_single, pad=0.02, fraction=0.046)
            cbar_single.ax.yaxis.set_tick_params(color="white")
            cbar_single.outline.set_edgecolor("white")
            plt.setp(plt.getp(cbar_single.ax.axes, "yticklabels"), color="white")
            fig_single.tight_layout()
            fig_single.savefig(slot_png, facecolor=fig_single.get_facecolor(), bbox_inches="tight")
            plt.close(fig_single)

    if not valid_time_bins:
        return

    ncols = max(1, len(behaviors_to_show))
    nrows = len(valid_time_bins)
    panel_w = 5.2
    panel_h = 4.6
    fig_w = max(11.0, panel_w * ncols)
    fig_h = max(5.0, panel_h * nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        dpi=180,
        facecolor="black",
        gridspec_kw={"wspace": 0.28, "hspace": 0.36},
    )
    axes_arr = np.array(axes, dtype=object).reshape(nrows, ncols)

    for r, (b_start, b_end, d_bin) in enumerate(valid_time_bins):
        time_label = f"{b_start:%H:%M}-{b_end:%H:%M}"
        for c, beh in enumerate(behaviors_to_show):
            ax = axes_arr[r, c]
            ax.set_facecolor("black")
            g = d_bin[d_bin["behavior_label"] == beh]
            if g.empty:
                ax.axis("off")
                continue

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MultipleLocator(_interval_step(x_min, x_max)))
            ax.yaxis.set_major_locator(MultipleLocator(_interval_step(y_min, y_max)))
            ax.set_xlabel("world_x", color="white")
            ax.set_ylabel("world_y", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.grid(False)
            ax.set_title(f"{LABEL_DISPLAY.get(beh, beh)} | {time_label}", color="white", fontsize=9)

            hb = ax.hexbin(
                g["world_x"].to_numpy(),
                g["world_y"].to_numpy(),
                gridsize=120,
                bins="log",
                cmap=cmap_for.get(beh, "Greys"),
                mincnt=1,
                alpha=0.90,
            )
            cbar = fig.colorbar(hb, ax=ax, pad=0.01, fraction=0.040)
            cbar.ax.yaxis.set_tick_params(color="white")
            cbar.outline.set_edgecolor("white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # Control top margin explicitly so title stays close to first row.
    fig.suptitle(title, color="white", y=0.985, fontsize=16)
    fig.subplots_adjust(top=0.965, bottom=0.04, left=0.05, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_trajectory_heatmaps_for_date(
    source_bouts: pd.DataFrame,
    refined_bouts: pd.DataFrame,
    out_dir: Path,
    output_stem: str,
    individual_label: str,
    night_start: pd.Timestamp,
    night_end: pd.Timestamp,
    bin_hours: float,
    traj_hourly_behaviors: Iterable[str] | None = None,
) -> tuple[Path, Path, Path] | None:
    df_traj = _build_identity_trajectory_points(source_bouts, refined_bouts)
    if df_traj.empty:
        return None

    heat_path = out_dir / f"{output_stem}_trajectory_heatmap_world_xy.png"
    traj_heat_path = out_dir / f"{output_stem}_trajectory_time_ordered_world_xy.png"
    standing_walking_hourly_path = out_dir / f"trajs" / f"{output_stem}.png"
    _plot_world_heatmap_by_behaviour(
        df_traj=df_traj,
        out_path=heat_path,
        title=heat_path.name.replace(".png", ""),
    )
    # _plot_world_time_ordered_trajectory(
    #     df_traj=df_traj,
    #     out_path=traj_heat_path,
    #     title=traj_heat_path.name.replace(".png", ""),
    # )
    _plot_world_heatmap_standing_walking_by_hour(
        df_traj=df_traj,
        out_path=standing_walking_hourly_path,
        title=standing_walking_hourly_path.name.replace(".png", ""),
        night_start=night_start,
        night_end=night_end,
        bin_hours=bin_hours,
        behaviors=traj_hourly_behaviors,
    )
    return heat_path, traj_heat_path, standing_walking_hourly_path


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
        "--bin_hours",
        type=float,
        default=1.0,
        help="Hour bin size for standing/walking trajectory heatmaps across the night.",
    )
    parser.add_argument(
        "--traj_hourly_behaviors",
        nargs="+",
        default=["standing", "walking"],
        help="Behaviors to include in hourly trajectory heatmaps (default: standing walking).",
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
        ### no 2025-0501, 2025-10-15 - wrong timestamp
        if date in ("20250501", "20251015", "20251030"):
            print(f"Skipping date {date} due to known timestamp issues in source CSVs.")
            continue


        output_stem = _per_date_output_stem(csv_sources, args.individual_group)
        gt_stem = _remove_date_suffix(output_stem, date)
        bouts, standing_diag, source_bouts, has_nonempty_source_csv = _load_bouts_for_date(
            csv_sources=csv_sources,
            individual_label=args.individual_group,
            standing_merge_gap_sec=args.standing_merge_gap_sec,
            walking_bin_minutes=args.walking_bin_minutes,
            walking_bin_distance_threshold=args.walking_bin_distance_threshold,
            movement_step_clip=args.movement_step_clip,
        )
        if not has_nonempty_source_csv:
            print(f"Skipped date {date}: all bout summary CSVs are empty; no analysis outputs created.")
            continue
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
        out_fig_dir = out_dir / "figures"
        out_fig_dir.mkdir(parents=True, exist_ok=True)

        save_csvs = False
        if save_csvs:
            out_csv_dir = out_dir / "csvs"
            out_csv_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_csv_dir / f"{output_stem}_activity_budget.csv"
            budget.to_csv(out_csv, index=False)

            timeline = _aggregate_timeline(labels, night_start, args.bin_minutes)
            timeline["date"] = date
            timeline_csv = out_csv_dir / f"{output_stem}_activity_timeline_{args.bin_minutes}min.csv"
            timeline.to_csv(timeline_csv, index=False)
            standing_diag_csv = out_csv_dir / f"{output_stem}_standing_walking_diagnostics.csv"
            standing_diag.to_csv(standing_diag_csv, index=False)

            print(f"Saved: {out_csv}")
            print(f"Saved: {timeline_csv}")
            print(f"Saved: {standing_diag_csv}")

        segments = _labels_to_segments(labels, night_start)
        segments["date"] = date
        segments["offset_start_sec"] = (segments["start_time"] - night_start).dt.total_seconds()
        segments["offset_end_sec"] = (segments["end_time"] - night_start).dt.total_seconds()
        # Keep y-axis dates in the all-night plot only for nights with actual bouts.
        if not bouts.empty:
            all_night_segments.append(segments)

        plot_path = None
        # plot_path = out_fig_dir / f"{output_stem}_activity_budget.png"
        # title = f"{output_stem} Activity Budget {date} ({night_start:%H:%M}–{night_end:%H:%M})"
        # _plot_activity_budget(budget, plot_path, title)

        ethogram_path = out_fig_dir / f"{output_stem}_activity_ethogram_{args.bin_minutes}min.png"
        ethogram_title = f"{output_stem} Activity Ethogram {date} ({night_start:%H:%M}–{night_end:%H:%M})"
        _plot_activity_timeline(segments, ethogram_path, ethogram_title, df_gt_segments=gt_segments)
        traj_paths = None
        traj_paths = _plot_trajectory_heatmaps_for_date(
            source_bouts=source_bouts,
            refined_bouts=bouts,
            out_dir=out_fig_dir,
            output_stem=output_stem,
            individual_label=args.individual_group,
            night_start=night_start,
            night_end=night_end,
            bin_hours=args.bin_hours,
            traj_hourly_behaviors=args.traj_hourly_behaviors,
        )
        print(f"Saved: {plot_path}")
        print(f"Saved: {ethogram_path}")
        if traj_paths is not None:
            print(f"Saved: {traj_paths[0]}")
            print(f"Saved: {traj_paths[1]}")
            print(f"Saved: {traj_paths[2]}")
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

    # if all_budgets:
    #     merged_budget = pd.concat(all_budgets, ignore_index=True)
    #     merged_budget = merged_budget.groupby("behavior_label", as_index=False)["duration_sec"].sum()
    #     merged_budget["duration_min"] = merged_budget["duration_sec"] / 60.0
    #     merged_budget["duration_hr"] = merged_budget["duration_sec"] / 3600.0
    #     merged_budget["percent"] = merged_budget["duration_sec"] / merged_budget["duration_sec"].sum() * 100.0

    #     merged_budget_csv = out_root / f"{args.individual_group}_activity_budget_all_nights.csv"
    #     merged_budget_png = out_root / f"{args.individual_group}_activity_budget_all_nights.png"
    #     merged_budget.to_csv(merged_budget_csv, index=False)
    #     _plot_activity_budget(
    #         merged_budget,
    #         merged_budget_png,
    #         f"{args.individual_group} Activity Budget Across All Nights",
    #     )
    #     print(f"Saved: {merged_budget_csv}")
    #     print(f"Saved: {merged_budget_png}")


if __name__ == "__main__":
    run_analysis(parse_args())
