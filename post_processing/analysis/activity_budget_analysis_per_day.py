import argparse
from pathlib import Path
from typing import Any, Iterable
import re
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

try:
    from post_processing.analysis.stereotype_classifier.inference import (
        InferenceBundle,
        load_model_for_inference,
        predict_label_from_image,
    )
except Exception:
    InferenceBundle = None
    load_model_for_inference = None
    predict_label_from_image = None


INDIVIDUALS_TO_ID = {
    "Chandra": 1,
    "Farha": 3,
    "Indi": 2,
    "Panang": 4,
    "Thai": 5,
    "Invalid": 0,
}

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

STEREOTYPY_MODEL_CHECKPOINT = Path(
    "/media/mu/zoo_vision/post_processing/analysis/stereotype_classifier/model.pt"
)
_STEREOTYPY_BUNDLE: Any = None


def _get_stereotypy_inference_bundle() -> Any:
    global _STEREOTYPY_BUNDLE
    if _STEREOTYPY_BUNDLE is not None:
        return _STEREOTYPY_BUNDLE
    if load_model_for_inference is None:
        print("Stereotypy inference import unavailable; falling back to label='no'.")
        return None
    if not STEREOTYPY_MODEL_CHECKPOINT.exists():
        print(f"Stereotypy checkpoint not found: {STEREOTYPY_MODEL_CHECKPOINT}; falling back to label='no'.")
        return None
    try:
        _STEREOTYPY_BUNDLE = load_model_for_inference(STEREOTYPY_MODEL_CHECKPOINT)
    except Exception as exc:
        print(f"Failed to load stereotypy model: {exc}; falling back to label='no'.")
        _STEREOTYPY_BUNDLE = None
    return _STEREOTYPY_BUNDLE


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
            # # ### TODO - NOW ONLY FOR 016-019,
            # if '016' not in str(csv_path) and '019' not in str(csv_path):
            #     continue
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


def _finalize_stereotypy_from_debug(
    df_debug: pd.DataFrame,
    min_consecutive_bins: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only consecutive stereotypy detections and merge them into windows."""
    if df_debug is None or df_debug.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["start_time", "end_time", "behavior_label"])

    dbg = df_debug.copy()
    dbg["start_timestamp"] = pd.to_datetime(dbg["start_timestamp"], errors="coerce")
    dbg["end_timestamp"] = pd.to_datetime(dbg["end_timestamp"], errors="coerce")
    dbg["final_stereotypy"] = False

    walk = dbg[(dbg["behavior_label"] == "walking") & dbg["start_timestamp"].notna() & dbg["end_timestamp"].notna()].copy()
    if walk.empty:
        return dbg, pd.DataFrame(columns=["start_time", "end_time", "behavior_label"])

    walk = walk.sort_values(["start_timestamp", "end_timestamp"])
    bin_span_sec = (walk["end_timestamp"] - walk["start_timestamp"]).dt.total_seconds().median()
    if not np.isfinite(bin_span_sec) or bin_span_sec <= 0:
        bin_span_sec = 120.0
    contiguous_gap = pd.Timedelta(seconds=max(1.0, 0.25 * float(bin_span_sec)))

    true_rows = walk[walk["is_stereotypy"].astype(bool)].copy()
    if true_rows.empty:
        return dbg, pd.DataFrame(columns=["start_time", "end_time", "behavior_label"])

    true_rows = true_rows.sort_values(["start_timestamp", "end_timestamp"])
    runs: list[list[int]] = []
    current: list[int] = []
    prev_end: pd.Timestamp | None = None
    for idx, row in true_rows.iterrows():
        s = pd.Timestamp(row["start_timestamp"])
        e = pd.Timestamp(row["end_timestamp"])
        if prev_end is None or s <= (prev_end + contiguous_gap):
            current.append(idx)
        else:
            if current:
                runs.append(current)
            current = [idx]
        prev_end = e
    if current:
        runs.append(current)

    keep_idxs: list[int] = []
    for run in runs:
        if len(run) >= int(min_consecutive_bins):
            keep_idxs.extend(run)

    if keep_idxs:
        dbg.loc[keep_idxs, "final_stereotypy"] = True

    final_rows = dbg[(dbg["behavior_label"] == "walking") & dbg["final_stereotypy"]].copy()
    if final_rows.empty:
        return dbg, pd.DataFrame(columns=["start_time", "end_time", "behavior_label"])

    final_rows = final_rows.sort_values(["start_timestamp", "end_timestamp"])
    merged: list[dict] = []
    cur_s = pd.Timestamp(final_rows.iloc[0]["start_timestamp"])
    cur_e = pd.Timestamp(final_rows.iloc[0]["end_timestamp"])
    for i in range(1, len(final_rows)):
        s = pd.Timestamp(final_rows.iloc[i]["start_timestamp"])
        e = pd.Timestamp(final_rows.iloc[i]["end_timestamp"])
        if s <= (cur_e + contiguous_gap):
            if e > cur_e:
                cur_e = e
        else:
            merged.append({"start_time": cur_s, "end_time": cur_e, "behavior_label": "stereotypy"})
            cur_s, cur_e = s, e
    merged.append({"start_time": cur_s, "end_time": cur_e, "behavior_label": "stereotypy"})
    merged_df = pd.DataFrame(merged).sort_values(["start_time", "end_time"]).reset_index(drop=True)
    return dbg, merged_df


def _apply_stereotypy_windows_to_labels(
    labels: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
    df_stereotypy_windows: pd.DataFrame | None,
) -> np.ndarray:
    if labels.size == 0 or df_stereotypy_windows is None or df_stereotypy_windows.empty:
        return labels

    out = labels.copy()
    label_codes = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    stereo_code = label_codes["stereotypy"]
    for _, row in df_stereotypy_windows.iterrows():
        s = pd.to_datetime(row.get("start_time"), errors="coerce")
        e = pd.to_datetime(row.get("end_time"), errors="coerce")
        if pd.isna(s) or pd.isna(e):
            continue
        s = max(pd.Timestamp(s), start)
        e = min(pd.Timestamp(e), end)
        if s >= e:
            continue
        s_idx = int((s - start).total_seconds())
        e_idx = int((e - start).total_seconds())
        s_idx = max(0, min(s_idx, len(out)))
        e_idx = max(0, min(e_idx, len(out)))
        if s_idx < e_idx:
            out[s_idx:e_idx] = stereo_code
    return out


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
    df_stereotypy_segments: pd.DataFrame | None = None,
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

    has_stereotypy_overlay = False
    if df_stereotypy_segments is not None and not df_stereotypy_segments.empty:
        st_df = df_stereotypy_segments.copy().sort_values(["start_time", "end_time"])
        starts = np.asarray(mdates.date2num(st_df["start_time"]), dtype=float)
        ends = np.asarray(mdates.date2num(st_df["end_time"]), dtype=float)
        ys = np.full(starts.shape[0], 0.94, dtype=float)
        ax.hlines(
            y=ys,
            xmin=starts,
            xmax=ends,
            colors=LABEL_COLORS["stereotypy"],
            linewidth=2.8,
            zorder=7,
        )
        has_stereotypy_overlay = True

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
                    # print(f"Skipping track {beh_csv_path} due to high proportion of bad quality frames ({n_bad}/{n_total})")
                    continue
                
                avg_conf = df_beh["behavior_conf"].mean()
                if avg_conf < 0.7:
                    # print(f"Skipping track {beh_csv_path} due to low behavior confidence ({avg_conf})")
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
            columns=["timestamp", "world_x", "world_y", "behavior_label", "camera_id", "track_filename", "label_priority"]
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
        seg["track_filename"] = str(track_filename)
        rows.append(seg)

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "world_x", "world_y", "behavior_label", "camera_id", "track_filename", "label_priority"]
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
        im_map = plt.imread(str(floorplan_path))
    except Exception as exc:
        print(f"Warning: failed to load map background ({exc}); fallback to world_x/world_y heatmap.")
        return None

    if im_map.ndim == 3 and im_map.shape[2] > 3:
        im_map = im_map[:, :, :3]

    submap_x = 1450
    submap_y = 1300
    submap_w = 1250
    submap_h = 900
    submap_scale = 0.25

    im_sub = im_map[submap_y:submap_y + submap_h, submap_x:submap_x + submap_w]
    # Match track_heatmap's 0.25 scale using stride downsample.
    im_sub = im_sub[::4, ::4]
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


def _get_camera_pairs(points: pd.DataFrame) -> pd.DataFrame:
    if points is None or points.empty or "camera_id" not in points.columns:
        return points
    cam_ids = points["camera_id"].map(_camera_id_to_int)
    return cam_ids


def _stereotypy_metrics(
    points: pd.DataFrame,
    axis_xlim: tuple[float, float] | None = None,
    axis_ylim: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Compute scale-aware trajectory metrics used for stereotypy scoring."""
    out = {
        "n_points": 0.0,
        "duration_sec": 0.0,
        "coverage": 0.0,
        "robust_coverage": 0.0,
        "elongation": 0.0,
        "revisit_ratio": 0.0,
        "reversals": 0.0,
        "reversal_density": 0.0,
        "shuttle_transitions": 0.0,
    }
    if points is None or points.empty:
        return out
    out["n_points"] = float(len(points))
    if len(points) < 3:
        return out

    p = points.copy()
    if "timestamp" in p.columns:
        p = p.sort_values("timestamp")
        ts = pd.to_datetime(p["timestamp"], errors="coerce").dropna()
        if not ts.empty:
            out["duration_sec"] = float((ts.max() - ts.min()).total_seconds())
    xy = p[["world_x", "world_y"]].to_numpy(dtype=float)
    if xy.shape[0] < 3:
        return out

    x_span = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
    y_span = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    bbox_diag = float(np.hypot(x_span, y_span))
    centered = xy - xy.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))
    minor = max(float(eigvals[0]), 1e-6)
    major = float(eigvals[1])
    elongation = major / minor
    out["elongation"] = float(elongation)

    scale = np.sqrt(max(major, 1e-6))
    bin_size = max(0.35, 0.20 * scale)
    gx = np.floor((xy[:, 0] - xy[:, 0].min()) / bin_size).astype(int)
    gy = np.floor((xy[:, 1] - xy[:, 1].min()) / bin_size).astype(int)
    unique_cells = len(set(zip(gx.tolist(), gy.tolist())))
    revisit_ratio = float(len(xy)) / float(max(unique_cells, 1))
    out["revisit_ratio"] = float(revisit_ratio)

    eigvecs = np.linalg.eigh(cov)[1]
    principal = eigvecs[:, 1]
    proj = centered @ principal
    dproj = np.diff(proj)
    signs = np.sign(dproj)
    signs = signs[signs != 0]
    reversals = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) >= 2 else 0
    reversal_density = float(reversals) / float(max(len(signs) - 1, 1))
    out["reversals"] = float(reversals)
    out["reversal_density"] = float(reversal_density)

    p_min = float(np.min(proj))
    p_max = float(np.max(proj))
    p_span = max(1e-6, p_max - p_min)
    p_norm = (proj - p_min) / p_span
    at_lo = p_norm <= 0.2
    at_hi = p_norm >= 0.8
    end_state = np.where(at_hi, 1, np.where(at_lo, -1, 0))
    end_state = end_state[end_state != 0]
    shuttle_transitions = int(np.sum(end_state[1:] * end_state[:-1] < 0)) if len(end_state) >= 2 else 0
    out["shuttle_transitions"] = float(shuttle_transitions)

    if axis_xlim is None or axis_ylim is None:
        axis_diag = bbox_diag
    else:
        axis_dx = max(1e-6, float(axis_xlim[1]) - float(axis_xlim[0]))
        axis_dy = max(1e-6, float(axis_ylim[1]) - float(axis_ylim[0]))
        axis_diag = float(np.hypot(axis_dx, axis_dy))
        print("axis_diag", axis_diag)
    coverage = float(bbox_diag) / float(max(axis_diag, 1e-6))
    out["coverage"] = float(coverage)
    qx = np.quantile(xy[:, 0], [0.05, 0.95])
    qy = np.quantile(xy[:, 1], [0.05, 0.95])
    robust_diag = float(np.hypot(float(qx[1] - qx[0]), float(qy[1] - qy[0])))
    out["robust_coverage"] = float(robust_diag) / float(max(axis_diag, 1e-6))
    return out


def _apply_dynamic_stereotypy_flags(metrics_df: pd.DataFrame, is_cam_17_18: bool=False) -> pd.DataFrame:
    """Recurrence-first detector for 10-min bins (single recurrence is sufficient)."""
    if metrics_df.empty:
        out = metrics_df.copy()
        out["stereotypy_score"] = pd.Series(dtype=float)
        out["is_stereotypy"] = pd.Series(dtype=bool)
        out["gate_base"] = pd.Series(dtype=bool)
        out["gate_short_guard"] = pd.Series(dtype=bool)
        out["gate_override"] = pd.Series(dtype=bool)
        return out

    out = metrics_df.copy()
    out["stereotypy_score"] = 0.0
    out["is_stereotypy"] = False
    out["gate_base"] = False
    out["gate_short_guard"] = False
    out["gate_override"] = False

    m = out.copy()
    m = m[m["behavior_label"].astype(str) == "walking"].copy()
    # 10-min bins can have sparse points; keep them.
    m = m[m["n_points"] >= 12].copy()
    if m.empty:
        return out

    m["is_cam_17_18"] = is_cam_17_18

    rank_cols = [
        "coverage",
        "robust_coverage",
        "elongation",
        "revisit_ratio",
        "reversals",
        "reversal_density",
        "shuttle_transitions",
    ]
    for col in rank_cols:
        m[f"{col}_rank"] = m[col].rank(method="average", pct=True)

    # Keep score for diagnostics; flagging uses recurrence-first logic below.
    m["stereotypy_score_raw"] = (
        0.25 * m["revisit_ratio_rank"]
        + 0.25 * m["shuttle_transitions_rank"]
        + 0.18 * m["reversal_density_rank"]
        + 0.14 * m["reversals_rank"]
        + 0.10 * m["elongation_rank"]
        + 0.08 * m["robust_coverage_rank"]
    )
    # Reduce score inflation for sparse windows while still allowing strong sparse patterns.
    point_conf = ((m["n_points"] - 12.0) / 78.0).clip(lower=0.0, upper=1.0)
    m["stereotypy_score"] = m["stereotypy_score_raw"] * (0.55 + 0.45 * point_conf)

    score_cut = float(max(0.65, m["stereotypy_score"].quantile(0.75)))
    revisit_cut = float(max(1.15, m["revisit_ratio"].quantile(0.40)))
    reversal_den_cut = float(max(0.07, m["reversal_density"].quantile(0.35)))

    # Cam17/18: stricter base thresholds to avoid false positives on incomplete trajectories.
    cam_revisit_cut = float(max(revisit_cut, 3.0))
    cam_reversal_den_cut = float(max(reversal_den_cut, 0.20))
    cam_robust_cov_min = 0.45
    cam_cov_min = 0.50
    cam_min_duration = 15.0
    cam_min_shuttle = 1.0

    # Single recurrence accepted: one end-to-end transition with loop-like revisits.
    base_rule = (
        (m["stereotypy_score"] >= score_cut)
        & (m["shuttle_transitions"] >= 1.0)
        & (
            (~m["is_cam_17_18"])
            | (m["shuttle_transitions"] >= cam_min_shuttle)
        )
        & (
            (~m["is_cam_17_18"] & (m["revisit_ratio"] >= revisit_cut))
            | (m["is_cam_17_18"] & (m["revisit_ratio"] >= cam_revisit_cut))
        )
        & (
            (~m["is_cam_17_18"] & ((m["reversals"] >= 2.0) | (m["reversal_density"] >= reversal_den_cut)))
            | (m["is_cam_17_18"] & ((m["reversals"] >= 2.0) | (m["reversal_density"] >= cam_reversal_den_cut)))
        )
        & (
            (~m["is_cam_17_18"] & (m["robust_coverage"] >= 0.045))
            | (m["is_cam_17_18"] & (m["robust_coverage"] >= cam_robust_cov_min))
        )
        & (
            (~m["is_cam_17_18"])
            | (m["coverage"] >= cam_cov_min)
        )
        & (
            (~m["is_cam_17_18"] | (m["duration_sec"] >= cam_min_duration))
        )
        & (m["elongation"] >= 1.15)
    )
    short_window = m["duration_sec"] <= np.where(m["is_cam_17_18"], 60.0, 70.0)
    short_guard = (
        (m["shuttle_transitions"] >= 2.0)
        & (m["reversals"] >= 3.0)
        & (m["robust_coverage"] >= np.where(m["is_cam_17_18"], 0.45, 0.065))
        & (
            (m["revisit_ratio"] >= float(max(1.20, revisit_cut)))
            | (m["is_cam_17_18"] & (m["revisit_ratio"] >= max(3.0, cam_revisit_cut)))
        )
    )
    strong_recurrence_override = (
        (m["shuttle_transitions"] >= 2.0)
        & (m["revisit_ratio"] >= 8.0)
        & (m["reversals"] >= 80.0)
        & (m["reversal_density"] >= 0.20)
        & (m["robust_coverage"] >= 0.18)
        & (m["duration_sec"] >= 70.0)
    )
    m["gate_base"] = base_rule
    m["gate_short_guard"] = short_guard
    m["gate_override"] = strong_recurrence_override
    m["is_stereotypy"] = (base_rule & (~short_window | short_guard)) | strong_recurrence_override

    # Prevent degenerate all/none labeling in very small samples.
    if len(m) < 4:
        m["is_stereotypy"] = (
            (m["stereotypy_score"] >= 0.65)
            & (m["shuttle_transitions"] >= 1)
            & (m["revisit_ratio"] >= 1.15)
            & ((m["reversals"] >= 2) | (m["reversal_density"] >= 0.07))
            & (m["robust_coverage"] >= 0.05)
            & (m["elongation"] >= 1.10)
        )

    out = out.merge(
        m[["panel_key", "stereotypy_score", "is_stereotypy", "gate_base", "gate_short_guard", "gate_override"]],
        on="panel_key",
        how="left",
        suffixes=("", "_new"),
    )
    out["stereotypy_score"] = out["stereotypy_score_new"].combine_first(out["stereotypy_score"]).fillna(0.0).infer_objects(copy=False)
    out["is_stereotypy"] = out["is_stereotypy_new"].combine_first(out["is_stereotypy"]).fillna(False).infer_objects(copy=False)
    out["gate_base"] = out["gate_base_new"].combine_first(out["gate_base"]).fillna(False).infer_objects(copy=False)
    out["gate_short_guard"] = out["gate_short_guard_new"].combine_first(out["gate_short_guard"]).fillna(False).infer_objects(copy=False)
    out["gate_override"] = out["gate_override_new"].combine_first(out["gate_override"]).fillna(False).infer_objects(copy=False)
    out = out.drop(
        columns=[
            "stereotypy_score_new",
            "is_stereotypy_new",
            "gate_base_new",
            "gate_short_guard_new",
            "gate_override_new",
        ]
    )
    return out


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
    bg_info = _get_submap_background()
    use_map_bg = bg_info is not None
    if use_map_bg:
        im_sub, t_sub_from_world2 = bg_info
        sub_h, sub_w = int(im_sub.shape[0]), int(im_sub.shape[1])
    else:
        x_min, x_max = float(d["world_x"].min()), float(d["world_x"].max())
        y_min, y_max = float(d["world_y"].min()), float(d["world_y"].max())

    for i, beh in enumerate(behaviors_to_show):
        ax = axes_arr[i]
        ax.set_facecolor("black")
        g = d[d["behavior_label"] == beh]
        if use_map_bg:
            xs, ys = _world_to_submap_xy(
                g["world_x"].to_numpy(dtype=float),
                g["world_y"].to_numpy(dtype=float),
                t_sub_from_world2,
                sub_w,
                sub_h,
            )
            if len(xs) == 0:
                ax.axis("off")
                continue
            ax.imshow(im_sub, zorder=0)
            hb = ax.hexbin(
                xs,
                ys,
                gridsize=140,
                bins="log",
                cmap=cmap_for.get(str(beh), "Greys"),
                mincnt=1,
                alpha=0.78,
                zorder=1,
            )
            ax.set_xlim(0, sub_w)
            ax.set_ylim(sub_h, 0)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        else:
            hb = ax.hexbin(
                g["world_x"].to_numpy(),
                g["world_y"].to_numpy(),
                gridsize=140,
                bins="log",
                cmap=cmap_for.get(str(beh), "Greys"),
                mincnt=1,
                alpha=0.85,
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        ax.set_title(LABEL_DISPLAY.get(beh, beh), color="white")
        ax.set_xlabel("world_x", color="white")
        ax.set_ylabel("world_y", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
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
    
    # mkdir out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
    bg_info = _get_submap_background()
    use_map_bg = False #bg_info is not None. TODO-rename it
    if use_map_bg:
        im_sub, t_sub_from_world2 = bg_info
        sub_h, sub_w = int(im_sub.shape[0]), int(im_sub.shape[1])

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
            if cam_ids_series is not None:
                cam_ids_list = sorted(pd.Series(cam_ids_series).dropna().astype(int).unique().tolist())
            cam_ids_str = ",".join(str(cid) for cid in cam_ids_list)

            slot_png = out_dir / f"{title}_{beh}_{b_start:%Y%m%d_%H%M%S}_{b_end:%Y%m%d_%H%M%S}.png"
            fig_single, ax_single = plt.subplots(figsize=(6, 6), dpi=180, facecolor="black")
            ax_single.set_facecolor("black")
            if use_map_bg:
                xs, ys = _world_to_submap_xy(
                    g["world_x"].to_numpy(dtype=float),
                    g["world_y"].to_numpy(dtype=float),
                    t_sub_from_world2,
                    sub_w,
                    sub_h,
                )
                if len(xs) == 0:
                    plt.close(fig_single)
                    continue
                ax_single.imshow(im_sub, zorder=0)
                ax_single.hexbin(
                    xs,
                    ys,
                    gridsize=120,
                    bins="log",
                    cmap=cmap_for.get(beh, "Greys"),
                    mincnt=1,
                    alpha=0.78,
                    zorder=1,
                )
                ax_single.set_xlim(0, sub_w)
                ax_single.set_ylim(sub_h, 0)
                ax_single.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                ax_single.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            else:
                ax_single.hexbin(
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
            if use_map_bg:
                ax_single.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                ax_single.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            else:
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
            
            ### DEBUG - mv slot_png to slot_png.parent / is_stereotypy / slot_png.name 
            if is_stereotypy:
                dest_dir = slot_png.parent / "stereotypy"
            else:
                dest_dir = slot_png.parent / "non_stereotypy"
                
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / slot_png.name
            try:
                slot_png.rename(dest_path)
                slot_png = dest_path
            except Exception as exc:
                print(f"Failed to move {slot_png} to {dest_path}: {exc}")

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
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame] | None:
    df_traj = _build_identity_trajectory_points(source_bouts, refined_bouts)
    if df_traj.empty:
        return None

    heat_path = out_dir / f"{output_stem}_trajectory_heatmap_world_xy.png"
    traj_heat_path = out_dir / f"{output_stem}_trajectory_time_ordered_world_xy.png"
    _plot_world_heatmap_by_behaviour(
        df_traj=df_traj,
        out_path=heat_path,
        title=heat_path.name.replace(".png", ""),
    )
    
    stereotypy_flags, stereotypy_debug = _plot_world_heatmap_standing_walking_bin(
        df_traj=df_traj,
        out_dir=out_dir / "trajs",
        title=output_stem,
        night_start=night_start,
        night_end=night_end,
        bin_hours=bin_hours,
        behaviors=traj_hourly_behaviors,
    )
    return heat_path, traj_heat_path, stereotypy_flags, stereotypy_debug


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
    parser = argparse.ArgumentParser(description="Generate per-date ethogram CSV with exact behavior start/end timestamps")
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
        "--date",
        required=True,
        help="Date to process, e.g. 2026-02-05 or 20260205.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/media/mu/zoo_vision/post_processing/analysis/activity_budgets-perday",
        help="Output root directory for generated CSVs.",
    )
    parser.add_argument(
        "--individual_group",
        type=str,
        default="Thai",
        help="Optional keyword to filter bout summary CSV files. Default is 'Thai'.",
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
        default=1/30,  ## digit-8 patterns: 2-min (cam016/019 ) or 4-min (cam017/018) bins work well; 
        help="Hour bin size for standing/walking trajectory heatmaps across the night.",
    )
    parser.add_argument(
        "--traj_hourly_behaviors",
        nargs="+",
        default=["standing", "walking"],
        help="Behaviors to include in hourly trajectory heatmaps (default: standing walking).",
    )
    parser.add_argument(
        "--plotting",
        action="store_true",
        help="If set, save ethogram and trajectory plots.",
    )
    return parser.parse_args()


def run_analysis(args: argparse.Namespace, plotting: bool | None = None) -> None:
    out_root = Path(args.out_root)
    if plotting is None:
        plotting = bool(getattr(args, "plotting", False))
    record_roots = _resolve_record_roots(args)
    source_configs = _build_source_configs(record_roots, args.output_dir)

    for output_dir, _ in source_configs:
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

    date = normalize_date(args.date)
    filename_keyword = args.individual_group if args.individual_group else None
    csv_sources: list[tuple[Path, Path]] = []
    for output_dir, track_dir in source_configs:
        bouts = get_bout_csvs(
            output_dir=output_dir,
            dates=[date],
            filename_keyword=filename_keyword,
            strict=False,
        )
        for _, csv_path in bouts:
            csv_sources.append((csv_path, track_dir))

    if not csv_sources:
        raise FileNotFoundError(f"No bout summary CSV files found for date={date}")

    output_stem = _per_date_output_stem(csv_sources, args.individual_group)
    bouts, standing_diag, source_bouts, has_nonempty_source_csv = _load_bouts_for_date(
        csv_sources=csv_sources,
        individual_label=args.individual_group,
        standing_merge_gap_sec=args.standing_merge_gap_sec,
        walking_bin_minutes=args.walking_bin_minutes,
        walking_bin_distance_threshold=args.walking_bin_distance_threshold,
        movement_step_clip=args.movement_step_clip,
    )
    if not has_nonempty_source_csv:
        raise ValueError(f"All source CSVs for date={date} are empty.")

    max_ts = None
    if not bouts.empty:
        max_ts = max(bouts["end_time"].max(), bouts["start_time"].max())
    night_start, night_end = _night_window(date, max_ts)
    labels = _build_timeline_labels(bouts, night_start, night_end)

    out_dir = out_root / date
    out_csv_dir = out_dir / "csvs"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir = out_dir / "figures"
    if plotting:
        out_fig_dir.mkdir(parents=True, exist_ok=True)

    final_stereotypy_windows = pd.DataFrame(columns=["start_time", "end_time", "behavior_label"])
    cam_ids = source_bouts["cam_id"].map(_camera_id_to_int).dropna().astype(int).unique().tolist() if not source_bouts.empty else []
    can_run_stereotypy = args.individual_group == "Thai"
    traj_paths = None
    if plotting:
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
        if traj_paths is not None:
            print(f"Saved: {traj_paths[0]}")

    if can_run_stereotypy:
        if traj_paths is not None:
            stereotypy_flags = traj_paths[2]
            stereotypy_debug = traj_paths[3]
        else:
            df_traj = _build_identity_trajectory_points(source_bouts, bouts)
            stereotypy_flags, stereotypy_debug = _plot_world_heatmap_standing_walking_bin(
                df_traj=df_traj,
                out_dir=out_csv_dir,
                title=output_stem,
                night_start=night_start,
                night_end=night_end,
                bin_hours=args.bin_hours,
                behaviors=args.traj_hourly_behaviors,
            )
        if not stereotypy_debug.empty:
            stereotypy_debug, final_stereotypy_windows = _finalize_stereotypy_from_debug(
                stereotypy_debug,
                min_consecutive_bins=2,
            )
            debug_csv = out_csv_dir / f"{output_stem}_stereotypy_debug.csv"
            stereotypy_debug.to_csv(debug_csv, index=False)
            print(f"Saved: {debug_csv}")
        if not final_stereotypy_windows.empty:
            final_windows_csv = out_csv_dir / f"{output_stem}_stereotypy_final_windows.csv"
            final_stereotypy_windows.to_csv(final_windows_csv, index=False)
            print(f"Saved: {final_windows_csv}")
            labels = _apply_stereotypy_windows_to_labels(
                labels=labels,
                start=night_start,
                end=night_end,
                df_stereotypy_windows=final_stereotypy_windows,
            )
            for _, row in final_stereotypy_windows.iterrows():
                print(f"stereotypy: {row['start_time']} -> {row['end_time']}")
        if not stereotypy_flags.empty:
            stereotypy_csv = out_csv_dir / f"{output_stem}_stereotypy_flags.csv"
            stereotypy_flags.to_csv(stereotypy_csv, index=False)
            print(f"Saved: {stereotypy_csv}")
    else:
        print(
            "Skipping stereotypy detection: supported only for individual_group='Thai' "
            "with camera IDs including 016 and 019."
        )

    segments = _labels_to_segments(labels, night_start)
    ethogram = pd.DataFrame({
        "identity-id": args.individual_group,
        "start-dt": segments["start_time"],
        "end-dt": segments["end_time"],
        "label": segments["behavior_label"],
    })
    ethogram_csv = out_csv_dir / f"{output_stem}_ethogram.csv"
    ethogram.to_csv(ethogram_csv, index=False)
    print(f"Saved: {ethogram_csv}")
    if plotting:
        ethogram_plot = out_fig_dir / f"{output_stem}_ethogram.png"
        ethogram_title = f"{output_stem} Activity Ethogram {date} ({night_start:%H:%M}–{night_end:%H:%M})"
        _plot_activity_timeline(
            segments,
            ethogram_plot,
            ethogram_title,
            df_gt_segments=None,
            df_stereotypy_segments=final_stereotypy_windows,
        )
        print(f"Saved: {ethogram_plot}")

    standing_diag_csv = out_csv_dir / f"{output_stem}_standing_walking_diagnostics.csv"
    standing_diag.to_csv(standing_diag_csv, index=False)
    print(f"Saved: {standing_diag_csv}")

    for _, row in ethogram[ethogram["label"].isin(["walking", "stereotypy"])].iterrows():
        print(f"{row['label']}: {row['start-dt']} -> {row['end-dt']}")


if __name__ == "__main__":
    run_analysis(parse_args())
