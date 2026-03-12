import argparse
from pathlib import Path
from typing import Iterable
import re
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from post_processing.analysis.utils import *

try:
    from post_processing.analysis.utils import (
        export_stereotypy_event_videos_from_csv,
        _plot_world_heatmap_standing_walking_bin,
        save_ethogram_csv,
        analyze_ethogram_and_plot_activity_budget,
        get_bout_csvs,
    )
except ImportError:
    from utils import (
        export_stereotypy_event_videos_from_csv,
        _plot_world_heatmap_standing_walking_bin,
        save_ethogram_csv,
        analyze_ethogram_and_plot_activity_budget,
        get_bout_csvs,
    )


def normalize_date(date_str: str) -> str:
    return date_str.replace("-", "")


def get_available_dates(output_dir: Path) -> list[str]:
    return sorted(
        p.name for p in output_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8
    )


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

    labels = np.full(total_seconds, label_codes["no_observation"], dtype=np.int16)
    priorities = np.full(total_seconds, LABEL_PRIORITY["no_observation"], dtype=np.int16)

    if bouts.empty:
        return pd.DataFrame({
            "behavior_label": ["no_observation"],
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
        label = code_to_label.get(code, "no_observation")
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
    labels = np.full(total_seconds, label_codes["no_observation"], dtype=np.int16)
    priorities = np.full(total_seconds, LABEL_PRIORITY["no_observation"], dtype=np.int16)

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
        label = code_to_label.get(code, "no_observation")
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
    group_info_by_date: dict[str, dict[str, str]] | None = None,
    LABEL_COLORS: dict[str, str] = LABEL_COLORS,
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

    anchor = pd.Timestamp("2000-01-01 18:00:00")
    date_to_y = {d: i for i, d in enumerate(dates_sorted)}

    y_labels: list[str] = []
    y_colors: list[str] = []
    for d_txt in dates_sorted:
        date_txt = pd.to_datetime(d_txt, format="%Y%m%d").strftime("%Y-%m-%d")
        info = {} if group_info_by_date is None else group_info_by_date.get(str(d_txt), {})
        group1_txt = str(info.get("group1", "unknown"))
        group2_txt = str(info.get("group2", "unknown"))
        raw_cam_txt = str(info.get("group1_camera_ids", "unknown"))
        cam_txt = camera_ids_to_room_label_text(raw_cam_txt)
        raw_cam_tokens = [t.strip() for t in raw_cam_txt.split(",") if t.strip()]
        if 'Panang' in group2_txt:
            text_color = "#B00020"
        else:
            text_color = "#333333"        
        y_labels.append(f"{date_txt}\nAnother Group:\n {group2_txt}")
        y_colors.append(text_color)

    gt_present_labels: list[str] = []
    if not gt_df.empty and "behavior_label" in gt_df.columns:
        for gt_label in sorted(gt_df["behavior_label"].dropna().astype(str).unique()):
            if gt_label in GT_LABEL_COLORS:
                gt_present_labels.append(gt_label)

    x0 = mdates.date2num(anchor)
    max_end_offset = 0.0
    if not df.empty:
        max_end_offset = max(max_end_offset, float(df["offset_end_sec"].max()))
    if not gt_df.empty:
        max_end_offset = max(max_end_offset, float(gt_df["offset_end_sec"].max()))
    x1 = mdates.date2num(anchor + pd.Timedelta(seconds=max_end_offset))
    fig_h = max(11.0, 0.7 * len(dates_sorted) + 2.0)
    fig, ax = plt.subplots(1, 1, figsize=(17, fig_h))

    if not df.empty:
        for label in LABEL_ORDER:
            dfl = df[df["behavior_label"] == label]
            if dfl.empty:
                continue
            starts = mdates.date2num(anchor + pd.to_timedelta(dfl["offset_start_sec"], unit="s"))
            widths = (dfl["offset_end_sec"] - dfl["offset_start_sec"]).to_numpy(dtype=float) / 86400.0
            ys = [date_to_y[d] for d in dfl["date"].astype(str)]
            ax.barh(
                ys,
                widths,
                left=starts,
                height=0.56,
                color=LABEL_COLORS[label],
                linewidth=0.3,
                label="_nolegend_",
            )

    if not gt_df.empty:
        gt_start_nums = mdates.date2num(anchor + pd.to_timedelta(gt_df["offset_start_sec"], unit="s"))
        gt_end_nums = mdates.date2num(anchor + pd.to_timedelta(gt_df["offset_end_sec"], unit="s"))
        for gt_label, dfg in gt_df.groupby("behavior_label"):
            color = GT_LABEL_COLORS.get(gt_label)
            if color is None:
                continue
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

    ax.set_xlim(x0, x1)
    ax.set_yticks(range(len(dates_sorted)))
    ax.set_yticklabels(y_labels, fontsize=9)
    for tick, color in zip(ax.get_yticklabels(), y_colors):
        tick.set_color(color)
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
    fig.suptitle(title, y=0.995, fontsize=14, fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 0.86, 0.985))
    # make fig - background transparent
    fig.patch.set_alpha(0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    # save svg
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, format="svg", dpi=150)
    plt.close(fig)


def _plot_activity_timeline_with_pie_multi_night(
    df_segments_all: pd.DataFrame,
    out_path: Path,
    title: str,
    group_info_by_date: dict[str, dict[str, str]] | None = None,
) -> None:
    df = df_segments_all.copy()
    if df.empty:
        raise ValueError("No multi-night timeline segments to plot")

    dates_sorted = sorted(df["date"].astype(str).unique())
    anchor = pd.Timestamp("2000-01-01 18:00:00")
    x0 = mdates.date2num(anchor)
    max_end_offset = float(df["offset_end_sec"].max()) if "offset_end_sec" in df.columns else 0.0
    x1 = mdates.date2num(anchor + pd.Timedelta(seconds=max_end_offset))

    n_rows = len(dates_sorted)
    fig_h = max(7.0, 2.2 * n_rows + 1.2)
    fig = plt.figure(figsize=(20, fig_h))
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=3,
        width_ratios=[1.4, 6.6, 5.0],
        hspace=0.24,
        wspace=0.26,
    )

    for i, date in enumerate(dates_sorted):
        day_df = df[df["date"].astype(str) == str(date)].copy()

        ax_date = fig.add_subplot(gs[i, 0])
        ax_date.axis("off")
        date_txt = pd.to_datetime(date, format="%Y%m%d").strftime("%Y-%m-%d")
        ax_date.text(
            0.5,
            0.70,
            date_txt,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        info = {}
        if group_info_by_date is not None:
            info = group_info_by_date.get(str(date), {})
        group1_txt = str(info.get("group1", "unknown"))
        group2_txt = str(info.get("group2", "unknown"))
        raw_cam_txt = str(info.get("group1_camera_ids", "unknown"))
        cam_txt = camera_ids_to_room_label_text(raw_cam_txt)
        raw_cam_tokens = [t.strip() for t in raw_cam_txt.split(",") if t.strip()]
        if any(t in {"16", "019", "19", "016"} for t in raw_cam_tokens):
            text_color = "#B00020"
        else:
            text_color = "#333333"
        ax_date.text(
            0.5,
            0.22,
            f"{group1_txt} | {group2_txt}\nroom: {cam_txt}",
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
        )
        if i == 0:
            ax_date.set_title("Date", fontsize=12, fontweight="bold")

        ax_eth = fig.add_subplot(gs[i, 1])
        for label in LABEL_ORDER:
            dfl = day_df[day_df["behavior_label"] == label]
            if dfl.empty:
                continue
            starts = mdates.date2num(anchor + pd.to_timedelta(dfl["offset_start_sec"], unit="s"))
            widths = (dfl["offset_end_sec"] - dfl["offset_start_sec"]).to_numpy(dtype=float) / 86400.0
            ax_eth.bar(
                starts,
                np.ones(len(dfl)),
                width=widths,
                bottom=0.0,
                align="edge",
                color=LABEL_COLORS.get(label, "#9E9E9E"),
                linewidth=0.2,
            )
        ax_eth.set_xlim(x0, x1)
        ax_eth.set_ylim(0, 1.0)
        ax_eth.set_yticks([])
        ax_eth.xaxis_date()
        ax_eth.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_eth.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_eth.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        ax_eth.spines["top"].set_visible(False)
        ax_eth.spines["right"].set_visible(False)
        ax_eth.spines["left"].set_visible(False)
        if i == 0:
            ax_eth.set_title("Ethogram", fontsize=12, fontweight="bold")
        legend_handles = [
            Patch(facecolor=LABEL_COLORS[label], edgecolor="none", label=LABEL_DISPLAY[label])
            for label in LABEL_ORDER
        ]
        ax_eth.legend(
            handles=legend_handles,
            ncol=1,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.05, 1.0),
            borderaxespad=0.0,
            title="Behavior",
        )
        if i < (n_rows - 1):
            ax_eth.tick_params(axis="x", labelbottom=False)

        ax_pie = fig.add_subplot(gs[i, 2])
        if day_df.empty:
            ax_pie.text(0.5, 0.5, "No observations", ha="center", va="center", fontsize=10)
            ax_pie.axis("off")
        else:
            budget = (
                day_df.groupby("behavior_label", as_index=False)["duration_sec"]
                .sum()
                .sort_values("duration_sec", ascending=False)
                .reset_index(drop=True)
            )
            values = (budget["duration_sec"].to_numpy(dtype=float) / 60.0)
            pie_labels = [
                f"{mins:.1f} min"
                for lbl, mins in zip(budget["behavior_label"].tolist(), values.tolist())
            ]
            pie_legend = budget["behavior_label"].tolist()
            pie_legend = [LABEL_DISPLAY.get(lbl, lbl) for lbl in pie_legend]
            pie_colors = [
                LABEL_COLORS.get(lbl, "#9E9E9E")
                for lbl in budget["behavior_label"].tolist()
            ]
            wedges, _, _ = ax_pie.pie(
                values,
                labels=pie_labels,
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
                labeldistance=1.06,
                pctdistance=0.72,
                colors=pie_colors,
                textprops={"fontsize": 8},
            )
            ax_pie.axis("equal")
        if i == 0:
            ax_pie.set_title("budget", fontsize=12, fontweight="bold")
    fig.suptitle(title, y=0.995, fontsize=14, fontweight="bold")
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
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
    out["behavior_label"] = "no_observation"
    out["label_priority"] = int(LABEL_PRIORITY["no_observation"])
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


def _filter_stereotypy_camera_points(points: pd.DataFrame) -> pd.DataFrame:
    if points is None or points.empty or "camera_id" not in points.columns:
        return points
    cam_ids = points["camera_id"].map(_camera_id_to_int)
    return points[cam_ids.isin(STEREOTYPY_CAMERA_IDS)].copy()


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
    label_root: Path | None = None,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame] | None:
    df_traj = _build_identity_trajectory_points(source_bouts, refined_bouts)
    if df_traj.empty:
        return None

    heat_path = out_dir / f"{output_stem}_trajectory_heatmap_world_xy.png"
    traj_heat_path = out_dir / f"{output_stem}_trajectory_time_ordered_world_xy.png"
    # _plot_world_heatmap_by_behaviour(
    #     df_traj=df_traj,
    #     out_path=heat_path,
    #     title=heat_path.name.replace(".png", ""),
    # )
    stereotypy_flags, stereotypy_debug = _plot_world_heatmap_standing_walking_bin(
        df_traj=df_traj,
        out_dir=out_dir / "trajs",
        title=output_stem,
        night_start=night_start,
        night_end=night_end,
        bin_hours=bin_hours,
        behaviors=traj_hourly_behaviors,
    )
    # TODO - plot world heatmap by behavior
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
        default="/media/mu/zoo_vision/post_processing/analysis/frames_for_slides/activity_budgets-map",
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
        "--gt_root",
        type=str,
        default="/media/mu/zoo_vision/data/GT_id_behavior/behavior_GTs",
        help="Root directory for GT CSV files. Files are read/written under <gt_root>/<individual_group>/.",
    )
    parser.add_argument(
        "--extract_video",
        action="store_true",
        help="If set, export stereotypy event videos under <out_root>/videos from merged stereotypy CSV.",
    )
    return parser.parse_args()


def run_analysis(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    merged_stereotypy_csv = out_root / f"{args.individual_group}_stereotypy_flags_all_nights.csv"
    if args.extract_video and merged_stereotypy_csv.exists():
        manifest_csv = export_stereotypy_event_videos_from_csv(
            stereotypy_csv=merged_stereotypy_csv,
            out_root=out_root,
            individual_group=args.individual_group,
        )
        if manifest_csv is not None:
            print(f"Saved: {manifest_csv}")
        else:
            print(f"No videos exported from: {merged_stereotypy_csv}")
        return

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
    gt_dir = Path(args.gt_root) / args.individual_group

    by_date: dict[str, list[tuple[Path, Path]]] = {}
    by_date_other_groups: dict[str, set[str]] = {}
    by_date_group1: dict[str, set[str]] = {}
    by_date_camera_ids: dict[str, set[int]] = {}
    for output_dir, track_dir in source_configs:
        bouts = get_bout_csvs(
            output_dir=output_dir,
            dates=dates,
            filename_keyword=filename_keyword,
            strict=False,
        )
        for date, csv_path, group1, other_group, camera_ids_txt in bouts:
            by_date.setdefault(date, []).append((csv_path, track_dir))
            grp1 = str(group1).strip()
            if grp1 and grp1.lower() != "unknown":
                by_date_group1.setdefault(date, set()).add(grp1)
            grp = str(other_group).strip()
            if grp and grp.lower() != "unknown":
                by_date_other_groups.setdefault(date, set()).add(grp)
            if str(camera_ids_txt).strip():
                for token in str(camera_ids_txt).split(","):
                    token = token.strip()
                    if token.isdigit():
                        by_date_camera_ids.setdefault(date, set()).add(int(token))

    if not by_date:
        raise FileNotFoundError("No bout summary CSV files found for selected roots/dates")

    all_night_segments: list[pd.DataFrame] = []
    all_night_gt_segments: list[pd.DataFrame] = []
    all_budgets: list[pd.DataFrame] = []
    all_stereotypy_flags: list[pd.DataFrame] = []
    group_info_by_date: dict[str, dict[str, str]] = {}

    for date, csv_sources in sorted(by_date.items()):
        ### no 2025-0501, 2025-10-15 - wrong timestamp
        if date in ("20250501", "20251015", "20251030"):
            print(f"Skipping date {date} due to known timestamp issues in source CSVs.")
            continue
        # ## DEBUG
        # if date not in ("20260204", "20260205"):
        #     continue
        

        output_stem = _per_date_output_stem(csv_sources, args.individual_group)
        group1_title = ", ".join(sorted(by_date_group1.get(date, set()))) or str(args.individual_group)
        other_group_title = ", ".join(sorted(by_date_other_groups.get(date, set()))) or "unknown"
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
        cam_ids = sorted(by_date_camera_ids.get(date, set()))
        labels = _build_timeline_labels(bouts, night_start, night_end)

        out_dir = out_root / date
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv_dir = out_dir / "csvs"
        out_csv_dir.mkdir(parents=True, exist_ok=True)
        out_fig_dir = out_dir / "figures"
        out_fig_dir.mkdir(parents=True, exist_ok=True)

        final_stereotypy_windows = pd.DataFrame(columns=["start_time", "end_time", "behavior_label"])
        
        # Determine group2 from semi-GT data if available
        final_group2_title = str(other_group_title)
        try:
            # Parse group1 individuals
            group1_individuals = set()
            for name in str(group1_title).split(","):
                name = name.strip()
                if name and name.lower() != "unknown":
                    group1_individuals.add(name)
            
            # Try to load semi-GT IDs for this date and camera(s)
            semi_gt_individuals = set()
            formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"  # Convert YYYYMMDD to YYYY-MM-DD
            for cam_id in cam_ids:
                cam_id_str = str(cam_id).zfill(3)  # Convert to '016', '017', etc.
                another_cam_id_str = "016" if cam_id_str == "016" or cam_id_str=='019' else "017"
                semi_gt_ids = load_semi_gt_ids(date=formatted_date, camera_id=cam_id_str)
                if semi_gt_ids:
                    semi_gt_individuals.update(semi_gt_ids)
            
            # If we have semi-GT data, calculate group2 as semi_gt - group1
            if semi_gt_individuals:
                group2_individuals = semi_gt_individuals - group1_individuals
                if group2_individuals:
                    final_group2_title = ", ".join(sorted(group2_individuals))
                    print(f"Date {date}: Using semi-GT data - Group1: {group1_title}, Group2: {final_group2_title}")
                else:
                    print(f"Date {date}: Semi-GT found but no other group detected, keeping original group2")
            else:
                print(f"Date {date}: No semi-GT data found, using original group2: {other_group_title}")
        except Exception as e:
            print(f"Date {date}: Error loading semi-GT data: {e}, using original group2: {other_group_title}")
        
        group_info_csv = out_csv_dir / "group_info.csv"
        pd.DataFrame(
            [
                {
                    "date": str(date),
                    "group1": str(group1_title),
                    "group2": final_group2_title,
                    "group1_camera_ids": ",".join(str(int(c)) for c in sorted(set(cam_ids))),
                }
            ]
        ).to_csv(group_info_csv, index=False)
        group_info_by_date[str(date)] = {
            "group1": str(group1_title),
            "group2": final_group2_title,
            "group1_camera_ids": ",".join(str(int(c)) for c in sorted(set(cam_ids))),
        }
        print(f"Saved: {group_info_csv}")

        plot_path = None

        ethogram_path = out_fig_dir / f"{output_stem}_activity_ethogram_{args.bin_minutes}min.png"
        ethogram_title = f"{output_stem} Activity Ethogram {date} ({night_start:%H:%M}–{night_end:%H:%M})"
        
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
            label_root=out_root,
        )
        if traj_paths is not None:
            # print(f"Saved: {traj_paths[1]}")
            stereotypy_flags = traj_paths[2]
            stereotypy_debug = traj_paths[3]
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
                    if not stereotypy_flags.empty:
                        sf = stereotypy_flags.copy()
                        sf["start_timestamp"] = pd.to_datetime(sf["start_timestamp"], errors="coerce")
                        sf["end_timestamp"] = pd.to_datetime(sf["end_timestamp"], errors="coerce")
                        keep_mask = np.zeros(len(sf), dtype=bool)
                        for i, row in sf.iterrows():
                            s = row["start_timestamp"]
                            e = row["end_timestamp"]
                            if pd.isna(s) or pd.isna(e):
                                continue
                            overlaps = (
                                (final_stereotypy_windows["start_time"] < e)
                                & (final_stereotypy_windows["end_time"] > s)
                            )
                            keep_mask[i] = bool(overlaps.any())
                        stereotypy_flags = sf[keep_mask].copy()
                else:
                    stereotypy_flags = pd.DataFrame(columns=stereotypy_flags.columns if not stereotypy_flags.empty else [])
            if not stereotypy_flags.empty:
                stereotypy_csv = out_csv_dir / f"{output_stem}_stereotypy_flags.csv"
                stereotypy_flags.to_csv(stereotypy_csv, index=False)
                print(f"Saved: {stereotypy_csv}")
                flags_all = stereotypy_flags.copy()
                flags_all["individual"] = args.individual_group
                all_stereotypy_flags.append(flags_all)
            heat_path = traj_paths[0]
            plot_world_heatmap_by_behaviour_after_stereotypy(
                df_traj=_build_identity_trajectory_points(source_bouts, bouts),
                out_path=heat_path,
                title=heat_path.name.replace(".png", ""),
                night_start=night_start,
                night_end=night_end,
                df_stereotypy_windows=final_stereotypy_windows,
            )
            if heat_path.exists():
                print(f"Saved: {heat_path}")

        
        budget = _summarize_labels(labels)
        budget["duration_min"] = budget["duration_sec"] / 60.0
        budget["duration_hr"] = budget["duration_sec"] / 3600.0
        budget["percent"] = budget["duration_sec"] / budget["duration_sec"].sum() * 100.0
        budget["night_start"] = night_start
        budget["night_end"] = night_end
        budget["date"] = date
        all_budgets.append(budget)

        save_csvs = True ### TODO
        if save_csvs:
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
        ethogram_csv = out_csv_dir / f"{output_stem}_ethogram.csv"
        save_ethogram_csv(
            identity_id=args.individual_group,
            segments=segments,
            out_csv=ethogram_csv,
        )
        print(f"Saved: {ethogram_csv}")
        pie_plot = out_fig_dir / f"{output_stem}_activity_budget_pie.png"
        budget_from_ethogram = analyze_ethogram_and_plot_activity_budget(
            ethogram_csv=ethogram_csv,
            out_plot=pie_plot,
            date=date,
            camera_ids=cam_ids,
            other_group=other_group_title,
            label_display_map=LABEL_DISPLAY,
            label_color_map=LABEL_COLORS,
        )
        print(f"Saved: {pie_plot}")
        pie_budget_csv = out_csv_dir / f"{output_stem}_activity_budget_from_ethogram.csv"
        budget_from_ethogram.to_csv(pie_budget_csv, index=False)
        print(f"Saved: {pie_budget_csv}")
        segments["date"] = date
        segments["offset_start_sec"] = (segments["start_time"] - night_start).dt.total_seconds()
        segments["offset_end_sec"] = (segments["end_time"] - night_start).dt.total_seconds()
        # Keep y-axis dates in the all-night plot only for nights with actual bouts.
        if not bouts.empty:
            all_night_segments.append(segments)
        _plot_activity_timeline(
            segments,
            ethogram_path,
            ethogram_title,
            df_gt_segments=gt_segments,
            df_stereotypy_segments=final_stereotypy_windows,
        )
        print(f"Saved: {ethogram_path}")
        if gt_created:
            print(f"Created empty GT template: {gt_csv}")

    if all_night_segments or all_night_gt_segments:
        all_segments = pd.concat(all_night_segments, ignore_index=True) if all_night_segments else pd.DataFrame()
        if not all_segments.empty:
            all_segments_csv = out_root / f"{args.individual_group}_activity_segments_all_nights.csv"
            all_segments.to_csv(all_segments_csv, index=False)
            print(f"Saved: {all_segments_csv}")

            budget_wide = (
                all_segments
                .groupby(["date", "behavior_label"], as_index=False)["duration_sec"]
                .sum()
                .pivot(index="date", columns="behavior_label", values="duration_sec")
                .fillna(0.0)
            )
            for label in LABEL_ORDER:
                if label not in budget_wide.columns:
                    budget_wide[label] = 0.0
            budget_wide = budget_wide.reset_index()
            for label in LABEL_ORDER:
                budget_wide.rename(columns={label: f"{label}_duration_sec"}, inplace=True)
            duration_cols = [f"{label}_duration_sec" for label in LABEL_ORDER]
            budget_wide["total_duration"] = budget_wide[duration_cols].sum(axis=1)
            budget_wide["other_group"] = budget_wide["date"].astype(str).map(
                lambda d: str(group_info_by_date.get(d, {}).get("group2", "unknown"))
            )
            budget_wide["camera_info"] = budget_wide["date"].astype(str).map(
                lambda d: str(group_info_by_date.get(d, {}).get("group1_camera_ids", "unknown"))
            )
            ordered_cols = ["date", *duration_cols, "other_group", "camera_info", "total_duration"]
            budget_wide = budget_wide[ordered_cols].sort_values("date").reset_index(drop=True)
            full_budget_csv = out_root / f"{args.individual_group}_activity_budget_all_nights_full.csv"
            budget_wide.to_csv(full_budget_csv, index=False)
            print(f"Saved: {full_budget_csv}")

        all_gt_segments = pd.concat(all_night_gt_segments, ignore_index=True) if all_night_gt_segments else pd.DataFrame()

        def _dates_for_room(room_ids: set[int]) -> set[str]:
            dates_for_room: set[str] = set()
            for d_txt, info in group_info_by_date.items():
                raw_cam_txt = str(info.get("group1_camera_ids", ""))
                cams: set[int] = set()
                for token in raw_cam_txt.split(","):
                    token = token.strip()
                    if token.isdigit():
                        cams.add(int(token))
                if cams & room_ids:
                    dates_for_room.add(str(d_txt))
            return dates_for_room

        room_defs = [
            ("room1_w_o_pool", "Room1 (w/o pool)", {16, 19}),
            ("room2_w_pool", "Room2 (w. pool)", {17, 18}),
        ]
        for room_tag, room_title, room_ids in room_defs:
            room_dates = _dates_for_room(room_ids)
            if not room_dates:
                continue
            room_segments = all_segments[all_segments["date"].astype(str).isin(room_dates)].copy()
            room_gt = (
                all_gt_segments[all_gt_segments["date"].astype(str).isin(room_dates)].copy()
                if not all_gt_segments.empty
                else pd.DataFrame()
            )
            if room_segments.empty and room_gt.empty:
                continue
            all_ethogram_path = out_root / f"{args.individual_group}_activity_ethogram_all_nights_{room_tag}.png"
            all_ethogram_title = f"{args.individual_group} Activity Ethogram Across Nights - {room_title}"
            _plot_activity_timeline_multi_night(
                room_segments,
                all_ethogram_path,
                all_ethogram_title,
                df_gt_segments_all=room_gt,
                group_info_by_date=group_info_by_date,
            )
            print(f"Saved: {all_ethogram_path}")

        # if not all_segments.empty:
        #     all_ethogram_pie_path = out_root / f"{args.individual_group}_activity_ethogram_all_nights_with_pies.png"
        #     all_ethogram_pie_title = f"{args.individual_group} Activity Ethogram + Budget Across Nights"
        #     _plot_activity_timeline_with_pie_multi_night(
        #         all_segments,
        #         all_ethogram_pie_path,
        #         all_ethogram_pie_title,
        #         group_info_by_date=group_info_by_date,
        #     )
        #     print(f"Saved: {all_ethogram_pie_path}")

    if all_stereotypy_flags:
        merged_stereotypy = pd.concat(all_stereotypy_flags, ignore_index=True)
        merged_stereotypy = merged_stereotypy.sort_values(["date", "start_timestamp"]).reset_index(drop=True)
        merged_stereotypy.to_csv(merged_stereotypy_csv, index=False)
        print(f"Saved: {merged_stereotypy_csv}")
        if args.extract_video:
            manifest_csv = export_stereotypy_event_videos_from_csv(
                stereotypy_csv=merged_stereotypy_csv,
                out_root=out_root,
                individual_group=args.individual_group,
            )
            if manifest_csv is not None:
                print(f"Saved: {manifest_csv}")


if __name__ == "__main__":
    run_analysis(parse_args())
