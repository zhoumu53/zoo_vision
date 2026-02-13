import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


VALID_LABELS = {
    "01_standing",
    "02_sleeping_left",
    "03_sleeping_right",
    "walking",
    "stereotypy",
}

LABEL_ORDER = [
    "01_standing",
    "02_sleeping_left",
    "03_sleeping_right",
    "walking",
    "stereotypy",
    "outside",
]

LABEL_DISPLAY = {
    "01_standing": "standing",
    "02_sleeping_left": "sleeping left",
    "03_sleeping_right": "sleeping right",
    "outside": "outside / no-detection",
    "walking": "walking",
    "stereotypy": "stereotypy",
}

LABEL_COLORS = {
    "02_sleeping_left": "#5FB13E",
    "03_sleeping_right": "#FF9A2A",
    "outside": "#F1F1F1",
    "stereotypy": "#FF2624",
    "walking": "#c372cb",
    "01_standing": "#8a00ac",
}

LABEL_PRIORITY = {
    "outside": 0,
    "02_sleeping_left": 1,
    "03_sleeping_right": 1,
    "01_standing": 2,
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
) -> list[tuple[str, Path]]:
    collected: list[tuple[str, Path]] = []
    for date in dates:
        date_dir = output_dir / date
        if not date_dir.exists():
            continue
        pattern = "*.csv" if not filename_keyword else f"*{filename_keyword}*.csv"
        for csv_path in sorted(date_dir.glob(pattern)):
            collected.append((date, csv_path))

    if not collected:
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

    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(mdates.date2num(df["start_time"].min()), mdates.date2num(df["end_time"].max()))
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    legend_handles = [
        Patch(facecolor=LABEL_COLORS[label], edgecolor="none", label=LABEL_DISPLAY[label])
        for label in LABEL_ORDER
    ]
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
) -> None:
    df = df_segments_all.copy()
    if df.empty:
        raise ValueError("No multi-night timeline segments to plot")

    dates_sorted = sorted(df["date"].astype(str).unique())
    date_to_y = {d: i for i, d in enumerate(dates_sorted)}
    y_labels = [pd.to_datetime(d, format="%Y%m%d").strftime("%Y-%m-%d") for d in dates_sorted]

    anchor = pd.Timestamp("2000-01-01 18:00:00")
    start_nums = mdates.date2num(anchor + pd.to_timedelta(df["offset_start_sec"], unit="s"))
    width_days = (df["offset_end_sec"] - df["offset_start_sec"]).to_numpy(dtype=float) / 86400.0

    fig_h = max(3.0, 0.5 * len(dates_sorted) + 1.2)
    fig, ax = plt.subplots(figsize=(13, fig_h))

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

    x0 = mdates.date2num(anchor)
    x1 = mdates.date2num(anchor + pd.Timedelta(seconds=float(df["offset_end_sec"].max())))
    ax.set_xlim(x0, x1)
    ax.set_title(title)
    ax.set_ylabel("Date")
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


def _load_bouts_for_date(csv_paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        label_col = _behavior_label_col(df)

        if "start_time" not in df.columns or "end_time" not in df.columns:
            raise ValueError(f"Missing start_time/end_time in {csv_path}")
        if "identity_label" not in df.columns:
            raise ValueError(f"Missing identity_label in {csv_path}")

        df = df.copy()
        df["behavior_label"] = df[label_col].astype(str)
        df = df[df["behavior_label"].isin(VALID_LABELS)]
        df = df[df["identity_label"].astype(str) == "Thai"]
        if df.empty:
            continue

        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
        df = df.dropna(subset=["start_time", "end_time"])
        if df.empty:
            continue

        frames.append(df[["start_time", "end_time", "behavior_label"]])

    if not frames:
        return pd.DataFrame(columns=["start_time", "end_time", "behavior_label"])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute activity budget per night from bout summary CSVs")
    parser.add_argument(
        "--record_root",
        type=str,
        default="/media/ElephantsWD/elephants/test_dan/results",
        help="Root directory of the record",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional direct path to night_bout_summary directory. Overrides --record_root.",
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (Path(args.record_root) / "demo" / "night_bout_summary")
    track_dir = Path(args.record_root) / "tracks"
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    dates = [normalize_date(d) for d in args.dates] if args.dates else get_available_dates(output_dir)
    if not dates:
        raise FileNotFoundError(f"No date folders found in: {output_dir}")

    filename_keyword = args.individual_group if args.individual_group else None
    bout_csvs = get_bout_csvs(output_dir=output_dir, dates=dates, filename_keyword=filename_keyword)

    out_root = Path(args.out_root)

    by_date: dict[str, list[Path]] = {}
    for date, csv_path in bout_csvs:
        by_date.setdefault(date, []).append(csv_path)

    all_night_segments: list[pd.DataFrame] = []

    for date, csv_paths in sorted(by_date.items()):
        bouts = _load_bouts_for_date(csv_paths)
        max_ts = None
        if not bouts.empty:
            max_ts = max(bouts["end_time"].max(), bouts["start_time"].max())

        night_start, night_end = _night_window(date, max_ts)
        labels = _build_timeline_labels(bouts, night_start, night_end)
        budget = _summarize_labels(labels)
        budget["duration_min"] = budget["duration_sec"] / 60.0
        budget["duration_hr"] = budget["duration_sec"] / 3600.0
        budget["percent"] = budget["duration_sec"] / budget["duration_sec"].sum() * 100.0
        budget["night_start"] = night_start
        budget["night_end"] = night_end
        budget["date"] = date

        out_dir = out_root / date
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{args.individual_group}_activity_budget.csv"
        budget.to_csv(out_csv, index=False)

        timeline = _aggregate_timeline(labels, night_start, args.bin_minutes)
        timeline["date"] = date
        timeline_csv = out_dir / f"{args.individual_group}_activity_timeline_{args.bin_minutes}min.csv"
        timeline.to_csv(timeline_csv, index=False)
        segments = _labels_to_segments(labels, night_start)
        segments["date"] = date
        segments["offset_start_sec"] = (segments["start_time"] - night_start).dt.total_seconds()
        segments["offset_end_sec"] = (segments["end_time"] - night_start).dt.total_seconds()
        all_night_segments.append(segments)

        plot_path = out_dir / f"{args.individual_group}_activity_budget.png"
        title = f"{args.individual_group} Activity Budget {date} ({night_start:%H:%M}–{night_end:%H:%M})"
        _plot_activity_budget(budget, plot_path, title)

        ethogram_path = out_dir / f"{args.individual_group}_activity_ethogram_{args.bin_minutes}min.png"
        ethogram_title = f"{args.individual_group} Activity Ethogram {date} ({night_start:%H:%M}–{night_end:%H:%M})"
        _plot_activity_timeline(segments, ethogram_path, ethogram_title)

        print(f"Saved: {out_csv}")
        print(f"Saved: {timeline_csv}")
        print(f"Saved: {plot_path}")
        print(f"Saved: {ethogram_path}")

    if all_night_segments:
        all_segments = pd.concat(all_night_segments, ignore_index=True)
        all_segments_csv = out_root / f"{args.individual_group}_activity_segments_all_nights.csv"
        all_segments.to_csv(all_segments_csv, index=False)

        all_ethogram_path = out_root / f"{args.individual_group}_activity_ethogram_all_nights.png"
        all_ethogram_title = f"{args.individual_group} Activity Ethogram Across Nights"
        _plot_activity_timeline_multi_night(all_segments, all_ethogram_path, all_ethogram_title)

        print(f"Saved: {all_segments_csv}")
        print(f"Saved: {all_ethogram_path}")
