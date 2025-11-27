from __future__ import annotations

import pandas as pd


def _track_column(tracks: pd.DataFrame) -> str:
    return "fixed_track_id" if "fixed_track_id" in tracks.columns else "canonical_track_id"


def _frame_seconds(meta: dict) -> float:
    fps = meta.get("fps")
    try:
        fps_val = float(fps)
        if fps_val > 0:
            return 1.0 / fps_val
    except Exception:
        pass
    return 1.0 / 30.0


def summarize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Return per-track statistics suitable for display in Streamlit."""
    if tracks.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "frames",
                "detections",
                "first_frame",
                "last_frame",
                "mean_score",
            ]
        )

    track_col = _track_column(tracks)
    summary = (
        tracks.groupby(track_col)
        .agg(
            frames=("frame_idx", "nunique"),
            detections=("frame_idx", "count"),
            first_frame=("frame_idx", "min"),
            last_frame=("frame_idx", "max"),
            mean_score=("score", "mean"),
        )
        .reset_index()
        .rename(columns={track_col: "track_id"})
    )
    summary.sort_values("frames", ascending=False, inplace=True)
    return summary


def behavior_time_by_track(tracks: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Aggregate time spent per behavior per track."""
    if tracks.empty or "behavior_label" not in tracks.columns:
        return pd.DataFrame(columns=["track_id", "behavior", "frames", "seconds"])
    track_col = _track_column(tracks)
    frame_sec = _frame_seconds(meta)
    subset = tracks.dropna(subset=["behavior_label"])
    if subset.empty:
        return pd.DataFrame(columns=["track_id", "behavior", "frames", "seconds"])
    agg = (
        subset.groupby([track_col, "behavior_label"])
        .agg(frames=("frame_idx", "count"))
        .reset_index()
        .rename(columns={track_col: "track_id", "behavior_label": "behavior"})
    )
    agg["seconds"] = agg["frames"] * frame_sec
    agg.sort_values(["track_id", "seconds"], ascending=[True, False], inplace=True)
    return agg


def behavior_heatmap(tracks: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Pivot to track_id x behavior with seconds as values."""
    agg = behavior_time_by_track(tracks, meta)
    if agg.empty:
        return agg
    pivot = agg.pivot_table(index="track_id", columns="behavior", values="seconds", fill_value=0.0)
    return pivot.reset_index()


def behavior_overall(tracks: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Overall behavior duration across all tracks."""
    agg = behavior_time_by_track(tracks, meta)
    if agg.empty:
        return agg
    overall = (
        agg.groupby("behavior")
        .agg(seconds=("seconds", "sum"), frames=("frames", "sum"))
        .reset_index()
        .sort_values("seconds", ascending=False)
    )
    return overall


def class_breakdown(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame(columns=["cls_name", "detections"])
    col = "cls_name" if "cls_name" in tracks.columns else "cls_id"
    summary = (
        tracks.groupby(col)
        .agg(detections=("frame_idx", "count"))
        .reset_index()
        .rename(columns={col: "cls_name"})
    )
    summary.sort_values("detections", ascending=False, inplace=True)
    return summary
