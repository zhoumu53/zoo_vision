from __future__ import annotations

import pandas as pd


def _track_column(tracks: pd.DataFrame) -> str:
    return "fixed_track_id" if "fixed_track_id" in tracks.columns else "canonical_track_id"


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
