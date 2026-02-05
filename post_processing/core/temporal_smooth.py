from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Tuple

import numpy as np
import pandas as pd

def _parse_timestamps(series: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    """
    Fast, consistent timestamp parsing.
    If fmt is None, falls back to pandas/dateutil but still coerces errors to NaT.
    """
    if fmt:
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")

@dataclass
class SmoothParams:
    # columns
    time_col: str = "timestamp"
    beh_col: str = "behavior_label"
    out_col: str = "behavior_label_smooth"

    # label sets
    invalid_label: str = "00_invalid"
    sleep_labels: Tuple[str, ...] = ("02_sleeping_left", "03_sleeping_right")

    # time handling
    max_gap: str = "3s"  # if time gap > max_gap, treat as a break (new block)

    # A) Base smoothing: time-based rolling majority vote
    # Use a fairly wide window to reflect "animals behave slow"
    rolling_window: str = "20s"
    rolling_min_periods: int = 5
    rolling_ignore_invalid: bool = True
    rolling_tie_break: str = "center"  # "center" | "keep"

    # B) Segment cleanup: remove tiny segments that are likely jitter
    # If a segment is shorter than this and neighbors agree, merge into neighbors.
    min_segment_duration: str = "2.0s"  # set higher if your FPS is high and jitter is frequent
    merge_if_neighbors_same: bool = True
    merge_short_to: str = "neighbor"  # "neighbor" | "invalid"

    # C) Simple sleeping check (your requested logic)
    enable_simple_sleep_check: bool = True
    simple_check_window: str = "5min"
    simple_check_mainly_threshold: float = 0.5  # majority
    simple_check_min_segment_s: float = 0.0     # act on all sleep segments by default; set >0 to only act on short bursts
    simple_check_ignore_invalid: bool = True
    simple_check_replace_to: str = "dominant"   # "dominant" | "invalid"

    # D) Optional: treat short invalid spikes similarly
    enable_invalid_spike_fix: bool = True
    invalid_spike_max_duration: str = "1.0s"


# -----------------------------
# Helpers
# -----------------------------

def _as_timedelta(x: str) -> pd.Timedelta:
    return pd.to_timedelta(x)


def _is_sleep(label: str, sleep_set: set[str]) -> bool:
    return label in sleep_set


def _dominant_label(
    labels: Iterable[str],
    invalid_label: str,
    ignore_invalid: bool = True,
) -> Optional[str]:
    s = pd.Series(list(labels), dtype="string")
    if ignore_invalid:
        s = s[s != invalid_label]
    if s.empty:
        return None
    return str(s.value_counts().idxmax())


def _split_blocks_by_gap(t: pd.Series, max_gap: pd.Timedelta) -> np.ndarray:
    """
    Returns block_id for each row; new block starts when time gap > max_gap.
    """
    dt = t.diff()
    new_block = (dt.isna()) | (dt > max_gap)
    return new_block.cumsum().to_numpy(dtype=np.int64)


def _segments_from_labels(labels: np.ndarray, block_id: np.ndarray) -> list[tuple[int, int, str, int]]:
    """
    Build segments as (start_idx, end_idx_exclusive, label, block_id_value)
    """
    segs: list[tuple[int, int, str, int]] = []
    n = len(labels)
    if n == 0:
        return segs

    s = 0
    cur_lab = labels[0]
    cur_blk = block_id[0]
    for i in range(1, n):
        if block_id[i] != cur_blk or labels[i] != cur_lab:
            segs.append((s, i, str(cur_lab), int(cur_blk)))
            s = i
            cur_lab = labels[i]
            cur_blk = block_id[i]
    segs.append((s, n, str(cur_lab), int(cur_blk)))
    return segs


def _segment_durations_seconds(t: pd.Series, segs: list[tuple[int, int, str, int]]) -> np.ndarray:
    """
    Approximate segment duration from timestamps: t[end-1] - t[start] (seconds).
    If segment has 1 row, duration 0.
    """
    out = np.zeros(len(segs), dtype=np.float64)
    tt = t.to_numpy()
    for k, (s, e, _, _) in enumerate(segs):
        if e - s <= 1:
            out[k] = 0.0
        else:
            out[k] = (tt[e - 1] - tt[s]).astype("timedelta64[ns]").astype(np.int64) / 1e9
    return out
def _rolling_majority_vote_timebased(
    df: pd.DataFrame,
    time_col: str,
    label_col: str,
    out_col: str,
    window: pd.Timedelta,
    min_periods: int,
    invalid_label: str,
    ignore_invalid: bool,
    tie_break: str,
) -> pd.DataFrame:
    """
    Time-based rolling majority vote for string labels.

    Implementation:
      - Sort by time.
      - For each row i, consider labels in [t[i]-window/2, t[i]+window/2].
      - If count < min_periods: keep original label.
      - Else: output majority label (optionally ignoring invalid).
    """
    d = df.copy()
    d = d.sort_values(time_col).reset_index(drop=True)

    t = d[time_col].to_numpy()
    labels = d[label_col].astype(str).to_numpy()

    if len(d) == 0:
        d[out_col] = []
        return d

    half = window / 2
    # Build symmetric bounds around each timestamp
    t_left = t - half.to_timedelta64()
    t_right = t + half.to_timedelta64()

    # Since t is sorted, use searchsorted to get window indices
    i_left = np.searchsorted(t, t_left, side="left")
    i_right = np.searchsorted(t, t_right, side="right")

    def roll_mode(arr: np.ndarray, center_label: str) -> str:
        s = pd.Series(arr, dtype="string")
        if ignore_invalid:
            s = s[s != invalid_label]
        if s.empty:
            return invalid_label
        vc = s.value_counts()
        top = vc.max()
        winners = vc[vc == top].index.astype(str).tolist()
        if len(winners) == 1:
            return winners[0]

        if tie_break == "keep" and center_label in winners:
            return center_label
        # deterministic tie break
        return sorted(winners)[0]

    out = np.empty(len(d), dtype=object)
    for i in range(len(d)):
        a = labels[i_left[i]:i_right[i]]
        if len(a) < min_periods:
            out[i] = labels[i]
        else:
            out[i] = roll_mode(a, center_label=labels[i])

    d[out_col] = out
    return d


def _fix_short_segments_by_neighbors(
    t: pd.Series,
    labels: np.ndarray,
    block_id: np.ndarray,
    min_seg_s: float,
    invalid_label: str,
    merge_if_neighbors_same: bool,
    merge_short_to: str,
) -> np.ndarray:
    """
    If a segment duration < min_seg_s, and neighbors (within same block) agree, overwrite it.
    """
    out = labels.copy()
    segs = _segments_from_labels(out, block_id)
    seg_dur = _segment_durations_seconds(t, segs)

    for k, (s, e, lab, blk) in enumerate(segs):
        if seg_dur[k] >= min_seg_s:
            continue

        # find prev/next segments in same block
        prev_k = k - 1
        next_k = k + 1
        if prev_k < 0 or next_k >= len(segs):
            continue
        ps, pe, plab, pblk = segs[prev_k]
        ns, ne, nlab, nblk = segs[next_k]
        if pblk != blk or nblk != blk:
            continue

        if merge_if_neighbors_same and plab == nlab:
            out[s:e] = plab
        else:
            if merge_short_to == "invalid":
                out[s:e] = invalid_label
            # else: keep as-is

    return out


def _window_stats_around_time(
    t: pd.Series,
    labels: np.ndarray,
    center: pd.Timestamp,
    window: pd.Timedelta,
    sleep_set: set[str],
    invalid_label: str,
    ignore_invalid: bool,
    mainly_threshold: float,
) -> tuple[bool, float, Optional[str]]:
    """
    Returns:
      is_mainly_sleeping, sleep_fraction, dominant_non_sleep_label
    """
    left = center - window
    right = center + window
    m = (t >= left) & (t <= right)
    if not m.any():
        return False, 0.0, None

    w = pd.Series(labels[m.to_numpy()], dtype="string")
    if ignore_invalid:
        w = w[w != invalid_label]
        if w.empty:
            return False, 0.0, None

    sleep_mask = w.isin(list(sleep_set))
    sleep_frac = float(sleep_mask.mean()) if len(w) else 0.0
    is_mainly_sleep = sleep_frac >= mainly_threshold

    non_sleep = w[~sleep_mask]
    dom_non_sleep = str(non_sleep.value_counts().idxmax()) if not non_sleep.empty else None
    return is_mainly_sleep, sleep_frac, dom_non_sleep


def _simple_sleep_spike_rule(
    df: pd.DataFrame,
    t: pd.Series,
    labels: np.ndarray,
    block_id: np.ndarray,
    sleep_set: set[str],
    params: SmoothParams,
) -> np.ndarray:
    """
    For each sleeping segment:
      - look at previous window (centered at segment start) and next window (centered at segment end)
      - if BOTH sides are NOT mainly sleeping -> replace sleeping in that segment
    """
    out = labels.copy()
    segs = _segments_from_labels(out, block_id)
    seg_dur = _segment_durations_seconds(t, segs)

    win = _as_timedelta(params.simple_check_window)

    for k, (s, e, lab, blk) in enumerate(segs):
        if not _is_sleep(lab, sleep_set):
            continue

        if params.simple_check_min_segment_s > 0 and seg_dur[k] >= params.simple_check_min_segment_s:
            # only act on short sleep bursts if requested
            continue

        # Use start and end times as anchors
        t_start = t.iloc[s]
        t_end = t.iloc[e - 1]

        left_main, _, left_dom = _window_stats_around_time(
            t=t, labels=out, center=t_start, window=win,
            sleep_set=sleep_set, invalid_label=params.invalid_label,
            ignore_invalid=params.simple_check_ignore_invalid,
            mainly_threshold=params.simple_check_mainly_threshold,
        )
        right_main, _, right_dom = _window_stats_around_time(
            t=t, labels=out, center=t_end, window=win,
            sleep_set=sleep_set, invalid_label=params.invalid_label,
            ignore_invalid=params.simple_check_ignore_invalid,
            mainly_threshold=params.simple_check_mainly_threshold,
        )

        # If BOTH sides are not mainly sleeping, treat this sleep segment as wrong
        if (not left_main) and (not right_main):
            if params.simple_check_replace_to == "invalid":
                out[s:e] = params.invalid_label
            else:
                # choose a replacement:
                # prefer the dominant label that exists; if both exist and differ, use the global dominant from (left+right)
                cands = [x for x in [left_dom, right_dom] if x is not None]
                if len(cands) == 0:
                    out[s:e] = params.invalid_label
                elif len(cands) == 1:
                    out[s:e] = cands[0]
                else:
                    # compute dominant over combined side windows
                    # (approximate by using their labels; stable + deterministic)
                    rep = _dominant_label(cands, invalid_label=params.invalid_label, ignore_invalid=False)
                    out[s:e] = rep if rep is not None else params.invalid_label

    return out


def _fix_short_invalid_spikes(
    t: pd.Series,
    labels: np.ndarray,
    block_id: np.ndarray,
    params: SmoothParams,
) -> np.ndarray:
    """
    If invalid segment is very short and neighbors (same block) agree, fill it.
    """
    out = labels.copy()
    segs = _segments_from_labels(out, block_id)
    seg_dur = _segment_durations_seconds(t, segs)

    max_s = _as_timedelta(params.invalid_spike_max_duration).total_seconds()

    for k, (s, e, lab, blk) in enumerate(segs):
        if lab != params.invalid_label:
            continue
        if seg_dur[k] > max_s:
            continue

        prev_k = k - 1
        next_k = k + 1
        if prev_k < 0 or next_k >= len(segs):
            continue
        _, _, plab, pblk = segs[prev_k]
        _, _, nlab, nblk = segs[next_k]
        if pblk != blk or nblk != blk:
            continue
        if plab == nlab and plab != params.invalid_label:
            out[s:e] = plab

    return out


# -----------------------------
# Public API
# -----------------------------

def suppress_isolated_sleeping(
    df: pd.DataFrame,
    in_col: str,
    time_col: str = "timestamp",
    out_col: str = "behavior_label_no_isolated_sleep",
    sleeping_labels: Tuple[str, ...] = ("02_sleeping_left", "03_sleeping_right"),
    lookback_window: str = "5min",
    lookahead_window: str = "5min",
    sleeping_threshold: float = 0.3,  # if < 30% sleeping in BOTH directions, suppress
    replace_with: str = "dominant",  # "dominant", "prev", or specific label like "01_standing"
    invalid_label: str = "00_invalid",
) -> pd.DataFrame:
    """
    Simple isolated sleeping suppression:
    For each sleeping frame, check ±5 min windows.
    If BOTH previous and next windows have low sleeping fraction, it's isolated -> suppress it.
    
    Args:
        sleeping_threshold: If sleeping fraction < this in BOTH prev AND next window, suppress
        replace_with: 
            - "dominant": most common non-sleeping label in combined ±5min window
            - "prev": previous non-sleeping label
            - specific label like "01_standing"
    """
    if df.empty:
        out = df.copy()
        out[out_col] = []
        return out
    
    out = df.copy()
    t = _ensure_datetime(out, time_col)
    out["_t"] = t
    out = out.sort_values("_t").reset_index(drop=True)
    
    s = out.set_index("_t")
    labels = s[in_col].astype(str)
    
    sleep_set = set(sleeping_labels)
    is_sleep = labels.isin(list(sleep_set))
    
    # Calculate sleeping fraction in lookback and lookahead windows (NOT centered)
    # Lookback: from (t - lookback_window) to t
    prev_sleep_frac = is_sleep.astype(float).rolling(
        lookback_window, closed='left', min_periods=1
    ).mean()
    
    # Lookahead: from t to (t + lookahead_window)
    # Reverse the series, roll, then reverse back
    reversed_sleep = is_sleep[::-1].astype(float)
    next_sleep_frac_reversed = reversed_sleep.rolling(
        lookahead_window, closed='left', min_periods=1
    ).mean()
    next_sleep_frac = next_sleep_frac_reversed[::-1]
    
    final = labels.to_numpy(dtype=object)
    
    for i in range(len(final)):
        if final[i] not in sleep_set:
            continue
        
        # Check if isolated: BOTH prev and next windows have low sleeping
        prev_frac = float(prev_sleep_frac.iloc[i]) if not pd.isna(prev_sleep_frac.iloc[i]) else 1.0
        next_frac = float(next_sleep_frac.iloc[i]) if not pd.isna(next_sleep_frac.iloc[i]) else 1.0
        
        is_isolated = (prev_frac < sleeping_threshold) and (next_frac < sleeping_threshold)
        
        if not is_isolated:
            continue
        
        # Suppress this isolated sleeping frame
        if replace_with == "prev":
            # Find previous non-sleeping label
            j = i - 1
            while j >= 0 and (final[j] in sleep_set):
                j -= 1
            final[i] = final[j] if j >= 0 else invalid_label
        elif replace_with == "dominant":
            # Find dominant non-sleeping label in combined ±window
            try:
                span_back = pd.Timedelta(lookback_window)
                span_ahead = pd.Timedelta(lookahead_window)
            except Exception:
                span_back = pd.Timedelta("5min")
                span_ahead = pd.Timedelta("5min")
            
            left = s.index[i] - span_back
            right = s.index[i] + span_ahead
            window_labels = labels.loc[left:right]
            dom = _dominant_label(window_labels, exclude=sleep_set)
            final[i] = dom if dom else invalid_label
        else:
            # Use specific label
            final[i] = replace_with
    
    out[out_col] = final
    out = out.drop(columns=["_t"])
    return out


def behavior_label_smooth(
    df: pd.DataFrame,
    beh_col: str = "behavior_label",
    time_col: str = "timestamp",
    out_col: str = "behavior_label_smooth",
    params: Optional[SmoothParams] = None,
) -> pd.DataFrame:
    """
    Smooth behavior labels in df.

    Notes
    -----
    - Does not rely on behavior_conf (you said it can be high even when wrong).
    - Works best when df represents a single continuous tracklet/identity/camera stream.
    - If multiple tracks are mixed, call this per-group (e.g., groupby track id) and concat.

    Returns
    -------
    df_out : DataFrame
      Copy of input df with `out_col` added.
    """
    if params is None:
        params = SmoothParams(time_col=time_col, beh_col=beh_col, out_col=out_col)
    else:
        # allow overriding via arguments
        params = SmoothParams(**{**params.__dict__, "time_col": time_col, "beh_col": beh_col, "out_col": out_col})

    if df is None or len(df) == 0:
        d0 = df.copy()
        d0[out_col] = []
        return d0

    d = df.copy()

    # Ensure timestamp dtype
    d[params.time_col] = _parse_timestamps(d[params.time_col], fmt=None)
    d = d.dropna(subset=[params.time_col]).sort_values(params.time_col).reset_index(drop=True)

    # Ensure behavior labels are strings
    d[params.beh_col] = d[params.beh_col].astype(str)

    # block by time gaps (prevents smoothing across discontinuities)
    max_gap_td = _as_timedelta(params.max_gap)
    t = d[params.time_col]
    block_id = _split_blocks_by_gap(t, max_gap_td)

    # A) Rolling majority vote (time-based)
    roll_td = _as_timedelta(params.rolling_window)
    dA = _rolling_majority_vote_timebased(
        df=d,
        time_col=params.time_col,
        label_col=params.beh_col,
        out_col=params.out_col,
        window=roll_td,
        min_periods=params.rolling_min_periods,
        invalid_label=params.invalid_label,
        ignore_invalid=params.rolling_ignore_invalid,
        tie_break=params.rolling_tie_break,
    )
    labels = dA[params.out_col].astype(str).to_numpy()

    # B) Merge short jitter segments
    min_seg_s = _as_timedelta(params.min_segment_duration).total_seconds()
    labels = _fix_short_segments_by_neighbors(
        t=t,
        labels=labels,
        block_id=block_id,
        min_seg_s=min_seg_s,
        invalid_label=params.invalid_label,
        merge_if_neighbors_same=params.merge_if_neighbors_same,
        merge_short_to=params.merge_short_to,
    )

    # C) Simple sleep spike rule (±5min by default)
    sleep_set = set(params.sleep_labels)
    if params.enable_simple_sleep_check:
        labels = _simple_sleep_spike_rule(
            df=dA,
            t=t,
            labels=labels,
            block_id=block_id,
            sleep_set=sleep_set,
            params=params,
        )

    # D) Optional: fill very short invalid spikes
    if params.enable_invalid_spike_fix:
        labels = _fix_short_invalid_spikes(
            t=t,
            labels=labels,
            block_id=block_id,
            params=params,
        )

    dA[params.out_col] = labels
    return dA


def smooth_behavior_cross_cameras(
    df: pd.DataFrame,
    id_col: str = "identity_label",
    cam_col: str = "camera_id",
    time_col: str = "timestamp",
    beh_col: str = "behavior_label",
    conf_col: str | None = "behavior_conf",
    standing_label: str = "01_standing",
    timestamp_resolution: str = "s", 
    only_if_multiple_cams: bool = True,
    apply_inplace: bool = False,
) -> pd.DataFrame:
    """
    Cross-camera smoothing rule (standing-dominance):
      For each (ID, timestamp_bucket), if ANY camera predicts standing_label,
      then set ALL non-standing behaviors at that same bucket to standing_label.

    Returns a copy unless apply_inplace=True.
    """
    if not apply_inplace:
        d = df.copy()
    else:
        d = df

    # Parse timestamp + bucket to second (or chosen resolution)
    ts = pd.to_datetime(d[time_col], errors="coerce", utc=False)
    d["_ts"] = ts
    d = d.dropna(subset=["_ts"])

    d["_ts_bucket"] = d["_ts"].dt.floor(timestamp_resolution)

    # Determine which (ID, bucket) has standing in any camera
    g = d.groupby([id_col, "_ts_bucket"], dropna=False)

    # flag standing rows
    d["_is_standing"] = d[beh_col].astype(str) == str(standing_label)

    # standing exists in group?
    standing_any = g["_is_standing"].transform("any")

    if only_if_multiple_cams:
        n_cams = g[cam_col].transform(lambda x: x.astype(str).nunique())
        eligible = standing_any & (n_cams >= 2)
    else:
        eligible = standing_any

    to_fix = eligible & (~d["_is_standing"])

    d.loc[to_fix, beh_col] = standing_label

    d = d.drop(columns=["_ts", "_ts_bucket", "_is_standing"], errors="ignore")

    return d


def behavior_label_smooth_old(
    df: pd.DataFrame,
    beh_col: str = "behavior_label",
    conf_col: str = "behavior_conf",
    window_size: int = 21,
    min_conf_threshold: float = 0.7,
    outlier_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Smooth behavior labels by detecting and fixing outliers using a sliding window.
    
    Args:
        df: DataFrame with behavior predictions
        beh_col: Column name for behavior labels
        conf_col: Column name for behavior confidence
        window_size: Size of sliding window (should be odd number, e.g., 11 = look at ±5 frames)
        min_conf_threshold: Minimum confidence to consider a prediction valid
        outlier_threshold: If a label appears less than this fraction in window, it's an outlier
        
    Returns:
        DataFrame with smoothed behavior labels
    """
    if len(df) < window_size:
        return df
    
    if beh_col not in df.columns:
        return df
    
    df = df.copy()
    
    # Ensure window size is odd for symmetric neighborhood
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    
    # Convert confidence to float if it's string
    if conf_col in df.columns:
        df[conf_col] = pd.to_numeric(df[conf_col], errors='coerce')
    
    # Create a working copy of labels
    labels = df[beh_col].values.copy()
    confidences = df[conf_col].values if conf_col in df.columns else np.ones(len(df))
    
    # Track which indices need smoothing
    smoothed_indices = []
    
    # Iterate through each position
    for i in range(len(labels)):
        # Skip if confidence is already very high
        if confidences[i] >= 0.99:
            continue
        
        # Define window bounds
        start_idx = max(0, i - half_window)
        end_idx = min(len(labels), i + half_window + 1)
        
        # Get window (excluding current position for comparison)
        window_labels = np.concatenate([labels[start_idx:i], labels[i+1:end_idx]])
        
        if len(window_labels) == 0:
            continue
        
        # Count occurrences of each label in window
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        
        if len(unique_labels) == 0:
            continue
        
        # Get the most common label in the window
        most_common_label = unique_labels[np.argmax(counts)]
        most_common_count = np.max(counts)
        
        current_label = labels[i]
        
        # Check if current label is an outlier
        # Count how many times current label appears in window
        current_count = np.sum(window_labels == current_label)
        current_fraction = current_count / len(window_labels)
        
        # If current label appears less than threshold in window, it's likely an outlier
        # Also check if confidence is low or below minimum threshold
        is_outlier = (
            current_fraction < outlier_threshold and
            most_common_count >= 2 and
            current_label != most_common_label and
            confidences[i] < min_conf_threshold
        )
        
        if is_outlier:
            labels[i] = most_common_label
            smoothed_indices.append(i)
    
    # Update the dataframe
    df[beh_col] = labels
    
    # Optionally mark smoothed rows (for debugging)
    if len(smoothed_indices) > 0:
        print(f"Smoothed {len(smoothed_indices)} outlier predictions out of {len(df)} frames")
    
    return df

