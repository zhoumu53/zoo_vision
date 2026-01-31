import pandas as pd
import numpy as np



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


def behavior_label_smooth(
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
    

if __name__ == "__main__":
    pass