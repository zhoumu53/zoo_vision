import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment
from datetime import timedelta, time
import re
from collections import defaultdict
from bisect import bisect_right


def parse_time_from_filename(filename):
    """
    Extract time from filename format: T180823_ID005991.csv -> 18:08:23
    
    Parameters
    ----------
    filename : str
        Filename in format THHMMSS_*.csv
    
    Returns
    -------
    datetime.time or None
    """
    match = re.match(r'T(\d{2})(\d{2})(\d{2})_', filename)
    if match:
        hour, minute, second = map(int, match.groups())
        try:
            return time(hour, minute, second)
        except ValueError:
            return None
    return None


def get_csv_files_by_time_window(directory, start_time, end_time):
    """
    Get CSV files within a time window based on filename timestamps.
    
    Parameters
    ----------
    directory : str
        Directory containing CSV files
    start_time : datetime.time
        Start of time window
    end_time : datetime.time
        End of time window
    
    Returns
    -------
    list
        List of filenames within the time window
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    files_in_window = []
    
    for csv_file in csv_files:
        file_time = parse_time_from_filename(csv_file)
        if file_time is None:
            continue
        
        # Handle time window crossing midnight
        if start_time <= end_time:
            if start_time <= file_time <= end_time:
                files_in_window.append(csv_file)
        else:  # Window crosses midnight
            if file_time >= start_time or file_time <= end_time:
                files_in_window.append(csv_file)
    
    return sorted(files_in_window)


def load_csv_files(directory, filenames):
    """
    Load multiple CSV files and combine them.
    
    Parameters
    ----------
    directory : str
        Directory containing CSV files
    filenames : list
        List of filenames to load
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe
    """
    if not filenames:
        return pd.DataFrame()
    
    dfs = []
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        try:
            df_temp = pd.read_csv(file_path)
            df_temp['filename'] = filename
            dfs.append(df_temp)
        except Exception as e:
            print(f"Warning: Failed to load {filename}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[df["identity_label"].notna()]
    
    return df


def match_identities_for_dataframe(
    cam1_df,
    cam2_df,
    time_window_seconds=0.5,
    distance_threshold=2.0,
    downsample_seconds=None
):
    """
    Match identities from cam1 to cam2 using Hungarian algorithm.
    
    Parameters
    ----------
    cam1_df : pd.DataFrame
        Camera 1 data (main ID source)
    cam2_df : pd.DataFrame
        Camera 2 data to update
    time_window_seconds : float
        Time matching window
    distance_threshold : float
        Distance threshold in meters
    
    Returns
    -------
    pd.DataFrame
        Updated cam2 with matched identities
    """
    if cam1_df.empty or cam2_df.empty:
        return cam2_df

    if downsample_seconds is not None and downsample_seconds > 0:
        cam1_df = cam1_df.copy()
        cam2_df = cam2_df.copy()
        cam1_df["timestamp"] = cam1_df["timestamp"].dt.floor(f"{int(downsample_seconds)}s")
        cam2_df["timestamp"] = cam2_df["timestamp"].dt.floor(f"{int(downsample_seconds)}s")
    
    cam2_updated = cam2_df.copy()
    cam2_updated['matched_identity'] = None
    cam2_updated['match_distance'] = np.nan
    
    cam2_timestamps = cam2_updated['timestamp'].unique()
    
    for cam2_time in cam2_timestamps:
        cam2_frame = cam2_updated[cam2_updated['timestamp'] == cam2_time]
        
        if len(cam2_frame) == 0:
            continue
        
        time_diff = (cam1_df['timestamp'] - cam2_time).abs()
        temporal_candidates = cam1_df[time_diff <= timedelta(seconds=time_window_seconds)]
        
        if len(temporal_candidates) == 0:
            continue
        
        # ### DEBUG -- PRINT OUT THE TEMPORAL CANDIDATES, WITH 1 UNIQUE ID and which is different from cam2_frame
        # ### double check those tracks with GTs -- we might label them wrong or stitch them wrong
        # unique_ids_cam2 = cam2_frame['identity_label'].unique()
        # unique_ids_candidates = temporal_candidates['identity_label'].unique()
        # if len(unique_ids_candidates) == 1 and unique_ids_candidates[0] not in unique_ids_cam2:
        #     print(f"DEBUG: Cam2 {unique_ids_cam2} has {len(cam2_frame)} detections.")
        #     print(f"DEBUG: Temporal candidates from Cam1 have unique ID: {unique_ids_candidates[0]}")
        #     # print(temporal_candidates)
        #     ## TODO - we need fix it with 'voted labels' from reid features??
        #     temporal_candidates['filename'] = temporal_candidates['filename'].astype(str)
        #     print ("DEBUG: Temporal candidates filenames:", temporal_candidates['filename'].unique())
        #     print("cam1 tracklets:", cam1_tracklets.keys())
        #     import sys; sys.exit(1)
        
        cam2_positions = cam2_frame[['world_x', 'world_y']].values
        cam1_positions = temporal_candidates[['world_x', 'world_y']].values
        
        distance_matrix = np.zeros((len(cam2_frame), len(temporal_candidates)))
        for i, cam2_pos in enumerate(cam2_positions):
            for j, cam1_pos in enumerate(cam1_positions):
                distance_matrix[i, j] = np.linalg.norm(cam2_pos - cam1_pos)
        
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        for cam2_idx, cam1_idx in zip(row_ind, col_ind):
            distance = distance_matrix[cam2_idx, cam1_idx]
            
            if distance <= distance_threshold:
                cam2_original_idx = cam2_frame.index[cam2_idx]
                matched_row = temporal_candidates.iloc[cam1_idx]
                
                cam2_updated.at[cam2_original_idx, 'matched_identity'] = matched_row['identity_label']
                cam2_updated.at[cam2_original_idx, 'match_distance'] = distance
    
    ## count mathched identities
    # get top matched identities
    top_identities = cam2_updated['matched_identity'].value_counts().head(10)
    # get top1 identity
    top1_identity = top_identities.index[0] if not top_identities.empty else None
    # print("Top matched identities:")
    # print(top_identities)
    
    ### TODO - Update identity labels
    # cam2_updated['matched_identity'] = cam2_updated['matched_identity'].astype(pd.StringDtype())
    
    cam2_updated['identity_label'] = cam2_updated['matched_identity'].fillna(cam2_updated['identity_label'])
    ### Remove temporary columns

    # count identity_label occurrences
    identity_counts = cam2_updated['matched_identity'].value_counts()
    # print("Identity counts after matching:")
    # print(identity_counts)
    
    return cam2_updated

def _flatten_tracklets(camera_data, use_voted_labels=False):
    tracklets = []
    for idlabel, items in (camera_data or {}).items():
        for t in items:
            t_copy = dict(t)
            if not t_copy.get("identity_label"):
                t_copy["identity_label"] = idlabel
            if use_voted_labels:
                t_copy["identity_label"] = t_copy["voted_track_label"]
            tracklets.append(t_copy)
    return tracklets


def _parse_timestamp(ts_value):
    if not ts_value:
        return None
    try:
        return pd.to_datetime(ts_value, errors="coerce")
    except Exception:
        return None


def _load_csv_cached(csv_path, cache):
    if not csv_path:
        return None
    if csv_path in cache:
        return cache[csv_path]
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found: {csv_path}")
        cache[csv_path] = None
        return None
    def _to_datetime_series(series):
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except TypeError:
            return pd.to_datetime(series, errors="coerce")
    try:
        df = pd.read_csv(csv_path)
        df["filename"] = os.path.basename(csv_path)
        df["timestamp"] = _to_datetime_series(df["timestamp"])
        df = df[df["identity_label"].notna()]
        df = df[df["timestamp"].notna()]
        cache[csv_path] = df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"Warning: Failed to load {csv_path}: {e}")
        cache[csv_path] = None
    return cache[csv_path]


def _build_cam1_index(cam1_tracklets):
    entries = []
    for t in cam1_tracklets:
        t_start = _parse_timestamp(t.get("start_timestamp"))
        t_end = _parse_timestamp(t.get("end_timestamp"))
        entries.append(
            {
                "start": t_start,
                "end": t_end,
                "csv_path": t.get("track_csv_path"),
            }
        )
    entries.sort(key=lambda x: x["start"] or pd.Timestamp.min)
    starts = [e["start"] or pd.Timestamp.min for e in entries]
    return entries, starts


def _select_cam1_paths(entries, starts, window_start, window_end):
    # Only scan entries that start before window_end.
    end_idx = bisect_right(starts, window_end)
    paths = []
    for e in entries[:end_idx]:
        if e["end"] is None or e["start"] is None:
            paths.append(e["csv_path"])
            continue
        if e["start"] <= window_end and e["end"] >= window_start:
            paths.append(e["csv_path"])
    return [p for p in paths if p]


def _resolve_id_conflicts(cam2_tracklets):
    """
    Resolve conflicts where the same ID is assigned to multiple tracks at the same time.
    
    Strategy:
    1. Find temporal overlaps for each identity_label
    2. For conflicting tracks, keep the one with best match quality
    3. Revert others to their original identity_label
    
    Parameters
    ----------
    cam2_tracklets : list
        List of tracklet dictionaries with identity_label and match metadata
    
    Returns
    -------
    list
        Updated tracklets with conflicts resolved
    """
    from collections import defaultdict
    
    # Group tracklets by identity_label
    identity_groups = defaultdict(list)
    for t in cam2_tracklets:
        idlabel = t.get("identity_label")
        if idlabel:
            identity_groups[idlabel].append(t)
    
    conflicts_resolved = 0
    
    for idlabel, tracklets in identity_groups.items():
        if len(tracklets) <= 1:
            continue
        
        # Parse timestamps for each tracklet
        tracklet_times = []
        for t in tracklets:
            start = _parse_timestamp(t.get("start_timestamp"))
            end = _parse_timestamp(t.get("end_timestamp"))
            if start and end:
                tracklet_times.append((t, start, end))
        
        if len(tracklet_times) <= 1:
            continue
        
        # Check for temporal overlaps
        for i in range(len(tracklet_times)):
            for j in range(i + 1, len(tracklet_times)):
                t1, start1, end1 = tracklet_times[i]
                t2, start2, end2 = tracklet_times[j]
                
                # Check if they overlap
                if start1 <= end2 and start2 <= end1:
                    # Conflict detected - resolve based on match quality
                    quality1 = t1.get("match_quality", 0)
                    quality2 = t2.get("match_quality", 0)
                    
                    # Keep the one with higher quality, revert the other
                    if quality1 >= quality2:
                        # Revert t2 to original ID
                        original_id = t2.get("original_identity_label", t2.get("identity_label"))
                        t2["identity_label"] = original_id
                        conflicts_resolved += 1
                    else:
                        # Revert t1 to original ID
                        original_id = t1.get("original_identity_label", t1.get("identity_label"))
                        t1["identity_label"] = original_id
                        conflicts_resolved += 1
    
    if conflicts_resolved > 0:
        print(f"Resolved {conflicts_resolved} ID conflicts due to temporal overlaps")
    
    return cam2_tracklets


def cross_camera_id_matching(
    cam1_data,
    cam2_data,
    window_hours=1,
    time_window_seconds=0.5,
    distance_threshold=2.0,
    print_summary=True,
    downsample_seconds=None,
    use_voted_labels=False
):
    """
    Cross-camera ID matching using stitched tracklet maps.

    Parameters
    ----------
    cam1_data : dict
        final_stitched_map for camera 1 (idlabel -> list[tracklet dict])
    cam2_data : dict
        final_stitched_map for camera 2 (idlabel -> list[tracklet dict])
    window_hours : float
        Time window (hours) around cam2 tracklet time to select cam1 tracklets
    time_window_seconds : float
        Temporal matching threshold (seconds)
    distance_threshold : float
        Distance matching threshold (meters)

    Returns
    -------
    dict
        Updated cam2 final_stitched_map with identity labels reassigned
    """
    

    from tqdm import tqdm
    
    cam1_tracklets = _flatten_tracklets(cam1_data, use_voted_labels=use_voted_labels)
    cam2_tracklets = _flatten_tracklets(cam2_data, use_voted_labels=use_voted_labels)
    
    ### filename2voted_id

    if not cam1_tracklets or not cam2_tracklets:
        return cam2_data

    csv_cache = {}
    stats = defaultdict(int)
    summary_rows = []

    cam1_entries, cam1_starts = _build_cam1_index(cam1_tracklets)

    for t in tqdm(cam2_tracklets, desc="Cross-Cam ID Matching"):
        cam2_csv = t.get("track_csv_path")
        cam2_df = _load_csv_cached(cam2_csv, csv_cache)
        if cam2_df is None or cam2_df.empty:
            continue

        # Store original identity for conflict resolution
        original_identity = t.get("identity_label", "unknown")
        t["original_identity_label"] = original_identity
        t["voted_track_label"] = t.get("voted_track_label", "unknown")

        cam2_start = _parse_timestamp(t.get("start_timestamp"))
        cam2_end = _parse_timestamp(t.get("end_timestamp"))
        if cam2_start is None:
            cam2_start = cam2_df["timestamp"].min()
        if cam2_end is None:
            cam2_end = cam2_df["timestamp"].max()

        window_start = cam2_start - timedelta(hours=window_hours)
        window_end = cam2_end + timedelta(hours=window_hours)

        cam1_paths = _select_cam1_paths(cam1_entries, cam1_starts, window_start, window_end)
        if not cam1_paths:
            continue

        cam1_dfs = []
        for p in cam1_paths:
            df = _load_csv_cached(p, csv_cache)
            if df is not None and not df.empty:
                cam1_dfs.append(df)

        if not cam1_dfs:
            continue

        cam1_df = pd.concat(cam1_dfs, ignore_index=True)
        cam1_df = cam1_df[
            (cam1_df["timestamp"] >= window_start)
            & (cam1_df["timestamp"] <= window_end)
        ]
        if cam1_df.empty:
            continue

        cam2_matched = match_identities_for_dataframe(
            cam1_df,
            cam2_df,
            time_window_seconds=time_window_seconds,
            distance_threshold=distance_threshold,
            downsample_seconds=downsample_seconds,
        )

        matched_identities = cam2_matched["matched_identity"].dropna()
        if len(matched_identities) > 0:
            identity_counts = matched_identities.value_counts()
            final_identity = identity_counts.idxmax()
            match_count = identity_counts.max()
            avg_distance = cam2_matched[cam2_matched["matched_identity"] == final_identity]["match_distance"].mean()
            
            # Calculate match quality score (higher is better)
            # Based on: frequency of matches and inverse of average distance
            match_quality = match_count / (1 + avg_distance)
            
            t["identity_label"] = final_identity
            t["match_quality"] = match_quality
            t["match_count"] = int(match_count)
            t["avg_match_distance"] = float(avg_distance)
            
            stats[f"{original_identity}->{final_identity}"] += 1
            summary_rows.append(
                {
                    "filename": os.path.basename(cam2_csv) if cam2_csv else "",
                    "original_identity": original_identity,
                    "matched_identity": final_identity,
                    "match_quality": match_quality,
                    "match_count": match_count,
                    "avg_distance": avg_distance,
                }
            )
        else:
            t["match_quality"] = 0
            stats[f"{original_identity}"] += 1
            summary_rows.append(
                {
                    "filename": os.path.basename(cam2_csv) if cam2_csv else "",
                    "original_identity": original_identity,
                    "matched_identity": None,
                    "match_quality": 0,
                    "match_count": 0,
                    "avg_distance": np.nan,
                }
            )
    
    # Resolve ID conflicts (same ID assigned to multiple tracks at same time)
    cam2_tracklets = _resolve_id_conflicts(cam2_tracklets)
    
    # save summary_rows to DataFrame
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("cross_cam_id_matching_summary.csv", index=False)

    updated_cam2_map = {}
    for t in cam2_tracklets:
        idlabel = t.get("identity_label", "unknown")
        updated_cam2_map.setdefault(idlabel, []).append(t)

    if print_summary:
        print("\n" + "=" * 60)
        print("MATCHING SUMMARY")
        print("=" * 60)
        if summary_rows:
            detailed_df = pd.DataFrame(summary_rows)
            print(detailed_df)
        if stats:
            summary_df = pd.DataFrame(
                [{"mapping": k, "count": v} for k, v in stats.items()]
            ).sort_values("count", ascending=False)
            print(summary_df)
        else:
            print("No matches found")

    return updated_cam2_map


if __name__ == "__main__":
    cam1_dir = "/Users/zhoumu/Downloads/vis/zag_elp_cam_016/2025-11-15"
    cam2_dir = "/Users/zhoumu/Downloads/vis/zag_elp_cam_019/2025-11-15"
    output_dir = "/Users/zhoumu/Downloads/vis/zag_elp_cam_019_matched/2025-11-15"
