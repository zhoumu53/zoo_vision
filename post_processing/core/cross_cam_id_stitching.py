import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment
from datetime import timedelta, time
import re
from collections import defaultdict


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
            if 'identity_label' not in df_temp.columns:
                continue
            if df_temp.empty:
                continue
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
    distance_threshold=2.0
):
    """
    Match identities from cam1 to cam2 using Hungarian algorithm.
    
    Parameters
    ----------
    cam1_df : pd.DataFrame
        Camera 1 data (ground truth)
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
    print("Top matched identities:")
    print(top_identities)
    
    ### Update identity labels
    # cam2_updated['matched_identity'] = cam2_updated['matched_identity'].astype(pd.StringDtype())
    
    cam2_updated['identity_label'] = cam2_updated['matched_identity'].fillna(cam2_updated['identity_label'])
    ### Remove temporary columns

    # count identity_label occurrences
    identity_counts = cam2_updated['matched_identity'].value_counts()
    print("Identity counts after matching:")
    print(identity_counts)
    
    return cam2_updated


def process_with_sliding_window(
    cam1_dir,
    cam2_dir,
    output_dir = None,
    start_time = "18:00:00",
    end_time = "08:00:00",
    window_hours=1,
    time_window_seconds=0.5,
    distance_threshold=2.0
):
    """
    Process cross-camera matching with sliding time windows.
    
    Parameters
    ----------
    cam1_dir : str
        Directory with cam1 CSV files (ground truth)
    cam2_dir : str
        Directory with cam2 CSV files to update
    output_dir : str or None
        Output directory for updated cam2 files (default: overwrite originals)
    window_hours : float
        Sliding window size in hours
    time_window_seconds : float
        Temporal matching threshold
    distance_threshold : float
        Distance matching threshold
    """
    if output_dir is None:
        output_dir = cam2_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all cam2 files
    cam2_files = sorted([f for f in os.listdir(cam2_dir) if f.endswith('.csv')])
    
    print(f"Found {len(cam2_files)} files in cam2")
    print(f"Using {window_hours}h sliding window for matching")
    print("=" * 60)
    
    stats = defaultdict(lambda: defaultdict(int))
    
    for i, cam2_file in enumerate(cam2_files, 1):
        file_time = parse_time_from_filename(cam2_file)
        if file_time is None:
            print(f"[{i}/{len(cam2_files)}] Skipping {cam2_file} - cannot parse time")
            continue
        
        # Calculate window bounds
        window_start_seconds = max(0, file_time.hour * 3600 + file_time.minute * 60 + file_time.second - window_hours * 3600)
        window_end_seconds = min(86399, file_time.hour * 3600 + file_time.minute * 60 + file_time.second + window_hours * 3600)
        
        start_time = time(window_start_seconds // 3600, (window_start_seconds % 3600) // 60, window_start_seconds % 60)
        end_time = time(window_end_seconds // 3600, (window_end_seconds % 3600) // 60, window_end_seconds % 60)
        
        # Get relevant cam1 files in window
        cam1_files_in_window = get_csv_files_by_time_window(cam1_dir, start_time, end_time)
        
        if not cam1_files_in_window:
            print(f"[{i}/{len(cam2_files)}] No cam1 files in window for {cam2_file}")
            continue
        
        # Load cam1 data for this window
        cam1_df = load_csv_files(cam1_dir, cam1_files_in_window)
        
        # Load cam2 file
        cam2_path = os.path.join(cam2_dir, cam2_file)
        try:
            cam2_df = pd.read_csv(cam2_path)
            cam2_df['timestamp'] = pd.to_datetime(cam2_df['timestamp'])
            cam2_df = cam2_df[cam2_df['identity_label'].notna()]
        except Exception as e:
            print(f"[{i}/{len(cam2_files)}] Error loading {cam2_file}: {e}")
            continue
        
        if cam2_df.empty:
            print(f"[{i}/{len(cam2_files)}] Skipping {cam2_file} - no valid data")
            continue
        
        # Match identities
        original_identity = cam2_df['identity_label'].iloc[0]  # Each CSV is one tracklet
        cam2_matched = match_identities_for_dataframe(
            cam1_df,
            cam2_df,
            time_window_seconds=time_window_seconds,
            distance_threshold=distance_threshold
        )
        
        # Vote for single identity for entire CSV (each CSV = one tracklet)
        matched_identities = cam2_matched['matched_identity'].dropna()
        
        if len(matched_identities) > 0:
            # Use majority voting
            identity_counts = matched_identities.value_counts()
            final_identity = identity_counts.idxmax()
            confidence = identity_counts.max() / len(matched_identities)
            
            # Apply final identity to ALL rows in this CSV
            cam2_matched['identity_label'] = final_identity
            
            # Track statistics
            stats[cam2_file][f"{original_identity}->{final_identity}"] = len(cam2_matched)
            
            status_msg = f"✓ {original_identity}->{final_identity} (conf: {confidence:.2%}, {identity_counts.max()}/{len(cam2_matched)} matched)"
        else:
            # No matches found, keep original
            status_msg = f"○ {original_identity} (no matches found)"
        
        # Save updated file
        output_path = os.path.join(output_dir, cam2_file)
        cam2_matched.to_csv(output_path, index=False)
        
        print(f"[{i}/{len(cam2_files)}] {status_msg}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MATCHING SUMMARY")
    print("=" * 60)
    
    all_mappings = []
    for filename, mappings in stats.items():
        for mapping, count in mappings.items():
            orig, matched = mapping.split('->')
            all_mappings.append({'filename': filename, 'original_identity': orig, 
                                'matched_identity': matched, 'count': count})
    
    if all_mappings:
        summary_df = pd.DataFrame(all_mappings)
        summary_df = summary_df.groupby(['filename', 'original_identity', 'matched_identity'])['count'].sum().reset_index()
        print(f"\nTotal files processed: {len(stats)}")
        print(f"Total mappings: {len(summary_df)}\n")
        print(summary_df.head(20))
        
        summary_path = os.path.join(output_dir, 'identity_mapping_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to: {summary_path}")
    else:
        print("No matches found")


# Example usage
if __name__ == "__main__":
    cam1_dir = "/Users/zhoumu/Downloads/vis/tracks/zag_elp_cam_017/2025-12-01"
    cam2_dir = "/Users/zhoumu/Downloads/vis/tracks/zag_elp_cam_018/2025-12-01"
    output_dir = "/Users/zhoumu/Downloads/vis/tracks/zag_elp_cam_018_matched/2025-12-01"
    
    process_with_sliding_window(
        cam1_dir,
        cam2_dir,
        output_dir=output_dir,
        window_hours=1,
        time_window_seconds=0.5,
        distance_threshold=2.0
    )
