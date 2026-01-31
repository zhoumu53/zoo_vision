import numpy as np
import pandas as pd
import os
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from datetime import timedelta


def load_camera_data(csv_path):
    """
    Load camera tracking data with proper timestamp parsing.
    
    Parameters
    ----------
    csv_path : str
        Path to a CSV file or directory containing multiple CSV files.
        If directory, all CSV files will be combined.
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with parsed timestamps
    """
    if os.path.isdir(csv_path):
        # Load all CSV files from directory
        csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
        
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in directory: {csv_path}")
        
        print(f"Loading {len(csv_files)} CSV files from {csv_path}...")
        
        dfs = []
        for csv_file in sorted(csv_files):
            file_path = os.path.join(csv_path, csv_file)
            try:
                df_temp = pd.read_csv(file_path)
                df_temp['filename'] = csv_file
                dfs.append(df_temp)
            except Exception as e:
                print(f"Warning: Failed to load {csv_file}: {e}")
                continue
        
        if len(dfs) == 0:
            raise ValueError(f"Failed to load any CSV files from: {csv_path}")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(dfs)} files into {len(df)} total rows")
    else:
        # Load single CSV file
        df = pd.read_csv(csv_path)
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp for efficiency
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove the rows with empty identity_label
    df = df[df["identity_label"].notna()]
    
    return df


def match_identities_knn(
    cam1_df,
    cam2_df,
    time_window_seconds=0.5,
    distance_threshold=2.0,
    k=3,
    output_path=None
):
    """
    Match identities using KNN approach with one-to-one constraint per frame.
    Finds top K matches for voting, then applies Hungarian algorithm per frame.
    
    Parameters
    ----------
    cam1_df : pd.DataFrame
        Camera 1 data (ground truth identities)
    cam2_df : pd.DataFrame
        Camera 2 data (to be updated)
    time_window_seconds : float
        Maximum time difference for matching (seconds)
    distance_threshold : float
        Maximum world distance for matching (meters)
    k : int
        Number of nearest neighbors to consider for voting
    output_path : str or None
        Path to save updated cam2 data
        
    Returns
    -------
    pd.DataFrame
        Updated cam2 data with matched identities
    """
    
    cam2_updated = cam2_df.copy()
    cam2_updated['matched_identity'] = None
    cam2_updated['match_confidence'] = np.nan
    
    print(f"Processing {len(cam2_updated)} detections from cam2 with KNN (k={k})...")
    
    # Get unique timestamps from cam2
    cam2_timestamps = cam2_updated['timestamp'].unique()
    
    matched_count = 0
    
    # Process each timestamp separately
    for cam2_time in cam2_timestamps:
        cam2_frame = cam2_updated[cam2_updated['timestamp'] == cam2_time]
        
        if len(cam2_frame) == 0:
            continue
        
        # Find temporal candidates
        time_diff = (cam1_df['timestamp'] - cam2_time).abs()
        temporal_candidates = cam1_df[time_diff <= timedelta(seconds=time_window_seconds)]
        
        if len(temporal_candidates) == 0:
            continue
        
        # For each cam2 detection, compute KNN-based identity scores
        cam2_identity_scores = []  # List of dicts: {identity: score}
        
        for idx, row in cam2_frame.iterrows():
            cam2_pos = np.array([row['world_x'], row['world_y']])
            
            # Calculate distances to all cam1 candidates
            cam1_positions = temporal_candidates[['world_x', 'world_y']].values
            distances = np.linalg.norm(cam1_positions - cam2_pos, axis=1)
            
            # Get valid neighbors within threshold
            valid_mask = distances <= distance_threshold
            valid_distances = distances[valid_mask]
            valid_candidates = temporal_candidates[valid_mask]
            
            if len(valid_candidates) == 0:
                cam2_identity_scores.append({})
                continue
            
            # Get top K nearest
            sorted_indices = np.argsort(valid_distances)[:k]
            nearest_identities = valid_candidates.iloc[sorted_indices]['identity_label'].values
            nearest_distances = valid_distances[sorted_indices]
            
            # Compute scores (inverse distance weighted voting)
            identity_scores = {}
            for identity, dist in zip(nearest_identities, nearest_distances):
                weight = 1.0 / (dist + 1e-6)  # Avoid division by zero
                identity_scores[identity] = identity_scores.get(identity, 0) + weight
            
            cam2_identity_scores.append(identity_scores)
        
        # Build cost matrix based on identity scores
        all_identities = list(set([id for scores in cam2_identity_scores for id in scores.keys()]))
        
        if len(all_identities) == 0:
            continue
        
        cost_matrix = np.full((len(cam2_frame), len(all_identities)), 1e6)
        
        for i, scores in enumerate(cam2_identity_scores):
            for j, identity in enumerate(all_identities):
                if identity in scores:
                    # Lower cost = better match (negate score)
                    cost_matrix[i, j] = -scores[identity]
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Apply matches
        for cam2_idx, identity_idx in zip(row_ind, col_ind):
            cost = cost_matrix[cam2_idx, identity_idx]
            
            if cost < 1e6:  # Valid match
                cam2_original_idx = cam2_frame.index[cam2_idx]
                matched_identity = all_identities[identity_idx]
                confidence = -cost  # Convert back from negative
                
                cam2_updated.at[cam2_original_idx, 'matched_identity'] = matched_identity
                cam2_updated.at[cam2_original_idx, 'match_confidence'] = confidence
                matched_count += 1
    
    # Update labels
    cam2_updated['original_identity'] = cam2_updated['identity_label']
    cam2_updated['identity_label'] = cam2_updated['matched_identity'].fillna(cam2_updated['identity_label'])
    
    match_rate = matched_count / len(cam2_updated) * 100
    print(f"\nMatched {matched_count}/{len(cam2_updated)} detections ({match_rate:.1f}%)")
    
    # Statistics
    print("\nIdentity mapping summary:")
    mapping = cam2_updated[cam2_updated['matched_identity'].notna()].groupby(
        ['original_identity', 'matched_identity']
    ).size().reset_index(name='count')
    print(mapping)
    
    if output_path:
        cam2_updated.to_csv(output_path, index=False)
        print(f"\nSaved updated data to: {output_path}")
        
    ## only groupby filename, then show unique of matched_identity, matched_identity
    mapping_by_file = cam2_updated[cam2_updated['matched_identity'].notna()].groupby(
        ['filename', 'original_identity', 'matched_identity']
    ).size().reset_index(name='count')
    print(mapping_by_file)
    
    return cam2_updated


if __name__ == "__main__":
    cam1_path = "/Users/zhoumu/Downloads/vis/zag_elp_cam_016/2025-11-15"
    cam2_path = "/Users/zhoumu/Downloads/vis/zag_elp_cam_019/2025-11-15"
    
    cam1_df = load_camera_data(cam1_path)
    cam2_df = load_camera_data(cam2_path)
    
    cam2_matched_knn = match_identities_knn(
        cam1_df,
        cam2_df,
        time_window_seconds=0.5,
        distance_threshold=2.0,
        k=3,
        output_path="cam2_matched_knn.csv"
    )
