import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment
from datetime import timedelta, time
import re
from collections import defaultdict
from bisect import bisect_right
from scipy.spatial.distance import cdist

def _preload_cam_df(tracklets, csv_cache):
    dfs = []
    for t in tracklets:
        p = t.get("track_csv_path")
        df = _load_csv_cached(p, csv_cache)
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all[df_all["timestamp"].notna()]
    return df_all.sort_values("timestamp").reset_index(drop=True)

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


def match_identities_for_dataframe_fast(
    cam1_df: pd.DataFrame,
    cam2_df: pd.DataFrame,
    time_window_seconds: float = 0.5,
    distance_threshold: float = 2.0,
    downsample_seconds: float | None = 1.0,
):
    """
    Faster version:
      - bin timestamps (floor)
      - only compare within nearby bins (±k)
      - use vectorized cdist for distance matrix
    """
    if cam1_df.empty or cam2_df.empty:
        return cam2_df

    cam1 = cam1_df.copy()
    cam2 = cam2_df.copy()

    # Ensure datetime
    cam1["timestamp"] = pd.to_datetime(cam1["timestamp"], errors="coerce")
    cam2["timestamp"] = pd.to_datetime(cam2["timestamp"], errors="coerce")
    cam1 = cam1[cam1["timestamp"].notna()]
    cam2 = cam2[cam2["timestamp"].notna()]

    cam2["matched_identity"] = None
    cam2["match_distance"] = np.nan

    # Choose a bin size; if None, use something close to time_window
    if downsample_seconds is None or downsample_seconds <= 0:
        bin_s = max(0.1, time_window_seconds)
    else:
        bin_s = float(downsample_seconds)

    cam1["_tbin"] = cam1["timestamp"].dt.floor(f"{bin_s}s")
    cam2["_tbin"] = cam2["timestamp"].dt.floor(f"{bin_s}s")

    # how many bins to look around to cover time_window_seconds?
    k = int(np.ceil(time_window_seconds / bin_s))

    # Pre-group cam1 by bin for quick access
    cam1_groups = {tb: g for tb, g in cam1.groupby("_tbin", sort=False)}

    # Process cam2 by bin
    for tb, cam2_frame in cam2.groupby("_tbin", sort=False):
        # Collect cam1 candidates from neighboring bins
        candidates = []
        for d in range(-k, k + 1):
            tb2 = tb + pd.to_timedelta(d * bin_s, unit="s")
            g = cam1_groups.get(tb2, None)
            if g is not None and len(g) > 0:
                candidates.append(g)

        if not candidates:
            continue

        temporal_candidates = pd.concat(candidates, ignore_index=False)

        # Vectorized distance matrix
        cam2_pos = cam2_frame[["world_x", "world_y"]].to_numpy(dtype=np.float32)
        cam1_pos = temporal_candidates[["world_x", "world_y"]].to_numpy(dtype=np.float32)

        if cam2_pos.size == 0 or cam1_pos.size == 0:
            continue

        dist = cdist(cam2_pos, cam1_pos)  # fast C implementation

        # Gate invalid pairs by setting to a large cost
        big = distance_threshold + 1e6
        cost = dist.copy()
        cost[cost > distance_threshold] = big

        # Hungarian
        row_ind, col_ind = linear_sum_assignment(cost)

        cam2_idx_list = cam2_frame.index.to_numpy()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            cam2_idx = cam2_idx_list[r]
            matched_row = temporal_candidates.iloc[c]
            cam2.at[cam2_idx, "matched_identity"] = matched_row["identity_label"]
            cam2.at[cam2_idx, "match_distance"] = float(dist[r, c])

    # Apply updates
    cam2["identity_label"] = cam2["matched_identity"].fillna(cam2["identity_label"])

    # Cleanup
    cam1.drop(columns=["_tbin"], errors="ignore", inplace=True)
    cam2.drop(columns=["_tbin"], errors="ignore", inplace=True)

    return cam2

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
        keep = ["timestamp", "world_x", "world_y", "identity_label", "filename"]
        df = df[keep]
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

def _tracklet_candidates_from_matches(
    cam2_matched: pd.DataFrame,
    topk: int = 5,
    min_votes: int = 5,
    score_mode: str = "exp",   # "exp" (distance-weighted) or "count"
    sigma: float = 1.0,        # meters, for exp weighting
):
    """
    Return ranked candidate identities for one cam2 tracklet based on frame matches.

    Output:
      candidates: list[(id, score, votes, avg_dist)]
      best: dict with best_id, best_score, margin, votes, avg_dist
    """
    m = cam2_matched.dropna(subset=["matched_identity"]).copy()
    if m.empty:
        return [], {
            "best_id": None, "best_score": 0.0, "margin": 0.0,
            "votes": 0, "avg_dist": np.nan
        }

    m["match_distance"] = pd.to_numeric(m["match_distance"], errors="coerce")
    m = m[m["match_distance"].notna()]

    if m.empty:
        return [], {
            "best_id": None, "best_score": 0.0, "margin": 0.0,
            "votes": 0, "avg_dist": np.nan
        }

    votes = m.groupby("matched_identity").size().rename("votes")
    avg_dist = m.groupby("matched_identity")["match_distance"].mean().rename("avg_dist")

    if score_mode == "count":
        score = votes.astype(float)
    else:
        # distance-weighted: sum(exp(-d/sigma))
        w = np.exp(-m["match_distance"].values / max(sigma, 1e-6))
        m["_w"] = w
        score = m.groupby("matched_identity")["_w"].sum().rename("score")
    
    if isinstance(min_votes, dict):
        # common mistake: min_votes=best_info or config dict
        if "min_votes" in min_votes:
            min_votes = int(min_votes["min_votes"])
        else:
            raise TypeError(f"min_votes must be int, got dict keys={list(min_votes.keys())}")
    try:
        min_votes = int(min_votes)
    except Exception:
        raise TypeError(f"min_votes must be int-like, got {type(min_votes)}: {min_votes}")

    df = pd.concat([votes, avg_dist, score], axis=1).reset_index().rename(columns={"index": "id"})
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    df = df[df["votes"] >= min_votes]

    if df.empty:
        return [], {
            "best_id": None, "best_score": 0.0, "margin": 0.0,
            "votes": 0, "avg_dist": np.nan
        }

    df = df.sort_values("score", ascending=False).head(topk)

    candidates = [(r["matched_identity"], float(r["score"]), int(r["votes"]), float(r["avg_dist"]))
                  for _, r in df.iterrows()]

    best_id, best_score, best_votes, best_avg = candidates[0]
    second_score = candidates[1][1] if len(candidates) > 1 else 0.0
    margin = float(best_score - second_score)

    return candidates, {
        "best_id": best_id,
        "best_score": float(best_score),
        "margin": margin,
        "votes": int(best_votes),
        "avg_dist": float(best_avg),
    }

def _accept_new_identity(original_id: str, best_id: str, best_info: dict,
                         min_margin: float = 0.5,
                         min_votes: int = 8,
                         max_avg_dist: float = 2.0):
    """
    Decide whether to overwrite original with best_id.
    This reduces spurious ID switches.
    """
    if best_id is None:
        return False
    # If it's same as original, always accept (no switch).
    if best_id == original_id:
        return True

    if best_info["votes"] < min_votes:
        return False
    if best_info["margin"] < min_margin:
        return False
    if not np.isfinite(best_info["avg_dist"]) or best_info["avg_dist"] > max_avg_dist:
        return False
    return True

def _resolve_id_conflicts(cam2_tracklets,
                          overlap_threshold_seconds: float = 1.0,
                          max_passes: int = 10):
    from collections import defaultdict

    def _overlap_seconds(a_start, a_end, b_start, b_end):
        latest = max(a_start, b_start)
        earliest = min(a_end, b_end)
        return max(0.0, (earliest - latest).total_seconds())

    # Parse times once
    for t in cam2_tracklets:
        t["_start"] = _parse_timestamp(t.get("start_timestamp")) or pd.Timestamp.min
        t["_end"] = _parse_timestamp(t.get("end_timestamp")) or pd.Timestamp.max
        t["match_quality"] = float(t.get("match_quality", 0.0))
        if t.get("candidate_id_scores") is None:
            t["candidate_id_scores"] = []
        if "original_identity_label" not in t:
            t["original_identity_label"] = t.get("identity_label", "unknown")

    def build_index():
        idx = defaultdict(list)  # idlabel -> list of indices
        for i, t in enumerate(cam2_tracklets):
            idx[t.get("identity_label", "unknown")].append(i)
        # sort by start time for determinism
        for k in idx:
            idx[k].sort(key=lambda ii: cam2_tracklets[ii]["_start"])
        return idx

    def would_conflict(i_tracklet, proposed_id, idx):
        ti = cam2_tracklets[i_tracklet]
        for j in idx.get(proposed_id, []):
            if j == i_tracklet:
                continue
            tj = cam2_tracklets[j]
            if _overlap_seconds(ti["_start"], ti["_end"], tj["_start"], tj["_end"]) >= overlap_threshold_seconds:
                return True
        return False

    for _ in range(max_passes):
        changed = False
        idx = build_index()

        for idlabel, group_idx in list(idx.items()):
            if len(group_idx) <= 1:
                continue

            # Sweep line to find conflicts (store groups as sets of indices)
            active = []
            conflict_groups = []

            for i in group_idx:
                ti = cam2_tracklets[i]

                # keep only active that still overlap with ti (>= threshold)
                new_active = []
                for a in active:
                    ta = cam2_tracklets[a]
                    if _overlap_seconds(ta["_start"], ta["_end"], ti["_start"], ti["_end"]) >= overlap_threshold_seconds:
                        new_active.append(a)
                active = new_active

                if active:
                    conflict_groups.append(set(active + [i]))

                active.append(i)

            # Merge intersecting groups
            merged = []
            for g in conflict_groups:
                placed = False
                for mg in merged:
                    if not g.isdisjoint(mg):
                        mg |= g
                        placed = True
                        break
                if not placed:
                    merged.append(set(g))

            # Resolve each merged conflict group
            for g in merged:
                if len(g) <= 1:
                    continue

                g_list = list(g)
                # Winner: highest match_quality, tie-breaker longest duration
                g_list.sort(
                    key=lambda ii: (
                        cam2_tracklets[ii].get("match_quality", 0.0),
                        (cam2_tracklets[ii]["_end"] - cam2_tracklets[ii]["_start"]).total_seconds(),
                    ),
                    reverse=True,
                )
                winner = g_list[0]
                losers = g_list[1:]

                # enforce winner keeps idlabel
                cam2_tracklets[winner]["identity_label"] = idlabel

                # rebuild idx after change
                idx = build_index()

                for lo in losers:
                    # try next-best candidate that doesn't conflict
                    reassigned = False
                    for cand in cam2_tracklets[lo].get("candidate_id_scores", []):
                        cand_id = cand[0]
                        if not cand_id or cand_id == idlabel:
                            continue
                        if not would_conflict(lo, cand_id, idx):
                            cam2_tracklets[lo]["identity_label"] = cand_id
                            changed = True
                            reassigned = True
                            break

                    if not reassigned:
                        orig = cam2_tracklets[lo].get("original_identity_label", "unknown")
                        if not would_conflict(lo, orig, idx):
                            cam2_tracklets[lo]["identity_label"] = orig
                        else:
                            cam2_tracklets[lo]["identity_label"] = "unknown"
                        changed = True

        if not changed:
            break

    # Cleanup internal fields
    for t in cam2_tracklets:
        t.pop("_start", None)
        t.pop("_end", None)

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

    # Preload entire cam1 once (huge speedup)
    cam1_df_all = _preload_cam_df(cam1_tracklets, csv_cache)
    if cam1_df_all.empty:
        return cam2_data
    
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

        cam1_df = cam1_df_all[
            (cam1_df_all["timestamp"] >= window_start) &
            (cam1_df_all["timestamp"] <= window_end)
        ]
        if cam1_df.empty:
            continue

        # cam2_matched = match_identities_for_dataframe(
        #     cam1_df,
        #     cam2_df,
        #     time_window_seconds=time_window_seconds,
        #     distance_threshold=distance_threshold,
        #     downsample_seconds=downsample_seconds,
        # )

        cam2_matched = match_identities_for_dataframe_fast(
            cam1_df,
            cam2_df,
            time_window_seconds=time_window_seconds,
            distance_threshold=distance_threshold,
            downsample_seconds=downsample_seconds if downsample_seconds is not None else 1.0,
        )

        matched_identities = cam2_matched["matched_identity"].dropna()
        if len(matched_identities) > 0:
            candidates, best = _tracklet_candidates_from_matches(
                cam2_matched,
                topk=5,
                min_votes=5,
                score_mode="exp",
                sigma=1.0,
            )

            # Store candidates for later conflict resolution
            t["candidate_id_scores"] = candidates

            # Define match_quality as best_score (or keep yours)
            t["match_quality"] = best["best_score"]

            # Decide whether to overwrite original ID
            proposed = best["best_id"]
            if _accept_new_identity(
                original_identity,
                proposed,
                best_info=best,
                min_margin=0.5,
                min_votes=8,
                max_avg_dist=distance_threshold,
            ):
                final_identity = proposed
            else:
                final_identity = original_identity

            t["identity_label"] = final_identity

            stats[f"{original_identity}->{final_identity}"] += 1
            summary_rows.append({
                "filename": os.path.basename(cam2_csv) if cam2_csv else "",
                "original_identity": original_identity,
                "matched_identity": final_identity,
                "match_quality": t["match_quality"],
                "votes": best["votes"],
                "margin": best["margin"],
                "avg_distance": best["avg_dist"],
            })
        else:
            t["candidate_id_scores"] = []
            t["match_quality"] = 0.0
            stats[f"{original_identity}"] += 1
            summary_rows.append({
                "filename": os.path.basename(cam2_csv) if cam2_csv else "",
                "original_identity": original_identity,
                "matched_identity": None,
                "match_quality": 0.0,
                "votes": 0,
                "margin": 0.0,
                "avg_distance": np.nan,
            })

    
    # Resolve ID conflicts (same ID assigned to multiple tracks at same time)
    cam2_tracklets = _resolve_id_conflicts(cam2_tracklets, overlap_threshold_seconds=1.0)
    
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
