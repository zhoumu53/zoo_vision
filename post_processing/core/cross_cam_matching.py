"""
Cross-Camera ID Matching Module (v2 – track-level)
====================================================
Matches individual identities across two cameras that observe the same room
using **raw track CSV data** (timestamp, world_x, world_y).

Unlike v1 (which relied on ``stitched_id`` and ``voted_track_label``), this
version works **track-by-track** using only the position time-series from each
CSV, so it is immune to upstream stitching or ReID errors.

Algorithm
---------
1. Load tracklet metadata from per-camera JSON (start / end timestamps,
   ``track_csv_path``).  Use time ranges for fast overlap filtering.
2. Load all track CSVs, extract ``(timestamp, world_x, world_y)`` per track.
3. Bin observations into temporal bins (default 1 s).
4. In each bin find active tracks from both cameras, build a world-position
   distance matrix, and run the Hungarian algorithm to get the optimal
   one-to-one assignment.
5. Accumulate per-bin votes ``(cam1_track, cam2_track) → count``.
6. Final global one-to-one assignment: greedy, sorted by vote count
   (highest first).
7. Assign ``cross_cam_id``: matched pairs share an ID; unmatched tracks
   get unique IDs.
8. Optionally propagate ``cross_cam_id`` + corrected ``identity_label``
   back into the source JSON files.

Public API
----------
* ``match_tracks_cross_camera``       – core matching (returns mapping + summary)
* ``run_cross_camera_matching_v2``    – top-level orchestrator (JSON → JSON)
* ``compute_cross_cam_stitched_id``   – Union-Find merger of stitched_id + cross_cam_id
* ``compute_cross_cam_individual``    – temporal max-cut partition into N individual slots
                                        (spurious detections → ``-1``)
* ``vote_identity_by_xcsid_reid``     – merged ReID voting per xcsid group
* ``smooth_behavior_cross_camera``    – cross-camera behaviour label alignment
* ``assign_identity_by_xcsid``        – conflict-aware identity assignment
* ``summarize_cross_cam_match``       – human-readable summary table
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from post_processing.utils import load_embedding
from post_processing.core.reid_inference import match_to_gallery
from post_processing.core.tracklet_manager import vote_identity_from_matched_labels


# ────────────────────────── helpers ─────────────────────────────────────────── #

def _load_track_positions(csv_path: str | Path) -> pd.DataFrame | None:
    """Load (timestamp, world_x, world_y) from a single track CSV.

    Returns *None* when the file doesn't exist or is empty.
    """
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, usecols=lambda c: c in {
            "timestamp", "world_x", "world_y",
        })
    except Exception:
        return None
    if df.empty or "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "world_x", "world_y"])
    if df.empty:
        return None
    return df.sort_values("timestamp").reset_index(drop=True)


def _build_track_index(tracklets_data: list[dict], cam_label: str) -> list[dict]:
    """Build a lightweight index: (track_filename, csv_path, start_ts, end_ts, camera)."""
    tracks = []
    for t in tracklets_data:
        csv_path = t.get("track_csv_path", "")
        if not csv_path:
            continue
        start = t.get("start_timestamp")
        end = t.get("end_timestamp")
        if not start or not end:
            continue
        try:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
        except Exception:
            continue
        tracks.append({
            "track_filename": t.get("track_filename", ""),
            "csv_path": csv_path,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "camera": cam_label,
        })
    return tracks


# ──────────────── 1. Track-level cross-camera matching ─────────────────────── #

def match_tracks_cross_camera(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    cam1_id: str,
    cam2_id: str,
    distance_threshold: float = 2.0,
    bin_seconds: float = 1.0,
    min_matched_bins: int = 5,
    logger: Optional[logging.Logger] = None,
) -> tuple[dict[str, int], pd.DataFrame]:
    """
    Match tracks across two cameras using **only** raw CSV position data.

    Algorithm
    ---------
    1. Load all track CSVs for both cameras; tag each frame with its
       ``track_filename`` and camera label.
    2. Bin all frames into temporal bins (``bin_seconds``).
    3. In each bin compute the mean (world_x, world_y) per active track,
       build the inter-camera distance matrix, and run Hungarian
       matching.
    4. Accumulate per-bin vote counts: ``(cam1_track, cam2_track) → N``.
    5. Global one-to-one assignment via greedy matching (sorted by vote
       count descending); resolve conflicts.
    6. Assign ``cross_cam_id``: matched pairs share an ID; unmatched
       tracks get unique IDs.

    Parameters
    ----------
    tracklets_cam1, tracklets_cam2 : list[dict]
        Tracklet dicts loaded from JSON for each camera.
    cam1_id, cam2_id : str
        Camera identifiers (e.g. ``'016'``, ``'019'``).
    distance_threshold : float
        Max world-distance (metres) to accept a bin-level match.
    bin_seconds : float
        Temporal bin width in seconds (default 1 s).
    min_matched_bins : int
        Minimum number of bins where a pair must match to be accepted.

    Returns
    -------
    (track_to_xcid, summary_df)
        - ``track_to_xcid``: dict mapping **every** track_filename to its
          ``cross_cam_id`` (int).
        - ``summary_df``: DataFrame with one row per matched / unmatched
          track showing match details.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # ── 1. Build track index & pre-filter ─────────────────────────────────── #
    idx1 = _build_track_index(tracklets_cam1, cam1_id)
    idx2 = _build_track_index(tracklets_cam2, cam2_id)
    logger.info("Track index: cam %s = %d tracks, cam %s = %d tracks",
                cam1_id, len(idx1), cam2_id, len(idx2))

    # ── 2. Load CSVs ──────────────────────────────────────────────────────── #
    frames: list[pd.DataFrame] = []
    loaded_tracks: set = set()

    for track_info in idx1 + idx2:
        tf = track_info["track_filename"]
        if tf in loaded_tracks:
            continue
        df = _load_track_positions(track_info["csv_path"])
        if df is None:
            continue
        df["_track"] = tf
        df["_camera"] = track_info["camera"]
        frames.append(df)
        loaded_tracks.add(tf)

    if not frames:
        logger.warning("No track CSVs loaded — nothing to match.")
        return {}, pd.DataFrame()

    all_frames = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d total frames from %d tracks.", len(all_frames), len(loaded_tracks))

    # ── 3. Bin by time ────────────────────────────────────────────────────── #
    all_frames["_tbin"] = all_frames["timestamp"].dt.floor(f"{bin_seconds}s")

    # Pre-compute mean position per (bin, track)
    agg = (
        all_frames
        .groupby(["_tbin", "_camera", "_track"], sort=False)
        .agg(wx=("world_x", "mean"), wy=("world_y", "mean"))
        .reset_index()
    )
    cam1_bins = agg[agg["_camera"] == cam1_id]
    cam2_bins = agg[agg["_camera"] == cam2_id]

    # Group by bin for fast lookup
    cam1_by_bin = {tb: g for tb, g in cam1_bins.groupby("_tbin", sort=False)}
    cam2_by_bin = {tb: g for tb, g in cam2_bins.groupby("_tbin", sort=False)}

    # ── 4. Per-bin Hungarian matching ─────────────────────────────────────── #
    BIG = distance_threshold + 1e6
    pair_votes: Counter = Counter()    # (cam1_track, cam2_track) -> count
    pair_dists: dict = defaultdict(list)  # same key -> list of distances

    common_bins = set(cam1_by_bin.keys()) & set(cam2_by_bin.keys())
    logger.info("Matching across %d common time bins ...", len(common_bins))

    for tbin in common_bins:
        g1 = cam1_by_bin[tbin]
        g2 = cam2_by_bin[tbin]

        pos1 = g1[["wx", "wy"]].values
        pos2 = g2[["wx", "wy"]].values
        tracks1 = g1["_track"].values
        tracks2 = g2["_track"].values

        dist = cdist(pos1, pos2)  # (n1, n2)
        cost = dist.copy()
        cost[cost > distance_threshold] = BIG

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= BIG:
                continue
            t1, t2 = tracks1[r], tracks2[c]
            pair_votes[(t1, t2)] += 1
            pair_dists[(t1, t2)].append(float(dist[r, c]))

    logger.info("Collected %d candidate pairs.", len(pair_votes))

    # ── 5. Global one-to-one assignment (greedy, by vote count) ───────────── #
    sorted_pairs = sorted(pair_votes.items(), key=lambda x: x[1], reverse=True)

    assigned_cam1: set = set()
    assigned_cam2: set = set()
    final_matches: list[dict] = []  # for summary

    for (t1, t2), votes in sorted_pairs:
        if votes < min_matched_bins:
            break
        if t1 in assigned_cam1 or t2 in assigned_cam2:
            continue
        dists = pair_dists[(t1, t2)]
        median_dist = float(np.median(dists))
        final_matches.append({
            "cam1_track": t1,
            "cam2_track": t2,
            "matched_bins": votes,
            "median_distance": round(median_dist, 4),
            "mean_distance": round(float(np.mean(dists)), 4),
        })
        assigned_cam1.add(t1)
        assigned_cam2.add(t2)

    logger.info("Final one-to-one matches: %d pairs.", len(final_matches))

    # ── 6. Assign cross_cam_id ────────────────────────────────────────────── #
    track_to_xcid: dict[str, int] = {}
    xcid = 0

    for m in final_matches:
        track_to_xcid[m["cam1_track"]] = xcid
        track_to_xcid[m["cam2_track"]] = xcid
        m["cross_cam_id"] = xcid
        xcid += 1

    # Unmatched tracks get unique IDs
    all_tracknames = {t["track_filename"] for t in idx1 + idx2}
    for tf in sorted(all_tracknames):
        if tf not in track_to_xcid:
            track_to_xcid[tf] = xcid
            xcid += 1

    logger.info("Assigned %d unique cross_cam_ids (%d matched + %d unmatched).",
                xcid, len(final_matches), xcid - len(final_matches))

    # ── 7. Build summary DataFrame ────────────────────────────────────────── #
    summary_rows: list[dict] = []

    # Matched
    for m in final_matches:
        summary_rows.append({
            "cross_cam_id": m["cross_cam_id"],
            "cam1_track": m["cam1_track"],
            "cam2_track": m["cam2_track"],
            "matched_bins": m["matched_bins"],
            "median_distance": m["median_distance"],
            "status": "matched",
        })

    # Unmatched cam1
    for t in idx1:
        tf = t["track_filename"]
        if tf not in assigned_cam1:
            summary_rows.append({
                "cross_cam_id": track_to_xcid[tf],
                "cam1_track": tf,
                "cam2_track": "",
                "matched_bins": 0,
                "median_distance": np.nan,
                "status": f"unmatched_cam{cam1_id}",
            })

    # Unmatched cam2
    for t in idx2:
        tf = t["track_filename"]
        if tf not in assigned_cam2:
            summary_rows.append({
                "cross_cam_id": track_to_xcid[tf],
                "cam1_track": "",
                "cam2_track": tf,
                "matched_bins": 0,
                "median_distance": np.nan,
                "status": f"unmatched_cam{cam2_id}",
            })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("matched_bins", ascending=False).reset_index(drop=True)
        logger.info("\nTrack matching summary:\n%s", summary_df.to_string(index=False))

    return track_to_xcid, summary_df


# ──────────────── 2. Top-level orchestrator (v2) ───────────────────────────── #

def run_cross_camera_matching_v2(
    record_root,
    camera_ids: list[str],
    start_datetime,
    end_datetime,
    distance_threshold: float = 2.0,
    bin_seconds: float = 1.0,
    min_matched_bins: int = 5,
    known_individuals: Optional[list[str]] = None,
    gallery_path: Optional[str | Path] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[dict[str, int], pd.DataFrame]:
    """
    End-to-end cross-camera matching for one camera pair.

    Loads per-camera JSON tracklet files, runs track-level matching
    using raw CSV position data, assigns ``cross_cam_id``, and writes
    the results (``cross_cam_id`` + corrected ``identity_label``) back
    into both JSON files.

    Parameters
    ----------
    record_root : Path
        Root directory for records.
    camera_ids : list[str]
        Exactly two camera IDs, e.g. ``['016', '019']``.
    start_datetime, end_datetime : pd.Timestamp or str
        Processing time window.
    distance_threshold : float
        Max world-distance (metres) to accept a match.
    bin_seconds : float
        Temporal bin width in seconds.
    min_matched_bins : int
        Minimum matched bins for a pair to be accepted.
    known_individuals : list[str] or None
        Allowed identity labels for this room (max 2).
    logger : logging.Logger, optional

    Returns
    -------
    (track_to_xcid, summary_df)
    """
    from post_processing.tools.utils import (
        load_tracklet_json_for_camera,
        update_tracklet_json_identity_labels,
    )

    if logger is None:
        logger = logging.getLogger(__name__)

    record_root = Path(record_root)
    start_datetime = pd.Timestamp(start_datetime)
    end_datetime = pd.Timestamp(end_datetime)

    if len(camera_ids) != 2:
        raise ValueError(f"Need exactly 2 camera IDs, got {len(camera_ids)}")

    cam1_id, cam2_id = camera_ids

    logger.info("=" * 70)
    logger.info("Cross-camera matching (v2 – track-level): cam %s ↔ cam %s",
                cam1_id, cam2_id)
    logger.info("=" * 70)

    # ── Load JSON for each camera ─────────────────────────────────────────── #
    json_data: dict[str, tuple[Path, list]] = {}
    for cam_id in camera_ids:
        try:
            jp, td = load_tracklet_json_for_camera(
                record_root=record_root,
                cam_id=cam_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )
            json_data[cam_id] = (jp, td)
            logger.info("Loaded %d tracklets from JSON for cam %s (%s)",
                        len(td), cam_id, jp.name)
        except FileNotFoundError as e:
            logger.error("JSON not found for cam %s: %s", cam_id, e)
            return {}, pd.DataFrame()

    tracklets_cam1 = json_data[cam1_id][1]
    tracklets_cam2 = json_data[cam2_id][1]

    # ── Run track-level matching ──────────────────────────────────────────── #
    track_to_xcid, summary_df = match_tracks_cross_camera(
        tracklets_cam1=tracklets_cam1,
        tracklets_cam2=tracklets_cam2,
        cam1_id=cam1_id,
        cam2_id=cam2_id,
        distance_threshold=distance_threshold,
        bin_seconds=bin_seconds,
        min_matched_bins=min_matched_bins,
        logger=logger,
    )

    # ── Write cross_cam_id to tracklet dicts (in-place, no save yet) ──────── #
    for cam_id in camera_ids:
        json_path, tracklets_data = json_data[cam_id]
        for tracklet in tracklets_data:
            tf = tracklet.get("track_filename", "")
            if tf in track_to_xcid:
                xcid_val = track_to_xcid[tf]
                try:
                    xcid_val = int(xcid_val)
                except (ValueError, TypeError):
                    pass
                tracklet["cross_cam_id"] = xcid_val

    # ── Compute cross_cam_stitched_id (Union-Find merge) ──────────────── #
    tracklets_cam1 = json_data[cam1_id][1]
    tracklets_cam2 = json_data[cam2_id][1]

    track_to_xcsid = compute_cross_cam_stitched_id(
        tracklets_cam1=tracklets_cam1,
        tracklets_cam2=tracklets_cam2,
        cam1_id=cam1_id,
        cam2_id=cam2_id,
        logger=logger,
    )

    # Write cross_cam_stitched_id to tracklet dicts (in-place, no save yet)
    for cam_id in camera_ids:
        json_path, tracklets_data = json_data[cam_id]
        for tracklet in tracklets_data:
            tf = tracklet.get("track_filename", "")
            if tf in track_to_xcsid:
                tracklet["cross_cam_stitched_id"] = track_to_xcsid[tf]

    # ── Compute cross_cam_individual (temporal max-cut partition) ──────── #
    tracklets_cam1 = json_data[cam1_id][1]
    tracklets_cam2 = json_data[cam2_id][1]

    n_individuals = len(known_individuals) if known_individuals else 2

    track_to_ind = compute_cross_cam_individual(
        tracklets_cam1=tracklets_cam1,
        tracklets_cam2=tracklets_cam2,
        n_individuals=n_individuals,
        logger=logger,
    )

    # Write cross_cam_individual to tracklet dicts (in-place, no save yet)
    for cam_id in camera_ids:
        json_path, tracklets_data = json_data[cam_id]
        for tracklet in tracklets_data:
            tf = tracklet.get("track_filename", "")
            if tf in track_to_ind:
                tracklet["cross_cam_individual"] = track_to_ind[tf]

    # ── ReID voting per cross_cam_stitched_id group ───────────────────────── #
    tracklets_cam1 = json_data[cam1_id][1]
    tracklets_cam2 = json_data[cam2_id][1]

    if gallery_path:
        xcsid_vote_results = vote_identity_by_xcsid_reid(
            tracklets_cam1=tracklets_cam1,
            tracklets_cam2=tracklets_cam2,
            gallery_path=gallery_path,
            known_individuals=known_individuals,
            logger=logger,
        )
    else:
        logger.warning("No gallery_path provided — skipping ReID voting "
                       "per cross_cam_stitched_id.")

    # ── Assign identity_label per partition via ReID vote ────────────────── #
    #  1) Aggregate the top-1 voted label per cross_cam_individual partition
    #  2) If both partitions share the same label → "confused"
    #  3) Otherwise → use partition's voted label
    tracklets_cam1 = json_data[cam1_id][1]
    tracklets_cam2 = json_data[cam2_id][1]
    all_tracklets_flat = tracklets_cam1 + tracklets_cam2

    track_to_label: dict[str, str] = {}
    partition_labels = _resolve_partition_labels(
        all_tracklets=all_tracklets_flat,
        known_individuals=known_individuals,
        logger=logger,
    )

    skip_vote = {"unknown", "spurious", ""}
    for t in all_tracklets_flat:
        tf = t.get("track_filename", "")
        if not tf:
            continue
        ind = t.get("cross_cam_individual")
        if ind is not None:
            if ind == -1:
                # Spurious detection → mark invalid
                track_to_label[tf] = "invalid"
            elif ind in partition_labels:
                lbl = partition_labels[ind]
                if lbl not in skip_vote:
                    track_to_label[tf] = lbl

    # ── Cross-camera behaviour smoothing ──────────────────────────────── #
    tracklets_cam1 = json_data[cam1_id][1]
    tracklets_cam2 = json_data[cam2_id][1]

    smooth_behavior_cross_camera(
        tracklets_cam1=tracklets_cam1,
        tracklets_cam2=tracklets_cam2,
        logger=logger,
        min_bout_seconds=300,   ## 5min
    )

    # ── Write identity_label + save JSON (single write) ───────────────── #
    skip_labels = {"unknown", "spurious", ""}

    for cam_id in camera_ids:
        json_path, tracklets_data = json_data[cam_id]
        n_xcid = sum(1 for t in tracklets_data if "cross_cam_id" in t)
        n_xcsid = sum(1 for t in tracklets_data if "cross_cam_stitched_id" in t)
        n_ind = sum(1 for t in tracklets_data
                    if t.get("cross_cam_individual") is not None)
        n_label = 0
        n_confused = 0
        n_invalid = 0

        for tracklet in tracklets_data:
            tf = tracklet.get("track_filename", "")
            track_npz_path = tracklet.get("track_csv_path", "").replace('.csv', '.npz')
            if not Path(track_npz_path).exists():
                continue
            if tf in track_to_label:
                new_label = track_to_label[tf]
                if new_label not in skip_labels:
                    tracklet["identity_label"] = new_label
                    n_label += 1
                    if new_label == "confused":
                        n_confused += 1
                    if new_label == "invalid":
                        n_invalid += 1

        logger.info(
            "Camera %s: wrote %d cross_cam_ids, %d cross_cam_stitched_ids, "
            "%d cross_cam_individuals, %d identity_labels "
            "(%d confused, %d invalid) → %s",
            cam_id, n_xcid, n_xcsid, n_ind, n_label, n_confused, n_invalid,
            json_path.name,
        )

    # # ── Verify identity conflicts across cameras ──────────────────────── #
    # #   If the same identity_label is assigned to different tracks at the
    # #   same time across cameras, append '_conflict' to the label.
    # _verify_identity_conflicts_across_cameras(
    #     json_data=json_data,
    #     camera_ids=camera_ids,
    #     logger=logger,
    # )

    # ── Save JSON (after conflict verification) ───────────────────────── #
    for cam_id in camera_ids:
        json_path, tracklets_data = json_data[cam_id]
        with open(json_path, "w") as f:
            json.dump(tracklets_data, f, indent=2)

    return track_to_xcid, summary_df


def _resolve_partition_labels(
    all_tracklets: list[dict],
    known_individuals: Optional[list[str]],
    logger: Optional[logging.Logger] = None,
) -> dict[int, str]:
    """
    Decide the identity label for each ``cross_cam_individual`` partition
    using the per-xcsid ReID votes (``cross_cam_stitched_id_vote_counts``).

    1. Aggregate vote counts per partition.
    2. If two (or more) partitions share the **same** top-1 label
       → all of those partitions get ``"confused"``.
    3. Otherwise → each partition gets its own top-1 label.

    Returns
    -------
    dict[int, str]
        ``partition_id → identity_label``
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # ── 1. Aggregate vote counts per partition ────────────────────────────── #
    partition_votes: dict[int, Counter] = defaultdict(Counter)
    seen_xcsids: dict[int, set] = defaultdict(set)

    for t in all_tracklets:
        ind = t.get("cross_cam_individual")
        xcsid = t.get("cross_cam_stitched_id")
        vc = t.get("cross_cam_stitched_id_vote_counts")
        if ind is None or xcsid is None or not vc:
            continue
        # Only count each xcsid once per partition (they all share the same vc)
        if xcsid in seen_xcsids[ind]:
            continue
        seen_xcsids[ind].add(xcsid)

        for lbl, info in vc.items():
            cnt = info["count"] if isinstance(info, dict) else int(info)
            partition_votes[ind][lbl] += cnt

    if not partition_votes:
        logger.warning("No ReID votes found in any partition.")
        return {}

    # ── 1b. Partition -1 is always "invalid" (spurious detections) ─────── #
    partition_labels: dict[int, str] = {}
    if -1 in partition_votes:
        partition_labels[-1] = "invalid"
        del partition_votes[-1]
        logger.info("  Partition -1 (spurious) → identity_label='invalid'")

    # ── 2. Find top-1 label per partition ─────────────────────────────────── #
    partition_top1: dict[int, str] = {}
    for ind, votes in partition_votes.items():
        if ind == -1:
            continue  # already handled
        if votes:
            top1 = votes.most_common(1)[0][0]
            partition_top1[ind] = top1
        else:
            partition_top1[ind] = "unknown"

    # ── 3. Check for duplicates → "confused" ─────────────────────────────── #
    #  Only flag "confused" when there are ≥ 2 known individuals.
    #  With a single known individual all partitions are the same elephant.
    n_known = len(known_individuals) if known_individuals else 0

    confused_partitions: set[int] = set()

    if n_known >= 2:
        label_to_partitions: dict[str, list[int]] = defaultdict(list)
        for ind, lbl in partition_top1.items():
            label_to_partitions[lbl].append(ind)

        for lbl, partitions in label_to_partitions.items():
            if lbl in ("unknown", "invalid", "spurious", ""):
                continue
            if len(partitions) > 1:
                # Same label voted for multiple partitions → confused
                for ind in partitions:
                    partition_labels[ind] = "confused"
                    confused_partitions.add(ind)

    # Assign non-confused partitions their top-1 label
    for ind, lbl in partition_top1.items():
        if ind not in confused_partitions:
            partition_labels[ind] = lbl

    # ── Log ───────────────────────────────────────────────────────────────── #
    logger.info("\n" + "=" * 80)
    logger.info("PARTITION → IDENTITY LABEL RESOLUTION")
    logger.info("=" * 80)
    for ind in sorted(partition_votes.keys()):
        top3 = partition_votes[ind].most_common(3)
        top3_str = ", ".join(f"{l}:{c}" for l, c in top3)
        final = partition_labels.get(ind, "unknown")
        logger.info("  Partition %d: votes=[%s] → identity_label='%s'",
                    ind, top3_str, final)
    if confused_partitions:
        logger.warning("  ⚠  Partitions %s share the same top-1 ReID label "
                       "→ marked 'confused'", sorted(confused_partitions))

    return partition_labels


def _check_per_camera_identity_conflicts(
    json_data: dict[str, tuple[Path, list]],
    camera_ids: list[str],
    track_to_label: dict[str, str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    For each camera independently, check whether two different xcsid groups
    that were assigned the **same** ``cross_cam_stitched_id_vote`` label
    overlap in time.  Print any conflicts found.

    This uses the ReID-voted label (``cross_cam_stitched_id_vote``) rather
    than the final ``identity_label`` (which already tries to resolve
    conflicts via greedy assignment).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    for cam_id in camera_ids:
        _, tracklets = json_data[cam_id]

        # Build: label → list of (start, end, xcsid, track_filename)
        label_intervals: dict[str, list[tuple]] = defaultdict(list)
        for t in tracklets:
            lbl = t.get("cross_cam_stitched_id_vote", "")
            xcsid = t.get("cross_cam_stitched_id")
            if not lbl or lbl in ("unknown", "invalid", "spurious", "") or xcsid is None:
                continue
            try:
                ts = pd.Timestamp(t["start_timestamp"])
                te = pd.Timestamp(t["end_timestamp"])
            except Exception:
                continue
            label_intervals[lbl].append((ts, te, xcsid, t.get("track_filename", "")))

        cam_conflicts = []
        for lbl, intervals in label_intervals.items():
            # Group intervals by xcsid
            xcsid_ranges: dict[int, tuple] = {}
            for ts, te, xcsid, tf in intervals:
                if xcsid not in xcsid_ranges:
                    xcsid_ranges[xcsid] = (ts, te, [tf])
                else:
                    old_s, old_e, tfs = xcsid_ranges[xcsid]
                    xcsid_ranges[xcsid] = (min(old_s, ts), max(old_e, te), tfs + [tf])

            xcsid_list = sorted(xcsid_ranges.keys())
            for i in range(len(xcsid_list)):
                for j in range(i + 1, len(xcsid_list)):
                    xa, xb = xcsid_list[i], xcsid_list[j]
                    sa, ea, tfs_a = xcsid_ranges[xa]
                    sb, eb, tfs_b = xcsid_ranges[xb]
                    if sa < eb and sb < ea:
                        overlap_s = max(sa, sb)
                        overlap_e = min(ea, eb)
                        cam_conflicts.append({
                            "camera": cam_id,
                            "label": lbl,
                            "xcsid_a": xa,
                            "xcsid_b": xb,
                            "overlap_seconds": (overlap_e - overlap_s).total_seconds(),
                            "overlap_start": str(overlap_s)[:19],
                            "overlap_end": str(overlap_e)[:19],
                        })

        if cam_conflicts:
            cdf = pd.DataFrame(cam_conflicts)
            logger.warning(
                "\n⚠  PER-CAMERA IDENTITY CONFLICTS (cam %s): %d conflicts\n%s",
                cam_id, len(cam_conflicts), cdf.to_string(index=False),
            )
        else:
            logger.info("Camera %s: no per-camera identity conflicts "
                        "(cross_cam_stitched_id_vote).", cam_id)


def _assign_identity_from_matches(
    track_to_xcid: dict[str, int],
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    known_individuals: list[str],
    track_to_label: dict[str, str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Legacy per-cross_cam_id assignment.  Kept for reference; prefer
    :func:`assign_identity_by_xcsid` which handles temporal conflicts."""
    pass


def _verify_identity_conflicts_across_cameras(
    json_data: dict[str, tuple[Path, list]],
    camera_ids: list[str],
    logger: Optional[logging.Logger] = None,
) -> int:
    """Verify identity labels across cameras; flag temporal conflicts.

    After all ``identity_label`` assignments are written, this function
    checks whether the **same** identity label is assigned to **different
    tracks on different cameras** at the same time.  When two elephants
    should not share a label simultaneously, this indicates an assignment
    error.

    For every conflicting track, ``identity_label`` is changed to
    ``{original_label}_conflict`` (in-place in the tracklet dicts, which
    are written to JSON by the caller).

    Algorithm
    ---------
    1. Collect all tracklet intervals ``(start, end, track_filename,
       camera, xcsid)`` grouped by ``identity_label``.
    2. For each label, find pairs of tracks **from different xcsid
       groups** that overlap in time.  (Tracks from the same xcsid on
       different cameras are the *same* individual — not a conflict.)
    3. Mark all conflicting tracks with ``{label}_conflict``.

    Parameters
    ----------
    json_data : dict
        ``cam_id → (json_path, tracklets_data)``
    camera_ids : list[str]
        Camera IDs being processed.
    logger : logging.Logger, optional

    Returns
    -------
    int
        Number of tracks whose ``identity_label`` was changed.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    skip = {"unknown", "invalid", "spurious", "confused", ""}

    # ── 1. Collect intervals per identity_label ───────────────────────────── #
    label_intervals: dict[str, list[dict]] = defaultdict(list)
    for cam_id in camera_ids:
        _, tracklets = json_data[cam_id]
        for t in tracklets:
            lbl = t.get("identity_label", "")
            if not lbl or lbl in skip or lbl.endswith("_conflict"):
                continue
            try:
                ts = pd.Timestamp(t["start_timestamp"])
                te = pd.Timestamp(t["end_timestamp"])
            except (KeyError, ValueError):
                continue
            label_intervals[lbl].append({
                "start": ts,
                "end": te,
                "track_filename": t.get("track_filename", ""),
                "camera": cam_id,
                "xcsid": t.get("cross_cam_stitched_id"),
                "tracklet_ref": t,  # direct reference for in-place edit
            })

    # ── 2. Find cross-xcsid temporal conflicts ───────────────────────────── #
    conflict_tracks: set[str] = set()   # track_filenames to mark
    conflict_rows: list[dict] = []

    for lbl, entries in label_intervals.items():
        # Group by xcsid
        xcsid_groups: dict[int, list[dict]] = defaultdict(list)
        for e in entries:
            xcsid = e["xcsid"]
            if xcsid is not None:
                xcsid_groups[xcsid].append(e)
            else:
                # No xcsid — treat each track as its own group
                xcsid_groups[id(e)] = [e]

        xcsid_list = sorted(xcsid_groups.keys())
        for i in range(len(xcsid_list)):
            for j in range(i + 1, len(xcsid_list)):
                xa, xb = xcsid_list[i], xcsid_list[j]
                ga, gb = xcsid_groups[xa], xcsid_groups[xb]

                # Compute time ranges for each xcsid group
                sa = min(e["start"] for e in ga)
                ea = max(e["end"] for e in ga)
                sb = min(e["start"] for e in gb)
                eb = max(e["end"] for e in gb)

                if sa < eb and sb < ea:
                    overlap_start = max(sa, sb)
                    overlap_end = min(ea, eb)
                    overlap_s = (overlap_end - overlap_start).total_seconds()
                    if overlap_s <= 0:
                        continue

                    conflict_rows.append({
                        "identity_label": lbl,
                        "xcsid_a": xa,
                        "xcsid_b": xb,
                        "overlap_seconds": round(overlap_s, 1),
                        "overlap_start": str(overlap_start)[:19],
                        "overlap_end": str(overlap_end)[:19],
                    })

                    # Mark ALL tracks in BOTH conflicting xcsid groups
                    for e in ga + gb:
                        conflict_tracks.add(e["track_filename"])

    # ── 3. Apply _conflict suffix ─────────────────────────────────────────── #
    n_changed = 0
    if conflict_tracks:
        for cam_id in camera_ids:
            _, tracklets = json_data[cam_id]
            for t in tracklets:
                tf = t.get("track_filename", "")
                if tf in conflict_tracks:
                    old_label = t.get("identity_label", "")
                    if old_label and not old_label.endswith("_conflict"):
                        t["identity_label"] = f"{old_label}_conflict"
                        n_changed += 1

    # ── Log ───────────────────────────────────────────────────────────────── #
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-CAMERA IDENTITY CONFLICT VERIFICATION")
    logger.info("=" * 80)
    if conflict_rows:
        cdf = pd.DataFrame(conflict_rows)
        logger.warning(
            "  ⚠  %d identity conflicts found across cameras (%d tracks → "
            "'_conflict'):\n%s",
            len(conflict_rows), n_changed, cdf.to_string(index=False),
        )
    else:
        logger.info("  ✓  No identity conflicts across cameras.")

    return n_changed


# ─────────── ReID voting per cross_cam_stitched_id group ──────────────────── #

def vote_identity_by_xcsid_reid(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    gallery_path: str | Path,
    known_individuals: Optional[list[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> dict[int, dict]:
    """
    Merge ReID features from all tracks sharing the same
    ``cross_cam_stitched_id``, match to gallery, and vote on identity.

    Only *good-quality* frames are used (``quality_label == 'good'``,
    ``behavior_label == '01_standing'``, ``behavior_conf >= 0.7``).

    Parameters
    ----------
    tracklets_cam1, tracklets_cam2 : list[dict]
        Tracklet dicts (already enriched with ``cross_cam_stitched_id``).
    gallery_path : str | Path
        Path to the gallery ``.npz`` file (must contain ``feature`` and
        ``label`` keys).
    known_individuals : list[str] | None
        Allowed identity labels for this room.

    Returns
    -------
    dict[int, dict]
        ``cross_cam_stitched_id → voted_results`` where ``voted_results``
        contains ``voted_label``, ``confidence``, ``ranked_labels``, etc.
        Also writes ``cross_cam_stitched_id_vote`` and
        ``cross_cam_stitched_id_vote_counts`` into each tracklet dict
        (in-place).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    gallery_path = Path(gallery_path)
    if not gallery_path.exists():
        logger.error("Gallery file not found: %s", gallery_path)
        return {}

    # ── Load gallery ─────────────────────────────────────────────────────── #
    gallery_data = np.load(gallery_path, allow_pickle=True)
    gallery_features_np = gallery_data["feature"]
    gallery_labels = gallery_data["label"].tolist()
    gallery_features = torch.from_numpy(gallery_features_np).float()

    logger.info("Loaded gallery: %d features, %d unique labels from %s",
                len(gallery_features), len(set(gallery_labels)), gallery_path.name)

    # ── Group tracklets by cross_cam_stitched_id ─────────────────────────── #
    all_tracklets = tracklets_cam1 + tracklets_cam2
    xcsid_groups: dict[int, list[dict]] = defaultdict(list)
    for t in all_tracklets:
        xcsid = t.get("cross_cam_stitched_id")
        if xcsid is not None:
            xcsid_groups[xcsid].append(t)

    xcsid_results: dict[int, dict] = {}

    for xcsid, members in xcsid_groups.items():
        merged_features_list: list[np.ndarray] = []

        for t in members:
            feature_path_str = t.get("feature_path", "")
            if not feature_path_str:
                continue
            feature_path = Path(feature_path_str)
            if not feature_path.exists():
                continue

            # Load features + frame_ids from npz
            features, frame_ids, _, _ = load_embedding(feature_path)

            # Filter by good-quality frames from behavior CSV
            csv_path_str = t.get("track_csv_path", "")
            if csv_path_str:
                behavior_csv = Path(csv_path_str).with_name(
                    Path(csv_path_str).stem + "_behavior.csv"
                )
                if behavior_csv.exists():
                    df_beh = pd.read_csv(behavior_csv)
                    good_indices = df_beh.index[
                        (df_beh["quality_label"] == "good")
                        & (df_beh["behavior_label"] == "01_standing")
                        & (df_beh["behavior_conf"].astype(float) >= 0.7)
                    ].tolist()

                    if good_indices:
                        feature_indices = [
                            i for i, fid in enumerate(frame_ids)
                            if fid in good_indices
                        ]
                        if feature_indices:
                            features = features[feature_indices]
                        else:
                            continue  # no matching feature indices
                    else:
                        continue  # no good frames in this track

            if len(features) > 0:
                merged_features_list.append(features)

        if not merged_features_list:
            # No features available for this group
            logger.debug("xcsid %d: no good-quality features available "
                         "(%d tracks)", xcsid, len(members))
            continue

        # ── Merge features from all tracks in group ──────────────────────── #
        merged_features = np.concatenate(merged_features_list, axis=0)
        merged_tensor = torch.from_numpy(merged_features).float()

        logger.debug("xcsid %d: merged %d features from %d tracks",
                     xcsid, len(merged_features), len(members))

        # ── Match to gallery ─────────────────────────────────────────────── #
        matched_labels = match_to_gallery(
            merged_tensor,
            gallery_features,
            gallery_labels=gallery_labels,
        )[-1]  # last element = matched_labels list

        # ── Vote on identity ─────────────────────────────────────────────── #
        voted_results = vote_identity_from_matched_labels(
            matched_labels=np.array(matched_labels),
            known_labels=known_individuals,
            sample_rate=1,
            return_details=True,
        )

        xcsid_results[xcsid] = voted_results

        # ── Write results into each tracklet in this group (in-place) ────── #
        top1_label = voted_results.get("voted_label", "unknown")
        # ranked_labels is list of (label, count, pct) tuples
        vote_counts = {
            lbl: {"count": int(cnt), "pct": round(pct, 2)}
            for lbl, cnt, pct in voted_results.get("ranked_labels", [])
        }

        for t in members:
            t["cross_cam_stitched_id_vote"] = top1_label
            t["cross_cam_stitched_id_vote_counts"] = vote_counts

    # ── Summary ───────────────────────────────────────────────────────────── #
    n_voted = len(xcsid_results)
    n_total = len(xcsid_groups)
    label_dist = Counter(
        r.get("voted_label", "unknown") for r in xcsid_results.values()
    )
    logger.info("\n" + "=" * 80)
    logger.info("ReID VOTING PER cross_cam_stitched_id")
    logger.info("=" * 80)
    logger.info("  Total xcsid groups:  %d", n_total)
    logger.info("  Voted (had features): %d", n_voted)
    logger.info("  Skipped (no features): %d", n_total - n_voted)
    logger.info("  Label distribution:  %s", dict(label_dist))

    # Per-xcsid vote table
    rows = []
    for xcsid in sorted(xcsid_groups.keys()):
        res = xcsid_results.get(xcsid)
        if res:
            top1 = res["voted_label"]
            conf = f"{res['confidence']:.1f}%"
            ranked = ", ".join(
                f"{lbl}:{cnt}" for lbl, cnt, _ in res.get("ranked_labels", [])[:3]
            )
        else:
            top1, conf, ranked = "-", "-", "no features"
        rows.append({
            "xcsid": xcsid,
            "n_tracks": len(xcsid_groups[xcsid]),
            "voted_label": top1,
            "confidence": conf,
            "top3_votes": ranked,
        })
    vote_df = pd.DataFrame(rows)
    logger.info("\n  Vote table:\n%s", vote_df.to_string(index=False))

    return xcsid_results


def assign_identity_by_xcsid(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    known_individuals: list[str],
    logger: Optional[logging.Logger] = None,
) -> tuple[dict[str, str], pd.DataFrame]:
    """
    Assign ``identity_label`` to every track using ``cross_cam_stitched_id``
    groups and ``cross_cam_stitched_id_vote`` evidence (from merged ReID
    features), **while preventing temporal conflicts** (the same individual
    cannot be in two groups at the same time).

    Algorithm
    ---------
    1. Group all tracklets by ``cross_cam_stitched_id``.
    2. For each group, use the ``cross_cam_stitched_id_vote`` (top-1 label
       from merged ReID voting) as the primary candidate.  Fall back to
       ``cross_cam_stitched_id_vote_counts`` for ranked alternatives.
    3. Sort groups by total evidence (descending) – strongest first.
    4. Greedily assign labels.  Before assigning a label to a group, check
       whether any *previously assigned* group with the same label overlaps
       in time.  If conflict:
       a. Try the next-best label from known_individuals.
       b. If all labels conflict, mark the group ``"unknown"``.
    5. Build ``track_filename → identity_label`` mapping.

    Parameters
    ----------
    tracklets_cam1, tracklets_cam2 : list[dict]
        Tracklet dicts enriched with ``cross_cam_stitched_id`` and
        ``cross_cam_stitched_id_vote`` / ``cross_cam_stitched_id_vote_counts``.
    known_individuals : list[str]
        Allowed identity labels for this room.

    Returns
    -------
    (track_to_label, conflict_df)
        - ``track_to_label``: ``track_filename → identity_label``
        - ``conflict_df``: DataFrame logging every detected conflict
          (xcsid pair, overlap duration, resolution).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    all_tracklets = tracklets_cam1 + tracklets_cam2
    skip = {"invalid", "unknown", "spurious", ""}

    # ── 1. Group by cross_cam_stitched_id ─────────────────────────────────── #
    xcsid_groups: dict[int, list[dict]] = defaultdict(list)
    for t in all_tracklets:
        xcsid = t.get("cross_cam_stitched_id")
        if xcsid is not None:
            xcsid_groups[xcsid].append(t)

    # ── 2. Per-group: vote, time range ────────────────────────────────────── #
    group_info: dict[int, dict] = {}
    for xcsid, members in xcsid_groups.items():
        vote = Counter()
        t_start = None
        t_end = None
        for t in members:
            # Use cross_cam_stitched_id_vote_counts (from merged ReID features)
            vc = t.get("cross_cam_stitched_id_vote_counts")
            if vc and isinstance(vc, dict):
                for lbl, info in vc.items():
                    if lbl and lbl not in skip:
                        cnt = info["count"] if isinstance(info, dict) else int(info)
                        vote[lbl] += cnt
                break  # all members share the same vote_counts, read once

        for t in members:
            s = t.get("start_timestamp")
            e = t.get("end_timestamp")
            try:
                ts = pd.Timestamp(s)
                te = pd.Timestamp(e)
            except Exception:
                continue
            if t_start is None or ts < t_start:
                t_start = ts
            if t_end is None or te > t_end:
                t_end = te

        group_info[xcsid] = {
            "vote": vote,
            "total_evidence": vote.total() if vote else 0,
            "start": t_start,
            "end": t_end,
            "n_tracks": len(members),
        }

    # ── 3. Sort by evidence (strongest first) ─────────────────────────────── #
    sorted_xcsids = sorted(
        group_info.keys(),
        key=lambda x: group_info[x]["total_evidence"],
        reverse=True,
    )

    # ── 4. Greedy assignment with temporal non-overlap ────────────────────── #
    # For each label, keep a sorted list of assigned intervals
    label_intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]] = (
        defaultdict(list)
    )
    xcsid_assignment: dict[int, str] = {}
    conflict_rows: list[dict] = []
    allowed = set(known_individuals) if known_individuals else set()

    def _overlaps_any(label: str, start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
        """Return list of xcsid that overlap with (start, end) for *label*."""
        hits = []
        for (s, e, xid) in label_intervals[label]:
            if s < end and start < e:
                hits.append(xid)
        return hits

    for xcsid in sorted_xcsids:
        gi = group_info[xcsid]
        if gi["start"] is None or gi["end"] is None:
            continue

        # Build candidate list: voted labels sorted by count, then remaining
        candidates = []
        if gi["vote"]:
            for lbl, _ in gi["vote"].most_common():
                if lbl in allowed:
                    candidates.append(lbl)
        # Add remaining known_individuals not already in candidates
        for ki in known_individuals or []:
            if ki not in candidates:
                candidates.append(ki)

        assigned = False
        for lbl in candidates:
            conflicting = _overlaps_any(lbl, gi["start"], gi["end"])
            if not conflicting:
                # No conflict → assign
                xcsid_assignment[xcsid] = lbl
                label_intervals[lbl].append((gi["start"], gi["end"], xcsid))
                assigned = True
                break
            else:
                # Record conflict (but keep trying next label)
                for cxcsid in conflicting:
                    cgi = group_info[cxcsid]
                    overlap_start = max(gi["start"], cgi["start"])
                    overlap_end = min(gi["end"], cgi["end"])
                    conflict_rows.append({
                        "xcsid_a": cxcsid,
                        "xcsid_b": xcsid,
                        "label": lbl,
                        "overlap_seconds": (overlap_end - overlap_start).total_seconds(),
                        "overlap_start": str(overlap_start),
                        "overlap_end": str(overlap_end),
                        "resolution": "try_next",
                    })

        if not assigned:
            xcsid_assignment[xcsid] = "unknown"
            # Log final conflict
            if conflict_rows and conflict_rows[-1]["xcsid_b"] == xcsid:
                conflict_rows[-1]["resolution"] = "set_unknown"

    # ── 5. Build track → label mapping ────────────────────────────────────── #
    track_to_label: dict[str, str] = {}
    for xcsid, label in xcsid_assignment.items():
        for t in xcsid_groups[xcsid]:
            tf = t.get("track_filename", "")
            if tf:
                track_to_label[tf] = label

    # Also assign "unknown" to any track whose xcsid had no assignment
    for t in all_tracklets:
        tf = t.get("track_filename", "")
        if tf and tf not in track_to_label:
            track_to_label[tf] = "unknown"

    conflict_df = pd.DataFrame(conflict_rows)

    # ── Summary ───────────────────────────────────────────────────────────── #
    label_counts = Counter(xcsid_assignment.values())
    n_groups = len(xcsid_groups)
    n_assigned = sum(1 for v in xcsid_assignment.values() if v != "unknown")
    n_unknown = sum(1 for v in xcsid_assignment.values() if v == "unknown")
    n_conflicts = len(conflict_df)

    logger.info("\n" + "=" * 80)
    logger.info("IDENTITY ASSIGNMENT BY cross_cam_stitched_id")
    logger.info("=" * 80)
    logger.info("  Groups (cross_cam_stitched_id): %d", n_groups)
    logger.info("  Assigned a known label:         %d", n_assigned)
    logger.info("  Assigned 'unknown':             %d", n_unknown)
    logger.info("  Temporal conflicts detected:    %d", n_conflicts)
    logger.info("  Label distribution:             %s", dict(label_counts))

    if not conflict_df.empty:
        logger.info("\n  Conflict details:\n%s",
                    conflict_df.to_string(index=False))

    # Per-xcsid summary
    rows = []
    for xcsid in sorted(xcsid_groups.keys()):
        gi = group_info[xcsid]
        lbl = xcsid_assignment.get(xcsid, "unassigned")
        majority_vote = gi["vote"].most_common(1)[0][0] if gi["vote"] else "-"
        rows.append({
            "xcsid": xcsid,
            "assigned_label": lbl,
            "majority_vote": majority_vote,
            "evidence": gi["total_evidence"],
            "n_tracks": gi["n_tracks"],
            "start": str(gi["start"])[:19] if gi["start"] else "",
            "end": str(gi["end"])[:19] if gi["end"] else "",
        })
    assign_df = pd.DataFrame(rows)
    logger.info("\n  Assignment table:\n%s", assign_df.to_string(index=False))

    return track_to_label, conflict_df


# ──────────────── 3. Cross-camera stitched ID (Union-Find) ────────────────── #

class _UnionFind:
    """Minimal Union-Find (disjoint-set) with path compression."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def compute_cross_cam_individual(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    n_individuals: int = 2,
    logger: Optional[logging.Logger] = None,
) -> dict[str, int]:
    """Partition all tracks into *n_individuals* temporal slots.

    Tracks that belong to the same ``cross_cam_stitched_id`` but are
    observed simultaneously on two cameras are the **same** physical
    individual — they must stay together.  Tracks whose xcsid groups
    overlap in time (and are from *different* xcsid groups) **must**
    represent different individuals.

    Spurious detection filtering
    ----------------------------
    When there are already ≥ ``n_individuals`` xcsid groups that overlap
    temporally with at least one other group, any xcsid group with
    **zero** overlap is likely a false detection (not an elephant).
    These are assigned ``cross_cam_individual = -1``, which downstream
    maps to ``identity_label = 'invalid'``.

    Algorithm
    ---------
    1. Merge overlapping same-xcsid tracks into *composite intervals*.
    2. Build a weighted overlap graph between xcsid groups
       (weight = total seconds of temporal overlap).
    3. Filter spurious xcsid groups (zero overlap when enough real
       groups exist) → assign partition ``-1``.
    4. Greedy **weighted max-cut** on remaining real groups, seeded by
       the heaviest overlapping pair.
    5. Iterative local search: try flipping each xcsid; accept if it
       reduces total within-partition overlap.
    6. **Per-camera overlap validation**: within each partition, if two
       xcsid groups have tracks that overlap on the *same* camera
       (physically impossible — one camera can't track the same
       elephant twice), evict the shorter-duration group to ``-1``.
    7. Map every track to its xcsid's partition
       → ``cross_cam_individual ∈ {-1, 0, 1, …, n_individuals − 1}``.
       ``-1`` = spurious / not an elephant.

    Parameters
    ----------
    tracklets_cam1, tracklets_cam2 : list[dict]
        Tracklet dicts, already enriched with ``cross_cam_stitched_id``.
    n_individuals : int
        Expected number of distinct individuals in the room.
        Currently only ``n_individuals == 2`` is fully supported.
    logger : logging.Logger, optional

    Returns
    -------
    dict[str, int]
        ``track_filename → cross_cam_individual`` for every track that
        has a ``cross_cam_stitched_id``.  Value ``-1`` means the track
        is spurious (not a real elephant).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    all_tracklets = tracklets_cam1 + tracklets_cam2

    # ── 1. Collect per-track intervals ──────────────────────────────────── #
    intervals: list[dict] = []
    for t in all_tracklets:
        try:
            s = pd.Timestamp(t["start_timestamp"])
            e = pd.Timestamp(t["end_timestamp"])
        except (KeyError, ValueError):
            continue
        xcsid = t.get("cross_cam_stitched_id")
        if xcsid is None:
            continue
        cam = t.get("camera_id", "")
        intervals.append({
            "start": s, "end": e,
            "tf": t["track_filename"],
            "xcsid": xcsid,
            "camera": cam,
        })

    if not intervals:
        logger.warning("compute_cross_cam_individual: no valid intervals")
        return {}

    # ── 2. Build composite intervals per xcsid ──────────────────────────── #
    #    Merge overlapping tracks within the same xcsid so that cross-camera
    #    simultaneous observations count as a single "slot usage".
    xcsid_tracks: dict[int, list[dict]] = defaultdict(list)
    for iv in intervals:
        xcsid_tracks[iv["xcsid"]].append(iv)

    xcsid_composites: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for xcsid, tracks in xcsid_tracks.items():
        tracks.sort(key=lambda x: x["start"])
        merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for t in tracks:
            if merged and t["start"] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], t["end"]))
            else:
                merged.append((t["start"], t["end"]))
        xcsid_composites[xcsid] = merged

    all_xcsids = sorted(xcsid_composites.keys())
    n = len(all_xcsids)
    xcsid_to_idx = {x: i for i, x in enumerate(all_xcsids)}

    logger.info(
        "cross_cam_individual: %d tracks, %d xcsid groups, "
        "%d composite intervals",
        len(intervals), n,
        sum(len(v) for v in xcsid_composites.values()),
    )

    # ── 3. Build weighted overlap matrix ────────────────────────────────── #
    overlap_matrix = np.zeros((n, n))
    for i, xa in enumerate(all_xcsids):
        for j in range(i + 1, n):
            xb = all_xcsids[j]
            total_ov = 0.0
            for sa, ea in xcsid_composites[xa]:
                for sb, eb in xcsid_composites[xb]:
                    ov = (min(ea, eb) - max(sa, sb)).total_seconds()
                    if ov > 0:
                        total_ov += ov
            overlap_matrix[i, j] = total_ov
            overlap_matrix[j, i] = total_ov

    total_overlap = np.sum(overlap_matrix) / 2

    # ── 3b. Filter spurious xcsid groups ────────────────────────────────── #
    #   An xcsid group is considered spurious (not a real elephant) if:
    #   - It has ZERO temporal overlap with every other xcsid group, AND
    #   - There are already ≥ n_individuals groups that DO overlap with
    #     at least one other group.
    #   Spurious groups get cross_cam_individual = -1 → identity_label = 'invalid'.
    per_node_overlap = np.sum(overlap_matrix, axis=1)  # total overlap per xcsid
    overlapping_mask = per_node_overlap > 0  # groups that overlap with ≥1 other
    n_overlapping = int(np.sum(overlapping_mask))

    spurious_indices: set[int] = set()
    if n_overlapping >= n_individuals:
        # Enough real groups exist → zero-overlap groups are spurious
        for i in range(n):
            if not overlapping_mask[i]:
                spurious_indices.add(i)
        if spurious_indices:
            spurious_xcsids = [all_xcsids[i] for i in sorted(spurious_indices)]
            logger.info(
                "  ⚠  Filtering %d spurious xcsid groups (zero overlap, "
                "likely not elephants): %s",
                len(spurious_indices), spurious_xcsids,
            )

    # Indices of real (non-spurious) xcsid groups
    real_indices = sorted(set(range(n)) - spurious_indices)
    n_real = len(real_indices)

    if n_real <= n_individuals:
        # Trivial: each real xcsid is its own individual; spurious → -1
        track_to_ind: dict[str, int] = {}
        real_idx_map = {orig: new for new, orig in enumerate(real_indices)}
        for iv in intervals:
            idx = xcsid_to_idx[iv["xcsid"]]
            if idx in spurious_indices:
                track_to_ind[iv["tf"]] = -1
            else:
                track_to_ind[iv["tf"]] = real_idx_map[idx]
        logger.info("  Trivial assignment — %d real xcsid ≤ %d individuals "
                    "(%d spurious → -1)",
                    n_real, n_individuals, len(spurious_indices))
        return track_to_ind

    # ── 4. Greedy max-cut (n_individuals == 2), real groups only ─────────── #
    # Build reduced overlap matrix for real groups only
    real_overlap = overlap_matrix[np.ix_(real_indices, real_indices)]

    max_i, max_j = np.unravel_index(
        np.argmax(real_overlap), real_overlap.shape)
    partition_real = np.full(n_real, -1, dtype=int)
    partition_real[max_i] = 0
    partition_real[max_j] = 1

    # Add remaining nodes sorted by total overlap weight (most constrained
    # first) so the initial greedy pass is as informed as possible.
    remaining = sorted(
        (i for i in range(n_real) if partition_real[i] < 0),
        key=lambda i: -np.sum(real_overlap[i]),
    )

    for idx in remaining:
        ov0 = np.dot(real_overlap[idx], (partition_real == 0).astype(float))
        ov1 = np.dot(real_overlap[idx], (partition_real == 1).astype(float))
        partition_real[idx] = 0 if ov0 <= ov1 else 1

    # ── 5. Local search (single-node flip) ──────────────────────────────── #
    for iteration in range(50):  # bounded
        improved = False
        for i in range(n_real):
            old_p = partition_real[i]
            new_p = 1 - old_p
            same_old = np.dot(real_overlap[i],
                              (partition_real == old_p).astype(float))
            same_new = np.dot(real_overlap[i],
                              (partition_real == new_p).astype(float))
            # Flipping removes same_old from within, adds same_new
            if same_new < same_old:
                partition_real[i] = new_p
                improved = True
        if not improved:
            break

    # ── 6. Quality metrics ──────────────────────────────────────────────── #
    within = 0.0
    cross = 0.0
    for i in range(n_real):
        for j in range(i + 1, n_real):
            w = real_overlap[i, j]
            if w == 0:
                continue
            if partition_real[i] == partition_real[j]:
                within += w
            else:
                cross += w

    if total_overlap > 0:
        logger.info(
            "  Max-cut quality: cross-partition %.1f%% (%d s), "
            "within-partition %.1f%% (%d s)",
            cross / total_overlap * 100, int(cross),
            within / total_overlap * 100, int(within),
        )
    else:
        logger.info("  No temporal overlaps between xcsid groups.")

    # ── 7. Map individual tracks → partition (spurious → -1) ────────────── #
    # Build full partition array: real groups get 0/1, spurious get -1
    partition = np.full(n, -1, dtype=int)
    for ri, orig_idx in enumerate(real_indices):
        partition[orig_idx] = partition_real[ri]

    # ── 8. Per-camera overlap validation ────────────────────────────────── #
    #   Within each partition, if two different xcsid groups have tracks
    #   that overlap on the SAME camera, one of them is a spurious
    #   detection (same camera can't see the same elephant twice).
    #   Keep the xcsid group with the longer total duration; mark the
    #   shorter one as spurious (-1).
    #
    #   Build per-camera track intervals per xcsid for this check.
    xcsid_per_cam: dict[int, dict[str, list[tuple]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for iv in intervals:
        xcsid_per_cam[iv["xcsid"]][iv["camera"]].append(
            (iv["start"], iv["end"])
        )

    # Total duration per xcsid (used to decide which survives)
    xcsid_duration: dict[int, float] = {}
    for xcsid, composites in xcsid_composites.items():
        xcsid_duration[xcsid] = sum(
            (e - s).total_seconds() for s, e in composites
        )

    for p_val in range(n_individuals):
        # Collect xcsid indices in this partition
        part_indices = [i for i in range(n) if partition[i] == p_val]
        if len(part_indices) < 2:
            continue

        # For each pair, check same-camera track overlap
        evicted: set[int] = set()
        for ii in range(len(part_indices)):
            idx_a = part_indices[ii]
            if idx_a in evicted:
                continue
            xa = all_xcsids[idx_a]
            for jj in range(ii + 1, len(part_indices)):
                idx_b = part_indices[jj]
                if idx_b in evicted:
                    continue
                xb = all_xcsids[idx_b]

                # Check same-camera overlap between xa and xb
                same_cam_overlap = 0.0
                for cam in set(xcsid_per_cam[xa].keys()) & set(xcsid_per_cam[xb].keys()):
                    for sa, ea in xcsid_per_cam[xa][cam]:
                        for sb, eb in xcsid_per_cam[xb][cam]:
                            ov = (min(ea, eb) - max(sa, sb)).total_seconds()
                            if ov > 0:
                                same_cam_overlap += ov

                if same_cam_overlap > 0:
                    # Conflict! Evict the shorter-duration xcsid
                    dur_a = xcsid_duration.get(xa, 0)
                    dur_b = xcsid_duration.get(xb, 0)
                    loser_idx = idx_b if dur_a >= dur_b else idx_a
                    loser_xcsid = all_xcsids[loser_idx]
                    winner_xcsid = xa if loser_idx == idx_b else xb
                    partition[loser_idx] = -1
                    evicted.add(loser_idx)
                    logger.info(
                        "  ⚠  Same-camera overlap in partition %d: "
                        "xcsid=%d (%.0fs) ↔ xcsid=%d (%.0fs) → "
                        "evicting xcsid=%d (shorter, %.1f min overlap)",
                        p_val, xa, dur_a, xb, dur_b,
                        loser_xcsid, same_cam_overlap / 60,
                    )

    track_to_ind: dict[str, int] = {}
    for iv in intervals:
        track_to_ind[iv["tf"]] = int(partition[xcsid_to_idx[iv["xcsid"]]])

    p_counts = Counter(track_to_ind.values())
    logger.info(
        "  Partition sizes: %s",
        ", ".join(f"slot {k}: {v} tracks" for k, v in sorted(p_counts.items())),
    )
    if -1 in p_counts:
        logger.info("  ⚠  %d tracks assigned to slot -1 (spurious / invalid)",
                    p_counts[-1])

    return track_to_ind


def compute_cross_cam_stitched_id(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    cam1_id: str,
    cam2_id: str,
    logger: Optional[logging.Logger] = None,
) -> dict[str, int]:
    """
    Merge ``stitched_id`` (per-camera) and ``cross_cam_id`` into a
    single unified ``cross_cam_stitched_id`` using a Union-Find.

    Edges
    -----
    1. **Same stitched_id on the same camera** → union those tracks.
    2. **Same cross_cam_id** → union those tracks (bridges cameras).

    Transitively, all connected tracks receive the **same**
    ``cross_cam_stitched_id`` (an integer starting from 0).

    Parameters
    ----------
    tracklets_cam1, tracklets_cam2 : list[dict]
        Tracklet dicts (already enriched with ``cross_cam_id``).
    cam1_id, cam2_id : str
        Camera identifiers (used only for logging/namespacing).

    Returns
    -------
    dict[str, int]
        ``track_filename → cross_cam_stitched_id`` for **every** track.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    uf = _UnionFind()

    # ── 1. Union tracks that share a stitched_id on the SAME camera ─────── #
    for cam_id, tracklets in [(cam1_id, tracklets_cam1),
                              (cam2_id, tracklets_cam2)]:
        sid_to_tracks: dict = defaultdict(list)
        for t in tracklets:
            tf = t.get("track_filename", "")
            sid = t.get("stitched_id")
            if tf and sid is not None:
                sid_to_tracks[(cam_id, sid)].append(tf)
        for key, tracks in sid_to_tracks.items():
            for i in range(1, len(tracks)):
                uf.union(tracks[0], tracks[i])

    # ── 2. Union tracks that share a cross_cam_id (across cameras) ──────── #
    xcid_to_tracks: dict[int, list[str]] = defaultdict(list)
    for t in tracklets_cam1 + tracklets_cam2:
        tf = t.get("track_filename", "")
        xcid = t.get("cross_cam_id")
        if tf and xcid is not None:
            xcid_to_tracks[xcid].append(tf)

    for xcid, tracks in xcid_to_tracks.items():
        for i in range(1, len(tracks)):
            uf.union(tracks[0], tracks[i])

    # ── 3. Assign contiguous cross_cam_stitched_id per component ────────── #
    all_tracks = []
    for t in tracklets_cam1 + tracklets_cam2:
        tf = t.get("track_filename", "")
        if tf:
            all_tracks.append(tf)

    root_to_id: dict[str, int] = {}
    track_to_xcsid: dict[str, int] = {}
    next_id = 0

    for tf in sorted(set(all_tracks)):  # sorted for deterministic IDs
        root = uf.find(tf)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        track_to_xcsid[tf] = root_to_id[root]

    logger.info(
        "cross_cam_stitched_id: %d tracks → %d unified IDs  "
        "(cam %s: %d tracks, cam %s: %d tracks)",
        len(track_to_xcsid), next_id,
        cam1_id, len(tracklets_cam1),
        cam2_id, len(tracklets_cam2),
    )

    return track_to_xcsid


# ──────────── 5. Cross-camera behaviour smoothing ─────────────────────────── #

def smooth_behavior_cross_camera(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    min_bout_seconds: float = 60.0,
    logger: Optional[logging.Logger] = None,
) -> dict[str, int]:
    """Align behaviour labels across cameras for the same individual.

    Four-stage smoothing:

    **Stage 1 — Cross-camera correction** (per matched pair):
        For every pair of tracks sharing ``cross_cam_id``, load both
        behaviour CSVs and find overlapping time windows.  Bout lengths
        are computed **per-camera on the original CSV** (not on merged
        data) so that a 10,000-frame continuous sleeping_right on cam A
        keeps its full weight even when cam B is fragmented.  The camera
        with the longer bout always wins.

    **Stage 2 — Majority-vote temporal filter** (per CSV):
        For each frame, look at a ±30 s window and replace the frame's
        label with the majority non-invalid label in that window.  This
        flattens rapid alternation (e.g. ``sleeping_left`` ↔
        ``sleeping_right`` every 1–5 frames) in a single pass.

    **Stage 3 — Within-track sandwich smoothing** (per CSV):
        Any remaining short bout (< ``min_bout_seconds``) that is
        sandwiched between two bouts of the same label is merged.  This
        catches isolated outliers that the majority window didn't fully
        resolve.

    **Stage 4 — Short bout elimination with context** (per CSV):
        Any bout shorter than 30 s whose label disagrees with the
        dominant behaviour in a ±5 min context window is replaced.
        This eliminates dense micro-bout clusters (e.g. hundreds of
        1–3 frame ``sleeping_left`` scattered in a long
        ``sleeping_right`` region) that survived the 30 s majority vote.

    Only ``behavior_label`` is modified; ``behavior_label_raw`` is
    preserved as the original prediction.

    Parameters
    ----------
    tracklets_cam1, tracklets_cam2 : list[dict]
        Tracklet dicts already enriched with ``cross_cam_id``.
    min_bout_seconds : float
        Minimum bout duration in seconds.  Shorter bouts sandwiched by
        the same label are merged in Stage 2.
    logger : logging.Logger, optional

    Returns
    -------
    dict[str, int]
        ``track_filename → n_corrected_frames`` for every track whose
        behaviour CSV was modified.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # ── 0. Snapshot behavior_label → behavior_label_old ───────────────── #
    #    Preserve the current (pre-smoothing) labels so that downstream
    #    consumers can compare before/after.  Smoothing then modifies
    #    behavior_label in-place; behavior_label_raw stays untouched.
    all_beh_csvs: set[Path] = set()
    for t in tracklets_cam1 + tracklets_cam2:
        csv_p_str = t.get("track_csv_path", "")
        if not csv_p_str:
            continue
        beh_p = Path(csv_p_str).with_name(
            Path(csv_p_str).stem + "_behavior.csv"
        )
        if beh_p.exists():
            all_beh_csvs.add(beh_p)

    n_snapshot = 0
    for beh_csv in all_beh_csvs:
        df_tmp = pd.read_csv(beh_csv)
        if "behavior_label" in df_tmp.columns:
            df_tmp["behavior_label_old"] = df_tmp["behavior_label"]
            df_tmp.to_csv(beh_csv, index=False)
            n_snapshot += 1
    logger.info("Snapshot behavior_label → behavior_label_old on %d CSVs",
                n_snapshot)

    # ── 1. Group tracks by cross_cam_id → find cross-camera pairs ────────── #
    xcid_to_tracks: dict[int, list[dict]] = defaultdict(list)
    for t in tracklets_cam1 + tracklets_cam2:
        xcid = t.get("cross_cam_id")
        tf = t.get("track_filename", "")
        csv = t.get("track_csv_path", "")
        if xcid is not None and tf and csv:
            xcid_to_tracks[xcid].append(t)

    cam1_filenames = {t.get("track_filename") for t in tracklets_cam1}
    cam2_filenames = {t.get("track_filename") for t in tracklets_cam2}

    total_xcam_corrections = 0
    total_pairs = 0
    track_corrections: dict[str, int] = {}

    # Collect all behaviour CSVs that were touched
    modified_beh_csvs: set[Path] = set()

    for xcid, members in xcid_to_tracks.items():
        cam1_members = [m for m in members if m["track_filename"] in cam1_filenames]
        cam2_members = [m for m in members if m["track_filename"] in cam2_filenames]

        if not cam1_members or not cam2_members:
            continue  # no cross-camera pair

        for t1 in cam1_members:
            for t2 in cam2_members:
                n_fixed = _smooth_behavior_pair(t1, t2, logger)
                total_pairs += 1
                total_xcam_corrections += n_fixed

                # Track which CSVs exist
                for t in (t1, t2):
                    csv_p = Path(t["track_csv_path"])
                    beh_p = csv_p.with_name(csv_p.stem + "_behavior.csv")
                    if beh_p.exists():
                        all_beh_csvs.add(beh_p)
                        if n_fixed > 0:
                            modified_beh_csvs.add(beh_p)

    # ── Stage 2 — Majority-vote temporal filter ─────────────────────────── #
    #    For each frame, look at a ±window_seconds neighbourhood and
    #    replace the frame's label with the majority label in that window.
    #    This naturally flattens rapid alternation (e.g. sleeping_left ↔
    #    sleeping_right at 1–5 frame granularity).
    total_majority_corrections = 0

    for beh_csv in all_beh_csvs:
        n_maj = _smooth_majority_vote(
            beh_csv, window_seconds=30.0, logger=logger,
        )
        total_majority_corrections += n_maj
        if n_maj > 0:
            tf = beh_csv.stem.replace("_behavior", "")
            track_corrections[tf] = track_corrections.get(tf, 0) + n_maj

    # ── Stage 3 — Within-track sandwich smoothing ─────────────────────────── #
    #    After majority-vote, any remaining short bout sandwiched by the
    #    same label is merged.  This catches isolated outliers that the
    #    majority window didn't fully resolve.
    total_within_corrections = 0

    for beh_csv in all_beh_csvs:
        n_within = _smooth_within_track(
            beh_csv, min_bout_seconds=min_bout_seconds, logger=logger,
        )
        total_within_corrections += n_within
        if n_within > 0:
            tf = beh_csv.stem.replace("_behavior", "")
            track_corrections[tf] = track_corrections.get(tf, 0) + n_within

    # ── Stage 4 — Short bout elimination with wide context ────────────────── #
    #    Any bout shorter than {max_short_bout_seconds} s that disagrees with the dominant
    #    behaviour in a ±{context_window_seconds // 60} min context window is replaced.  This catches
    #    dense micro-bout clusters that survived the {max_short_bout_seconds} s majority vote.
    total_shortbout_corrections = 0

    for beh_csv in all_beh_csvs:
        n_short = _smooth_short_bouts(
            beh_csv,
            max_short_bout_seconds=300.0,   # catch bouts up to 5 min
            context_window_seconds=1800.0,  # ±30 min context
            logger=logger,
        )
        total_shortbout_corrections += n_short
        if n_short > 0:
            tf = beh_csv.stem.replace("_behavior", "")
            track_corrections[tf] = track_corrections.get(tf, 0) + n_short

    # ── Summary ───────────────────────────────────────────────────────────── #
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-CAMERA BEHAVIOUR SMOOTHING")
    logger.info("=" * 80)
    logger.info("  Matched pairs processed:      %d", total_pairs)
    logger.info("  Stage 1 (cross-cam) corrected: %d frames", total_xcam_corrections)
    logger.info("  Stage 2 (majority-vote) corrected: %d frames",
                total_majority_corrections)
    logger.info("  Stage 3 (within-track) corrected: %d frames",
                total_within_corrections)
    logger.info("  Stage 4 (short-bout elim) corrected: %d frames",
                total_shortbout_corrections)
    logger.info("  Total tracks modified:         %d", len(track_corrections))
    if track_corrections:
        top10 = sorted(track_corrections.items(), key=lambda x: -x[1])[:10]
        for tf, n in top10:
            logger.info("    %s → %d frames corrected", tf, n)

    # ── Behaviour bout summary ────────────────────────────────────────────── #
    all_tracklets = tracklets_cam1 + tracklets_cam2
    bout_summary_df = summarize_behavior_bouts(
        tracklets=all_tracklets,
        min_bout_seconds=0.0,
        logger=logger,
    )

    return track_corrections


# --------------------------------------------------------------------------- #
#  Stage 1 helpers – cross-camera pairwise smoothing                          #
# --------------------------------------------------------------------------- #

def _smooth_behavior_pair(
    t1: dict,
    t2: dict,
    logger: logging.Logger,
) -> int:
    """Smooth behaviour between two cross-camera matched tracks.

    Bout lengths are computed **per-camera on the original CSV** so that
    a long continuous sleeping_right on one camera keeps its full weight
    even when the other camera is heavily fragmented.

    Uses ``merge_asof`` (nearest-timestamp within 200 ms) instead of an
    exact inner join to avoid cartesian-product explosions from duplicate
    timestamps that appear when multiple detections share the same frame.

    Returns the total number of frames corrected (across both CSVs).
    """
    csv1 = Path(t1["track_csv_path"])
    csv2 = Path(t2["track_csv_path"])
    beh_csv1 = csv1.with_name(csv1.stem + "_behavior.csv")
    beh_csv2 = csv2.with_name(csv2.stem + "_behavior.csv")

    if not beh_csv1.exists() or not beh_csv2.exists():
        return 0

    df1 = pd.read_csv(beh_csv1)
    df2 = pd.read_csv(beh_csv2)

    if "timestamp" not in df1.columns or "timestamp" not in df2.columns:
        return 0
    if "behavior_label" not in df1.columns or "behavior_label" not in df2.columns:
        return 0

    df1["_ts"] = pd.to_datetime(df1["timestamp"])
    df2["_ts"] = pd.to_datetime(df2["timestamp"])

    # ── Compute bout lengths on the ORIGINAL per-camera data ─────────────── #
    bout_orig1 = _compute_bout_lengths(df1["behavior_label"].values)
    bout_orig2 = _compute_bout_lengths(df2["behavior_label"].values)
    df1["_bout"] = bout_orig1
    df2["_bout"] = bout_orig2
    df1["_orig_idx"] = df1.index
    df2["_orig_idx"] = df2.index

    # ── Deduplicate by timestamp (keep first occurrence) ─────────────────── #
    #    Multiple detections per frame produce duplicate timestamps which
    #    cause cartesian-product explosions in joins.
    df1_dedup = df1.drop_duplicates(subset="timestamp", keep="first") \
                    .sort_values("_ts").reset_index(drop=True)
    df2_dedup = df2.drop_duplicates(subset="timestamp", keep="first") \
                    .sort_values("_ts").reset_index(drop=True)

    # ── Align by nearest timestamp (merge_asof, 200 ms tolerance) ────────── #
    #    Direction 1: for each cam1 frame find nearest cam2 frame
    merged_1to2 = pd.merge_asof(
        df1_dedup[["_ts", "behavior_label", "behavior_conf", "_bout", "_orig_idx"]],
        df2_dedup[["_ts", "behavior_label", "behavior_conf", "_bout", "_orig_idx"]],
        on="_ts",
        tolerance=pd.Timedelta("200ms"),
        direction="nearest",
        suffixes=("_1", "_2"),
    )
    #    Direction 2: for each cam2 frame find nearest cam1 frame
    merged_2to1 = pd.merge_asof(
        df2_dedup[["_ts", "behavior_label", "behavior_conf", "_bout", "_orig_idx"]],
        df1_dedup[["_ts", "behavior_label", "behavior_conf", "_bout", "_orig_idx"]],
        on="_ts",
        tolerance=pd.Timedelta("200ms"),
        direction="nearest",
        suffixes=("_2", "_1"),
    )

    # ── Collect corrections ───────────────────────────────────────────────── #
    fix1: dict[int, str] = {}   # orig_idx → corrected label for df1
    fix2: dict[int, str] = {}   # orig_idx → corrected label for df2

    def _process_merged(merged_df: pd.DataFrame,
                        lbl_col_self: str, bout_col_self: str,
                        conf_col_self: str, idx_col_self: str,
                        lbl_col_other: str, bout_col_other: str,
                        conf_col_other: str,
                        fix_self: dict) -> None:
        """For each disagreeing row in *merged_df*, overwrite self's label
        if other camera has a longer bout (= more stable evidence)."""
        has_match = merged_df[lbl_col_other].notna()
        disagree = has_match & (merged_df[lbl_col_self] != merged_df[lbl_col_other])

        for i in merged_df.index[disagree]:
            l_self = merged_df.at[i, lbl_col_self]
            l_other = merged_df.at[i, lbl_col_other]

            # Never propagate invalid
            if l_self == "00_invalid" or l_other == "00_invalid":
                continue

            b_self = int(merged_df.at[i, bout_col_self])
            b_other = int(merged_df.at[i, bout_col_other])
            c_self = float(merged_df.at[i, conf_col_self])
            c_other = float(merged_df.at[i, conf_col_other])

            # Other camera wins if its bout is longer (or tie + higher conf)
            if b_other > b_self or (b_other == b_self and c_other > c_self):
                orig_idx = int(merged_df.at[i, idx_col_self])
                fix_self[orig_idx] = l_other

    _process_merged(
        merged_1to2,
        "behavior_label_1", "_bout_1", "behavior_conf_1", "_orig_idx_1",
        "behavior_label_2", "_bout_2", "behavior_conf_2",
        fix1,
    )
    _process_merged(
        merged_2to1,
        "behavior_label_2", "_bout_2", "behavior_conf_2", "_orig_idx_2",
        "behavior_label_1", "_bout_1", "behavior_conf_1",
        fix2,
    )

    if not fix1 and not fix2:
        return 0

    # ── Apply fixes — also propagate to duplicate-timestamp rows ──────────── #
    n_fixed_1 = _apply_fixes(df1, fix1, beh_csv1)
    n_fixed_2 = _apply_fixes(df2, fix2, beh_csv2)

    n_total = n_fixed_1 + n_fixed_2

    if n_total > 0:
        logger.debug(
            "  xcid pair %s ↔ %s: cam1 %d rows, cam2 %d rows, "
            "corrected %d (cam1→%d, cam2→%d)",
            t1["track_filename"], t2["track_filename"],
            len(df1_dedup), len(df2_dedup),
            n_total, n_fixed_1, n_fixed_2,
        )

    return n_total


def _apply_fixes(
    df: pd.DataFrame,
    fixes: dict[int, str],
    csv_path: Path,
) -> int:
    """Apply corrections to *df* and save.  Also propagate fixes to rows
    with duplicate timestamps (same frame, different detection)."""
    if not fixes:
        return 0

    # Build timestamp → new_label for propagation to duplicates
    ts_to_label: dict[str, str] = {}
    for idx, new_lbl in fixes.items():
        ts_to_label[df.at[idx, "timestamp"]] = new_lbl

    n_fixed = 0
    for idx in df.index:
        ts = df.at[idx, "timestamp"]
        if ts in ts_to_label:
            df.at[idx, "behavior_label"] = ts_to_label[ts]
            n_fixed += 1

    if n_fixed > 0:
        cols_to_drop = [c for c in df.columns if c.startswith("_")]
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        df.to_csv(csv_path, index=False)

    return n_fixed


# --------------------------------------------------------------------------- #
#  Stage 2 helper – majority-vote temporal smoothing                          #
# --------------------------------------------------------------------------- #

def _smooth_majority_vote(
    beh_csv: Path,
    window_seconds: float = 30.0,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Replace each frame's label with the majority label in a ±window.

    For each row *i*, collect all rows whose timestamp falls within
    ``[ts_i − window_seconds, ts_i + window_seconds]`` and count the
    occurrence of each non-invalid label.  If the majority label is
    different from the current label (and not ``00_invalid``), replace it.

    This naturally flattens rapid alternation (e.g. ``sleeping_left`` ↔
    ``sleeping_right`` every 1–5 frames) in a single pass.

    Parameters
    ----------
    beh_csv : Path
        Path to the behaviour CSV (modified in-place).
    window_seconds : float
        Half-width of the majority-vote window in seconds.
    logger : logging.Logger, optional

    Returns
    -------
    int
        Number of frames whose label was corrected.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    df = pd.read_csv(beh_csv)
    if "behavior_label" not in df.columns or "timestamp" not in df.columns:
        return 0

    ts = pd.to_datetime(df["timestamp"])
    labels = df["behavior_label"].values.copy()
    n = len(labels)
    if n < 3:
        return 0

    # Convert timestamps to float seconds for fast windowing
    ts_sec = (ts - ts.iloc[0]).dt.total_seconds().values

    window = window_seconds
    total_fixed = 0

    # Iterate until stable (majority vote can oscillate at boundaries)
    for _pass in range(10):
        new_labels = labels.copy()

        # Use two-pointer approach for sliding window
        left = 0
        right = 0
        n_fixed = 0

        for i in range(n):
            t_i = ts_sec[i]

            # Advance left/right pointers
            while left < n and ts_sec[left] < t_i - window:
                left += 1
            while right < n and ts_sec[right] <= t_i + window:
                right += 1

            # Count labels in [left, right)
            # Skip 00_invalid in counting
            counts: dict[str, int] = {}
            for j in range(left, right):
                lbl = labels[j]
                if lbl != "00_invalid":
                    counts[lbl] = counts.get(lbl, 0) + 1

            if not counts:
                continue

            # Find majority
            majority_lbl = max(counts, key=counts.get)  # type: ignore[arg-type]

            current = labels[i]
            if current != majority_lbl:
                new_labels[i] = majority_lbl
                n_fixed += 1

        labels = new_labels
        total_fixed += n_fixed
        if n_fixed == 0:
            break

    if total_fixed > 0:
        df["behavior_label"] = labels
        df.to_csv(beh_csv, index=False)
        logger.debug("  Majority-vote smooth %s: %d frames corrected (window=%.0fs, %d passes)",
                     beh_csv.name, total_fixed, window_seconds, _pass + 1)

    return total_fixed


# --------------------------------------------------------------------------- #
#  Stage 3 helper – within-track temporal smoothing                           #
# --------------------------------------------------------------------------- #

def _smooth_within_track(
    beh_csv: Path,
    min_bout_seconds: float = 60.0,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Remove short noisy bouts from a single behaviour CSV.

    A bout shorter than *min_bout_seconds* that is sandwiched between
    two bouts of the **same** label is replaced with that label.

    Runs iteratively until no more short bouts can be merged (a single
    pass may expose new short bouts after merging).

    Returns the number of frames corrected.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    df = pd.read_csv(beh_csv)
    if "behavior_label" not in df.columns or "timestamp" not in df.columns:
        return 0

    ts = pd.to_datetime(df["timestamp"])
    labels = df["behavior_label"].values.copy()
    n = len(labels)
    if n < 3:
        return 0

    total_fixed = 0

    for _iteration in range(20):  # bounded iterations
        bouts = _get_bouts(labels, ts)
        fixed_this_pass = 0

        for bi in range(1, len(bouts) - 1):  # skip first and last
            bout_start, bout_end, bout_lbl, bout_dur = bouts[bi]
            prev_lbl = bouts[bi - 1][2]
            next_lbl = bouts[bi + 1][2]

            # Merge if neighbours have the same valid label AND this
            # bout is short.  00_invalid bouts are also eligible for
            # replacement — they represent momentary detection drops,
            # not real behavioural transitions.
            if (prev_lbl == next_lbl
                    and bout_lbl != prev_lbl
                    and bout_dur < min_bout_seconds
                    and prev_lbl != "00_invalid"):
                labels[bout_start:bout_end] = prev_lbl
                fixed_this_pass += (bout_end - bout_start)

        total_fixed += fixed_this_pass
        if fixed_this_pass == 0:
            break

    if total_fixed > 0:
        df["behavior_label"] = labels
        df.to_csv(beh_csv, index=False)
        logger.debug("  Within-track smooth %s: %d frames corrected",
                     beh_csv.name, total_fixed)

    return total_fixed


# --------------------------------------------------------------------------- #
#  Stage 4 helper – short bout elimination with wide context                  #
# --------------------------------------------------------------------------- #

def _smooth_short_bouts(
    beh_csv: Path,
    max_short_bout_seconds: float = 30.0,
    context_window_seconds: float = 300.0,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Eliminate short bouts that disagree with the dominant local behaviour.

    For each bout shorter than *max_short_bout_seconds*, compute the
    dominant (most frequent) non-invalid label within a ±*context_window_seconds*
    window centred on the bout.  If the bout's label differs from that
    dominant label, replace the bout with the dominant label.

    This catches patterns that survive majority-vote and sandwich
    smoothing: dense micro-bout clusters (e.g. hundreds of 1–3 frame
    ``sleeping_left`` bouts scattered in a long ``sleeping_right``
    region) where the local 30 s majority-vote was insufficient because
    the noise density was too high within the vote window.

    Runs iteratively (up to 20 passes) until convergence.

    Parameters
    ----------
    beh_csv : Path
        Path to the behaviour CSV (modified in-place).
    max_short_bout_seconds : float
        Maximum duration of a bout to be considered "short" (candidates
        for replacement).
    context_window_seconds : float
        Half-width of the context window (in seconds) used to determine
        the dominant behaviour around a short bout.
    logger : logging.Logger, optional

    Returns
    -------
    int
        Number of frames whose label was corrected.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    df = pd.read_csv(beh_csv)
    if "behavior_label" not in df.columns or "timestamp" not in df.columns:
        return 0

    ts = pd.to_datetime(df["timestamp"])
    labels = df["behavior_label"].values.copy()
    n = len(labels)
    if n < 3:
        return 0

    ts_sec = (ts - ts.iloc[0]).dt.total_seconds().values
    total_fixed = 0

    for _iteration in range(20):
        bouts = _get_bouts(labels, ts)
        fixed_this_pass = 0

        for bi, (b_start, b_end, b_lbl, b_dur) in enumerate(bouts):
            # Skip long bouts; 00_invalid IS eligible for replacement
            # (short invalid bouts are detection artifacts, not real states)
            if b_dur >= max_short_bout_seconds:
                continue

            # Determine the time range of the context window
            bout_mid_time = (ts_sec[b_start] + ts_sec[b_end - 1]) / 2.0
            ctx_lo = bout_mid_time - context_window_seconds
            ctx_hi = bout_mid_time + context_window_seconds

            # Count label durations (seconds) in the context window,
            # excluding the current bout itself
            dur_by_label: dict[str, float] = {}
            for bj, (s, e, lbl, dur) in enumerate(bouts):
                if bj == bi or lbl == "00_invalid":
                    continue
                # Compute overlap of bout [ts_sec[s], ts_sec[e-1]] with
                # the context window [ctx_lo, ctx_hi]
                bout_t0 = ts_sec[s]
                bout_t1 = ts_sec[e - 1]
                overlap_lo = max(bout_t0, ctx_lo)
                overlap_hi = min(bout_t1, ctx_hi)
                if overlap_lo <= overlap_hi:
                    overlap_dur = overlap_hi - overlap_lo
                    # For single-frame bouts, overlap_dur can be 0; count
                    # at least the number of frames * avg_frame_time
                    if overlap_dur < 0.01:
                        overlap_dur = max(0.3, dur) if dur > 0 else 0.3
                    dur_by_label[lbl] = dur_by_label.get(lbl, 0.0) + overlap_dur

            if not dur_by_label:
                continue

            dominant_lbl = max(dur_by_label, key=dur_by_label.get)  # type: ignore[arg-type]
            dominant_dur = dur_by_label[dominant_lbl]
            total_dur = sum(dur_by_label.values())

            # Only replace if the dominant label covers > 60% of context
            # and the bout label is different
            if b_lbl != dominant_lbl and dominant_dur > 0.6 * total_dur:
                labels[b_start:b_end] = dominant_lbl
                fixed_this_pass += (b_end - b_start)

        total_fixed += fixed_this_pass
        if fixed_this_pass == 0:
            break

    if total_fixed > 0:
        df["behavior_label"] = labels
        df.to_csv(beh_csv, index=False)
        logger.debug("  Short-bout smooth %s: %d frames corrected "
                     "(max_bout=%.0fs, context=%.0fs)",
                     beh_csv.name, total_fixed, max_short_bout_seconds,
                     context_window_seconds)

    return total_fixed


def _get_bouts(
    labels: np.ndarray,
    timestamps: pd.Series,
) -> list[tuple[int, int, str, float]]:
    """Return list of (start_idx, end_idx, label, duration_seconds)."""
    n = len(labels)
    if n == 0:
        return []

    bouts: list[tuple[int, int, str, float]] = []
    run_start = 0
    for i in range(1, n):
        if labels[i] != labels[run_start]:
            dur = (timestamps.iloc[i - 1] - timestamps.iloc[run_start]).total_seconds()
            bouts.append((run_start, i, labels[run_start], dur))
            run_start = i
    # last bout
    dur = (timestamps.iloc[n - 1] - timestamps.iloc[run_start]).total_seconds()
    bouts.append((run_start, n, labels[run_start], dur))
    return bouts


def _compute_bout_lengths(labels: np.ndarray) -> np.ndarray:
    """For each element, compute the length of its enclosing run (bout).

    Example: ``['A','A','A','B','B','A'] → [3,3,3,2,2,1]``
    """
    n = len(labels)
    bout_len = np.ones(n, dtype=int)

    if n == 0:
        return bout_len

    # Forward pass: find run boundaries and lengths
    run_starts: list[int] = [0]
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            run_starts.append(i)
    run_starts.append(n)  # sentinel

    for r in range(len(run_starts) - 1):
        s = run_starts[r]
        e = run_starts[r + 1]
        bout_len[s:e] = e - s

    return bout_len


# ──────────────────── 6. Behaviour bout summary ────────────────────────────── #

def summarize_behavior_bouts(
    tracklets: list[dict],
    min_bout_seconds: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Summarise smoothed behaviour bouts across all tracks.

    Scans every behaviour CSV referenced by *tracklets*, computes the
    run-length-encoded bouts from the (smoothed) ``behavior_label``
    column, and returns a tidy DataFrame — one row per bout.

    This makes it easy to spot residual un-smoothed noise (short bouts,
    rapid alternation) and to review long-term sleeping patterns at a
    glance.

    Parameters
    ----------
    tracklets : list[dict]
        Tracklet dicts (must contain ``track_csv_path``, ``camera_id``,
        ``identity_label``, ``track_filename``).
    min_bout_seconds : float
        Only include bouts longer than this (default 0 = include all).
    logger : logging.Logger, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``start_time``, ``end_time``, ``duration_s``,
        ``behavior_label``, ``n_frames``, ``cam_id``,
        ``identity_label``, ``track_filename``.
        Sorted by ``cam_id``, ``start_time``.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    rows: list[dict] = []
    seen_csvs: set[Path] = set()

    for t in tracklets:
        csv_p_str = t.get("track_csv_path", "")
        if not csv_p_str:
            continue
        beh_csv = Path(csv_p_str).with_name(
            Path(csv_p_str).stem + "_behavior.csv"
        )
        if not beh_csv.exists() or beh_csv in seen_csvs:
            continue
        seen_csvs.add(beh_csv)

        cam_id = t.get("camera_id", "")
        identity = t.get("identity_label", "")
        track_fn = t.get("track_filename", "")

        df = pd.read_csv(beh_csv)
        if "behavior_label" not in df.columns or "timestamp" not in df.columns:
            continue

        ts = pd.to_datetime(df["timestamp"])
        labels = df["behavior_label"].values
        n = len(labels)
        if n == 0:
            continue

        # Run-length encoding
        run_start = 0
        for i in range(1, n):
            if labels[i] != labels[run_start]:
                dur = (ts.iloc[i - 1] - ts.iloc[run_start]).total_seconds()
                if dur >= min_bout_seconds:
                    rows.append({
                        "start_time": ts.iloc[run_start],
                        "end_time": ts.iloc[i - 1],
                        "duration_s": round(dur, 1),
                        "behavior_label": labels[run_start],
                        "n_frames": i - run_start,
                        "cam_id": cam_id,
                        "identity_label": identity,
                        "track_filename": track_fn,
                    })
                run_start = i
        # Last bout
        dur = (ts.iloc[n - 1] - ts.iloc[run_start]).total_seconds()
        if dur >= min_bout_seconds:
            rows.append({
                "start_time": ts.iloc[run_start],
                "end_time": ts.iloc[n - 1],
                "duration_s": round(dur, 1),
                "behavior_label": labels[run_start],
                "n_frames": n - run_start,
                "cam_id": cam_id,
                "identity_label": identity,
                "track_filename": track_fn,
            })

    if not rows:
        logger.info("No behaviour bouts found.")
        return pd.DataFrame()

    bout_df = pd.DataFrame(rows)
    bout_df.sort_values(["cam_id", "start_time"], inplace=True)
    bout_df.reset_index(drop=True, inplace=True)

    # ── Log summary ───────────────────────────────────────────────────────── #
    # Sleeping bouts overview (the most useful for checking smoothing quality)
    sleep_mask = bout_df["behavior_label"].str.contains("sleeping", case=False)
    sleep_df = bout_df[sleep_mask].copy()

    n_total_bouts = len(bout_df)
    n_sleep_bouts = len(sleep_df)
    n_short_sleep = (sleep_df["duration_s"] < 60).sum() if n_sleep_bouts > 0 else 0

    logger.info("\n" + "=" * 80)
    logger.info("BEHAVIOUR BOUT SUMMARY (after smoothing)")
    logger.info("=" * 80)
    logger.info("  Total bouts:          %d", n_total_bouts)
    logger.info("  Sleeping bouts:       %d", n_sleep_bouts)
    logger.info("  Short sleeping (<60s): %d  %s",
                n_short_sleep,
                "⚠ may need more smoothing" if n_short_sleep > 0 else "✓ clean")

    # Per-camera breakdown
    for cam_id in sorted(bout_df["cam_id"].unique()):
        cam_bouts = bout_df[bout_df["cam_id"] == cam_id]
        cam_sleep = cam_bouts[cam_bouts["behavior_label"].str.contains("sleeping")]
        n_short = (cam_sleep["duration_s"] < 60).sum() if len(cam_sleep) > 0 else 0
        logger.info("  cam %s: %d bouts (%d sleeping, %d short)",
                    cam_id, len(cam_bouts), len(cam_sleep), n_short)

    # Print sleeping bout table (compact)
    if n_sleep_bouts > 0:
        display_cols = ["start_time", "end_time", "duration_s",
                        "behavior_label", "cam_id", "identity_label"]
        tbl = sleep_df[display_cols].copy()
        tbl["start_time"] = tbl["start_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        tbl["end_time"] = tbl["end_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        logger.info("\n  Sleeping bouts:\n%s", tbl.to_string(index=False))

    # Flag suspicious short bouts (any label, < 30s)
    short_all = bout_df[bout_df["duration_s"] < 30]
    if len(short_all) > 0:
        logger.warning(
            "\n  ⚠ %d bouts shorter than 30s remain (possible smoothing gaps):",
            len(short_all),
        )
        flag_cols = ["start_time", "end_time", "duration_s",
                     "behavior_label", "cam_id", "track_filename"]
        flag_tbl = short_all[flag_cols].copy()
        flag_tbl["start_time"] = flag_tbl["start_time"].dt.strftime("%H:%M:%S")
        flag_tbl["end_time"] = flag_tbl["end_time"].dt.strftime("%H:%M:%S")
        logger.warning("\n%s", flag_tbl.to_string(index=False))

    return bout_df


# ──────────────────── 7. Summary / reporting ───────────────────────────────── #

def summarize_cross_cam_match(
    summary_df: pd.DataFrame,
    camera_ids: list[str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Print a human-readable summary of the track-level matching results."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if summary_df.empty:
        logger.info("No matching results to summarize.")
        return

    n_matched = (summary_df["status"] == "matched").sum()
    n_unmatched = len(summary_df) - n_matched

    logger.info("\n" + "=" * 80)
    logger.info("CROSS-CAMERA TRACK MATCHING SUMMARY  (cam %s ↔ cam %s)",
                camera_ids[0], camera_ids[1])
    logger.info("=" * 80)
    logger.info("  Matched pairs:   %d", n_matched)
    logger.info("  Unmatched tracks: %d", n_unmatched)

    if n_matched > 0:
        matched = summary_df[summary_df["status"] == "matched"]
        logger.info("\n  Matched pairs:\n%s",
                    matched[["cross_cam_id", "cam1_track", "cam2_track",
                             "matched_bins", "median_distance"]].to_string(index=False))

    if n_unmatched > 0:
        unmatched = summary_df[summary_df["status"] != "matched"]
        logger.info("\n  Unmatched tracks:\n%s",
                    unmatched[["cross_cam_id", "cam1_track", "cam2_track",
                               "status"]].to_string(index=False))
