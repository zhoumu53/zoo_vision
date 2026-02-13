"""
Cross-Camera ID Matching Module (production)
===========================================

Purpose
-------
Cross-camera stitching and identity assignment for a two-camera room setup using:
- World coordinates from raw track CSV (timestamp, world_x, world_y)
- Optional behavior CSV for sleep-aware ReID frame selection
- ReID voting per cross-camera stitched group (xcsid)
- Strict temporal exclusivity constraints for final identity labels

Key invariants enforced
-----------------------
1) xcsid merges must NOT create same-camera temporal overlap between different stitched_id.
2) xcsid merges must NOT violate world-coordinate hard constraints during temporal overlap.
3) Final identity labels must be temporally exclusive per label (no duplicate label at same time).
4) Deterministic ordering for reproducibility.

Public API
----------
- run_cross_camera_matching_v2
- match_tracks_cross_camera
- compute_cross_cam_stitched_id
- compute_cross_cam_individual
- vote_identity_by_xcsid_reid
- assign_identity_by_xcsid_strict
- smooth_behavior_cross_camera
- summarize_cross_cam_match
- summarize_behavior_bouts
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from post_processing.utils import load_embedding
from post_processing.core.reid_inference import match_to_gallery
from post_processing.core.tracklet_manager import vote_identity_from_matched_labels


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CrossCamConfig:
    bin_seconds: float = 1.0

    # Track-level cross-camera matching (cross_cam_id)
    distance_threshold_m: float = 2.0
    min_matched_bins: int = 5
    max_time_offset_s: int = 60  # time shift search range for clock offset correction

    # xcsid constraints
    xcsid_max_overlap_dist_m: float = 2.0
    xcsid_min_overlap_bins: int = 8

    # ReID selection
    reid_conf_thresh: float = 0.9
    sleep_min_seconds: float = 1800.0
    sleep_edge_window_minutes: float = 15.0
    prefer_behavior_labels: tuple[str, ...] = ("01_standing",)

    # Behavior smoothing
    behavior_pair_tolerance_ms: int = 200
    majority_window_seconds: float = 30.0
    within_track_min_bout_seconds: float = 300.0
    shortbout_max_seconds: float = 300.0
    shortbout_context_seconds: float = 1800.0

    # Monitoring
    fail_on_timestamp_nat_rate: float = 0.01


# --------------------------------------------------------------------------- #
# Timestamp normalization
# --------------------------------------------------------------------------- #

def norm_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    df["timestamp"] = df["timestamp"].dt.floor("ms")
    return df


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_track_positions(csv_path: str | Path, *, logger: Optional[logging.Logger] = None) -> pd.DataFrame | None:
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, usecols=lambda c: c in {"timestamp", "world_x", "world_y"})
    except Exception:
        return None
    if df.empty or "timestamp" not in df.columns:
        return None
    df = norm_timestamp(df)
    nat_rate = float(df["timestamp"].isna().mean())
    if logger is not None and nat_rate > 0:
        logger.debug("NaT rate in %s: %.4f", p.name, nat_rate)
    df = df.dropna(subset=["timestamp", "world_x", "world_y"])
    if df.empty:
        return None
    return df.sort_values("timestamp").reset_index(drop=True)


def _build_track_index(tracklets_data: list[dict], cam_label: str) -> list[dict]:
    tracks: list[dict] = []
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
        tf = t.get("track_filename", "")
        if not tf:
            continue
        tracks.append(
            {
                "track_filename": tf,
                "csv_path": csv_path,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "camera": cam_label,
            }
        )
    tracks.sort(key=lambda d: (d["start_ts"], d["track_filename"]))
    return tracks


# --------------------------------------------------------------------------- #
# Optional time offset estimation
# --------------------------------------------------------------------------- #

def _activity_series_from_index(idx: list[dict], *, freq_s: int = 1) -> pd.Series:
    if not idx:
        return pd.Series(dtype=int)
    t0 = min(t["start_ts"] for t in idx).floor(f"{freq_s}s")
    t1 = max(t["end_ts"] for t in idx).ceil(f"{freq_s}s")
    if t1 <= t0:
        return pd.Series(dtype=int)

    bins = pd.date_range(t0, t1, freq=f"{freq_s}s")
    counts = pd.Series(0, index=bins, dtype=int)

    for t in idx:
        s = t["start_ts"].floor(f"{freq_s}s")
        e = t["end_ts"].ceil(f"{freq_s}s")
        if e <= s:
            continue
        counts.loc[s:e] += 1
    return counts


def estimate_time_offset_seconds(
    idx1: list[dict],
    idx2: list[dict],
    *,
    max_offset_s: int,
    logger: Optional[logging.Logger] = None,
) -> int:
    if logger is None:
        logger = logging.getLogger(__name__)

    s1 = _activity_series_from_index(idx1, freq_s=1)
    s2 = _activity_series_from_index(idx2, freq_s=1)
    if s1.empty or s2.empty:
        return 0

    common = s1.index.intersection(s2.index)
    if len(common) < 60:
        return 0

    a = s1.loc[common].astype(float).values
    b = s2.loc[common].astype(float).values

    if np.all(a == 0) or np.all(b == 0):
        return 0

    best_off = 0
    best_score = -1e18

    for off in range(-max_offset_s, max_offset_s + 1):
        if off < 0:
            aa = a[-off:]
            bb = b[: len(aa)]
        elif off > 0:
            bb = b[off:]
            aa = a[: len(bb)]
        else:
            aa = a
            bb = b
        if len(aa) < 60:
            continue
        score = float(np.dot(aa - aa.mean(), bb - bb.mean()))
        if score > best_score:
            best_score = score
            best_off = off

    logger.info("Estimated camera time offset: %ds (apply to cam2 timestamps).", best_off)
    return int(best_off)


# --------------------------------------------------------------------------- #
# Track-level cross-camera matching (cross_cam_id)
# --------------------------------------------------------------------------- #

def match_tracks_cross_camera(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    cam1_id: str,
    cam2_id: str,
    *,
    cfg: CrossCamConfig = CrossCamConfig(),
    logger: Optional[logging.Logger] = None,
) -> tuple[dict[str, int], pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)

    idx1 = _build_track_index(tracklets_cam1, cam1_id)
    idx2 = _build_track_index(tracklets_cam2, cam2_id)
    logger.info("Track index: cam %s=%d tracks, cam %s=%d tracks", cam1_id, len(idx1), cam2_id, len(idx2))

    time_offset_s = 0
    if cfg.max_time_offset_s > 0:
        time_offset_s = estimate_time_offset_seconds(idx1, idx2, max_offset_s=cfg.max_time_offset_s, logger=logger)

    frames: list[pd.DataFrame] = []
    loaded: set[str] = set()

    for track_info in (idx1 + idx2):
        tf = track_info["track_filename"]
        if tf in loaded:
            continue
        df = _load_track_positions(track_info["csv_path"], logger=logger)
        if df is None:
            continue

        if track_info["camera"] == cam2_id and time_offset_s != 0:
            df["timestamp"] = df["timestamp"] + pd.Timedelta(seconds=time_offset_s)

        df["_track"] = tf
        df["_camera"] = track_info["camera"]
        frames.append(df)
        loaded.add(tf)

    if not frames:
        logger.warning("No track CSVs loaded.")
        return {}, pd.DataFrame()

    all_frames = pd.concat(frames, ignore_index=True)

    nat_rate = float(all_frames["timestamp"].isna().mean())
    if nat_rate > cfg.fail_on_timestamp_nat_rate:
        raise RuntimeError(f"Timestamp NaT rate too high: {nat_rate:.4f}")

    all_frames["_tbin"] = all_frames["timestamp"].dt.floor(f"{cfg.bin_seconds}s")

    agg = (
        all_frames.groupby(["_tbin", "_camera", "_track"], sort=False)
        .agg(wx=("world_x", "median"), wy=("world_y", "median"))
        .reset_index()
    )

    c1 = agg[agg["_camera"] == cam1_id]
    c2 = agg[agg["_camera"] == cam2_id]

    c1_by_bin = {tb: g for tb, g in c1.groupby("_tbin", sort=False)}
    c2_by_bin = {tb: g for tb, g in c2.groupby("_tbin", sort=False)}

    BIG = cfg.distance_threshold_m + 1e6
    pair_votes: Counter[tuple[str, str]] = Counter()
    pair_dists: dict[tuple[str, str], list[float]] = defaultdict(list)

    common_bins = sorted(set(c1_by_bin.keys()) & set(c2_by_bin.keys()))
    logger.info("Common time bins: %d", len(common_bins))

    for tbin in common_bins:
        g1 = c1_by_bin[tbin]
        g2 = c2_by_bin[tbin]
        if g1.empty or g2.empty:
            continue

        pos1 = g1[["wx", "wy"]].values
        pos2 = g2[["wx", "wy"]].values
        t1s = g1["_track"].astype(str).values
        t2s = g2["_track"].astype(str).values

        dist = cdist(pos1, pos2)
        cost = dist.copy()
        cost[cost > cfg.distance_threshold_m] = BIG

        r_idx, c_idx = linear_sum_assignment(cost)
        for r, c in zip(r_idx, c_idx):
            if cost[r, c] >= BIG:
                continue
            a = str(t1s[r])
            b = str(t2s[c])
            pair_votes[(a, b)] += 1
            pair_dists[(a, b)].append(float(dist[r, c]))

    if not pair_votes:
        logger.warning("No candidate cross-camera pairs found.")
        return {}, pd.DataFrame()

    cand = [(k, v) for k, v in pair_votes.items() if v >= cfg.min_matched_bins]
    if not cand:
        logger.warning("All candidate pairs below min_matched_bins=%d.", cfg.min_matched_bins)
        return {}, pd.DataFrame()

    tracks1 = sorted({k[0] for k, _ in cand})
    tracks2 = sorted({k[1] for k, _ in cand})
    i1 = {t: i for i, t in enumerate(tracks1)}
    i2 = {t: i for i, t in enumerate(tracks2)}

    W = np.zeros((len(tracks1), len(tracks2)), dtype=float)
    for (a, b), votes in cand:
        W[i1[a], i2[b]] = float(votes)

    maxW = float(W.max()) if W.size else 0.0
    if maxW <= 0:
        return {}, pd.DataFrame()

    cost = (maxW - W)
    large = maxW + 1e6
    cost[W <= 0] = large

    rr, cc = linear_sum_assignment(cost)

    final_matches: list[dict] = []
    for r, c in zip(rr, cc):
        if W[r, c] <= 0:
            continue
        a = tracks1[r]
        b = tracks2[c]
        votes = int(W[r, c])
        dists = pair_dists.get((a, b), [])
        med = float(np.median(dists)) if dists else float("nan")
        final_matches.append(
            {
                "cam1_track": a,
                "cam2_track": b,
                "matched_bins": votes,
                "median_distance": round(med, 4) if np.isfinite(med) else np.nan,
            }
        )

    track_to_xcid: dict[str, int] = {}
    xcid = 0
    for m in sorted(final_matches, key=lambda d: (-d["matched_bins"], d["cam1_track"], d["cam2_track"])):
        track_to_xcid[m["cam1_track"]] = xcid
        track_to_xcid[m["cam2_track"]] = xcid
        m["cross_cam_id"] = xcid
        xcid += 1

    all_tracknames = sorted({t["track_filename"] for t in (idx1 + idx2) if t.get("track_filename")})
    for tf in all_tracknames:
        if tf not in track_to_xcid:
            track_to_xcid[tf] = xcid
            xcid += 1

    rows: list[dict] = []
    for m in final_matches:
        rows.append(
            {
                "cross_cam_id": m["cross_cam_id"],
                "cam1_track": m["cam1_track"],
                "cam2_track": m["cam2_track"],
                "matched_bins": m["matched_bins"],
                "median_distance": m["median_distance"],
                "status": "matched",
            }
        )

    assigned1 = {m["cam1_track"] for m in final_matches}
    assigned2 = {m["cam2_track"] for m in final_matches}

    for t in idx1:
        tf = t["track_filename"]
        if tf not in assigned1:
            rows.append(
                {
                    "cross_cam_id": track_to_xcid[tf],
                    "cam1_track": tf,
                    "cam2_track": "",
                    "matched_bins": 0,
                    "median_distance": np.nan,
                    "status": f"unmatched_cam{cam1_id}",
                }
            )
    for t in idx2:
        tf = t["track_filename"]
        if tf not in assigned2:
            rows.append(
                {
                    "cross_cam_id": track_to_xcid[tf],
                    "cam1_track": "",
                    "cam2_track": tf,
                    "matched_bins": 0,
                    "median_distance": np.nan,
                    "status": f"unmatched_cam{cam2_id}",
                }
            )

    summary_df = pd.DataFrame(rows).sort_values(["matched_bins", "cross_cam_id"], ascending=[False, True]).reset_index(drop=True)
    logger.info("Final cross_cam_id matched pairs: %d", len(final_matches))
    return track_to_xcid, summary_df


# --------------------------------------------------------------------------- #
# Union-Find
# --------------------------------------------------------------------------- #

class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
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


# --------------------------------------------------------------------------- #
# xcsid hard constraints
# --------------------------------------------------------------------------- #

def _bin_positions(df: pd.DataFrame, bin_seconds: float) -> pd.DataFrame:
    d = df.copy()
    d["_tbin"] = d["timestamp"].dt.floor(f"{bin_seconds}s")
    return d.groupby("_tbin", sort=False).agg(wx=("world_x", "median"), wy=("world_y", "median")).reset_index()


def _overlap_distance_stats(
    csv_a: str | Path,
    csv_b: str | Path,
    *,
    bin_seconds: float,
    max_overlap_dist_m: float,
    min_overlap_bins: int,
    cache: dict,
) -> tuple[float, int]:
    key = ("ov", str(csv_a), str(csv_b), float(bin_seconds), float(max_overlap_dist_m), int(min_overlap_bins))
    if key in cache:
        return cache[key]

    df_a = _load_track_positions(csv_a)
    df_b = _load_track_positions(csv_b)
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        cache[key] = (float("inf"), 0)
        return cache[key]

    ba = _bin_positions(df_a, bin_seconds)
    bb = _bin_positions(df_b, bin_seconds)

    m = ba.merge(bb, on="_tbin", how="inner", suffixes=("_a", "_b"))
    if m.empty:
        cache[key] = (float("inf"), 0)
        return cache[key]

    dx = m["wx_a"].values - m["wx_b"].values
    dy = m["wy_a"].values - m["wy_b"].values
    dist = np.sqrt(dx * dx + dy * dy)

    good = dist[np.isfinite(dist)]
    if good.size == 0:
        cache[key] = (float("inf"), 0)
        return cache[key]

    n = int(np.sum(good <= max_overlap_dist_m))
    if n < min_overlap_bins:
        cache[key] = (float("inf"), n)
        return cache[key]

    med = float(np.median(good[good <= max_overlap_dist_m]))
    cache[key] = (med, n)
    return cache[key]


def _would_create_same_cam_overlap(root_a: str, root_b: str, *, uf: _UnionFind, tf_to_meta: dict[str, dict]) -> bool:
    mem_a = [tf for tf in tf_to_meta if uf.find(tf) == root_a]
    mem_b = [tf for tf in tf_to_meta if uf.find(tf) == root_b]
    for ta in mem_a:
        ma = tf_to_meta[ta]
        for tb in mem_b:
            mb = tf_to_meta[tb]
            if ma["cam"] != mb["cam"]:
                continue
            if ma["sid"] == mb["sid"]:
                continue
            if ma["start"] < mb["end"] and mb["start"] < ma["end"]:
                return True
    return False


def _would_create_world_conflict_between_components(
    root_a: str,
    root_b: str,
    *,
    uf: _UnionFind,
    tf_to_meta: dict[str, dict],
    bin_seconds: float,
    max_overlap_dist_m: float,
    min_overlap_bins: int,
    cache: dict,
) -> bool:
    mem_a = [tf for tf in tf_to_meta if uf.find(tf) == root_a]
    mem_b = [tf for tf in tf_to_meta if uf.find(tf) == root_b]
    if not mem_a or not mem_b:
        return False

    sa = min(tf_to_meta[tf]["start"] for tf in mem_a)
    ea = max(tf_to_meta[tf]["end"] for tf in mem_a)
    sb = min(tf_to_meta[tf]["start"] for tf in mem_b)
    eb = max(tf_to_meta[tf]["end"] for tf in mem_b)
    if not (sa < eb and sb < ea):
        return False

    pairs: list[tuple[str, str]] = []
    for ta in mem_a:
        ma = tf_to_meta[ta]
        for tb in mem_b:
            mb = tf_to_meta[tb]
            if ma["start"] < mb["end"] and mb["start"] < ma["end"]:
                pairs.append((ma["csv_path"], mb["csv_path"]))

    pairs.sort()
    for a_csv, b_csv in pairs:
        med, n_ok = _overlap_distance_stats(
            a_csv,
            b_csv,
            bin_seconds=bin_seconds,
            max_overlap_dist_m=max_overlap_dist_m,
            min_overlap_bins=min_overlap_bins,
            cache=cache,
        )
        if n_ok >= min_overlap_bins and np.isfinite(med) and med > max_overlap_dist_m:
            return True

    return False


def compute_cross_cam_stitched_id(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    cam1_id: str,
    cam2_id: str,
    *,
    cfg: CrossCamConfig = CrossCamConfig(),
    logger: Optional[logging.Logger] = None,
) -> dict[str, int]:
    if logger is None:
        logger = logging.getLogger(__name__)

    uf = _UnionFind()

    tf_to_meta: dict[str, dict] = {}
    for cam_id, tracklets in [(cam1_id, tracklets_cam1), (cam2_id, tracklets_cam2)]:
        for t in tracklets:
            tf = t.get("track_filename", "")
            csvp = t.get("track_csv_path", "")
            if not tf or not csvp:
                continue
            try:
                s = pd.Timestamp(t["start_timestamp"])
                e = pd.Timestamp(t["end_timestamp"])
            except Exception:
                continue
            tf_to_meta[tf] = {
                "cam": cam_id,
                "sid": t.get("stitched_id"),
                "start": s,
                "end": e,
                "csv_path": csvp,
            }

    for cam_id, tracklets in [(cam1_id, tracklets_cam1), (cam2_id, tracklets_cam2)]:
        sid_to_tracks: dict[tuple[str, int], list[str]] = defaultdict(list)
        for t in tracklets:
            tf = t.get("track_filename", "")
            sid = t.get("stitched_id")
            if tf and sid is not None:
                sid_to_tracks[(cam_id, int(sid))].append(tf)
        for _, tracks in sorted(sid_to_tracks.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            tracks = sorted(tracks)
            for i in range(1, len(tracks)):
                uf.union(tracks[0], tracks[i])

    xcid_to_tracks: dict[int, list[str]] = defaultdict(list)
    for t in tracklets_cam1 + tracklets_cam2:
        tf = t.get("track_filename", "")
        xcid = t.get("cross_cam_id")
        if tf and xcid is not None:
            xcid_to_tracks[int(xcid)].append(tf)

    cache: dict = {}
    n_skipped_samecam = 0
    n_skipped_world = 0

    for xcid, tracks in sorted(xcid_to_tracks.items(), key=lambda kv: kv[0]):
        tracks = sorted(set(tracks))
        if len(tracks) < 2:
            continue
        anchor = tracks[0]
        for b in tracks[1:]:
            ra = uf.find(anchor)
            rb = uf.find(b)
            if ra == rb:
                continue

            if _would_create_same_cam_overlap(ra, rb, uf=uf, tf_to_meta=tf_to_meta):
                n_skipped_samecam += 1
                continue

            if _would_create_world_conflict_between_components(
                ra,
                rb,
                uf=uf,
                tf_to_meta=tf_to_meta,
                bin_seconds=cfg.bin_seconds,
                max_overlap_dist_m=cfg.xcsid_max_overlap_dist_m,
                min_overlap_bins=cfg.xcsid_min_overlap_bins,
                cache=cache,
            ):
                n_skipped_world += 1
                continue

            uf.union(anchor, b)

    all_tracks = sorted({t.get("track_filename", "") for t in (tracklets_cam1 + tracklets_cam2) if t.get("track_filename")})
    root_to_id: dict[str, int] = {}
    track_to_xcsid: dict[str, int] = {}
    next_id = 0

    for tf in all_tracks:
        r = uf.find(tf)
        if r not in root_to_id:
            root_to_id[r] = next_id
            next_id += 1
        track_to_xcsid[tf] = root_to_id[r]

    logger.info(
        "xcsid: %d tracks -> %d groups (skipped samecam=%d, world=%d)",
        len(track_to_xcsid),
        next_id,
        n_skipped_samecam,
        n_skipped_world,
    )

    return track_to_xcsid


# --------------------------------------------------------------------------- #
# cross_cam_individual (bout-aware partitioning, n_individuals==2)
# --------------------------------------------------------------------------- #

def compute_cross_cam_individual(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    *,
    n_individuals: int = 2,
    known_individuals: Optional[list[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> dict[str, int]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if n_individuals <= 0:
        raise ValueError("n_individuals must be positive")

    all_tracklets = tracklets_cam1 + tracklets_cam2

    intervals: list[dict] = []
    for t in all_tracklets:
        xcsid = t.get("cross_cam_stitched_id")
        if xcsid is None:
            continue
        try:
            s = pd.Timestamp(t["start_timestamp"])
            e = pd.Timestamp(t["end_timestamp"])
        except Exception:
            continue
        intervals.append(
            {
                "start": s,
                "end": e,
                "tf": t.get("track_filename", ""),
                "xcsid": int(xcsid),
                "camera": t.get("camera_id", t.get("camera", "")),
            }
        )

    intervals = [iv for iv in intervals if iv["tf"]]
    if not intervals:
        return {}

    xcsid_tracks: dict[int, list[dict]] = defaultdict(list)
    for iv in intervals:
        xcsid_tracks[iv["xcsid"]].append(iv)

    xcsid_composites: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for xcsid, tracks in xcsid_tracks.items():
        tracks = sorted(tracks, key=lambda x: x["start"])
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

    overlap = np.zeros((n, n), dtype=float)
    for i, xa in enumerate(all_xcsids):
        for j in range(i + 1, n):
            xb = all_xcsids[j]
            total = 0.0
            for sa, ea in xcsid_composites[xa]:
                for sb, eb in xcsid_composites[xb]:
                    ov = (min(ea, eb) - max(sa, sb)).total_seconds()
                    if ov > 0:
                        total += ov
            overlap[i, j] = total
            overlap[j, i] = total

    adjacency = overlap > 0
    visited = [False] * n
    bouts: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        comp: list[int] = []
        q = [start]
        visited[start] = True
        while q:
            node = q.pop(0)
            comp.append(node)
            for nb in range(n):
                if not visited[nb] and adjacency[node, nb]:
                    visited[nb] = True
                    q.append(nb)
        bouts.append(sorted(comp))

    def _earliest(comp: list[int]) -> pd.Timestamp:
        tmin = pd.Timestamp.max
        for idx in comp:
            x = all_xcsids[idx]
            for s, _ in xcsid_composites[x]:
                if s < tmin:
                    tmin = s
        return tmin

    bouts.sort(key=_earliest)

    partition = np.full(n, -1, dtype=int)

    for bout_idx, members in enumerate(bouts):
        offset = bout_idx * n_individuals
        sub = overlap[np.ix_(members, members)]
        per_node = np.sum(sub, axis=1)
        overlapping_nodes = per_node > 0
        n_overlapping = int(np.sum(overlapping_nodes))

        spurious_local: set[int] = set()
        if n_overlapping >= n_individuals:
            for li in range(len(members)):
                if not overlapping_nodes[li]:
                    spurious_local.add(li)

        real_local = sorted(set(range(len(members))) - spurious_local)
        for li in spurious_local:
            partition[members[li]] = -1

        if not real_local:
            continue

        if len(real_local) <= n_individuals:
            for slot, li in enumerate(real_local):
                partition[members[li]] = offset + slot
            continue

        sub_real = sub[np.ix_(real_local, real_local)]

        max_i, max_j = np.unravel_index(np.argmax(sub_real), sub_real.shape)
        part_local = np.full(len(real_local), -1, dtype=int)
        part_local[max_i] = 0
        part_local[max_j] = 1

        remaining = sorted(
            (i for i in range(len(real_local)) if part_local[i] < 0),
            key=lambda i: -float(np.sum(sub_real[i])),
        )
        for i in remaining:
            ov0 = float(np.dot(sub_real[i], (part_local == 0).astype(float)))
            ov1 = float(np.dot(sub_real[i], (part_local == 1).astype(float)))
            part_local[i] = 0 if ov0 <= ov1 else 1

        for _ in range(50):
            improved = False
            for i in range(len(real_local)):
                old = int(part_local[i])
                new = 1 - old
                same_old = float(np.dot(sub_real[i], (part_local == old).astype(float)))
                same_new = float(np.dot(sub_real[i], (part_local == new).astype(float)))
                if same_new < same_old:
                    part_local[i] = new
                    improved = True
            if not improved:
                break

        for pos, li in enumerate(real_local):
            partition[members[li]] = offset + int(part_local[pos])

    tf_to_ind: dict[str, int] = {}
    for iv in intervals:
        tf_to_ind[iv["tf"]] = int(partition[xcsid_to_idx[iv["xcsid"]]])

    return tf_to_ind


# --------------------------------------------------------------------------- #
# ReID voting per xcsid (sleep-edge aware)
# --------------------------------------------------------------------------- #

def _is_sleep_label(lbl: str) -> bool:
    return isinstance(lbl, str) and ("sleep" in lbl.lower())


def _get_behavior_label_col(df: pd.DataFrame) -> str:
    if "behavior_label_raw" in df.columns:
        return "behavior_label_raw"
    if "behavior_label_old" in df.columns:
        return "behavior_label_old"
    return "behavior_label"


def _find_longest_sleep_bout(
    df_beh: pd.DataFrame,
    *,
    ts_col: str,
    lbl_col: str,
    min_sleep_seconds: float,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if df_beh.empty or ts_col not in df_beh.columns or lbl_col not in df_beh.columns:
        return None

    df = df_beh.sort_values(ts_col).reset_index(drop=True)
    ts = df[ts_col]
    labels = df[lbl_col].astype(str).values
    n = len(df)
    if n == 0:
        return None

    best_dur = -1.0
    best = None

    run_start = 0
    for i in range(1, n):
        if labels[i] != labels[run_start]:
            lbl = labels[run_start]
            if _is_sleep_label(lbl):
                dur = (ts.iloc[i - 1] - ts.iloc[run_start]).total_seconds()
                if dur >= min_sleep_seconds and dur > best_dur:
                    best_dur = dur
                    best = (ts.iloc[run_start], ts.iloc[i - 1])
            run_start = i

    lbl = labels[run_start]
    if _is_sleep_label(lbl):
        dur = (ts.iloc[n - 1] - ts.iloc[run_start]).total_seconds()
        if dur >= min_sleep_seconds and dur > best_dur:
            best = (ts.iloc[run_start], ts.iloc[n - 1])

    if best is None:
        return None
    return pd.Timestamp(best[0]), pd.Timestamp(best[1])


def _select_reid_frame_indices_from_behavior_csv(
    behavior_csv: Path,
    *,
    cfg: CrossCamConfig,
) -> list[int]:
    df = _safe_read_csv(behavior_csv)
    if df is None or df.empty or "timestamp" not in df.columns:
        return []

    df = norm_timestamp(df)
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    lbl_col = _get_behavior_label_col(df)
    if lbl_col not in df.columns:
        return []

    if "behavior_conf" not in df.columns:
        df["behavior_conf"] = 1.0
    if "quality_label" not in df.columns:
        df["quality_label"] = "good"

    labels = df[lbl_col].astype(str)
    conf = pd.to_numeric(df["behavior_conf"], errors="coerce").fillna(0.0)
    qual = df["quality_label"].astype(str)

    base = (qual == "good") & (conf >= cfg.reid_conf_thresh) & (labels != "00_invalid")

    sleep_bout = _find_longest_sleep_bout(
        df, ts_col="timestamp", lbl_col=lbl_col, min_sleep_seconds=cfg.sleep_min_seconds
    )

    ts = df["timestamp"]
    non_sleep = ~labels.map(_is_sleep_label)

    if sleep_bout is not None:
        s0, s1 = sleep_bout
        W = pd.Timedelta(minutes=float(cfg.sleep_edge_window_minutes))
        pre = (ts >= (s0 - W)) & (ts < s0)
        post = (ts > s1) & (ts <= (s1 + W))
        m = base & (pre | post) & non_sleep
        idx = df.index[m].tolist()
        if idx:
            return idx

    for pref in cfg.prefer_behavior_labels:
        m = base & (labels == pref)
        idx = df.index[m].tolist()
        if idx:
            return idx

    return df.index[base & non_sleep].tolist() or df.index[base].tolist()


def vote_identity_by_xcsid_reid(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    *,
    gallery_path: str | Path,
    known_individuals: Optional[list[str]] = None,
    cfg: CrossCamConfig = CrossCamConfig(),
    logger: Optional[logging.Logger] = None,
) -> dict[int, dict]:
    if logger is None:
        logger = logging.getLogger(__name__)

    known_individuals = known_individuals or []
    gallery_path = Path(gallery_path)
    if not gallery_path.exists():
        logger.error("Gallery file not found: %s", gallery_path)
        return {}

    g = np.load(gallery_path, allow_pickle=True)
    gallery_features = torch.from_numpy(g["feature"]).float()
    gallery_labels = g["label"].tolist()

    if known_individuals:
        keep = [i for i, lbl in enumerate(gallery_labels) if lbl in set(known_individuals)]
        gallery_features = gallery_features[keep]
        gallery_labels = [gallery_labels[i] for i in keep]

    all_tracklets = tracklets_cam1 + tracklets_cam2
    xcsid_groups: dict[int, list[dict]] = defaultdict(list)
    for t in all_tracklets:
        xcsid = t.get("cross_cam_stitched_id")
        if xcsid is not None:
            xcsid_groups[int(xcsid)].append(t)

    results: dict[int, dict] = {}

    for xcsid, members in sorted(xcsid_groups.items(), key=lambda kv: kv[0]):
        merged_list: list[np.ndarray] = []

        for t in members:
            fp = t.get("feature_path", "")
            if not fp:
                continue
            feature_path = Path(fp)
            if not feature_path.exists():
                continue

            features, frame_ids, _, _ = load_embedding(feature_path)
            if features is None or len(features) == 0:
                continue

            csv_path = t.get("track_csv_path", "")
            if not csv_path:
                continue
            beh_csv = Path(csv_path).with_name(Path(csv_path).stem + "_behavior.csv")
            good_idx = _select_reid_frame_indices_from_behavior_csv(beh_csv, cfg=cfg)
            if not good_idx:
                continue

            fidx = [i for i, fid in enumerate(frame_ids) if fid in set(good_idx)]
            if not fidx:
                continue

            features = features[fidx]
            if len(features) > 0:
                merged_list.append(features)

        if not merged_list:
            continue

        merged = np.concatenate(merged_list, axis=0)
        merged_tensor = torch.from_numpy(merged).float()

        matched_labels = match_to_gallery(
            merged_tensor,
            gallery_features,
            gallery_labels=gallery_labels,
        )[-1]

        voted = vote_identity_from_matched_labels(
            matched_labels=np.array(matched_labels),
            known_labels=known_individuals,
            sample_rate=1,
            return_details=True,
        )

        results[xcsid] = voted
        top1 = voted.get("voted_label", "unknown")
        vote_counts = {
            lbl: {"count": int(cnt), "pct": round(pct, 2)}
            for lbl, cnt, pct in voted.get("ranked_labels", [])
        }

        for t in members:
            t["cross_cam_stitched_id_vote"] = top1
            t["cross_cam_stitched_id_vote_counts"] = vote_counts

    return results


# --------------------------------------------------------------------------- #
# Strict identity assignment (temporal exclusivity per label)
# --------------------------------------------------------------------------- #

def assign_identity_by_xcsid_strict(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    *,
    known_individuals: list[str],
    logger: Optional[logging.Logger] = None,
) -> tuple[dict[str, str], pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)

    allowed = list(known_individuals or [])
    all_tracklets = tracklets_cam1 + tracklets_cam2

    xcsid_groups: dict[int, list[dict]] = defaultdict(list)
    for t in all_tracklets:
        x = t.get("cross_cam_stitched_id")
        if x is not None:
            xcsid_groups[int(x)].append(t)

    group_info: dict[int, dict] = {}
    for xcsid, members in xcsid_groups.items():
        vote = Counter()
        vc = None
        for m in members:
            vc = m.get("cross_cam_stitched_id_vote_counts")
            if isinstance(vc, dict) and vc:
                break
        if isinstance(vc, dict):
            for lbl, info in vc.items():
                if lbl in set(allowed):
                    cnt = info["count"] if isinstance(info, dict) else int(info)
                    vote[lbl] += int(cnt)

        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []
        for m in members:
            try:
                starts.append(pd.Timestamp(m["start_timestamp"]))
                ends.append(pd.Timestamp(m["end_timestamp"]))
            except Exception:
                continue

        s = min(starts) if starts else None
        e = max(ends) if ends else None
        dur = (e - s).total_seconds() if (s is not None and e is not None) else 0.0
        evidence = float(vote.total()) + 0.001 * float(dur)

        group_info[xcsid] = {"vote": vote, "start": s, "end": e, "evidence": evidence, "n_tracks": len(members)}

    sorted_xcsids = sorted(group_info.keys(), key=lambda x: group_info[x]["evidence"], reverse=True)

    label_intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]] = defaultdict(list)

    def overlaps(label: str, s: pd.Timestamp, e: pd.Timestamp) -> list[int]:
        hits: list[int] = []
        for ss, ee, xid in label_intervals[label]:
            if ss < e and s < ee:
                hits.append(xid)
        return hits

    xcsid_to_label: dict[int, str] = {}
    conflict_rows: list[dict] = []

    for xcsid in sorted_xcsids:
        gi = group_info[xcsid]
        s, e = gi["start"], gi["end"]
        if s is None or e is None:
            xcsid_to_label[xcsid] = "unknown"
            continue

        candidates = [lbl for lbl, _ in gi["vote"].most_common() if lbl in set(allowed)]
        for lbl in allowed:
            if lbl not in candidates:
                candidates.append(lbl)

        assigned = False
        for lbl in candidates:
            hits = overlaps(lbl, s, e)
            if not hits:
                xcsid_to_label[xcsid] = lbl
                label_intervals[lbl].append((s, e, xcsid))
                assigned = True
                break

            for h in hits:
                hs, he = group_info[h]["start"], group_info[h]["end"]
                if hs is None or he is None:
                    continue
                os = max(s, hs)
                oe = min(e, he)
                if oe > os:
                    conflict_rows.append(
                        {
                            "label": lbl,
                            "xcsid_kept": h,
                            "xcsid_blocked": xcsid,
                            "overlap_seconds": (oe - os).total_seconds(),
                            "overlap_start": str(os)[:19],
                            "overlap_end": str(oe)[:19],
                            "resolution": "try_next",
                        }
                    )

        if not assigned:
            xcsid_to_label[xcsid] = "unknown"
            if conflict_rows:
                conflict_rows[-1]["resolution"] = "set_unknown"

    track_to_label: dict[str, str] = {}
    for xcsid, lbl in xcsid_to_label.items():
        for t in xcsid_groups[xcsid]:
            tf = t.get("track_filename", "")
            if tf:
                track_to_label[tf] = lbl

    conflict_df = pd.DataFrame(conflict_rows)
    return track_to_label, conflict_df


# --------------------------------------------------------------------------- #
# Behavior smoothing (4 stages)
# --------------------------------------------------------------------------- #

def _compute_bout_lengths(labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    if n == 0:
        return np.array([], dtype=int)
    out = np.ones(n, dtype=int)
    run_starts = [0]
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            run_starts.append(i)
    run_starts.append(n)
    for r in range(len(run_starts) - 1):
        s = run_starts[r]
        e = run_starts[r + 1]
        out[s:e] = e - s
    return out


def _get_bouts(labels: np.ndarray, timestamps: pd.Series) -> list[tuple[int, int, str, float]]:
    n = len(labels)
    if n == 0:
        return []
    bouts: list[tuple[int, int, str, float]] = []
    run_start = 0
    for i in range(1, n):
        if labels[i] != labels[run_start]:
            dur = (timestamps.iloc[i - 1] - timestamps.iloc[run_start]).total_seconds()
            bouts.append((run_start, i, str(labels[run_start]), float(dur)))
            run_start = i
    dur = (timestamps.iloc[n - 1] - timestamps.iloc[run_start]).total_seconds()
    bouts.append((run_start, n, str(labels[run_start]), float(dur)))
    return bouts


def _apply_fixes(df: pd.DataFrame, fixes: dict[int, str], csv_path: Path) -> int:
    if not fixes:
        return 0
    ts_to_lbl: dict[pd.Timestamp, str] = {}
    for idx, new_lbl in fixes.items():
        ts_to_lbl[df.at[idx, "timestamp"]] = new_lbl

    n_fixed = 0
    for i in df.index:
        ts = df.at[i, "timestamp"]
        if ts in ts_to_lbl:
            df.at[i, "behavior_label"] = ts_to_lbl[ts]
            n_fixed += 1

    if n_fixed > 0:
        drop_cols = [c for c in df.columns if c.startswith("_")]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        df.to_csv(csv_path, index=False)
    return n_fixed


def _smooth_behavior_pair(t1: dict, t2: dict, *, tol_ms: int, logger: logging.Logger) -> int:
    csv1 = Path(t1["track_csv_path"])
    csv2 = Path(t2["track_csv_path"])
    b1 = csv1.with_name(csv1.stem + "_behavior.csv")
    b2 = csv2.with_name(csv2.stem + "_behavior.csv")
    if not b1.exists() or not b2.exists():
        return 0

    df1 = _safe_read_csv(b1)
    df2 = _safe_read_csv(b2)
    if df1 is None or df2 is None:
        return 0
    if "timestamp" not in df1.columns or "timestamp" not in df2.columns:
        return 0
    if "behavior_label" not in df1.columns or "behavior_label" not in df2.columns:
        return 0
    if "behavior_conf" not in df1.columns:
        df1["behavior_conf"] = 1.0
    if "behavior_conf" not in df2.columns:
        df2["behavior_conf"] = 1.0

    df1 = norm_timestamp(df1)
    df2 = norm_timestamp(df2)
    df1["_ts"] = df1["timestamp"]
    df2["_ts"] = df2["timestamp"]
    df1["_bout"] = _compute_bout_lengths(df1["behavior_label"].astype(str).values)
    df2["_bout"] = _compute_bout_lengths(df2["behavior_label"].astype(str).values)
    df1["_orig_idx"] = df1.index
    df2["_orig_idx"] = df2.index

    df1d = df1.drop_duplicates(subset="timestamp", keep="first").sort_values("_ts").reset_index(drop=True)
    df2d = df2.drop_duplicates(subset="timestamp", keep="first").sort_values("_ts").reset_index(drop=True)

    tol = pd.Timedelta(milliseconds=int(tol_ms))

    m12 = pd.merge_asof(
        df1d[["_ts", "behavior_label", "behavior_conf", "_bout", "_orig_idx"]],
        df2d[["_ts", "behavior_label", "behavior_conf", "_bout", "_orig_idx"]],
        on="_ts",
        tolerance=tol,
        direction="nearest",
        suffixes=("_1", "_2"),
    )
    m21 = pd.merge_asof(
        df2d[["_ts", "behavior_label", "behavior_conf", "_bout", "_orig_idx"]],
        df1d[["_ts", "behavior_label", "behavior_conf", "_bout", "_orig_idx"]],
        on="_ts",
        tolerance=tol,
        direction="nearest",
        suffixes=("_2", "_1"),
    )

    fix1: dict[int, str] = {}
    fix2: dict[int, str] = {}

    def proc(merged: pd.DataFrame, self_suf: str, other_suf: str, fix: dict[int, str]) -> None:
        lbl_self = f"behavior_label{self_suf}"
        lbl_other = f"behavior_label{other_suf}"
        conf_self = f"behavior_conf{self_suf}"
        conf_other = f"behavior_conf{other_suf}"
        bout_self = f"_bout{self_suf}"
        bout_other = f"_bout{other_suf}"
        idx_self = f"_orig_idx{self_suf}"

        has = merged[lbl_other].notna()
        disagree = has & (merged[lbl_self] != merged[lbl_other])

        for i in merged.index[disagree]:
            a = str(merged.at[i, lbl_self])
            b = str(merged.at[i, lbl_other])
            if a == "00_invalid" or b == "00_invalid":
                continue
            bs = int(merged.at[i, bout_self])
            bo = int(merged.at[i, bout_other])
            cs = float(merged.at[i, conf_self])
            co = float(merged.at[i, conf_other])
            if bo > bs or (bo == bs and co > cs):
                oi = int(merged.at[i, idx_self])
                fix[oi] = b

    proc(m12, "_1", "_2", fix1)
    proc(m21, "_2", "_1", fix2)

    n1 = _apply_fixes(df1, fix1, b1)
    n2 = _apply_fixes(df2, fix2, b2)
    return int(n1 + n2)


def _smooth_majority_vote(beh_csv: Path, *, window_seconds: float, logger: logging.Logger) -> int:
    df = _safe_read_csv(beh_csv)
    if df is None or df.empty:
        return 0
    if "timestamp" not in df.columns or "behavior_label" not in df.columns:
        return 0

    df = norm_timestamp(df)
    ts = df["timestamp"]
    labels = df["behavior_label"].astype(str).values.copy()
    n = len(labels)
    if n < 3:
        return 0

    ts_sec = (ts - ts.iloc[0]).dt.total_seconds().values
    window = float(window_seconds)
    total_fixed = 0

    for _pass in range(10):
        new = labels.copy()
        left = 0
        right = 0
        fixed = 0

        for i in range(n):
            t = ts_sec[i]
            while left < n and ts_sec[left] < t - window:
                left += 1
            while right < n and ts_sec[right] <= t + window:
                right += 1

            counts: dict[str, int] = {}
            for j in range(left, right):
                lbl = labels[j]
                if lbl != "00_invalid":
                    counts[lbl] = counts.get(lbl, 0) + 1
            if not counts:
                continue
            maj = max(counts, key=counts.get)
            if labels[i] != maj:
                new[i] = maj
                fixed += 1

        labels = new
        total_fixed += fixed
        if fixed == 0:
            break

    if total_fixed > 0:
        df["behavior_label"] = labels
        df.to_csv(beh_csv, index=False)
    return int(total_fixed)


def _smooth_within_track(beh_csv: Path, *, min_bout_seconds: float, logger: logging.Logger) -> int:
    df = _safe_read_csv(beh_csv)
    if df is None or df.empty:
        return 0
    if "timestamp" not in df.columns or "behavior_label" not in df.columns:
        return 0

    df = norm_timestamp(df)
    ts = df["timestamp"]
    labels = df["behavior_label"].astype(str).values.copy()
    n = len(labels)
    if n < 3:
        return 0

    total = 0
    for _ in range(20):
        bouts = _get_bouts(labels, ts)
        fixed = 0
        for bi in range(1, len(bouts) - 1):
            s, e, lbl, dur = bouts[bi]
            prev_lbl = bouts[bi - 1][2]
            next_lbl = bouts[bi + 1][2]
            if prev_lbl == next_lbl and lbl != prev_lbl and dur < float(min_bout_seconds) and prev_lbl != "00_invalid":
                labels[s:e] = prev_lbl
                fixed += (e - s)
        total += fixed
        if fixed == 0:
            break

    if total > 0:
        df["behavior_label"] = labels
        df.to_csv(beh_csv, index=False)
    return int(total)


def _smooth_short_bouts(
    beh_csv: Path,
    *,
    max_short_bout_seconds: float,
    context_window_seconds: float,
    logger: logging.Logger,
) -> int:
    df = _safe_read_csv(beh_csv)
    if df is None or df.empty:
        return 0
    if "timestamp" not in df.columns or "behavior_label" not in df.columns:
        return 0

    df = norm_timestamp(df)
    ts = df["timestamp"]
    labels = df["behavior_label"].astype(str).values.copy()
    n = len(labels)
    if n < 3:
        return 0

    ts_sec = (ts - ts.iloc[0]).dt.total_seconds().values
    total = 0

    for _ in range(20):
        bouts = _get_bouts(labels, ts)
        fixed = 0
        for bi, (bs, be, bl, bd) in enumerate(bouts):
            if bd >= float(max_short_bout_seconds):
                continue

            mid = 0.5 * (ts_sec[bs] + ts_sec[be - 1])
            lo = mid - float(context_window_seconds)
            hi = mid + float(context_window_seconds)

            dur_by: dict[str, float] = {}
            for bj, (s, e, lbl, dur) in enumerate(bouts):
                if bj == bi or lbl == "00_invalid":
                    continue
                t0 = ts_sec[s]
                t1 = ts_sec[e - 1]
                ol = max(t0, lo)
                oh = min(t1, hi)
                if oh >= ol:
                    d = oh - ol
                    if d < 0.01:
                        d = max(0.3, float(dur)) if dur > 0 else 0.3
                    dur_by[lbl] = dur_by.get(lbl, 0.0) + float(d)

            if not dur_by:
                continue
            dom = max(dur_by, key=dur_by.get)
            dom_dur = dur_by[dom]
            tot = sum(dur_by.values())
            if bl != dom and dom_dur > 0.6 * tot:
                labels[bs:be] = dom
                fixed += (be - bs)

        total += fixed
        if fixed == 0:
            break

    if total > 0:
        df["behavior_label"] = labels
        df.to_csv(beh_csv, index=False)
    return int(total)


def smooth_behavior_cross_camera(
    tracklets_cam1: list[dict],
    tracklets_cam2: list[dict],
    *,
    cfg: CrossCamConfig = CrossCamConfig(),
    logger: Optional[logging.Logger] = None,
) -> tuple[dict[str, int], pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)

    all_tracklets = tracklets_cam1 + tracklets_cam2

    all_beh: set[Path] = set()
    for t in all_tracklets:
        csvp = t.get("track_csv_path", "")
        if not csvp:
            continue
        beh = Path(csvp).with_name(Path(csvp).stem + "_behavior.csv")
        if beh.exists():
            all_beh.add(beh)

    for beh_csv in sorted(all_beh):
        df = _safe_read_csv(beh_csv)
        if df is None or df.empty:
            continue
        changed = False

        if "behavior_label_raw" in df.columns:
            if "behavior_label" in df.columns and "behavior_label_old" not in df.columns:
                df["behavior_label_old"] = df["behavior_label"]
                changed = True
            raw_vals = df["behavior_label_raw"].astype(str)
            if "behavior_label" not in df.columns or not df["behavior_label"].astype(str).equals(raw_vals):
                df["behavior_label"] = raw_vals
                changed = True
        elif "behavior_label" in df.columns and "behavior_label_old" not in df.columns:
            df["behavior_label_old"] = df["behavior_label"]
            changed = True

        if changed:
            df.to_csv(beh_csv, index=False)

    xcid_to_tracks: dict[int, list[dict]] = defaultdict(list)
    for t in all_tracklets:
        xcid = t.get("cross_cam_id")
        tf = t.get("track_filename", "")
        csvp = t.get("track_csv_path", "")
        if xcid is None or not tf or not csvp:
            continue
        xcid_to_tracks[int(xcid)].append(t)

    cam1_files = {t.get("track_filename") for t in tracklets_cam1}
    cam2_files = {t.get("track_filename") for t in tracklets_cam2}

    track_corrections: dict[str, int] = {}

    for xcid, members in sorted(xcid_to_tracks.items(), key=lambda kv: kv[0]):
        a = [m for m in members if m.get("track_filename") in cam1_files]
        b = [m for m in members if m.get("track_filename") in cam2_files]
        if not a or not b:
            continue
        for t1 in a:
            for t2 in b:
                nfix = _smooth_behavior_pair(t1, t2, tol_ms=cfg.behavior_pair_tolerance_ms, logger=logger)
                if nfix > 0:
                    tf1 = t1.get("track_filename", "")
                    tf2 = t2.get("track_filename", "")
                    if tf1:
                        track_corrections[tf1] = track_corrections.get(tf1, 0) + nfix
                    if tf2:
                        track_corrections[tf2] = track_corrections.get(tf2, 0) + nfix

    for beh_csv in sorted(all_beh):
        n = _smooth_majority_vote(beh_csv, window_seconds=cfg.majority_window_seconds, logger=logger)
        if n > 0:
            tf = beh_csv.stem.replace("_behavior", "")
            track_corrections[tf] = track_corrections.get(tf, 0) + n

    for beh_csv in sorted(all_beh):
        n = _smooth_within_track(beh_csv, min_bout_seconds=cfg.within_track_min_bout_seconds, logger=logger)
        if n > 0:
            tf = beh_csv.stem.replace("_behavior", "")
            track_corrections[tf] = track_corrections.get(tf, 0) + n

    for beh_csv in sorted(all_beh):
        n = _smooth_short_bouts(
            beh_csv,
            max_short_bout_seconds=cfg.shortbout_max_seconds,
            context_window_seconds=cfg.shortbout_context_seconds,
            logger=logger,
        )
        if n > 0:
            tf = beh_csv.stem.replace("_behavior", "")
            track_corrections[tf] = track_corrections.get(tf, 0) + n

    bout_df = summarize_behavior_bouts(all_tracklets, logger=logger)
    track_to_beh_csv: dict[str, Path] = {}
    for t in all_tracklets:
        tf = t.get("track_filename", "")
        csvp = t.get("track_csv_path", "")
        if not tf or not csvp:
            continue
        beh = Path(csvp).with_name(Path(csvp).stem + "_behavior.csv")
        if beh.exists():
            track_to_beh_csv[tf] = beh

    bout_df, n_bout_conflict_fixed, bout_track_corrections = _smooth_bout_conflicts(
        bout_df,
        track_to_beh_csv=track_to_beh_csv,
        min_context_seconds=0.0,
        max_gap_seconds=max(cfg.shortbout_max_seconds, 60.0),
        focus_keyword="sleeping",
        logger=logger,
    )
    for tf, nfix in bout_track_corrections.items():
        track_corrections[tf] = track_corrections.get(tf, 0) + int(nfix)
    if n_bout_conflict_fixed > 0:
        logger.info("Cross-camera bout conflict smoothing fixed %d short outlier bouts.", n_bout_conflict_fixed)
    return track_corrections, bout_df


# --------------------------------------------------------------------------- #
# Behavior bout summary
# --------------------------------------------------------------------------- #

def _smooth_bout_conflicts(
    bout_df: pd.DataFrame,
    *,
    track_to_beh_csv: Optional[dict[str, Path]] = None,
    min_context_seconds: float = 0.0,
    max_gap_seconds: float = 300.0,
    focus_keyword: Optional[str] = "sleeping",
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    """Smooth short outlier behavior bouts across cameras per individual.

    Example:
        sleeping_right (20m) -> sleeping_left (2m) -> sleeping_right (30m)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if bout_df is None or bout_df.empty:
        return bout_df, 0, {}
    if "behavior_label" not in bout_df.columns or "start_time" not in bout_df.columns or "end_time" not in bout_df.columns:
        return bout_df, 0, {}

    out = bout_df.copy()
    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    out = out.dropna(subset=["start_time", "end_time"]).copy()
    if out.empty:
        return out, 0, {}

    invalid_ids = {"", "unknown", "invalid", "spurious", "confused"}
    if "identity_label" in out.columns:
        id_vals = out["identity_label"].astype(str).str.strip()
    else:
        id_vals = pd.Series([""] * len(out), index=out.index)

    if "cross_cam_stitched_id" in out.columns:
        xcsid_vals = out["cross_cam_stitched_id"]
    else:
        xcsid_vals = pd.Series([np.nan] * len(out), index=out.index)

    if "cross_cam_id" in out.columns:
        xcid_vals = out["cross_cam_id"]
    else:
        xcid_vals = pd.Series([np.nan] * len(out), index=out.index)

    subject_keys: list[str] = []
    for i in out.index:
        idv = id_vals.loc[i]
        if idv and idv.lower() not in invalid_ids:
            subject_keys.append(f"id:{idv}")
            continue
        xv = xcsid_vals.loc[i]
        if pd.notna(xv):
            subject_keys.append(f"xcsid:{xv}")
            continue
        cv = xcid_vals.loc[i]
        if pd.notna(cv):
            subject_keys.append(f"xcid:{cv}")
            continue
        subject_keys.append(f"row:{i}")
    out["_subject_key"] = subject_keys

    out = out.sort_values(["_subject_key", "start_time", "end_time"], kind="mergesort").copy()
    labels = out["behavior_label"].astype(str).copy()
    fixed = 0
    focus = (focus_keyword or "").strip().lower()
    fixes: list[dict] = []

    def _row_duration_seconds(idx: int) -> float:
        if "duration_s" in out.columns and pd.notna(out.loc[idx, "duration_s"]):
            return float(out.loc[idx, "duration_s"])
        return float((out.loc[idx, "end_time"] - out.loc[idx, "start_time"]).total_seconds())

    for _, g in out.groupby("_subject_key", sort=False):
        idx = g.index.to_list()
        if len(idx) < 3:
            continue

        for _ in range(10):
            changed = 0
            for j in range(1, len(idx) - 1):
                i_prev = idx[j - 1]
                i_cur = idx[j]
                i_next = idx[j + 1]

                prev_lbl = str(labels.loc[i_prev])
                cur_lbl = str(labels.loc[i_cur])
                next_lbl = str(labels.loc[i_next])

                if cur_lbl == "00_invalid":
                    continue
                if prev_lbl != next_lbl or cur_lbl == prev_lbl:
                    continue

                dur_cur = _row_duration_seconds(i_cur)
                dur_prev = _row_duration_seconds(i_prev)
                dur_next = _row_duration_seconds(i_next)

                if float(min_context_seconds) > 0.0:
                    if dur_prev < float(min_context_seconds) or dur_next < float(min_context_seconds):
                        continue
                if (dur_prev + dur_next) <= (2.0 * dur_cur):
                    continue

                gap_prev = float((out.loc[i_cur, "start_time"] - out.loc[i_prev, "end_time"]).total_seconds())
                gap_next = float((out.loc[i_next, "start_time"] - out.loc[i_cur, "end_time"]).total_seconds())
                if gap_prev > float(max_gap_seconds) or gap_next > float(max_gap_seconds):
                    continue

                if focus:
                    if focus not in prev_lbl.lower() or focus not in cur_lbl.lower():
                        continue

                labels.loc[i_cur] = prev_lbl
                fixes.append(
                    {
                        "row_idx": int(i_cur),
                        "track_filename": str(out.loc[i_cur, "track_filename"]) if "track_filename" in out.columns else "",
                        "start_time": out.loc[i_cur, "start_time"],
                        "end_time": out.loc[i_cur, "end_time"],
                        "old_label": cur_lbl,
                        "new_label": prev_lbl,
                    }
                )
                changed += 1

            fixed += changed
            if changed == 0:
                break

    if fixed > 0:
        out["behavior_label"] = labels
        if "duration_s" in out.columns:
            out["duration_s"] = (out["end_time"] - out["start_time"]).dt.total_seconds().round(1)

    track_frame_corrections: dict[str, int] = {}
    if fixes and track_to_beh_csv:
        fixes_by_track: dict[str, list[dict]] = defaultdict(list)
        for fx in fixes:
            tf = fx.get("track_filename", "")
            if tf:
                fixes_by_track[tf].append(fx)

        for tf, tf_fixes in fixes_by_track.items():
            beh_csv = track_to_beh_csv.get(tf)
            if beh_csv is None or not Path(beh_csv).exists():
                continue
            df = _safe_read_csv(Path(beh_csv))
            if df is None or df.empty:
                continue
            if "timestamp" not in df.columns or "behavior_label" not in df.columns:
                continue

            df = norm_timestamp(df)
            lbl = df["behavior_label"].astype(str).copy()
            changed_frames = 0

            for fx in tf_fixes:
                t0 = pd.Timestamp(fx["start_time"])
                t1 = pd.Timestamp(fx["end_time"])
                old_lbl = str(fx["old_label"])
                new_lbl = str(fx["new_label"])
                mask = (df["timestamp"] >= t0) & (df["timestamp"] <= t1)
                mask &= (lbl == old_lbl)
                n = int(mask.sum())
                if n <= 0:
                    continue
                lbl.loc[mask] = new_lbl
                changed_frames += n

            if changed_frames > 0:
                df["behavior_label"] = lbl.values
                df.to_csv(beh_csv, index=False)
                track_frame_corrections[tf] = track_frame_corrections.get(tf, 0) + changed_frames

    out = out.drop(columns=["_subject_key"], errors="ignore")
    out = out.sort_values(["start_time", "end_time"], kind="mergesort").reset_index(drop=True)
    return out, int(fixed), track_frame_corrections


def summarize_behavior_bouts(
    tracklets: list[dict],
    *,
    min_bout_seconds: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger(__name__)

    rows: list[dict] = []
    seen: set[Path] = set()

    for t in tracklets:
        csvp = t.get("track_csv_path", "")
        if not csvp:
            continue
        beh = Path(csvp).with_name(Path(csvp).stem + "_behavior.csv")
        if not beh.exists() or beh in seen:
            continue
        seen.add(beh)

        df = _safe_read_csv(beh)
        if df is None or df.empty:
            continue
        if "timestamp" not in df.columns or "behavior_label" not in df.columns:
            continue

        df = norm_timestamp(df)
        ts = df["timestamp"]
        labels = df["behavior_label"].astype(str).values
        n = len(labels)
        if n == 0:
            continue

        cam_id = t.get("camera_id", t.get("camera", ""))
        identity = t.get("identity_label", "")
        track_fn = t.get("track_filename", "")

        run_start = 0
        for i in range(1, n):
            if labels[i] != labels[run_start]:
                dur = (ts.iloc[i - 1] - ts.iloc[run_start]).total_seconds()
                if dur >= float(min_bout_seconds):
                    rows.append(
                        {
                            "start_time": ts.iloc[run_start],
                            "end_time": ts.iloc[i - 1],
                            "duration_s": round(float(dur), 1),
                            "behavior_label": labels[run_start],
                            "n_frames": i - run_start,
                            "cam_id": cam_id,
                            "identity_label": identity,
                            "track_filename": track_fn,
                        }
                    )
                run_start = i

        dur = (ts.iloc[n - 1] - ts.iloc[run_start]).total_seconds()
        if dur >= float(min_bout_seconds):
            rows.append(
                {
                    "start_time": ts.iloc[run_start],
                    "end_time": ts.iloc[n - 1],
                    "duration_s": round(float(dur), 1),
                    "behavior_label": labels[run_start],
                    "n_frames": n - run_start,
                    "cam_id": cam_id,
                    "identity_label": identity,
                    "cross_cam_id": t.get("cross_cam_id"),
                    "cross_cam_stitched_id": t.get("cross_cam_stitched_id"),
                    "track_filename": track_fn,
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["cam_id", "start_time"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Matching summary
# --------------------------------------------------------------------------- #

def summarize_cross_cam_match(summary_df: pd.DataFrame, camera_ids: list[str], *, logger: Optional[logging.Logger] = None) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
    if summary_df.empty:
        logger.info("No matching results.")
        return
    n_matched = int((summary_df["status"] == "matched").sum())
    n_unmatched = int(len(summary_df) - n_matched)
    logger.info("Cross-camera match summary cam %s <-> %s: matched=%d, unmatched=%d", camera_ids[0], camera_ids[1], n_matched, n_unmatched)


# --------------------------------------------------------------------------- #
# Monitoring: concurrent xcsid count
# --------------------------------------------------------------------------- #
def find_intervals_over_max_concurrency(
    tracklets: list[dict],
    *,
    id_key: str,
    max_concurrent: int,
    bin_seconds: float,
) -> pd.DataFrame:
    rows = []
    for t in tracklets:
        xid = t.get(id_key)
        if xid is None:
            continue
        try:
            s = pd.Timestamp(t["start_timestamp"])
            e = pd.Timestamp(t["end_timestamp"])
        except Exception:
            continue

        # Keep as label (string/int), do not cast to int
        xid_label = str(xid)

        rows.append((s, e, xid_label))

    if not rows:
        return pd.DataFrame()

    t0 = min(r[0] for r in rows).floor(f"{bin_seconds}s")
    t1 = max(r[1] for r in rows).ceil(f"{bin_seconds}s")
    bins = pd.date_range(t0, t1, freq=f"{bin_seconds}s")

    events = pd.DataFrame(rows, columns=["start", "end", "xid"])
    counts = []
    for tb in bins:
        te = tb + pd.Timedelta(seconds=float(bin_seconds))
        active = events[(events["start"] < te) & (events["end"] > tb)]
        counts.append((tb, int(active["xid"].nunique())))

    cdf = pd.DataFrame(counts, columns=["tbin", "n_active"])
    bad = cdf[cdf["n_active"] > int(max_concurrent)].copy()
    if bad.empty:
        return bad

    bad["run"] = (bad["tbin"].diff() != pd.Timedelta(seconds=float(bin_seconds))).cumsum()
    out_rows = []
    for _, g in bad.groupby("run"):
        out_rows.append(
            {
                "start": g["tbin"].iloc[0],
                "end": g["tbin"].iloc[-1] + pd.Timedelta(seconds=float(bin_seconds)),
                "max_active": int(g["n_active"].max()),
                "n_bins": int(len(g)),
            }
        )
    return pd.DataFrame(out_rows).sort_values("start").reset_index(drop=True)

# --------------------------------------------------------------------------- #
# Top-level orchestrator (JSON -> JSON)
# --------------------------------------------------------------------------- #

def run_cross_camera_matching_v2(
    *,
    record_root: str | Path,
    camera_ids: list[str],
    start_datetime,
    end_datetime,
    known_individuals: Optional[list[str]] = None,
    gallery_path: Optional[str | Path] = None,
    cfg: CrossCamConfig = CrossCamConfig(),
    logger: Optional[logging.Logger] = None,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    from post_processing.tools.utils import load_tracklet_json_for_camera

    if logger is None:
        logger = logging.getLogger(__name__)

    if len(camera_ids) != 2:
        raise ValueError("camera_ids must have exactly 2 elements")

    known_individuals = known_individuals or []

    record_root = Path(record_root)
    start_datetime = pd.Timestamp(start_datetime)
    end_datetime = pd.Timestamp(end_datetime)
    cam1_id, cam2_id = camera_ids

    json_data: dict[str, tuple[Path, list]] = {}
    for cam_id in camera_ids:
        jp, td = load_tracklet_json_for_camera(
            record_root=record_root,
            cam_id=cam_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
        json_data[cam_id] = (jp, td)
        logger.info("Loaded %d tracklets for cam %s (%s)", len(td), cam_id, jp.name)

    tracklets_cam1 = json_data[cam1_id][1]
    tracklets_cam2 = json_data[cam2_id][1]

    track_to_xcid, match_df = match_tracks_cross_camera(
        tracklets_cam1,
        tracklets_cam2,
        cam1_id,
        cam2_id,
        cfg=cfg,
        logger=logger,
    )

    for cam_id in camera_ids:
        _, tds = json_data[cam_id]
        for t in tds:
            tf = t.get("track_filename", "")
            if tf in track_to_xcid:
                t["cross_cam_id"] = int(track_to_xcid[tf])

    track_to_xcsid = compute_cross_cam_stitched_id(
        json_data[cam1_id][1],
        json_data[cam2_id][1],
        cam1_id,
        cam2_id,
        cfg=cfg,
        logger=logger,
    )

    for cam_id in camera_ids:
        _, tds = json_data[cam_id]
        for t in tds:
            tf = t.get("track_filename", "")
            if tf in track_to_xcsid:
                t["cross_cam_stitched_id"] = int(track_to_xcsid[tf])

    track_to_ind = compute_cross_cam_individual(
        json_data[cam1_id][1],
        json_data[cam2_id][1],
        n_individuals=max(2, len(known_individuals) if known_individuals else 2),
        known_individuals=known_individuals,
        logger=logger,
    )
    for cam_id in camera_ids:
        _, tds = json_data[cam_id]
        for t in tds:
            tf = t.get("track_filename", "")
            if tf in track_to_ind:
                t["cross_cam_individual"] = int(track_to_ind[tf])

    if gallery_path is not None:
        vote_identity_by_xcsid_reid(
            json_data[cam1_id][1],
            json_data[cam2_id][1],
            gallery_path=gallery_path,
            known_individuals=known_individuals,
            cfg=cfg,
            logger=logger,
        )

    track_to_label, conflict_df = assign_identity_by_xcsid_strict(
        json_data[cam1_id][1],
        json_data[cam2_id][1],
        known_individuals=known_individuals,
        logger=logger,
    )

    skip = {"unknown", "invalid", "spurious", "confused", ""}
    known_set = set(known_individuals)
    for cam_id in camera_ids:
        _, tds = json_data[cam_id]
        for t in tds:
            tf = t.get("track_filename", "")
            if not tf:
                continue
            lbl = track_to_label.get(tf, "unknown")
            if lbl not in skip:
                t["identity_label"] = lbl
            else:
                ### double-check that we don't have a conflict where both tracks in an xcsid group are labeled as the same known individual 
                # (should not happen with the strict assignment, but just in case)
                if len(known_individuals) == 2:
                    t["identity_label"] = "confused"  # if only 2 known individuals, any conflict means we can't disambiguate, so label as confused
                elif len(known_individuals) ==1:
                    prev_lbl = str(t.get("voted_track_label", "unknown"))
                    ### Thai -> if matches with previous voted label, keep it, otherwise set to unknown (to avoid conflicts where strict assignment fails to assign a known label but the vote suggests a known label)
                    t["identity_label"] = prev_lbl if prev_lbl in known_set else "unknown"
                else:
                    t["identity_label"] = "unknown"
                
    _, bout_df = smooth_behavior_cross_camera(
        json_data[cam1_id][1],
        json_data[cam2_id][1],
        cfg=cfg,
        logger=logger,
    )

    all_tracklets = json_data[cam1_id][1] + json_data[cam2_id][1]
    bad_xcsid = find_intervals_over_max_concurrency(
        all_tracklets,
        id_key="cross_cam_stitched_id",
        max_concurrent=2,
        bin_seconds=cfg.bin_seconds,
    )
    if not bad_xcsid.empty:
        logger.warning("Detected >2 concurrent xcsid intervals:\n%s", bad_xcsid.to_string(index=False))

    bad_lbl = find_intervals_over_max_concurrency(
        [t for t in all_tracklets if t.get("identity_label") not in {"unknown", "invalid", "spurious", "confused", ""}],
        id_key="identity_label",
        max_concurrent=1,
        bin_seconds=cfg.bin_seconds,
    )
    if not bad_lbl.empty:
        logger.warning("Detected identity_label concurrency violations:\n%s", bad_lbl.to_string(index=False))


    for cam_id in camera_ids:
        json_path, tds = json_data[cam_id]
        with open(json_path, "w") as f:
            json.dump(tds, f, indent=2)
        logger.info("Saved JSON: %s", json_path)

    return track_to_xcid, match_df, bout_df
