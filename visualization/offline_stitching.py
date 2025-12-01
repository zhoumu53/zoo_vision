"""
Offline Track Stitching Pipeline

This script performs hierarchical track stitching on JSONL outputs from the tracking stage:

Stage 1: Within-video stitching using ReID features
  - Merge fragmented tracks in the same video/camera
  - Uses ReID model to compute feature similarity
  - Fixes ID switches caused by tracking errors

Stage 2: Cross-camera stitching using room constraints
  - Merge tracks across cameras in the same room
  - Room pairs: (016, 019) and (017, 018)
  - Uses temporal overlap, size similarity, and ReID features

Outputs:
  - Fixed JSONL with both original_track_id and stitched_track_id
  - Summary statistics showing merge counts per stage
  - Visualization frames comparing before/after stitching
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Import utilities from your existing codebase
sys.path.insert(0, str(Path(__file__).parent))
import utils
from utils import build_reid_model, preprocess_patches, extract_features, load_gallery_database, match_gallery, SOCIAL_GROUPS, SOCIAL_GROUPS

# Room configuration
ROOM_PAIRS = {
    "room1": ["016", "019"],
    "room2": ["017", "018"],
}

CAMERA_TO_ROOM = {
    "016": "room1",
    "019": "room1",
    "017": "room2",
    "018": "room2",
}


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("offline_stitching")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline track stitching with ReID and room constraints"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing tracking results (auto-discovers .jsonl files).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for stitched results (default: {input-dir}/stitched_tracks).",
    )
    parser.add_argument(
        "--reid-config",
        required=True,
        help="Path to PoseGuidedReID config (.yml).",
    )
    parser.add_argument(
        "--reid-checkpoint",
        required=True,
        help="PoseGuidedReID checkpoint (.pth).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for ReID model (cuda / cpu).",
    )
    parser.add_argument(
        "--gallery",
        default=None,
        help="Gallery features npz file for identity anchoring (optional).",
    )
    parser.add_argument(
        "--gallery-device",
        default="cpu",
        help="Device for gallery embeddings (cpu / cuda).",
    )
    parser.add_argument(
        "--gallery-threshold",
        type=float,
        default=0.7,
        help="Minimum cosine similarity to accept gallery identity match.",
    )
    parser.add_argument(
        "--gallery-top-k",
        type=int,
        default=3,
        help="Number of top gallery matches to store per track.",
    )
    parser.add_argument(
        "--stage1-sim-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for within-video stitching (Stage 1).",
    )
    parser.add_argument(
        "--stage1-time-gap",
        type=float,
        default=15.0,
        help="Maximum time gap (seconds) for within-video stitching (Stage 1).",
    )
    parser.add_argument(
        "--stage1-size-ratio",
        type=float,
        default=0.4,
        help="Minimum size ratio for within-video stitching (Stage 1).",
    )
    parser.add_argument(
        "--stage2-sim-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for cross-camera stitching (Stage 2).",
    )
    parser.add_argument(
        "--stage2-time-gap",
        type=float,
        default=5.0,
        help="Maximum time gap (seconds) for cross-camera stitching (Stage 2).",
    )
    parser.add_argument(
        "--stage2-size-ratio",
        type=float,
        default=0.6,
        help="Minimum size ratio for cross-camera stitching (Stage 2).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only first N frames per track (for debugging/testing).",
    )
    parser.add_argument(
        "--use-social-groups",
        action="store_true",
        help="Enable social group validation (elephants from different groups shouldn't appear together).",
    )
    parser.add_argument(
        "--social-group-action",
        choices=["report", "remove"],
        default="report",
        help="Action for social group conflicts: 'report' (log only) or 'remove' (remove lower-confidence identity).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate comparison visualizations (before/after stitching).",
    )
    parser.add_argument(
        "--vis-max-frames",
        type=int,
        default=100,
        help="Maximum frames to visualize per video.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity.",
    )
    return parser.parse_args()


# ==============================================================================
# Data Loading & Parsing
# ==============================================================================


def discover_jsonl_files(input_dir: Path, logger: logging.Logger) -> List[Path]:
    """Discover all .jsonl files in the input directory and subdirectories."""
    jsonl_files = []
    
    # Search in the directory and immediate subdirectories
    for pattern in ["*.jsonl", "*/*.jsonl"]:
        jsonl_files.extend(input_dir.glob(pattern))
    
    # Filter to only include track files (exclude stitched output)
    track_files = [
        f for f in jsonl_files 
        if "_tracks.jsonl" in f.name and f.name != "stitched_tracks.jsonl"
    ]
    
    if not track_files:
        # Fallback: accept any .jsonl file except stitched output
        track_files = [f for f in jsonl_files if f.name != "stitched_tracks.jsonl"]
    
    logger.info("Discovered %d JSONL files in %s", len(track_files), input_dir)
    for f in sorted(track_files):
        logger.info("  - %s", f.name)
    
    return sorted(track_files)


def load_jsonl(path: Path, max_frames: Optional[int] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load JSONL file and return (metadata, records).
    
    First line contains metadata with video path.
    Subsequent lines contain detection results.
    Only loads frames that have tracks (skips empty frames for speed).
    
    Args:
        path: Path to JSONL file
        max_frames: Optional limit on number of frames to load (for debugging)
    """
    metadata = {}
    records = []
    
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            if i == 0 and "meta" in data:
                # First line is metadata
                metadata = data["meta"]
            elif "results" in data:
                # Only append records that have tracks (skip empty frames)
                result = data["results"]
                if "tracks" in result and result["tracks"]:
                    records.append(result)
                    # Stop if we've reached max_frames limit
                    if max_frames is not None and len(records) >= max_frames:
                        break
    
    return metadata, records


def group_detections_by_track(
    file_records: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]]]], 
    logger: logging.Logger
) -> Dict[Tuple[str, str, int], List[Dict[str, Any]]]:
    """
    Group detections by (video_path, camera_id, track_id).
    
    Args:
        file_records: List of (video_path, camera_id, metadata, records) tuples
    
    Returns:
        Dict mapping (video_path, camera_id, track_id) -> list of detection dicts
    """
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    
    records_with_tracks = 0
    total_detections = 0
    
    for video_path, camera_id, metadata, records in file_records:
        for record in records:
            # Records are already filtered (only have tracks)
            tracks = record.get("tracks", [])
            if not tracks or len(tracks) == 0:
                continue
            
            records_with_tracks += 1
            frame_idx = record.get("frame_idx", 0)
            timestamp = record.get("timestamp", "")
            
            for track in tracks:
                # Use canonical_track_id (the stable ID after tracking)
                track_id = track.get("canonical_track_id")
                if track_id is None or track_id < 0:
                    continue
                
                total_detections += 1
                
                # Add frame-level info to each detection
                detection = {
                    **track,
                    "track_id": track_id,  # Normalize field name
                    "video_path": video_path,
                    "camera_id": camera_id,
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                }
                
                key = (video_path, camera_id, track_id)
                groups[key].append(detection)
    
    logger.info("Grouping stats:")
    logger.info("  Records with tracks: %d", records_with_tracks)
    logger.info("  Total detections: %d", total_detections)
    logger.info("  Unique tracks: %d", len(groups))
    
    return groups


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO format timestamp."""
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        return datetime.now()


# ==============================================================================
# ReID Feature Extraction
# ==============================================================================


class ReIDFeatureExtractor:
    """Extract ReID features from detection crops."""
    
    def __init__(
        self,
        reid_config: str,
        reid_checkpoint: str,
        device: torch.device,
        logger: logging.Logger,
    ):
        self.device = device
        self.logger = logger
        
        # Load ReID model
        self.model, self.transform = build_reid_model(
            reid_config, 
            reid_checkpoint, 
            num_classes=5,
            device=device,
            logger=logger
        )
        self.model.eval()
        self.logger.info("Loaded ReID model on %s", device)
    
    def extract_from_detections(
        self,
        detections: List[Dict[str, Any]],
        video_path: str,
    ) -> torch.Tensor:
        """
        Extract ReID features for a list of detections from the same track.
        
        Returns averaged feature vector (1, feat_dim).
        """
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.warning("Cannot open video: %s", video_path)
            return torch.zeros(1, 2048, device=self.device)  # Dummy feature
        
        features_list = []
        
        # Sample frames uniformly (max 10 frames per track)
        sampled_dets = detections[:: max(1, len(detections) // 10)][:10]
        
        for det in sampled_dets:
            frame_idx = det.get("frame_idx", 0)
            bbox = det.get("bbox")
            
            if bbox is None or len(bbox) != 4:
                continue
            
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Extract features using preprocess_patches
            with torch.no_grad():
                # preprocess_patches expects (frame, boxes, transform)
                boxes = np.array([bbox])
                patches, _, _ = preprocess_patches(frame, boxes, self.transform)
                
                if not patches:
                    continue
                
                # Stack patches and move to device
                patches_tensor = torch.stack(patches).to(self.device)
                feat = extract_features(self.model, patches_tensor)
                features_list.append(feat)
        
        cap.release()
        
        if not features_list:
            return torch.zeros(1, 2048, device=self.device)
        
        # Average features
        avg_feat = torch.stack(features_list).mean(dim=0)
        # L2 normalize
        avg_feat = avg_feat / (torch.norm(avg_feat, dim=1, keepdim=True) + 1e-9)
        
        return avg_feat


# ==============================================================================
# Track Statistics
# ==============================================================================


def compute_track_stats(
    detections: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute statistics for a track."""
    if not detections:
        return {
            "start_time": None,
            "end_time": None,
            "median_area": 0.0,
            "num_frames": 0,
        }
    
    # Temporal range
    timestamps = [parse_timestamp(d["timestamp"]) for d in detections]
    start_time = min(timestamps)
    end_time = max(timestamps)
    
    # Median bbox area
    areas = []
    for d in detections:
        bbox = d.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            areas.append(width * height)
    
    median_area = float(np.median(areas)) if areas else 0.0
    
    return {
        "start_time": start_time,
        "end_time": end_time,
        "median_area": median_area,
        "num_frames": len(detections),
    }


# ==============================================================================
# Union-Find for Track Merging
# ==============================================================================


class UnionFind:
    """Disjoint set data structure for track merging."""
    
    def __init__(self, keys: List[Any]):
        self.parent = {k: k for k in keys}
    
    def find(self, x: Any) -> Any:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: Any, y: Any) -> None:
        """Union by smaller key (for determinism)."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        if root_x < root_y:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
    
    def get_clusters(self) -> Dict[Any, List[Any]]:
        """Return mapping from root -> list of members."""
        clusters = defaultdict(list)
        for key in self.parent:
            root = self.find(key)
            clusters[root].append(key)
        return dict(clusters)


# ==============================================================================
# Stage 1: Within-Video Stitching (ReID-based)
# ==============================================================================


def stitch_within_video(
    track_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    reid_extractor: ReIDFeatureExtractor,
    sim_threshold: float,
    time_gap: float,
    size_ratio: float,
    logger: logging.Logger,
) -> Tuple[Dict[Tuple[str, str, int], List[Dict[str, Any]]], int]:
    """
    Stage 1: Merge fragmented tracks within the same video/camera using ReID.
    
    Returns:
        (stitched_groups, num_merges)
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: Within-Video Stitching (ReID-based)")
    logger.info("=" * 80)
    logger.info("  Similarity threshold: %.2f", sim_threshold)
    logger.info("  Time gap threshold: %.1f seconds", time_gap)
    logger.info("  Size ratio threshold: %.2f", size_ratio)
    
    # Group by (video_path, camera_id) first
    video_camera_groups = defaultdict(list)
    for key, dets in track_groups.items():
        video_path, camera_id, track_id = key
        video_camera_groups[(video_path, camera_id)].append((key, dets))
    
    stitched_groups = {}
    total_merges = 0
    
    # Process each video/camera combination
    for (video_path, camera_id), tracks in tqdm(
        video_camera_groups.items(),
        desc="Stage 1: Within-video stitching",
    ):
        logger.info("  Processing: %s (camera %s)", Path(video_path).stem, camera_id)
        
        if len(tracks) < 2:
            # No merging possible
            for key, dets in tracks:
                stitched_groups[key] = dets
            continue
        
        # Compute stats for each track
        track_stats = {}
        for key, dets in tracks:
            track_stats[key] = compute_track_stats(dets)
        
        # Extract ReID features for each track
        track_features = {}
        for key, dets in tracks:
            feat = reid_extractor.extract_from_detections(dets, video_path)
            track_features[key] = feat
        
        # Union-Find for merging
        uf = UnionFind([key for key, _ in tracks])
        
        # Compare all pairs
        keys = [key for key, _ in tracks]
        n = len(keys)
        
        for i in range(n):
            key1 = keys[i]
            stats1 = track_stats[key1]
            feat1 = track_features[key1]
            
            for j in range(i + 1, n):
                key2 = keys[j]
                stats2 = track_stats[key2]
                feat2 = track_features[key2]
                
                # Check temporal proximity
                start1, end1 = stats1["start_time"], stats1["end_time"]
                start2, end2 = stats2["start_time"], stats2["end_time"]
                
                if start1 is None or end1 is None or start2 is None or end2 is None:
                    continue
                
                latest_start = max(start1, start2)
                earliest_end = min(end1, end2)
                overlap = (earliest_end - latest_start).total_seconds()
                
                gap = None
                if overlap < 0:
                    if end1 < start2:
                        gap = (start2 - end1).total_seconds()
                    else:
                        gap = (start1 - end2).total_seconds()
                
                temporal_match = (overlap >= 0) or (gap is not None and gap <= time_gap)
                
                if not temporal_match:
                    continue
                
                # Check size similarity
                area1, area2 = stats1["median_area"], stats2["median_area"]
                if area1 == 0 or area2 == 0:
                    continue
                
                ratio = min(area1, area2) / max(area1, area2)
                if ratio < size_ratio:
                    continue
                
                # Check ReID similarity
                sim = torch.cosine_similarity(feat1, feat2).item()
                if sim < sim_threshold:
                    continue
                
                # Merge!
                uf.union(key1, key2)
                total_merges += 1
                logger.debug(
                    "    Merged %s <-> %s (sim=%.3f, gap=%.1fs, ratio=%.2f)",
                    key1[2], key2[2], sim, gap or 0.0, ratio,
                )
        
        # Build merged groups
        clusters = uf.get_clusters()
        for root, members in clusters.items():
            all_dets = []
            for key in members:
                dets = next(d for k, d in tracks if k == key)
                all_dets.extend(dets)
            # Sort by frame index
            all_dets.sort(key=lambda d: d.get("frame_idx", 0))
            stitched_groups[root] = all_dets
        
        logger.info(
            "    Tracks: %d -> %d (merged %d)",
            len(tracks), len(clusters), len(tracks) - len(clusters),
        )
    
    logger.info("Stage 1 complete: %d merges total", total_merges)
    logger.info("  Tracks before: %d", len(track_groups))
    logger.info("  Tracks after: %d", len(stitched_groups))
    logger.info("")
    
    return stitched_groups, total_merges


# ==============================================================================
# Gallery Matching (Identity Anchoring)
# ==============================================================================


def match_tracks_to_gallery(
    track_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    reid_extractor: ReIDFeatureExtractor,
    gallery_path: str,
    gallery_device: torch.device,
    gallery_threshold: float,
    top_k: int,
    logger: logging.Logger,
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    """
    Match stitched tracks to gallery database for identity anchoring.
    
    This step assigns known elephant identities to tracks based on ReID feature similarity.
    
    Args:
        track_groups: Stitched track groups from Stage 1
        reid_extractor: ReID feature extractor
        gallery_path: Path to gallery features .npz file
        gallery_device: Device for gallery embeddings
        gallery_threshold: Minimum cosine similarity to accept identity match
        top_k: Number of top matches to store
        logger: Logger instance
    
    Returns:
        Dict mapping track_key -> {
            'gallery_identity': str or None,
            'gallery_score': float or None,
            'gallery_matches': List[(label, score), ...]
        }
    """
    logger.info("=" * 80)
    logger.info("GALLERY MATCHING: Identity Anchoring")
    logger.info("=" * 80)
    logger.info("  Gallery path: %s", gallery_path)
    logger.info("  Similarity threshold: %.2f", gallery_threshold)
    logger.info("  Top-K matches: %d", top_k)
    
    # Load gallery database
    gallery = load_gallery_database(gallery_path, gallery_device)
    logger.info("  Gallery size: %d embeddings", len(gallery.labels))
    logger.info("  Unique identities: %d", len(set(gallery.labels.tolist())))
    
    
    track_identities = {}
    matched_count = 0
    total_tracks = len(track_groups)
    

    # Extract video start time for AM/PM filtering (if needed)
    # For now, we'll pass ampm=None to disable time-based filtering
    # ampm = utils.extract_metadata_from_video_path(next(iter(track_groups.items()))[0])[-1]
    ampm = None
    
    # Process each track
    for key, dets in tqdm(track_groups.items(), desc="Gallery matching"):
        video_path, camera_id, track_id = key
        
        # Extract ReID feature for this track
        feat = reid_extractor.extract_from_detections(dets, video_path)
        
        # Move feature to gallery device
        feat = feat.to(gallery.features.device)
        
        # Match against gallery
        scores_mat, indices_mat = match_gallery(feat, gallery, top_k, ampm=ampm)
        
        # Get top-K matches
        match_scores = scores_mat[0].cpu().tolist()
        match_indices = indices_mat[0].cpu().numpy()
        
        matches = [
            (str(gallery.labels[idx]), float(match_scores[i]))
            for i, idx in enumerate(match_indices)
        ]
        
        # Accept best match if above threshold
        best_label, best_score = matches[0] if matches else (None, 0.0)
        
        if best_score >= gallery_threshold:
            gallery_identity = best_label
            gallery_score = best_score
            matched_count += 1
        else:
            gallery_identity = None
            gallery_score = None
        
        track_identities[key] = {
            'gallery_identity': gallery_identity,
            'gallery_score': gallery_score,
            'gallery_matches': matches,
        }
        
        if gallery_identity:
            logger.debug(
                "  Track (cam %s, id %d): matched to %s (score=%.3f)",
                camera_id, track_id, gallery_identity, gallery_score,
            )
    
    logger.info("Gallery matching complete:")
    logger.info("  Total tracks: %d", total_tracks)
    logger.info("  Matched to gallery: %d (%.1f%%)", matched_count, 100 * matched_count / max(total_tracks, 1))
    logger.info("  Unmatched: %d", total_tracks - matched_count)
    logger.info("")
    
    return track_identities


# ==============================================================================
# Social Group Validation
# ==============================================================================


def build_identity_to_group_map(social_groups: Dict[int, List[str]]) -> Dict[str, int]:
    """Build mapping from identity name to group ID."""
    identity_to_group = {}
    for group_id, members in social_groups.items():
        for member in members:
            identity_to_group[member] = group_id
    return identity_to_group


def validate_social_groups(
    stitched_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    track_identities: Dict[Tuple[str, str, int], Dict[str, Any]],
    social_groups: Dict[int, List[str]],
    action: str,
    logger: logging.Logger,
) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate social group constraints: elephants from different groups shouldn't appear together.
    
    Args:
        stitched_groups: Track groups after stitching
        track_identities: Gallery identity matches for each track
        social_groups: Social group definitions {group_id: [member_names]}
        action: 'report' (log only) or 'remove' (remove lower-confidence matches)
        logger: Logger instance
    
    Returns:
        (updated_track_identities, conflicts_list)
        - updated_track_identities: Modified identities (same as input if action='report')
        - conflicts_list: List of detected conflicts for reporting
    """
    logger.info("=" * 80)
    logger.info("SOCIAL GROUP VALIDATION")
    logger.info("=" * 80)
    logger.info("  Social groups: %s", social_groups)
    logger.info("  Action: %s", action)
    
    # Build identity -> group_id mapping
    identity_to_group = build_identity_to_group_map(social_groups)
    logger.info("  Identity mapping: %s", identity_to_group)
    
    # Group detections by (camera_id, timestamp)
    frame_identities = defaultdict(list)  # (camera_id, timestamp) -> [(track_key, identity, score)]
    
    for track_key, identity_data in track_identities.items():
        identity = identity_data.get('gallery_identity')
        score = identity_data.get('gallery_score')
        
        if not identity:
            continue
        
        # Get all detections for this track
        detections = stitched_groups.get(track_key, [])
        for det in detections:
            camera_id = det.get('camera_id')
            timestamp = det.get('timestamp')
            if camera_id and timestamp:
                frame_identities[(camera_id, timestamp)].append((track_key, identity, score))
    
    # Check for social group conflicts
    conflicts = []
    updated_identities = dict(track_identities)
    
    for (camera_id, timestamp), identity_list in frame_identities.items():
        if len(identity_list) < 2:
            continue  # No conflict possible with single elephant
        
        # Get all groups present in this frame
        groups_present = {}  # group_id -> [(track_key, identity, score)]
        unassigned = []  # Identities not in any social group
        
        for track_key, identity, score in identity_list:
            group_id = identity_to_group.get(identity)
            if group_id is not None:
                if group_id not in groups_present:
                    groups_present[group_id] = []
                groups_present[group_id].append((track_key, identity, score))
            else:
                unassigned.append((track_key, identity, score))
        
        # Conflict: multiple groups present simultaneously
        if len(groups_present) > 1:
            conflict_info = {
                'camera_id': camera_id,
                'timestamp': timestamp,
                'groups': groups_present,
                'unassigned': unassigned,
            }
            conflicts.append(conflict_info)
            
            logger.warning(
                "SOCIAL GROUP CONFLICT at camera %s, timestamp %s:",
                camera_id, timestamp
            )
            for group_id, members in groups_present.items():
                member_strs = [f"{identity} ({score:.2f})" for _, identity, score in members]
                logger.warning("  Group %d: %s", group_id, ", ".join(member_strs))
            
            if action == "remove":
                # Remove lower-confidence identities, keep highest confidence from most represented group
                # Find group with most members
                largest_group = max(groups_present.keys(), key=lambda g: len(groups_present[g]))
                
                # Keep largest group, remove others
                for group_id, members in groups_present.items():
                    if group_id != largest_group:
                        for track_key, identity, score in members:
                            logger.warning(
                                "  REMOVED: %s (group %d, score %.2f) - keeping group %d",
                                identity, group_id, score, largest_group
                            )
                            # Remove gallery identity but keep track in results
                            updated_identities[track_key] = {
                                'gallery_identity': None,
                                'gallery_score': None,
                                'gallery_matches': updated_identities[track_key]['gallery_matches'],
                                'removed_by_social_group': True,
                                'original_identity': identity,
                                'original_score': score,
                            }
    
    logger.info("")
    logger.info("Social group validation complete:")
    logger.info("  Total conflicts found: %d", len(conflicts))
    if action == "remove":
        removed_count = sum(
            1 for v in updated_identities.values() 
            if v.get('removed_by_social_group')
        )
        logger.info("  Identities removed: %d", removed_count)
    logger.info("")
    
    return updated_identities, conflicts


# ==============================================================================
# Stage 2: Cross-Camera Stitching (Room-level)
# ==============================================================================


def stitch_cross_camera(
    track_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    reid_extractor: ReIDFeatureExtractor,
    sim_threshold: float,
    time_gap: float,
    size_ratio: float,
    logger: logging.Logger,
    track_identities: Optional[Dict[Tuple[str, str, int], Dict[str, Any]]] = None,
) -> Tuple[Dict[Tuple[str, str, int], List[Dict[str, Any]]], int]:
    """
    Stage 2: Merge tracks across cameras within the same room using ReID + room constraints.
    
    Returns:
        (stitched_groups, num_merges)
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: Cross-Camera Stitching (Room-level)")
    logger.info("=" * 80)
    logger.info("  Similarity threshold: %.2f", sim_threshold)
    logger.info("  Time gap threshold: %.1f seconds", time_gap)
    logger.info("  Size ratio threshold: %.2f", size_ratio)
    
    # Group by room
    room_groups = defaultdict(list)
    for key, dets in track_groups.items():
        video_path, camera_id, track_id = key
        room = CAMERA_TO_ROOM.get(camera_id)
        if room:
            room_groups[room].append((key, dets))
    
    stitched_groups = {}
    total_merges = 0
    
    # Process each room
    for room, tracks in tqdm(room_groups.items(), desc="Stage 2: Cross-camera stitching"):
        logger.info("  Processing room: %s", room)
        
        if len(tracks) < 2:
            for key, dets in tracks:
                stitched_groups[key] = dets
            continue
        
        # Compute stats for each track
        track_stats = {}
        for key, dets in tracks:
            track_stats[key] = compute_track_stats(dets)
        
        # Extract ReID features for each track
        track_features = {}
        for key, dets in tracks:
            video_path = key[0]
            feat = reid_extractor.extract_from_detections(dets, video_path)
            track_features[key] = feat
        
        # Union-Find for merging
        uf = UnionFind([key for key, _ in tracks])
        
        # Compare all pairs (only across different cameras)
        keys = [key for key, _ in tracks]
        n = len(keys)
        
        for i in range(n):
            key1 = keys[i]
            cam1 = key1[1]
            stats1 = track_stats[key1]
            feat1 = track_features[key1]
            
            for j in range(i + 1, n):
                key2 = keys[j]
                cam2 = key2[1]
                stats2 = track_stats[key2]
                feat2 = track_features[key2]
                
                # Only merge across different cameras
                if cam1 == cam2:
                    continue
                
                # Check temporal proximity (stricter for cross-camera)
                start1, end1 = stats1["start_time"], stats1["end_time"]
                start2, end2 = stats2["start_time"], stats2["end_time"]
                
                if start1 is None or end1 is None or start2 is None or end2 is None:
                    continue
                
                latest_start = max(start1, start2)
                earliest_end = min(end1, end2)
                overlap = (earliest_end - latest_start).total_seconds()
                
                gap = None
                if overlap < 0:
                    if end1 < start2:
                        gap = (start2 - end1).total_seconds()
                    else:
                        gap = (start1 - end2).total_seconds()
                
                temporal_match = (overlap >= 0) or (gap is not None and gap <= time_gap)
                
                if not temporal_match:
                    continue
                
                # Check size similarity (stricter)
                area1, area2 = stats1["median_area"], stats2["median_area"]
                if area1 == 0 or area2 == 0:
                    continue
                
                ratio = min(area1, area2) / max(area1, area2)
                if ratio < size_ratio:
                    continue
                
                # Check ReID similarity (stricter)
                sim = torch.cosine_similarity(feat1, feat2).item()
                
                # Identity-aware matching: if both tracks have gallery identities
                if track_identities:
                    identity1 = track_identities.get(key1, {}).get('gallery_identity')
                    identity2 = track_identities.get(key2, {}).get('gallery_identity')
                    
                    if identity1 and identity2:
                        if identity1 != identity2:
                            # Different known elephants - never merge
                            logger.debug(
                                "    Blocked merge cam%s:%d <-> cam%s:%d (different identities: %s vs %s)",
                                cam1, key1[2], cam2, key2[2], identity1, identity2,
                            )
                            continue
                        else:
                            # Same known elephant - relax similarity threshold
                            identity_boost = 0.1
                            effective_threshold = max(0.5, sim_threshold - identity_boost)
                            if sim < effective_threshold:
                                continue
                            logger.debug(
                                "    Identity-boosted merge: %s (sim=%.3f, threshold=%.3f)",
                                identity1, sim, effective_threshold,
                            )
                    else:
                        # At least one unmatched track - use standard threshold
                        if sim < sim_threshold:
                            continue
                else:
                    # No gallery matching - use standard threshold
                    if sim < sim_threshold:
                        continue
                
                # Merge!
                uf.union(key1, key2)
                total_merges += 1
                logger.debug(
                    "    Merged cam%s:%d <-> cam%s:%d (sim=%.3f, gap=%.1fs, ratio=%.2f)",
                    cam1, key1[2], cam2, key2[2], sim, gap or 0.0, ratio,
                )
        
        # Build merged groups
        clusters = uf.get_clusters()
        for root, members in clusters.items():
            all_dets = []
            for key in members:
                dets = next(d for k, d in tracks if k == key)
                all_dets.extend(dets)
            all_dets.sort(key=lambda d: d.get("frame_idx", 0))
            stitched_groups[root] = all_dets
        
        logger.info(
            "    Tracks: %d -> %d (merged %d)",
            len(tracks), len(clusters), len(tracks) - len(clusters),
        )
    
    logger.info("Stage 2 complete: %d merges total", total_merges)
    logger.info("  Tracks before: %d", len(track_groups))
    logger.info("  Tracks after: %d", len(stitched_groups))
    logger.info("")
    
    return stitched_groups, total_merges


# ==============================================================================
# Output Generation
# ==============================================================================


def assign_global_track_ids(
    stitched_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]]
) -> Dict[Tuple[str, str, int], int]:
    """Assign sequential global track IDs to stitched groups."""
    mapping = {}
    next_id = 1
    
    # Sort by room and then by key for determinism
    sorted_keys = sorted(stitched_groups.keys(), key=lambda k: (CAMERA_TO_ROOM.get(k[1], ""), k))
    
    for key in sorted_keys:
        mapping[key] = next_id
        next_id += 1
    
    return mapping


def compact_track_ids(
    stitched_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    global_id_mapping: Dict[Tuple[str, str, int], int],
) -> Dict[Tuple[str, str, int], int]:
    """
    Compact track IDs to be continuous from 1 to N.
    
    This ensures IDs are sequential with no gaps, making it easy to count
    total unique individuals (max ID = total count).
    
    Args:
        stitched_groups: Track groups after stitching
        global_id_mapping: Initial ID mapping (may have gaps)
    
    Returns:
        Compacted mapping with continuous IDs 1, 2, 3, ..., N
    """
    # Get all unique IDs and sort them
    unique_ids = sorted(set(global_id_mapping.values()))
    
    # Create mapping from old ID to new compact ID
    compact_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
    
    # Apply compaction to global_id_mapping
    compacted_global_mapping = {
        key: compact_mapping[old_id] 
        for key, old_id in global_id_mapping.items()
    }
    
    return compacted_global_mapping



def write_stitched_jsonl(
    file_records: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]]]],
    stitched_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    global_id_mapping: Dict[Tuple[str, str, int], int],
    output_path: Path,
    logger: logging.Logger,
    track_identities: Optional[Dict[Tuple[str, str, int], Dict[str, Any]]] = None,
) -> None:
    """Write stitched JSONL with original_track_id, stitched_track_id, and gallery fields."""
    
    # Build lookup: (video_path, camera_id, frame_idx, original_track_id) -> (stitched_track_id, gallery_info)
    lookup = {}
    for key, dets in stitched_groups.items():
        stitched_id = global_id_mapping[key]
        
        # Get gallery info for this stitched track
        gallery_info = {}
        if track_identities and key in track_identities:
            gallery_data = track_identities[key]
            gallery_info = {
                'gallery_identity': gallery_data.get('gallery_identity'),
                'gallery_score': gallery_data.get('gallery_score'),
                'gallery_top_matches': gallery_data.get('gallery_matches', []),
            }
        
        for det in dets:
            lookup_key = (
                det["video_path"],
                det["camera_id"],
                det["frame_idx"],
                det.get("canonical_track_id", det.get("track_id")),  # original track_id
            )
            lookup[lookup_key] = (stitched_id, gallery_info)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for video_path, camera_id, metadata, records in file_records:
            # Write metadata first
            f.write(json.dumps({"meta": metadata}) + "\n")
            
            # Write each frame's results
            for record in records:
                if "tracks" not in record or not record["tracks"]:
                    # Write record as-is if no tracks
                    f.write(json.dumps({"results": record}) + "\n")
                    continue
                
                frame_idx = record.get("frame_idx", 0)
                
                # Update tracks with stitched_track_id and gallery fields
                updated_tracks = []
                for track in record["tracks"]:
                    original_track_id = track.get("canonical_track_id", -1)
                    
                    lookup_key = (video_path, camera_id, frame_idx, original_track_id)
                    result = lookup.get(lookup_key, (original_track_id, {}))
                    stitched_id, gallery_info = result
                    
                    updated_track = {
                        **track,
                        "original_track_id": original_track_id,
                        "stitched_track_id": stitched_id,
                    }
                    
                    # Add gallery fields if available
                    if gallery_info.get('gallery_identity'):
                        updated_track['gallery_identity'] = gallery_info['gallery_identity']
                        updated_track['gallery_score'] = gallery_info['gallery_score']
                        updated_track['gallery_top_matches'] = gallery_info['gallery_top_matches']
                    
                    updated_tracks.append(updated_track)
                
                updated_record = {
                    **record,
                    "tracks": updated_tracks,
                }
                f.write(json.dumps({"results": updated_record}) + "\n")
    
    logger.info("Wrote stitched JSONL to: %s", output_path)


def write_summary_stats(
    output_dir: Path,
    stage1_merges: int,
    stage2_merges: int,
    original_tracks: int,
    stage1_tracks: int,
    final_tracks: int,
    original_track_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    final_track_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    global_id_mapping: Dict[Tuple[str, str, int], int],
    logger: logging.Logger,
    track_identities: Optional[Dict[Tuple[str, str, int], Dict[str, Any]]] = None,
    social_group_conflicts: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Write summary statistics to a text file."""
    summary_path = output_dir / "stitching_summary.txt"
    
    # Compute per-camera statistics
    camera_original = defaultdict(int)
    camera_final = defaultdict(int)
    
    for key in original_track_groups.keys():
        camera_id = key[1]
        camera_original[camera_id] += 1
    
    for key in final_track_groups.keys():
        camera_id = key[1]
        camera_final[camera_id] += 1
    
    # Get max track ID (which equals total unique individuals)
    max_track_id = max(global_id_mapping.values()) if global_id_mapping else 0
    
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("OFFLINE TRACK STITCHING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original tracks: {original_tracks}\n")
        f.write(f"After Stage 1 (within-video): {stage1_tracks}\n")
        f.write(f"After Stage 2 (cross-camera): {final_tracks}\n\n")
        
        f.write(f"Stage 1 merges: {stage1_merges}\n")
        f.write(f"Stage 2 merges: {stage2_merges}\n")
        f.write(f"Total merges: {stage1_merges + stage2_merges}\n\n")
        
        reduction = original_tracks - final_tracks
        pct = (reduction / original_tracks * 100) if original_tracks > 0 else 0.0
        f.write(f"Track reduction: {reduction} ({pct:.1f}%)\n\n")
        
        f.write(f"UNIQUE INDIVIDUALS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total unique individuals identified: {max_track_id}\n")
        f.write(f"  (Track IDs range from 1 to {max_track_id}, continuous)\n\n")
        
        # Gallery matching statistics
        if track_identities:
            matched_count = sum(1 for v in track_identities.values() if v.get('gallery_identity'))
            total_count = len(track_identities)
            
            # Count unique gallery identities
            unique_identities = set(
                v.get('gallery_identity') 
                for v in track_identities.values() 
                if v.get('gallery_identity')
            )
            
            # Per-identity track counts
            identity_counts = defaultdict(int)
            for v in track_identities.values():
                identity = v.get('gallery_identity')
                if identity:
                    identity_counts[identity] += 1
            
            f.write(f"GALLERY MATCHING RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total tracks after Stage 1: {total_count}\n")
            f.write(f"Matched to gallery: {matched_count} ({100*matched_count/max(total_count,1):.1f}%)\n")
            f.write(f"Unmatched: {total_count - matched_count}\n")
            f.write(f"Unique gallery identities found: {len(unique_identities)}\n\n")
            
            if identity_counts:
                f.write("Tracks per identity:\n")
                for identity in sorted(identity_counts.keys()):
                    count = identity_counts[identity]
                    f.write(f"  {identity}: {count} track(s)\n")
                f.write("\n")
        
        # Social group conflicts
        if social_group_conflicts:
            f.write("SOCIAL GROUP CONFLICTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total conflicts detected: {len(social_group_conflicts)}\n")
            f.write("(Elephants from different social groups appeared together)\n\n")
            
            for idx, conflict in enumerate(social_group_conflicts, 1):
                camera_id = conflict['camera_id']
                timestamp = conflict['timestamp']
                groups = conflict['groups']
                
                f.write(f"Conflict #{idx}:\n")
                f.write(f"  Camera: {camera_id}\n")
                f.write(f"  Timestamp: {timestamp}\n")
                f.write(f"  Groups present: {len(groups)}\n")
                
                for group_id, members in groups.items():
                    member_strs = [f"{identity} (score: {score:.2f})" for _, identity, score in members]
                    f.write(f"    Group {group_id}: {', '.join(member_strs)}\n")
                f.write("\n")
        
        f.write("PER-CAMERA STATISTICS\n")
        f.write("-" * 80 + "\n")
        
        for camera_id in sorted(set(camera_original.keys()) | set(camera_final.keys())):
            orig = camera_original.get(camera_id, 0)
            final = camera_final.get(camera_id, 0)
            cam_reduction = orig - final
            cam_pct = (cam_reduction / orig * 100) if orig > 0 else 0.0
            
            f.write(f"Camera {camera_id}:\n")
            f.write(f"  Original tracks: {orig}\n")
            f.write(f"  Final tracks: {final}\n")
            f.write(f"  Reduction: {cam_reduction} ({cam_pct:.1f}%)\n\n")
    
    logger.info("Wrote summary statistics to: %s", summary_path)


# ==============================================================================
# Visualization
# ==============================================================================


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate consistent BGR color for a track ID."""
    # Use hash to ensure seed is within valid range [0, 2**32 - 1]
    seed = abs(hash(track_id)) % (2**32)
    np.random.seed(seed)
    return tuple(np.random.randint(0, 255, 3).tolist())


def visualize_stitching(
    file_records: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]]]],
    stitched_groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    global_id_mapping: Dict[Tuple[str, str, int], int],
    output_dir: Path,
    max_frames: int,
    logger: logging.Logger,
) -> None:
    """Generate side-by-side comparison visualizations."""
    
    logger.info("Generating visualizations...")
    
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Group records by video
    video_records = {}
    for video_path, camera_id, metadata, records in file_records:
        video_records[video_path] = (camera_id, records)
    
    # Build lookup for stitched IDs
    lookup = {}
    for key, dets in stitched_groups.items():
        stitched_id = global_id_mapping[key]
        for det in dets:
            lookup_key = (
                det["video_path"],
                det["camera_id"],
                det["frame_idx"],
                det.get("canonical_track_id", det.get("track_id")),
            )
            lookup[lookup_key] = stitched_id
    
    # Visualize each video
    for video_path, (camera_id, records) in tqdm(list(video_records.items())[:5], desc="Generating visualizations"):
        video_name = Path(video_path).stem
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", video_path)
            continue
        
        # Sample frames
        sampled_records = records[:: max(1, len(records) // max_frames)][:max_frames]
        
        for record in sampled_records:
            frame_idx = record.get("frame_idx", 0)
            
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Create two copies for before/after
            frame_before = frame.copy()
            frame_after = frame.copy()
            
            # Draw tracks
            tracks = record.get("tracks", [])
            for track in tracks:
                bbox = track.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                original_id = track.get("canonical_track_id", track.get("track_id", -1))
                
                # Before: draw with original ID
                cv2.rectangle(frame_before, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_before,
                    f"ID:{original_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                
                # After: draw with stitched ID
                lookup_key = (video_path, camera_id, frame_idx, original_id)
                stitched_id = lookup.get(lookup_key, original_id)
                
                # Color based on stitched ID
                color = get_track_color(stitched_id)
                
                cv2.rectangle(frame_after, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame_after,
                    f"ID:{stitched_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            
            # Add titles
            cv2.putText(
                frame_before,
                "Before Stitching",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_after,
                "After Stitching",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            
            # Concatenate side-by-side and save
            vis_frame = np.hstack([frame_before, frame_after])
            out_path = vis_dir / f"{video_name}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), vis_frame)
        
        cap.release()
    
    logger.info("Visualization complete")
def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)
    
    logger.info("=" * 80)
    logger.info("OFFLINE TRACK STITCHING PIPELINE")
    logger.info("=" * 80)
    
    # Parse input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)
    
    logger.info("Input directory: %s", input_dir)
    
    # Auto-discover JSONL files
    jsonl_files = discover_jsonl_files(input_dir, logger)
    if not jsonl_files:
        logger.error("No JSONL files found in %s", input_dir)
        sys.exit(1)
    
    logger.info("")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / "stitched_tracks"
    
    logger.info("Output directory: %s", output_dir)
    logger.info("")
    
    # Debug mode notification
    if args.max_frames:
        logger.info("DEBUG MODE: Processing only first %d frames per file", args.max_frames)
        logger.info("")
    
    # Load all input JSONL files
    file_records = []
    total_records = 0
    
    for jsonl_path in jsonl_files:
        logger.info("Loading: %s", jsonl_path.name)
        metadata, records = load_jsonl(jsonl_path, max_frames=args.max_frames)
        
        # Extract camera ID from filename (e.g., "ZAG-ELP-CAM-016")
        filename = jsonl_path.stem
        camera_parts = filename.split("-")
        if len(camera_parts) >= 4:
            camera_id = camera_parts[3]  # "016", "017", etc.
        else:
            camera_id = "unknown"
        
        video_path = metadata.get("video", str(jsonl_path))
        
        file_records.append((video_path, camera_id, metadata, records))
        total_records += len(records)
        logger.info("  Loaded %d records (camera: %s)", len(records), camera_id)
    
    logger.info("Total records: %d", total_records)
    logger.info("")
    
    # Group detections by track
    track_groups = group_detections_by_track(file_records, logger)
    original_track_count = len(track_groups)
    logger.info("Original unique tracks: %d", original_track_count)
    logger.info("")
    
    # Initialize ReID feature extractor
    device = torch.device(args.device)
    reid_extractor = ReIDFeatureExtractor(
        args.reid_config,
        args.reid_checkpoint,
        device,
        logger,
    )
    
    # Stage 1: Within-video stitching
    stage1_groups, stage1_merges = stitch_within_video(
        track_groups,
        reid_extractor,
        args.stage1_sim_threshold,
        args.stage1_time_gap,
        args.stage1_size_ratio,
        logger,
    )
    stage1_track_count = len(stage1_groups)
    
    # Gallery matching: Identity anchoring (optional, after Stage 1)
    track_identities = None
    social_group_conflicts = []
    if args.gallery:
        gallery_device = torch.device(args.gallery_device)
        track_identities = match_tracks_to_gallery(
            stage1_groups,
            reid_extractor,
            args.gallery,
            gallery_device,
            args.gallery_threshold,
            args.gallery_top_k,
            logger,
        )
        
        # Social group validation (optional, after gallery matching)
        if args.use_social_groups and track_identities:
            track_identities, social_group_conflicts = validate_social_groups(
                stage1_groups,
                track_identities,
                SOCIAL_GROUPS,
                args.social_group_action,
                logger,
            )
    
    # Stage 2: Cross-camera stitching (identity-aware if gallery matching was performed)
    stage2_groups, stage2_merges = stitch_cross_camera(
        stage1_groups,
        reid_extractor,
        args.stage2_sim_threshold,
        args.stage2_time_gap,
        args.stage2_size_ratio,
        logger,
        track_identities=track_identities,
    )
    final_track_count = len(stage2_groups)
    
    # Assign global IDs and compact them to be continuous
    global_id_mapping = assign_global_track_ids(stage2_groups)
    global_id_mapping = compact_track_ids(stage2_groups, global_id_mapping)
    
    # Log ID statistics
    max_id = max(global_id_mapping.values()) if global_id_mapping else 0
    logger.info("Compacted track IDs: 1 to %d (total: %d unique individuals)", max_id, max_id)
    logger.info("")
    
    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_jsonl = output_dir / "stitched_tracks.jsonl"
    write_stitched_jsonl(
        file_records,
        stage2_groups,
        global_id_mapping,
        output_jsonl,
        logger,
        track_identities=track_identities,
    )
    
    write_summary_stats(
        output_dir,
        stage1_merges,
        stage2_merges,
        original_track_count,
        stage1_track_count,
        final_track_count,
        track_groups,
        stage2_groups,
        global_id_mapping,
        logger,
        track_identities=track_identities,
        social_group_conflicts=social_group_conflicts,
    )
    
    # Visualizations
    if args.visualize:
        visualize_stitching(
            file_records,
            stage2_groups,
            global_id_mapping,
            output_dir,
            args.vis_max_frames,
            logger,
        )
    
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info("Final results:")
    logger.info("  Original tracks: %d", original_track_count)
    logger.info("  After Stage 1: %d (-%d)", stage1_track_count, original_track_count - stage1_track_count)
    logger.info("  After Stage 2: %d (-%d)", final_track_count, stage1_track_count - final_track_count)
    logger.info("  Total reduction: %d tracks (%.1f%%)", 
                original_track_count - final_track_count,
                (original_track_count - final_track_count) / original_track_count * 100 if original_track_count > 0 else 0)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
