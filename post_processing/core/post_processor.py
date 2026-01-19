"""
Post-Processing Module

Handles data cleaning and behavior smoothing:
- Fix identity switches and spurious detections
- Smooth behavior labels using temporal windows
- Remove invalid labels using cross-camera information
- Apply various filtering and cleaning strategies
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ==============================================================================
# Configuration
# ==============================================================================

ROOM_PAIRS = {
    "room1": ["016", "019"],
    "room2": ["017", "018"],
}

CAMERA_TO_ROOM = {
    "016": "room1",
    "019": "room1",
    "017": "room2",
    "018": "room2",
    "16": "room1",
    "19": "room1",
    "17": "room2",
    "18": "room2",
}


# ==============================================================================
# Invalid Zone Handler
# ==============================================================================

class InvalidZoneHandler:
    """Handle invalid zone detection and filtering."""

    def __init__(self, invalid_zones_dir: Optional[Path], logger: Optional[logging.Logger] = None):
        self.invalid_zones_dir = invalid_zones_dir
        self.logger = logger or logging.getLogger(__name__)
        self.zone_cache: Dict[str, List[np.ndarray]] = {}

    def load_zones(self, camera_id: str) -> Optional[List[np.ndarray]]:
        """Load invalid zone polygons for a camera."""
        if camera_id in self.zone_cache:
            return self.zone_cache[camera_id]

        if not self.invalid_zones_dir:
            return None

        json_path = self.invalid_zones_dir / f"cam{camera_id}_invalid_zones.json"
        if not json_path.exists():
            self.zone_cache[camera_id] = None
            return None

        try:
            import json
            with open(json_path, "r") as f:
                data = json.load(f)

            polygons = []
            for zone in data.get("zones", []):
                points = zone.get("points", [])
                if points:
                    poly_array = np.array(points, dtype=np.int32)
                    polygons.append(poly_array)

            self.zone_cache[camera_id] = polygons
            self.logger.info("Loaded %d invalid zones for camera %s", len(polygons), camera_id)
            return polygons

        except Exception as e:
            self.logger.warning("Failed to load invalid zones for camera %s: %s", camera_id, e)
            self.zone_cache[camera_id] = None
            return None

    def is_in_invalid_zone(self, bbox: List[float], camera_id: str) -> bool:
        """Check if a bounding box overlaps with any invalid zone."""
        import cv2

        polygons = self.load_zones(camera_id)
        if not polygons or not bbox or len(bbox) != 4:
            return False

        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        center_point = (center_x, center_y)

        # Check if center point is inside any polygon
        for polygon in polygons:
            result = cv2.pointPolygonTest(polygon, center_point, False)
            if result >= 0:  # Inside or on the boundary
                return True

        return False


# ==============================================================================
# Identity Switch Fixer
# ==============================================================================

class IdentitySwitchFixer:
    """Fix identity switches and remove spurious detections."""

    def __init__(
        self,
        time_window_minutes: int = 5,
        min_spurious_duration_seconds: int = 30,
        logger: Optional[logging.Logger] = None,
    ):
        self.time_window_minutes = time_window_minutes
        self.min_spurious_duration_seconds = min_spurious_duration_seconds
        self.logger = logger or logging.getLogger(__name__)

    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix identity switches and remove spurious detections."""
        df = df.copy()
        fixes_count = 0

        # Process each camera separately
        for camera_id in df['camera_id'].unique():
            camera_mask = df['camera_id'] == camera_id
            camera_df = df[camera_mask].copy()

            # Find single-detection frames
            frame_counts = camera_df.groupby('timestamp').size()
            single_frames = frame_counts[frame_counts == 1].index

            for timestamp in single_frames:
                frame_mask = camera_df['timestamp'] == timestamp
                if frame_mask.sum() != 1:
                    continue

                idx = camera_df[frame_mask].index[0]
                current_id = camera_df.loc[idx, 'stitched_id']

                # Look for dominant ID in time window
                time_window = timedelta(minutes=self.time_window_minutes)
                window_mask = (
                    (camera_df['timestamp'] >= timestamp - time_window) &
                    (camera_df['timestamp'] <= timestamp + time_window) &
                    (camera_df['stitched_id'] != current_id)
                )

                if window_mask.sum() > 0:
                    nearby_ids = camera_df.loc[window_mask, 'stitched_id'].value_counts()
                    if len(nearby_ids) > 0:
                        dominant_id = nearby_ids.index[0]
                        df.loc[idx, 'stitched_id'] = dominant_id
                        fixes_count += 1

            # Remove spurious detections
            for stitched_id in camera_df['stitched_id'].unique():
                track_mask = camera_df['stitched_id'] == stitched_id
                track_df = camera_df[track_mask]

                if len(track_df) == 0:
                    continue

                # Calculate duration
                duration = (track_df['timestamp'].max() - track_df['timestamp'].min()).total_seconds()

                # Check if it's a spurious detection
                if duration < self.min_spurious_duration_seconds:
                    # Check if there's a dominant track in the same time period
                    time_mask = (
                        (camera_df['timestamp'] >= track_df['timestamp'].min()) &
                        (camera_df['timestamp'] <= track_df['timestamp'].max()) &
                        (camera_df['stitched_id'] != stitched_id)
                    )

                    if time_mask.sum() > len(track_df) * 2:  # Dominant track is 2x longer
                        # Remove spurious track
                        df = df[~((df['camera_id'] == camera_id) & (df['stitched_id'] == stitched_id))]
                        fixes_count += 1

        self.logger.info("Fixed %d identity switches/spurious detections", fixes_count)
        return df


# ==============================================================================
# Behavior Smoother
# ==============================================================================

class BehaviorSmoother:
    """Smooth behavior labels using temporal windows."""

    def __init__(
        self,
        time_window_seconds: float = 1.5,
        invalid_window_seconds: float = 10.0,
        max_consecutive_frames: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        self.time_window_seconds = time_window_seconds
        self.invalid_window_seconds = invalid_window_seconds
        self.max_consecutive_frames = max_consecutive_frames
        self.logger = logger or logging.getLogger(__name__)

    def smooth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smooth behavior labels within time windows."""
        df = df.copy()
        df['behavior_original'] = df['behavior']
        smoothed_count = 0

        # Process each camera-track combination
        for (camera_id, stitched_id), group in df.groupby(['camera_id', 'stitched_id']):
            if len(group) < 2:
                continue

            group = group.sort_values('timestamp')
            indices = group.index.tolist()

            # Step 1: Smooth invalid labels
            for i, idx in enumerate(indices):
                if df.loc[idx, 'behavior'] != 'invalid':
                    continue

                # Look for valid behaviors in larger window
                current_time = df.loc[idx, 'timestamp']
                window = timedelta(seconds=self.invalid_window_seconds)

                nearby_mask = (
                    (group['timestamp'] >= current_time - window) &
                    (group['timestamp'] <= current_time + window) &
                    (group['behavior'] != 'invalid')
                )

                nearby_behaviors = group.loc[nearby_mask, 'behavior']
                if len(nearby_behaviors) > 0:
                    most_common = nearby_behaviors.mode()[0]
                    df.loc[idx, 'behavior'] = most_common
                    smoothed_count += 1

            # Step 2: Smooth brief behavior spikes
            group = df.loc[indices].copy()
            i = 0
            while i < len(indices):
                current_behavior = df.loc[indices[i], 'behavior']

                # Count consecutive frames with same behavior
                j = i
                while j < len(indices) and df.loc[indices[j], 'behavior'] == current_behavior:
                    j += 1

                consecutive_count = j - i

                # If it's a brief spike, smooth it
                if consecutive_count <= self.max_consecutive_frames:
                    # Get surrounding behavior
                    prev_behavior = df.loc[indices[i - 1], 'behavior'] if i > 0 else None
                    next_behavior = df.loc[indices[j], 'behavior'] if j < len(indices) else None

                    if prev_behavior == next_behavior and prev_behavior is not None:
                        for k in range(i, j):
                            df.loc[indices[k], 'behavior'] = prev_behavior
                            smoothed_count += 1

                i = j

        self.logger.info("Smoothed %d behavior labels", smoothed_count)
        return df


# ==============================================================================
# Invalid Label Remover
# ==============================================================================

class InvalidLabelRemover:
    """Remove remaining invalid labels using track history."""

    def __init__(
        self,
        lookback_seconds: float = 60.0,
        default_behavior: str = "standing",
        logger: Optional[logging.Logger] = None,
    ):
        self.lookback_seconds = lookback_seconds
        self.default_behavior = default_behavior
        self.logger = logger or logging.getLogger(__name__)

    def remove(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid labels using lookback strategy."""
        df = df.copy()
        removed_count = 0

        # Process each camera-track combination
        for (camera_id, stitched_id), group in df.groupby(['camera_id', 'stitched_id']):
            group = group.sort_values('timestamp')
            indices = group.index.tolist()

            for idx in indices:
                if df.loc[idx, 'behavior'] != 'invalid':
                    continue

                current_time = df.loc[idx, 'timestamp']
                lookback = timedelta(seconds=self.lookback_seconds)

                # Look back for valid behavior
                past_mask = (
                    (group['timestamp'] < current_time) &
                    (group['timestamp'] >= current_time - lookback) &
                    (group['behavior'] != 'invalid')
                )

                past_behaviors = group.loc[past_mask, 'behavior']
                if len(past_behaviors) > 0:
                    # Use most recent valid behavior
                    df.loc[idx, 'behavior'] = past_behaviors.iloc[-1]
                else:
                    # Use default
                    df.loc[idx, 'behavior'] = self.default_behavior

                removed_count += 1

        self.logger.info("Removed %d invalid labels", removed_count)
        return df


# ==============================================================================
# Cross-Camera Invalid Fixer
# ==============================================================================

class CrossCameraInvalidFixer:
    """Fix invalid behaviors by checking paired cameras."""

    def __init__(
        self,
        time_tolerance_seconds: float = 3.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.time_tolerance_seconds = time_tolerance_seconds
        self.logger = logger or logging.getLogger(__name__)

    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix invalid behaviors using cross-camera information."""
        df = df.copy()
        df['room'] = df['camera_id'].map(CAMERA_TO_ROOM)
        df['timestamp_sec'] = df['timestamp'].dt.floor('1S')

        fixed_count = 0

        # Find invalid behaviors
        invalid_mask = df['behavior'] == 'invalid'
        invalid_rows = df[invalid_mask].copy()

        for idx, row in invalid_rows.iterrows():
            camera_id = row['camera_id']
            room = row['room']
            stitched_id = row['stitched_id']
            timestamp = row['timestamp']

            # Find other cameras in the same room
            room_cameras = ROOM_PAIRS.get(room, [])
            other_cameras = [cam for cam in room_cameras if cam != camera_id]

            if not other_cameras:
                continue

            # Look for same elephant in other cameras
            tolerance = timedelta(seconds=self.time_tolerance_seconds)
            cross_mask = (
                (df['camera_id'].isin(other_cameras)) &
                (df['stitched_id'] == stitched_id) &
                (df['timestamp'] >= timestamp - tolerance) &
                (df['timestamp'] <= timestamp + tolerance) &
                (df['behavior'] != 'invalid')
            )

            cross_matches = df[cross_mask]
            if len(cross_matches) > 0:
                # Use most common valid behavior from other cameras
                valid_behavior = cross_matches['behavior'].mode()[0]
                df.loc[idx, 'behavior'] = valid_behavior
                fixed_count += 1

        df = df.drop(columns=['room', 'timestamp_sec'])
        self.logger.info("Fixed %d invalid behaviors using cross-camera info", fixed_count)
        return df


# ==============================================================================
# Post-Processor (Main Class)
# ==============================================================================

class PostProcessor:
    """
    Post-processing pipeline for cleaning and smoothing tracking data.
    
    This is the main class for Stage 3 of the pipeline.
    """

    def __init__(
        self,
        invalid_zones_dir: Optional[Path] = None,
        time_window_minutes: int = 5,
        min_spurious_duration_seconds: int = 30,
        time_window_seconds: float = 1.5,
        invalid_window_seconds: float = 10.0,
        max_consecutive_frames: int = 3,
        lookback_seconds: float = 60.0,
        default_behavior: str = "standing",
        time_tolerance_seconds: float = 3.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

        # Initialize handlers
        self.invalid_zone_handler = InvalidZoneHandler(invalid_zones_dir, self.logger)
        self.identity_fixer = IdentitySwitchFixer(
            time_window_minutes,
            min_spurious_duration_seconds,
            self.logger,
        )
        self.behavior_smoother = BehaviorSmoother(
            time_window_seconds,
            invalid_window_seconds,
            max_consecutive_frames,
            self.logger,
        )
        self.invalid_remover = InvalidLabelRemover(
            lookback_seconds,
            default_behavior,
            self.logger,
        )
        self.cross_camera_fixer = CrossCameraInvalidFixer(
            time_tolerance_seconds,
            self.logger,
        )

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Run the complete post-processing pipeline.
        
        Returns:
            (processed_df, stats)
        """
        self.logger.info("=" * 80)
        self.logger.info("Post-Processing Pipeline")
        self.logger.info("=" * 80)
        self.logger.info("Input rows: %d", len(df))

        original_count = len(df)
        stats = {}

        # Step 1: Fix identity switches
        df = self.identity_fixer.fix(df)
        stats['identity_fixes'] = original_count - len(df)

        # Step 2: Smooth behavior labels
        df = self.behavior_smoother.smooth(df)
        stats['behavior_smoothing'] = (df['behavior'] != df['behavior_original']).sum()

        # Step 3: Fix invalid labels using cross-camera info
        df = self.cross_camera_fixer.fix(df)

        # Step 4: Remove remaining invalid labels
        invalid_before = (df['behavior'] == 'invalid').sum()
        df = self.invalid_remover.remove(df)
        invalid_after = (df['behavior'] == 'invalid').sum()
        stats['invalid_removed'] = invalid_before - invalid_after

        # Step 5: Mark detections in invalid zones
        if self.invalid_zone_handler.invalid_zones_dir:
            in_zone_count = 0
            for idx, row in df.iterrows():
                if self.invalid_zone_handler.is_in_invalid_zone(row['bbox'], row['camera_id']):
                    df.loc[idx, 'behavior'] = 'invalid'
                    in_zone_count += 1
            stats['marked_invalid_zone'] = in_zone_count

        self.logger.info("Post-processing complete:")
        self.logger.info("  Identity fixes: %d", stats.get('identity_fixes', 0))
        self.logger.info("  Behavior smoothing: %d", stats.get('behavior_smoothing', 0))
        self.logger.info("  Invalid removed: %d", stats.get('invalid_removed', 0))
        self.logger.info("  Marked in invalid zone: %d", stats.get('marked_invalid_zone', 0))
        self.logger.info("Final rows: %d", len(df))

        return df, stats
