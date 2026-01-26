import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from pathlib import Path
import numpy as np
import cv2
import logging
logger = logging.getLogger(__name__)
from post_processing.utils import load_embedding, load_semi_gt_ids
from post_processing.core.reid_inference import match_to_gallery
from post_processing.tools.run_reid_feature_extraction import vote_matched_labels
from post_processing.tools.reranking import assign_identities_from_trackid2label_counts
import json

seed = 42
np.random.seed(seed)

## TODO: remove redundant items in Tracklet dataclass
@dataclass
class Tracklet:
    track_id: str
    raw_track_id: str  # original track id before stitching -- linked to track_file name (*.mkv)
    track_filename: str = ""  # video file name of the tracklet
    track_csv_path: Optional[Path] = None  # path to the per-track CSV
    track_video_path: Optional[Path] = None  # optional raw video path for visualization
    
    camera_id: str = ""
    identity_label: str = "invalid" # final identity label after voting
    voted_track_label: str = "invalid" # identity label voted from ReID features (from current tracklet only)
    semi_gt_labels: List[str] = field(default_factory=list)  # list of semi GT IDs matched to this tracklet
    stitched_id: Optional[int] = None
    invalid_flag: bool = False    ## whether the tracklet is invalid (e.g. in invalid zone, else?)

    ## no need -- we can link it back to track file (*.csv)
    # frames: List[int] = field(default_factory=list)  # frame indices within the video
    start_timestamp: Optional[datetime] = None    
    end_timestamp: Optional[datetime] = None
    start_frame_id: Optional[int] = None
    end_frame_id: Optional[int] = None
    # bboxes: List[List[float]] = field(default_factory=list)
    # scores: List[float] = field(default_factory=list)
    feature_path: Optional[Path] = None  # path to the .npz file with ReID features




def init_tracklets_from_online_results(track_csv: str, camera_id: str | None = None) -> List[Tracklet]:
    """
    Load a track CSV (frame_id,timestamp,bbox_top,bbox_left,bbox_bottom,bbox_right,score)
    into Tracklet objects. Uses file stem as raw_track_id/track_id.
    """
    track_id = Path(track_csv).stem
    df = pd.read_csv(track_csv)
    if df.empty:
        logger.warning(f"Track CSV {track_csv} is empty. Skipping.")
        return []
    
    start_timestamp = datetime.fromisoformat(df['timestamp'].iloc[0])
    end_timestamp = datetime.fromisoformat(df['timestamp'].iloc[-1])
    start_frame_id = int(df['frame_id'].iloc[0])
    end_frame_id = int(df['frame_id'].iloc[-1])
    track_filename = Path(track_csv).stem

    ## load identity label from npz if exists
    voted_track_label = "invalid"
    npz_path = Path(track_csv).with_suffix(".npz")
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            voted_track_label = data.get('voted_track_labels', 'invalid')
            if isinstance(voted_track_label, np.ndarray):
                voted_track_label = voted_track_label.item()
        except Exception as e:
            logger.warning(f"Failed to load identity label from {npz_path}: {e}")
    else:
        ### we didn't save npz file for invalid tracks
        voted_track_label = "invalid"
         
        
    t = Tracklet(track_id=track_id, 
                 raw_track_id=track_id, 
                 camera_id=camera_id or "",
                 start_timestamp=start_timestamp,
                 end_timestamp=end_timestamp,
                 feature_path=Path(track_csv).with_suffix(".npz"),
                 start_frame_id=start_frame_id,
                 end_frame_id=end_frame_id,
                 track_filename=track_filename,
                 track_csv_path=Path(track_csv),
                 voted_track_label=voted_track_label
                 )
    
    # print(f"Initialized tracklet {track_id} from {track_csv} with {len(df)} frames.", start_timestamp, end_timestamp)

    return [t]


def track_csv2identity(final_stitched_map: Dict[str, List[Dict[str, object]]]) -> Dict[str, str]:
    
    csv2identity = {}
    
    for idlabel, tracklets in final_stitched_map.items():
        for t in tracklets:
            track_csv = t.get("track_csv_path", "")
            if track_csv:
                csv2identity[track_csv] = t.get("identity_label", "invalid")
    
    return csv2identity



################# TODO: segmentation model for the invalid zone detection if camera view is changed (now the polygon points are fixed) ##################
class InvalidZoneHandler:
    """Handle invalid zone detection and filtering."""

    def __init__(self, invalid_zones_dir: Optional[Path], logger: Optional[logging.Logger] = None,
                 original_height: int = 1520, original_width: int = 2688):
        self.invalid_zones_dir = invalid_zones_dir
        self.logger = logger or logging.getLogger(__name__)
        self.zone_cache: Dict[str, List[np.ndarray]] = {}
        self.original_height = original_height  # Original image height when zones were annotated -- fixed
        self.original_width = original_width    # Original image width when zones were annotated -- fixed

    def load_zones(self, camera_id: str) -> Optional[List[np.ndarray]]:
        """Load invalid zone polygons for a camera."""
        if camera_id in self.zone_cache:
            return self.zone_cache[camera_id]

        if not self.invalid_zones_dir:
            return None

        json_path = self.invalid_zones_dir / f"cam{camera_id}_invalid_zones.json"
        if not json_path.exists():
            self.logger.debug(f"No invalid zones file found at {json_path}")
            self.zone_cache[camera_id] = None
            return None

        try:
            
            with open(json_path, "r") as f:
                data = json.load(f)

            polygons = []
            for zone in data.get("polygons", []):
                points = zone.get("points", [])
                if points:
                    poly_array = np.array(points, dtype=np.int32)
                    print(f"Loaded invalid zone '{zone.get('name', 'unnamed')}' for camera {camera_id}: {len(points)} points")
                    polygons.append(poly_array)

            self.zone_cache[camera_id] = polygons if polygons else None
            print(f"Loaded {len(polygons)} invalid zone(s) for camera {camera_id}")
            return polygons if polygons else None

        except Exception as e:
            self.logger.error(f"Failed to load invalid zones for camera {camera_id}: {e}")
            self.zone_cache[camera_id] = None
            return None

    def is_in_invalid_zone(self, bbox: List[float], camera_id: str, height: int, width: int) -> bool:
        """Check if a bounding box overlaps with any invalid zone.
        
        Args:
            bbox: Bounding box in format [top, left, bottom, right]
            camera_id: Camera identifier
            height: Current image height
            width: Current image width
        
        Returns:
            True if bbox center is inside any invalid zone polygon
        """

        polygons = self.load_zones(camera_id)

        if not polygons or not bbox or len(bbox) != 4:
            return False
        
        # bbox format: [top, left, bottom, right]
        top, left, bottom, right = bbox
        center_x = int((left + right) / 2)
        center_y = int((top + bottom) / 2)
        center_point = (center_x, center_y)
        
        # print("Checking bbox center point", center_point, "for invalid zones in camera", camera_id, "with frame size", height, "x", width)
        scaled_polygons = self.get_scaled_polygons(polygons, camera_id, height, width)
        
        # # ### DEBUG -- plot invalid zones + box for debugging
        # frame = self.plot_invalid_zones(np.zeros((height, width, 3), dtype=np.uint8), scaled_polygons, camera_id, height=height, width=width)
        # # plot box on frame
        # if frame is not None and frame.size != 0:
        #     cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
        #     cv2.circle(frame, center_point, 5, (0, 255, 0), -1)
        #     cv2.imwrite(f"invalid_zone_debug_cam{camera_id}.jpg", frame)
        #     print(f"Saved debug image with invalid zones and bbox to invalid_zone_debug_cam{camera_id}.jpg")
            
        # import sys; sys.exit(0)
        
        # Check if center point is inside any polygon
        for scaled_polygon in scaled_polygons:
            result = cv2.pointPolygonTest(scaled_polygon, center_point, False)
            if result >= 0:  # Inside or on the boundary
                return True
            
        return False

    def get_scaled_polygons(self, polygons: List[np.ndarray], camera_id: str, height: int, width: int) -> Optional[List[np.ndarray]]:
        """Get polygons scaled to match the target image size.
        
        Args:
            camera_id: Camera identifier
            height: Target image height
            width: Target image width
            
        Returns:
            List of scaled polygon arrays, or None if no zones found
        """
        
        if not polygons:
            return None
        
        # Calculate scaling factors
        scale_x = width / self.original_width
        scale_y = height / self.original_height
        
        # print("original size:", self.original_width, self.original_height)
        # print(f"Scaling factors for camera {camera_id}: scale_x={scale_x}, scale_y={scale_y}")
        
        # Only log if scaling is significant (more than 1% difference)
        if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
            self.logger.debug(
                f"Scaling polygons: original ({self.original_width}x{self.original_height}) -> "
                f"current ({width}x{height}), scale_x={scale_x:.3f}, scale_y={scale_y:.3f}"
            )
            
        scaled_polygons = []
        for polygon in polygons:
            # Scale polygon points to match target image size
            scaled_polygon = polygon.copy().astype(np.float32)
            scaled_polygon[:, 0] *= scale_x  # Scale x coordinates
            scaled_polygon[:, 1] *= scale_y  # Scale y coordinates
            scaled_polygon = scaled_polygon.astype(np.int32)
            scaled_polygons.append(scaled_polygon)
        
        return scaled_polygons

    def plot_invalid_zones(self, 
                           frame: np.ndarray, 
                           polygons: List[np.ndarray],
                           camera_id: str, 
                           height: int,
                           width: int,
                           color: tuple = (0, 0, 255), 
                           thickness: int = 2,
                           fill_alpha: float = 0.3) -> np.ndarray:
        """Draw invalid zones on a frame.
        
        Args:
            frame: Input image (H, W, C) in RGB or BGR format
            camera_id: Camera identifier
            color: Polygon color in RGB format (default: red)
            thickness: Line thickness for polygon borders (default: 2)
            fill_alpha: Transparency for filled polygons (0-1, default: 0.3)
            
        Returns:
            Frame with invalid zones drawn
        """
        
        
        if frame is None or frame.size == 0:
            self.logger.warning("Invalid frame provided for plotting")
            return frame
        
        height, width = frame.shape[:2]

        # Create a copy to draw on
        output_frame = frame.copy()
        
        # Create overlay for semi-transparent fill
        overlay = output_frame.copy()
        
        for i, polygon in enumerate(polygons):
            # Fill polygon with transparency
            cv2.fillPoly(overlay, [polygon], color)
            
            # Draw polygon border
            cv2.polylines(output_frame, [polygon], isClosed=True, 
                         color=color, thickness=thickness)
        
        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, fill_alpha, output_frame, 1 - fill_alpha, 
                       0, output_frame)
        
        return output_frame


class TrackletManager:
    """
    Tracklet Manager for loading and stitching tracklets within single camera.

    """
    def __init__(self, 
                 track_dirs: list[Path],
                 camera_id: str, 
                 num_identities: int = 2,
                 start_time: Optional[datetime] = "17:00:00",
                 end_time: Optional[datetime] = "07:00:00",
                 logger =None,
                 invalid_zones_dir: Path = Path('/media/mu/zoo_vision/data/invalid_zones'),
                 height: int = 1520,
                 width: int = 1920,
                 ):
        
        """Initialize TrackletManager.

        Args:
            track_dirs: List of directories containing track CSV files.
            camera_id: Camera identifier.
            num_identities: Expected number of unique identities for stitching.
            start_time: Optional start time to filter tracklets.
            end_time: Optional end time to filter tracklets.
            logger: Optional logger for logging messages.
            invalid_zones_dir: Directory containing invalid zone JSON files.
            height: Image height of current video resolution for invalid zone scaling.
            width: Image width of current video resolution for invalid zone scaling.

        """
        
        self.track_dirs = track_dirs
        self.camera_id = camera_id
        self.num_identities = num_identities
        self.start_time = self._format_time(start_time) if start_time else None
        self.end_time = self._format_time(end_time) if end_time else None
        self.logger = logger
        if not logger:
            import logging
            self.logger = logging.getLogger(__name__)
        self.tracklets: List[Tracklet] = []
        self.height = height
        self.width = width

        self.invalid_zone_handler = InvalidZoneHandler(invalid_zones_dir, 
                                                       self.logger)
        # self.invalid_zone_handler.plot_invalid_zones(frame, self.camera_id)

    def _format_time(self, timestamp) -> datetime:
        """Convert time string to datetime object.
        
        Supports formats:
        - 'HH:MM:SS' (time only, uses date from self.tracklets if available, otherwise uses today)
        - 'yyyymmdd HH:MM:SS' (full datetime)
        """
        if isinstance(timestamp, str):
            # Try full datetime format first
            if len(timestamp) > 8:  # Has date component
                try:
                    return datetime.strptime(timestamp, "%Y%m%d %H:%M:%S")
                except ValueError:
                    pass
            
            # Handle time-only format
            try:
                time_obj = datetime.strptime(timestamp, "%H:%M:%S").time()
                # Use date from first tracklet if available, otherwise use today
                if self.tracklets and self.tracklets[0].start_timestamp:
                    date_part = self.tracklets[0].start_timestamp.date()
                else:
                    date_part = datetime.now().date()
                return datetime.combine(date_part, time_obj)
            except ValueError:
                raise ValueError(f"Invalid time format: {timestamp}. Expected 'HH:MM:SS' or 'yyyymmdd HH:MM:SS'")
        elif isinstance(timestamp, datetime):
            return timestamp
        else:
            raise TypeError(f"Expected str or datetime, got {type(timestamp)}")

    def validate_tracklet_timestamp(self, start_dt: datetime) -> bool:
        """Check if tracklet timestamp is within the time window.
        
        Args:
            start_dt: Start timestamp of the tracklet
            
        Returns:
            True if tracklet should be skipped, False if it should be kept
        """
        # If no time filtering, keep all tracklets
        if not self.start_time and not self.end_time:
            return False
        
        if not start_dt:
            return False
        
        # If only start_time is set
        if self.start_time and not self.end_time:
            return start_dt < self.start_time
        
        # If only end_time is set
        if self.end_time and not self.start_time:
            return start_dt > self.end_time
        
        # Both start_time and end_time are set
        # Check if time range crosses midnight (e.g., 17:00 to 07:00)
        if self.start_time > self.end_time:
            # Crosses midnight: keep if >= start_time OR <= end_time
            in_range = start_dt >= self.start_time or start_dt <= self.end_time
        else:
            # Normal range: keep if >= start_time AND <= end_time
            in_range = self.start_time <= start_dt <= self.end_time
        
        # Return True to skip, False to keep
        return not in_range
    
    def vote_identity_label(self, features, avg_embedding, gallery_features, gallery_labels, known_labels=None) -> str:
        
        # Sample 20% if too many queries
        if len(features) > 10000:
            sample_size = int(len(features) * 0.2)
            sample_indices = np.random.choice(len(features), size=sample_size, replace=False)
            features = features[sample_indices]
        
        matched_labels = match_to_gallery(features, gallery_features, gallery_labels=gallery_labels)[-1]
        avg_matched_labels = match_to_gallery(avg_embedding, gallery_features, gallery_labels=gallery_labels)[-1]
        
        voted_track_labels = vote_matched_labels(matched_labels)

        count_dict = {}
        for label in voted_track_labels:
            count_dict[label] = count_dict.get(label, 0) + 1
        # sort by count
        count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
        # get final voted label if in known_labels, else 'invalid'
        for label in count_dict.keys():
            if known_labels and label in known_labels:
                voted_track_label = label
                break
            else:
                voted_track_label = "invalid"

        return voted_track_label

    def load_tracklets_for_camera(self, track_dirs: list[Path], camera_id: str | None = None, filter_invalids: bool = True) -> None:

        camera_id = camera_id or self.camera_id            
        # track_files = sorted(track_dir.glob("*.csv"))
        track_files = []
        for track_dir in track_dirs:
            files = sorted(track_dir.glob("*.csv"))
            files = [file for file in files if f"_id_behavior" not in file.stem]
            track_files.extend(files)
                    
        n_invalid = 0
        polygons = self.invalid_zone_handler.load_zones(camera_id)
        
        for track_file in track_files:

            track_filename = track_file.stem

            try:
                s_time, raw_track_id = track_filename.split("T")[1].split("_ID")
            except Exception as e:
                self.logger.warning(f"Unexpected track file name format: {track_filename}, skipping.")
                continue                

            if str(raw_track_id) == '-00001':
                # skip invalid tracks
                continue
            
            tracklet = init_tracklets_from_online_results(
                track_csv=str(track_file),
                camera_id=camera_id
            )
            
            if not tracklet:
                continue

            start_dt = tracklet[0].start_timestamp       # datetime
            end_dt   = tracklet[0].end_timestamp         # datetime

            # print(f"Tracklet {track_filename} start time: {start_dt}, end time: {end_dt}")
            
            # Check if tracklet is within time window
            skip_tracklet = self.validate_tracklet_timestamp(start_dt)

            ## skip tracklet in a short time period - less than 1 min
            tracklet_duration = (end_dt - start_dt).total_seconds()
            if tracklet_duration < 60:
                skip_tracklet = True
                # print(f"Skipping tracklet {track_filename} - duration {tracklet_duration} seconds is less than 60 seconds.")
            
            if skip_tracklet:
                # print(f"Skipping tracklet {track_filename}{start_dt} - outside time window [{self.start_time} to {self.end_time}]")
                continue

            if filter_invalids:
                # Check bbox locations - if in invalid zone, mark invalid_flag=True
                df = pd.read_csv(track_file)
                invalid_flag = False
                ### compute the mean bbox of whole tracklet  TODO: update this when we only have yolo-xywh format box
                box_mean = df[['bbox_top', 'bbox_left', 'bbox_bottom', 'bbox_right']].mean().tolist()
                if self.invalid_zone_handler.is_in_invalid_zone(box_mean, camera_id, self.height, self.width):
                    invalid_flag = True
                    n_invalid += 1
                    print(f"Tracklet {track_filename} marked as invalid (starts in invalid zone)")
                ### set invalid_flag
                tracklet[0].invalid_flag = invalid_flag

            tracklet[0].track_filename = track_filename
            tracklet[0].track_csv_path = track_file

            self.tracklets.extend(tracklet)
        print(f"Loaded {len(self.tracklets)} tracklets for camera {camera_id}, "
              f"marked {n_invalid} as invalid due to invalid zones."
              f"valid tracklets: {len(self.tracklets)-n_invalid}")


    def aggregate_tracklet_features(self) -> None:
        """Aggregate features for all loaded tracklets, and store the average features."""

        self.track_ids = set([t.track_id for t in self.tracklets if t.stitched_id is not None])

        agg_features = {
            t_id: [] for t_id in self.track_ids
        }

        for t in self.tracklets:
            if t.feature_path and t.feature_path.exists():
                if t.stitched_id is None:
                    continue
                feats, frame_ids, _, avg_feat = load_embedding(t.feature_path)
                agg_features[t.stitched_id].append(avg_feat)
            
        # now N, 1, D -> N, D -> avg to (1, D)

        for t_id in self.track_ids:
            if len(agg_features[t_id]) > 0:
                features = np.array(agg_features[t_id]).squeeze(1)
                agg_features[t_id] = features.mean(axis=0, keepdims=True)  # (1, D)

            else:
                agg_features[t_id] = None

        return agg_features

                

    def get_stitched_tracklets(self, 
                                gallery_features: np.ndarray = None,
                                gallery_labels: List[str] = None,
                                 ) -> None:
        """
        Save stitched tracklets metadata to json file.
        This does NOT save per-frame data or features,
        only high-level info including stitched_id.

        dict format: stiched_id -> {
            ["track_id": str,   
            "stitched_id": int,
            "camera_id": str,
            "start_timestamp": str,
            "end_timestamp": str,
            ...]
        }
        """

        known_labels = load_semi_gt_ids(date=self.start_time.date(), camera_id=self.camera_id)
        self.stitched_map: Dict[str, List[Dict[str, object]]] = {}
        for t in self.tracklets:
            
            voted_track_label = 'invalid'
            # if voted_track_label is None or voted_track_label == "invalid":
            if t.feature_path and t.feature_path.exists():
                feats, frame_ids, _, avg_feat = load_embedding(t.feature_path)
                voted_track_label = self.vote_identity_label(
                    features=feats,
                    avg_embedding=avg_feat,
                    gallery_features=gallery_features,
                    gallery_labels=gallery_labels,
                    known_labels=known_labels
                )
                                        
            sid = "unassigned" if t.invalid_flag else str(t.stitched_id)
            
            self.stitched_map.setdefault(sid, []).append(
                {
                    "track_id": t.track_id,
                    "raw_track_id": t.raw_track_id,
                    "track_filename": t.track_filename,
                    "track_csv_path": str(t.track_csv_path) if t.track_csv_path else "",
                    "track_video_path": str(t.track_video_path) if t.track_video_path else "",
                    "stitched_id": t.stitched_id,
                    "camera_id": t.camera_id,
                    # "avg_tracklet_label": avg_tracklet_label,  ### avg embedding from all tracklets
                    "voted_track_label": voted_track_label,   ### voted label from this track
                    "semi_gt_labels": known_labels,
                    # "identity_label": identity_label,   ### final label
                    "invalid_flag": t.invalid_flag,
                    "start_timestamp": t.start_timestamp.isoformat() if isinstance(t.start_timestamp, datetime) else "",
                    "end_timestamp": t.end_timestamp.isoformat() if isinstance(t.end_timestamp, datetime) else "",
                    "start_frame_id": t.start_frame_id,
                    "end_frame_id": t.end_frame_id,
                    "feature_path": str(t.feature_path) if t.feature_path else "",
                }
            )
            
        ### update stitched_map - get final voted label for each stitched_id 
        start_time =None
        end_time = None
        trackid2idlabel = self.vote_tracklet_identity_labels(start_time = start_time, end_time = end_time, known_labels=known_labels)
        ### update self.stitched_map: voted_track_label -> tracklets, not track_id -> tracklets

        self.final_stitched_map: Dict[str, List[Dict[str, object]]] = {}
        ### self.final_stitched_map -> update final identity_label for each tracklet
        for sid in self.stitched_map:
            if sid in trackid2idlabel:
                idlabel = trackid2idlabel[sid]
            else:
                continue
            
            tracklets = self.stitched_map[sid]
            
            for t in tracklets:
                t["identity_label"] = idlabel
                # remove 'voted_track_label' field
                # if 'voted_track_label' in t:
                #     del t['voted_track_label']
            
            self.final_stitched_map.setdefault(idlabel, []).extend(tracklets)


        return self.final_stitched_map
    
    def save_stitched_tracklets(self, 
                                stitched_tracklets: Dict[str, List[Dict[str, object]]],
                                output_dir : Path):

        if self.start_time and self.end_time:
            date_str = '%Y%m%d_%H%M%S'
            time_range_str = f"{self.start_time.strftime(date_str)}_{self.end_time.strftime(date_str)}"
        else:
            raise ValueError("Both start_time and end_time must be specified to save stitched tracklets.")
            
        if output_dir is None:
            raise ValueError("output_dir must be specified to save stitched tracklets.")
        else:
            camera_str = f"zag_elp_cam_{self.camera_id}"
            date_str = self.start_time.strftime("%Y-%m-%d") if self.start_time else "unknown_date"
            save_path = output_dir / camera_str / date_str / f"stitched_tracklets_cam{self.camera_id}_{time_range_str}.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)

        with save_path.open("w") as f:
            json.dump(stitched_tracklets, f, indent=2)

        if self.logger:
            self.logger.info("Saved stitched tracklets JSON to %s", save_path)

    def load_stitched_tracklets_from_dir(self, dir: Path = None) -> Dict[str, List[Dict[str, object]]]:
        """Load stitched tracklets from a JSON file."""        
        
        if dir is None:
            raise ValueError("output_dir must be specified to save stitched tracklets.")
        
        time_range_str = self.start_time.strftime("%Y%m%d_%H%M%S") + "_" + self.end_time.strftime("%Y%m%d_%H%M%S") if self.start_time and self.end_time else "fullday"
        camera_str = f"zag_elp_cam_{self.camera_id}"
        date_str = self.start_time.strftime("%Y-%m-%d") if self.start_time else "unknown_date"
        json_path = dir / camera_str / date_str / f"stitched_tracklets_cam{self.camera_id}_{time_range_str}.json"

        if not json_path.exists():
            self.logger.error(f"Stitched tracklets JSON file not found: {json_path}")
            return {}

        with json_path.open("r") as f:
            stitched_tracklets = json.load(f)

        self.logger.info(f"Loaded stitched tracklets from {json_path}")
        return stitched_tracklets


    def vote_tracklet_identity_labels(self, start_time: Optional[datetime | str] = None, 
                                       end_time: Optional[datetime | str] = None,
                                       filter_labels: List[str] = ["Thai"], 
                                       known_labels: List[str] = None) -> Dict[str, str]:
        """Vote on final identity labels for each stitched tracklet ID based on a time window.
        
        Args:
            start_time: Start time for filtering tracklets (datetime or string "HH:MM:SS" or "yyyymmdd HH:MM:SS")
            end_time: End time for filtering tracklets (datetime or string "HH:MM:SS" or "yyyymmdd HH:MM:SS")
            filter_labels: Labels to filter out when multiple IDs are detected (default: ["Thai"])
            
        Returns:
            Dict mapping stitched_id (str) -> final identity label (str)
        """
        from collections import Counter
        
        if not hasattr(self, 'stitched_map') or not self.stitched_map:
            self.logger.warning("No stitched_map available. Please run save_stitched_tracklets first.")
            return {}
        
        # Convert string inputs to datetime objects
        if start_time and isinstance(start_time, str):
            start_time = self._format_time(start_time)
        if end_time and isinstance(end_time, str):
            end_time = self._format_time(end_time)
        
        trackid2idlabel = {}
        trackid2label_counts = {}
        
        # Process each stitched ID
        for stitched_id, tracklets in self.stitched_map.items():
            if stitched_id == "unassigned":
                continue
            
            # Filter tracklets by time range
            filtered_tracklets = []
            for t in tracklets:
                if start_time or end_time:
                    t_start = datetime.fromisoformat(t["start_timestamp"]) if t.get("start_timestamp") else None
                    if not t_start:
                        continue
                    
                    # Apply time filtering logic
                    if start_time and end_time:
                        if start_time > end_time:
                            if not (t_start >= start_time or t_start <= end_time):
                                continue
                        else:
                            if not (start_time <= t_start <= end_time):
                                continue
                    elif start_time and t_start < start_time:
                        continue
                    elif end_time and t_start > end_time:
                        continue
                
                filtered_tracklets.append(t)

            # Skip invalid tracklets
            filtered_tracklets = [t for t in filtered_tracklets if not t.get("voted_track_label") == 'invalid']
            
            # Skip if no tracklets in time window
            if not filtered_tracklets:
                trackid2idlabel[stitched_id] = "invalid"
                print("=================== No tracklets for stitched_id", stitched_id, "in time window ===================")
                continue
            
            # Count voted labels
            label_counts = Counter()
            for t in filtered_tracklets:
                label = t.get("voted_track_label", "invalid") or "invalid"
                if label != "invalid":
                    label_counts[label] += 1
            
            # Get unique labels (excluding "invalid")
            unique_labels = [label for label in label_counts.keys() if label != "invalid"]
            
            # Apply filtering logic -- filter out the wrong labels from group if not Thai in the room
            if len(unique_labels) >= 2 and ("Thai" not in known_labels if known_labels else True):
                for filter_label in filter_labels:
                    if filter_label in label_counts:
                        del label_counts[filter_label]
                        self.logger.debug(f"Filtered out '{filter_label}' from stitched_id {stitched_id}")
                        
            ### TODO: if known_labels provided, further filter the labels not in known_labels
            if known_labels:
                for label in list(label_counts.keys()):
                    if label not in known_labels:
                        del label_counts[label]
                        self.logger.debug(f"Filtered out invalid label '{label}' from stitched_id {stitched_id} based on known_labels")
            
            trackid2label_counts[stitched_id] = dict(label_counts)
            
            # # Get top voted label
            # if label_counts:
            #     final_label = label_counts.most_common(1)[0][0]
            #     self.logger.info(f"Stitched ID {stitched_id}: voted label = {final_label} (from {dict(label_counts)})")
            # else:
            #     final_label = "invalid"
            #     self.logger.warning(f"Stitched ID {stitched_id}: no valid labels after filtering")
            
            # trackid2idlabel[stitched_id] = final_label
            
        ### re-ranking based on counts
        if known_labels is not None:
            if len(known_labels) == 1:
                for sid in trackid2label_counts.keys():
                    trackid2idlabel[sid] = 'Thai'
            else:
                ranks = assign_identities_from_trackid2label_counts(trackid2label_counts, known_labels, score_mode="ratio")
                for sid, info in ranks.items():
                    trackid2idlabel[sid] = info["assigned_label"]
        
        print("====== Known labels used for voting: ", known_labels, "======")
        self.analyze_stitching_results(trackid2idlabel=trackid2idlabel, start_time=start_time, end_time=end_time)
        
        return trackid2idlabel

    def stitch_tracklets_bidirectional(
        self,
        max_gap_frames: int = 600,
        max_long_gap_frames : int = 30000,
        local_sim_th: float = 0.5,
        gallery_sim_th: float = 0.45,
        long_gap_gallery_th: float = 0.6,
        head_k: int = 5,
        tail_k: int = 5,
        gallery_k: int = 10,
        w_local: float = 0.6,
        w_gallery: float = 0.4,
    ):
        """
        """
        from post_processing.core.stitching import stitch_tracklets_bidirectional_gallery
        
        cam_tracklets = [t for t in self.tracklets if t.camera_id == self.camera_id and t.invalid_flag==False]
        if not cam_tracklets:
            print(
                f"[bidirectional] No tracklets found for camera {self.camera_id}."
            )
            return
        print("Total tracklets for camera", self.camera_id, ":", len(cam_tracklets))
        stitch_tracklets_bidirectional_gallery(
            cam_tracklets,
            max_gap_frames=max_gap_frames, 
            max_long_gap_frames=max_long_gap_frames,
            local_sim_th=local_sim_th,
            gallery_sim_th=gallery_sim_th,
            long_gap_gallery_th=long_gap_gallery_th,
            head_k=head_k,
            tail_k=tail_k,
            gallery_k=gallery_k,
            w_local=w_local,
            w_gallery=w_gallery,
            logger_=self.logger,
        )


        num_ids = len(set(t.stitched_id for t in cam_tracklets))
        print(
            f"[bidirectional] camera {self.camera_id}: "
            f"{len(cam_tracklets)} tracklets -> {num_ids} stitched_ids "
            f"(max_gap_frames={max_gap_frames}, local_th={local_sim_th}, "
            f"gallery_th={gallery_sim_th})"
        )

        num_ids = len(set(t.stitched_id for t in cam_tracklets))
        print(
            f"[bidirectional] camera {self.camera_id}: "
            f"{len(cam_tracklets)} tracklets -> {num_ids} stitched_ids "
            f"(max_gap_frames={max_gap_frames}, local_th={local_sim_th}, "
            f"gallery_th={gallery_sim_th})"
        )
        
        
    def analyze_stitching_results(self, trackid2idlabel: Dict[str, str], 
                                  start_time: Optional[datetime | str] = None, 
                                 end_time: Optional[datetime | str] = None) -> None:
        """
        Analyze and print stitching results using self.stitched_map.
        Groups tracklets by stitched_id and shows voted identity labels.
        
        Args:
            start_time: Optional start time to filter tracklets (datetime object or string "HH:MM:SS" or "yyyymmdd HH:MM:SS")
            end_time: Optional end time to filter tracklets (datetime object or string "HH:MM:SS" or "yyyymmdd HH:MM:SS")
        """
        from collections import Counter
        
        if not hasattr(self, 'stitched_map') or not self.stitched_map:
            print("No stitched_map available. Please run save_stitched_tracklets first.")
            return
        
        print("\n" + "="*80)
        print("STITCHING ANALYSIS RESULTS")
        if start_time or end_time:
            time_filter_str = f" (Time filter: {start_time or 'start'} to {end_time or 'end'})"
            print(time_filter_str)
        print("="*80)
        
        # Sort keys, putting "unassigned" at the end
        sorted_keys = sorted([k for k in self.stitched_map.keys() if k != "unassigned"])
        if "unassigned" in self.stitched_map:
            sorted_keys.append("unassigned")
        
        # Helper function to filter tracklets by time
        def filter_by_time(tracklets, start_time, end_time):
            if not start_time and not end_time:
                return tracklets
            
            filtered = []
            for t in tracklets:
                t_start = datetime.fromisoformat(t["start_timestamp"]) if t.get("start_timestamp") else None
                if not t_start:
                    continue
                
                # Apply time filtering logic
                if start_time and end_time:
                    if start_time > end_time:  # Midnight crossing
                        if not (t_start >= start_time or t_start <= end_time):
                            continue
                    else:
                        if not (start_time <= t_start <= end_time):
                            continue
                elif start_time and t_start < start_time:
                    continue
                elif end_time and t_start > end_time:
                    continue
                
                filtered.append(t)
            return filtered
        
        # Analyze each stitched group
        for stitched_id in sorted_keys:
            if stitched_id == "unassigned":
                continue  # Skip unassigned for main analysis
                
            tracklets = self.stitched_map[stitched_id]
            filtered_tracklets = filter_by_time(tracklets, start_time, end_time)
            
            # Skip if no tracklets after filtering
            if not filtered_tracklets:
                continue
            
            # Count voted labels for display
            label_counts = Counter()
            for t in filtered_tracklets:
                label = t.get("voted_track_label", "invalid") or "invalid"
                label_counts[label] += 1
            
            # Calculate time span
            start_times = [datetime.fromisoformat(t["start_timestamp"]) 
                          for t in filtered_tracklets if t.get("start_timestamp")]
            end_times = [datetime.fromisoformat(t["end_timestamp"]) 
                        for t in filtered_tracklets if t.get("end_timestamp")]
            
            if start_times and end_times:
                time_span = max(end_times) - min(start_times)
                duration_str = str(time_span)
            else:
                duration_str = "N/A"
            
            # Get final voted identity from vote_tracklet_identity_labels
            final_identity = trackid2idlabel.get(stitched_id, "invalid")
            
            print(f"\nStitched ID: {stitched_id}")
            print(f"  Final Identity: {final_identity}")
            print(f"  Number of tracklets: {len(filtered_tracklets)}")
            print(f"  Time span: {duration_str}")
            print(f"  Voted labels distribution:")
            for label, count in label_counts.most_common():
                percentage = (count / len(filtered_tracklets)) * 100
                print(f"    {label}: {count} ({percentage:.1f}%)")
        
        # Summary statistics
        total_tracklets = 0
        unassigned_count = 0
        
        for sid, tracklets in self.stitched_map.items():
            filtered = filter_by_time(tracklets, start_time, end_time)
            total_tracklets += len(filtered)
            if sid == "unassigned":
                unassigned_count += len(filtered)
        
        assigned_tracklets = total_tracklets - unassigned_count
        num_stitched_ids = len([k for k in trackid2idlabel.keys()])
        
        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)
        print(f"Total tracklets (in time range): {total_tracklets}")
        print(f"Assigned tracklets: {assigned_tracklets} ({assigned_tracklets/total_tracklets*100:.1f}%)" if total_tracklets > 0 else "Assigned tracklets: 0")
        print(f"Unassigned tracklets: {unassigned_count} ({unassigned_count/total_tracklets*100:.1f}%)" if total_tracklets > 0 else "Unassigned tracklets: 0")
        print(f"Number of unique stitched IDs: {num_stitched_ids}")
        print(f"Unique identities: {set(trackid2idlabel.values())}")
        print("="*80 + "\n")





if __name__ == "__main__":


    camera_id = "016"
    date = "2025-11-29"

    # start_time = "01:00:00"
    # end_time   = "04:59:59"

    start_time = None
    end_time   = None

    track_dir=Path(f"/media/ElephantsWD/elephants/test/tracks/zag_elp_cam_{camera_id}/{date}")
    
    tracklet_manager = TrackletManager(
        track_dir=track_dir,
        camera_id=camera_id,
        num_identities=2,
        logger=logger,
        start_time=start_time,
        end_time=end_time,
    )
    tracklet_manager.load_tracklets_for_camera()
    print(f"Loaded {len(tracklet_manager.tracklets)} tracklets for camera {tracklet_manager.camera_id}")

    # Test the new bidirectional gallery-based stitching
    print("\n" + "="*80)
    print("Testing BIDIRECTIONAL GALLERY-based stitching (recommended for elephants)")
    print("="*80)
    tracklet_manager.stitch_tracklets_bidirectional(
        max_gap_frames=600,
        local_sim_th=0.5,
        gallery_sim_th=0.45,
        head_k=5,
        tail_k=5,
        gallery_k=10,
        w_local=0.6,
        w_gallery=0.4,
    )

    # tracklet_manager.save_stitched_tracklets()
    tracklet_manager.aggregate_tracklet_features()

    # # Visualize results
    # from post_processing.tools.visualization import visualize_stitched_tracks_pairs, plot_stitched_ids_on_original_frames
    # visualize_stitched_tracks_pairs(
    #     tracklet_manager.tracklets,
    #     output_dir=Path("/media/mu/zoo_vision/post_processing/scrips/out"),
    #     camera_id=camera_id,
    #     head_k=3,
    #     tail_k=3,
    #     max_chains=None,
    #     max_tracklets_per_chain=None,
    #     cell_h=256,
    #     cell_w=256,
    #     logger_=logger,
    # )
    
    # from tests.validate_stitching_timeline import validate_stitched_timelines
    # validate_stitched_timelines(
    #     tracklet_manager.tracklets,
    #     camera_id=camera_id,
    #     logger_=logger,
    # )


    # print("validate_stitched_timelines", validate_stitched_timelines)


    # ## TODO - vis - tracks

    # ampm = 'AM'

    # # video1= f'/mnt/camera_nas/ZAG-ELP-CAM-{camera_id}/20251129{ampm}/ZAG-ELP-CAM-{camera_id}-20251129-001949-1764371989420-7.mp4'
    # # video2= f'/mnt/camera_nas/ZAG-ELP-CAM-{camera_id}/20251129{ampm}/ZAG-ELP-CAM-{camera_id}-20251129-011949-1764375589549-7.mp4'

    # plot_stitched_ids_on_original_frames(
    #     tracklet_manager.tracklets,
    #     camera_id=camera_id,         # or "016", or None for all
    #     max_ids=5,               # plot first 5 identities
    #     max_tracklets_per_id=8,  # up to 8 tracklets per identity
    #     head_k=3,
    #     cols=4,
    # )
