import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from pathlib import Path
import numpy as np

import logging
logger = logging.getLogger(__name__)


## TODO: remove redundant items in Tracklet dataclass
@dataclass
class Tracklet:
    track_id: str
    raw_track_id: str  # original track id before stitching -- linked to track_file name (*.mkv)
    track_filename: str = ""  # video file name of the tracklet
    track_csv_path: Optional[Path] = None  # path to the per-track CSV
    track_video_path: Optional[Path] = None  # optional raw video path for visualization
    
    camera_id: str = ""
    identity_label: str = "unknown"
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
    start_timestamp = datetime.fromisoformat(df['timestamp'].iloc[0])
    end_timestamp = datetime.fromisoformat(df['timestamp'].iloc[-1])
    start_frame_id = int(df['frame_id'].iloc[0])
    end_frame_id = int(df['frame_id'].iloc[-1])
    track_filename = Path(track_csv).stem

    ## load identity label from npz if exists
    identity_label = "unknown"
    npz_path = Path(track_csv).with_suffix(".npz")
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=True)
            identity_label = data.get('voted_labels', 'unknown')
            if isinstance(identity_label, np.ndarray):
                identity_label = identity_label.item()
        except Exception as e:
            logger.warning(f"Failed to load identity label from {npz_path}: {e}")

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
                 )
    
    # print(f"Initialized tracklet {track_id} from {track_csv} with {len(df)} frames.", start_timestamp, end_timestamp)

    return [t]

################# TODO: segmentation model for the invalid zone detection if camera view is changed (now the polygon points are fixed) ##################
class InvalidZoneHandler:
    """Handle invalid zone detection and filtering."""

    def __init__(self, invalid_zones_dir: Optional[Path], logger: Optional[logging.Logger] = None,
                 original_height: int = 1520, original_width: int = 2688):
        self.invalid_zones_dir = invalid_zones_dir
        self.logger = logger or logging.getLogger(__name__)
        self.zone_cache: Dict[str, List[np.ndarray]] = {}
        self.original_height = original_height  # Original image height when zones were annotated
        self.original_width = original_width    # Original image width when zones were annotated

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
            import json
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
        import cv2

        polygons = self.load_zones(camera_id)

        if not polygons or not bbox or len(bbox) != 4:
            return False

        # Calculate scaling factors if image size differs from original
        scale_x = width / self.original_width
        scale_y = height / self.original_height
        
        # Only log if scaling is significant (more than 1% difference)
        if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
            self.logger.debug(
                f"Scaling polygons: original ({self.original_width}x{self.original_height}) -> "
                f"current ({width}x{height}), scale_x={scale_x:.3f}, scale_y={scale_y:.3f}"
            )

        # bbox format: [top, left, bottom, right]
        top, left, bottom, right = bbox
        center_x = int((left + right) / 2)
        center_y = int((top + bottom) / 2)
        center_point = (center_x, center_y)

        # Check if center point is inside any polygon
        for polygon in polygons:
            # Scale polygon points to match current image size
            scaled_polygon = polygon.copy().astype(np.float32)
            scaled_polygon[:, 0] *= scale_x  # Scale x coordinates
            scaled_polygon[:, 1] *= scale_y  # Scale y coordinates
            scaled_polygon = scaled_polygon.astype(np.int32)
            
            result = cv2.pointPolygonTest(scaled_polygon, center_point, False)
            if result >= 0:  # Inside or on the boundary
                return True

        return False

    def get_scaled_polygons(self, camera_id: str, height: int, width: int) -> Optional[List[np.ndarray]]:
        """Get polygons scaled to match the target image size.
        
        Args:
            camera_id: Camera identifier
            height: Target image height
            width: Target image width
            
        Returns:
            List of scaled polygon arrays, or None if no zones found
        """
        polygons = self.load_zones(camera_id)
        
        if not polygons:
            return None
        
        # Calculate scaling factors
        scale_x = width / self.original_width
        scale_y = height / self.original_height
        
        scaled_polygons = []
        for polygon in polygons:
            # Scale polygon points to match target image size
            scaled_polygon = polygon.copy().astype(np.float32)
            scaled_polygon[:, 0] *= scale_x  # Scale x coordinates
            scaled_polygon[:, 1] *= scale_y  # Scale y coordinates
            scaled_polygon = scaled_polygon.astype(np.int32)
            scaled_polygons.append(scaled_polygon)
        
        return scaled_polygons


class TrackletManager:
    """
    Tracklet Manager for loading and stitching tracklets within single camera.

    """
    def __init__(self, 
                 track_dir: Path,
                 camera_id: str, 
                 num_identities: int = 2,
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None,
                 logger =None,
                 invalid_zones_dir: Path = Path('/media/mu/zoo_vision/data/invalid_zones'),
                 height: int = 1520,
                 width: int = 1920
                 ):
        
        """Initialize TrackletManager.

        Args:
            track_dir: Directory containing track CSV files.
            camera_id: Camera identifier.
            num_identities: Expected number of unique identities for stitching.
            start_time: Optional start time to filter tracklets.
            end_time: Optional end time to filter tracklets.
            logger: Optional logger for logging messages.
            invalid_zones_dir: Directory containing invalid zone JSON files.
            height: Image height for invalid zone scaling.
            width: Image width for invalid zone scaling.

        """
        
        self.track_dir = track_dir
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

        self.invalid_zone_handler = InvalidZoneHandler(invalid_zones_dir, self.logger)

    def _format_time(self, timestamp) -> datetime:
        """Convert time string 'HH:MM:SS' to datetime object, removing date component."""
        if isinstance(timestamp, str):
            return datetime.strptime(timestamp, "%H:%M:%S").time()
        elif isinstance(timestamp, datetime):
            return timestamp.time()
            

    def load_tracklets_for_camera(self, camera_id: str | None = None, filter_invalids: bool = True) -> None:

        camera_id = camera_id or self.camera_id            
        track_files = sorted(self.track_dir.glob("*.csv"))


        n_invalid = 0
        
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

            start_dt = tracklet[0].start_timestamp       # datetime
            end_dt   = tracklet[0].end_timestamp         # datetime

            start_t = start_dt.time() if start_dt else None
            end_t   = end_dt.time() if end_dt else None

            # ### note -- bug: we may miss tracklets from a different track file 
            # # 1) Tracklet ends BEFORE time window → skip
            # if self.start_time and end_t and end_t < self.start_time:
            #     print(f"Skipping tracklet {track_filename} ending at {end_t} before start_time {self.start_time}")
            #     continue

            # # 2) Tracklet starts AFTER time window → skip
            # if self.end_time and start_t and start_t > self.end_time:
            #     print(f"Skipping tracklet {track_filename} starting at {start_t} after end_time {self.end_time}")
            #     continue

            ### we only keep the tracklets starting WITHIN the time window, after start_time before end_time
            if self.start_time and start_t and self.start_time > start_t:
                print(f"Skipping tracklet {track_filename} starting at {start_t} before start_time {self.start_time}")
                continue

            if self.end_time and start_t and self.end_time < start_t:
                print(f"Skipping tracklet {track_filename} starting at {start_t} after end_time {self.end_time}")
                continue

            if filter_invalids:
                # Check bbox locations - if in invalid zone, mark invalid_flag=True
                df = pd.read_csv(track_file)
                invalid_flag = False
                ### compute the mean bbox of whole tracklet
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


    def save_stitched_tracklets(self, save_path: Path = None) -> None:
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
        import json

        if self.start_time and self.end_time:
            time_range_str = f"{self.start_time.strftime('%H%M%S')}_{self.end_time.strftime('%H%M%S')}"
        else:
            time_range_str = "fullday"
        save_path = Path(save_path) if save_path else self.track_dir / f"stitched_tracklets_cam{self.camera_id}_{time_range_str}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        stitched_map: Dict[str, List[Dict[str, object]]] = {}
        for t in self.tracklets:
            sid = "unassigned" if t.stitched_id is None else str(t.stitched_id)
            stitched_map.setdefault(sid, []).append(
                {
                    "track_id": t.track_id,
                    "raw_track_id": t.raw_track_id,
                    "track_filename": t.track_filename,
                    "track_csv_path": str(t.track_csv_path) if t.track_csv_path else "",
                    "track_video_path": str(t.track_video_path) if t.track_video_path else "",
                    "stitched_id": t.stitched_id,
                    "camera_id": t.camera_id,
                    "identity_label": t.identity_label,
                    "invalid_flag": t.invalid_flag,
                    "start_timestamp": t.start_timestamp.isoformat() if isinstance(t.start_timestamp, datetime) else "",
                    "end_timestamp": t.end_timestamp.isoformat() if isinstance(t.end_timestamp, datetime) else "",
                    "start_frame_id": t.start_frame_id,
                    "end_frame_id": t.end_frame_id,
                    "feature_path": str(t.feature_path) if t.feature_path else "",
                }
            )

        with save_path.open("w") as f:
            json.dump(stitched_map, f, indent=2)

        if self.logger:
            self.logger.info("Saved stitched tracklets JSON to %s", save_path)


    def stitch_tracklets_bidirectional(
        self,
        max_gap_frames: int = 600,
        local_sim_th: float = 0.5,
        gallery_sim_th: float = 0.45,
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
            num_identities=self.num_identities,
            max_gap_frames=max_gap_frames,
            local_sim_th=local_sim_th,
            gallery_sim_th=gallery_sim_th,
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




if __name__ == "__main__":


    camera_id = "016"
    date = "2025-11-29"

    # start_time = "01:00:00"
    # end_time   = "04:59:59"

    start_time = None
    end_time   = None

    track_dir=Path(f"/media/dherrera/ElephantsWD/elephants/test/tracks/zag_elp_cam_{camera_id}/{date}")
    
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

    # Visualize results
    from post_processing.tools.visualization import visualize_stitched_tracks_pairs, plot_stitched_ids_on_original_frames
    visualize_stitched_tracks_pairs(
        tracklet_manager.tracklets,
        output_dir=Path("/media/mu/zoo_vision/post_processing/scrips/out"),
        camera_id=camera_id,
        head_k=3,
        tail_k=3,
        max_chains=None,
        max_tracklets_per_chain=None,
        cell_h=256,
        cell_w=256,
        logger_=logger,
    )
    
    from tests.validate_stitching_timeline import validate_stitched_timelines
    validate_stitched_timelines(
        tracklet_manager.tracklets,
        camera_id=camera_id,
        logger_=logger,
    )


    print("validate_stitched_timelines", validate_stitched_timelines)


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
