import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from pathlib import Path


import logging
logger = logging.getLogger(__name__)


@dataclass
class Tracklet:
    track_id: str
    raw_track_id: str  # original track id before stitching -- linked to track_file name (*.mkv)
    
    camera_id: str = ""
    behavior_label: str = "unknown"
    behavior_conf: float = 0.0
    identity_label: str = "unknown"
    ori_identity_label: str = "unknown"   ## TODO: ?
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
    t = Tracklet(track_id=track_id, 
                 raw_track_id=track_id, 
                 camera_id=camera_id or "",
                 start_timestamp=start_timestamp,
                 end_timestamp=end_timestamp,
                 feature_path=Path(track_csv).with_suffix(".npz"),
                 start_frame_id=start_frame_id,
                 end_frame_id=end_frame_id
                 )
    
    # print(f"Initialized tracklet {track_id} from {track_csv} with {len(df)} frames.", start_timestamp, end_timestamp)

    return [t]




class TrackletManager:
    """
    Tracklet Manager for loading and stitching tracklets within single camera.

    """
    def __init__(self, 
                 track_dir: Path,
                 camera_id: str, 
                 num_identities: int = 2,
                 logger =None):
        
        self.track_dir = track_dir
        self.camera_id = camera_id
        self.num_identities = num_identities
        self.logger = logger
        if not logger:
            import logging
            self.logger = logging.getLogger(__name__)
        self.tracklets: List[Tracklet] = []


    def load_tracklets_for_camera(self, camera_id: str | None = None) -> None:

        camera_id = camera_id or self.camera_id            
        track_files = sorted(self.track_dir.glob("*.csv"))
        
        
        for track_file in track_files:
            tracklet = init_tracklets_from_online_results(
                track_csv=str(track_file),
                camera_id=camera_id
            )
            self.tracklets.extend(tracklet)


    def stitch_tracklets_within_camera(self, camera_id: str) -> None:
        """
        Stitch tracklets that belong to the given camera_id.

        Note:
        - This operates in-place: it sets Tracklet.stitched_id.
        - If TrackletManager only holds tracklets from this camera,
          you could ignore the camera_id argument; here we filter for safety.
        """

        from stitching import stitch_tracklets

        print(f"Stitching tracklets for camera {camera_id}...")

        cam_tracklets = [t for t in self.tracklets if t.camera_id == self.camera_id]
        self.logger.info(f"Found {len(cam_tracklets)} tracklets for camera {self.camera_id} to stitch.")

        if not cam_tracklets:
            self.logger.warning(
                f"No tracklets found for camera {self.camera_id} to stitch."
            )
            return

        stitch_tracklets(cam_tracklets,
                         num_identities=2)

        num_chains = len(set(t.stitched_id for t in cam_tracklets))
        self.logger.info(
            f"Stitched {len(cam_tracklets)} tracklets into {num_chains} stitched_ids "
            f"for camera {self.camera_id}."
        )

            

    def save_stitched_tracklets(self, save_path: Path) -> None:
        """
        Save stitched tracklets metadata to a CSV file.

        This does NOT save per-frame data or features,
        only high-level info including stitched_id.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "camera_id",
            "track_id",
            "raw_track_id",
            "stitched_id",
            "behavior_label",
            "behavior_conf",
            "identity_label",
            "ori_identity_label",
            "invalid_flag",
            "start_timestamp",
            "end_timestamp",
            "feature_path",
        ]

        with save_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for t in self.tracklets:
                row = {
                    "camera_id": t.camera_id,
                    "track_id": t.track_id,
                    "raw_track_id": t.raw_track_id,
                    "stitched_id": t.stitched_id,
                    "behavior_label": t.behavior_label,
                    "behavior_conf": t.behavior_conf,
                    "identity_label": t.identity_label,
                    "ori_identity_label": t.ori_identity_label,
                    "invalid_flag": t.invalid_flag,
                    "start_timestamp": (
                        t.start_timestamp.isoformat()
                        if isinstance(t.start_timestamp, datetime)
                        else ""
                    ),
                    "end_timestamp": (
                        t.end_timestamp.isoformat()
                        if isinstance(t.end_timestamp, datetime)
                        else ""
                    ),
                    "feature_path": str(t.feature_path) if t.feature_path else "",
                }
                writer.writerow(row)

        if self.logger:
            self.logger.info(
                f"Saved {len(self.tracklets)} stitched tracklets to {save_path}."
            )





if __name__ == "__main__":

    track_dir=Path("/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20250318/20250318_15/ZAG-ELP-CAM-019-20250318-155551-1742309751619-7/tracks")
    
    tracklet_manager = TrackletManager(
        track_dir=track_dir,
        camera_id="019",
        num_identities=2,
        logger=logger
    )
    tracklet_manager.load_tracklets_for_camera()
    print(f"Loaded {len(tracklet_manager.tracklets)} tracklets for camera {tracklet_manager.camera_id}")

    tracklet_manager.stitch_tracklets_within_camera(camera_id="019")
    # tracklet_manager.save_stitched_tracklets(Path("stitched_tracklets_CAM_016.csv"))
