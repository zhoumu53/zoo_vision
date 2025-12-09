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
                 end_frame_id=end_frame_id,
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

            track_id = track_file.stem
            if int(track_id) < 0:
                # skip invalid tracks
                continue

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
        print(f"Found {len(cam_tracklets)} tracklets for camera {self.camera_id} to stitch.")

        if not cam_tracklets:
            print(
                f"No tracklets found for camera {self.camera_id} to stitch."
            )
            return

        stitch_tracklets(cam_tracklets,
                         num_identities=2)

        num_chains = len(set(t.stitched_id for t in cam_tracklets))
        print(
            f"Stitched {len(cam_tracklets)} tracklets into {num_chains} stitched_ids "
            f"for camera {self.camera_id}."
        )

    def stitch_tracklets_within_camera_global(self):
        from stitching import stitch_tracklets_global_frames

        cam_tracklets = [t for t in self.tracklets if t.camera_id == self.camera_id]
        if not cam_tracklets:
            print(f"No tracklets for camera {self.camera_id}")
            return

        stitch_tracklets_global_frames(
            cam_tracklets,
            num_identities=self.num_identities,  # 一般是 2
            head_k=5,
            tail_k=5,
            w_global=0.7,
            w_local=0.3,
            assign_th=0.6,
            ema_alpha=0.7,
            logger_=self.logger,
        )

        num_ids = len(set(t.stitched_id for t in cam_tracklets))
        print(
            f"[global_frames] camera {self.camera_id}: "
            f"{len(cam_tracklets)} tracklets -> {num_ids} stitched_ids"
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
            print(
                f"Saved {len(self.tracklets)} stitched tracklets to {save_path}."
            )

    def compare_stitching_methods(
        self,
        sim_th: float = 0.6,
        head_k: int = 1,
        tail_k: int = 1,
        ema_alpha: float = 0.7,
    ):
        import copy
        from collections import Counter
        """
        同时跑 cluster(head+tail) 和 global_frames 两种算法，
        比较每个 track 的 stitched_id 分配是否一致。

        不会修改 self.tracklets 内当前的 stitched_id，
        只在 copy 的对象上跑算法。
        """
        from stitching import (
            stitch_tracklets,
            stitch_tracklets_global_frames,
        )

        cam_tracklets = [t for t in self.tracklets if t.camera_id == self.camera_id]
        if not cam_tracklets:
            print(
                f"[compare] No tracklets for camera {self.camera_id}."
            )
            return

        # 深拷贝两份，用于分别跑两个算法
        cluster_tracklets = copy.deepcopy(cam_tracklets)
        global_tracklets = copy.deepcopy(cam_tracklets)

        # run cluster-based
        stitch_tracklets(
            cluster_tracklets,
            num_identities=self.num_identities,
            head_k=head_k,
            tail_k=tail_k,
        )

        # run global-frames
        stitch_tracklets_global_frames(
            global_tracklets,
            num_identities=self.num_identities,
            head_k=5,
            tail_k=5,
            w_global=0.7,
            w_local=0.3,
            assign_th=0.6,
            ema_alpha=0.7,
        )

        # mapping: track_id -> stitched_id
        cluster_map = {
            t.track_id: t.stitched_id for t in cluster_tracklets
        }
        global_map = {
            t.track_id: t.stitched_id for t in global_tracklets
        }

        # 统计差异
        all_ids = sorted(cluster_map.keys())
        diff_tracks = []
        for tid in all_ids:
            c_id = cluster_map.get(tid, None)
            g_id = global_map.get(tid, None)
            if c_id != g_id:
                diff_tracks.append((tid, c_id, g_id))

        #### print all track ids for debugging  -- map stitch -> track ids
        print("All stitched IDs of tracks:")
        stitch_to_tracks_cluster: Dict[int, List[str]] = {}
        stitch_to_tracks_global: Dict[int, List[str]] = {}
        for tid in all_ids:
            c_id = cluster_map.get(tid, None)
            g_id = global_map.get(tid, None)
            if c_id is not None:
                stitch_to_tracks_cluster.setdefault(c_id, []).append(tid)
            if g_id is not None:
                stitch_to_tracks_global.setdefault(g_id, []).append(tid)
        print("Cluster-based stitching:")
        for sid, tids in stitch_to_tracks_cluster.items():
            print(f"  Stitched ID {sid}: tracks {tids}")
        print("Global-frames stitching:")
        for sid, tids in stitch_to_tracks_global.items():
            print(f"  Stitched ID {sid}: tracks {tids}")

        print(
            f"[compare] Camera {self.camera_id}: "
            f"{len(cam_tracklets)} tracklets. "
            f"cluster unique IDs={len(set(cluster_map.values()))}, "
            f"global unique IDs={len(set(global_map.values()))}, "
            f"different assignments for {len(diff_tracks)} tracklets."
        )

        # 打印每个 stitched_id 下有多少 track（方便 sanity check）
        c_counter = Counter(cluster_map.values())
        g_counter = Counter(global_map.values())
        print(f"[compare] cluster counts: {c_counter}")
        print(f"[compare] global  counts: {g_counter}")

        if diff_tracks:
            # 只打前几条，避免 log 太长
            preview = diff_tracks[:20]
            print(
                "[compare] example differences (track_id, cluster_id, global_id): "
                + ", ".join(str(x) for x in preview)
            )

        # 可以选择返回结果，方便你在 notebook 里直接看
        return cluster_map, global_map, diff_tracks




if __name__ == "__main__":

    track_dir=Path("/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20251129/20251129_01/ZAG-ELP-CAM-016-20251129-011949-1764375589549-7/tracks")
    
    tracklet_manager = TrackletManager(
        track_dir=track_dir,
        camera_id="016",
        num_identities=3,
        logger=logger
    )
    tracklet_manager.load_tracklets_for_camera()
    print(f"Loaded {len(tracklet_manager.tracklets)} tracklets for camera {tracklet_manager.camera_id}")

    # tracklet_manager.stitch_tracklets_within_camera(camera_id="016")
    # tracklet_manager.stitch_tracklets_within_camera_global()
    # tracklet_manager.save_stitched_tracklets(Path("stitched_tracklets_CAM_016.csv"))

    cluster_map, global_map, diff = tracklet_manager.compare_stitching_methods()
    print("num diff tracks:", len(diff))