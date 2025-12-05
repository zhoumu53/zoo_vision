import torch
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Tracklet:
    track_id: int
    frames: List[int] = field(default_factory=list)
    camera_id: str = ""
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    bboxes: List[List[float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    # behavior_label: str = "unknown"
    # behavior_conf: float = 0.0
    identity_label: str = "unknown"
    ori_identity_label: str = "unknown"
    features: Optional[torch.Tensor] = None
    stitched_id: Optional[int] = None


class TrackletManager:
    def __init__(self):
        self.tracklets: List[Tracklet] = []

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamps of the form YYYYMMDD_HHMMSS."""
        try:
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except Exception:
            return None

    def _camera_from_frame(self, frame_name: str) -> str:
        # frame name pattern: ZAG-ELP-CAM-017-20251129-001907-...jpg
        parts = frame_name.split("-")
        return parts[3] if len(parts) > 3 else ""

    def from_tracking_results(self, frames: List[Dict]) -> List[Tracklet]:
        """
        Build tracklets from tracking JSONL records (frame-wise dicts).

        Each frame dict should contain `frame_idx`, `timestamp`, and a `tracks` list
        with per-track dictionaries as in the sample JSON.
        """
        tracklets_by_id: Dict[int, Tracklet] = {}


        print(frames[0])

        unique_track_ids = set()

        for frame in frames:
            frame = frame.get("results", frame)

            frame_idx = frame.get("frame_idx")
            

            tracks = frame.get("tracks", [])

            if len(tracks) == 0:
                continue

            for trk in tracks:
                track_id = trk.get("raw_track_id")
                if track_id is None:
                    continue
                unique_track_ids.add(track_id)

                tl = tracklets_by_id.setdefault(
                    track_id,
                    Tracklet(track_id=track_id, 
                             camera_id=self._camera_from_frame(trk.get("frame_name", ""))),
                )

                tl.frames.append(frame_idx)
                tl.bboxes.append(trk.get("bbox", []))
                tl.scores.append(trk.get("score", 0.0))

                frame_ts = self._parse_timestamp(frame.get("timestamp", ""))

                if tl.start_timestamp is None or (frame_ts and frame_ts < tl.start_timestamp):
                    tl.start_timestamp = frame_ts
                if tl.end_timestamp is None or (frame_ts and frame_ts > tl.end_timestamp):
                    tl.end_timestamp = frame_ts


        ### add in timestamps for each tracklet
        tracklets = list(tracklets_by_id.values())
        return tracklets

    def stitch_tracklets(self, stitching_map: Dict[int, int]) -> List[Tracklet]:
        """
        Merge tracklets according to a mapping {track_id: stitched_id}.

        The caller can supply a stitching algorithm elsewhere and pass the mapping here.
        """
        stitched: Dict[int, Tracklet] = {}
        for tl in self.tracklets:
            target_id = stitching_map.get(tl.track_id, tl.track_id)
            merged = stitched.get(target_id)

            if merged is None:
                merged = Tracklet(
                    track_id=tl.track_id,
                    stitched_id=target_id,
                    camera_id=tl.camera_id,
                    frames=list(tl.frames),
                    bboxes=list(tl.bboxes),
                    scores=list(tl.scores),
                    behavior_label=tl.behavior_label,
                    behavior_conf=tl.behavior_conf,
                    identity_label=tl.identity_label,
                    ori_identity_label=tl.ori_identity_label,
                    start_timestamp=tl.start_timestamp,
                    end_timestamp=tl.end_timestamp,
                    features=tl.features,
                )
                stitched[target_id] = merged
                continue

            merged.frames.extend(tl.frames)
            merged.frames.sort()
            merged.bboxes.extend(tl.bboxes)
            merged.scores.extend(tl.scores)
            if tl.start_timestamp and (merged.start_timestamp is None or tl.start_timestamp < merged.start_timestamp):
                merged.start_timestamp = tl.start_timestamp
            if tl.end_timestamp and (merged.end_timestamp is None or tl.end_timestamp > merged.end_timestamp):
                merged.end_timestamp = tl.end_timestamp

            if tl.behavior_conf > merged.behavior_conf:
                merged.behavior_conf = tl.behavior_conf
                merged.behavior_label = tl.behavior_label

        self.tracklets = list(stitched.values())
        return self.tracklets



if __name__ == "__main__":

    from core.file_manager import FileManager

    fm = FileManager("20251129")
    _, records = fm.load_online_tracking_jsonl(fm.get_online_tracking_json_path(cam_id="017", hour=0))
    print(f"Loaded {len(records)} frames from online tracking results", fm.get_online_tracking_json_path(cam_id="017", hour=0))
    tm = TrackletManager()

    tracklets = tm.from_tracking_results(records)
    # stitched = tm.stitch_tracklets({1: 100, 2: 100})  # example mapping
    print(f"Extracted {len(tracklets)} tracklets\n")