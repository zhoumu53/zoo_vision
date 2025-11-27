from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

DEFAULT_FIXES_NAME = "tracks_fixes.json"


def load_track_log(path: str | Path) -> Tuple[pd.DataFrame, dict]:
    """Load the JSONL track log produced by video_tracks_reid_improved.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Track log not found: {path}")

    rows = []
    meta: dict = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msg_type = obj.get("type")
            if msg_type == "meta":
                meta = obj
                continue
            if msg_type != "frame":
                continue

            frame_idx = int(obj.get("frame_idx", -1))
            timestamp_s = obj.get("timestamp_s")
            for trk in obj.get("tracks", []):
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_s": timestamp_s,
                        "raw_track_id": int(trk.get("raw_track_id", -1)),
                        "canonical_track_id": int(
                            trk.get("canonical_track_id", trk.get("raw_track_id", -1))
                        ),
                        "display_track_id": int(
                            trk.get("display_track_id", trk.get("raw_track_id", -1))
                        ),
                        "bbox": trk.get("bbox"),
                        "score": trk.get("score"),
                        "cls_id": trk.get("cls_id"),
                        "cls_name": trk.get("cls_name"),
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["frame_idx", "canonical_track_id", "raw_track_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df, meta


def load_fixes(path: str | Path) -> Dict[int, int]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    mapping = data.get("canonical_to_fixed", data)
    return {int(k): int(v) for k, v in mapping.items()}


def save_fixes(path: str | Path, fixes: Dict[int, int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"canonical_to_fixed": {str(k): int(v) for k, v in fixes.items()}}
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def apply_fixes(tracks: pd.DataFrame, fixes: Dict[int, int]) -> pd.DataFrame:
    if tracks.empty:
        return tracks.assign(fixed_track_id=pd.Series(dtype=int))
    updated = tracks.copy()
    updated["fixed_track_id"] = updated["canonical_track_id"].apply(
        lambda tid: int(fixes.get(int(tid), tid))
    )
    updated["fixed_display_track_id"] = updated["fixed_track_id"]
    return updated


def frame_tracks(tracks: pd.DataFrame, frame_idx: int) -> pd.DataFrame:
    if tracks.empty:
        return tracks
    return tracks.loc[tracks["frame_idx"] == frame_idx]
