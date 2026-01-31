
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import re

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
POSE_REID_ROOT = PROJECT_ROOT / "training" / "PoseGuidedReID"

for path in (PROJECT_ROOT, POSE_REID_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))



GT_IMAGES_DIR = '/media/ElephantsWD/elephants/reid_gt_cleaned_data'
SEMI_GT_ID_CSV = '/media/mu/zoo_vision/data/elephants_sandbox.csv'

ROOM_PAIRS = {
    "ohne": ["016", "019"],
    "mit": ["017", "018"],
    'both': ["016", "017", "018", "019"],
}

CAMERA_TO_ROOM = {
    "016": "ohne",
    "019": "ohne",
    "017": "mit",
    "018": "mit",
}

COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (231, 76, 60),
    (46, 205, 113),
    (52, 152, 219),
    (155, 89, 182),
    (241, 196, 15),
]


ID2NAMES = {
    '01': 'Chandra',
    '02': 'Indi',
    '03': 'Fahra',
    '04': 'Panang',
    '05': 'Thai',
    '06': 'Zali',
}

SOCIAL_GROUPS = {
    1: ['Chandra', 'Indi',],
    2: ['Panang',  'Fahra'],
    3: ['Thai'],
}


IDENTITY_NAMES = [
    "Chandra",
    "Indi",
    "Fahra",
    "Panang",
    "Thai",
    "Zali",
]


DEFAULT_IDENTITY_LABELS = [
    "01_Chandra",
    "02_Indi",
    "03_Fahra",
    "04_Panang",
    "05_Thai",
    "06_Zali",
]


CAMERA_PARIS = {
    '016': '019',
    '019': '016',
    '017': '018',
    '018': '017',
}


# type fix
TYPO = {
    'Farha': 'Fahra',
}

ALIKE_PAIRS = {
    ('Indi', 'Panang', 'Thai'),
    ('Chandra', 'Fahra'),
}



def load_feature_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Path]:
    """Load all embeddings and frame ids from NPZ; return (features, frame_ids, video_path)."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature NPZ file not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return data


def load_embedding(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Path]:
    """Load all embeddings and frame ids from NPZ; return (features, frame_ids, video_path)."""
    data = np.load(npz_path, allow_pickle=True)
    feats = data["features"]
    avg_feat = data['avg_embedding'] if 'avg_embedding' in data else feats.mean(axis=0)
    frame_ids = data.get("frame_ids", np.arange(len(feats)))
    video_path = npz_path.with_suffix(".mkv")
    metadata = data.get("metadata", None)
    if metadata is not None:
        try:
            meta_obj = metadata.item() if hasattr(metadata, "item") else metadata
            if isinstance(meta_obj, dict) and "video" in meta_obj:
                video_path = Path(meta_obj["video"])
        except Exception:
            pass
    return feats.astype(np.float32), frame_ids, video_path, avg_feat.astype(np.float32) if avg_feat is not None else None


## LOAD SANDBOX GTs

def parse_first_datetime(s: str) -> str:
    if not isinstance(s, str):
        return s
    # split on our aggregation delimiter
    parts = [p.strip() for p in re.split(r"\s*\|\|\s*", s) if p.strip()]
    return parts[0] if parts else s


def load_sandbox_gts(sandbox_gt_path: Path = Path(SEMI_GT_ID_CSV), 
                     columns: List[str] = ['date', 'individual', 'room']) -> List[Path]:


    df = pd.read_csv(sandbox_gt_path, usecols=columns)
    # remove the rows with 'room' as 'others'
    df = df[df['room'] != 'others']
    df["datetime"] = df["date"].astype(str).map(parse_first_datetime)
    # Your timestamps are ISO-like, so dayfirst must be False
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise", dayfirst=False)
    # remove date
    df["date"] = df["datetime"].dt.date

    # check if 'individual' has typos
    df['individual'] = df['individual'].replace(TYPO)
    
    return df

def load_semi_gt_ids(sandbox_gt_path: Path = Path(SEMI_GT_ID_CSV),
                     date: str = '', 
                     camera_id: str = '') -> List[str]:
    try:
        df = load_sandbox_gts()
        df = df[df['date'] == pd.to_datetime(date).date()]
        room = CAMERA_TO_ROOM[camera_id]
        df = df[df['room'] == room]
        return df['individual'].unique().tolist()
    except Exception as e:
        print(f"No semi GT IDs found for date {date} and camera {camera_id}.")
        return []
    
    
def get_dates_with_semi_gt(sandbox_gt_path: Path = Path(SEMI_GT_ID_CSV),
                           processed_dir: Path = Path('/media/ElephantsWD/elephants/xmas/tracks/zag_elp_cam_016')) -> List[str]:
    
    df = load_sandbox_gts()
    gt_dates = [date.strftime('%Y-%m-%d') for date in df['date'].unique().tolist()]
    
    # list all folder under processed_dir
    folder_dates = [p.name for p in processed_dir.iterdir() if p.is_dir()]
    
    # intersect gt_dates and folder_dates
    gt_dates = list(set(gt_dates).intersection(set(folder_dates)))
    gt_dates.sort()
    
    return gt_dates

if __name__ == "__main__":
    import pandas as pd
    # ids = load_semi_gt_ids(date='2025-12-01', camera_id='016')
    dates = get_dates_with_semi_gt()
    print(dates)