
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import re

from post_processing.core.reid_inference import match_to_gallery
from post_processing.tools.run_reid_feature_extraction import vote_matched_labels

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



def vote_known_individuals(track_dir: Path, 
                           date: str, 
                           start_time: str, 
                           end_time: str,
                           gallery_features: np.ndarray,
                           gallery_labels: np.ndarray) -> list[str]:
    
    track_files = list(track_dir.glob("*.csv"))
    track_files = [file for file in track_files if f"_behavior" not in file.stem and f"part_" not in file.stem]
    track_files = sorted(track_files)
    ### get track_files between start_time and end_time
    start_dt = pd.to_datetime(f"{date} {start_time}", format="%Y%m%d %H%M%S")
    end_dt = pd.to_datetime(f"{date} {end_time}", format="%Y%m%d %H%M%S")
    
    votes = []
    valid_track_files = []
    for track_file in track_files:
        track_filename = track_file.stem
        behavior_csv = track_file.parent / f"{track_filename}_behavior.csv"
        if not behavior_csv.exists():
            continue
        df_behavior = pd.read_csv(behavior_csv)
        track_start_dt = pd.to_datetime(df_behavior['timestamp'].min())
        # check if track overlaps with the time window
        if (track_start_dt >= start_dt) and (track_start_dt <= end_dt):
            feature_path = track_file.parent / f"{track_filename}.npz"
            features, frame_ids = load_embedding(feature_path)[0:2]
            # now voting based on good quality frames + high confidence behavior + no 'standing' 
            good_indices = df_behavior.index[(df_behavior['quality_label'] == 'good') 
                                                & (df_behavior['behavior_label'] == '01_standing')
                                                & (df_behavior['behavior_label'] != '00_invalid')
                                                & (df_behavior['behavior_conf'].astype(float) >= 0.7)].tolist()
            if len(good_indices) == 0:  ### all bad quality frames -- no voting
                voted_track_label = "invalid"
            ### feature_indices for reid voting - pick the features with good quality
            feature_indices = [i for i, fid in enumerate(frame_ids) if fid in good_indices]
            features = features[feature_indices]
            if len(features) == 0:
                continue
            
            matched_labels = match_to_gallery(features, 
                                            gallery_features, 
                                            gallery_labels=gallery_labels)[-1]
            voted_labels = vote_matched_labels(matched_labels)
            voted_label = max(set(voted_labels), key=voted_labels.count)
            votes.append(voted_label)
            
    vote_counts = pd.Series(votes).value_counts()
    return vote_counts
    

def vote_known_individuals_per_camera_pair(
    vote_results_dict: Dict[str, pd.Series],
    camera_pairs: Dict[str, list] = None
) -> Dict[str, list]:
    """
    Aggregate voting results across camera pairs considering social groups.
    Returns only the TOP 1 social group per camera pair.
    
    Args:
        vote_results_dict: Dict mapping camera_id to pd.Series (or list) of vote counts
        camera_pairs: Dict mapping room name to list of camera IDs
        
    Returns:
        Dict mapping room name to list of individuals from top voted group
    """
    
    if camera_pairs is None:
        camera_pairs = {
            "ohne": ["016", "019"],
            "mit": ["017", "018"],
        }
    
    # Invert SOCIAL_GROUPS to get individual -> group_id mapping
    individual_to_group = {}
    for group_id, individuals in SOCIAL_GROUPS.items():
        for individual in individuals:
            individual_to_group[individual] = group_id
    
    room_individuals = {}
    
    for room_name, cam_ids in camera_pairs.items():
        # Aggregate votes across all cameras in this room
        aggregated_votes = {}
        
        for cam_id in cam_ids:
            if cam_id not in vote_results_dict:
                continue
                
            votes = vote_results_dict[cam_id]
            if votes is None or len(votes) == 0:
                continue
            
            # Handle both pd.Series and list inputs
            if isinstance(votes, pd.Series):
                vote_items = votes.items()
            elif isinstance(votes, list):
                # If it's a list, treat each item with equal weight (1 vote)
                vote_items = [(individual, 1) for individual in votes]
            else:
                continue
            
            # Add votes to aggregated count
            for individual, count in vote_items:
                if individual not in aggregated_votes:
                    aggregated_votes[individual] = 0
                aggregated_votes[individual] += count
        
        if not aggregated_votes:
            room_individuals[room_name] = []
            continue
        
        # Group individuals by their social group
        group_votes = {}
        for individual, count in aggregated_votes.items():
            group_id = individual_to_group.get(individual)
            if group_id is not None:
                if group_id not in group_votes:
                    group_votes[group_id] = []
                group_votes[group_id].append((individual, count))
        
        if not group_votes:
            room_individuals[room_name] = []
            continue
        
        # Calculate total votes per social group
        group_total_votes = {
            group_id: sum(count for _, count in members)
            for group_id, members in group_votes.items()
        }
        
        # Select ONLY the top 1 group with highest votes
        top_group_id = max(group_total_votes.items(), key=lambda x: x[1])[0]
        
        # Get all individuals from the top group, sorted by votes (descending)
        top_group_members = sorted(
            group_votes[top_group_id],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return all individuals from the top group
        selected_individuals = [individual for individual, _ in top_group_members]
        
        room_individuals[room_name] = selected_individuals
    
    return room_individuals

def assign_known_individuals_to_cameras(
    vote_results_dict: Dict[str, pd.Series],
    camera_pairs: Dict[str, list] = None
) -> Dict[str, list]:
    """
    Assign known individuals to each camera based on voting and room assignments.
    
    Args:
        vote_results_dict: Dict mapping camera_id to pd.Series of vote counts
        camera_pairs: Dict mapping room name to list of camera IDs
        
    Returns:
        Dict mapping camera_id to list of known individuals
    """
    from post_processing.utils import CAMERA_TO_ROOM
    
    # Get top individuals per room
    room_individuals = vote_known_individuals_per_camera_pair(
        vote_results_dict, 
        camera_pairs
    )
    
    # Assign individuals to each camera based on room
    camera_individuals = {}
    for camera_id in vote_results_dict.keys():
        room = CAMERA_TO_ROOM.get(camera_id)
        if room and room in room_individuals:
            camera_individuals[camera_id] = room_individuals[room]
        else:
            camera_individuals[camera_id] = []
    
    return camera_individuals


if __name__ == "__main__":
    import pandas as pd
    # ids = load_semi_gt_ids(date='2025-12-01', camera_id='016')
    dates = get_dates_with_semi_gt()
    print(dates)