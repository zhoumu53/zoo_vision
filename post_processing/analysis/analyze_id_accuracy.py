from post_processing.analysis.evaluate_behavior import build_gt_behavior_for_results
from post_processing.analysis.analyze_lateral_sleeping import load_gt_ids
import pandas as pd
from pathlib import Path
import numpy as np
import json


def load_id_from_csv(
    csv_path: Path) -> str:
    # print("Loading ID from CSV:", csv_path)
    df = pd.read_csv(csv_path)
    if 'identity_label' in df.columns:
        return df['identity_label'].iloc[0]
    else:
        return None


def compute_id_performance(date : str = '2025-11-15',
                           cam_id: str = '016',
                           start_time: str = '15:30:00',
                            end_time: str = '08:00:00'
    ):
    
    GT_dir = Path('/media/mu/zoo_vision/data/GT_id_behavior/id_GTs')

    track_root = Path('/media/mu/zoo_vision/xmas')
    stitched_dir = Path(track_root / 'demo')
    tracks_dir = Path(stitched_dir / 'tracks')

    ## EVALUATION
    df_gt_id = load_gt_ids(
        dir=GT_dir,
        camera_ids=[cam_id],
        date=date
    )
    df_gt_id['filename'] = df_gt_id['filename'].apply(lambda x: x.replace('.csv', ''))
    # print("Loaded GT IDs:", df_gt_id.head())
    
    # raise error if df_gt_id is empty
    if df_gt_id.empty:
        raise ValueError(f"No GT IDs found for date {date} and cam_id {cam_id}")
    
    ### load stitched results & voted results
    json_path = stitched_dir / f'zag_elp_cam_{cam_id}' / date / f'*{start_time.replace(":", "")}*{end_time.replace(":", "")}*.json'
    # if not exist, raise error
    try:
        json_path = list(json_path.parent.glob(json_path.name))[0]
    except IndexError:
        json_path = Path(str(json_path).replace('demo', 'demo-'))
        json_path = list(json_path.parent.glob(json_path.name))[0]
    print("Loading stitched results from JSON:", json_path)
    # read json
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"No data found in JSON file: {json_path}")
        
    data_items = [i for _d in data.values() for i in _d ]
    
    columns = ['track_filename', 'track_csv_path', 'identity_label', 'voted_track_label']
    df_stitched = pd.DataFrame(data_items, columns=columns)
    # rename 'identity_label' to 'stitched_id'
    df_stitched = df_stitched.rename(columns={'identity_label': 'stitched_id'})
    

    ### load smoothed results (from csvs)
    df_stitched['smoothed_id'] = df_stitched['track_csv_path'].apply(
        lambda x: load_id_from_csv(Path(x))
    )
    
    if start_time == '16:30:00' and end_time == '07:30:00':
        df_stitched['smoothed_id'] = df_stitched['track_csv_path'].apply(
            lambda x: load_id_from_csv(Path(str(x).replace('.csv', '_id_behavior.csv')))
        )
    
    # print("Stitched DataFrame head:\n", df_stitched.head())
    
    # merge with GT
    df_eval = pd.merge(
        df_stitched,
        df_gt_id,
        how='inner',
        left_on='track_filename',
        right_on='filename'
    )
    
    if df_eval.empty:
        raise ValueError(f"No matching tracks found between stitched results and GT for date {date} and cam_id {cam_id}")
    # print("Evaluation DataFrame head:\n", df_eval.head())
    # remove the items with NaN gt
    df_eval = df_eval[~df_eval['gt'].isna()]

    ### fix typo for gt, voted_track_label, stitched_id, smoothed_id
    from post_processing.utils import TYPO
    for col in ['gt', 'voted_track_label', 'stitched_id', 'smoothed_id']:
        df_eval[col] = df_eval[col].replace(TYPO)
    
    # compute accuracy -- sklearn accuracy_score
    from sklearn.metrics import accuracy_score
    acc_voted = accuracy_score(df_eval['gt'], df_eval['voted_track_label'])
    acc_stitched = accuracy_score(df_eval['gt'], df_eval['stitched_id'])
    acc_smoothed = accuracy_score(df_eval['gt'], df_eval['smoothed_id'])
    # acc_2stage_smoothed = accuracy_score(df_eval['gt'], df_eval['2stage_smoothed_id'])
    
    columns = [
        'track_filename',
        'gt',
        'voted_track_label',
        'stitched_id',
        'smoothed_id',
    ]
    df_eval = df_eval[columns]
    output_csv_dir = Path('/media/mu/zoo_vision/post_processing/analysis/csvs')
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_csv_dir = output_csv_dir / f'{cam_id}_{date}_id_evaluation_{start_time.replace(":", "")}_{end_time.replace(":", "")}.csv'
    df_eval.to_csv(
        output_csv_dir,
        index=False
    )
    print(f"Saved evaluation results to CSV for date {date}, cam_id {cam_id}, time {start_time}-{end_time}")
    
    # print(f"Accuracy (step1: reid + voting): {acc_voted}")
    # print(f"Accuracy (step2: stitched     ): {acc_stitched}")
    # print(f"Accuracy (step3: smoothed     ): {acc_smoothed}")
    # print(f"Accuracy (2-stage smoothed): {acc_2stage_smoothed}")
    
    return {
        'acc_voted': acc_voted,
        'acc_stitched': acc_stitched,
        'acc_smoothed': acc_smoothed,
    }
    
    
if __name__ == "__main__":
    
    data_with_gts = {
        
        # '2025-11-15': ['016', '019'],
        '2025-11-15': ['017', '018'],
        # '2025-11-30': ['017', '018'],
    }
    
    new_time_split = ['15:30:00', '08:00:00']
    old_time_split = ['16:30:00', '07:30:00']   
    
    splits = [old_time_split, new_time_split]
    
    
    
    results_dict = { date : { cam_id: {} for cam_id in cam_ids} for date, cam_ids in data_with_gts.items()}
        
    
    
    for date, cam_ids in data_with_gts.items():
        
        
        for cam_id in cam_ids:
            print("==================================================")
            print(f"Computing ID performance for date: {date}, cam_id: {cam_id}")
            # print("Evaluating with time splits:")
                        
            for time_split in splits:
                dtype = "Old" if time_split == old_time_split else "New"
                # print(f"Starting [{dtype}] model")
                
                res = compute_id_performance(date=date, cam_id=cam_id, start_time=time_split[0], end_time=time_split[1])
                # print("--------------------------------------------------")
                results_dict[date][cam_id][dtype] = res


    print("\n" + "=" * 80 + "\n")

    # print final results
    print("\n\nFinal Results Summary:")
    for date, cam_ids in data_with_gts.items():
        for cam_id in cam_ids:
            print("==================================================")
            print(f"Date: {date}, Cam ID: {cam_id}")
            for dtype in ["Old", "New"]:
                res = results_dict[date][cam_id][dtype]
                print(f"Results for [{dtype}] model:")
                print(f"  Accuracy (voted): {res['acc_voted']}")
                print(f"  Accuracy (stitched): {res['acc_stitched']}")
                print(f"  Accuracy (smoothed): {res['acc_smoothed']}")
            print("--------------------------------------------------")
            
    # compute average accuracy across all cameras per date
    print("\n\nAverage Accuracy Summary:")
    for date, cam_ids in data_with_gts.items():
        print("==================================================")
        print(f"Date: {date}")
        for dtype in ["Old", "New"]:
            acc_voted_list = []
            acc_stitched_list = []
            acc_smoothed_list = []
            for cam_id in cam_ids:
                res = results_dict[date][cam_id][dtype]
                acc_voted_list.append(res['acc_voted'])
                acc_stitched_list.append(res['acc_stitched'])
                acc_smoothed_list.append(res['acc_smoothed'])
            avg_acc_voted = np.mean(acc_voted_list)
            avg_acc_stitched = np.mean(acc_stitched_list)
            avg_acc_smoothed = np.mean(acc_smoothed_list)
            print(f"Average Results for [{dtype}] model:")
            print(f"  Average Accuracy (voted): {avg_acc_voted}")
            print(f"  Average Accuracy (stitched): {avg_acc_stitched}")
            print(f"  Average Accuracy (smoothed): {avg_acc_smoothed}")
        print("--------------------------------------------------")
            
    