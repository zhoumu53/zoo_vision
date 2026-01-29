from post_processing.analysis.evaluate_behavior import build_gt_behavior_for_results
from post_processing.analysis.analyze_lateral_sleeping import load_gt_ids
import pandas as pd
from pathlib import Path
import numpy as np
import json
from post_processing.utils import IDENTITY_NAMES, TYPO

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
                            end_time: str = '08:00:00',
                            output_dir: Path = None
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
            
    columns = ['track_filename', 'track_csv_path', 'stitched_label', 'stitched_id', 'voted_track_label', 'start_timestamp', 'end_timestamp']
    df_stitched = pd.DataFrame(data, columns=columns)

    if df_stitched.empty:
        raise ValueError(f"No stitched data found in JSON file: {json_path}")
    

    ### load smoothed results (from csvs)
    df_stitched['smoothed_label'] = df_stitched['track_csv_path'].apply(
        lambda x: load_id_from_csv(Path(x))
    )
    
    if start_time == '16:30:00' and end_time == '07:30:00':
        df_stitched['smoothed_label'] = df_stitched['track_csv_path'].apply(
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

    ### fix typo for gt, voted_track_label, stitched_label, smoothed_label
    for col in ['gt', 'voted_track_label', 'stitched_label', 'smoothed_label']:
        df_eval[col] = df_eval[col].replace(TYPO)
    # remove gt not in IDENTITY_NAMES
    df_eval = df_eval[df_eval['gt'].isin(IDENTITY_NAMES)]

    # print(f"After merging, evaluation DataFrame shape: {df_eval.shape}")
    # print(df_eval)
    # fill rows with None values in prediction columns
    for col in ['voted_track_label', 'stitched_label', 'smoothed_label']:
        df_eval[col] = df_eval[col].fillna('None')

    # add n_length column -- length of the track in number of frames
    def compute_track_length(track_csv_path: str) -> int:
        df_track = pd.read_csv(track_csv_path)
        return len(df_track)
    df_eval['n_length'] = df_eval['track_csv_path'].apply(compute_track_length)

    # compute accuracy -- sklearn accuracy_score
    from sklearn.metrics import accuracy_score
    acc_voted = accuracy_score(df_eval['gt'], df_eval['voted_track_label'])
    acc_stitched = accuracy_score(df_eval['gt'], df_eval['stitched_label'])
    acc_smoothed = accuracy_score(df_eval['gt'], df_eval['smoothed_label'])
    # acc_2stage_smoothed = accuracy_score(df_eval['gt'], df_eval['2stage_smoothed_label'])
    
    columns = [
        'track_filename',
        'gt',
        'voted_track_label',
        'stitched_label',
        'stitched_id',
        'smoothed_label',
        'n_length',
        'start_timestamp', 'end_timestamp'
    ]
    df_eval = df_eval[columns]

         
    # from post_processing.core.temporal_smooth import smooth_identity_labels, reassign_stitched_ids
    # df_eval = smooth_identity_labels(df_eval, max_time_gap='10min')
    # # Reassign IDs to be consistent
    # df_eval = reassign_stitched_ids(df_eval)

    output_csv_path = output_dir / f'{cam_id}_{start_time.replace(":", "")}_{end_time.replace(":", "")}.csv'
    df_eval.to_csv(
        output_csv_path,
        index=False
    )
    print(f"Saved evaluation results to CSV for date {date}, cam_id {cam_id}, time {start_time}-{end_time}")
    
    return {
        'acc_voted': acc_voted,
        'acc_stitched': acc_stitched,
        'acc_smoothed': acc_smoothed,
    }



def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze ID accuracy across different methods"
    )
    parser.add_argument(
        "--date",
        type=str,
        required=False,
        default='2025-11-15',
        help="Date to analyze (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--cam_ids",
        type=str,
        required=False,
        default='016',
        nargs='+',
        help="Camera IDs to analyze (comma-separated if multiple)"
    )
    return parser.parse_args()
    

def main(date: str, cam_ids: list):

    output_dir = Path(f'/media/mu/zoo_vision/post_processing/analysis/csvs/{date}/{cam_ids[0]}_{cam_ids[-1]}')
    output_dir.mkdir(parents=True, exist_ok=True)

    new_time_split = ['15:30:00', '08:00:00']
    old_time_split = ['16:30:00', '07:30:00']   
    
    splits = [old_time_split, new_time_split]
    
    dtype2splits={
        # "Old": old_time_split,
        "New": new_time_split
    }
    
    results_dict = { date : { cam_id: {} for cam_id in cam_ids} }
        
    print(f"Processing date: {date} for cam_ids: {cam_ids}", results_dict[date].keys())
    
    for cam_id in cam_ids:
        print("==================================================")
        print(f"Computing ID performance for date: {date}, cam_id: {cam_id}")
        # print("Evaluating with time splits:")
                    
        for dtype, time_split in dtype2splits.items():
        
            res = compute_id_performance(date=date, cam_id=cam_id, start_time=time_split[0], end_time=time_split[1], output_dir=output_dir)
            # print("--------------------------------------------------")
            results_dict[date][cam_id][dtype] = res


    print("\n" + "=" * 80 + "\n")

    # print final results
    print("\n\nFinal Results Summary:")
    for cam_id in cam_ids:
        print("==================================================")
        print(f"Date: {date}, Cam ID: {cam_id}")
        for dtype in dtype2splits.keys():
            res = results_dict[date][cam_id][dtype]
            print(f"Results for [{dtype}] model:")
            print(f"  Accuracy (voted): {res['acc_voted']}")
            print(f"  Accuracy (stitched): {res['acc_stitched']}")
            print(f"  Accuracy (smoothed): {res['acc_smoothed']}")
        print("--------------------------------------------------")
            
    # compute average accuracy across all cameras per date
    print("\n\nAverage Accuracy Summary:")
    print("==================================================")
    print(f"Date: {date}")
    for dtype in dtype2splits.keys():
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
    

    # save printed results to a text file
    output_txt_dir = output_dir / f'id_accuracy_summary.txt'
    with open(output_txt_dir, 'w') as f:
        f.write("\n\nFinal Results Summary:\n")
        for cam_id in cam_ids:
            f.write("==================================================\n")
            f.write(f"Date: {date}, Cam ID: {cam_id}\n")
            for dtype in dtype2splits.keys():
                res = results_dict[date][cam_id][dtype]
                f.write(f"Results for [{dtype}] model:\n")
                f.write(f"  Accuracy (voted): {res['acc_voted']}\n")
                f.write(f"  Accuracy (stitched): {res['acc_stitched']}\n")
                f.write(f"  Accuracy (smoothed): {res['acc_smoothed']}\n")
            f.write("--------------------------------------------------\n")
                
        f.write("\n\nAverage Accuracy Summary:\n")
        f.write("==================================================\n")
        f.write(f"Date: {date}\n")
        for dtype in dtype2splits.keys():
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
            f.write(f"Average Results for [{dtype}] model:\n")
            f.write(f"  Average Accuracy (voted): {avg_acc_voted}\n")
            f.write(f"  Average Accuracy (stitched): {avg_acc_stitched}\n")
            f.write(f"  Average Accuracy (smoothed): {avg_acc_smoothed}\n")
        f.write("--------------------------------------------------\n")



if __name__ == "__main__":
    args = get_args()

    date = args.date
    cam_ids = args.cam_ids


    gt_dates_cams= {
        '2025-11-15': [['017', '018'], ['016', '019']],
        '2025-11-30': [['017', '018'], ['016', '019']],
        '2025-12-01': [['016', '019']],
        '2025-12-15': [['017', '018']],
    }


    for date, cam_id_groups in gt_dates_cams.items():

        if '2025-12-01' == date and cam_id_groups == [['016', '019']]:
            continue
        if date == '2025-12-15' and cam_id_groups == [['017', '018']]:
            continue

        for cam_ids in cam_id_groups:
            main(date, cam_ids)