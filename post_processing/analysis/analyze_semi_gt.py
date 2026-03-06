import numpy as np

import pandas as pd
import os
import glob

from post_processing.utils import ID2NAMES, CAMERA_TO_ROOM, SEMI_GT_ID_CSV, load_semi_gt_ids
from pathlib import Path





def get_semi_gts_on_tracks(processed_dir = '/media/ElephantsWD/elephants/xmas/tracks/',
                           sandbox_gt_path=Path(SEMI_GT_ID_CSV),
                           output = '/media/mu/zoo_vision/data/semi_gts/semi_gt_analysis.csv'):
    

    if os.path.exists(output):
        df = pd.read_csv(output)
        return df

    cameras = os.listdir(processed_dir)
    dates = os.listdir(os.path.join(processed_dir, cameras[0]))

    data = {
        'camera': [],
        'date': [],
        'n_tracks': [],
        'IDs': []
    }

    for cam in cameras:
        for date in dates:
            if 'csv' in date:
                continue
            next_day = pd.to_datetime(date) + pd.Timedelta(days=1)
            next_day_str = next_day.strftime('%Y-%m-%d')
            paths = [os.path.join(processed_dir, cam, d) for d in [date, next_day_str]]
            track_csvs = []
            for path in paths:
                track_csvs.extend(glob.glob(os.path.join(path, '*.csv')))


            n_tracks = len(track_csvs)
            
            cam_id = cam.split('_')[-1]
            gts = load_semi_gt_ids(sandbox_gt_path=sandbox_gt_path,
                date=date, 
                camera_id=cam_id)

            data['camera'].append(cam)
            data['date'].append(date)
            data['n_tracks'].append(n_tracks)
            data['IDs'].append(gts)


    df = pd.DataFrame(data)

    df.to_csv(output, index=False)

    return df


def _parse_ids(x):
    """
    Robustly parse IDs column that may be:
    - NaN
    - string: "['Thai']"
    - list: ['Thai']
    - numpy array
    """
    # Case 1: true missing
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # Case 2: already list / tuple / np array
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(v) for v in x if pd.notna(v)]

    # Case 3: string representation
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(vv) for vv in v]
            return [str(v)]
        except Exception:
            # fallback: treat as single label
            return [s]

    # Fallback (should not happen)
    return [str(x)]

def build_paired_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: camera, date, n_tracks, IDs
    Output columns:
      date, IDs, camera1, n_tracks1, camera2, n_tracks2
    Where camera2 is the paired camera of camera1 (PAIR mapping).
    """
    df = df.copy()

    # normalize / parse
    df["date"] = df["date"].astype(str).str.slice(0, 10)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise").dt.date.astype(str)
    # df["date"] = pd.to_datetime(df["date"], errors="raise").dt.date.astype(str)
    df["IDs_list"] = df["IDs"].apply(_parse_ids)
    df["IDs"] = df["IDs_list"].apply(lambda xs: ",".join(xs) if xs else "")

    # ensure numeric
    df["n_tracks"] = pd.to_numeric(df["n_tracks"], errors="coerce").fillna(0).astype(int)

    # add partner camera and a canonical pair key so each pair appears once
    df["camera_partner"] = df["camera"].map(PAIR)
    df["pair_key"] = df.apply(
        lambda r: "|".join(sorted([r["camera"], r["camera_partner"]])) if pd.notna(r["camera_partner"]) else r["camera"],
        axis=1,
    )

    # keep only cameras in PAIR
    df = df[df["camera"].isin(PAIR.keys())].copy()

    # aggregate per (date, IDs, pair_key): pick values for each camera in the pair
    def agg_pair(g: pd.DataFrame) -> pd.Series:
        cams = sorted(g["camera"].unique())
        # canonical order: the smaller string is camera1
        cam1 = cams[0]
        cam2 = PAIR[cam1]

        n1 = int(g.loc[g["camera"] == cam1, "n_tracks"].sum()) if (g["camera"] == cam1).any() else 0
        n2 = int(g.loc[g["camera"] == cam2, "n_tracks"].sum()) if (g["camera"] == cam2).any() else 0

        return pd.Series(
            {
                "camera1": cam1,
                "n_tracks1": n1,
                "camera2": cam2,
                "n_tracks2": n2,
            }
        )

    out = (
        df.groupby(["date", "IDs", "pair_key"], as_index=False)
          .apply(agg_pair)
          .reset_index(drop=True)
    )

    out = out[["date", "IDs", "camera1", "n_tracks1", "camera2", "n_tracks2"]].sort_values(
        ["date", "IDs", "camera1"]
    )

    return out


def get_valid_dates_with_semi_gt(df):

    # df = IDs is not empty and n_tracks > 0
    df = df[df['IDs'].map(lambda x: len(eval(x)) > 0 and x != '[]')]

    # group by date
    df_valid = df[df['n_tracks'] > 10]

    # then count the number of cameras per date
    date_counts = df_valid.groupby('date').size().reset_index(name='n_cameras')
    # keep only dates with 4 cameras
    valid_dates = date_counts[date_counts['n_cameras'] == 4]['date'].tolist()


    ### count the unique IDs, per date, across all cameras
    unique_ids = df['IDs'].unique()
    id_counts = {}
    for ids in unique_ids:

        unique_dates = df_valid[df_valid['IDs'] == ids]['date'].unique().tolist()
        # sort unique dates
        unique_dates.sort()
        for date in unique_dates:

            # count cameras for this id on this date
            n_cameras = df_valid[(df_valid['date'] == date) & (df_valid['IDs'] == ids)].shape[0]
            print(f"Date: {date}, ID: {ids}, n_cameras: {n_cameras}")

        # print("ids", ids)

    ### get new df columns: date, IDs, camera1, n_tracks1, camera2, n_tracks2
    ### note: cam 016-019, 017-018 one group


    for data in df_valid.itertuples():
        date = data.date
        print(f"Date: {date}, Camera: {data.camera}, n_tracks: {data.n_tracks}, IDs: {data.IDs}")
    
    print(f"Found {len(valid_dates)} valid dates with semi GTs and more than 10 tracks per camera.")

    print("Valid dates:", valid_dates)
    return df_valid


PAIR = {
    'zag_elp_cam_016': 'zag_elp_cam_019',
    'zag_elp_cam_017': 'zag_elp_cam_018',
    'zag_elp_cam_018': 'zag_elp_cam_017',
    'zag_elp_cam_019': 'zag_elp_cam_016',
}



def main():
    
    df = get_semi_gts_on_tracks()
    print("df", df)
    df_to_process = get_valid_dates_with_semi_gt(df)
    print("df_to_process", df_to_process)
    df = df[df['IDs'].map(lambda x: x != '[]')]
    summary = build_paired_summary(df)
    summary.to_csv("/media/mu/zoo_vision/data/semi_gts/paired_summary_test.csv", index=False)
    print(summary)

if __name__ == "__main__":
    main()

            

