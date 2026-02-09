# load json file - night-date
# load csv files (tracks, behaviors)


from __future__ import annotations
from pathlib import Path
from datetime import datetime
import enlighten
import pandas as pd
import psycopg2


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Patch

mpl.rcParams["agg.path.chunksize"] = 20000 
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 0.5


@dataclass
class PlotConfig:
    gap_minutes: float = 5.0
    dpi: int = 180
    marker_size: float = 2.0
    line_width: float = 1.0
    alpha_points: float = 0.9
    alpha_lines: float = 0.6
    heatmap_gridsize: int = 140  # hexbin resolution

### for visualization + bg image
IMG_H = 600
IMG_W = 1060
    
def ensure_datetime_sorted(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col)
    return out


def add_bbox_pixels_and_center(
    df: pd.DataFrame,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
    *,
    top_col: str = "bbox_top2",
    left_col: str = "bbox_left2",
    bottom_col: str = "bbox_bottom2",
    right_col: str = "bbox_right2",
) -> pd.DataFrame:
    """
    Converts normalized bbox coords (0..1) to pixels and adds bbox center (cx, cy).
    """
    out = df.copy()

    out["bbox_top"] = out[top_col] * img_h
    out["bbox_left"] = out[left_col] * img_w
    out["bbox_bottom"] = out[bottom_col] * img_h
    out["bbox_right"] = out[right_col] * img_w

    out["cx"] = (out["bbox_left"] + out["bbox_right"]) / 2.0
    out["cy"] = (out["bbox_top"] + out["bbox_bottom"]) / 2.0

    # clip to image bounds (optional but keeps plots sane)
    out["cx"] = out["cx"].clip(0, img_w - 1)
    out["cy"] = out["cy"].clip(0, img_h - 1)

    return out


def normalize_behaviour_label(s: str) -> str:
    if s is None or str(s).strip().lower() == "00_invalid":
        return "invalid"
    return str(s).strip().lower()


def behaviour_cmap_map() -> Dict[str, str]:
    return {
        "01_standing": "winter",
        "02_sleeping_left": "magma",
        "03_sleeping_right": "plasma",
        "invalid": "Greys",
    }


# ----------------------------
#    Heatmap of "staying same location" (density) with different colors per behaviour
#    Using bbox center (cx, cy)
# ----------------------------

def plot_bbox_center_heatmap_by_behaviour(
    df_individual: pd.DataFrame,
    out_path: Path,
    *,
    time_col: str = "timestamp",
    beh_col: str = "behavior_label",
    cx_col: str = "cx",
    cy_col: str = "cy",
    cfg: PlotConfig = PlotConfig(),
    title: Optional[str] = None,
    invert_y: bool = True,
    cam_id: Optional[str] = None,
) -> Path:
    d = ensure_datetime_sorted(df_individual, time_col=time_col)
    d = add_bbox_pixels_and_center(d, IMG_H, IMG_W)

    d = d.dropna(subset=[cx_col, cy_col])
    if d.empty:
        raise ValueError("No valid bbox centers to plot.")

    d[beh_col] = d[beh_col].map(normalize_behaviour_label)
    # check if only one behaviour present, if so, use 'rainbow'
    behaviors_to_show = d[beh_col].unique()
    cmap_for = behaviour_cmap_map() if len(behaviors_to_show) > 1 else {behaviors_to_show[0]: "rainbow"}

    fig = plt.figure(figsize=(10, 7), dpi=cfg.dpi, facecolor="black")
    ax = fig.add_subplot(111, facecolor="black")

    hexbin_collections = []
    for beh, g in d.groupby(beh_col):
        if g.empty:
            continue
        hb = ax.hexbin(
            g[cx_col].to_numpy(),
            g[cy_col].to_numpy(),
            gridsize=cfg.heatmap_gridsize,
            bins="log",
            cmap=cmap_for.get(beh, "Greys"),
            mincnt=1,
            alpha=0.85,
        )
        hexbin_collections.append(hb)

    ax.set_title(title or "Heatmap (bbox center) colored by behaviour", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_xlabel("", color="white")
    ax.set_ylabel("", color="white")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(0, IMG_H)
    if invert_y:
        ax.invert_yaxis()
    ax.grid(False)
    
    ## remove the axis ticks and labels for a cleaner look (optional)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # colorbar for density
    if hexbin_collections:
        cbar = fig.colorbar(hexbin_collections[-1], ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label('Density (log scale)', color='white', rotation=270, labelpad=20)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    
    if cam_id is not None:
        cam_img_path = f'/media/mu/zoo_vision/data/cameras/zag_elp_cam_{cam_id}/sample.jpg'
        # read and overlay cam image as background (with some transparency)
        if Path(cam_img_path).exists():
            img = plt.imread(cam_img_path)
            ### the img is up-side down, so flip it vertically
            img = np.flipud(img)
            ax.imshow(img, extent=[0, IMG_W, 0, IMG_H], alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    

    return out_path


def plot_time_ordered_trajectory_segmented(
    df_individual: pd.DataFrame,
    out_path: Path,
    *,
    time_col: str = "timestamp",
    cx_col: str = "cx",
    cy_col: str = "cy",
    cfg: PlotConfig = PlotConfig(),
    title: Optional[str] = None,
    invert_y: bool = True,
    cam_id: Optional[str] = None,
    break_if_gap_s: float = 10.0,   # if time gap > this, consider it a break
    break_if_jump_px: float = 200.0,   # if spatial jump > this, consider it a break
    draw_density: bool = True,
    density_alpha: float = 0.30,
    point_size: float = 10.0,
    line_width: float = 2.0,
) -> Path:
    d = ensure_datetime_sorted(df_individual, time_col=time_col)
    d = add_bbox_pixels_and_center(d, IMG_H, IMG_W)

    d = d.dropna(subset=[cx_col, cy_col, time_col])
    if d.empty:
        raise ValueError("No valid bbox centers to plot.")

    t = pd.to_datetime(d[time_col])
    dt = (t - t.min()).dt.total_seconds().to_numpy()
    dt_range = max(dt.max(), 1e-9)
    t_norm = dt / dt_range  # [0,1] for coloring

    x = d[cx_col].to_numpy()
    y = d[cy_col].to_numpy()

    # break mask: True at indices where a new segment starts
    breaks = np.zeros(len(x), dtype=bool)
    if len(x) >= 2:
        gaps = np.diff(dt)
        jumps = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        breaks[1:] = (gaps > break_if_gap_s) | (jumps > break_if_jump_px)

    fig = plt.figure(figsize=(10, 7), dpi=cfg.dpi, facecolor="black")
    ax = fig.add_subplot(111, facecolor="black")

    if cam_id is not None:
        cam_id_str = str(cam_id).zfill(3)
        cam_img_path = f"/media/mu/zoo_vision/data/cameras/zag_elp_cam_{cam_id_str}/sample.jpg"
        if Path(cam_img_path).exists():
            img = plt.imread(cam_img_path)
            img = np.flipud(img)
            ax.imshow(img, extent=[0, IMG_W, 0, IMG_H], alpha=0.30)

    if draw_density:
        ax.hexbin(
            x, y,
            gridsize=cfg.heatmap_gridsize,
            bins="log",
            cmap="Greys",
            mincnt=1,
            alpha=density_alpha,
        )

    # draw segmented lines + time-colored points
    sc = None
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or breaks[i]:
            xs = x[start:i]
            ys = y[start:i]
            ts = t_norm[start:i]
            if len(xs) >= 2:
                ax.plot(xs, ys, linewidth=line_width, alpha=0.9)  # no cross-gap lines
            sc = ax.scatter(xs, ys, c=ts, s=point_size, cmap="turbo",
                            alpha=0.95, linewidths=0)
            start = i

    ax.set_title(title or "Segmented time-ordered trajectory (bbox center)", color="white")
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(0, IMG_H)
    if invert_y:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("white")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Time (normalized)", color="white", rotation=270, labelpad=20)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.outline.set_edgecolor("white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path



def analyze_and_save_individual_plots_bbox_center(
    df_individual: pd.DataFrame,
    *,
    individual_name: str = None,
    cam_id: Optional[str] = None,
    heat_path: Path = None,
    traj_heat_path: Path = None,
    cfg: PlotConfig = PlotConfig(),
) -> Tuple[Path, Path]:
    
    
    plot_bbox_center_heatmap_by_behaviour(
        df_individual,
        heat_path,
        cfg=cfg,
        cam_id=cam_id,
        title=heat_path.name.replace(".png", "")
    )
    
    plot_time_ordered_trajectory_segmented(
        df_individual,
        traj_heat_path,
        cfg=cfg,
        cam_id=cam_id,
        title=traj_heat_path.name.replace(".png", "")
    )



INDIVIDUALS_TO_ID = {
    "Chandra": 1,
    "Farha": 3,
    "Indi": 2,
    "Panang": 4,
    "Thai": 5,
    "Invalid": 0,
}
BEHAVIOURS_TO_ID = {
    "00_invalid": 0,
    "01_standing": 1,
    "02_sleeping_left": 2,
    "03_sleeping_right": 3,
}

CAMERA_TO_ID = {
    "zag_elp_cam_016": 0,
    "zag_elp_cam_017": 1,
    "zag_elp_cam_018": 2,
    "zag_elp_cam_019": 3,
}
TYPO = {
    'Fahra': 'Farha',
}


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="", add_help=add_help
    )

    parser.add_argument(
        "--record_root", default='/media/ElephantsWD/elephants/test_dan/results' , type=Path, help="Path to tracks data", required=False
    )
    parser.add_argument(
        "--dates", type=str, nargs="+", help="Specific dates to process", default=None
    )
    parser.add_argument(
        "--start_timestamp", default="180000", type=str, help="Start timestamp for processing, e.g., '180000'", required=False
    )
    parser.add_argument(
        "--end_timestamp", default="080000", type=str, help="End timestamp for processing, e.g., '080000'", required=False
    )
    parser.set_defaults(delete_existing_day=False)
    return parser



def merge_csv_tracklets(record_root: Path, 
                        start_datetime: pd.Timestamp ,  ## date time
                        end_datetime: pd.Timestamp,
                        id_col: str = 'stitched_label',
                        camera_ids: list[str] = None) -> pd.DataFrame:

    from post_processing.tools.utils import (load_valid_tracks, 
                                             load_identity_labels_from_json)
    
    df_behavior = load_valid_tracks(
        record_root=record_root,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        camera_ids=camera_ids,
        behavior_csv_suffix='_behavior_smoothed.csv'
    )
    
    df_label_predictions = load_identity_labels_from_json(
        record_root=Path(record_root),
        camera_ids=camera_ids,
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    # merge identity labels into df_behavior, on track_filename
    df_results = df_behavior.merge(
        df_label_predictions[['track_filename', id_col]],
        on='track_filename',
        how='left'
    )
    
    return df_results



def normalize_time_string(time_str: str) -> str:
    """Convert various time formats to HHMMSS format.
    
    Handles: 18, 1800, 180000, 18:00:00 -> all convert to 180000
    """
    # Remove colons if present
    time_str = time_str.replace(':', '')
    
    # Pad to 6 digits (HHMMSS)
    if len(time_str) == 1:
        time_str = time_str.zfill(2) + '0000'  # "18" -> "180000"
    elif len(time_str) == 2:
        time_str = time_str + '0000'  # "18" -> "180000"
    elif len(time_str) == 4:
        time_str = time_str + '00'  # "1800" -> "180000"
    elif len(time_str) == 6:
        pass  # Already correct format
    else:
        raise ValueError(f"Invalid time format: {time_str}")
    
    return time_str

def list_track_files_for_night(camera_dir: Path, 
                               date: str) -> list[Path]:
    """List all track files for a given night (date + time range)"""
    track_files = []
    date_str = date
    # add '-' to date string if not present
    if '-' not in date_str:
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    date_dir = camera_dir / date_str
    next_date = (pd.to_datetime(date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    next_date_dir = camera_dir / next_date
    # List files from date_dir
    
    if date_dir.exists():
        track_files.extend(list(date_dir.glob("*.csv")))
    # List files from next_date_dir
    if next_date_dir.exists():
        track_files.extend(list(next_date_dir.glob("*.csv")))
        
    return track_files


def main(args):
    record_root = args.record_root
    dates = args.dates
    if dates is None:
        dates = ['2026-02-04', '2026-02-05', '2026-02-06', '2026-02-07', '2026-02-08']
    
    # Normalize and convert timestamp strings to pd.Timestamp
    start_timestamp = normalize_time_string(args.start_timestamp)
    end_timestamp = normalize_time_string(args.end_timestamp)
    
    start_timestamp = pd.to_datetime(start_timestamp, format="%H%M%S")
    end_timestamp = pd.to_datetime(end_timestamp, format="%H%M%S")

    cam_pairs = [
        ['016', '019'],
        ['018', '017']
    ]
    
    for date in dates:
        next_date = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for cam_pair in cam_pairs:
            print(f"Processing date {date} for cameras {cam_pair} with time range {start_timestamp.time()} to {end_timestamp.time()}")
            df_results = merge_csv_tracklets(
                        record_root= args.record_root,
                        start_datetime= pd.Timestamp(date + " " + start_timestamp.strftime("%H:%M:%S")),
                        end_datetime= pd.Timestamp(next_date + " " + end_timestamp.strftime("%H:%M:%S")),
                        id_col= 'stitched_label',
                        camera_ids= cam_pair,
                    )
            ## test
            # df_results = pd.read_csv('/media/mu/zoo_vision/post_processing/analysis/trajs/2026-02-04/csvs/camera_016_019_label_Thai.csv')
            
            cfg = PlotConfig(gap_minutes=5.0)
            out_dir = Path('/media/mu/zoo_vision/post_processing/analysis/trajs') / date
            out_dir.mkdir(parents=True, exist_ok=True)

            #### Per-individual
            for label, group in df_results.groupby('stitched_label'):
                if str(label).strip().lower() == 'invalid':
                    continue

                out_csv_dir = out_dir / "csvs"
                out_csv_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_csv_dir / f"camera_{cam_pair[0]}_{cam_pair[1]}_label_{label}.csv"
                group.to_csv(out_file, index=False)
                
                #### Per-behavior
                
                for beh, group_beh in group.groupby('behavior_label'):
                    if str(beh).strip().lower() in ('00_invalid', 'unknown', 'invalid'):
                        continue
                    
                    #### Per-camera
                    for cam_id, group_beh_cam in group_beh.groupby('camera_id'):
                        
                        out_img_dir = out_dir / label
                        out_img_dir.mkdir(parents=True, exist_ok=True)
                        
                        traj_heat_path = out_img_dir / f"camera_{cam_id}_label_{label}_beh_{beh}_traj.png"
                        heat_path = out_img_dir / f"camera_{cam_id}_label_{label}_beh_{beh}_heat.png"
                        
                        analyze_and_save_individual_plots_bbox_center(
                            group_beh_cam,
                            cam_id=cam_id,
                            traj_heat_path= traj_heat_path,
                            heat_path= heat_path,
                            individual_name=str(label),
                            cfg=cfg,
                        )
                
                        print(f"saved: {traj_heat_path}, \n {heat_path}")
            # print("df_results", df_results.head())


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
