from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from post_processing.utils import IDENTITY_NAMES
# ---------------------------
# Label normalization
# ---------------------------

PRED_LABEL_MAP = {
    # keep your native labels as-is
    "01_standing": "01_standing",
    "02_sleeping_left": "02_sleeping_left",
    "03_sleeping_right": "03_sleeping_right",
    "00_invalid": "00_invalid",
}

GT_SLEEP_MAP = {
    "sleep_left": "02_sleeping_left",
    "sleep_right": "03_sleeping_right",
}

GT_DEFAULT = "01_standing"


ID_INVALID = "Invalid"
BEH_INVALID = "00_invalid"
BEH_LABELS = ["01_standing", "02_sleeping_left", "03_sleeping_right"]


def normalize_pred_behavior(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    # if already standardized, keep; otherwise map if needed
    return s.map(PRED_LABEL_MAP).fillna(s)


def build_track_to_gt_id_map(df_gt_id: pd.DataFrame) -> dict[str, str]:
    """
    df_gt_id columns: filename (no .csv), gt (Indi/Chandra/...)
    Returns map: 'Txxxxxx_IDyyyyyy.csv' -> 'Indi'
    """
    # check if 'csv' suffix is present in 'filename' column; if not, add it
    if not df_gt_id["filename"].astype(str).str.endswith(".csv").all():
        df_gt_id["filename"] = df_gt_id["filename"].astype(str) + ".csv"
    m = dict(zip(df_gt_id["filename"].astype(str), df_gt_id["gt"].astype(str)))
    return m


# ---------------------------
# GT construction
# ---------------------------

def _to_dt64(series: pd.Series) -> pd.Series:
    return series if pd.api.types.is_datetime64_any_dtype(series) else pd.to_datetime(series, errors="coerce")


def _to_time(series: pd.Series) -> pd.Series:
    # parse "23:05:27" -> datetime.time
    return pd.to_datetime(series.astype(str), format="%H:%M:%S", errors="coerce").dt.time


def build_gt_behavior_for_results(
    df_results: pd.DataFrame,
    df_gt_id: pd.DataFrame,
    df_gt_behavior: pd.DataFrame,
    track_col: str = "track_filename",
    ts_col: str = "timestamp",
    pred_beh_col: str = "behavior_label",
) -> pd.DataFrame:
    """
    Adds columns:
      - gt_identity
      - pred_behavior
      - gt_behavior

    GT rule:
      df_gt_behavior only annotates sleeping intervals; everything else = standing.
      Interval membership uses time-of-day, with wrap-around support for midnight crossing:
        if start<=end: start <= t <= end
        else: t >= start OR t <= end
    """
    d = df_results.copy()
    d[ts_col] = _to_dt64(d[ts_col])
    d["tod"] = d[ts_col].dt.time

    # GT identity from track filename
    track2id = build_track_to_gt_id_map(df_gt_id)
    d["gt_identity"] = d[track_col].astype(str).map(track2id)

    # normalize predictions
    d["pred_behavior"] = normalize_pred_behavior(d[pred_beh_col])

    # prepare GT behavior table
    gb = df_gt_behavior.copy()
    gb["id"] = gb["id"].astype(str)
    gb["gt_norm"] = gb["gt"].astype(str).map(GT_SLEEP_MAP)
    gb["start_t"] = _to_time(gb["start_timestamp"])
    gb["end_t"] = _to_time(gb["end_timestamp"])

    # default GT = standing
    d["gt_behavior"] = GT_DEFAULT

    # assign sleeping GT intervals per id (vectorized per-id, per-interval)
    # Note: loops over intervals are typically small (few rows), so this is fast even for big df_results.
    for _, r in gb.iterrows():
        gid = r["id"]
        gt_label = r["gt_norm"]
        st, et = r["start_t"], r["end_t"]
        if pd.isna(st) or pd.isna(et) or not isinstance(gt_label, str):
            continue

        mask_id = d["gt_identity"].eq(gid)
        if not mask_id.any():
            continue

        if st <= et:
            mask_time = (d["tod"] >= st) & (d["tod"] <= et)
        else:
            # crosses midnight
            mask_time = (d["tod"] >= st) | (d["tod"] <= et)

        d.loc[mask_id & mask_time, "gt_behavior"] = gt_label
    
    
    # print("d columns:", d.columns.tolist())
    columns_to_keep = [track_col, 
                       ts_col, 
                       "camera_id",
                       "gt_identity", 
                       "identity_label",
                       "gt_behavior",
                       "pred_behavior", 
                       "behavior_conf"]
    
    if 'quality_label' in d.columns:
        columns_to_keep.extend([
                       "quality_label",
                       "quality_conf"])
    d = d[columns_to_keep]
    
    # filter the items with NaN gt_identity
    d = d.dropna(subset=["gt_identity"])
    
    ### remove the rows where pred_behavior is invalid -- '00_invalid'
    d = d[d["pred_behavior"] != "00_invalid"]
    
    ### remove the rows where gt_identits is not in IDENTITY_NAMES
    d = d[d["gt_identity"].isin(IDENTITY_NAMES)]

    return d




def behavior_metrics_per_id(
    df_eval: pd.DataFrame,
    id_col: str = "gt_identity",          # evaluate "per ID" using ground-truth identity
    gt_beh_col: str = "gt_behavior",
    pred_beh_col: str = "pred_behavior",
    pred_id_col: str = "identity_label",
    gt_id_col: str = "gt_identity",
    wrong_id_as: str = "__WRONG_ID__",    # sentinel label so wrong-ID rows count as behavior-wrong
    labels: list | None = None,           # optional fixed label order (recommended)
    verbose: bool = True,
):
    df = df_eval.copy()

    # Only count behavior as correct if identity AND behavior are correct:
    # If identity is wrong, force the predicted behavior to a sentinel wrong class.
    df["_pred_beh_joint"] = np.where(
        df[pred_id_col].astype(str) == df[gt_id_col].astype(str),
        df[pred_beh_col].astype(str),
        wrong_id_as,
    )

    results = {}

    for _id, g in df.groupby(id_col):
        gts = g[gt_beh_col].astype(str).tolist()
        preds = g["_pred_beh_joint"].astype(str).tolist()

        if labels is None:
            lab = sorted(set(gts) | set(preds))
        else:
            lab = labels

        acc = accuracy_score(gts, preds)
        cm = confusion_matrix(gts, preds, labels=lab)
        
        report = classification_report(gts, preds, labels=lab, zero_division=0)

        results[_id] = {
            "n": len(g),
            "accuracy": acc,
            "labels": lab,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        if verbose:
            print(f"\n===== ID: {_id} (n={len(g)} frames) =====")
            print(f"Joint Accuracy (ID & Behavior): {acc:.4f}")
            print("Labels:", lab)
            print("\nConfusion Matrix (normalized):")
            ## print cm with 3 decimal places
            cm = np.array2string(cm, formatter={'float_kind':lambda x: "%.3f" % x})
            print(cm)
            # print("\nClassification Report:")
            # print(report)

    return results



######


def _to_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

def estimate_sampling_interval_seconds(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    fps: float = 25.0,
    log_every_n_frames: int = 5,
    group_for_estimation: list[str] | None = None,
) -> float:
    fallback = float(log_every_n_frames) / float(fps)  # e.g. 5/25 = 0.2s

    d = df.copy()
    d["_ts"] = _to_datetime_series(d[time_col])
    d = d.dropna(subset=["_ts"])
    if len(d) < 2:
        return fallback

    if group_for_estimation:
        diffs = []
        for _, g in d.sort_values("_ts").groupby(group_for_estimation):
            if len(g) < 2:
                continue
            dt = g["_ts"].diff().dt.total_seconds().to_numpy()
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if len(dt):
                diffs.append(dt)
        if not diffs:
            return fallback
        dt_all = np.concatenate(diffs)
    else:
        d = d.sort_values("_ts")
        dt_all = d["_ts"].diff().dt.total_seconds().to_numpy()
        dt_all = dt_all[np.isfinite(dt_all) & (dt_all > 0)]

    if len(dt_all) == 0:
        return fallback

    med = float(np.median(dt_all))
    if not (0.25 * fallback <= med <= 10.0 * fallback):
        return fallback
    return med

def add_joint_fields(
    df_eval: pd.DataFrame,
    gt_id_col: str = "gt_identity",
    pred_id_col: str = "identity_label",
    gt_beh_col: str = "gt_behavior",
    pred_beh_col: str = "pred_behavior",
) -> pd.DataFrame:
    df = df_eval.copy()
    df["id_correct"] = df[pred_id_col].astype(str) == df[gt_id_col].astype(str)
    df["beh_correct"] = df[pred_beh_col].astype(str) == df[gt_beh_col].astype(str)
    df["joint_correct"] = df["id_correct"] & df["beh_correct"]
    return df

def build_behavior_bouts(
    df: pd.DataFrame,
    id_col: str,
    behavior_col: str,
    time_col: str = "timestamp",
    dt_seconds: float = 0.2,
    max_gap_seconds: float | None = None,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if max_gap_seconds is None:
        max_gap_seconds = 2.5 * dt_seconds

    d = df.copy()
    d["_ts"] = _to_datetime_series(d[time_col])
    d = d.dropna(subset=["_ts"]).sort_values("_ts")

    group_cols = [id_col] + (extra_group_cols or [])
    bout_rows = []

    for _, g in d.groupby(group_cols, dropna=False):
        g = g.sort_values("_ts").copy()
        beh = g[behavior_col].astype(str)
        gap = g["_ts"].diff().dt.total_seconds().fillna(0.0)

        new_bout = (beh != beh.shift(1)) | (gap > max_gap_seconds)
        bout_id = new_bout.cumsum()

        for _, gb in g.groupby(bout_id):
            n = len(gb)
            bout_rows.append({
                "id": gb[id_col].iloc[0],
                "behavior": gb[behavior_col].astype(str).iloc[0],
                "n_samples": n,
                "duration_sec": n * dt_seconds,
            })

    return pd.DataFrame(bout_rows)

def summarize_hours_per_id_behavior(bouts: pd.DataFrame) -> pd.DataFrame:
    if bouts.empty:
        return pd.DataFrame(columns=["id", "behavior", "hours"])

    s = (bouts.groupby(["id", "behavior"], as_index=False)
         .agg(total_sec=("duration_sec", "sum")))
    s["hours"] = s["total_sec"] / 3600.0
    return s.drop(columns=["total_sec"])

def summarize_hours_per_id(bouts: pd.DataFrame) -> pd.DataFrame:
    if bouts.empty:
        return pd.DataFrame(columns=["id", "wrong_id_hours"])

    s = (bouts.groupby(["id"], as_index=False)
         .agg(total_sec=("duration_sec", "sum")))
    s["wrong_id_hours"] = s["total_sec"] / 3600.0
    return s.drop(columns=["total_sec"])

def compute_gt_joint_wrongid_hours(
    df_eval: pd.DataFrame,
    time_col: str = "timestamp",
    gt_id_col: str = "gt_identity",
    gt_beh_col: str = "gt_behavior",
    pred_id_col: str = "identity_label",
    pred_beh_col: str = "pred_behavior",
    fps: float = 25.0,
    log_every_n_frames: int = 5,
    extra_group_cols: list[str] | None = None,   # e.g. ["camera_id"] if you want separation in dt estimation/bouting
) -> dict:
    """
    Outputs ONLY:
      1) gt_hours_per_id_behavior
      2) both_correct_h_per_id_behavior
      3) wrong_id_hours_per_id
    All durations are in HOURS.
    """
    df = add_joint_fields(
        df_eval,
        gt_id_col=gt_id_col,
        pred_id_col=pred_id_col,
        gt_beh_col=gt_beh_col,
        pred_beh_col=pred_beh_col,
    )

    dt = estimate_sampling_interval_seconds(
        df,
        time_col=time_col,
        fps=fps,
        log_every_n_frames=log_every_n_frames,
        group_for_estimation=extra_group_cols,
    )

    # (1) GT hours per (ID, behavior)
    gt_bouts = build_behavior_bouts(
        df=df,
        id_col=gt_id_col,
        behavior_col=gt_beh_col,
        time_col=time_col,
        dt_seconds=dt,
        extra_group_cols=extra_group_cols,
    )
    gt_hours = summarize_hours_per_id_behavior(gt_bouts).rename(
        columns={"id": "gt_identity", "behavior": "gt_behavior", "hours": "gt_hours"}
    )

    # (2) Joint-correct hours per (ID, behavior)
    df_joint = df[df["joint_correct"]].copy()
    joint_bouts = build_behavior_bouts(
        df=df_joint,
        id_col=gt_id_col,
        behavior_col=gt_beh_col,
        time_col=time_col,
        dt_seconds=dt,
        extra_group_cols=extra_group_cols,
    )
    joint_hours = summarize_hours_per_id_behavior(joint_bouts).rename(
        columns={"id": "gt_identity", "behavior": "gt_behavior", "hours": "both_correct_h"}
    )

    # (3) Wrong-ID hours per ID (irrespective of behavior)
    df_wrong = df[~df["id_correct"]].copy()
    if len(df_wrong) > 0:
        df_wrong["_wrong_id_flag"] = "wrong_id"
    wrong_bouts = build_behavior_bouts(
        df=df_wrong,
        id_col=gt_id_col,
        behavior_col="_wrong_id_flag",   # constant, so bouts split only by time gaps
        time_col=time_col,
        dt_seconds=dt,
        extra_group_cols=extra_group_cols,
    )
    wrong_hours = summarize_hours_per_id(wrong_bouts).rename(columns={"id": "gt_identity"})

    return {
        "dt_seconds": dt,
        "gt_hours_per_id_behavior": gt_hours.sort_values(["gt_identity", "gt_behavior"]).reset_index(drop=True),
        "both_correct_h_per_id_behavior": joint_hours.sort_values(["gt_identity", "gt_behavior"]).reset_index(drop=True),
        "wrong_id_hours_per_id": wrong_hours.sort_values(["gt_identity"]).reset_index(drop=True),
    }
    
    
def merge_behavior_hours_tables(
    gt_hours_per_id_behavior: pd.DataFrame,
    both_correct_h_per_id_behavior: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge per-(ID, behavior) GT hours with per-(ID, behavior) joint-correct hours.

    Output columns:
      gt_identity, gt_behavior, gt_hours, both_correct_h
    """
    merged = gt_hours_per_id_behavior.merge(
        both_correct_h_per_id_behavior,
        on=["gt_identity", "gt_behavior"],
        how="left",
    )
    if "both_correct_h" in merged.columns:
        merged["both_correct_h"] = merged["both_correct_h"].fillna(0.0)
    else:
        merged["both_correct_h"] = 0.0

    # Ensure missing gt_hours are 0 (should not happen if GT table is the base)
    merged["gt_hours"] = merged["gt_hours"].fillna(0.0)

    return merged.sort_values(["gt_identity", "gt_behavior"]).reset_index(drop=True)


def build_overall_hours_per_id(
    merged_per_id_behavior: pd.DataFrame,
    wrong_id_hours_per_id: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build overall per-ID summary:
      - total GT hours (sum over behaviors)
      - total joint-correct hours (sum over behaviors)
      - wrong_id_hours (from wrong_id_hours_per_id)
      - n_gt_behaviors (number of behaviors present in GT for that ID)
      - joint_time_recall = joint_correct / gt
    """
    overall = (
        merged_per_id_behavior
        .groupby("gt_identity", as_index=False)
        .agg(
            n_gt_behaviors=("gt_behavior", "nunique"),
            gt_hours=("gt_hours", "sum"),
            both_correct_h=("both_correct_h", "sum"),
        )
    )

    overall = overall.merge(
        wrong_id_hours_per_id[["gt_identity", "wrong_id_hours"]],
        on="gt_identity",
        how="left",
    )
    overall["wrong_id_hours"] = overall["wrong_id_hours"].fillna(0.0)

    overall["joint_time_recall"] = np.where(
        overall["gt_hours"] > 0,
        overall["both_correct_h"] / overall["gt_hours"],
        np.nan,
    )

    return overall.sort_values("gt_identity").reset_index(drop=True)



def postprocess_time_outputs(out: dict) -> dict:
    """
    Given output from compute_gt_joint_wrongid_hours(...),
    returns:
      - per_id_behavior_merged
      - overall
    """
    per_id_behavior_merged = merge_behavior_hours_tables(
        out["gt_hours_per_id_behavior"],
        out["both_correct_h_per_id_behavior"],
    )
    overall = build_overall_hours_per_id(
        per_id_behavior_merged,
        out["wrong_id_hours_per_id"],
    )
    return {
        "per_id_behavior_merged": per_id_behavior_merged,
        "overall": overall,
    }
    
def compute_wrong_id_hours_not_behavior(
    df_eval: pd.DataFrame,
    excluded_gt_behaviors: set[str] | list[str] = ("01_standing",),
    time_col: str = "timestamp",
    gt_id_col: str = "gt_identity",
    gt_beh_col: str = "gt_behavior",
    pred_id_col: str = "identity_label",
    fps: float = 25.0,
    log_every_n_frames: int = 5,
    extra_group_cols: list[str] | None = None,
    dt_seconds: float | None = None,  # if you already computed dt, pass it in
) -> pd.DataFrame:
    """
    Per GT ID: wrong-id hours but ONLY on rows whose GT behavior is NOT in excluded_gt_behaviors.
    Returns columns: gt_identity, wrong_id_hours_not_excluded
    """
    excl = set(map(str, excluded_gt_behaviors))

    df = df_eval.copy()
    df["_ts"] = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    df = df.dropna(subset=["_ts"])

    if dt_seconds is None:
        dt_seconds = estimate_sampling_interval_seconds(
            df,
            time_col=time_col,
            fps=fps,
            log_every_n_frames=log_every_n_frames,
            group_for_estimation=extra_group_cols,
        )

    # wrong ID AND GT behavior not excluded
    mask = (df[pred_id_col].astype(str) != df[gt_id_col].astype(str)) & (~df[gt_beh_col].astype(str).isin(excl))
    d = df.loc[mask, [gt_id_col, time_col, gt_beh_col] + (extra_group_cols or [])].copy()
    if d.empty:
        return pd.DataFrame(columns=["gt_identity", "wrong_id_hours_not_excluded"])

    # Build bouts over time (constant behavior label so only time gaps split)
    d["_wrong_id_flag"] = "wrong_id_not_excluded"

    wrong_bouts = build_behavior_bouts(
        df=d,
        id_col=gt_id_col,
        behavior_col="_wrong_id_flag",
        time_col=time_col,
        dt_seconds=dt_seconds,
        extra_group_cols=extra_group_cols,
    )

    s = (wrong_bouts.groupby(["id"], as_index=False)
         .agg(total_sec=("duration_sec", "sum")))
    s["wrong_id_hours_not_excluded"] = s["total_sec"] / 3600.0
    s = s.drop(columns=["total_sec"]).rename(columns={"id": "gt_identity"})

    return s.sort_values("gt_identity").reset_index(drop=True)

def add_wrong_id_not_standing_to_outputs(
    df_eval: pd.DataFrame,
    out: dict,
    standing_label: str = "01_standing",
) -> dict:
    """
    Adds a new table and also merges it into `overall`:
      - wrong_id_hours_not_standing (per ID)
    """
    wrong_not_standing = compute_wrong_id_hours_not_behavior(
        df_eval=df_eval,
        excluded_gt_behaviors={standing_label},
        dt_seconds=out.get("dt_seconds"),
        extra_group_cols=None,  # or match what you used previously (e.g. ["camera_id"])
    )

    return {
        **out,
        "wrong_id_hours_not_standing_per_id": wrong_not_standing,
    }



def append_wrong_id_not_standing_rows(
    per_id_behavior_merged: pd.DataFrame,
    wrong_id_hours_not_standing_per_id: pd.DataFrame,
    id_col: str = "gt_identity",
    behavior_col: str = "gt_behavior",
    gt_hours_col: str = "gt_hours",
    joint_hours_col: str = "both_correct_h",
    wrong_col_name: str = "wrong_id_hours_not_standing_per_id",
    wrong_behavior_label: str = "__WRONG_ID_NOT_STANDING__",
) -> pd.DataFrame:
    """
    Appends ONE extra row per ID to per_id_behavior_merged containing the wrong-id-not-standing hours.

    Result:
      - original rows unchanged
      - plus rows where gt_behavior == wrong_behavior_label
      - column `wrong_id_hours_not_standing_per_id` filled for those rows
      - gt_hours / both_correct_h set to NaN on those appended rows (as requested)
    """
    base = per_id_behavior_merged.copy()

    w = wrong_id_hours_not_standing_per_id.copy()
    # Accept either naming:
    #   gt_identity + wrong_id_hours_not_standing
    # or gt_identity + wrong_id_hours_not_excluded
    if "wrong_id_hours_not_standing" in w.columns:
        val_col = "wrong_id_hours_not_standing"
    elif "wrong_id_hours_not_excluded" in w.columns:
        val_col = "wrong_id_hours_not_excluded"
    elif wrong_col_name in w.columns:
        val_col = wrong_col_name
    else:
        raise ValueError(
            f"Cannot find wrong-id hours column in wrong_id_hours_not_standing_per_id. "
            f"Got columns: {list(w.columns)}"
        )

    appended = pd.DataFrame({
        id_col: w[id_col].astype(str),
        behavior_col: wrong_behavior_label,
        gt_hours_col: np.nan,
        joint_hours_col: np.nan,
        wrong_col_name: w[val_col].astype(float),
    })

    # Ensure the column exists on base, empty where not applicable
    if wrong_col_name not in base.columns:
        base[wrong_col_name] = np.nan

    out = pd.concat([base, appended], ignore_index=True, axis=0)

    return out.sort_values([id_col, behavior_col]).reset_index(drop=True)



def wrong_id_hours_not_standing_per_id(
    df_eval: pd.DataFrame,
    standing_label: str = "01_standing",
    time_col: str = "timestamp",
    gt_id_col: str = "gt_identity",
    gt_beh_col: str = "gt_behavior",
    pred_id_col: str = "identity_label",
    fps: float = 25.0,
    log_every_n_frames: int = 5,
    extra_group_cols: list[str] | None = None,
    dt_seconds: float | None = None,  # pass precomputed dt if you have it
) -> pd.DataFrame:
    """
    Count hours where (GT behavior != standing) AND (predicted ID is wrong), per GT ID.
    Output columns: gt_identity, wrong_id_hours_not_standing
    """
    d = df_eval.copy()
    d["_ts"] = pd.to_datetime(d[time_col], errors="coerce", utc=False)
    d = d.dropna(subset=["_ts"])

    if dt_seconds is None:
        dt_seconds = estimate_sampling_interval_seconds(
            d,
            time_col=time_col,
            fps=fps,
            log_every_n_frames=log_every_n_frames,
            group_for_estimation=extra_group_cols,
        )

    mask_wrong_id = d[pred_id_col].astype(str) != d[gt_id_col].astype(str)
    mask_not_standing = d[gt_beh_col].astype(str) != str(standing_label)
    d = d.loc[mask_wrong_id & mask_not_standing].copy()

    if d.empty:
        return pd.DataFrame(columns=[gt_id_col, "wrong_id_hours_not_standing"]).rename(
            columns={gt_id_col: "gt_identity"}
        )

    # Make a constant label so bouts only split by time gaps
    d["_flag"] = "wrong_id_not_standing"

    bouts = build_behavior_bouts(
        df=d,
        id_col=gt_id_col,
        behavior_col="_flag",
        time_col=time_col,
        dt_seconds=dt_seconds,
        extra_group_cols=extra_group_cols,
    )

    s = (bouts.groupby("id", as_index=False)
         .agg(total_sec=("duration_sec", "sum")))
    s["wrong_id_hours_not_standing"] = s["total_sec"] / 3600.0
    s = s.drop(columns=["total_sec"]).rename(columns={"id": "gt_identity"})

    return s.sort_values("gt_identity").reset_index(drop=True)


def compute_predicted_hours_per_id_behavior(
    df_eval: pd.DataFrame,
    time_col: str = "timestamp",
    pred_id_col: str = "identity_label",
    pred_beh_col: str = "pred_behavior",
    fps: float = 25.0,
    log_every_n_frames: int = 5,
    extra_group_cols: list[str] | None = None,
    dt_seconds: float | None = None,
) -> pd.DataFrame:
    """
    Returns:
      gt_identity, gt_behavior, predicted_hours
    (column names aligned for merging with per_id_behavior_merged)
    """
    d = df_eval.copy()
    d["_ts"] = pd.to_datetime(d[time_col], errors="coerce", utc=False)
    d = d.dropna(subset=["_ts"])

    if dt_seconds is None:
        dt_seconds = estimate_sampling_interval_seconds(
            d,
            time_col=time_col,
            fps=fps,
            log_every_n_frames=log_every_n_frames,
            group_for_estimation=extra_group_cols,
        )

    # Build bouts on predicted (ID, behavior)
    d["_pred_id"] = d[pred_id_col].astype(str)
    d["_pred_beh"] = d[pred_beh_col].astype(str)

    bouts = build_behavior_bouts(
        df=d,
        id_col="_pred_id",
        behavior_col="_pred_beh",
        time_col=time_col,
        dt_seconds=dt_seconds,
        extra_group_cols=extra_group_cols,
    )

    s = (
        bouts.groupby(["id", "behavior"], as_index=False)
        .agg(total_sec=("duration_sec", "sum"))
    )
    s["predicted_hours"] = s["total_sec"] / 3600.0

    return (
        s.drop(columns=["total_sec"])
         .rename(columns={
             "id": "gt_identity",
             "behavior": "gt_behavior",
         })
         .sort_values(["gt_identity", "gt_behavior"])
         .reset_index(drop=True)
    )
    
    
def add_predicted_hours_to_per_id_behavior(
    per_id_behavior_merged: pd.DataFrame,
    predicted_hours_per_id_behavior: pd.DataFrame,
) -> pd.DataFrame:
    out = per_id_behavior_merged.merge(
        predicted_hours_per_id_behavior,
        on=["gt_identity", "gt_behavior"],
        how="left",
    )

    out["predicted_hours"] = out["predicted_hours"].fillna(0.0)
    return out


def wrong_id_hours_per_behavior(
    df_eval: pd.DataFrame,
    time_col: str = "timestamp",
    gt_id_col: str = "gt_identity",
    gt_beh_col: str = "gt_behavior",
    pred_id_col: str = "identity_label",
    pred_beh_col: str = "pred_behavior",
    fps: float = 25.0,
    log_every_n_frames: int = 5,
    extra_group_cols: list[str] | None = None,
    dt_seconds: float | None = None,      # pass out["dt_seconds"] to keep consistent
    behavior_source: str = "gt",          # "gt" or "pred"
    per_id: bool = True,                 # True: per GT ID; False: overall only
) -> pd.DataFrame:
    """
    Wrong-ID prediction hours, broken down by behavior.

    Wrong-ID rows: identity_label != gt_identity

    behavior_source:
      - "gt": group wrong-ID hours by GT behavior (what was actually happening)
      - "pred": group wrong-ID hours by predicted behavior (what the model said)

    Returns columns:
      if per_id:
        gt_identity, behavior, wrong_id_hours
      else:
        behavior, wrong_id_hours
    """
    d = df_eval.copy()
    d["_ts"] = pd.to_datetime(d[time_col], errors="coerce", utc=False)
    d = d.dropna(subset=["_ts"])

    if dt_seconds is None:
        dt_seconds = estimate_sampling_interval_seconds(
            d,
            time_col=time_col,
            fps=fps,
            log_every_n_frames=log_every_n_frames,
            group_for_estimation=extra_group_cols,
        )

    # wrong ID mask
    mask_wrong = d[pred_id_col].astype(str) != d[gt_id_col].astype(str)
    d = d.loc[mask_wrong].copy()
    if d.empty:
        cols = ["behavior", "wrong_id_hours"] if not per_id else ["gt_identity", "behavior", "wrong_id_hours"]
        return pd.DataFrame(columns=cols)

    # choose which behavior label to attribute wrong-ID time to
    if behavior_source == "gt":
        d["_beh"] = d[gt_beh_col].astype(str)
    elif behavior_source == "pred":
        d["_beh"] = d[pred_beh_col].astype(str)
    else:
        raise ValueError('behavior_source must be "gt" or "pred"')

    if per_id:
        id_for_group = gt_id_col  # attribute wrong-ID time to the GT individual
    else:
        d["_all"] = "ALL"
        id_for_group = "_all"

    # build bouts per (id, behavior)
    bouts = build_behavior_bouts(
        df=d,
        id_col=id_for_group,
        behavior_col="_beh",
        time_col=time_col,
        dt_seconds=dt_seconds,
        extra_group_cols=extra_group_cols,
    )

    # summarize hours
    s = (bouts.groupby(["id", "behavior"], as_index=False)
         .agg(total_sec=("duration_sec", "sum")))
    s["wrong_id_hours"] = s["total_sec"] / 3600.0
    s = s.drop(columns=["total_sec"])

    if per_id:
        s = s.rename(columns={"id": "gt_identity"})
        # change column 'behavior' to gt_behavior for clarity
        s = s.rename(columns={"behavior": "gt_behavior"})
        return s.sort_values(["gt_identity", "gt_behavior"]).reset_index(drop=True)
    else:
        s = s.drop(columns=["id"]).rename(columns={"behavior": "gt_behavior"})
        return s.sort_values(["gt_behavior"]).reset_index(drop=True)



def correct_id_wrong_behavior_hours_per_id_behavior(
    df_eval: pd.DataFrame,
    time_col: str = "timestamp",
    gt_id_col: str = "gt_identity",
    gt_beh_col: str = "gt_behavior",
    pred_id_col: str = "identity_label",
    pred_beh_col: str = "pred_behavior",
    fps: float = 25.0,
    log_every_n_frames: int = 5,
    extra_group_cols: list[str] | None = None,
    dt_seconds: float | None = None,
) -> pd.DataFrame:
    """
    Hours where:
      - predicted ID == GT ID
      - predicted behavior != GT behavior

    Aggregated per (GT ID, GT behavior).

    Returns:
      gt_identity, gt_behavior, correct_id_wrong_behavior
    """
    d = df_eval.copy()
    d["_ts"] = pd.to_datetime(d[time_col], errors="coerce", utc=False)
    d = d.dropna(subset=["_ts"])

    if dt_seconds is None:
        dt_seconds = estimate_sampling_interval_seconds(
            d,
            time_col=time_col,
            fps=fps,
            log_every_n_frames=log_every_n_frames,
            group_for_estimation=extra_group_cols,
        )

    # mask: ID correct, behavior wrong (and behavior is sleeping - L or R)
    mask = (
        (d[pred_id_col].astype(str) == d[gt_id_col].astype(str)) &
        (d[pred_beh_col].astype(str) != d[gt_beh_col].astype(str)) &
        (d[gt_beh_col].astype(str).str.startswith("02_sleeping_left") | d[gt_beh_col].astype(str).str.startswith("03_sleeping_right"))
    )
    d = d.loc[mask].copy()

    if d.empty:
        return pd.DataFrame(
            columns=["gt_identity", "gt_behavior", "correct_id_wrong_behavior"]
        )

    # Build bouts on GT (ID, behavior)
    bouts = build_behavior_bouts(
        df=d,
        id_col=gt_id_col,
        behavior_col=gt_beh_col,
        time_col=time_col,
        dt_seconds=dt_seconds,
        extra_group_cols=extra_group_cols,
    )

    s = (
        bouts.groupby(["id", "behavior"], as_index=False)
        .agg(total_sec=("duration_sec", "sum"))
    )
    s["cor_id_wrong_beh"] = s["total_sec"] / 3600.0

    return (
        s.drop(columns=["total_sec"])
         .rename(columns={"id": "gt_identity", "behavior": "gt_behavior"})
         .sort_values(["gt_identity", "gt_behavior"])
         .reset_index(drop=True)
    )
    
def add_correct_id_wrong_behavior_to_per_id_behavior(
    per_id_behavior_merged: pd.DataFrame,
    correct_id_wrong_behavior: pd.DataFrame,
) -> pd.DataFrame:
    out = per_id_behavior_merged.merge(
        correct_id_wrong_behavior,
        on=["gt_identity", "gt_behavior"],
        how="left",
    )
    out["cor_id_wrong_beh"] = out["cor_id_wrong_beh"].fillna(0.0)
    return out