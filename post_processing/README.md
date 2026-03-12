# Post-processing pipeline

This folder contains the offline post-processing pipeline after detection + tracking:

1. Start from per-track clips/CSVs produced by C++/YOLO+ByteTrack.
2. Run per-track inference:
   - behavior classification
   - ReID feature extraction + per-track identity voting
3. Stitch tracklets within each camera (same animal across fragmented tracks).
4. (Optional) match identities across camera pairs.
5. Run downstream analysis (sleeping/stereotypy/activity budgets).

## End-to-end data flow

```text
tracking JSONL (+ raw video)
  -> tools/extract_tracklets.py
     -> tracks/*.csv + tracks/*.mkv
  -> scripts/go_live/run_extraction.sh (live) OR scripts/offline/offline_extraction*.sh (offline)
     -> *_behavior.csv + *.npz per track
  -> scripts/go_live/run_post_processing_db_update.sh (live) OR scripts/offline/offline_post_processing_db_update.sh (offline)
     -> demo/zag_elp_cam_XXX/YYYY-MM-DD/stitched_tracklets_camXXX_*.json
     -> demo/night_bout_summary/YYYYMMDD/*.csv (cross-camera on)
  -> analysis/*.py
     -> reports/figures/csv summaries
```

## Expected data layout

`record_root` is the main input/output base used by most scripts.

```text
<record_root>/
  tracks/
    zag_elp_cam_016/
      2025-11-29/
        T180001_ID000001.csv
        T180001_ID000001.mkv
        T180001_ID000001_behavior.csv
        T180001_ID000001.npz
        ...
    zag_elp_cam_017/
    zag_elp_cam_018/
    zag_elp_cam_019/
  demo/
    zag_elp_cam_016/2025-11-29/stitched_tracklets_cam016_*.json
    ...
```

Track directory resolution is implemented in `core/file_manager.py` (`tracks/zag_elp_cam_<cam_id>/<YYYY-MM-DD>`).

## File formats

### 1) Track CSV (`Txxxxxx_IDxxxxxx.csv`)
From C++/YOLO pipeline:

Columns:
- `frame_id`
- `timestamp`
- `bbox_top`, `bbox_left`, `bbox_bottom`, `bbox_right`
- `score`

### 2) Behavior CSV (`*_behavior.csv`)
From `tools/extract_features_single_cam.py`.

Columns:
- `timestamp`
- `behavior_label_raw`
- `behavior_label` (smoothed)
- `behavior_conf`
- `quality_label`, `quality_conf`

### 3) ReID features NPZ (`*.npz`)
Produced by `tools/run_reid_feature_extraction.py`.

Typical keys:
- `features`
- `frame_ids`
- `avg_embedding`
- `matched_labels`
- `avg_matched_labels`
- `voted_labels`
- optional `metadata`

### 4) Stitched tracklet JSON (`stitched_tracklets_cam*.json`)
Produced by `core/tracklet_manager.py::save_stitched_tracklets`.

Typical fields per item:
- `track_id`, `raw_track_id`, `track_filename`
- `track_csv_path`, `track_video_path`
- `camera_id`
- `stitched_id`
- `stitched_label`, `voted_track_label`, `smoothed_label`, `identity_label`
- `invalid_flag`
- `start_timestamp`, `end_timestamp`
- `feature_path`

## Main entrypoints

Use shell wrappers under `scripts/go_live` and `scripts/offline` as the main entry points.

### A) Live videos (`scripts/go_live`)

Feature extraction:
```bash
bash post_processing/scripts/go_live/run_extraction.sh
```

Optional camera subset:
```bash
bash post_processing/scripts/go_live/run_extraction.sh 016 019
```

Post-processing + ethogram + DB update:
```bash
bash post_processing/scripts/go_live/run_post_processing_db_update.sh
```

Explicit date:
```bash
bash post_processing/scripts/go_live/run_post_processing_db_update.sh 20251129
```

### B) Offline videos (`scripts/offline`)

Offline extraction (date lists are maintained inside the script files):
```bash
bash post_processing/scripts/offline/offline_extraction.sh
```
or
```bash
bash post_processing/scripts/offline/offline_extraction_split3.sh
```

Offline post-processing / DB update:
```bash
bash post_processing/scripts/offline/offline_post_processing_db_update.sh
```

### C) Low-level tools (internal)

These are used by the wrappers and are useful for debugging/custom runs:
- `tools/extract_tracklets.py`
- `tools/extract_features_single_cam.py`
- `tools/offline_extract_features_single_cam.py`
- `tools/run_post_processing_full_night.py`
- `tools/offline_post_processing_full_night.py`

## Code structure

### `core/`
Core logic for stitching/matching/smoothing.
- `tracklet_manager.py`: tracklet loading, invalid-zone filtering, within-camera stitching, JSON export.
- `cross_cam_matching.py`: cross-camera track matching (`run_cross_camera_matching_v2`) + summaries.
- `temporal_smooth.py`: behavior smoothing (`behavior_label_smooth`, cross-camera smoothing utilities).
- `behavior_inference.py`: behavior model wrapper.
- `reid_inference.py`: ReID feature extraction + gallery matching.
- `file_manager.py`: canonical path helpers for tracks/videos.
- `config/`: YAML config + loader.

### `tools/`
CLI-style scripts used in pipeline execution.
- `extract_tracklets.py`: JSONL -> per-track clip+CSV.
- `extract_features_single_cam.py`: per-camera night processing (behavior + ReID).
- `run_feature_extraction_id_behavior.py`: older/all-camera batch variant.
- `run_post_processing_full_night.py`: stitching + optional cross-camera.
- `run_reid_feature_extraction.py`: reusable ReID extraction/voting utilities.

### `analysis/`
Post-hoc analysis and visualization scripts.
- Activity budgets: `activity_budget_analysis.py`, `activity_budget_analysis_per_day.py`
- Sleeping: `sleeping_analysis.py`, `sleeping_duration_analysis.py`, `sleeping_bout_summary.py`
- Stereotypy: `stereotypy_analysis.py`
- Stereotypy classifier model: `analysis/stereotype_classifier/` (`training.py`, `inference.py`)
- Visualization: `video_results_visualization.py`, `trajectory_heatmap.py`

#### `analysis/stereotype_classifier/`
Image classifier for stereotypy labels (ResNet18-based).

Training (`training.py`):
- Loads labels from `gt.csv` and images from `images/`.
- Uses a strict year split: train years vs one test year (`--train_years`, `--test_year`).
- Saves best checkpoint to `model.pt` and metadata to `model.json`.
- Default split in code: train on `2025`, test on `2026`.

Example:
```bash
python post_processing/analysis/stereotype_classifier/training.py \
  --gt_csv /media/mu/zoo_vision/data/stereotype/gt.csv \
  --image_dir /media/mu/zoo_vision/data/stereotype/images \
  --train_years 2025 \
  --test_year 2026 \
  --epochs 20 \
  --checkpoint_path post_processing/analysis/stereotype_classifier/model.pt
```

Inference (`inference.py`):
- Loads `model.pt` checkpoint and class mapping from saved metadata.
- Predicts top-1 label and top-k probabilities for one image.

Example:
```bash
python post_processing/analysis/stereotype_classifier/inference.py \
  --checkpoint_path post_processing/analysis/stereotype_classifier/model.pt \
  --image_path /path/to/image.jpg \
  --topk 3
```


## Configuration

Primary config file:
- `post_processing/core/config/configs.yaml`

Important sections:
- `directories.record_root`, `directories.output_dir`
- `models.reid.*`, `models.behavior.model_path`
- `processing.batch_size`, `processing.device`, overwrite flags
- `time_window.*`
- `stitching.*`

Many scripts also expose CLI overrides for ad-hoc runs.

## Notes and caveats

- Time windows may cross midnight (for example `16:00 -> 08:00` next day).
- Some scripts expect absolute paths in local environment defaults; pass explicit CLI args in shared/repro runs.
- Cross-camera logic is currently designed around room pairs `016<->019` and `017<->018`.
- Behavior smoothing is important because frame-wise predictions can jitter.
