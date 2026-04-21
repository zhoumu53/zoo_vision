# Empty Video Scanner

Scans NAS camera videos and identifies the ones **without elephants** so they can be deleted to free up storage.

- Uses a **custom-trained YOLO model** (`models/segmentation/yolo/all_v3/weights/best.onnx`) for elephant detection.
- Already-scanned videos are **automatically skipped** on re-run (results are saved per folder).
- Results are saved **immediately** after each video — safe to interrupt and resume at any time.

## Usage

### Scan a specific folder

```bash
cd clean_nas_empty_videos
./scripts/run_direct_scan.sh "ZAG-ELP-CAM-019/20260410AM"
```

Scan an entire camera (all date subfolders):

```bash
./scripts/run_direct_scan.sh "ZAG-ELP-CAM-019"
```

Scan everything on the NAS:

```bash
./scripts/run_direct_scan.sh
```

### Re-scan (ignore previous results)

Re-scan a specific folder, ignoring cached results in `report.json`:

```bash
./scripts/run_direct_scan.sh "ZAG-ELP-CAM-019/20260410AM" --rescan
```

Re-scan all cameras (re-processes everything, overwriting previous results):

```bash
./scripts/rescan_all.sh
```

Re-scan only cameras matching a pattern:

```bash
./scripts/rescan_all.sh ZAG-ELP-CAM
```

### Interactive CLI

For manual folder selection (browse and pick from a menu):

```bash
./scripts/run_cli_historical_scan.sh
```

### Delete the empty videos

After scanning, run the delete script to remove files listed in the CSVs:

```bash
./scripts/DELETE_SCANNED_EMPTY.sh
```

**Note:** The actual `rm` command inside the script is commented out by default. Edit and uncomment it before it will delete anything.

## Where results are saved

All output goes under `OUTPUT_ROOT` (default: `/media/ElephantsWD/empty_videos_to_be_deleted`):

```
OUTPUT_ROOT/{camera}/{date}/
├── report.json            # one entry per video with to_delete: true/false
├── {date}.csv             # list of empty videos for the delete script
└── to_delete_preview/     # preview images (only for empty videos)
    ├── {video_name}.jpg
    └── ...
```

- Re-running the scan on the same folder **skips already-processed videos** based on `report.json`. Use `--rescan` to force re-processing.

## How the scanning algorithm works

For each video, the tool takes a series of snapshots (frames) and uses a YOLO model to check whether an elephant appears in each frame. The process has three stages:

### Stage 1 — Quick check (coarse sampling)

The tool takes one snapshot every **2 minutes** throughout the video. For example, a 1-hour video gets about 30 snapshots. Each snapshot is checked for elephants.

If elephants are found in very few snapshots (less than 20%), the tool moves to Stage 2 for a closer look.

### Stage 2 — Closer look (refinement)

The tool takes **additional snapshots between** the ones from Stage 1 (at the midpoints). This doubles the number of checked frames and gives a more accurate picture of how much of the video contains elephants.

### Stage 3 — Confirmation (dense validation)

If any elephant was spotted in Stage 1 or 2, the tool zooms into those time windows and takes snapshots every **30 seconds** around the detections. It then measures the **longest continuous stretch** of time where elephants are confirmed.

### Final decision

A video is marked as **to_delete = true** (no elephants) only when **both** of these are true:

1. Less than 80% of the sampled frames contain an elephant
2. The longest continuous stretch of elephant detections is shorter than 2 minutes

If either condition fails — meaning elephants are seen in most frames, or they appear for a long enough stretch — the video is kept.

For videos marked as empty, a **preview image** (grid of sampled frames) is saved in `to_delete_preview/` so you can visually double-check the result before deleting.
