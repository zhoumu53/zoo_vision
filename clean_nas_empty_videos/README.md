# Empty Video Scanner

Scans NAS camera videos and identifies the ones **without elephants** so they can be deleted to free up storage.

- Already-scanned videos are **automatically skipped** on re-run (results are saved per folder).
- Results are saved **immediately** after each video — safe to interrupt and resume at any time.

## Usage

### [SCAN 1] - Direct scan (recommended)

LAZY VERSION - when you want to scan all folders under NAS system:

```bash
cd clean_nas_empty_videos
./scripts/run_direct_scan.sh
```

Provide the folder path directly:

```bash
cd clean_nas_empty_videos
./scripts/run_direct_scan.sh "ELP-Kamera-01/20250603PM"
```

You can also scan a top-level folder (all date subfolders):

```bash
cd clean_nas_empty_videos
./scripts/run_direct_scan.sh "ELP-Kamera-01"
```


### [SCAN 2] - Interactive CLI

For manual folder selection (browse and pick from a menu). Use this if you want to look around before deciding which folder to scan, or if you want to delete videos from another disk:

```bash
cd clean_nas_empty_videos
./scripts/run_cli_historical_scan.sh
```

### [DELETE] - Delete the empty videos

After scanning, run the delete script to remove files listed in the CSVs:

```bash
cd clean_nas_empty_videos
./scripts/DELETE_SCANNED_EMPTY.sh
```

**Note:** The actual `rm` command inside the script is commented out by default. You must edit and uncomment it before it will delete anything.

## Where results are saved

| What | Location |
|------|----------|
| Scan results (per folder/date) | `./runs/{camera_folder}/{date_folder}/report.json` |
| Preview images (empty videos only) | `./runs/{camera_folder}/{date_folder}/{video_name}.jpg` |
| Final CSV for deletion | `/media/ElephantsWD/empty_videos_to_be_deleted/{camera_folder}/{date}.csv` |

- `report.json` contains one entry per video with `to_delete: true/false`.
- The CSV at the export path is what the delete script reads.
- Re-running the scan on the same folder **skips already-processed videos** based on `report.json`.

## How the scanning algorithm works

For each video, the tool takes a series of snapshots (frames) and uses a YOLO AI model to check whether an elephant appears in each frame. The process has three stages:

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

For videos marked as empty, a **preview image** (grid of sampled frames) is saved so you can visually double-check the result before deleting.
