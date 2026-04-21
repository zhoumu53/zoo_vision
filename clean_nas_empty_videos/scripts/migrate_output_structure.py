#!/usr/bin/env python3
"""One-time migration: reorganize existing output files into the new structure.

Before:
  runs/{camera}/{date}/report.json
  runs/{camera}/{date}/{video}.jpg          (all previews, mixed)
  EMPTY_EXPORT_ROOT/{camera}/{date_key}.csv (separate location)

After:
  runs/{camera}/{date}/report.json          (unchanged)
  runs/{camera}/{date}/{date}.csv           (moved from EMPTY_EXPORT_ROOT)
  runs/{camera}/{date}/to_delete_preview/   (only to_delete JPGs)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
EMPTY_EXPORT_ROOT = Path("/media/ElephantsWD/empty_videos_to_be_deleted")

DRY_RUN = "--dry-run" in sys.argv


def log(msg: str) -> None:
    print(f"  {'[DRY RUN] ' if DRY_RUN else ''}{msg}")


def load_to_delete_stems(report_path: Path) -> set[str]:
    """Read report.json and return stems of videos whose LATEST result is to_delete=True.

    report.json may contain duplicate entries from multiple scans (e.g. rescan).
    The last entry for each video wins.
    """
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    # build a map of video_path -> latest to_delete value
    latest: dict[str, bool] = {}
    for result in data.get("results", []):
        video_path = result.get("video_path", "")
        to_delete = result.get("to_delete")
        if video_path and to_delete is not None:
            latest[video_path] = to_delete
    return {Path(vp).stem for vp, td in latest.items() if td is True}


def migrate_jpgs(date_dir: Path, to_delete_stems: set[str]) -> None:
    """Move to_delete JPGs into to_delete_preview/, remove the rest."""
    jpgs = list(date_dir.glob("*.jpg"))
    if not jpgs:
        return

    preview_dir = date_dir / "to_delete_preview"

    for jpg in jpgs:
        if jpg.stem in to_delete_stems:
            dest = preview_dir / jpg.name
            log(f"MOVE {jpg.name} -> to_delete_preview/")
            if not DRY_RUN:
                preview_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(jpg), str(dest))
        else:
            log(f"DELETE {jpg.name} (not to_delete)")
            if not DRY_RUN:
                jpg.unlink()


def migrate_csvs() -> None:
    """Copy CSVs from EMPTY_EXPORT_ROOT/{camera}/{date_key}.csv into matching run dirs."""
    if not EMPTY_EXPORT_ROOT.is_dir():
        print(f"EMPTY_EXPORT_ROOT not found: {EMPTY_EXPORT_ROOT}, skipping CSV migration.")
        return

    for csv_path in sorted(EMPTY_EXPORT_ROOT.rglob("*.csv")):
        camera = csv_path.parent.name      # e.g. "ELP-Kamera-06"
        date_key = csv_path.stem            # e.g. "20240903"

        camera_dir = RUNS_DIR / camera
        if not camera_dir.is_dir():
            log(f"SKIP CSV {csv_path.name} — no camera dir {camera}")
            continue

        # Find matching date folder(s): e.g. 20240903AM, 20240903PM, or just 20240903
        matching_dirs = sorted(camera_dir.glob(f"{date_key}*"))
        if not matching_dirs:
            log(f"SKIP CSV {csv_path.name} — no matching date dir for {camera}/{date_key}*")
            continue

        for date_dir in matching_dirs:
            dest = date_dir / f"{date_dir.name}.csv"
            if dest.exists():
                log(f"SKIP CSV -> {dest.relative_to(RUNS_DIR)} (already exists)")
                continue
            log(f"COPY {csv_path} -> {dest.relative_to(RUNS_DIR)}")
            if not DRY_RUN:
                shutil.copy2(str(csv_path), str(dest))


def main() -> None:
    if DRY_RUN:
        print("=== DRY RUN — no files will be changed ===\n")

    if not RUNS_DIR.is_dir():
        print(f"Runs directory not found: {RUNS_DIR}")
        return

    # 1. Migrate JPGs in each date folder
    print("--- Step 1: Reorganize preview JPGs ---")
    moved = 0
    deleted = 0
    for camera_dir in sorted(RUNS_DIR.iterdir()):
        if not camera_dir.is_dir():
            continue
        for date_dir in sorted(camera_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            report = date_dir / "report.json"
            if not report.exists():
                continue
            to_delete_stems = load_to_delete_stems(report)
            jpgs = list(date_dir.glob("*.jpg"))
            if not jpgs:
                continue
            print(f"\n{date_dir.relative_to(RUNS_DIR)}/  ({len(jpgs)} JPGs, {len(to_delete_stems)} to_delete)")
            for jpg in jpgs:
                if jpg.stem in to_delete_stems:
                    moved += 1
                else:
                    deleted += 1
            migrate_jpgs(date_dir, to_delete_stems)

    print(f"\nJPG summary: {moved} moved to to_delete_preview/, {deleted} deleted (not to_delete)")

    # 2. Copy CSVs from EMPTY_EXPORT_ROOT
    print("\n--- Step 2: Copy CSVs from EMPTY_EXPORT_ROOT ---")
    migrate_csvs()

    print("\nDone!")


if __name__ == "__main__":
    main()
