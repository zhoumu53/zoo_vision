from __future__ import annotations

import json
import re
from pathlib import Path

from .models import ScanConfig, VideoResult


DATE_FOLDER_PATTERN = re.compile(r"^\d{8}(?:AM|PM)?$")


def run_dir_for_video(output_root: Path, data_root: Path, video_path: Path) -> Path:
    """Derive ./runs/{top_level_folder}/{date_folder}/ from a video's absolute path."""
    relative = video_path.relative_to(data_root)
    parts = relative.parts
    if len(parts) >= 2:
        return output_root / parts[0] / parts[1]
    return output_root / (parts[0] if parts else "unknown")


def load_processed_video_paths(run_dir: Path) -> set[str]:
    """Load the set of video_path values already present in a run_dir's report.json."""
    json_path = run_dir / "report.json"
    if not json_path.exists():
        return set()
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    results = payload.get("results")
    if not isinstance(results, list):
        return set()
    return {
        str(row.get("video_path") or "")
        for row in results
        if isinstance(row, dict) and row.get("video_path")
    }


def load_report_results(run_dir: Path) -> list[dict]:
    """Load existing results list from report.json, or empty list."""
    json_path = run_dir / "report.json"
    if not json_path.exists():
        return []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    results = payload.get("results")
    return results if isinstance(results, list) else []


def append_result_to_report(run_dir: Path, result: VideoResult, *, config: ScanConfig | None = None) -> Path:
    """Append a single VideoResult to report.json (creating it if needed)."""
    json_path = run_dir / "report.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    existing_results = load_report_results(run_dir)
    existing_results.append(result.to_dict())

    payload: dict = {"results": existing_results}
    if config is not None:
        payload["scan_config"] = {
            "data_root": str(config.data_root),
            "interval_minutes": config.interval_minutes,
            "confidence_threshold": config.confidence_threshold,
            "weights_path": config.weights_path,
            "model_source": config.model_source,
        }

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return json_path
