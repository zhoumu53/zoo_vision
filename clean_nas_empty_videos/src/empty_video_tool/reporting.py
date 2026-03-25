from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .exporting import (
    EMPTY_VIDEO_EXPORT_FIELDNAMES,
    build_empty_video_export_rows,
    grouped_empty_video_export_path,
    grouped_empty_video_log_path,
    write_grouped_empty_video_exports,
)
from .models import ScanConfig, VideoResult


INDEX_FILENAME = "scan_index.json"
EMPTY_VIDEO_LOG_FILENAME = "empty_video_paths.log"
EMPTY_VIDEO_CSV_FILENAME = "empty_videos.csv"


def create_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = output_root / f"scan-{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"scan-{timestamp}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate

def _index_path(output_root: Path) -> Path:
    return output_root / INDEX_FILENAME


def _load_index_payload(output_root: Path) -> dict:
    index_path = _index_path(output_root)
    if not index_path.exists():
        return {"version": 1, "entries": []}

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "entries": []}

    if not isinstance(payload, dict):
        return {"version": 1, "entries": []}

    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {"version": 1, "entries": []}
    return {"version": 1, "entries": entries}


def _write_index_payload(output_root: Path, payload: dict) -> Path:
    index_path = _index_path(output_root)
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return index_path


def _scan_entry(
    *,
    generated_at: str,
    run_dir: Path,
    report_json_path: Path,
    empty_video_log_path: Path,
    empty_video_csv_path: Path,
    config: ScanConfig,
    result_count: int,
) -> dict:
    return {
        "generated_at": generated_at,
        "run_dir": str(run_dir),
        "report_json_path": str(report_json_path),
        "empty_video_log_path": str(empty_video_log_path),
        "empty_video_csv_path": str(empty_video_csv_path),
        "target_folder": str(config.target_folder),
        "target_folder_relative": config.target_folder.relative_to(config.data_root).as_posix(),
        "recursive": config.recursive,
        "filename_substring": config.filename_substring,
        "interval_minutes": config.interval_minutes,
        "confidence_threshold": config.confidence_threshold,
        "weights_path": config.weights_path,
        "model_source": config.model_source,
        "result_count": result_count,
    }


def _append_scan_index_entry(
    output_root: Path,
    *,
    generated_at: str,
    run_dir: Path,
    report_json_path: Path,
    empty_video_log_path: Path,
    empty_video_csv_path: Path,
    config: ScanConfig,
    result_count: int,
) -> Path:
    payload = _load_index_payload(output_root)
    payload["entries"].append(
        _scan_entry(
            generated_at=generated_at,
            run_dir=run_dir,
            report_json_path=report_json_path,
            empty_video_log_path=empty_video_log_path,
            empty_video_csv_path=empty_video_csv_path,
            config=config,
            result_count=result_count,
        )
    )
    return _write_index_payload(output_root, payload)


def load_saved_scan(report_json_path: Path) -> dict | None:
    try:
        payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("results"), list):
        return None
    return payload


def load_scan_index_entries(output_root: Path) -> list[dict]:
    payload = _load_index_payload(output_root)
    return [
        entry
        for entry in payload["entries"]
        if isinstance(entry, dict)
        and Path(str(entry.get("report_json_path", ""))).exists()
    ]


def folder_has_saved_scan_coverage(
    entries: list[dict],
    *,
    target_folder: Path,
    recursive: bool,
    filename_substring: str | None,
    interval_minutes: int,
    confidence_threshold: float,
    weights_path: str | None,
    include_ancestor_recursive_coverage: bool = True,
) -> bool:
    resolved_target_folder = target_folder.expanduser().resolve()

    for entry in entries:
        if entry.get("interval_minutes") != interval_minutes:
            continue
        if entry.get("confidence_threshold") != confidence_threshold:
            continue
        if entry.get("weights_path") != weights_path:
            continue

        entry_target_raw = entry.get("target_folder")
        if not entry_target_raw:
            continue

        entry_target_folder = Path(str(entry_target_raw)).expanduser().resolve()
        entry_recursive = bool(entry.get("recursive"))

        if entry_target_folder == resolved_target_folder:
            if recursive and not entry_recursive:
                continue
            if entry.get("filename_substring") != filename_substring:
                continue
            return True

        if entry_recursive and include_ancestor_recursive_coverage:
            try:
                resolved_target_folder.relative_to(entry_target_folder)
            except ValueError:
                continue
            return True

    return False


def find_latest_saved_scan(
    output_root: Path,
    *,
    target_folder: Path,
    recursive: bool,
    filename_substring: str | None,
    interval_minutes: int | None = None,
    confidence_threshold: float | None = None,
    weights_path: str | None = None,
    exact_settings: bool = False,
) -> dict | None:
    payload = {"entries": load_scan_index_entries(output_root)}
    target_folder_str = str(target_folder.expanduser().resolve())
    matches = [
        entry
        for entry in payload["entries"]
        if isinstance(entry, dict)
        and entry.get("target_folder") == target_folder_str
        and bool(entry.get("recursive")) == recursive
        and entry.get("filename_substring") == filename_substring
        and (not exact_settings or entry.get("interval_minutes") == interval_minutes)
        and (not exact_settings or entry.get("confidence_threshold") == confidence_threshold)
        and (not exact_settings or entry.get("weights_path") == weights_path)
        and Path(str(entry.get("report_json_path", ""))).exists()
    ]
    if not matches:
        return None
    matches.sort(key=lambda entry: str(entry.get("generated_at", "")), reverse=True)
    return matches[0]


def write_empty_video_log(run_dir: Path, results: list[VideoResult]) -> Path:
    log_path = run_dir / EMPTY_VIDEO_LOG_FILENAME
    empty_paths = [result.host_video_path for result in results if result.status == "ok" and result.empty_video]
    with log_path.open("w", encoding="utf-8") as handle:
        for path in empty_paths:
            handle.write(f"{path}\n")
    return log_path


def empty_video_csv_path(run_dir: Path) -> Path:
    return run_dir / EMPTY_VIDEO_CSV_FILENAME


def empty_video_export_log_path(export_root: Path) -> Path:
    return export_root / EMPTY_VIDEO_LOG_FILENAME


def _existing_csv_values(csv_path: Path, unique_field: str) -> set[str]:
    if not csv_path.exists():
        return set()

    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            return {
                str(row.get(unique_field) or "").strip()
                for row in rows
                if str(row.get(unique_field) or "").strip()
            }
    except OSError:
        return set()


def _append_unique_csv_rows(
    csv_path: Path,
    rows: list[dict],
    *,
    fieldnames: list[str],
    unique_field: str,
) -> None:
    if not rows:
        return

    existing_values = _existing_csv_values(csv_path, unique_field)
    pending_rows = []
    for row in rows:
        unique_value = str(row.get(unique_field) or "").strip()
        if not unique_value or unique_value in existing_values:
            continue
        existing_values.add(unique_value)
        pending_rows.append(row)

    if not pending_rows and csv_path.exists():
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a" if file_exists else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in pending_rows:
            writer.writerow(row)


def _append_unique_log_line(log_path: Path, line: str) -> None:
    value = line.strip()
    if not value:
        return

    existing_values: set[str] = set()
    if log_path.exists():
        try:
            existing_values = {
                current.strip()
                for current in log_path.read_text(encoding="utf-8").splitlines()
                if current.strip()
            }
        except OSError:
            existing_values = set()

    if value in existing_values:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{value}\n")


def _append_export_log_rows(export_root: Path, export_rows: list[dict]) -> Path:
    log_path = empty_video_export_log_path(export_root)
    for row in export_rows:
        host_path = str(row.get("host_path") or "")
        _append_unique_log_line(log_path, host_path)
        _append_unique_log_line(
            grouped_empty_video_log_path(
                export_root,
                row,
                log_filename=EMPTY_VIDEO_LOG_FILENAME,
            ),
            host_path,
        )
    return log_path


def append_empty_video_artifacts(
    run_dir: Path,
    result: VideoResult,
    *,
    config: ScanConfig | None = None,
) -> None:
    if result.status != "ok" or not result.empty_video:
        return

    host_root = config.host_data_root_display if config is not None else None
    export_rows = build_empty_video_export_rows(
        [result.to_report_row()],
        host_root=host_root,
    )
    _append_unique_log_line(run_dir / EMPTY_VIDEO_LOG_FILENAME, result.host_video_path)

    _append_unique_csv_rows(
        empty_video_csv_path(run_dir),
        export_rows,
        fieldnames=EMPTY_VIDEO_EXPORT_FIELDNAMES,
        unique_field="host_path",
    )

    if config is None or config.empty_export_root is None:
        return

    _append_export_log_rows(config.empty_export_root, export_rows)
    for row in export_rows:
        export_path = grouped_empty_video_export_path(config.empty_export_root, row)
        _append_unique_csv_rows(
            export_path,
            [row],
            fieldnames=EMPTY_VIDEO_EXPORT_FIELDNAMES,
            unique_field="host_path",
        )


def write_empty_video_csv(run_dir: Path, results: list[VideoResult], *, host_root: str | None = None) -> Path:
    export_path = empty_video_csv_path(run_dir)
    export_rows = build_empty_video_export_rows(
        [result.to_report_row() for result in results],
        host_root=host_root,
    )
    with export_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EMPTY_VIDEO_EXPORT_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in export_rows:
            writer.writerow(row)
    return export_path


def write_scan_report(
    run_dir: Path,
    results: list[VideoResult],
    *,
    config: ScanConfig | None = None,
) -> tuple[Path, Path]:
    json_path = run_dir / "report.json"
    empty_video_log_path = write_empty_video_log(run_dir, results)
    empty_video_csv_export_path = write_empty_video_csv(
        run_dir,
        results,
        host_root=config.host_data_root_display if config is not None else None,
    )
    if config is not None and config.empty_export_root is not None:
        export_rows = build_empty_video_export_rows(
            [result.to_report_row() for result in results],
            host_root=config.host_data_root_display,
        )
        _append_export_log_rows(config.empty_export_root, export_rows)
        write_grouped_empty_video_exports(
            config.empty_export_root,
            [result.to_report_row() for result in results],
            host_root=config.host_data_root_display,
        )
    generated_at = datetime.now().isoformat(timespec="seconds")

    payload = {
        "run_dir": str(run_dir),
        "generated_at": generated_at,
        "empty_video_csv_path": str(empty_video_csv_export_path),
        "scan_config": None,
        "results": [result.to_dict() for result in results],
    }
    if config is not None:
        payload["scan_config"] = {
            "data_root": str(config.data_root),
            "target_folder": str(config.target_folder),
            "target_folder_relative": config.target_folder.relative_to(config.data_root).as_posix(),
            "empty_export_root": str(config.empty_export_root) if config.empty_export_root is not None else None,
            "recursive": config.recursive,
            "filename_substring": config.filename_substring,
            "interval_minutes": config.interval_minutes,
            "confidence_threshold": config.confidence_threshold,
            "weights_path": config.weights_path,
            "model_source": config.model_source,
            "target_labels": list(config.target_labels),
            "non_empty_ratio_threshold": config.non_empty_ratio_threshold,
            "refine_ratio_threshold": config.refine_ratio_threshold,
            "min_non_empty_minutes": config.min_non_empty_minutes,
            "dense_validation_stride_seconds": config.dense_validation_stride_seconds,
        }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if config is not None:
        _append_scan_index_entry(
            run_dir.parent,
            generated_at=generated_at,
            run_dir=run_dir,
            report_json_path=json_path,
            empty_video_log_path=empty_video_log_path,
            empty_video_csv_path=empty_video_csv_export_path,
            config=config,
            result_count=len(results),
        )

    return json_path, empty_video_log_path
