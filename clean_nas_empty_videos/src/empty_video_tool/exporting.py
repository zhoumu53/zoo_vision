from __future__ import annotations

from datetime import datetime
import csv
from pathlib import Path
import re


DATE_PART_PATTERN = re.compile(r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?:AM|PM)?$")
EMPTY_VIDEO_EXPORT_FIELDNAMES = [
    "host_path",
    "date",
    "year",
    "folder_name",
    "video_filename",
]


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


def _normalized_root(host_root: str | None) -> str:
    if not host_root:
        return ""
    return Path(host_root).expanduser().as_posix()


def _resolved_file_path(scan_row: dict) -> Path | None:
    for key in ("video_path", "host_video_path"):
        raw_path = str(scan_row.get(key) or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _extract_date_year_from_relative_path(relative_path: str) -> tuple[str, str]:
    for part in Path(relative_path).parts:
        match = DATE_PART_PATTERN.match(part)
        if match is None:
            continue
        year = match.group("year")
        month = match.group("month")
        day = match.group("day")
        return f"{year}-{month}-{day}", year
    return "", ""


def _fallback_date_year(file_path: Path | None) -> tuple[str, str]:
    if file_path is None:
        return "", ""
    modified_at = datetime.fromtimestamp(file_path.stat().st_mtime)
    return modified_at.strftime("%Y-%m-%d"), modified_at.strftime("%Y")


def _date_key(date_value: str) -> str:
    normalized = str(date_value).strip().replace("-", "")
    return normalized or "unknown-date"


def _size_bytes(file_path: Path | None) -> int | None:
    if file_path is None:
        return None
    return file_path.stat().st_size


def _normalized_relative_path(scan_row: dict, *, normalized_root: str) -> str:
    relative_path = str(scan_row.get("relative_path") or "").strip()
    if relative_path:
        return relative_path

    host_video_path = str(scan_row.get("host_video_path") or "").strip()
    if host_video_path and normalized_root:
        host_path = Path(host_video_path)
        try:
            return host_path.relative_to(Path(normalized_root)).as_posix()
        except ValueError:
            return host_path.name
    if host_video_path:
        return Path(host_video_path).name
    return ""


def _top_level_folder(relative_path: str, *, host_root: str) -> str:
    path_parts = Path(relative_path).parts if relative_path else ()
    if len(path_parts) > 1:
        top_level = path_parts[0]
        if top_level not in {".", ""}:
            return top_level
    if host_root:
        return Path(host_root).name
    return "root"


def build_empty_video_export_rows(scan_rows: list[dict], *, host_root: str | None = None) -> list[dict]:
    normalized_root = _normalized_root(host_root)
    export_rows: list[dict] = []

    for row in scan_rows:
        if row.get("status") != "ok" or not row.get("empty_video"):
            continue

        host_video_path = str(row.get("host_video_path") or "").strip()
        if not host_video_path:
            continue

        host_path = Path(host_video_path)
        relative_path = _normalized_relative_path(row, normalized_root=normalized_root)
        relative_folder = Path(relative_path).parent.as_posix() if relative_path else ""
        if relative_folder == ".":
            relative_folder = ""
        top_level_folder = _top_level_folder(relative_path, host_root=normalized_root)
        file_path = _resolved_file_path(row)
        date_value, year_value = _extract_date_year_from_relative_path(relative_path)
        if not date_value:
            date_value, year_value = _fallback_date_year(file_path)
        size_bytes = _size_bytes(file_path)
        size_mb = round(size_bytes / (1024 * 1024), 3) if size_bytes is not None else None

        export_rows.append(
            {
                "host_path": host_video_path,
                "host_video_path": host_video_path,
                "top_level_folder": top_level_folder,
                "date": date_value,
                "date_key": _date_key(date_value),
                "year": year_value,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "root": normalized_root,
                "folder": host_path.parent.as_posix(),
                "folder_name": host_path.parent.name,
                "relative_folder": relative_folder,
                "video_filename": host_path.name,
                "relative_path": relative_path,
                "classification_reason": str(row.get("classification_reason") or ""),
                "model_source": str(row.get("model_source") or ""),
            }
        )

    return export_rows


def grouped_empty_video_export_path(export_root: Path, row: dict) -> Path:
    top_level_folder = str(row.get("top_level_folder") or "").strip() or "root"
    date_key = _date_key(str(row.get("date_key") or row.get("date") or ""))
    return export_root / top_level_folder / f"{date_key}.csv"


def grouped_empty_video_log_path(
    export_root: Path,
    row: dict,
    *,
    log_filename: str = "empty_video_paths.log",
) -> Path:
    top_level_folder = str(row.get("top_level_folder") or "").strip() or "root"
    date_key = _date_key(str(row.get("date_key") or row.get("date") or ""))
    return export_root / top_level_folder / date_key / log_filename


def write_grouped_empty_video_exports(
    export_root: Path,
    scan_rows: list[dict],
    *,
    host_root: str | None = None,
) -> list[Path]:
    export_root.mkdir(parents=True, exist_ok=True)
    export_rows = build_empty_video_export_rows(scan_rows, host_root=host_root)
    rows_by_group: dict[Path, list[dict]] = {}
    for row in export_rows:
        export_path = grouped_empty_video_export_path(export_root, row)
        rows_by_group.setdefault(export_path, []).append(row)

    export_paths: list[Path] = []
    for export_path in sorted(rows_by_group.keys(), key=lambda path: path.as_posix()):
        _append_unique_csv_rows(
            export_path,
            rows_by_group[export_path],
            fieldnames=EMPTY_VIDEO_EXPORT_FIELDNAMES,
            unique_field="host_path",
        )
        export_paths.append(export_path)
    return export_paths
