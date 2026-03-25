from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import TextIO

from .discovery import contains_video_files, discover_video_files
from .fs import ensure_within_root, list_subdirectories
from .models import ScanConfig, ScanProgress
from .pipeline import scan_videos
from .reporting import (
    empty_video_csv_path,
    find_latest_saved_scan,
    folder_has_saved_scan_coverage,
    load_saved_scan,
    load_scan_index_entries,
)

MAX_RAW_LOG_LINES = 20
DIRECTORY_GRID_COLUMNS = 10
DIRECTORY_GRID_PAGE_SIZE = 100
DIRECTORY_GRID_MAX_CELL_WIDTH = 24


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return float(raw_value)
    except ValueError:
        return default


def _load_settings() -> dict[str, str | int | float]:
    return {
        "data_root": os.getenv("DATA_ROOT", "/mnt/camera_nas/"),
        "host_data_root_display": os.getenv("HOST_DATA_ROOT_DISPLAY", "/mnt/camera_nas/"),
        "output_root": os.getenv("OUTPUT_ROOT", "./runs"),
        "empty_export_root": os.getenv(
            "EMPTY_EXPORT_ROOT",
            os.getenv("HOST_EMPTY_EXPORT_ROOT", "/media/ElephantsWD/empty_videos_to_be_deleted"),
        ),
        "default_interval": _env_int("DEFAULT_INTERVAL_MINUTES", 2),
        "default_confidence": _env_float("DEFAULT_CONFIDENCE", 0.65),
        "default_weights_path": os.getenv("DEFAULT_WEIGHTS_PATH", "").strip(),
        "top_level_folder_glob": os.getenv("TOP_LEVEL_FOLDER_GLOB", "").strip(),
    }


def _selected_folder_video_name_filter(data_root: Path, selected_folder: Path) -> str | None:
    try:
        relative_parts = selected_folder.relative_to(data_root).parts
    except ValueError:
        return None
    if len(relative_parts) == 1:
        return "ELP-"
    return None


def _build_parser(defaults: dict[str, str | int | float]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="empty-video-cli",
        description="Interactive CLI for empty-video review scans.",
    )
    parser.add_argument(
        "--folder",
        help="Folder to scan, either relative to DATA_ROOT or an absolute path under DATA_ROOT. "
        "If omitted, the CLI opens an interactive folder chooser.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan subfolders recursively. Enabled by default.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=int(defaults["default_interval"]),
        help="Coarse uniform sample interval in minutes.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(defaults["default_confidence"]),
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--weights-path",
        default=str(defaults["default_weights_path"]),
        help="Optional custom YOLO weights path.",
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Ignore any saved report and run a fresh scan.",
    )
    return parser


def _print_line(output_stream: TextIO, message: str = "") -> None:
    print(message, file=output_stream)


def _truncate_label(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return f"{value[:width - 3]}..."


def _interactive_select_folder(
    data_root: Path,
    *,
    scan_index_entries: list[dict],
    recursive: bool,
    interval_minutes: int,
    confidence_threshold: float,
    weights_path: str | None,
    top_level_folder_glob: str | None,
    input_fn=input,
    output_stream: TextIO,
) -> Path:
    current_folder = data_root
    page_index = 0

    while True:
        relative_label = current_folder.relative_to(data_root).as_posix() if current_folder != data_root else "/"
        _print_line(output_stream)
        _print_line(output_stream, f"Current folder: {relative_label}")

        current_folder_scanned = folder_has_saved_scan_coverage(
            scan_index_entries,
            target_folder=current_folder,
            recursive=recursive,
            filename_substring=_selected_folder_video_name_filter(data_root, current_folder),
            interval_minutes=interval_minutes,
            confidence_threshold=confidence_threshold,
            weights_path=weights_path,
        )
        visible_directories, hidden_scanned_count, hidden_empty_count = _visible_child_directories(
            current_folder,
            data_root=data_root,
            scan_index_entries=scan_index_entries,
            recursive=recursive,
            interval_minutes=interval_minutes,
            confidence_threshold=confidence_threshold,
            weights_path=weights_path,
            top_level_folder_glob=top_level_folder_glob,
        )

        if current_folder_scanned:
            _print_line(output_stream, "Current folder already has a matching saved scan.")
            _print_line(output_stream, "0. Use this folder anyway")
        else:
            _print_line(output_stream, "0. Use this folder")

        if hidden_scanned_count:
            _print_line(output_stream, f"Hiding {hidden_scanned_count} already-scanned folder(s).")
        if hidden_empty_count:
            _print_line(output_stream, f"Hiding {hidden_empty_count} empty folder(s) with no matching videos.")

        is_leaf_folder = not visible_directories

        if visible_directories:
            total_pages = _directory_page_count(visible_directories)
            page_index = min(page_index, total_pages - 1)
            page_start, page_directories = _directory_page(visible_directories, page_index)
            if total_pages > 1:
                _print_line(
                    output_stream,
                    f"Showing folders {page_start + 1}-{page_start + len(page_directories)} "
                    f"of {len(visible_directories)} (page {page_index + 1}/{total_pages}).",
                )
                _print_line(output_stream, "**Use n/p to move between pages. [n]ext, [p]revious.**")
            _print_directory_grid(
                page_directories,
                output_stream=output_stream,
                start_index=page_start + 1,
            )
        else:
            total_pages = 1
            page_start = 0
            page_directories = []
            _print_line(output_stream, "No unscanned subfolders were found here.")
            _print_video_preview(
                output_stream=output_stream,
                data_root=data_root,
                selected_folder=current_folder,
                recursive=recursive,
                filename_substring=_selected_folder_video_name_filter(data_root, current_folder),
            )

        if is_leaf_folder:
            prompt = "Press Enter or 0 to scan this folder, or b to go back: "
        elif current_folder == data_root:
            prompt = "Choose a folder number, or 0 to scan the current folder: "
        else:
            prompt = "Choose a folder number, 0 to scan the current folder, or b to go back: "

        raw_choice = input_fn(prompt).strip()
        lowered_choice = raw_choice.lower()
        if raw_choice in {"", "0"}:
            return current_folder
        if lowered_choice in {"q", "quit", "exit"}:
            raise SystemExit(1)
        if lowered_choice in {"b", "back"}:
            if current_folder == data_root:
                _print_line(output_stream, "Already at the root folder.")
            else:
                current_folder = current_folder.parent
                page_index = 0
            continue
        if lowered_choice == "n":
            if not visible_directories or total_pages == 1:
                _print_line(output_stream, "No additional pages are available.")
            elif page_index + 1 >= total_pages:
                _print_line(output_stream, "Already on the last page.")
            else:
                page_index += 1
            continue
        if lowered_choice == "p":
            if not visible_directories or total_pages == 1:
                _print_line(output_stream, "No previous pages are available.")
            elif page_index == 0:
                _print_line(output_stream, "Already on the first page.")
            else:
                page_index -= 1
            continue
        if not raw_choice.isdigit():
            _print_line(output_stream, "Enter a valid number.")
            continue

        choice = int(raw_choice)
        visible_start = page_start + 1
        visible_end = page_start + len(page_directories)
        if visible_start <= choice <= visible_end:
            current_folder = visible_directories[choice - 1]
            page_index = 0
            continue

        if visible_directories:
            _print_line(
                output_stream,
                f"Choose a visible number between {visible_start} and {visible_end}, "
                "or use n/p for more folders.",
            )
        else:
            _print_line(output_stream, "Use 0 to scan this folder, or b to go back.")


def _directory_page_count(directories: list[Path], *, page_size: int = DIRECTORY_GRID_PAGE_SIZE) -> int:
    return max(1, (len(directories) + page_size - 1) // page_size)


def _directory_page(
    directories: list[Path],
    page_index: int,
    *,
    page_size: int = DIRECTORY_GRID_PAGE_SIZE,
) -> tuple[int, list[Path]]:
    page_start = max(page_index, 0) * page_size
    page_end = page_start + page_size
    return page_start, directories[page_start:page_end]


def _directory_grid_cell_width(
    directories: list[Path],
    *,
    start_index: int = 1,
    max_cell_width: int = DIRECTORY_GRID_MAX_CELL_WIDTH,
) -> int:
    if not directories:
        return 0
    label_lengths = [
        len(f"{index}. {directory.name}")
        for index, directory in enumerate(directories, start=start_index)
    ]
    return min(max(label_lengths), max_cell_width)


def _print_directory_grid(
    directories: list[Path],
    *,
    output_stream: TextIO,
    start_index: int = 1,
) -> None:
    if not directories:
        return

    columns = min(DIRECTORY_GRID_COLUMNS, len(directories))
    cell_width = _directory_grid_cell_width(directories, start_index=start_index)
    option_labels = [
        _truncate_label(f"{index}. {directory.name}", cell_width)
        for index, directory in enumerate(directories, start=start_index)
    ]

    for row_start in range(0, len(option_labels), columns):
        row = option_labels[row_start:row_start + columns]
        _print_line(output_stream, "  ".join(f"{label:<{cell_width}}" for label in row))


def _visible_child_directories(
    current_folder: Path,
    *,
    data_root: Path,
    scan_index_entries: list[dict],
    recursive: bool,
    interval_minutes: int,
    confidence_threshold: float,
    weights_path: str | None,
    top_level_folder_glob: str | None,
) -> tuple[list[Path], int, int]:
    filter_empty_directories = current_folder == data_root
    child_directories = list_subdirectories(
        current_folder,
        name_glob=top_level_folder_glob if current_folder == data_root else None,
    )
    visible_directories: list[Path] = []
    hidden_scanned_count = 0
    hidden_empty_count = 0

    for child in child_directories:
        filename_substring = _selected_folder_video_name_filter(data_root, child)
        child_scanned = folder_has_saved_scan_coverage(
            scan_index_entries,
            target_folder=child,
            recursive=recursive,
            filename_substring=filename_substring,
            interval_minutes=interval_minutes,
            confidence_threshold=confidence_threshold,
            weights_path=weights_path,
            include_ancestor_recursive_coverage=False,
        )
        if child_scanned:
            hidden_scanned_count += 1
            continue

        if filter_empty_directories and not contains_video_files(
            child,
            recursive=recursive,
            filename_substring=filename_substring,
        ):
            hidden_empty_count += 1
            continue
        visible_directories.append(child)

    return visible_directories, hidden_scanned_count, hidden_empty_count


def _resolve_selected_folder(
    data_root: Path,
    folder_arg: str | None,
    *,
    scan_index_entries: list[dict] | None = None,
    recursive: bool = True,
    interval_minutes: int = 2,
    confidence_threshold: float = 0.65,
    weights_path: str | None = None,
    top_level_folder_glob: str | None = None,
    input_fn=input,
    output_stream: TextIO,
) -> Path:
    if not folder_arg:
        if not sys.stdin.isatty():
            raise ValueError("No interactive terminal is available. Pass --folder to choose the target folder.")
        return _interactive_select_folder(
            data_root,
            scan_index_entries=scan_index_entries or [],
            recursive=recursive,
            interval_minutes=interval_minutes,
            confidence_threshold=confidence_threshold,
            weights_path=weights_path,
            top_level_folder_glob=top_level_folder_glob,
            input_fn=input_fn,
            output_stream=output_stream,
        )

    candidate = Path(folder_arg).expanduser()
    if not candidate.is_absolute():
        candidate = data_root / candidate
    return ensure_within_root(data_root, candidate)


def _host_folder_display(selected_folder: Path, data_root: Path, host_data_root_display: str) -> str:
    try:
        relative_path = selected_folder.relative_to(data_root)
    except ValueError:
        return str(selected_folder)
    return str((Path(host_data_root_display) / relative_path).as_posix())


def _resolve_empty_video_csv_path(run_dir: str | None, empty_video_csv: str | None) -> str:
    candidate = str(empty_video_csv or "").strip()
    if candidate:
        return candidate
    if not run_dir:
        return ""
    derived = empty_video_csv_path(Path(run_dir))
    return str(derived) if derived.exists() else ""


def _progress_handler(output_stream: TextIO):
    def handle(progress: ScanProgress) -> None:
        if progress.phase == "discovering":
            _print_line(output_stream, progress.message)
            return
        if progress.phase == "video_started":
            _print_line(
                output_stream,
                f"[{progress.video_index}/{progress.total_videos}] {progress.message}",
            )
            return
        if progress.phase == "video_completed":
            _print_line(
                output_stream,
                f"[{progress.video_index}/{progress.total_videos}] {progress.message}",
            )
            return
        if progress.phase == "scan_completed":
            _print_line(output_stream, progress.message)

    return handle


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _group_paths_by_folder(
    paths: list[Path],
    *,
    base_folder: Path | None = None,
    current_folder_label: str = "",
) -> list[tuple[str, int, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        if base_folder is not None and path.is_relative_to(base_folder):
            relative_parent = path.parent.relative_to(base_folder)
            folder_name = current_folder_label if relative_parent == Path(".") else relative_parent.as_posix()
        else:
            folder_name = path.parent.as_posix()
        grouped[folder_name].append(path.name)

    rows: list[tuple[str, int, str]] = []
    for folder_name in sorted(grouped.keys(), key=str.lower):
        filenames = sorted(grouped[folder_name], key=str.lower)
        rows.append((folder_name, len(filenames), filenames[0]))
    return rows


def _print_folder_table(
    rows: list[tuple[str, int, str]],
    *,
    output_stream: TextIO,
    count_header: str,
    sample_header: str,
) -> None:
    if not rows:
        return

    _print_line(output_stream, f"| Folder | {count_header} | {sample_header} |")
    _print_line(output_stream, "| --- | ---: | --- |")
    for folder_name, count, sample_name in rows:
        _print_line(
            output_stream,
            f"| {_markdown_cell(folder_name)} | {count} | {_markdown_cell(sample_name)} |",
        )


def _print_video_preview(
    *,
    output_stream: TextIO,
    data_root: Path,
    selected_folder: Path,
    recursive: bool,
    filename_substring: str | None,
) -> int:
    video_paths = discover_video_files(
        selected_folder,
        recursive=recursive,
        filename_substring=filename_substring,
    )
    _print_line(output_stream)
    _print_line(output_stream, f"Selected folder: {selected_folder}")
    _print_line(output_stream, f"Videos queued: {len(video_paths)}")
    _print_line(output_stream, f"Scan mode: {'recursive' if recursive else 'current folder only'}")
    if filename_substring:
        _print_line(output_stream, f"Active filename filter: *{filename_substring}*")

    if not video_paths:
        return 0

    current_folder_label = selected_folder.name or selected_folder.as_posix()
    folder_rows = _group_paths_by_folder(
        video_paths,
        base_folder=selected_folder,
        current_folder_label=current_folder_label,
    )
    _print_line(output_stream, f"Folders queued: {len(folder_rows)}")
    _print_folder_table(
        folder_rows,
        output_stream=output_stream,
        count_header="Videos",
        sample_header="Example video",
    )
    return len(video_paths)


def _print_scan_summary(rows: list[dict], run_meta: dict, *, output_stream: TextIO) -> None:
    empty_count = sum(1 for row in rows if row.get("empty_video"))
    error_count = sum(1 for row in rows if row.get("status") == "error")

    _print_line(output_stream)
    _print_line(output_stream, "Scan Summary")
    if run_meta.get("source") == "saved_report":
        _print_line(output_stream, "Source: saved report")
    generated_at = run_meta.get("generated_at")
    if generated_at:
        _print_line(output_stream, f"Generated at: {generated_at}")
    _print_line(output_stream, f"Run directory: {run_meta['run_dir']}")
    _print_line(output_stream, f"JSON report: {run_meta['report_json_path']}")
    _print_line(output_stream, f"Empty video log: {run_meta['empty_video_log_path']}")
    empty_csv_path = str(run_meta.get("empty_video_csv_path") or "").strip()
    if empty_csv_path:
        _print_line(output_stream, f"Empty video CSV: {empty_csv_path}")
    _print_line(output_stream, f"Processed videos: {len(rows)}")
    _print_line(output_stream, f"Empty videos: {empty_count}")
    _print_line(output_stream, f"Errors: {error_count}")


def _print_empty_video_log(log_path: Path, *, output_stream: TextIO) -> None:
    _print_line(output_stream)
    _print_line(output_stream, "Empty Video Log")
    _print_line(output_stream, str(log_path))
    if not log_path.exists():
        _print_line(output_stream, "(log file not found)")
        return

    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        _print_line(output_stream, "(no empty videos)")
        return

    log_paths = [Path(line) for line in lines]
    folder_rows = _group_paths_by_folder(log_paths, base_folder=None)
    _print_folder_table(
        folder_rows,
        output_stream=output_stream,
        count_header="Files",
        sample_header="Example file",
    )

    if len(lines) <= MAX_RAW_LOG_LINES:
        _print_line(output_stream)
        _print_line(output_stream, "Raw paths")
        for line in lines:
            _print_line(output_stream, line)
    else:
        _print_line(output_stream)
        _print_line(output_stream, f"Raw paths are stored in the log file. Skipped printing {len(lines)} lines here.")


def main(argv: list[str] | None = None, *, input_fn=input, output_stream: TextIO | None = None) -> int:
    output_stream = output_stream or sys.stdout
    settings = _load_settings()
    parser = _build_parser(settings)
    args = parser.parse_args(argv)

    data_root = Path(str(settings["data_root"])).expanduser().resolve()
    output_root = Path(str(settings["output_root"])).expanduser().resolve()
    scan_index_entries = load_scan_index_entries(output_root)

    if not data_root.exists():
        _print_line(output_stream, f"Mounted data root does not exist: {data_root}")
        return 1

    try:
        selected_folder = _resolve_selected_folder(
            data_root,
            args.folder,
            scan_index_entries=scan_index_entries,
            recursive=args.recursive,
            interval_minutes=int(args.interval_minutes),
            confidence_threshold=float(args.confidence_threshold),
            weights_path=args.weights_path.strip() or None,
            top_level_folder_glob=str(settings.get("top_level_folder_glob") or "").strip() or None,
            input_fn=input_fn,
            output_stream=output_stream,
        )
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        _print_line(output_stream, str(exc))
        return 1

    filename_substring = _selected_folder_video_name_filter(data_root, selected_folder)
    _print_line(output_stream, f"Host folder: {_host_folder_display(selected_folder, data_root, str(settings['host_data_root_display']))}")
    _print_video_preview(
        output_stream=output_stream,
        data_root=data_root,
        selected_folder=selected_folder,
        recursive=args.recursive,
        filename_substring=filename_substring,
    )

    weights_path = args.weights_path.strip() or None

    if not args.rescan:
        saved_scan = find_latest_saved_scan(
            output_root,
            target_folder=selected_folder,
            recursive=args.recursive,
            filename_substring=filename_substring,
            interval_minutes=int(args.interval_minutes),
            confidence_threshold=float(args.confidence_threshold),
            weights_path=weights_path,
            exact_settings=True,
        )
        if saved_scan is not None:
            payload = load_saved_scan(Path(str(saved_scan["report_json_path"])))
            if payload is not None:
                rows = payload["results"]
                run_meta = {
                    "run_dir": str(saved_scan["run_dir"]),
                    "report_json_path": str(saved_scan["report_json_path"]),
                    "empty_video_log_path": str(saved_scan["empty_video_log_path"]),
                    "empty_video_csv_path": _resolve_empty_video_csv_path(
                        str(saved_scan["run_dir"]),
                        str(saved_scan.get("empty_video_csv_path") or payload.get("empty_video_csv_path") or ""),
                    ),
                    "generated_at": payload.get("generated_at") or saved_scan.get("generated_at"),
                    "source": "saved_report",
                }
                _print_line(output_stream)
                _print_line(output_stream, "Loaded the latest saved report for this folder and settings. Use --rescan to refresh it.")
                _print_scan_summary(rows, run_meta, output_stream=output_stream)
                _print_empty_video_log(Path(run_meta["empty_video_log_path"]), output_stream=output_stream)
                return 0

    try:
        config = ScanConfig(
            data_root=data_root,
            target_folder=selected_folder,
            output_root=output_root,
            empty_export_root=Path(str(settings["empty_export_root"])).expanduser(),
            host_data_root_display=str(settings["host_data_root_display"]),
            filename_substring=filename_substring,
            recursive=args.recursive,
            interval_minutes=int(args.interval_minutes),
            confidence_threshold=float(args.confidence_threshold),
            weights_path=weights_path,
        )
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        _print_line(output_stream, f"Invalid scan configuration: {exc}")
        return 1

    try:
        run_result = scan_videos(config, progress_callback=_progress_handler(output_stream))
    except Exception as exc:
        _print_line(output_stream, f"Scan failed: {exc}")
        return 1

    saved_payload = load_saved_scan(Path(run_result.report_json_path))
    rows = saved_payload["results"] if saved_payload is not None else [result.to_dict() for result in run_result.results]
    run_meta = {
        "run_dir": run_result.run_dir,
        "report_json_path": run_result.report_json_path,
        "empty_video_log_path": run_result.empty_video_log_path,
        "empty_video_csv_path": _resolve_empty_video_csv_path(
            run_result.run_dir,
            (
                str(saved_payload.get("empty_video_csv_path") or "")
                if saved_payload is not None
                else run_result.empty_video_csv_path
            ),
        ),
        "generated_at": saved_payload.get("generated_at") if saved_payload is not None else None,
        "source": "scan",
    }
    _print_scan_summary(rows, run_meta, output_stream=output_stream)
    _print_empty_video_log(Path(run_result.empty_video_log_path), output_stream=output_stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
