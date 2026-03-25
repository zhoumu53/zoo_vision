from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TextIO

from .discovery import discover_video_files
from .fs import ensure_within_root, list_subdirectories
from .models import ScanConfig, ScanProgress
from .pipeline import scan_videos

DIRECTORY_GRID_COLUMNS = 5
DIRECTORY_GRID_PAGE_SIZE = 50
DIRECTORY_GRID_MAX_CELL_WIDTH = 28

# ANSI colors
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"
WHITE = "\033[97m"


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
        "output_root": os.getenv("OUTPUT_ROOT", "./runs"),
        "empty_export_root": os.getenv(
            "EMPTY_EXPORT_ROOT", "/media/ElephantsWD/empty_videos_to_be_deleted",
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
    return parser


def _print_line(output_stream: TextIO, message: str = "") -> None:
    print(message, file=output_stream, flush=True)


def _print_box(output_stream: TextIO, lines: list[str], *, color: str = CYAN) -> None:
    """Print text inside a rounded box with color."""
    content_width = max((len(line) for line in lines), default=0)
    width = max(content_width + 4, 20)
    _print_line(output_stream, f"{color}{'=' * width}{RESET}")
    for line in lines:
        _print_line(output_stream, f"{color}  {line:<{width - 4}}  {RESET}")
    _print_line(output_stream, f"{color}{'=' * width}{RESET}")


def _print_header(output_stream: TextIO, text: str) -> None:
    _print_line(output_stream)
    _print_line(output_stream, f"{BOLD}{BG_BLUE}{WHITE} {text} {RESET}")
    _print_line(output_stream)


def _print_success(output_stream: TextIO, text: str) -> None:
    _print_line(output_stream, f"{GREEN}{BOLD}{text}{RESET}")


def _print_warn(output_stream: TextIO, text: str) -> None:
    _print_line(output_stream, f"{YELLOW}{text}{RESET}")


def _print_error(output_stream: TextIO, text: str) -> None:
    _print_line(output_stream, f"{RED}{BOLD}{text}{RESET}")


def _truncate_label(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return f"{value[:width - 3]}..."


def _interactive_select_folder(
    data_root: Path,
    *,
    top_level_folder_glob: str | None,
    recursive: bool,
    filename_substring_fn,
    input_fn=input,
    output_stream: TextIO,
) -> Path:
    current_folder = data_root
    page_index = 0

    while True:
        relative_label = current_folder.relative_to(data_root).as_posix() if current_folder != data_root else "/"

        _print_header(output_stream, "FOLDER BROWSER")
        _print_line(output_stream, f"  {DIM}Location:{RESET}  {BOLD}{relative_label}{RESET}")
        _print_line(output_stream)
        _print_line(output_stream, f"  {CYAN}{BOLD}[0]{RESET}  {BOLD}>>> Scan this folder <<<{RESET}")

        visible_directories = list_subdirectories(
            current_folder,
            name_glob=top_level_folder_glob if current_folder == data_root else None,
        )

        is_leaf_folder = not visible_directories

        if visible_directories:
            _print_line(output_stream)
            _print_line(output_stream, f"  {DIM}Subfolders ({len(visible_directories)}):{RESET}")
            _print_line(output_stream)
            total_pages = _directory_page_count(visible_directories)
            page_index = min(page_index, total_pages - 1)
            page_start, page_directories = _directory_page(visible_directories, page_index)
            if total_pages > 1:
                _print_line(
                    output_stream,
                    f"  {DIM}Page {page_index + 1}/{total_pages}  "
                    f"({page_start + 1}-{page_start + len(page_directories)} of {len(visible_directories)})"
                    f"  |  [n]ext  [p]rev{RESET}",
                )
            _print_directory_grid(
                page_directories,
                output_stream=output_stream,
                start_index=page_start + 1,
            )
        else:
            total_pages = 1
            page_start = 0
            page_directories = []
            _print_line(output_stream)
            # count videos in this leaf folder
            video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
            try:
                video_count = sum(
                    1 for f in current_folder.iterdir()
                    if f.is_file() and f.suffix.lower() in video_extensions
                )
            except OSError:
                video_count = 0
            if video_count > 0:
                _print_success(output_stream, f"  {video_count} video(s) found in this folder.")
            else:
                _print_warn(output_stream, "  No subfolders or videos here.")

        _print_line(output_stream)
        hints = []
        if not is_leaf_folder:
            hints.append("number = enter folder")
        hints.append("0 or Enter = scan")
        if current_folder != data_root:
            hints.append("b = back")
        hints.append("q = quit")
        _print_line(output_stream, f"  {DIM}{' | '.join(hints)}{RESET}")

        try:
            raw_choice = input_fn(f"\n  {BOLD}{YELLOW}>>>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(1)
        lowered_choice = raw_choice.lower()
        if raw_choice in {"", "0"}:
            return current_folder
        if lowered_choice in {"q", "quit", "exit"}:
            raise SystemExit(1)
        if lowered_choice in {"b", "back"}:
            if current_folder == data_root:
                _print_warn(output_stream, "  Already at the root folder.")
            else:
                current_folder = current_folder.parent
                page_index = 0
            continue
        if lowered_choice == "n":
            if page_index + 1 >= total_pages:
                _print_warn(output_stream, "  Already on the last page.")
            else:
                page_index += 1
            continue
        if lowered_choice == "p":
            if page_index == 0:
                _print_warn(output_stream, "  Already on the first page.")
            else:
                page_index -= 1
            continue
        if not raw_choice.isdigit():
            _print_error(output_stream, "  Enter a valid number.")
            continue

        choice = int(raw_choice)
        visible_start = page_start + 1
        visible_end = page_start + len(page_directories)
        if visible_start <= choice <= visible_end:
            current_folder = visible_directories[choice - 1]
            page_index = 0
            continue

        _print_error(output_stream, "  Invalid choice.")


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
        _print_line(output_stream, "  " + "  ".join(f"{CYAN}{label:<{cell_width}}{RESET}" for label in row))


def _resolve_selected_folder(
    data_root: Path,
    folder_arg: str | None,
    *,
    top_level_folder_glob: str | None = None,
    recursive: bool = True,
    input_fn=input,
    output_stream: TextIO,
) -> Path:
    if not folder_arg:
        if not sys.stdin.isatty():
            raise ValueError("No interactive terminal is available. Pass --folder to choose the target folder.")
        return _interactive_select_folder(
            data_root,
            top_level_folder_glob=top_level_folder_glob,
            recursive=recursive,
            filename_substring_fn=_selected_folder_video_name_filter,
            input_fn=input_fn,
            output_stream=output_stream,
        )

    candidate = Path(folder_arg).expanduser()
    if not candidate.is_absolute():
        candidate = data_root / candidate
    return ensure_within_root(data_root, candidate)


def _progress_handler(output_stream: TextIO):
    def handle(progress: ScanProgress) -> None:
        if progress.phase == "discovering":
            _print_line(output_stream, f"  {DIM}{progress.message}{RESET}")
            return
        if progress.phase == "video_started":
            _print_line(output_stream, f"  {CYAN}[{progress.video_index}]{RESET} {progress.message}")
            return
        if progress.phase == "video_completed":
            msg = progress.message
            if "to_delete=True" in msg:
                tag = f"{RED}{BOLD}TO DELETE{RESET}"
            elif "to_delete=False" in msg:
                tag = f"{GREEN}KEEP{RESET}"
            elif "Skipped" in msg:
                tag = f"{DIM}SKIPPED{RESET}"
            else:
                tag = f"{YELLOW}???{RESET}"
            _print_line(output_stream, f"  {CYAN}[{progress.video_index}]{RESET} {tag}  {DIM}{msg}{RESET}")
            return
        if progress.phase == "scan_completed":
            _print_line(output_stream)
            _print_success(output_stream, f"  {progress.message}")

    return handle


def main(argv: list[str] | None = None, *, input_fn=input, output_stream: TextIO | None = None) -> int:
    output_stream = output_stream or sys.stdout
    settings = _load_settings()
    parser = _build_parser(settings)
    args = parser.parse_args(argv)

    data_root = Path(str(settings["data_root"])).expanduser().resolve()
    output_root = Path(str(settings["output_root"])).expanduser().resolve()

    if not data_root.exists():
        _print_error(output_stream, f"Data root does not exist: {data_root}")
        return 1

    try:
        selected_folder = _resolve_selected_folder(
            data_root,
            args.folder,
            top_level_folder_glob=str(settings.get("top_level_folder_glob") or "").strip() or None,
            recursive=args.recursive,
            input_fn=input_fn,
            output_stream=output_stream,
        )
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        _print_error(output_stream, str(exc))
        return 1

    filename_substring = _selected_folder_video_name_filter(data_root, selected_folder)

    _print_header(output_stream, "SCANNING")
    _print_box(output_stream, [
        f"Target:  {selected_folder}",
        f"Output:  {output_root}",
        f"Export:  {settings['empty_export_root']}",
    ])
    _print_line(output_stream)

    weights_path = args.weights_path.strip() or None

    try:
        config = ScanConfig(
            data_root=data_root,
            target_folder=selected_folder,
            output_root=output_root,
            empty_export_root=Path(str(settings["empty_export_root"])).expanduser(),
            filename_substring=filename_substring,
            recursive=args.recursive,
            interval_minutes=int(args.interval_minutes),
            confidence_threshold=float(args.confidence_threshold),
            weights_path=weights_path,
        )
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        _print_error(output_stream, f"Invalid scan configuration: {exc}")
        return 1

    try:
        results = scan_videos(config, progress_callback=_progress_handler(output_stream))
    except Exception as exc:
        _print_error(output_stream, f"Scan failed: {exc}")
        return 1

    new_count = len(results)
    delete_count = sum(1 for r in results if r.to_delete)
    keep_count = sum(1 for r in results if r.to_delete is False)
    error_count = sum(1 for r in results if r.error)

    _print_header(output_stream, "RESULTS")
    summary_lines = [f"New videos processed:  {new_count}"]
    if delete_count:
        summary_lines.append(f"To delete (empty):     {delete_count}")
    if keep_count:
        summary_lines.append(f"To keep (elephants):   {keep_count}")
    if error_count:
        summary_lines.append(f"Errors:                {error_count}")
    if new_count == 0:
        summary_lines.append("All videos already processed!")

    color = GREEN if error_count == 0 else YELLOW
    _print_box(output_stream, summary_lines, color=color)
    _print_line(output_stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
