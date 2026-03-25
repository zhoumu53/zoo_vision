from __future__ import annotations

import argparse
import os
import sys
from typing import TextIO

from pathlib import Path

from .cli import (
    _print_line,
    _progress_handler,
    _resolve_selected_folder,
    _selected_folder_video_name_filter,
    _env_float,
    _env_int,
)
from .models import ScanConfig
from .pipeline import scan_videos


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
    }


def _build_parser(defaults: dict[str, str | int | float]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="empty-video-scan",
        description="Non-interactive empty-video scanner for a directly provided target folder.",
    )
    parser.add_argument(
        "target_folder",
        nargs="?",
        help="Folder to scan, either relative to DATA_ROOT or an absolute path under DATA_ROOT.",
    )
    parser.add_argument(
        "--folder",
        dest="folder_option",
        help="Alias for target_folder.",
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


def main(argv: list[str] | None = None, *, output_stream: TextIO | None = None) -> int:
    output_stream = output_stream or sys.stdout
    settings = _load_settings()
    parser = _build_parser(settings)
    args = parser.parse_args(argv)

    folder_arg = str(args.folder_option or args.target_folder or "").strip()

    data_root = Path(str(settings["data_root"])).expanduser().resolve()
    output_root = Path(str(settings["output_root"])).expanduser().resolve()

    if not data_root.exists():
        _print_line(output_stream, f"Data root does not exist: {data_root}")
        return 1

    try:
        if folder_arg:
            selected_folder = _resolve_selected_folder(
                data_root,
                folder_arg,
                output_stream=output_stream,
            )
        else:
            selected_folder = data_root
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        _print_line(output_stream, str(exc))
        return 1

    filename_substring = _selected_folder_video_name_filter(data_root, selected_folder)
    _print_line(output_stream, f"Target folder: {selected_folder}")

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
        _print_line(output_stream, f"Invalid scan configuration: {exc}")
        return 1

    try:
        results = scan_videos(config, progress_callback=_progress_handler(output_stream))
    except Exception as exc:
        _print_line(output_stream, f"Scan failed: {exc}")
        return 1

    empty_count = sum(1 for r in results if r.to_delete)
    error_count = sum(1 for r in results if r.error)
    _print_line(output_stream)
    _print_line(output_stream, f"New videos processed: {len(results)}")
    _print_line(output_stream, f"To delete: {empty_count}")
    if error_count:
        _print_line(output_stream, f"Errors: {error_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
