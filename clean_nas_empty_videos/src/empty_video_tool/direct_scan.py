from __future__ import annotations

import argparse
import os
import sys
from typing import TextIO

from pathlib import Path

from .cli import (
    _host_folder_display,
    _print_empty_video_log,
    _print_line,
    _print_scan_summary,
    _print_video_preview,
    _progress_handler,
    _resolve_empty_video_csv_path,
    _resolve_selected_folder,
    _selected_folder_video_name_filter,
    _env_float,
    _env_int,
)
from .models import ScanConfig
from .pipeline import scan_videos
from .reporting import (
    find_latest_saved_scan,
    load_saved_scan,
)


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
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Ignore any saved report and run a fresh scan.",
    )
    return parser


def main(argv: list[str] | None = None, *, output_stream: TextIO | None = None) -> int:
    output_stream = output_stream or sys.stdout
    settings = _load_settings()
    parser = _build_parser(settings)
    args = parser.parse_args(argv)

    folder_arg = str(args.folder_option or args.target_folder or "").strip()
    if not folder_arg:
        parser.error("provide a target folder or use --folder")

    data_root = Path(str(settings["data_root"])).expanduser().resolve()
    output_root = Path(str(settings["output_root"])).expanduser().resolve()

    if not data_root.exists():
        _print_line(output_stream, f"Mounted data root does not exist: {data_root}")
        return 1

    try:
        selected_folder = _resolve_selected_folder(
            data_root,
            folder_arg,
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
