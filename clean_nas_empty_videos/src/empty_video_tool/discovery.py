from __future__ import annotations

import os
from pathlib import Path


VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv"})


def is_video_file(path: Path) -> bool:
    try:
        return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    except OSError:
        return False


def _matches_filename_filter(filename: str, filename_substring: str | None) -> bool:
    if not filename_substring:
        return True
    return filename_substring in filename


def contains_video_files(folder: Path, recursive: bool = True, filename_substring: str | None = None) -> bool:
    if recursive:
        def _ignore_walk_error(_: OSError) -> None:
            return None

        for root, dirnames, filenames in os.walk(
            folder,
            topdown=True,
            onerror=_ignore_walk_error,
            followlinks=False,
        ):
            dirnames.sort(key=str.lower)
            filenames.sort(key=str.lower)
            for filename in filenames:
                candidate = Path(root) / filename
                if candidate.suffix.lower() in VIDEO_EXTENSIONS and _matches_filename_filter(candidate.name, filename_substring):
                    return True
        return False

    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if (
                        entry.is_file(follow_symlinks=False)
                        and Path(entry.name).suffix.lower() in VIDEO_EXTENSIONS
                        and _matches_filename_filter(entry.name, filename_substring)
                    ):
                        return True
                except OSError:
                    continue
    except OSError:
        return False
    return False


def discover_video_files(folder: Path, recursive: bool = True, filename_substring: str | None = None) -> list[Path]:
    videos: list[Path] = []

    if recursive:
        def _ignore_walk_error(_: OSError) -> None:
            return None

        for root, dirnames, filenames in os.walk(
            folder,
            topdown=True,
            onerror=_ignore_walk_error,
            followlinks=False,
        ):
            dirnames.sort(key=str.lower)
            filenames.sort(key=str.lower)
            for filename in filenames:
                candidate = Path(root) / filename
                if candidate.suffix.lower() in VIDEO_EXTENSIONS and _matches_filename_filter(candidate.name, filename_substring):
                    videos.append(candidate)
    else:
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        if (
                            entry.is_file(follow_symlinks=False)
                            and Path(entry.name).suffix.lower() in VIDEO_EXTENSIONS
                            and _matches_filename_filter(entry.name, filename_substring)
                        ):
                            videos.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            return []

    return sorted(videos, key=lambda path: str(path).lower())
