from __future__ import annotations

import fnmatch
import os
from pathlib import Path


def ensure_within_root(root: Path, candidate: Path) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved_candidate = candidate.expanduser().resolve()
    resolved_candidate.relative_to(resolved_root)
    return resolved_candidate


def resolve_child(root: Path, relative_path: str = "") -> Path:
    if relative_path in {"", "."}:
        return ensure_within_root(root, root)
    return ensure_within_root(root, root / relative_path)


def list_subdirectories(folder: Path, *, name_glob: str | None = None) -> list[Path]:
    subdirectories: list[Path] = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if name_glob and not fnmatch.fnmatch(entry.name, name_glob):
                            continue
                        subdirectories.append(Path(entry.path))
                except OSError:
                    continue
    except OSError:
        return []

    return sorted(subdirectories, key=lambda item: item.name.lower())


def relative_to_root(root: Path, path: Path) -> str:
    return ensure_within_root(root, path).relative_to(root.expanduser().resolve()).as_posix()
