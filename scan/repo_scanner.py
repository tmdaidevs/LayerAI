"""Repository scanning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


MAX_FILE_SIZE_BYTES = 500 * 1024
ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".json", ".yaml", ".yml", ".md"}
IGNORED_DIRS = {"node_modules", ".git", "dist", "build", ".venv"}


@dataclass(frozen=True)
class FileObject:
    path: str
    extension: str
    size: int
    content: str


def scan_repo(path: str) -> List[FileObject]:
    """Scan a repository and return metadata for selected files."""
    root = Path(path)
    results: List[FileObject] = []

    for current_root, dirs, files in _walk_dirs(root):
        for file_name in files:
            file_path = current_root / file_name
            extension = file_path.suffix.lower()
            if extension not in ALLOWED_EXTENSIONS:
                continue

            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size > MAX_FILE_SIZE_BYTES:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            results.append(
                FileObject(
                    path=str(file_path),
                    extension=extension,
                    size=size,
                    content=content,
                )
            )

    return results


def _walk_dirs(root: Path):
    for current_root, dirs, files in root.walk():
        dirs[:] = [dir_name for dir_name in dirs if dir_name not in IGNORED_DIRS]
        yield current_root, dirs, files
