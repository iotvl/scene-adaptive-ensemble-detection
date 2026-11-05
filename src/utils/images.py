"""
Image file discovery utilities.
"""

from __future__ import annotations

from pathlib import Path

from src.constants import IMAGE_EXTENSIONS


def find_images(directory: Path | str) -> list[Path]:
    """Return sorted list of image files in *directory*.

    Only files whose suffix (lowercased) is in ``IMAGE_EXTENSIONS`` are
    included.  Non-recursive — only the immediate directory is scanned.
    """
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
