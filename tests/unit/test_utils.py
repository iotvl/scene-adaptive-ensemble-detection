"""Tests for src/utils/images.py and src/data/converter.py."""

from __future__ import annotations

from pathlib import Path

from src.utils.images import find_images


class TestFindImages:
    def test_finds_images(self, tmp_path: Path):
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.txt").touch()
        (tmp_path / "d.bmp").touch()
        images = find_images(tmp_path)
        names = {p.name for p in images}
        assert "a.jpg" in names
        assert "b.png" in names
        assert "d.bmp" in names
        assert "c.txt" not in names

    def test_empty_dir(self, tmp_path: Path):
        assert find_images(tmp_path) == []

    def test_case_insensitive_extension(self, tmp_path: Path):
        (tmp_path / "upper.JPG").touch()
        images = find_images(tmp_path)
        assert len(images) == 1
