"""Tests for src/constants.py."""

from __future__ import annotations

from src.constants import (
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    VISDRONE_CLASSES,
    VISDRONE_TO_YOLO_CLASS,
    SceneType,
)


class TestVisDroneClasses:
    def test_class_count(self):
        assert NUM_CLASSES == 8
        assert len(VISDRONE_CLASSES) == NUM_CLASSES

    def test_known_classes(self):
        assert "pedestrian" in VISDRONE_CLASSES
        assert "car" in VISDRONE_CLASSES
        assert "bus" in VISDRONE_CLASSES

    def test_dropped_classes(self):
        # tricycle and awning-tricycle are intentionally absent.
        assert "tricycle" not in VISDRONE_CLASSES
        assert "awning-tricycle" not in VISDRONE_CLASSES

    def test_mapping_covers_all_classes(self):
        assert set(VISDRONE_TO_YOLO_CLASS.values()) == set(range(NUM_CLASSES))

    def test_mapping_keys_skip_tricycles(self):
        # VisDrone categories 7 (tricycle) and 8 (awning-tricycle) are dropped.
        assert set(VISDRONE_TO_YOLO_CLASS.keys()) == {1, 2, 3, 4, 5, 6, 9, 10}


class TestSceneType:
    def test_enum_values(self):
        assert SceneType.DAY == "day"
        assert SceneType.NIGHT == "night"
        assert SceneType.RAIN == "rain"

    def test_iterable(self):
        scenes = list(SceneType)
        assert len(scenes) == 3

    def test_str_enum(self):
        # StrEnum members should work as strings directly
        assert f"scene_{SceneType.DAY}" == "scene_day"


class TestImageExtensions:
    def test_common_formats(self):
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS

    def test_frozen(self):
        assert isinstance(IMAGE_EXTENSIONS, frozenset)


class TestImageNetConstants:
    def test_mean_length(self):
        assert len(IMAGENET_MEAN) == 3

    def test_std_length(self):
        assert len(IMAGENET_STD) == 3

    def test_values_in_range(self):
        for v in IMAGENET_MEAN + IMAGENET_STD:
            assert 0.0 < v < 1.0
