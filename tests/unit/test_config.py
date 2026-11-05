"""Tests for src/config.py — Pydantic validated configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import (
    AppConfig,
    DatasetConfig,
    EnsembleSceneConfig,
    ScenesConfig,
    load_config,
)


class TestLoadConfig:
    def test_load_valid(self, tmp_yaml_config: Path):
        cfg = load_config(tmp_yaml_config)
        assert isinstance(cfg, AppConfig)
        assert cfg.project.name == "test"
        assert cfg.dataset.num_classes == 8

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")


class TestDatasetValidation:
    def test_class_count_mismatch(self):
        with pytest.raises(ValidationError, match="num_classes"):
            DatasetConfig(
                classes=["a", "b", "c"],
                num_classes=5,
            )

    def test_class_count_ok(self):
        dc = DatasetConfig(classes=["a", "b"], num_classes=2)
        assert dc.num_classes == 2


class TestScenesValidation:
    def test_threshold_order(self):
        with pytest.raises(ValidationError, match="brightness_night_threshold"):
            ScenesConfig(
                brightness_night_threshold=150,
                brightness_day_threshold=100,
            )

    def test_valid_thresholds(self):
        sc = ScenesConfig(
            brightness_night_threshold=60,
            brightness_day_threshold=130,
        )
        assert sc.brightness_night_threshold < sc.brightness_day_threshold


class TestEnsembleSceneValidation:
    def test_weights_models_mismatch(self):
        with pytest.raises(ValidationError, match="weights"):
            EnsembleSceneConfig(
                models=["a", "b"],
                weights=[1.0],
            )

    def test_valid(self):
        esc = EnsembleSceneConfig(
            models=["yolov8n", "yolov8s"],
            weights=[0.6, 0.4],
        )
        assert len(esc.weights) == len(esc.models)
