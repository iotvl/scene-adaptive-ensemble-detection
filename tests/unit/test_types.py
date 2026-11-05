"""Tests for src/types.py."""

from __future__ import annotations

from src.constants import SceneType
from src.types import DetectionResult, EvalMetrics, ProfileResult, ScenePrediction


class TestDetectionResult:
    def test_basic_properties(self, sample_boxes, sample_scores, sample_labels):
        det = DetectionResult(
            boxes=sample_boxes,
            scores=sample_scores,
            labels=sample_labels,
        )
        assert det.num_detections == 2
        assert not det.is_empty

    def test_empty(self):
        det = DetectionResult.empty()
        assert det.num_detections == 0
        assert det.is_empty
        assert det.boxes.shape == (0, 4)
        assert det.scores.shape == (0,)
        assert det.labels.shape == (0,)


class TestScenePrediction:
    def test_fields(self):
        pred = ScenePrediction(
            scene=SceneType.NIGHT,
            confidence=0.95,
            features={"brightness": 40.0},
        )
        assert pred.scene == SceneType.NIGHT
        assert pred.confidence == 0.95
        assert "brightness" in pred.features

    def test_default_features(self):
        pred = ScenePrediction(scene=SceneType.DAY, confidence=0.8)
        assert pred.features == {}


class TestEvalMetrics:
    def test_defaults(self):
        m = EvalMetrics()
        assert m.map50 == 0.0
        assert m.f1 == 0.0
        assert m.per_class_ap == {}

    def test_custom_values(self):
        m = EvalMetrics(map50=0.42, precision=0.8, recall=0.7)
        assert m.map50 == 0.42


class TestProfileResult:
    def test_defaults(self):
        p = ProfileResult()
        assert p.model_name == ""
        assert p.fps == 0.0
        assert p.gpu_memory_mb is None
