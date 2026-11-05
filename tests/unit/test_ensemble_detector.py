"""Tests for src/ensemble/ensemble_detector.py — construction-time guards.

These cover the cheap, model-free validation path: the weight-count check runs
before any model weights are loaded, so no real checkpoints are needed (audit C5).
"""

from __future__ import annotations

import pytest

from src.ensemble.ensemble_detector import EnsembleDetector


class TestWeightCountValidation:
    def test_mismatched_weights_raise(self):
        with pytest.raises(ValueError, match="len.weights"):
            EnsembleDetector(model_paths=["a.pt", "b.pt"], weights=[1.0])

    def test_too_many_weights_raise(self):
        with pytest.raises(ValueError, match="len.weights"):
            EnsembleDetector(model_paths=["a.pt"], weights=[1.0, 2.0])
