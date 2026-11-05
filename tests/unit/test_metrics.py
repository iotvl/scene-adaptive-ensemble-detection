"""Tests for src/evaluation/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_ap,
    compute_iou,
    compute_iou_matrix,
    remap_10class_to_8class,
)


class TestRemap10to8:
    def _preds(self, labels):
        n = len(labels)
        boxes = np.tile([0, 0, 10, 10], (n, 1)).astype(float)
        scores = np.linspace(0.9, 0.1, n)
        return boxes, scores, np.array(labels, dtype=int)

    def test_maps_bus_motor_and_drops_tricycle(self):
        # 10-class ids: 0..5 identity, 6=tricycle, 7=awning, 8=bus, 9=motor
        boxes, scores, labels = self._preds([0, 3, 6, 7, 8, 9])
        b, s, lbl = remap_10class_to_8class(boxes, scores, labels)
        # tricycle(6)/awning(7) dropped; 8->6 (bus), 9->7 (motor)
        assert list(lbl) == [0, 3, 6, 7]
        assert len(b) == len(s) == 4

    def test_empty(self):
        b, s, lbl = remap_10class_to_8class(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
        assert len(lbl) == 0

    def test_all_dropped(self):
        boxes, scores, labels = self._preds([6, 7])
        b, s, lbl = remap_10class_to_8class(boxes, scores, labels)
        assert len(lbl) == 0 and b.shape == (0, 4)


class TestComputeIoU:
    def test_identical_boxes(self):
        box = np.array([0, 0, 100, 100], dtype=float)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        box1 = np.array([0, 0, 50, 50], dtype=float)
        box2 = np.array([100, 100, 200, 200], dtype=float)
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        box1 = np.array([0, 0, 100, 100], dtype=float)
        box2 = np.array([50, 50, 150, 150], dtype=float)
        # intersection = 50*50 = 2500, union = 10000+10000-2500 = 17500
        assert compute_iou(box1, box2) == pytest.approx(2500 / 17500)

    def test_contained(self):
        outer = np.array([0, 0, 100, 100], dtype=float)
        inner = np.array([25, 25, 75, 75], dtype=float)
        # inter = 2500, union = 10000
        assert compute_iou(outer, inner) == pytest.approx(2500 / 10000)

    def test_zero_area(self):
        box1 = np.array([0, 0, 0, 0], dtype=float)
        box2 = np.array([0, 0, 100, 100], dtype=float)
        assert compute_iou(box1, box2) == pytest.approx(0.0)


class TestComputeIoUMatrix:
    def test_self_iou(self):
        boxes = np.array(
            [
                [0, 0, 100, 100],
                [200, 200, 300, 300],
            ],
            dtype=float,
        )
        mat = compute_iou_matrix(boxes, boxes)
        assert mat.shape == (2, 2)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[1, 1] == pytest.approx(1.0)
        assert mat[0, 1] == pytest.approx(0.0)

    def test_asymmetric_shapes(self):
        a = np.array([[0, 0, 50, 50]], dtype=float)
        b = np.array([[0, 0, 50, 50], [100, 100, 200, 200]], dtype=float)
        mat = compute_iou_matrix(a, b)
        assert mat.shape == (1, 2)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[0, 1] == pytest.approx(0.0)


class TestComputeAP:
    def test_perfect_detector(self):
        recall = np.array([0.5, 1.0])
        precision = np.array([1.0, 1.0])
        assert compute_ap(recall, precision) == pytest.approx(1.0)

    def test_empty(self):
        recall = np.array([])
        precision = np.array([])
        # With no detections, AP should be very low
        ap = compute_ap(recall, precision)
        assert 0.0 <= ap <= 1.0
