"""Tests for src/ensemble/methods.py — box fusion methods."""

from __future__ import annotations

import numpy as np
import pytest

from src.ensemble.methods import (
    apply_nms,
    apply_wbf,
    denormalize_boxes,
    fusion_extra_kwargs,
    get_ensemble_method,
    normalize_boxes,
)


class TestNormalisation:
    def test_normalize_and_denormalize_roundtrip(self):
        boxes = np.array([[100.0, 200.0, 300.0, 400.0]])
        w, h = 640, 480
        norm = normalize_boxes(boxes, w, h)
        assert norm[0, 0] == pytest.approx(100 / 640)
        assert norm[0, 1] == pytest.approx(200 / 480)
        back = denormalize_boxes(norm, w, h)
        np.testing.assert_allclose(back, boxes, atol=1e-6)

    def test_clipping(self):
        boxes = np.array([[-10.0, -10.0, 700.0, 500.0]])
        norm = normalize_boxes(boxes, 640, 480)
        assert np.all(norm >= 0)
        assert np.all(norm <= 1)


class TestApplyWBF:
    def test_single_model(self):
        boxes_list = [np.array([[100.0, 100.0, 200.0, 200.0]])]
        scores_list = [np.array([0.9])]
        labels_list = [np.array([0])]
        fused_boxes, fused_scores, fused_labels = apply_wbf(
            boxes_list, scores_list, labels_list, img_size=(640, 640)
        )
        assert len(fused_scores) == 1
        assert fused_labels[0] == 0
        assert fused_scores[0] > 0

    def test_empty_input(self):
        boxes_list = [np.zeros((0, 4))]
        scores_list = [np.zeros(0)]
        labels_list = [np.zeros(0, dtype=int)]
        fused_boxes, fused_scores, fused_labels = apply_wbf(
            boxes_list, scores_list, labels_list, img_size=(640, 640)
        )
        assert len(fused_scores) == 0


class TestApplyNMS:
    def test_single_model(self):
        boxes_list = [np.array([[50.0, 50.0, 150.0, 150.0]])]
        scores_list = [np.array([0.85])]
        labels_list = [np.array([2])]
        fused_boxes, fused_scores, fused_labels = apply_nms(
            boxes_list, scores_list, labels_list, img_size=(640, 640)
        )
        assert len(fused_scores) >= 1
        assert fused_labels[0] == 2


class TestFusionExtraKwargs:
    """Each method accepts a different subset of tuning params (audit C2/C3)."""

    def test_wbf_and_nmw_get_skip_box_thr(self):
        for method in ("wbf", "nmw"):
            kw = fusion_extra_kwargs(method, skip_box_thr=0.2, sigma=0.5)
            assert kw == {"skip_box_thr": 0.2}

    def test_soft_nms_gets_sigma_and_thresh(self):
        kw = fusion_extra_kwargs("soft_nms", skip_box_thr=0.2, sigma=0.7)
        assert kw == {"sigma": 0.7, "thresh": 0.2}

    def test_nms_gets_nothing(self):
        assert fusion_extra_kwargs("nms", skip_box_thr=0.2, sigma=0.7) == {}

    def test_routed_kwargs_are_accepted_by_each_method(self):
        # Forwarding the wrong kwarg would TypeError; this is the regression guard.
        boxes_list = [np.array([[100.0, 100.0, 200.0, 200.0]])]
        scores_list = [np.array([0.9])]
        labels_list = [np.array([0])]
        for method in ("wbf", "nms", "soft_nms", "nmw"):
            fn = get_ensemble_method(method)
            fn(
                boxes_list,
                scores_list,
                labels_list,
                weights=[1.0],
                iou_thr=0.55,
                img_size=(640, 640),
                **fusion_extra_kwargs(method, skip_box_thr=0.01, sigma=0.5),
            )


class TestGetEnsembleMethod:
    def test_known_methods(self):
        for name in ("wbf", "nms", "soft_nms", "nmw"):
            fn = get_ensemble_method(name)
            assert callable(fn)

    def test_unknown_method(self):
        get_ensemble_method.cache_clear()
        with pytest.raises(ValueError, match="Unknown ensemble method"):
            get_ensemble_method("invalid_method_xyz")
