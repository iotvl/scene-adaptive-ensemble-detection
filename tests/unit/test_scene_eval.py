"""Unit tests for src/ensemble/scene_eval.py disk cache + overlap helpers."""

import numpy as np
import pytest

from src.constants import VISDRONE_TO_YOLO_CLASS
from src.ensemble.scene_eval import (
    cache_path,
    detr_label_mapping,
    filter_by_conf,
    load_predictions,
    prediction_overlap,
    save_predictions,
)


def _pred(boxes, labels, scores=None):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    labels = np.asarray(labels, dtype=int)
    if scores is None:
        scores = np.ones(len(labels))
    return {"boxes": boxes, "scores": np.asarray(scores, dtype=float), "labels": labels}


class TestPredictionCacheRoundTrip:
    def test_round_trip(self, tmp_path):
        preds = [
            _pred([[0, 0, 10, 10], [5, 5, 20, 20]], [0, 3], [0.9, 0.4]),
            {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=int)},
        ]
        img_whs = [(1920, 1080), (640, 480)]
        path = tmp_path / "m_day_val.npz"

        save_predictions(path, preds, img_whs)
        loaded, loaded_whs = load_predictions(path)

        assert loaded_whs == img_whs
        assert len(loaded) == 2
        np.testing.assert_allclose(loaded[0]["boxes"], preds[0]["boxes"])
        np.testing.assert_allclose(loaded[0]["scores"], preds[0]["scores"])
        np.testing.assert_array_equal(loaded[0]["labels"], preds[0]["labels"])
        assert loaded[1]["boxes"].shape == (0, 4)
        assert loaded[1]["labels"].size == 0

    def test_cache_path_encodes_protocol(self, tmp_path):
        p1 = cache_path(tmp_path, "yolov9e_vd", "day", "val", 0.001, 1280)
        p2 = cache_path(tmp_path, "yolov9e_vd", "day", "val", 0.25, 640)
        assert p1 != p2
        assert "yolov9e_vd" in p1.name and "day" in p1.name and "val" in p1.name


class TestPredictionOverlap:
    def test_identical_predictions(self):
        a = _pred([[0, 0, 10, 10]], [2])
        assert prediction_overlap(a, a) == 1.0

    def test_disjoint_boxes(self):
        a = _pred([[0, 0, 10, 10]], [2])
        b = _pred([[100, 100, 110, 110]], [2])
        assert prediction_overlap(a, b) == 0.0

    def test_same_box_different_class_is_no_match(self):
        a = _pred([[0, 0, 10, 10]], [2])
        b = _pred([[0, 0, 10, 10]], [3])
        assert prediction_overlap(a, b) == 0.0

    def test_both_empty_is_none(self):
        e = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=int)}
        assert prediction_overlap(e, e) is None

    def test_partial_overlap(self):
        a = _pred([[0, 0, 10, 10], [50, 50, 60, 60]], [1, 1])
        b = _pred([[0, 0, 10, 10]], [1])
        # matched: 1 of a's 2 + 1 of b's 1 -> 2 / 3
        assert prediction_overlap(a, b) == pytest.approx(2 / 3)


class TestDetrLabelMapping:
    def test_named_zero_indexed_10class(self):
        id2label = {
            0: "pedestrian",
            1: "people",
            2: "bicycle",
            3: "car",
            4: "van",
            5: "truck",
            6: "tricycle",
            7: "awning-tricycle",
            8: "bus",
            9: "motor",
        }
        m = detr_label_mapping(id2label)
        assert m == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 8: 6, 9: 7}  # tricycles dropped

    def test_named_visdrone_categories(self):
        id2label = {
            1: "pedestrian",
            2: "people",
            3: "bicycle",
            4: "car",
            5: "van",
            6: "truck",
            9: "bus",
            10: "motor",
        }
        assert detr_label_mapping(id2label) == VISDRONE_TO_YOLO_CLASS

    def test_generic_labels_fall_back_to_visdrone_categories(self):
        id2label = {i: f"LABEL_{i}" for i in range(10)}
        assert detr_label_mapping(id2label) == VISDRONE_TO_YOLO_CLASS

    def test_string_keys_and_name_normalisation(self):
        id2label = {str(i): n.upper() for i, n in enumerate(
            ["pedestrian", "people", "bicycle", "car", "van", "truck", "bus", "motor"]
        )}
        id2label["3"] = "Car"
        m = detr_label_mapping(id2label)
        assert m == {i: i for i in range(8)}

    def test_partial_name_match_falls_back(self):
        # Only 2 recognizable names -> different taxonomy, trust the id fallback.
        id2label = {0: "car", 1: "bus", 2: "boat", 3: "plane"}
        assert detr_label_mapping(id2label) == VISDRONE_TO_YOLO_CLASS

    def test_empty_falls_back(self):
        assert detr_label_mapping({}) == VISDRONE_TO_YOLO_CLASS


def test_filter_by_conf():
    p = _pred([[0, 0, 1, 1], [1, 1, 2, 2]], [0, 1], [0.9, 0.1])
    f = filter_by_conf(p, 0.5)
    assert len(f["boxes"]) == 1
    assert f["labels"].tolist() == [0]
