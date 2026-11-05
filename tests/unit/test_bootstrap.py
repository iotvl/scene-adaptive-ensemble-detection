"""Unit tests for src/evaluation/bootstrap.py.

The load-bearing property is exactness: SceneClassStats.map50 with unit counts
must reproduce compute_map's mAP50 bit-for-bit, and duplicated counts must be
equivalent to duplicating the image in a fresh compute_map call. Everything the
bootstrap reports rests on those two identities.
"""

import numpy as np
import pytest

from src.evaluation.bootstrap import (
    SceneClassStats,
    _fast_ap,
    match_image_by_class,
    paired_bootstrap_map50,
    summarize,
)
from src.evaluation.metrics import compute_ap, compute_map

NUM_CLASSES = 8


def _random_scene(rng, n_images, max_pred=30, max_gt=15):
    """Random predictions/GT in the compute_map contract (xyxy abs pixels)."""
    preds, gts = [], []
    for i in range(n_images):
        n_p = int(rng.integers(0, max_pred))
        n_g = int(rng.integers(0, max_gt))

        def boxes(n):
            xy = rng.uniform(0, 900, (n, 2))
            wh = rng.uniform(5, 120, (n, 2))
            return np.concatenate([xy, xy + wh], axis=1)

        preds.append(
            {
                "image_id": i,
                "boxes": boxes(n_p),
                "scores": rng.uniform(0.001, 1.0, n_p),
                "labels": rng.integers(0, NUM_CLASSES, n_p),
            }
        )
        gts.append(
            {
                "image_id": i,
                "boxes": boxes(n_g),
                "labels": rng.integers(0, NUM_CLASSES, n_g),
            }
        )
    return preds, gts


def _overlapping_scene(rng, n_images):
    """Scene where predictions are GT boxes plus jitter, so TPs actually occur."""
    preds, gts = [], []
    for i in range(n_images):
        n_g = int(rng.integers(1, 12))
        xy = rng.uniform(0, 900, (n_g, 2))
        wh = rng.uniform(20, 120, (n_g, 2))
        gt_boxes = np.concatenate([xy, xy + wh], axis=1)
        labels = rng.integers(0, NUM_CLASSES, n_g)
        jitter = rng.uniform(-6, 6, (n_g, 4))
        preds.append(
            {
                "image_id": i,
                "boxes": gt_boxes + jitter,
                "scores": rng.uniform(0.05, 1.0, n_g),
                "labels": labels.copy(),
            }
        )
        gts.append({"image_id": i, "boxes": gt_boxes, "labels": labels})
    return preds, gts


class TestFastAp:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_compute_ap(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 200))
        tp = rng.random(n) < 0.4
        total_gt = int(tp.sum()) + int(rng.integers(1, 20))
        tp_cumsum = np.cumsum(tp.astype(float))
        recall = tp_cumsum / total_gt
        precision = tp_cumsum / np.arange(1, n + 1)
        assert _fast_ap(recall, precision) == pytest.approx(
            compute_ap(recall, precision), abs=1e-12
        )


class TestIdentityWithComputeMap:
    @pytest.mark.parametrize("maker", [_random_scene, _overlapping_scene])
    def test_unit_counts_reproduce_map50(self, maker):
        rng = np.random.default_rng(7)
        preds, gts = maker(rng, 12)
        stats = SceneClassStats(preds, gts, iou_threshold=0.5, num_classes=NUM_CLASSES)
        expected = compute_map(preds, gts, iou_thresholds=[0.5], num_classes=NUM_CLASSES)["mAP50"]
        assert stats.map50(np.ones(12, dtype=int)) == pytest.approx(expected, abs=1e-12)

    def test_duplicated_counts_equal_duplicated_images(self):
        rng = np.random.default_rng(3)
        preds, gts = _overlapping_scene(rng, 6)
        counts = np.array([2, 0, 1, 3, 0, 1])

        stats = SceneClassStats(preds, gts, num_classes=NUM_CLASSES)
        got = stats.map50(counts)

        dup_preds, dup_gts = [], []
        new_id = 0
        for i, c in enumerate(counts):
            for _ in range(c):
                dup_preds.append({**preds[i], "image_id": new_id})
                dup_gts.append({**gts[i], "image_id": new_id})
                new_id += 1
        expected = compute_map(dup_preds, dup_gts, iou_thresholds=[0.5], num_classes=NUM_CLASSES)[
            "mAP50"
        ]
        assert got == pytest.approx(expected, abs=1e-12)


class TestMatchImageByClass:
    def test_perfect_predictions_all_tp(self):
        gt = {
            "boxes": np.array([[0, 0, 10, 10], [50, 50, 80, 90]], dtype=float),
            "labels": np.array([0, 3]),
        }
        pred = {
            "boxes": gt["boxes"].copy(),
            "scores": np.array([0.9, 0.8]),
            "labels": gt["labels"].copy(),
        }
        out = match_image_by_class(pred, gt, num_classes=NUM_CLASSES)
        assert out[0][1].tolist() == [True] and out[0][2] == 1
        assert out[3][1].tolist() == [True] and out[3][2] == 1
        assert out[1][2] == 0 and len(out[1][0]) == 0

    def test_duplicate_prediction_second_is_fp(self):
        gt = {"boxes": np.array([[0, 0, 10, 10]], dtype=float), "labels": np.array([0])}
        pred = {
            "boxes": np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float),
            "scores": np.array([0.5, 0.9]),
            "labels": np.array([0, 0]),
        }
        out = match_image_by_class(pred, gt, num_classes=NUM_CLASSES)
        # Higher-scored duplicate matches first; the other is a false positive.
        assert out[0][0].tolist() == [0.9, 0.5]
        assert out[0][1].tolist() == [True, False]


class TestPairedBootstrap:
    def test_paired_replicates_and_summary(self):
        rng = np.random.default_rng(11)
        preds_a, gts = _overlapping_scene(rng, 10)
        # Config B = config A with every score halved: identical ranking,
        # identical mAP50, so every paired delta must be exactly zero.
        preds_b = [{**p, "scores": p["scores"] * 0.5} for p in preds_a]
        stats = {
            "a": SceneClassStats(preds_a, gts, num_classes=NUM_CLASSES),
            "b": SceneClassStats(preds_b, gts, num_classes=NUM_CLASSES),
        }
        reps = paired_bootstrap_map50(stats, n_boot=25, rng=np.random.default_rng(0))
        assert reps["a"].shape == (25,)
        np.testing.assert_allclose(reps["a"], reps["b"], atol=1e-12)

        s = summarize(reps["a"])
        assert s["ci_lo"] <= s["mean"] <= s["ci_hi"]

    def test_mismatched_image_counts_rejected(self):
        rng = np.random.default_rng(5)
        preds, gts = _overlapping_scene(rng, 4)
        stats = {
            "a": SceneClassStats(preds, gts, num_classes=NUM_CLASSES),
            "b": SceneClassStats(preds[:3], gts[:3], num_classes=NUM_CLASSES),
        }
        with pytest.raises(AssertionError):
            paired_bootstrap_map50(stats, n_boot=2, rng=np.random.default_rng(0))
