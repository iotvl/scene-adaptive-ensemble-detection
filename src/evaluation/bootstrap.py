"""Paired bootstrap confidence intervals for mAP50 over precomputed detections.

compute_per_class_ap's greedy matching is per-image independent: the global
score sort only fixes the order in which each image's own predictions are
matched (score-descending, ties in original order), and matching consumes only
that image's ground truth. So TP flags can be computed once per image and a
bootstrap replicate (images resampled with replacement) only re-aggregates
flags — no re-matching, no re-fusion. `SceneClassStats.map50(counts)` with unit
counts reproduces `compute_map(...)["mAP50"]` exactly; callers should assert
that identity once per configuration before trusting replicate values.

Replicates are paired by sharing one resample-count vector across all
configurations of a scene, so deltas between configurations are computed on
identical image multisets.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_iou_matrix


def _fast_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Vectorized twin of metrics.compute_ap (COCO 101-point interpolation).

    np.maximum.accumulate replaces the backward max loop and searchsorted
    replaces the per-point np.where scan; results are bit-identical because
    mrec is non-decreasing.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    recall_points = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, recall_points, side="left")
    return float(mpre[idx].mean())


def match_image_by_class(
    pred: dict, gt: dict, iou_threshold: float = 0.5, num_classes: int = 8
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Greedy-match one image's predictions to its GT, per class.

    Mirrors compute_per_class_ap's matching order exactly: predictions are
    processed score-descending with ties kept in original order (the global
    sort there is stable, so within an image the order is the same).

    Returns per class: (scores sorted desc, tp flags in that order, n_gt).
    """
    boxes, scores, labels = pred["boxes"], pred["scores"], pred["labels"]
    gt_boxes, gt_labels = gt["boxes"], gt["labels"]
    out = []
    for cls_id in range(num_classes):
        pmask = labels == cls_id
        cls_scores = scores[pmask]
        order = np.argsort(-cls_scores, kind="stable")
        cls_scores = cls_scores[order]
        cls_boxes = boxes[pmask][order]
        g = gt_boxes[gt_labels == cls_id]
        tp = np.zeros(len(cls_scores), dtype=bool)
        if len(g) > 0 and len(cls_scores) > 0:
            used = np.zeros(len(g), dtype=bool)
            ious = compute_iou_matrix(cls_boxes, g)
            for i in range(len(cls_scores)):
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= iou_threshold and not used[j]:
                    used[j] = True
                    tp[i] = True
        out.append((cls_scores, tp, int(len(g))))
    return out


class SceneClassStats:
    """Per-configuration match statistics for one scene, aggregatable under
    any image resample-count vector without re-matching."""

    def __init__(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
        iou_threshold: float = 0.5,
        num_classes: int = 8,
    ) -> None:
        self.num_classes = num_classes
        self.n_images = len(predictions)
        per_image = [
            match_image_by_class(p, g, iou_threshold, num_classes)
            for p, g in zip(predictions, ground_truths)
        ]
        self.ngt = np.zeros((num_classes, self.n_images), dtype=int)
        self._img_ids: list[np.ndarray] = []
        self._tp: list[np.ndarray] = []
        for c in range(num_classes):
            scores = np.concatenate([per_image[i][c][0] for i in range(self.n_images)])
            tp = np.concatenate([per_image[i][c][1] for i in range(self.n_images)])
            img_ids = np.concatenate(
                [np.full(len(per_image[i][c][0]), i, dtype=int) for i in range(self.n_images)]
            )
            self.ngt[c] = [per_image[i][c][2] for i in range(self.n_images)]
            # One global stable score sort per class, shared by every replicate.
            order = np.argsort(-scores, kind="stable")
            self._img_ids.append(img_ids[order])
            self._tp.append(tp[order])

    def map50(self, counts: np.ndarray) -> float:
        """mAP50 of the image multiset given by `counts` (occurrences per image).

        Unit counts == the full sample == compute_map's mAP50. Duplicated
        images just repeat their (already sorted) TP flags in place, which is
        equivalent to matching each copy independently.
        """
        aps = []
        for c in range(self.num_classes):
            total_gt = int(self.ngt[c] @ counts)
            tp = np.repeat(self._tp[c], counts[self._img_ids[c]])
            if total_gt == 0 or len(tp) == 0:
                aps.append(0.0)
                continue
            tp_cumsum = np.cumsum(tp)
            recall = tp_cumsum / total_gt
            precision = tp_cumsum / np.arange(1, len(tp) + 1)
            aps.append(_fast_ap(recall, precision))
        return float(np.mean(aps))


def paired_bootstrap_map50(
    stats_by_config: dict[str, SceneClassStats],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Bootstrap mAP50 replicates for every configuration of one scene.

    Each replicate draws one resample of the scene's images and evaluates all
    configurations on it (paired), so replicate-wise deltas are valid.
    Returns config -> (n_boot,) array of mAP50 values.
    """
    n = {s.n_images for s in stats_by_config.values()}
    assert len(n) == 1, f"configs disagree on image count: {n}"
    n_images = n.pop()
    out = {k: np.empty(n_boot) for k in stats_by_config}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_images, n_images), minlength=n_images)
        for key, stats in stats_by_config.items():
            out[key][b] = stats.map50(counts)
    return out


def summarize(values: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Percentile-CI summary of a replicate array."""
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std(ddof=1)), 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
    }
