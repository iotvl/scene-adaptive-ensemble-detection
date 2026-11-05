"""Per-frame, per-class matching of fused boxes to GT, and statistics that can
be re-aggregated under any image resample without re-matching.

Mirrors src/evaluation/bootstrap.py (match_image_by_class / SceneClassStats)
but keeps, per prediction, the index of the matched GT object (needed for the
object-wise ASR) and TP flags at the ten COCO IoU thresholds (for mAP@50-95).
`ConfigStats.map50(ones)` == `compute_map(...)["mAP50"]` on the same fused
boxes — asserted in tests/test_metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.bootstrap import _fast_ap
from src.evaluation.metrics import compute_iou_matrix

IOU_THRS = np.round(np.arange(0.5, 1.0, 0.05), 2)  # 10 COCO thresholds, index 0 == 0.5


@dataclass
class FrameClassMatch:
    """Fused boxes of one class on one frame, matched to that frame's GT."""

    scores: np.ndarray  # (n,) float32, sorted desc (stable)
    tp10: np.ndarray  # (n, 10) bool, TP at IOU_THRS
    gt_idx: np.ndarray  # (n,) int32, matched GT index at IoU 0.5, -1 = FP
    n_gt: int

    @property
    def n(self) -> int:
        return len(self.scores)


def match_frame_class(
    fboxes: np.ndarray, fscores: np.ndarray, gt_boxes: np.ndarray
) -> FrameClassMatch:
    """Greedy score-descending matching (same order as compute_per_class_ap):
    each prediction takes the unmatched GT with the highest IoU if it clears
    the threshold. Done independently at every IoU threshold."""
    fscores = np.asarray(fscores, dtype=float).reshape(-1)
    fboxes = np.asarray(fboxes, dtype=float).reshape(-1, 4)
    order = np.argsort(-fscores, kind="stable")
    fscores, fboxes = fscores[order], fboxes[order]
    n, g = len(fscores), len(gt_boxes)
    tp10 = np.zeros((n, len(IOU_THRS)), dtype=bool)
    gt_idx = np.full(n, -1, dtype=np.int32)
    if n and g:
        ious = compute_iou_matrix(fboxes, np.asarray(gt_boxes, dtype=float).reshape(-1, 4))
        for t, thr in enumerate(IOU_THRS):
            used = np.zeros(g, dtype=bool)
            for i in range(n):
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= thr and not used[j]:
                    used[j] = True
                    tp10[i, t] = True
                    if t == 0:
                        gt_idx[i] = j
    return FrameClassMatch(fscores.astype(np.float32), tp10, gt_idx, int(g))


class ClassRun:
    """One (scene, method, class, variant): FrameClassMatch for every frame,
    stored flat for fast npz round-trips."""

    def __init__(self, frames: list[FrameClassMatch]):
        self.n_images = len(frames)
        self.lengths = np.array([f.n for f in frames], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.scores = (
            np.concatenate([f.scores for f in frames]) if frames else np.zeros(0, np.float32)
        )
        self.tp10 = np.concatenate([f.tp10 for f in frames]) if frames else np.zeros((0, 10), bool)
        self.gt_idx = (
            np.concatenate([f.gt_idx for f in frames]) if frames else np.zeros(0, np.int32)
        )
        self.n_gt = np.array([f.n_gt for f in frames], dtype=np.int64)

    def frame(self, i: int) -> FrameClassMatch:
        a, b = self.offsets[i], self.offsets[i + 1]
        return FrameClassMatch(
            self.scores[a:b], self.tp10[a:b], self.gt_idx[a:b], int(self.n_gt[i])
        )

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            lengths=self.lengths,
            scores=self.scores,
            tp10=np.packbits(self.tp10, axis=1),
            gt_idx=self.gt_idx,
            n_gt=self.n_gt,
        )

    @classmethod
    def load(cls, path) -> ClassRun:
        z = np.load(path)
        run = cls.__new__(cls)
        run.lengths = z["lengths"]
        run.n_images = len(run.lengths)
        run.offsets = np.concatenate([[0], np.cumsum(run.lengths)])
        run.scores = z["scores"]
        run.tp10 = np.unpackbits(z["tp10"], axis=1, count=len(IOU_THRS)).astype(bool)
        run.gt_idx = z["gt_idx"]
        run.n_gt = z["n_gt"]
        return run


class ConfigStats:
    """A full configuration on one scene: per class, the frames' matches
    (possibly mixed from several ClassRuns frame by frame, e.g. rho < 1).

    Built once per configuration
    every metric is then a cheap aggregation,
    and bootstrap replicates only pass different `counts` vectors."""

    def __init__(self, per_class_frames: list[list[FrameClassMatch]]):
        self.num_classes = len(per_class_frames)
        self.n_images = len(per_class_frames[0])
        self._scores, self._tp10, self._gt_idx, self._img = [], [], [], []
        self.ngt = np.zeros((self.num_classes, self.n_images), dtype=np.int64)
        for c, frames in enumerate(per_class_frames):
            scores = np.concatenate([f.scores for f in frames])
            tp10 = np.concatenate([f.tp10 for f in frames])
            gt_idx = np.concatenate([f.gt_idx for f in frames])
            img = np.concatenate([np.full(f.n, i, dtype=np.int32) for i, f in enumerate(frames)])
            self.ngt[c] = [f.n_gt for f in frames]
            order = np.argsort(-scores, kind="stable")  # one global stable sort per class
            self._scores.append(scores[order])
            self._tp10.append(tp10[order])
            self._gt_idx.append(gt_idx[order])
            self._img.append(img[order])

    # ---- mAP --------------------------------------------------------------
    def _ap(self, c: int, t: int, counts: np.ndarray) -> float:
        total_gt = int(self.ngt[c] @ counts)
        tp = np.repeat(self._tp10[c][:, t], counts[self._img[c]])
        if total_gt == 0 or len(tp) == 0:
            return 0.0  # compute_map convention: absent class scores 0
        cum = np.cumsum(tp)
        return _fast_ap(cum / total_gt, cum / np.arange(1, len(tp) + 1))

    def per_class_ap50(self, counts: np.ndarray | None = None) -> np.ndarray:
        counts = np.ones(self.n_images, dtype=int) if counts is None else counts
        return np.array([self._ap(c, 0, counts) for c in range(self.num_classes)])

    def map50(self, counts: np.ndarray | None = None) -> float:
        return float(self.per_class_ap50(counts).mean())

    def map5095(self, counts: np.ndarray | None = None) -> float:
        counts = np.ones(self.n_images, dtype=int) if counts is None else counts
        return float(
            np.mean(
                [
                    np.mean([self._ap(c, t, counts) for c in range(self.num_classes)])
                    for t in range(len(IOU_THRS))
                ]
            )
        )

    # ---- operating-point metrics (deploy threshold) --------------------------
    def detected(self, c: int, thr: float) -> set[tuple[int, int]]:
        """GT objects of class c hit by a prediction with score >= thr (IoU 0.5)."""
        m = (self._scores[c] >= thr) & (self._gt_idx[c] >= 0)
        return set(zip(self._img[c][m].tolist(), self._gt_idx[c][m].tolist()))

    def n_gt_total(self, c: int) -> int:
        return int(self.ngt[c].sum())

    def recall(self, c: int, thr: float) -> float:
        n = self.n_gt_total(c)
        return len(self.detected(c, thr)) / n if n else float("nan")

    def false_positives(self, c: int, thr: float) -> int:
        m = (self._scores[c] >= thr) & (~self._tp10[c][:, 0])
        return int(m.sum())

    def n_boxes_per_image(self) -> float:
        return float(sum(len(s) for s in self._scores) / self.n_images)
