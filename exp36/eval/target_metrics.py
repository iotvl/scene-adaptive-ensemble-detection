"""Operating-point metrics of the attacked class (spec stage 4).

    target_recall  = |detected by the configuration| / |GT objects of the class|
    ASR            = |detected by clean \\ detected by poisoned| / |detected by clean|
    FPPI delta     = (FP_poisoned - FP_clean) / n_images          (A2 only)

All at the DEPLOY threshold (0.25) with IoU 0.5 matching
"detected" is the
set of (frame, gt index) pairs, so ASR is a paired, object-wise quantity.
"""

from __future__ import annotations

from exp36.eval.matching import ConfigStats


def target_recall(stats: ConfigStats, cls: int, thr: float) -> float:
    return stats.recall(cls, thr)


def attack_success_rate(clean: ConfigStats, poisoned: ConfigStats, cls: int, thr: float) -> float:
    d_clean = clean.detected(cls, thr)
    if not d_clean:
        return float("nan")
    d_pois = poisoned.detected(cls, thr)
    return len(d_clean - d_pois) / len(d_clean)


def fppi_delta(clean: ConfigStats, poisoned: ConfigStats, cls: int, thr: float) -> float:
    return (poisoned.false_positives(cls, thr) - clean.false_positives(cls, thr)) / clean.n_images
