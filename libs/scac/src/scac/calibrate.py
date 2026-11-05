"""Threshold calibration without hand tuning and without data leakage.

The threshold ``tau`` is never chosen directly: for each escalation fraction
``p`` on a grid, ``tau`` is the (1 - p)-quantile of the signal on the
calibration set, the quality of the resulting mixture is scored, and the best
``p*`` wins. ``crossfit`` does this on two folds split at the GROUP level
(video sequences), so every frame is escalated under a threshold its own
sequence never influenced. This reproduces scripts/12c exactly (same fold
construction, same quantile rule, same tie-breaking: first best on the grid).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

ArrayLike = Any  # anything np.asarray accepts (list, tuple, ndarray)

ScoreFn = Callable[[np.ndarray, np.ndarray], float]
"""score_fn(escalate_mask, eval_indices) -> quality of the mixture (higher is better).

``escalate_mask`` is a boolean array over ALL frames (the mixture is formed on
all frames because the metric may be non-additive), ``eval_indices`` says which
frames the score is computed on (the calibration fold, or everything).
"""


def p_grid(step: float = 0.05) -> list[float]:
    return [round(float(p), 2) for p in np.arange(0.0, 1.0 + 1e-9, step)]


def tau_for_fraction(signals: np.ndarray, p: float) -> float:
    """Threshold that escalates (about) a fraction p of the given frames."""
    return float(np.quantile(signals, 1.0 - p)) if p > 0 else float("inf")


def choose_threshold(
    signals: ArrayLike,
    score_fn: ScoreFn,
    cal_indices: ArrayLike | None = None,
    grid: Sequence[float] | None = None,
) -> dict:
    """Pick p* (and its tau) maximising ``score_fn`` on the calibration frames.

    Returns ``{"p": p*, "tau": tau, "cal_score": best}``. The first grid value
    reaching the maximum wins (ties favour the cheaper cascade).
    """
    sig = np.asarray(signals, dtype=float)
    n = len(sig)
    cal = np.arange(n) if cal_indices is None else np.asarray(cal_indices, dtype=int)
    grid = p_grid() if grid is None else list(grid)
    best_p, best_tau, best_val = 0.0, float("inf"), -np.inf
    for p in grid:
        tau = tau_for_fraction(sig[cal], p)
        val = float(score_fn(sig >= tau, cal))
        if val > best_val:
            best_p, best_tau, best_val = float(p), tau, val
    return {"p": best_p, "tau": best_tau, "cal_score": best_val}


def stratified_folds(
    group_of: Sequence, scene_of: Sequence, rng: np.random.Generator, n_folds: int = 2
) -> np.ndarray:
    """Fold id per frame: groups (sequences) are shuffled within each scene and
    dealt round-robin into folds, so every fold holds whole groups and the
    scene proportions are preserved. Scene iteration order = first appearance.
    """
    group_of = list(group_of)
    scene_of = list(scene_of)
    scenes = list(dict.fromkeys(scene_of))
    fold_of_group: dict = {}
    for scene in scenes:
        groups = sorted({g for g, s in zip(group_of, scene_of) if s == scene})
        order = rng.permutation(len(groups))
        for k, j in enumerate(order):
            fold_of_group[groups[j]] = k % n_folds
    return np.array([fold_of_group[g] for g in group_of], dtype=int)


def crossfit(
    signals: ArrayLike,
    fold_of: ArrayLike,
    score_fn: ScoreFn,
    grid: Sequence[float] | None = None,
) -> dict:
    """Two-fold cross-fitted calibration.

    Chooses (p*, tau) on each fold with :func:`choose_threshold`, then
    escalates every frame under the OTHER fold's tau. Returns
    ``{"folds": {0: {...}, 1: {...}}, "escalate": mask, "realized": fraction}``.
    """
    sig = np.asarray(signals, dtype=float)
    folds_arr = np.asarray(fold_of, dtype=int)
    folds = sorted(set(folds_arr.tolist()))
    if len(folds) != 2:
        raise ValueError("crossfit expects exactly two folds")
    chosen = {}
    for f in folds:
        cal = np.flatnonzero(folds_arr == f)
        chosen[f] = choose_threshold(sig, score_fn, cal, grid)
    other = {folds[0]: folds[1], folds[1]: folds[0]}
    esc = np.array([sig[i] >= chosen[other[folds_arr[i]]]["tau"] for i in range(len(sig))])
    return {"folds": chosen, "escalate": esc, "realized": float(esc.mean())}


def realized_by_scene(escalate: np.ndarray, scene_of: Sequence) -> dict:
    scenes = np.asarray(list(scene_of))
    return {s: float(escalate[scenes == s].mean()) for s in dict.fromkeys(scenes.tolist())}
