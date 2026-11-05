"""Oracle feasibility gate (dissertation §4.1).

Before building a cascade, ask whether ANY escalation policy could pay off:
rank frames by the realised per-frame gain of the ensemble over tier 1 and
escalate the top fraction p. If even this label-informed reference does not
beat the strongest uniform configuration at the same average budget, no
realizable signal will. Everything here is fusion/aggregation over cached
predictions — no inference.

The gate's summary statistic — the fraction of frames with a positive gain —
predicts the verdict: dense gain (75–90 % on VisDrone) closes the cascade,
sparse gain (41 % on UAVDT) opens it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .calibrate import p_grid, tau_for_fraction
from .policy import average_budget

MixtureScore = Callable[[np.ndarray], float]
"""score(escalate_mask) -> quality of the mixture on the whole evaluation set."""


def gate_stats(delta: Sequence[float]) -> dict:
    """Density and size of the per-frame ensemble gain (NaN = frame without GT)."""
    d = np.asarray(delta, dtype=float)
    d = d[~np.isnan(d)]
    return {
        "n": int(d.size),
        "frac_positive": float((d > 0).mean()) if d.size else float("nan"),
        "frac_negative": float((d < 0).mean()) if d.size else float("nan"),
        "mean_gain": float(d.mean()) if d.size else float("nan"),
    }


def _curve(order_key: np.ndarray, score: MixtureScore, grid, b1, b2) -> list[dict]:
    n = len(order_key)
    out = []
    for p in grid:
        if p <= 0:
            mask = np.zeros(n, dtype=bool)
        elif p >= 1:
            mask = np.ones(n, dtype=bool)
        else:
            mask = order_key >= tau_for_fraction(order_key, p)
        row = {"p": float(p), "score": float(score(mask)), "realized": float(mask.mean())}
        if b1 is not None and b2 is not None:
            row["budget"] = average_budget(row["realized"], b1, b2)
        out.append(row)
    return out


def oracle_curve(
    delta: Sequence[float], score: MixtureScore, grid=None, b1=None, b2=None
) -> list[dict]:
    """Greedy per-frame oracle: escalate frames in decreasing order of gain.

    NaN gains (frames without ground truth) are ranked lowest. This curve is a
    reference, not an upper bound of a non-additive metric such as mAP (see
    §4.1); a block-greedy refinement over the pool-level metric lies above it.
    """
    d = np.asarray(delta, dtype=float)
    key = np.where(np.isnan(d), -np.inf, d)
    return _curve(key, score, p_grid() if grid is None else grid, b1, b2)


def signal_curve(
    signals: Sequence[float], score: MixtureScore, grid=None, b1=None, b2=None
) -> list[dict]:
    """Quality vs. escalation fraction for a realizable signal (higher = escalate first)."""
    return _curve(
        np.asarray(signals, dtype=float), score, p_grid() if grid is None else grid, b1, b2
    )


def random_curve(
    n: int,
    score: MixtureScore,
    rng: np.random.Generator,
    grid=None,
    draws: int = 3,
    b1=None,
    b2=None,
) -> list[dict]:
    """Random-escalation control averaged over ``draws`` draws per fraction."""
    grid = p_grid() if grid is None else grid
    out = []
    for p in grid:
        k = int(round(p * n))
        vals = []
        for _ in range(draws):
            mask = np.zeros(n, dtype=bool)
            mask[rng.choice(n, size=k, replace=False)] = True
            vals.append(float(score(mask)))
        row = {"p": float(p), "score": float(np.mean(vals)), "realized": k / n}
        if b1 is not None and b2 is not None:
            row["budget"] = average_budget(row["realized"], b1, b2)
        out.append(row)
    return out
