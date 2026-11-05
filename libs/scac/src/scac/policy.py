"""The cascade itself: a threshold rule on the complexity signal and a runner
that wires tier-1 / tier-2 callables together.

The runner is framework-agnostic: a tier is any callable ``image -> prediction``
and the fusion is any callable ``list[prediction] -> prediction``. A prediction
is a dict with at least a ``"scores"`` array (the signal reads only that). This
keeps the library free of Ultralytics / torch; the project wires its own
detectors and WBF into it (see README).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .signals import DEPLOY_CONF, SIGNALS

Prediction = dict[str, Any]
Tier = Callable[[Any], Prediction]
Fuser = Callable[[list[Prediction]], Prediction]


def average_budget(p: float, b1: float, b2: float) -> float:
    """Average per-frame cost B(p) = B1 + p * B2 (eq. 2.6 / 4.2).

    ``b1`` is the tier-1 cost paid on every frame, ``b2`` the cost of the
    ADDITIONAL tier-2 members (tier-1 predictions are reused in the fusion).
    """
    return float(b1) + float(p) * float(b2)


@dataclass(frozen=True)
class CascadePolicy:
    """Threshold rule: escalate when signal(tier-1 scores) >= tau.

    ``tau = inf`` never escalates (pure tier 1); ``tau = -inf`` always does
    (pure ensemble). Obtain ``tau`` from :func:`scac.calibrate.choose_threshold`
    or :func:`scac.calibrate.crossfit` — never by hand.
    """

    tau: float
    signal: str = "C-count"
    deploy_conf: float = DEPLOY_CONF

    def __post_init__(self):
        if self.signal not in SIGNALS:
            raise ValueError(f"unknown signal {self.signal!r}; choose from {sorted(SIGNALS)}")

    def value(self, scores) -> float:
        return SIGNALS[self.signal](scores, self.deploy_conf)

    def escalate(self, scores) -> bool:
        return bool(self.value(scores) >= self.tau)

    def escalate_many(self, signals) -> np.ndarray:
        """Vectorised decision for precomputed signal values."""
        return np.asarray(signals, dtype=float) >= self.tau


@dataclass
class Cascade:
    """Two-tier cascade over callables.

    ``tier1(image)`` runs on every frame; if the policy escalates, every member
    of ``tier2`` runs and ``fuse([tier1_pred, *tier2_preds])`` is returned;
    otherwise the tier-1 prediction is returned unchanged. ``guard`` is an
    optional :class:`scac.guard.EscalationGuard` that can veto escalation when
    the realised fraction exceeds its cap (resource-exhaustion countermeasure).
    """

    tier1: Tier
    tier2: list[Tier]
    fuse: Fuser
    policy: CascadePolicy
    guard: Any | None = None
    _n_frames: int = field(default=0, init=False)
    _n_escalated: int = field(default=0, init=False)

    def __call__(self, image) -> tuple[Prediction, bool]:
        pred1 = self.tier1(image)
        want = self.policy.escalate(pred1["scores"])
        allowed = want if self.guard is None else self.guard.admit(want)
        self._n_frames += 1
        if not allowed:
            return pred1, False
        self._n_escalated += 1
        preds = [pred1] + [m(image) for m in self.tier2]
        return self.fuse(preds), True

    @property
    def realized_fraction(self) -> float:
        return self._n_escalated / self._n_frames if self._n_frames else 0.0

    def reset(self) -> None:
        self._n_frames = self._n_escalated = 0
