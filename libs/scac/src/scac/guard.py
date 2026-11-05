"""Escalation guard — the configuration-level countermeasure to resource
exhaustion (threat T5, dissertation §4.5).

An adversary who inflates the tier-1 detection count (phantom boxes, sponge
inputs) drives the realised escalation fraction towards 1 and the average cost
towards the full ensemble (κ up to 2.0 measured on UAVDT). The guard caps the
fraction of escalated frames in a sliding window: once the cap is reached the
cascade degrades to tier 1 until the window drains. It never changes behaviour
while the realised fraction stays below the cap.
"""

from __future__ import annotations

from collections import deque


class EscalationGuard:
    def __init__(self, p_max: float, window: int = 100):
        if not 0.0 < p_max <= 1.0:
            raise ValueError("p_max must be in (0, 1]")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.p_max = float(p_max)
        self.window = int(window)
        self._hist: deque[bool] = deque(maxlen=window)
        self.vetoed = 0

    @property
    def realized_fraction(self) -> float:
        return sum(self._hist) / len(self._hist) if self._hist else 0.0

    def admit(self, want_escalate: bool) -> bool:
        """Record the frame and return whether escalation is allowed now."""
        if not want_escalate:
            self._hist.append(False)
            return False
        # would escalating this frame push the window above the cap? Unfilled
        # slots of a fresh window count as non-escalated frames, so the guard
        # admits up to p_max * window escalations before it ever vetoes.
        future = (sum(self._hist) + 1) / self.window
        if future > self.p_max + 1e-12:
            self._hist.append(False)
            self.vetoed += 1
            return False
        self._hist.append(True)
        return True

    def budget_overrun(self, p_planned: float, b1: float, b2: float) -> float:
        """κ = B(realised) / B(planned) (eq. 2.15)."""
        return (b1 + self.realized_fraction * b2) / (b1 + p_planned * b2)

    def reset(self) -> None:
        self._hist.clear()
        self.vetoed = 0
