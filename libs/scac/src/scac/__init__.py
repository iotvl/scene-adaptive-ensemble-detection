"""scac — Scene-Complexity Adaptive Cascade.

A model-agnostic library for the two-tier detector cascade studied in the
dissertation (chapter 4): a cheap tier-1 detector runs on every frame, a
complexity signal computed from its output decides whether the frame is
escalated to an expensive tier-2 ensemble.

Public API (all NumPy-only, no inference framework dependency):

- signals:   ``c_count``, ``c_conf``, ``c_entropy``, ``signal_values``
- policy:    ``CascadePolicy`` (threshold rule), ``Cascade`` (runs callables),
             ``average_budget`` (B(p) = B1 + p * B2)
- calibrate: ``choose_threshold`` (p* on a calibration set),
             ``stratified_folds`` (2-fold by group, stratified by scene),
             ``crossfit`` (fold-wise thresholds, cross-applied),
             ``realized_by_scene`` (diagnostic: escalation fraction per scene)
- gate:      ``gate_stats`` (density of the ensemble gain),
             ``oracle_curve`` / ``signal_curve`` (quality vs. escalation fraction)
- guard:     ``EscalationGuard`` (sliding-window cap on the realised fraction)
"""

from .calibrate import choose_threshold, crossfit, realized_by_scene, stratified_folds
from .gate import gate_stats, oracle_curve, signal_curve
from .guard import EscalationGuard
from .policy import Cascade, CascadePolicy, average_budget
from .signals import c_conf, c_count, c_entropy, signal_values

__all__ = [
    "Cascade",
    "CascadePolicy",
    "EscalationGuard",
    "average_budget",
    "c_conf",
    "c_count",
    "c_entropy",
    "choose_threshold",
    "crossfit",
    "gate_stats",
    "oracle_curve",
    "realized_by_scene",
    "signal_curve",
    "signal_values",
    "stratified_folds",
]

__version__ = "0.1.0"
