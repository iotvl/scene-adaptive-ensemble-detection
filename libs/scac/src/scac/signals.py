"""Annotation-free complexity signals computed from tier-1 detection scores.

Only detections visible at the deployment threshold are used: this is exactly
what a deployed system can observe. Higher signal = more complex = escalate
first. The definitions are identical to the experiment code the dissertation
reports on (src/ensemble/scac.py), so numbers are directly comparable.
"""

from __future__ import annotations

import numpy as np

DEPLOY_CONF = 0.25
TOPK_CONF = 10


def _visible(scores, deploy_conf: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    return s[s >= deploy_conf]


def c_count(scores, deploy_conf: float = DEPLOY_CONF) -> float:
    """Number of deploy-visible detections (scene density).

    The best realizable signal in both studied domains; an empty frame has
    complexity 0 (an empty frame is not a dense frame).
    """
    return float(_visible(scores, deploy_conf).size)


def c_conf(scores, deploy_conf: float = DEPLOY_CONF, topk: int = TOPK_CONF) -> float:
    """1 - mean confidence of the top-k visible detections (tier-1 uncertainty).

    Consistently no better than random escalation on dense aerial imagery;
    provided for ablations. An empty frame has maximal uncertainty (1.0).
    """
    vis = _visible(scores, deploy_conf)
    if vis.size == 0:
        return 1.0
    return 1.0 - float(np.sort(vis)[::-1][:topk].mean())


def c_entropy(scores, deploy_conf: float = DEPLOY_CONF) -> float:
    """Mean binary entropy of the visible detection confidences.

    An empty frame has maximal entropy (1.0).
    """
    vis = _visible(scores, deploy_conf)
    if vis.size == 0:
        return 1.0
    p = np.clip(vis, 1e-6, 1 - 1e-6)
    return float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))


def signal_values(
    scores, deploy_conf: float = DEPLOY_CONF, topk: int = TOPK_CONF
) -> dict[str, float]:
    """All three signals for one frame, keyed as in the dissertation."""
    return {
        "C-count": c_count(scores, deploy_conf),
        "C-conf": c_conf(scores, deploy_conf, topk),
        "C-entropy": c_entropy(scores, deploy_conf),
    }


SIGNALS = {"C-count": c_count, "C-conf": c_conf, "C-entropy": c_entropy}
