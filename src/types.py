"""
Shared data structures used across the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import SceneType


@dataclass
class DetectionResult:
    """Result from a single image detection (single model or ensemble)."""

    boxes: np.ndarray  # (N, 4) xyxy absolute pixels
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) int class indices

    @property
    def num_detections(self) -> int:
        return len(self.scores)

    @property
    def is_empty(self) -> bool:
        return self.num_detections == 0

    @staticmethod
    def empty() -> DetectionResult:
        return DetectionResult(
            boxes=np.zeros((0, 4)),
            scores=np.zeros(0),
            labels=np.zeros(0, dtype=int),
        )


@dataclass
class ScenePrediction:
    """Result from a scene classifier."""

    scene: SceneType
    confidence: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    """Evaluation metrics for a model or ensemble on a dataset."""

    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    per_class_ap: dict[str, float] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Resource profiling result for a model."""

    model_name: str = ""
    params: int = 0
    params_m: float = 0.0
    flops_g: float | None = None
    latency_ms: float = 0.0
    fps: float = 0.0
    gpu_memory_mb: float | None = None
    cpu_memory_mb: float | None = None
