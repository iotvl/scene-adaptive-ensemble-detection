"""
Optimize ensemble weights using grid search or Bayesian optimization.
"""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np
from tqdm import tqdm

from src.evaluation.metrics import compute_map

from .ensemble_detector import EnsembleDetector

logger = logging.getLogger(__name__)


class EnsembleWeightOptimizer:
    """
    Optimize per-model weights for ensemble detection.

    Approaches:
      1. Grid search over weight combinations
      2. Random search
      3. Bayesian optimization (via scipy)
    """

    def __init__(
        self,
        ensemble: EnsembleDetector,
        val_images: list[np.ndarray],
        val_ground_truths: list[dict],
        metric: str = "mAP50",
    ):
        """
        Args:
            ensemble: EnsembleDetector to optimize
            val_images: list of validation images
            val_ground_truths: list of GT dicts with keys:
                "image_id", "boxes" (M,4), "labels" (M,)
            metric: metric to optimize
        """
        self.ensemble = ensemble
        self.val_images = val_images
        self.val_ground_truths = val_ground_truths
        self.metric = metric
        self.history: list[dict] = []

    def _evaluate_with_weights(self, weights: list[float]) -> float:
        """Evaluate ensemble with given weights and return metric."""
        self.ensemble.weights = weights

        predictions = []
        for i, img in enumerate(self.val_images):
            boxes, scores, labels = self.ensemble.predict(img)
            predictions.append(
                {
                    "image_id": i,
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                }
            )

        metrics = compute_map(predictions, self.val_ground_truths)
        return metrics.get(self.metric, 0.0)

    def grid_search(
        self,
        weight_range: tuple[float, float] = (0.5, 2.0),
        step: float = 0.25,
    ) -> tuple[list[float], float]:
        """
        Grid search over weight combinations.

        Args:
            weight_range: (min, max) for weights
            step: step size

        Returns:
            (best_weights, best_metric)
        """
        n_models = len(self.ensemble.models)
        values = np.arange(weight_range[0], weight_range[1] + step, step).tolist()

        best_weights: list[float] = []
        best_metric = -1

        combinations = list(itertools.product(values, repeat=n_models))

        if len(combinations) > 1000:
            logger.warning(
                "Grid search has %d combinations — this may take a very long time. "
                "Consider increasing step or reducing weight_range.",
                len(combinations),
            )

        logger.info(
            "Grid search: %d combinations for %d models",
            len(combinations),
            n_models,
        )

        for weights in tqdm(combinations, desc="Grid search"):
            weights = list(weights)
            metric_val = self._evaluate_with_weights(weights)

            self.history.append({"weights": weights, self.metric: metric_val})

            if metric_val > best_metric:
                best_metric = metric_val
                best_weights = weights
                logger.info(
                    "  New best: weights=%s, %s=%.4f",
                    weights,
                    self.metric,
                    metric_val,
                )

        logger.info(
            "Best weights: %s -> %s=%.4f",
            best_weights,
            self.metric,
            best_metric,
        )
        self.ensemble.weights = best_weights
        return best_weights, best_metric

    def random_search(
        self,
        n_trials: int = 100,
        weight_range: tuple[float, float] = (0.5, 2.0),
    ) -> tuple[list[float], float]:
        """Random search over weight combinations."""
        n_models = len(self.ensemble.models)

        best_weights: list[float] = []
        best_metric = -1

        for i in tqdm(range(n_trials), desc="Random search"):
            weights = np.random.uniform(weight_range[0], weight_range[1], n_models).tolist()

            metric_val = self._evaluate_with_weights(weights)
            self.history.append({"weights": weights, self.metric: metric_val})

            if metric_val > best_metric:
                best_metric = metric_val
                best_weights = weights

        logger.info(
            "Best weights: %s -> %s=%.4f",
            [f"{w:.3f}" for w in best_weights],
            self.metric,
            best_metric,
        )
        self.ensemble.weights = best_weights
        return best_weights, best_metric

    def bayesian_search(
        self,
        n_trials: int = 50,
        weight_range: tuple[float, float] = (0.5, 2.0),
    ) -> tuple[list[float], float]:
        """Bayesian optimization using scipy minimize."""
        from scipy.optimize import minimize

        n_models = len(self.ensemble.models)

        def objective(weights):
            return -self._evaluate_with_weights(weights.tolist())

        [(weight_range[0], weight_range[1])] * n_models
        x0 = np.ones(n_models)

        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": n_trials, "adaptive": True},
        )

        best_weights = result.x.tolist()
        best_metric = -result.fun

        logger.info(
            "Best weights: %s -> %s=%.4f",
            [f"{w:.3f}" for w in best_weights],
            self.metric,
            best_metric,
        )
        self.ensemble.weights = best_weights
        return best_weights, best_metric

    def save_history(self, output_path: str):
        """Save optimization history."""
        with open(output_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
