"""
Ensemble object detector combining multiple models with configurable fusion.

Supports:
  - WBF, NMS, Soft-NMS, NMW fusion
  - Per-model weights
  - Per-scene ensemble configurations
  - Weight optimization via grid/Bayesian search
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .methods import fusion_extra_kwargs, get_ensemble_method

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    Ensemble of multiple object detection models.

    Usage:
        detector = EnsembleDetector(
            model_paths=["yolov8n.pt", "yolov8s.pt", "yolov5n.pt"],
            method="wbf",
            weights=[1.0, 1.5, 1.0],
        )
        boxes, scores, labels = detector.predict(image)
    """

    def __init__(
        self,
        model_paths: list[str],
        model_names: list[str] | None = None,
        method: str = "wbf",
        weights: list[float] | None = None,
        iou_thr: float = 0.55,
        skip_box_thr: float = 0.01,
        sigma: float = 0.5,
        conf_thr: float = 0.25,
        img_size: int = 640,
        device: str = "auto",
    ):
        """
        Args:
            model_paths:  paths to model weights
            model_names:  human-readable names
            method:       ensemble method ("wbf", "nms", "soft_nms", "nmw")
            weights:      per-model weights for fusion
            iou_thr:      IoU threshold for fusion
            skip_box_thr: minimum score threshold (wbf/nmw; soft_nms thresh)
            sigma:        Gaussian decay for soft_nms
            conf_thr:     confidence threshold for individual models
            img_size:     inference image size
            device:       compute device
        """
        self.model_paths = model_paths
        self.model_names = model_names or [Path(p).stem for p in model_paths]
        self.method = method
        if weights is not None and len(weights) != len(model_paths):
            raise ValueError(
                f"len(weights)={len(weights)} != len(model_paths)={len(model_paths)}"
            )
        self.weights = weights or [1.0] * len(model_paths)
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.sigma = sigma
        self.conf_thr = conf_thr
        self.img_size = img_size
        self.device = device

        self.models = []
        self._load_models()

        self.ensemble_fn = get_ensemble_method(method)

    def _load_models(self):
        """Load all detection models."""
        from ultralytics import YOLO

        for path, name in zip(self.model_paths, self.model_names):
            logger.info("  Loading %s: %s", name, path)
            model = YOLO(path)
            self.models.append(model)

        logger.info(
            "  Loaded %d models for ensemble (%s)",
            len(self.models),
            self.method,
        )

    def predict_single(self, model, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions from a single model.

        Returns:
            (boxes_xyxy, scores, labels) as numpy arrays
        """
        results = model.predict(
            image,
            imgsz=self.img_size,
            conf=self.conf_thr,
            device=self.device,
            verbose=False,
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy().astype(int)

        return boxes, scores, labels

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run ensemble prediction on a single image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            (boxes_xyxy, scores, labels)
        """
        h, w = image.shape[:2]

        all_boxes = []
        all_scores = []
        all_labels = []

        for model in self.models:
            boxes, scores, labels = self.predict_single(model, image)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        # Apply ensemble fusion
        fused_boxes, fused_scores, fused_labels = self.ensemble_fn(
            all_boxes,
            all_scores,
            all_labels,
            weights=self.weights,
            iou_thr=self.iou_thr,
            img_size=(w, h),
            **fusion_extra_kwargs(
                self.method, skip_box_thr=self.skip_box_thr, sigma=self.sigma
            ),
        )

        return fused_boxes, fused_scores, fused_labels

    def predict_batch(
        self, images: list[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run ensemble on a batch of images."""
        return [self.predict(img) for img in images]

    def benchmark(self, image: np.ndarray, num_runs: int = 50) -> dict:
        """
        Benchmark the ensemble speed.

        Returns dict with timing information.
        """
        # Warmup
        for _ in range(5):
            self.predict(image)

        # Individual model times
        per_model_times = {name: [] for name in self.model_names}
        fusion_times = []
        total_times = []

        for _ in range(num_runs):
            h, w = image.shape[:2]
            all_boxes, all_scores, all_labels = [], [], []

            total_start = time.perf_counter()

            for model, name in zip(self.models, self.model_names):
                t0 = time.perf_counter()
                boxes, scores, labels = self.predict_single(model, image)
                per_model_times[name].append((time.perf_counter() - t0) * 1000)
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

            t_fusion_start = time.perf_counter()
            self.ensemble_fn(
                all_boxes,
                all_scores,
                all_labels,
                weights=self.weights,
                iou_thr=self.iou_thr,
                img_size=(w, h),
                **fusion_extra_kwargs(
                    self.method, skip_box_thr=self.skip_box_thr, sigma=self.sigma
                ),
            )
            fusion_times.append((time.perf_counter() - t_fusion_start) * 1000)
            total_times.append((time.perf_counter() - total_start) * 1000)

        result = {
            "method": self.method,
            "num_models": len(self.models),
            "total_mean_ms": float(np.mean(total_times)),
            "total_std_ms": float(np.std(total_times)),
            "total_fps": float(1000.0 / np.mean(total_times)),
            "fusion_mean_ms": float(np.mean(fusion_times)),
        }

        for name, times in per_model_times.items():
            result[f"{name}_mean_ms"] = float(np.mean(times))

        return result


class AdaptiveEnsembleDetector:
    """
    Adaptive ensemble that selects different model configurations
    based on scene classification.
    """

    def __init__(
        self,
        scene_ensembles: dict[str, EnsembleDetector],
        scene_classifier=None,
        default_scene: str = "day",
    ):
        """
        Args:
            scene_ensembles: dict mapping scene name -> EnsembleDetector
            scene_classifier: SceneClassifier instance (or None for manual)
            default_scene: fallback scene if classifier fails
        """
        self.scene_ensembles = scene_ensembles
        self.scene_classifier = scene_classifier
        self.default_scene = default_scene

    def predict(
        self, image: np.ndarray, scene: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Predict with adaptive ensemble selection.

        Args:
            image: BGR image
            scene: force scene type (skip classifier)

        Returns:
            (boxes, scores, labels, detected_scene)
        """
        # Classify scene
        if scene is None and self.scene_classifier is not None:
            scene, _ = self.scene_classifier.classify(image)

        if scene is None or scene not in self.scene_ensembles:
            scene = self.default_scene

        # Run appropriate ensemble
        ensemble = self.scene_ensembles[scene]
        boxes, scores, labels = ensemble.predict(image)

        return boxes, scores, labels, scene
