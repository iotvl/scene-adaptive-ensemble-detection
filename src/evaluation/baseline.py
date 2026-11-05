"""
Baseline evaluation of individual object detection models on VisDrone.

Supports:
  - Ultralytics models (YOLOv5, YOLOv8, RT-DETR)
  - Per-scene evaluation
  - Full metrics reporting (mAP, precision, recall, per-class AP)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from src.constants import SceneType
from src.utils.general import ensure_dir, get_device, load_config, set_seed

logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """Evaluate detection models on the VisDrone dataset."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.cfg = load_config(config_path)
        set_seed(self.cfg.project.seed)
        self.device = get_device()
        self.results: list[dict] = []

    def evaluate_ultralytics_model(
        self,
        model_name: str,
        weights: str,
        data_yaml: str,
        img_size: int = 640,
        batch_size: int = 16,
        conf_thresh: float = 0.001,
    ) -> dict:
        """
        Evaluate an Ultralytics model (YOLOv5/v8, RT-DETR).

        Args:
            model_name: descriptive name
            weights: path to .pt weights
            data_yaml: path to dataset.yaml
            img_size: inference image size
            batch_size: batch size
            conf_thresh: confidence threshold for mAP integration. Keep this low
                (~0.001); the deployment threshold (e.g. 0.25) truncates the
                precision-recall curve and under-reports mAP.

        Returns:
            Dict with evaluation results
        """
        import yaml as _yaml
        from ultralytics import YOLO

        logger.info("Evaluating: %s (weights=%s)", model_name, weights)

        model = YOLO(weights)

        # Guard: a model whose class space differs from the dataset's will be
        # scored by raw class index, producing meaningless mAP (e.g. COCO `car`
        # at index 2 matched against VisDrone `bicycle` at index 2). Warn loudly.
        model_nc = getattr(getattr(model, "model", None), "nc", None)
        try:
            with open(data_yaml) as _f:
                data_nc = len(_yaml.safe_load(_f).get("names", {}))
        except Exception:
            data_nc = None
        if model_nc is not None and data_nc is not None and model_nc != data_nc:
            logger.warning(
                "CLASS-SPACE MISMATCH: %s has %d classes but %s declares %d. "
                "mAP will be matched by raw index and is NOT a valid measure of "
                "detection quality for this dataset.",
                model_name,
                model_nc,
                data_yaml,
                data_nc,
            )

        # Measure inference time
        start_time = time.time()
        metrics = model.val(
            data=data_yaml,
            imgsz=img_size,
            batch=batch_size,
            conf=conf_thresh,
            device=str(self.device),
            verbose=False,
        )
        eval_time = time.time() - start_time

        result = {
            "model": model_name,
            "weights": weights,
            "dataset": data_yaml,
            "img_size": img_size,
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "eval_time_s": eval_time,
        }

        # Per-class AP
        if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
            class_names = self.cfg.dataset.classes
            for i, name in enumerate(class_names):
                if i < len(metrics.box.ap50):
                    result[f"AP50_{name}"] = float(metrics.box.ap50[i])

        self.results.append(result)
        self._print_result(result)

        return result

    def evaluate_all_models(
        self,
        data_yaml: str,
        model_configs: dict | None = None,
        category: str = "all",
    ) -> pd.DataFrame:
        """
        Evaluate all configured models.

        Args:
            data_yaml: path to dataset.yaml
            model_configs: dict of model configs (default: from config)
            category: "small", "large", or "all"
        """
        if model_configs is None:
            model_configs = {}
            if category in ("small", "all"):
                for key, cfg in self.cfg.small_models.items():
                    model_configs[key] = cfg
            if category in ("large", "all"):
                for key, cfg in self.cfg.large_models.items():
                    model_configs[key] = cfg

        for key, mcfg in model_configs.items():
            if mcfg.framework == "ultralytics":
                self.evaluate_ultralytics_model(
                    model_name=mcfg.name,
                    weights=mcfg.weights,
                    data_yaml=data_yaml,
                    img_size=self.cfg.dataset.img_size,
                    batch_size=self.cfg.evaluation.batch_size,
                    conf_thresh=self.cfg.evaluation.confidence_threshold,
                )
            else:
                logger.info(
                    "Framework '%s' not yet supported for %s",
                    mcfg.framework,
                    mcfg.name,
                )

        return pd.DataFrame(self.results)

    def evaluate_per_scene(
        self,
        scenes_root: str,
        model_configs: dict | None = None,
        category: str = "all",
    ) -> pd.DataFrame:
        """
        Evaluate models on each scene (day, night, rain) + overall.

        Args:
            scenes_root: root of scene-split dataset (with day/, night/, rain/ subdirs)
            model_configs: dict of model configs
            category: "small", "large", or "all"
        """
        scenes_dir = Path(scenes_root)
        all_results = []

        if model_configs is None:
            model_configs = {}
            if category in ("small", "all"):
                for key, cfg in self.cfg.small_models.items():
                    model_configs[key] = cfg
            if category in ("large", "all"):
                for key, cfg in self.cfg.large_models.items():
                    model_configs[key] = cfg

        for scene in SceneType:
            scene_yaml = scenes_dir / scene.value / "dataset.yaml"
            if not scene_yaml.exists():
                logger.info("%s not found, skipping", scene_yaml)
                continue

            logger.info("Scene: %s", scene.value.upper())

            for key, mcfg in model_configs.items():
                if mcfg.framework == "ultralytics":
                    result = self.evaluate_ultralytics_model(
                        model_name=mcfg.name,
                        weights=mcfg.weights,
                        data_yaml=str(scene_yaml),
                    )
                    result["scene"] = scene.value
                    all_results.append(result)

        df = pd.DataFrame(all_results)
        return df

    def save_results(self, output_path: str = "experiments/baseline/results.csv"):
        """Save accumulated results to CSV."""
        out_path = Path(output_path)
        ensure_dir(out_path.parent)

        df = pd.DataFrame(self.results)
        df.to_csv(out_path, index=False)
        logger.info("Results saved to %s", out_path)

        # Also save as JSON
        json_path = out_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)

        return df

    def _print_result(self, result: dict):
        """Pretty-print a single result."""
        logger.info(
            "%s: mAP50=%.4f, mAP50-95=%.4f, P=%.4f, R=%.4f (%.1fs)",
            result["model"],
            result["mAP50"],
            result["mAP50-95"],
            result["precision"],
            result["recall"],
            result["eval_time_s"],
        )


def select_top_models(
    results_df: pd.DataFrame,
    metric: str = "mAP50",
    n: int = 3,
    category: str = "small",
) -> list[str]:
    """
    Select top-N models based on a metric.

    Args:
        results_df: evaluation results
        metric: metric to sort by
        n: number of models to select
        category: filter by model category

    Returns:
        List of model names
    """
    df = results_df.sort_values(metric, ascending=False)
    top = df.head(n)["model"].tolist()
    logger.info("Top-%d models by %s: %s", n, metric, top)
    return top


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/VisDrone-YOLO/dataset.yaml")
    parser.add_argument("--scenes", default=None, help="Scene-split root dir")
    parser.add_argument("--category", choices=["small", "large", "all"], default="all")
    parser.add_argument("--output", default="experiments/baseline/results.csv")
    args = parser.parse_args()

    evaluator = BaselineEvaluator(args.config)

    if args.scenes:
        df = evaluator.evaluate_per_scene(args.scenes, category=args.category)
    else:
        df = evaluator.evaluate_all_models(args.data, category=args.category)

    evaluator.save_results(args.output)
    print("\n" + df.to_string())
