"""
Main pipeline: Adaptive Ensemble Object Detection System.

Full inference pipeline:
  Image -> Scene Classifier -> Scene-specific Ensemble -> Detections
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.constants import IMAGE_EXTENSIONS, SceneType
from src.ensemble.ensemble_detector import AdaptiveEnsembleDetector, EnsembleDetector
from src.scene_classifier.classifier import HeuristicSceneClassifier, SceneClassifier
from src.utils.general import get_device, load_config
from src.utils.visualization import plot_detections

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Full adaptive object detection pipeline.

    Flow:
        1. Input image
        2. Scene classification (CNN or heuristic)
        3. Route to scene-specific ensemble
        4. Ensemble fusion
        5. Output detections
    """

    def __init__(self, config_path: str = "configs/default.yaml", remap_10to8: bool = False):
        """
        Args:
            config_path: path to YAML config
            remap_10to8: if the routed weights are 10-class VisDrone checkpoints
                (e.g. mshamrai/yolov8*-visdrone), remap detections into the
                project's 8-class space on output (drop tricycle/awning, bus 8->6,
                motor 9->7). Leave False for models that already emit 8 classes —
                remapping then would wrongly drop real bus/motor at indices 6/7.
        """
        self.cfg = load_config(config_path)
        self.device = get_device()
        self.remap_10to8 = remap_10to8

        self.scene_classifier = None
        self.scene_ensembles: dict[str, EnsembleDetector] = {}
        self.adaptive_detector = None

        self._initialized = False

    def setup(
        self,
        scene_classifier_weights: str | None = None,
        use_heuristic: bool = False,
    ):
        """
        Initialize the full pipeline.

        Args:
            scene_classifier_weights: path to CNN classifier weights
            use_heuristic: use heuristic classifier instead of CNN
        """
        logger.info("Initializing Detection Pipeline")

        # 1. Scene classifier
        if use_heuristic:
            logger.info("[1/2] Using heuristic scene classifier")
            self.scene_classifier = HeuristicSceneClassifier(
                brightness_night=self.cfg.scenes.brightness_night_threshold,
                brightness_day=self.cfg.scenes.brightness_day_threshold,
                saturation_rain=self.cfg.scenes.saturation_rain_threshold,
                contrast_rain=self.cfg.scenes.contrast_rain_threshold,
            )
        else:
            logger.info(
                "[1/2] Loading CNN scene classifier: %s",
                scene_classifier_weights,
            )
            self.scene_classifier = SceneClassifier(
                model_path=scene_classifier_weights,
                backbone=self.cfg.scenes.classifier.backbone,
                device=str(self.device),
            )

        # 2. Per-scene ensembles
        logger.info("[2/2] Loading scene-specific ensembles...")
        for scene in SceneType:
            scene_cfg = getattr(self.cfg.ensemble, scene.value)

            # Resolve model weights
            model_keys = scene_cfg.models
            weights_paths = []
            model_names = []

            for key in model_keys:
                if key in self.cfg.small_models:
                    mcfg = self.cfg.small_models[key]
                elif key in self.cfg.large_models:
                    mcfg = self.cfg.large_models[key]
                elif key in self.cfg.finetuned_models:
                    mcfg = self.cfg.finetuned_models[key]
                else:
                    raise ValueError(
                        f"Ensemble model key '{key}' for scene '{scene.value}' not found "
                        "in small_models / large_models / finetuned_models. Silently "
                        "skipping it would desync the per-model weights (audit C1)."
                    )
                weights_paths.append(mcfg.weights)
                model_names.append(mcfg.name)

            if not weights_paths:
                logger.info("No models configured for scene '%s'", scene.value)
                continue

            logger.info(
                "Scene '%s': %s (method=%s, weights=%s)",
                scene.value,
                model_names,
                scene_cfg.method,
                list(scene_cfg.weights),
            )

            self.scene_ensembles[scene.value] = EnsembleDetector(
                model_paths=weights_paths,
                model_names=model_names,
                method=scene_cfg.method,
                weights=list(scene_cfg.weights),
                iou_thr=self.cfg.ensemble.iou_threshold,
                skip_box_thr=self.cfg.ensemble.skip_box_threshold,
                sigma=self.cfg.ensemble.sigma,
                conf_thr=self.cfg.evaluation.confidence_threshold,
                img_size=self.cfg.dataset.img_size,
                device=str(self.device),
            )

        # 3. Adaptive detector
        self.adaptive_detector = AdaptiveEnsembleDetector(
            scene_ensembles=self.scene_ensembles,
            scene_classifier=self.scene_classifier,
            default_scene="day",
        )

        self._initialized = True
        logger.info("Pipeline initialized!")

    def detect(
        self,
        image: np.ndarray,
        force_scene: str | None = None,
        visualize: bool = False,
        save_path: str | None = None,
    ) -> dict:
        """
        Run detection on a single image.

        Args:
            image: BGR image (H, W, 3)
            force_scene: force a specific scene
            visualize: draw boxes on image
            save_path: save annotated image

        Returns:
            Dict with keys: "scene", "boxes", "scores", "labels", "num_detections", "time_ms"
        """
        if not self._initialized or self.adaptive_detector is None:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        start = time.perf_counter()

        boxes, scores, labels, scene = self.adaptive_detector.predict(image, scene=force_scene)

        if self.remap_10to8:
            from src.evaluation.metrics import remap_10class_to_8class

            boxes, scores, labels = remap_10class_to_8class(boxes, scores, labels)

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = {
            "scene": scene,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "num_detections": len(boxes),
            "time_ms": elapsed_ms,
        }

        if visualize or save_path:
            annotated = plot_detections(
                image,
                boxes,
                labels,
                scores,
                save_path=save_path,
            )
            result["annotated_image"] = annotated

        return result

    def detect_batch(
        self,
        images: list[np.ndarray],
        force_scene: str | None = None,
    ) -> list[dict]:
        """Run detection on a batch of images."""
        return [self.detect(img, force_scene=force_scene) for img in images]

    def detect_directory(
        self,
        images_dir: str,
        output_dir: str | None = None,
        force_scene: str | None = None,
        max_images: int | None = None,
    ) -> list[dict]:
        """
        Run detection on all images in a directory.
        """
        from tqdm import tqdm

        images_path = Path(images_dir)
        image_files = sorted(
            p for p in images_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if max_images:
            image_files = image_files[:max_images]

        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        results = []
        for img_path in tqdm(image_files, desc="Detecting"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            save_path = str(output_path / img_path.name) if output_path else None
            result = self.detect(
                image,
                force_scene=force_scene,
                save_path=save_path,
            )
            result["filename"] = img_path.name
            results.append(result)

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run detection pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True, help="Image or directory path")
    parser.add_argument("--output", default=None, help="Output directory for visualizations")
    parser.add_argument("--scene-weights", default=None, help="Scene classifier weights")
    parser.add_argument("--heuristic", action="store_true", help="Use heuristic classifier")
    parser.add_argument("--force-scene", default=None, choices=["day", "night", "rain"])
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    pipeline = DetectionPipeline(args.config)
    pipeline.setup(
        scene_classifier_weights=args.scene_weights,
        use_heuristic=args.heuristic,
    )

    input_path = Path(args.input)

    if input_path.is_dir():
        results = pipeline.detect_directory(
            str(input_path),
            output_dir=args.output,
            force_scene=args.force_scene,
            max_images=args.max_images,
        )
        print(f"\nProcessed {len(results)} images")

        # Summary
        scenes = [r["scene"] for r in results]
        for scene in set(scenes):
            count = scenes.count(scene)
            avg_dets = np.mean([r["num_detections"] for r in results if r["scene"] == scene])
            avg_time = np.mean([r["time_ms"] for r in results if r["scene"] == scene])
            print(f"  {scene}: {count} images, avg {avg_dets:.0f} detections, {avg_time:.1f}ms")
    else:
        image = cv2.imread(str(input_path))
        if image is None:
            raise SystemExit(f"Could not read image: {input_path}")
        result = pipeline.detect(
            image,
            force_scene=args.force_scene,
            visualize=True,
            save_path=str(Path(args.output) / input_path.name) if args.output else None,
        )
        print(f"\nScene: {result['scene']}")
        print(f"Detections: {result['num_detections']}")
        print(f"Time: {result['time_ms']:.1f}ms")
