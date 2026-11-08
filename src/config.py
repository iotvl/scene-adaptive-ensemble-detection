"""
Pydantic-validated configuration for the ensemble object detection system.

Loads YAML via OmegaConf, then validates through Pydantic v2 models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from pydantic import BaseModel, model_validator

# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------


class ProjectConfig(BaseModel):
    name: str = "ensemble-object-detection"
    root: str = "."
    seed: int = 42


class DatasetConfig(BaseModel):
    name: str = "VisDrone"
    root: str = "data/VisDrone"
    train_images: str = ""
    train_labels: str = ""
    val_images: str = ""
    val_labels: str = ""
    test_images: str = ""
    test_labels: str = ""
    classes: list[str] = []
    num_classes: int = 8
    img_size: int = 640

    @model_validator(mode="after")
    def check_class_count(self) -> DatasetConfig:
        if self.classes and self.num_classes != len(self.classes):
            msg = f"num_classes ({self.num_classes}) != len(classes) ({len(self.classes)})"
            raise ValueError(msg)
        return self


class SceneClassifierConfig(BaseModel):
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = True
    input_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    lr: float = 0.001


class AugmentationConfig(BaseModel):
    enabled: bool = True
    ratio: float = 0.3
    effects: list[str] = ["rain", "fog", "rain_fog"]
    intensity_range: list[float] = [0.3, 0.8]
    seed: int = 42


class ScenesConfig(BaseModel):
    classes: list[str] = ["day", "night", "rain"]
    brightness_night_threshold: float = 80
    brightness_day_threshold: float = 120
    saturation_rain_threshold: float = 50
    contrast_rain_threshold: float = 200
    classifier: SceneClassifierConfig = SceneClassifierConfig()
    augmentation: AugmentationConfig = AugmentationConfig()

    @model_validator(mode="after")
    def check_thresholds(self) -> ScenesConfig:
        if self.brightness_night_threshold >= self.brightness_day_threshold:
            msg = (
                f"brightness_night_threshold ({self.brightness_night_threshold}) "
                f"must be < brightness_day_threshold ({self.brightness_day_threshold})"
            )
            raise ValueError(msg)
        return self


class ModelConfig(BaseModel, extra="allow"):
    name: str
    weights: str
    framework: str = "ultralytics"
    params_m: float = 0.0


class EnsembleSceneConfig(BaseModel):
    models: list[str]
    method: str = "wbf"
    weights: list[float]

    @model_validator(mode="after")
    def check_weights_length(self) -> EnsembleSceneConfig:
        if len(self.weights) != len(self.models):
            msg = f"len(weights)={len(self.weights)} != len(models)={len(self.models)}"
            raise ValueError(msg)
        return self


class EnsembleConfig(BaseModel):
    methods: list[str] = ["wbf", "nms", "soft_nms"]
    default_method: str = "wbf"
    iou_threshold: float = 0.55
    skip_box_threshold: float = 0.01
    sigma: float = 0.5
    day: EnsembleSceneConfig = EnsembleSceneConfig(models=[], weights=[])
    night: EnsembleSceneConfig = EnsembleSceneConfig(models=[], weights=[])
    rain: EnsembleSceneConfig = EnsembleSceneConfig(models=[], weights=[])


class EvaluationConfig(BaseModel):
    metrics: list[str] = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
    iou_thresholds: list[float] = [0.5, 0.75]
    # Deployment/inference confidence threshold (used by the pipeline at runtime).
    confidence_threshold: float = 0.25
    # mAP-evaluation confidence: keep low so the full precision-recall curve is
    # integrated. Using the deployment threshold (0.25) truncates the curve and
    # systematically *under*-reports mAP — they are deliberately separate knobs.
    eval_confidence_threshold: float = 0.001
    # Inference image size used only for metric evaluation. VisDrone images are
    # ~2000x1500 with many tiny objects, so 640 crushes small-object recall;
    # 1280+ recovers it. None falls back to dataset.img_size.
    eval_img_size: int | None = None
    batch_size: int = 16
    num_workers: int = 4


class ProfilingConfig(BaseModel):
    num_warmup: int = 10
    num_iterations: int = 100
    input_size: list[int] = [1, 3, 640, 640]
    device: str = "cuda"
    measure: list[str] = ["latency", "fps", "gpu_memory", "cpu_memory", "flops", "params"]


class TrainingConfig(BaseModel):
    batch_size: int = 16
    epochs: int = 100
    optimizer: str = "AdamW"
    lr: float = 0.001
    weight_decay: float = 0.0005
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    amp: bool = True
    device: str = "0"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel, extra="allow"):
    project: ProjectConfig = ProjectConfig()
    dataset: DatasetConfig = DatasetConfig()
    scenes: ScenesConfig = ScenesConfig()
    small_models: dict[str, ModelConfig] = {}
    large_models: dict[str, ModelConfig] = {}
    finetuned_models: dict[str, ModelConfig] = {}
    ensemble: EnsembleConfig = EnsembleConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    profiling: ProfilingConfig = ProfilingConfig()
    training: TrainingConfig = TrainingConfig()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


# Prediction-cache image-size token for models that do NOT run through the shared
# Ultralytics letterbox at ``evaluation.eval_img_size`` (e.g. RT-DETRv4 at its own fixed
# 960x960 resize). Filled by ``load_config`` from ``finetuned_models.<key>.framework`` /
# ``eval_size``; consulted by ``src.ensemble.scene_eval.cache_path`` so every cache-driven
# experiment resolves such a member's file without per-script special cases.
NATIVE_CACHE_IMG_TOKENS: dict[str, str] = {}


def register_native_cache_tokens(cfg: AppConfig) -> None:
    for key, m in cfg.finetuned_models.items():
        if m.framework != "ultralytics":
            size = getattr(m, "eval_size", None)
            NATIVE_CACHE_IMG_TOKENS[key] = f"native{size}" if size else "native"


def load_config(config_path: str | Path = "configs/default.yaml") -> AppConfig:
    """Load YAML config via OmegaConf and validate through Pydantic.

    Args:
        config_path: path to the YAML configuration file.

    Returns:
        Validated ``AppConfig`` instance.

    Raises:
        FileNotFoundError: if the config file does not exist.
        pydantic.ValidationError: if the config fails validation.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = cast(
        "dict[str, Any]",
        OmegaConf.to_container(OmegaConf.load(config_path), resolve=True),
    )

    cfg = AppConfig.model_validate(raw)
    register_native_cache_tokens(cfg)
    return cfg
