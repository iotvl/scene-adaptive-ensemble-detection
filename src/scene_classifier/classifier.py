"""
Scene classifier for adaptive ensemble selection.

Classifies input images into scene categories:
  - "day"   : good weather, daytime
  - "night" : low-light, nighttime
  - "rain"  : rain, fog, haze

Supports:
  - Heuristic-based (fast, no training needed)
  - CNN-based (MobileNetV3/EfficientNet, requires training)
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.constants import IMAGENET_MEAN, IMAGENET_STD, SceneType

logger = logging.getLogger(__name__)

SCENE_CLASSES = [s.value for s in SceneType]
SCENE_TO_IDX = {s.value: i for i, s in enumerate(SceneType)}
IDX_TO_SCENE = {i: s for i, s in enumerate(SceneType)}


class HeuristicSceneClassifier:
    """
    Rule-based scene classifier using image statistics.

    Single source of truth for heuristic scene classification — used both at
    inference time (DetectionPipeline) and at dataset-build time
    (SceneSplitter). Defaults mirror configs/default.yaml so a no-args
    construction is consistent with the config.

    Decision tree (in order):
      1. brightness < night_thresh                                   -> NIGHT
      2. low saturation AND low contrast AND high dark-channel       -> RAIN
      3. brightness < day_thresh AND low saturation                  -> RAIN  (twilight/overcast)
      4. otherwise                                                   -> DAY
    """

    # Tuned for VisDrone; the dark-channel cut-off separates fog/haze (high
    # mean dark channel) from genuine darkness (low dark channel).
    DARK_CHANNEL_RAIN_MIN: float = 80.0

    def __init__(
        self,
        brightness_night: float = 80,
        brightness_day: float = 120,
        saturation_rain: float = 50,
        contrast_rain: float = 200,
    ):
        if brightness_night >= brightness_day:
            raise ValueError(
                f"brightness_night ({brightness_night}) must be < brightness_day ({brightness_day})"
            )
        self.brightness_night = brightness_night
        self.brightness_day = brightness_day
        self.saturation_rain = saturation_rain
        self.contrast_rain = contrast_rain

    def extract_features(self, image: np.ndarray) -> dict:
        """Compute the per-image stats the heuristic relies on."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2].mean()
        brightness_std = hsv[:, :, 2].std()
        saturation = hsv[:, :, 1].mean()
        saturation_std = hsv[:, :, 1].std()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = cv2.Laplacian(gray, cv2.CV_64F).var()
        dark_channel = np.min(image, axis=2).mean()

        return {
            "brightness": float(brightness),
            "brightness_std": float(brightness_std),
            "saturation": float(saturation),
            "saturation_std": float(saturation_std),
            "contrast": float(contrast),
            "dark_channel": float(dark_channel),
        }

    def classify_features(self, features: dict) -> SceneType:
        """Run the decision tree on a precomputed feature dict."""
        brightness = features["brightness"]
        saturation = features["saturation"]
        contrast = features["contrast"]
        dark_channel = features["dark_channel"]

        if brightness < self.brightness_night:
            return SceneType.NIGHT
        if (
            saturation < self.saturation_rain
            and contrast < self.contrast_rain
            and dark_channel > self.DARK_CHANNEL_RAIN_MIN
        ):
            return SceneType.RAIN
        if brightness < self.brightness_day and saturation < self.saturation_rain:
            return SceneType.RAIN
        return SceneType.DAY

    def classify(self, image: np.ndarray) -> tuple[SceneType, dict]:
        """Classify a single BGR image. Returns (scene, features)."""
        features = self.extract_features(image)
        return self.classify_features(features), features


class SceneClassifierCNN(nn.Module):
    """
    CNN-based scene classifier using pretrained backbone.
    """

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        num_classes: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "mobilenet_v3_small":
            from torchvision.models import (
                MobileNet_V3_Small_Weights,
                mobilenet_v3_small,
            )

            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes),
            )

        elif backbone == "efficientnet_b0":
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes),
            )

        elif backbone == "shufflenet_v2_x0_5":
            from torchvision.models import (
                ShuffleNet_V2_X0_5_Weights,
                shufflenet_v2_x0_5,
            )

            weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
            self.backbone = shufflenet_v2_x0_5(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SceneClassifier:
    """
    High-level scene classifier wrapping CNN model with pre/post-processing.
    """

    def __init__(
        self,
        model_path: str | None = None,
        backbone: str = "mobilenet_v3_small",
        device: str = "auto",
        input_size: int = 224,
    ):
        from src.utils.general import get_device

        self.input_size = input_size
        self.device = get_device(device)

        # Model
        self.model = SceneClassifierCNN(backbone=backbone)

        if model_path and Path(model_path).exists():
            state_dict = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state_dict)
            logger.info("Loaded scene classifier from %s", model_path)

        self.model.to(self.device).eval()

        # Image normalization (ImageNet stats)
        self.mean = np.array(IMAGENET_MEAN)
        self.std = np.array(IMAGENET_STD)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for CNN."""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(img).unsqueeze(0).float()

    def classify(self, image: np.ndarray) -> tuple[str, dict]:
        """
        Classify an image.

        Returns:
            (scene_label, {"probabilities": {scene: prob}, "predicted": scene})
        """
        tensor = self.preprocess(image).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        scene = IDX_TO_SCENE[pred_idx]

        probabilities = {IDX_TO_SCENE[i]: float(p) for i, p in enumerate(probs)}

        return scene, {
            "probabilities": probabilities,
            "predicted": scene,
            "confidence": float(probs[pred_idx]),
        }
