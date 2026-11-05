"""Shared pytest fixtures for the ensemble object detection test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_boxes() -> np.ndarray:
    """Two sample xyxy bounding boxes."""
    return np.array(
        [
            [100.0, 100.0, 200.0, 200.0],
            [300.0, 300.0, 400.0, 400.0],
        ]
    )


@pytest.fixture
def sample_scores() -> np.ndarray:
    return np.array([0.9, 0.7])


@pytest.fixture
def sample_labels() -> np.ndarray:
    return np.array([0, 3])


@pytest.fixture
def tmp_yaml_config(tmp_path: Path) -> Path:
    """Write a minimal valid config YAML and return its path."""
    cfg = tmp_path / "test_config.yaml"
    cfg.write_text(
        """\
project:
  name: test
  root: "."
  seed: 42

dataset:
  name: VisDrone
  root: data/VisDrone
  classes:
    - pedestrian
    - people
    - bicycle
    - car
    - van
    - truck
    - bus
    - motor
  num_classes: 8
  img_size: 640

scenes:
  classes: [day, night, rain]
  brightness_night_threshold: 80
  brightness_day_threshold: 120
  saturation_rain_threshold: 50
  contrast_rain_threshold: 200
  classifier:
    backbone: mobilenet_v3_small
    pretrained: true
    input_size: 224
    batch_size: 32
    epochs: 30
    lr: 0.001
  augmentation:
    enabled: true
    ratio: 0.3
    effects: [rain, fog, rain_fog]
    intensity_range: [0.3, 0.8]
    seed: 42

ensemble:
  methods: [wbf, nms, soft_nms]
  default_method: wbf
  iou_threshold: 0.55
  skip_box_threshold: 0.01
  sigma: 0.5
  day:
    models: [yolov8n]
    method: wbf
    weights: [1.0]
  night:
    models: [yolov8n]
    method: wbf
    weights: [1.0]
  rain:
    models: [yolov8n]
    method: wbf
    weights: [1.0]

evaluation:
  metrics: [mAP50]
  iou_thresholds: [0.5]
  confidence_threshold: 0.25
  batch_size: 4
  num_workers: 0

profiling:
  num_warmup: 2
  num_iterations: 5
  input_size: [1, 3, 640, 640]
  device: cpu
  measure: [latency, fps]

training:
  batch_size: 4
  epochs: 1
  optimizer: AdamW
  lr: 0.001
  weight_decay: 0.0005
  scheduler: cosine
  warmup_epochs: 0
  amp: false
  device: cpu
"""
    )
    return cfg
