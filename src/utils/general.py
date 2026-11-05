"""
General utility functions for the project.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

from src.config import AppConfig
from src.config import load_config as _load_config


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str = "configs/default.yaml") -> AppConfig:
    """Load and validate YAML configuration."""
    return _load_config(config_path)


def get_device(device_str: str = "auto") -> torch.device:
    """Get compute device (CUDA → MPS → CPU)."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def protocol_metadata(cfg: AppConfig, **overrides) -> dict:
    """Evaluation-protocol provenance to embed in every result JSON (audit §3.2).

    Captures the knobs that make metrics comparable across phases — device, conf
    threshold, image size, seed, and night split. Pass per-run specifics
    (e.g. ``night_split="test"``, ``device="mps"``) via ``overrides``.
    """
    meta = {
        "seed": cfg.project.seed,
        "eval_confidence_threshold": cfg.evaluation.eval_confidence_threshold,
        "eval_img_size": cfg.evaluation.eval_img_size,
        "iou_threshold": cfg.ensemble.iou_threshold,
        "skip_box_threshold": cfg.ensemble.skip_box_threshold,
    }
    meta.update(overrides)
    return meta
