"""
Ground truth loading from YOLO-format label files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.constants import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def load_ground_truth(
    labels_dir: str | Path,
    images_dir: str | Path,
) -> tuple[list[dict], list[str]]:
    """Load ground truth boxes from YOLO label files.

    For each label file, finds the matching image (any supported extension),
    reads image dimensions, and converts normalised YOLO boxes to absolute
    ``xyxy`` pixel coordinates.

    Args:
        labels_dir: directory containing ``*.txt`` YOLO label files.
        images_dir: directory containing corresponding images.

    Returns:
        ``(gt_list, image_files)`` where *gt_list* is a list of dicts
        ``{"boxes": ndarray(N,4), "labels": ndarray(N,)}`` and
        *image_files* is a list of absolute image paths (str).
    """
    labels_path = Path(labels_dir)
    images_path = Path(images_dir)

    gt_list: list[dict] = []
    image_files: list[str] = []

    for label_file in sorted(labels_path.glob("*.txt")):
        # Find matching image with any supported extension
        img_file = _find_image(images_path, label_file.stem)
        if img_file is None:
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            logger.warning("Cannot read image: %s", img_file)
            continue
        h, w = image.shape[:2]

        boxes: list[list[float]] = []
        labels: list[int] = []

        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

        gt_list.append(
            {
                "boxes": np.array(boxes) if boxes else np.zeros((0, 4)),
                "labels": np.array(labels, dtype=int) if labels else np.zeros(0, dtype=int),
            }
        )
        image_files.append(str(img_file))

    logger.info("Loaded %d ground-truth entries from %s", len(gt_list), labels_path)
    return gt_list, image_files


def _find_image(images_dir: Path, stem: str) -> Path | None:
    """Find an image file matching *stem* with any supported extension."""
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None
