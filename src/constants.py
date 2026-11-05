"""
Central constants for the ensemble object detection project.

All shared values (class names, mappings, enums, thresholds) live here
to avoid duplication across modules.
"""

from enum import Enum

# ---------------------------------------------------------------------------
# VisDrone-DET class definitions
# ---------------------------------------------------------------------------

VISDRONE_CLASSES: list[str] = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "bus",
    "motor",
]

NUM_CLASSES: int = len(VISDRONE_CLASSES)

# VisDrone annotation category → YOLO 0-indexed class id.
# Categories 0 (ignored regions), 7 (tricycle), 8 (awning-tricycle) and 11 (others)
# are skipped — tricycle classes are dropped from this project.
VISDRONE_TO_YOLO_CLASS: dict[int, int] = {
    1: 0,  # pedestrian
    2: 1,  # people
    3: 2,  # bicycle
    4: 3,  # car
    5: 4,  # van
    6: 5,  # truck
    9: 6,  # bus
    10: 7,  # motor
}

# Legacy 10-class layout (used by externally pretrained VisDrone checkpoints
# such as mshamrai/yolov8*-visdrone). Maps old YOLO class ids → new 8-class ids,
# with tricycle (6) and awning-tricycle (7) intentionally absent.
OLD10_TO_NEW8: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 8: 6, 9: 7}
KEPT_OLD10_IDS: list[int] = sorted(OLD10_TO_NEW8.keys())
DROPPED_OLD10_IDS: frozenset[int] = frozenset({6, 7})

# ---------------------------------------------------------------------------
# Scene classification
# ---------------------------------------------------------------------------


class SceneType(str, Enum):
    """Scene categories for adaptive ensemble routing."""

    DAY = "day"
    NIGHT = "night"
    RAIN = "rain"


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
    }
)

# ---------------------------------------------------------------------------
# ImageNet normalisation (used by scene classifier CNN)
# ---------------------------------------------------------------------------

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
