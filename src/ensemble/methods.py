"""
Ensemble fusion methods for object detection bounding boxes.

Implements:
  - NMS (Non-Maximum Suppression)
  - Soft-NMS
  - Weighted Boxes Fusion (WBF)
  - Non-Maximum Weighted (NMW)
"""

from __future__ import annotations

import functools

import numpy as np


def normalize_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Normalize boxes to [0, 1] range (required by ensemble_boxes library)."""
    norm = boxes.copy().astype(float)
    norm[:, [0, 2]] /= img_w
    norm[:, [1, 3]] /= img_h
    return np.clip(norm, 0, 1)


def denormalize_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Denormalize boxes back to pixel coordinates."""
    denorm = boxes.copy().astype(float)
    denorm[:, [0, 2]] *= img_w
    denorm[:, [1, 3]] *= img_h
    return denorm


def apply_wbf(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    weights: list[float] | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.01,
    img_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Weighted Boxes Fusion.

    Reference: Solovyev et al., "Weighted boxes fusion: Ensembling boxes
    from different object detection models", arXiv:1910.13302

    Args:
        boxes_list:  list of (N_i, 4) arrays in xyxy pixel coords
        scores_list: list of (N_i,) score arrays
        labels_list: list of (N_i,) label arrays
        weights:     per-model weights
        iou_thr:     IoU threshold for fusion
        skip_box_thr: min score to keep a box
        img_size:    (width, height) for normalization

    Returns:
        (boxes, scores, labels) - fused results in pixel coords
    """
    from ensemble_boxes import weighted_boxes_fusion

    img_w, img_h = img_size

    # Normalize boxes to [0, 1]
    norm_boxes = []
    for boxes in boxes_list:
        if len(boxes) > 0:
            norm_boxes.append(normalize_boxes(boxes, img_w, img_h))
        else:
            norm_boxes.append(np.zeros((0, 4)))

    # Convert to lists of lists (required by ensemble_boxes)
    boxes_in = [b.tolist() for b in norm_boxes]
    scores_in = [s.tolist() for s in scores_list]
    labels_in = [lbl.tolist() for lbl in labels_list]

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_in,
        scores_in,
        labels_in,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    # Denormalize back to pixel coords
    fused_boxes = denormalize_boxes(np.array(fused_boxes), img_w, img_h)

    return fused_boxes, np.array(fused_scores), np.array(fused_labels, dtype=int)


def apply_nms(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    weights: list[float] | None = None,
    iou_thr: float = 0.5,
    img_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply standard NMS ensemble.

    Concatenates all boxes from all models, then applies NMS.
    """
    from ensemble_boxes import nms

    img_w, img_h = img_size

    norm_boxes = [
        normalize_boxes(b, img_w, img_h).tolist() if len(b) > 0 else [] for b in boxes_list
    ]
    scores_in = [s.tolist() for s in scores_list]
    labels_in = [lbl.tolist() for lbl in labels_list]

    fused_boxes, fused_scores, fused_labels = nms(
        norm_boxes,
        scores_in,
        labels_in,
        weights=weights,
        iou_thr=iou_thr,
    )

    fused_boxes = denormalize_boxes(np.array(fused_boxes), img_w, img_h)
    return fused_boxes, np.array(fused_scores), np.array(fused_labels, dtype=int)


def apply_soft_nms(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    weights: list[float] | None = None,
    iou_thr: float = 0.5,
    sigma: float = 0.5,
    thresh: float = 0.01,
    img_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Soft-NMS ensemble.
    Uses Gaussian decay instead of hard suppression.
    """
    from ensemble_boxes import soft_nms

    img_w, img_h = img_size

    norm_boxes = [
        normalize_boxes(b, img_w, img_h).tolist() if len(b) > 0 else [] for b in boxes_list
    ]
    scores_in = [s.tolist() for s in scores_list]
    labels_in = [lbl.tolist() for lbl in labels_list]

    fused_boxes, fused_scores, fused_labels = soft_nms(
        norm_boxes,
        scores_in,
        labels_in,
        weights=weights,
        iou_thr=iou_thr,
        sigma=sigma,
        thresh=thresh,
    )

    fused_boxes = denormalize_boxes(np.array(fused_boxes), img_w, img_h)
    return fused_boxes, np.array(fused_scores), np.array(fused_labels, dtype=int)


def apply_nmw(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    weights: list[float] | None = None,
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.01,
    img_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Non-Maximum Weighted (NMW) ensemble.
    """
    from ensemble_boxes import non_maximum_weighted

    img_w, img_h = img_size

    norm_boxes = [
        normalize_boxes(b, img_w, img_h).tolist() if len(b) > 0 else [] for b in boxes_list
    ]
    scores_in = [s.tolist() for s in scores_list]
    labels_in = [lbl.tolist() for lbl in labels_list]

    fused_boxes, fused_scores, fused_labels = non_maximum_weighted(
        norm_boxes,
        scores_in,
        labels_in,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    fused_boxes = denormalize_boxes(np.array(fused_boxes), img_w, img_h)
    return fused_boxes, np.array(fused_scores), np.array(fused_labels, dtype=int)


# Registry of methods
ENSEMBLE_METHODS = {
    "wbf": apply_wbf,
    "nms": apply_nms,
    "soft_nms": apply_soft_nms,
    "nmw": apply_nmw,
}


def fusion_extra_kwargs(
    method_name: str,
    *,
    skip_box_thr: float = 0.01,
    sigma: float = 0.5,
) -> dict:
    """Build the method-specific fusion kwargs.

    Each fusion function accepts a different subset of tuning params:
      - wbf / nmw  -> skip_box_thr
      - soft_nms   -> sigma, thresh (the score floor, mapped from skip_box_thr)
      - nms        -> none

    Passing the wrong kwarg to a method raises TypeError, so callers must route
    config values (ensemble.skip_box_threshold, ensemble.sigma) through here
    instead of forwarding blindly. See audit findings C2/C3.
    """
    if method_name in ("wbf", "nmw"):
        return {"skip_box_thr": skip_box_thr}
    if method_name == "soft_nms":
        return {"sigma": sigma, "thresh": skip_box_thr}
    return {}


@functools.cache
def get_ensemble_method(method_name: str):
    """Get ensemble method function by name."""
    if method_name not in ENSEMBLE_METHODS:
        raise ValueError(
            f"Unknown ensemble method: {method_name}. Available: {list(ENSEMBLE_METHODS.keys())}"
        )
    return ENSEMBLE_METHODS[method_name]
