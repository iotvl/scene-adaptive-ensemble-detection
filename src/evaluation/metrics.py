"""
Evaluation metrics for object detection.
Includes mAP calculation, per-class AP, and additional metrics.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.constants import OLD10_TO_NEW8


def remap_10class_to_8class(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a 10-class detector's predictions into the project's 8-class space.

    The externally-pretrained VisDrone checkpoints (e.g. ``mshamrai/yolov8*-visdrone``)
    output the original 10-class layout where 6=tricycle, 7=awning-tricycle, 8=bus,
    9=motor. This project keeps 8 classes (6=bus, 7=motor). Scoring the raw 10-class
    predictions against the 8-class ground truth silently mismatches bus/motor and
    turns tricycle/awning into phantom false positives, so callers must remap first.

    Drops predictions whose class id is not in ``OLD10_TO_NEW8`` (tricycle/awning).

    Args:
        boxes: (N, 4) xyxy.
        scores: (N,) confidences.
        labels: (N,) 10-class ids.

    Returns:
        (boxes, scores, labels) filtered and remapped into 0..7.
    """
    if labels.size == 0:
        return boxes, scores, labels.astype(int)
    keep = np.array([int(c) in OLD10_TO_NEW8 for c in labels])
    if not keep.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    new_labels = np.array([OLD10_TO_NEW8[int(c)] for c in labels[keep]], dtype=int)
    return boxes[keep], scores[keep], new_labels


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in xyxy format.

    Args:
        box1: (4,) [x1, y1, x2, y2]
        box2: (4,) [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU matrix.

    Args:
        boxes1: (N, 4) array
        boxes2: (M, 4) array

    Returns:
        (N, M) IoU matrix
    """
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / np.maximum(union, 1e-8)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision (area under P-R curve with interpolation).
    Uses the 101-point interpolation (COCO-style).
    """
    # Add sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    precision_interp = np.zeros_like(recall_points)

    for i, r in enumerate(recall_points):
        idx = np.where(mrec >= r)[0]
        if len(idx) > 0:
            precision_interp[i] = mpre[idx[0]]

    return precision_interp.mean()


def compute_per_class_ap(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    num_classes: int = 8,
) -> dict[int, float]:
    """
    Compute per-class Average Precision.

    Args:
        predictions: list of dicts with keys:
            "image_id", "boxes" (N,4), "scores" (N,), "labels" (N,)
        ground_truths: list of dicts with keys:
            "image_id", "boxes" (M,4), "labels" (M,)
        iou_threshold: IoU threshold
        num_classes: total number of classes

    Returns:
        Dict mapping class_id -> AP
    """
    # Organize GT by image and class
    gt_by_image = defaultdict(list)
    for gt in ground_truths:
        gt_by_image[gt["image_id"]].append(gt)

    ap_per_class = {}

    for cls_id in range(num_classes):
        # Collect all predictions for this class
        class_preds = []
        for pred in predictions:
            mask = pred["labels"] == cls_id
            if mask.any():
                for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
                    class_preds.append(
                        {
                            "image_id": pred["image_id"],
                            "box": box,
                            "score": score,
                        }
                    )

        if not class_preds:
            ap_per_class[cls_id] = 0.0
            continue

        # Sort by confidence (descending)
        class_preds.sort(key=lambda x: x["score"], reverse=True)

        # Count total GT for this class
        total_gt = 0
        gt_matched = defaultdict(lambda: np.zeros(0))
        for img_id, gts in gt_by_image.items():
            for gt in gts:
                mask = gt["labels"] == cls_id
                total_gt += mask.sum()
                gt_matched[img_id] = np.zeros(int(mask.sum()), dtype=bool)

        if total_gt == 0:
            ap_per_class[cls_id] = 0.0
            continue

        # Compute TP/FP
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, pred in enumerate(class_preds):
            img_id = pred["image_id"]
            pred_box = pred["box"]

            # Find GT boxes for this class in this image
            gt_boxes_for_img = []
            for gt in gt_by_image.get(img_id, []):
                mask = gt["labels"] == cls_id
                if mask.any():
                    gt_boxes_for_img.append(gt["boxes"][mask])

            if not gt_boxes_for_img:
                fp[i] = 1
                continue

            gt_boxes = np.concatenate(gt_boxes_for_img, axis=0)

            # Compute IoU with all GT boxes
            ious = np.array([compute_iou(pred_box, gb) for gb in gt_boxes])
            best_idx = np.argmax(ious)
            best_iou = ious[best_idx]

            if best_iou >= iou_threshold and not gt_matched[img_id][best_idx]:
                tp[i] = 1
                gt_matched[img_id][best_idx] = True
            else:
                fp[i] = 1

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / total_gt
        denom = tp_cumsum + fp_cumsum
        precision = np.divide(
            tp_cumsum, denom, where=denom > 0, out=np.zeros_like(tp_cumsum, dtype=float)
        )

        ap_per_class[cls_id] = compute_ap(recall, precision)

    return ap_per_class


def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_thresholds: list[float] | None = None,
    num_classes: int = 8,
) -> dict[str, float]:
    """
    Compute mAP at various IoU thresholds.

    Returns:
        Dict with "mAP50", "mAP75", "mAP50-95", and per-class APs
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    results = {}
    all_aps = []

    for iou_thr in iou_thresholds:
        per_class = compute_per_class_ap(predictions, ground_truths, iou_thr, num_classes)
        mean_ap = np.mean(list(per_class.values()))
        all_aps.append(mean_ap)

        if abs(iou_thr - 0.5) < 1e-6:
            results["mAP50"] = mean_ap
            for cls_id, ap in per_class.items():
                results[f"AP50_class{cls_id}"] = ap

        if abs(iou_thr - 0.75) < 1e-6:
            results["mAP75"] = mean_ap

    results["mAP50-95"] = np.mean(all_aps)

    return results
