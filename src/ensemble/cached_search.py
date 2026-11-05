"""Pure fusion + weight search over cached per-model predictions (no models loaded).

Decouples inference from fusion: callers cache each model's predictions once, then
this module fuses/searches over those cached boxes. Fusion uses the exact same
helpers as the live EnsembleDetector (src/ensemble/methods.py), so cached-path
results match the live path.
"""

from __future__ import annotations

import itertools

import numpy as np

from src.ensemble.methods import fusion_extra_kwargs, get_ensemble_method
from src.evaluation.metrics import compute_map


def fuse_cached_image(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    img_wh: tuple[int, int],
    method: str = "wbf",
    weights: list[float] | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.01,
    sigma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse one image's per-member predictions. Mirrors EnsembleDetector.predict's
    fusion call: fn(boxes_list, scores_list, labels_list, weights, iou_thr, img_size=(w,h)).

    img_wh = (width, height) — matches methods.py::apply_wbf which documents
    img_size as (width, height). skip_box_thr/sigma are routed per method so the
    cached search tunes weights under the same thresholds as deployment (audit C3).
    """
    fn = get_ensemble_method(method)
    return fn(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        img_size=img_wh,
        **fusion_extra_kwargs(method, skip_box_thr=skip_box_thr, sigma=sigma),
    )


def evaluate_cached_ensemble(
    cache: dict[str, list[dict]],
    members: list[str],
    method: str,
    weights: list[float] | None,
    gt_list: list[dict],
    img_whs: list[tuple[int, int]],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.01,
    sigma: float = 0.5,
    num_classes: int = 8,
) -> dict:
    """Fuse `members`' cached predictions per image and score with compute_map.

    cache[model][i] = {"boxes","scores","labels"} for image i (already 10->8 remapped).
    gt_list[i]      = {"boxes","labels"} for image i.
    num_classes: label space of preds AND gt (8 = VisDrone; UAVDT callers pass 3 —
    the default silently deflates 3-class mAP to 3/8 otherwise).
    """
    predictions = []
    for i, gt_entry in enumerate(gt_list):
        boxes_list = [cache[m][i]["boxes"] for m in members]
        scores_list = [cache[m][i]["scores"] for m in members]
        labels_list = [cache[m][i]["labels"] for m in members]
        fb, fs, fl = fuse_cached_image(
            boxes_list,
            scores_list,
            labels_list,
            img_whs[i],
            method,
            weights,
            iou_thr,
            skip_box_thr,
            sigma,
        )
        predictions.append({"image_id": i, "boxes": fb, "scores": fs, "labels": fl})

    gt = [{"image_id": i, **gt_entry} for i, gt_entry in enumerate(gt_list)]
    return compute_map(predictions, gt, num_classes=num_classes)


def grid_search_weights(
    cache: dict[str, list[dict]],
    members: list[str],
    method: str,
    gt_list: list[dict],
    img_whs: list[tuple[int, int]],
    grid: list[float],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.01,
    sigma: float = 0.5,
    num_classes: int = 8,
) -> tuple[list[float], dict]:
    """Exhaustive grid over per-member weights; maximize mAP50 on cached preds.

    Returned weights are always drawn from `grid`. With 2 members and a 4-value
    grid this is 16 evaluations; 3 members -> 64. Each evaluation only re-runs
    fusion (cheap), never inference.
    """
    best_weights: list[float] | None = None
    best_metrics: dict = {}
    best_map = -1.0
    for combo in itertools.product(grid, repeat=len(members)):
        weights = list(combo)
        metrics = evaluate_cached_ensemble(
            cache, members, method, weights, gt_list, img_whs, iou_thr, skip_box_thr,
            sigma, num_classes,
        )
        if metrics["mAP50"] > best_map:
            best_map = metrics["mAP50"]
            best_weights, best_metrics = weights, metrics
    assert best_weights is not None  # grid is always non-empty
    return best_weights, best_metrics
