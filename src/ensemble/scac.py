"""Shared core of the SCAC (scene-complexity adaptive cascade) experiments.

The complexity-signal definitions and the per-image/mixture scoring below ARE
the method the E9 gates report on, so scripts 12 (VisDrone), 12b (UAVDT gate)
and 12c (UAVDT crossfit) must all use this single implementation — divergent
copies would silently make the gates measure different signals and break the
cross-dataset comparison.

All mAP scoring here is mAP@50-only (iou_thresholds=[0.5]): the sweeps read
nothing else, and the default 10-threshold compute_map would pay ~10x the
matching cost for discarded numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import compute_map

DEPLOY_CONF = 0.25  # complexity signals see only deploy-visible detections
TOPK_CONF = 10
IOU50 = [0.5]


def map50(preds: list[dict], gts: list[dict], num_classes: int) -> float:
    """mAP@50 of one prediction set (adds image_ids, single IoU threshold)."""
    return compute_map(
        [{"image_id": i, **p} for i, p in enumerate(preds)],
        [{"image_id": i, **g} for i, g in enumerate(gts)],
        iou_thresholds=IOU50,
        num_classes=num_classes,
    )["mAP50"]


def per_image_ap50(pred: dict, gt: dict, num_classes: int) -> float | None:
    """AP@50 of one image against its own GT; None when the image has no GT.

    NOTE: the value averages over ALL num_classes (zero-GT classes contribute
    0), so magnitudes are NOT comparable across label spaces (VisDrone /8 vs
    UAVDT /3); only rankings and same-space deltas are.
    """
    if len(gt["labels"]) == 0:
        return None
    return map50([pred], [gt], num_classes)


def c_count(pred: dict) -> float:
    """Deploy-visible detection count — the gates' best realizable signal."""
    return float((np.asarray(pred["scores"]) >= DEPLOY_CONF).sum())


def signal_values(t1_pred: dict) -> dict[str, float]:
    """GT-free complexity signals from tier-1 predictions.
    Higher = more complex = escalate first."""
    s = np.asarray(t1_pred["scores"], dtype=float)
    vis = s[s >= DEPLOY_CONF]
    if vis.size == 0:
        # tier 1 sees nothing it trusts: maximal uncertainty for conf/entropy,
        # but zero density for count (an empty frame is not a dense frame).
        return {"C-conf": 1.0, "C-count": 0.0, "C-entropy": 1.0}
    topk = np.sort(vis)[::-1][:TOPK_CONF]
    p = np.clip(vis, 1e-6, 1 - 1e-6)
    bin_entropy = float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))
    return {
        "C-conf": 1.0 - float(topk.mean()),
        "C-count": float(vis.size),
        "C-entropy": bin_entropy,
    }


def eval_split(
    scene_slices: dict[str, tuple[int, int]],
    mixed_preds: list[dict],
    gt_all: list[dict],
    n_images: dict[str, int],
    num_classes: int,
    include_avg: bool = False,
) -> dict:
    """Per-scene mAP@50 (+ optional scene-balanced avg) + image-weighted avg."""
    out = {}
    for scene, (lo, hi) in scene_slices.items():
        out[scene] = round(
            map50(mixed_preds[lo:hi], gt_all[lo:hi], num_classes), 4
        )
    total = sum(n_images.values())
    if include_avg:
        out["avg"] = round(float(np.mean([out[s] for s in n_images])), 4)
    out["avg_weighted"] = round(
        sum(out[s] * n_images[s] for s in n_images) / total, 4
    )
    return out


def load_sweep_checkpoint(
    ckpt_path: Path, fingerprint: dict, curve_names: set[str]
) -> dict | None:
    """Resume-safe read of a partial sweep: the checkpoint is reused only when
    the FULL run fingerprint (p-grid, tier1, members, draw count, ...) matches
    — a stale checkpoint from a differently-configured run must never be
    spliced into new results."""
    if not ckpt_path.exists():
        return None
    saved = json.load(open(ckpt_path))
    if saved.get("fingerprint") != fingerprint:
        return None
    if set(saved.get("curves", {})) != curve_names:
        return None
    return saved["curves"]


def save_sweep_checkpoint(ckpt_path: Path, fingerprint: dict, curves: dict) -> None:
    with open(ckpt_path, "w") as f:
        json.dump({"fingerprint": fingerprint, "curves": curves}, f)
