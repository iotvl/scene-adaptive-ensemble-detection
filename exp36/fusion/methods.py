"""Box-fusion methods for exp36, applied PER CLASS.

All five methods treat classes independently (ensemble_boxes groups by label
before anything else; our WBF does the same), so fusing one class at a time
and concatenating is exactly equivalent to fusing everything at once — this is
what lets the grid reuse per-class fusions across attack configurations
(tests/test_fusion.py::test_per_class_decomposition proves the identity).

    nms, soft_nms, nmw  -> ensemble_boxes through src.ensemble.cached_search
                           (identical to article TIT, scripts/14)
    wbf                 -> own implementation mirroring ensemble_boxes' WBF
                           (conf_type='avg', allows_overflow=False), so that the
                           confirmation multiplier can be switched off
    wbf_no_conf         -> same with confirmation=False (the ablation)

fusion/crosscheck.py verifies own wbf == ensemble_boxes wbf on real data.
"""

from __future__ import annotations

import re

import numpy as np

from src.ensemble.cached_search import fuse_cached_image

METHODS = ["nms", "soft_nms", "nmw", "wbf", "wbf_no_conf"]

# Mitigation for A1 (thesis §3.6 / §3.8): a capped confirmation multiplier
# min(T, N - k_max) / (N - k_max) that assumes up to k_max members may be
# compromised, so a cluster supported by N - k_max honest members already gets
# full confidence. Method names "wbf_cap1", "wbf_cap2", ... ; k_max is parsed.
_CAP_RE = re.compile(r"^wbf_cap(\d+)$")


def cap_kmax(method: str) -> int | None:
    """k_max of a capped-WBF method name, else None."""
    m = _CAP_RE.match(method)
    return int(m.group(1)) if m else None


def is_wbf_family(method: str) -> bool:
    """Methods whose confirmation multiplier depends on the pool size N."""
    return method in ("wbf", "wbf_no_conf") or cap_kmax(method) is not None


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xa = np.maximum(boxes[:, 0], box[0])
    ya = np.maximum(boxes[:, 1], box[1])
    xb = np.minimum(boxes[:, 2], box[2])
    yb = np.minimum(boxes[:, 3], box[3])
    inter = np.maximum(xb - xa, 0) * np.maximum(yb - ya, 0)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    return inter / (area + area_b - inter)


def wbf_single_class(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    n_models: int,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.01,
    confirmation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted boxes fusion of ONE class, boxes already normalised to [0, 1].

    Mirrors ensemble_boxes.weighted_boxes_fusion with equal weights,
    conf_type='avg', allows_overflow=False:
      1. pool all boxes (score >= skip_box_thr, positive area), sort by score desc;
      2. greedy clustering: a box joins the cluster whose CURRENT fused box has
         the highest IoU > iou_thr, else starts a new cluster; the fused box is
         recomputed after every join (coords = score-weighted mean, conf = mean);
      3. confirmation (WBF proper): conf *= min(T, N) / N with T = cluster size,
         N = number of models in the pool — a majority-vote damping of clusters
         supported by few models. `confirmation=False` is the ablation
         (wbf_no_conf): conf stays the plain cluster mean.
    Returns (boxes (m, 4) normalised, scores (m,)), sorted by score desc.
    """
    if not boxes_list:
        return np.zeros((0, 4)), np.zeros(0)
    b = np.concatenate([np.asarray(x, dtype=float).reshape(-1, 4) for x in boxes_list])
    s = np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in scores_list])
    if len(b) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    b = np.clip(b, 0.0, 1.0)
    # ensemble_boxes swaps inverted coords; our caches never have them, but keep parity
    x1 = np.minimum(b[:, 0], b[:, 2])
    x2 = np.maximum(b[:, 0], b[:, 2])
    y1 = np.minimum(b[:, 1], b[:, 3])
    y2 = np.maximum(b[:, 1], b[:, 3])
    b = np.stack([x1, y1, x2, y2], axis=1)
    keep = (s >= skip_box_thr) & ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) > 0)
    b, s = b[keep], s[keep]
    if len(b) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    order = np.argsort(-s, kind="stable")  # argsort()[::-1] in the library; ties are rare
    b, s = b[order], s[order]

    fused = np.zeros((0, 4))  # current fused coords per cluster
    conf_sum = np.zeros(0)  # sum of member scores per cluster
    coord_sum = np.zeros((0, 4))  # sum of score * coords per cluster
    size = np.zeros(0, dtype=int)
    for j in range(len(b)):
        idx = -1
        if len(fused):
            ious = _iou_one_to_many(b[j], fused)
            k = int(np.argmax(ious))
            if ious[k] > iou_thr:
                idx = k
        if idx == -1:
            fused = np.vstack([fused, b[j][None]])
            conf_sum = np.append(conf_sum, s[j])
            coord_sum = np.vstack([coord_sum, (s[j] * b[j])[None]])
            size = np.append(size, 1)
        else:
            conf_sum[idx] += s[j]
            coord_sum[idx] += s[j] * b[j]
            size[idx] += 1
            fused[idx] = coord_sum[idx] / conf_sum[idx]
    conf = conf_sum / size
    if confirmation:
        conf = conf * np.minimum(size, n_models) / n_models
    order = np.argsort(-conf, kind="stable")
    return fused[order], conf[order]


def fuse_one_class(
    method: str,
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    cls: int,
    img_wh: tuple[int, int],
    n_models: int,
    iou_thr: float,
    skip_box_thr: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse the boxes of one class coming from several members.

    `boxes_list[m]` are the class-`cls` boxes of member m in abs pixels (the
    member list must already reflect the attack: suppressed members contribute
    empty arrays or are absent, injected members carry their phantoms).
    `n_models` is the pool size N used by WBF's confirmation multiplier — it is
    the pool size, not the number of members that still have boxes (an absent
    member is exactly what the multiplier is supposed to penalise).
    Returns (boxes abs pixels, scores).
    """
    boxes_list = [np.asarray(x, dtype=float).reshape(-1, 4) for x in boxes_list]
    scores_list = [np.asarray(x, dtype=float).reshape(-1) for x in scores_list]
    if sum(len(x) for x in boxes_list) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    w, h = img_wh
    if is_wbf_family(method):
        norm = [x / np.array([w, h, w, h], dtype=float) for x in boxes_list]
        kmax = cap_kmax(method)
        n_eff = n_models if kmax is None else max(1, n_models - kmax)
        fb, fs = wbf_single_class(
            norm, scores_list, n_eff, iou_thr, skip_box_thr, confirmation=(method != "wbf_no_conf")
        )
        return fb * np.array([w, h, w, h], dtype=float), fs
    if method in ("nms", "soft_nms", "nmw"):
        # ensemble_boxes drops boxes whose clipped area is zero and then crashes if a class
        # is left with nothing; drop them here first (identical result, no crash).
        kept = []
        for bx, sc in zip(boxes_list, scores_list):
            c = np.clip(bx, [0, 0, 0, 0], [w, h, w, h])
            ok = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1]) > 0
            kept.append((bx[ok], sc[ok]))
        boxes_list = [bx for bx, _ in kept]
        scores_list = [sc for _, sc in kept]
        if sum(len(x) for x in scores_list) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        # ensemble_boxes' nms path cannot mix empty members with non-empty ones (scripts/14).
        # weights=None on purpose: with weights, ensemble_boxes' nms_method rescales every
        # score by w_i / sum(w) (i.e. divides by N) — invisible to mAP but fatal for the
        # deploy-threshold metrics and for mixing runs of different member counts.
        nonempty = [(bx, sc) for bx, sc in zip(boxes_list, scores_list) if len(sc) > 0]
        bl = [bx for bx, _ in nonempty]
        sl = [sc for _, sc in nonempty]
        ll = [np.full(len(sc), cls, dtype=int) for sc in sl]
        fb, fs, fl = fuse_cached_image(
            bl, sl, ll, img_wh, method, None, iou_thr, skip_box_thr, sigma
        )
        return np.asarray(fb, dtype=float).reshape(-1, 4), np.asarray(fs, dtype=float).reshape(-1)
    raise ValueError(method)


def fuse_all_classes(
    method, preds_list, img_wh, num_classes, n_models, iou_thr, skip_box_thr, sigma
):
    """Convenience: fuse every class of an image; returns one prediction dict.
    Used by the cross-checks; the grid works per class."""
    boxes, scores, labels = [], [], []
    for c in range(num_classes):
        bl = [p["boxes"][p["labels"] == c] for p in preds_list]
        sl = [p["scores"][p["labels"] == c] for p in preds_list]
        fb, fs = fuse_one_class(method, bl, sl, c, img_wh, n_models, iou_thr, skip_box_thr, sigma)
        boxes.append(fb)
        scores.append(fs)
        labels.append(np.full(len(fs), c, dtype=int))
    return {
        "boxes": np.concatenate(boxes) if boxes else np.zeros((0, 4)),
        "scores": np.concatenate(scores) if scores else np.zeros(0),
        "labels": np.concatenate(labels) if labels else np.zeros(0, dtype=int),
    }
