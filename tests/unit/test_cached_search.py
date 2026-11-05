import numpy as np

from src.ensemble.cached_search import (
    evaluate_cached_ensemble,
    fuse_cached_image,
    grid_search_weights,
)
from src.evaluation.metrics import compute_map


def _toy_scene():
    gt_list = [
        {"boxes": np.array([[10.0, 10.0, 50.0, 50.0]]), "labels": np.array([3])},
        {"boxes": np.array([[20.0, 20.0, 60.0, 60.0]]), "labels": np.array([3])},
    ]
    img_whs = [(640, 640), (640, 640)]
    cache = {
        "m1": [
            {
                "boxes": np.array([[10.0, 10.0, 50.0, 50.0]]),
                "scores": np.array([0.9]),
                "labels": np.array([3]),
            },
            {
                "boxes": np.array([[20.0, 20.0, 60.0, 60.0]]),
                "scores": np.array([0.8]),
                "labels": np.array([3]),
            },
        ]
    }
    return cache, gt_list, img_whs


def test_fuse_two_identical_boxes_merges_to_one():
    boxes_list = [np.array([[10.0, 10.0, 50.0, 50.0]]), np.array([[10.0, 10.0, 50.0, 50.0]])]
    scores_list = [np.array([0.9]), np.array([0.8])]
    labels_list = [np.array([3]), np.array([3])]

    boxes, scores, labels = fuse_cached_image(
        boxes_list, scores_list, labels_list, img_wh=(640, 640), method="wbf", weights=[1.0, 1.0]
    )

    assert len(boxes) == 1
    assert int(labels[0]) == 3
    assert 0.8 <= float(scores[0]) <= 0.9  # WBF averages member scores: (0.9 + 0.8) / 2 ≈ 0.85


def test_single_member_fusion_reproduces_solo_map():
    cache, gt_list, img_whs = _toy_scene()
    solo_preds = [{"image_id": i, **cache["m1"][i]} for i in range(len(gt_list))]
    solo_gt = [{"image_id": i, **gt_list[i]} for i in range(len(gt_list))]
    solo = compute_map(solo_preds, solo_gt)["mAP50"]

    fused = evaluate_cached_ensemble(cache, ["m1"], "wbf", [1.0], gt_list, img_whs, iou_thr=0.55)[
        "mAP50"
    ]

    assert abs(fused - solo) < 1e-6


def test_grid_search_returns_weights_and_metrics():
    cache, gt_list, img_whs = _toy_scene()
    best_weights, best_metrics = grid_search_weights(
        cache, ["m1"], "wbf", gt_list, img_whs, grid=[0.5, 1.0, 2.0], iou_thr=0.55
    )
    assert isinstance(best_weights, list) and len(best_weights) == 1
    assert best_weights[0] in (0.5, 1.0, 2.0)
    assert "mAP50" in best_metrics
    assert best_metrics["mAP50"] > 0.0


def test_grid_search_two_members_not_worse_than_uniform():
    # 1 image, 1 GT. m1 predicts it correctly; m2 injects a far-away false positive.
    gt_list = [{"boxes": np.array([[10.0, 10.0, 50.0, 50.0]]), "labels": np.array([3])}]
    img_whs = [(640, 640)]
    cache = {
        "m1": [
            {
                "boxes": np.array([[10.0, 10.0, 50.0, 50.0]]),
                "scores": np.array([0.9]),
                "labels": np.array([3]),
            }
        ],
        "m2": [
            {
                "boxes": np.array([[300.0, 300.0, 340.0, 340.0]]),
                "scores": np.array([0.9]),
                "labels": np.array([3]),
            }
        ],
    }
    grid = [0.5, 1.0, 2.0]
    uniform = evaluate_cached_ensemble(cache, ["m1", "m2"], "wbf", [1.0, 1.0], gt_list, img_whs)[
        "mAP50"
    ]
    best_weights, best_metrics = grid_search_weights(
        cache, ["m1", "m2"], "wbf", gt_list, img_whs, grid=grid
    )
    assert len(best_weights) == 2
    assert all(w in grid for w in best_weights)
    assert best_metrics["mAP50"] >= uniform
