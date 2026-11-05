"""pycocotools reference scorer (cross-check only). The grid itself uses the
project's compute_map-equivalent ConfigStats, which scripts/09 already
verified against COCOeval (|diff| <= 0.0006 mAP@50); this module re-runs that
verification on the exp36 fusion path so the number can be quoted."""

from __future__ import annotations

import contextlib
import io

from exp36.loaders.gt import scene_to_coco


def coco_map(fused_preds: list[dict], gt_list: list[dict], num_classes: int) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dets = []
    for i, pred in enumerate(fused_preds):
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            x1, y1, x2, y2 = (float(v) for v in box)
            dets.append(
                {
                    "image_id": i,
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = scene_to_coco(gt_list, num_classes)
        coco_gt.createIndex()
        if not dets:
            return {"mAP50": 0.0, "mAP50-95": 0.0}
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 10, 1000]  # fused outputs can exceed 300 per image
        ev.evaluate()
        ev.accumulate()
    precision = ev.eval["precision"][:, :, :, 0, -1]
    valid = precision > -1
    p50 = precision[0]
    return {"mAP50": float(p50[p50 > -1].mean()), "mAP50-95": float(precision[valid].mean())}
