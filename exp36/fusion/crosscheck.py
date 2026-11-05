"""Cross-checks that make the numbers quotable (spec stage 2 + stage 4):

1. own WBF vs ensemble_boxes.weighted_boxes_fusion on 100 random images of
   the clean pool: |dmAP@50| < 0.002, |d#boxes| < 2 %, plus the max abs
   coordinate/score difference of the matched outputs;
2. ConfigStats.map50 (grid scorer) vs src compute_map vs pycocotools on the
   clean pools, every scene: the grid scorer must equal compute_map exactly
   and stay within the scripts/09 tolerance of pycocotools.
"""

from __future__ import annotations

import numpy as np

from exp36.eval.coco_map import coco_map
from exp36.eval.matching import ConfigStats, match_frame_class
from exp36.fusion.methods import fuse_all_classes
from exp36.loaders.cache import Dataset
from src.ensemble.cached_search import fuse_cached_image
from src.evaluation.metrics import compute_map


def _fuse_scene(method, sd, pool, num_classes, fp, own=True):
    out = []
    for i in range(sd.n_images):
        preds = [sd.preds[m][i] for m in pool]
        if own:
            out.append(
                fuse_all_classes(
                    method,
                    preds,
                    sd.whs[i],
                    num_classes,
                    len(pool),
                    fp["iou_thr"],
                    fp["skip_box_thr"],
                    fp["soft_nms_sigma"],
                )
            )
        else:
            fb, fs, fl = fuse_cached_image(
                [p["boxes"] for p in preds],
                [p["scores"] for p in preds],
                [p["labels"] for p in preds],
                sd.whs[i],
                "wbf",
                [1.0] * len(pool),
                fp["iou_thr"],
                fp["skip_box_thr"],
                fp["soft_nms_sigma"],
            )
            out.append(
                {
                    "boxes": np.asarray(fb).reshape(-1, 4),
                    "scores": np.asarray(fs).reshape(-1),
                    "labels": np.asarray(fl).astype(int).reshape(-1),
                }
            )
    return out


def wbf_vs_library(
    ds: Dataset, pool: list[str], fp: dict, n_images: int = 100, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    report = {"n_images": 0, "scenes": {}}
    tot_own = tot_lib = 0
    max_coord = max_score = 0.0
    for scene, sd in ds.scenes.items():
        idx = rng.choice(
            sd.n_images, size=min(n_images // len(ds.scenes) + 1, sd.n_images), replace=False
        )
        sub = type(sd)(
            sd.name,
            [sd.gt_list[i] for i in idx],
            [sd.whs[i] for i in idx],
            [sd.image_files[i] for i in idx],
            {m: [sd.preds[m][i] for i in idx] for m in pool},
        )
        own = _fuse_scene("wbf", sub, pool, ds.num_classes, fp, own=True)
        lib = _fuse_scene("wbf", sub, pool, ds.num_classes, fp, own=False)
        gt = [{"image_id": i, **g} for i, g in enumerate(sub.gt_list)]
        m_own = compute_map(
            [{"image_id": i, **p} for i, p in enumerate(own)], gt, num_classes=ds.num_classes
        )["mAP50"]
        m_lib = compute_map(
            [{"image_id": i, **p} for i, p in enumerate(lib)], gt, num_classes=ds.num_classes
        )["mAP50"]
        n_own = sum(len(p["scores"]) for p in own)
        n_lib = sum(len(p["scores"]) for p in lib)
        # element-wise comparison after sorting both outputs identically
        for a, b in zip(own, lib):
            if len(a["scores"]) == len(b["scores"]) and len(a["scores"]):
                ka = np.lexsort((a["boxes"][:, 0], a["labels"], -a["scores"]))
                kb = np.lexsort((b["boxes"][:, 0], b["labels"], -b["scores"]))
                max_coord = max(max_coord, float(np.abs(a["boxes"][ka] - b["boxes"][kb]).max()))
                max_score = max(max_score, float(np.abs(a["scores"][ka] - b["scores"][kb]).max()))
        report["scenes"][scene] = {
            "n_images": int(len(idx)),
            "map50_own": round(m_own, 4),
            "map50_lib": round(m_lib, 4),
            "n_boxes_own": n_own,
            "n_boxes_lib": n_lib,
        }
        report["n_images"] += int(len(idx))
        tot_own += n_own
        tot_lib += n_lib
    d = [abs(v["map50_own"] - v["map50_lib"]) for v in report["scenes"].values()]
    report["max_abs_dmap50"] = round(max(d), 4)
    report["rel_dboxes"] = round(abs(tot_own - tot_lib) / max(tot_lib, 1), 4)
    report["max_abs_coord_diff_px"] = round(max_coord, 4)
    report["max_abs_score_diff"] = round(max_score, 6)
    report["pass"] = bool(report["max_abs_dmap50"] < 0.002 and report["rel_dboxes"] < 0.02)
    return report


def scorer_vs_reference(ds: Dataset, pool: list[str], fp: dict, methods=("wbf", "nms")) -> dict:
    """grid scorer (ConfigStats) == compute_map (exact) and ≈ pycocotools."""
    report = {}
    for method in methods:
        for scene, sd in ds.scenes.items():
            fused = _fuse_scene(method, sd, pool, ds.num_classes, fp, own=True)
            gt = [{"image_id": i, **g} for i, g in enumerate(sd.gt_list)]
            ref = compute_map(
                [{"image_id": i, **p} for i, p in enumerate(fused)], gt, num_classes=ds.num_classes
            )
            per_class = []
            for c in range(ds.num_classes):
                per_class.append(
                    [
                        match_frame_class(
                            p["boxes"][p["labels"] == c],
                            p["scores"][p["labels"] == c],
                            g["boxes"][g["labels"] == c],
                        )
                        for p, g in zip(fused, sd.gt_list)
                    ]
                )
            st = ConfigStats(per_class)
            coco = coco_map(fused, sd.gt_list, ds.num_classes)
            report[f"{method}/{scene}"] = {
                "grid_map50": round(st.map50(), 6),
                "compute_map_map50": round(ref["mAP50"], 6),
                "pycocotools_map50": round(coco["mAP50"], 4),
                "grid_map5095": round(st.map5095(), 4),
                "compute_map_map5095": round(ref["mAP50-95"], 4),
                "pycocotools_map5095": round(coco["mAP50-95"], 4),
            }
    report["max_abs_diff_grid_vs_compute_map"] = round(
        max(abs(v["grid_map50"] - v["compute_map_map50"]) for k, v in report.items() if "/" in k), 6
    )
    report["max_abs_diff_grid_vs_pycocotools_map50"] = round(
        max(abs(v["grid_map50"] - v["pycocotools_map50"]) for k, v in report.items() if "/" in k), 4
    )
    return report
