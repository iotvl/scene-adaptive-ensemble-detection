"""Ground-truth helpers: COCO export (for pycocotools cross-checks) and the
empirical box-size prior of a class on the TRAIN split (for the injection
attack A2 — phantoms must look like real objects of the target class)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exp36.loaders.cache import ROOT, Dataset, _load_script08

VISDRONE_CAT_TO_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 9: 6, 10: 7}  # VISDRONE_TO_YOLO_CLASS


def scene_to_coco(scene_gt: list[dict], num_classes: int) -> dict:
    images, anns = [], []
    ann_id = 1
    for i, gt in enumerate(scene_gt):
        images.append({"id": i})
        for box, label in zip(gt["boxes"], gt["labels"]):
            x1, y1, x2, y2 = (float(v) for v in box)
            anns.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "images": images,
        "annotations": anns,
        "categories": [{"id": c + 1} for c in range(num_classes)],
    }


def export_coco(ds: Dataset, out_dir: Path) -> dict[str, Path]:
    """Writes data/{dataset}_{scene}_coco.json per scene (spec 1.2)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for scene, sd in ds.scenes.items():
        p = out_dir / f"{ds.name}_{scene}_coco.json"
        with open(p, "w") as f:
            json.dump(scene_to_coco(sd.gt_list, ds.num_classes), f)
        paths[scene] = p
    return paths


def box_prior(dataset: str, cls: int) -> np.ndarray:
    """(n, 2) array of (w, h) in pixels of every TRAIN instance of `cls`."""
    if dataset == "visdrone":
        ann = ROOT / "data/VisDrone/VisDrone2019-DET-train/annotations"
        wh = []
        for f in sorted(ann.glob("*.txt")):
            for line in open(f):
                p = line.strip().split(",")
                if len(p) < 6:
                    continue
                w, h, cat = float(p[2]), float(p[3]), int(p[5])
                if VISDRONE_CAT_TO_IDX.get(cat) == cls and w > 0 and h > 0:
                    wh.append((w, h))
        return np.asarray(wh, dtype=float)
    if dataset == "uavdt":
        u08 = _load_script08()
        root = ROOT / "data/UAVDT"
        weather = u08.read_weather(root / "M_attr")
        train_seqs = {
            f.stem.replace("_attr", "").strip()
            for f in (root / "M_attr" / "train").glob("*_attr.txt")
        }
        wh = []
        for seq in sorted(train_seqs):
            if seq not in weather:
                continue
            gt = u08.parse_boxes_by_frame(
                root / "UAV-benchmark-MOTD_v1.0" / "GT" / f"{seq}_gt_whole.txt", True
            )
            for entries in gt.values():
                for (x1, y1, x2, y2), c in entries:
                    if c == cls and x2 > x1 and y2 > y1:
                        wh.append((x2 - x1, y2 - y1))
        return np.asarray(wh, dtype=float)
    raise ValueError(dataset)
