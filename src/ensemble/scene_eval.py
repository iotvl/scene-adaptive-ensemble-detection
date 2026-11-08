"""Shared per-scene evaluation helpers for the 05b/06* ensemble experiments.

Decouples the reusable bits — GT loading, one-pass per-model prediction caching,
and the honest single-meter FPS benchmark — from any single phase script so the
ensemble-search (05b), scene-agnostic baseline (06a) and routing ablations (06b)
all share one implementation. Fusion itself lives in src/ensemble/cached_search.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import NATIVE_CACHE_IMG_TOKENS
from src.constants import VISDRONE_TO_YOLO_CLASS
from src.ensemble.ensemble_detector import EnsembleDetector
from src.evaluation.metrics import remap_10class_to_8class

# ensemble_boxes spams a per-box UserWarning for degenerate (zero-area) boxes during
# NMW/WBF; at eval conf 0.001 on dense VisDrone there are thousands. Silence it.
warnings.filterwarnings("ignore", message="Zero area box skipped")

# 4 values -> 16 evals for top2, 64 for top3 (cheap: fusion only, no inference).
WEIGHT_GRID = [0.5, 1.0, 1.5, 2.0]


def load_ground_truth(labels_dir: str, images_dir: str):
    """Load YOLO-format GT as xyxy abs-pixel boxes (mirrors scripts/05a)."""
    labels_path = Path(labels_dir)
    images_path = Path(images_dir)
    gt_list, image_files = [], []
    for label_file in sorted(labels_path.glob("*.txt")):
        img_file = images_path / (label_file.stem + ".jpg")
        if not img_file.exists():
            continue
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        h, w = image.shape[:2]
        boxes, labels = [], []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append(
                    [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                )
                labels.append(cls_id)
        gt_list.append(
            {
                "boxes": np.array(boxes) if boxes else np.zeros((0, 4)),
                "labels": np.array(labels, dtype=int) if labels else np.zeros(0, dtype=int),
            }
        )
        image_files.append(str(img_file))
    return gt_list, image_files


def cache_model_predictions(weights_path, image_files, conf, img_size, device):
    """Run one YOLO model over the images once; return remapped (10->8) per-image
    preds + (w, h) list. Decouples inference from fusion: cache once, fuse/search many."""
    model = YOLO(weights_path)
    preds, img_whs = [], []
    for f in image_files:
        image = cv2.imread(f)
        if image is None:
            raise FileNotFoundError(f"cv2 could not read image: {f}")
        h, w = image.shape[:2]
        r = model(f, conf=conf, imgsz=img_size, device=device, verbose=False)[0]
        if len(r.boxes) > 0:
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            lbl = r.boxes.cls.cpu().numpy().astype(int)
        else:
            b, s, lbl = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        b, s, lbl = remap_10class_to_8class(b, s, lbl)
        preds.append({"boxes": b, "scores": s, "labels": lbl})
        img_whs.append((w, h))
    del model
    return preds, img_whs


def _load_transformers_detector(hf_id):
    """Load a transformers object-detection model, tolerating legacy configs.

    Some community VisDrone checkpoints (e.g. okhytrov/detr-finetuned-visdrone)
    were saved with `dilation: null`, which newer transformers' strict config
    validation rejects (expects bool). Fall back to patching the raw config dict.
    """
    from transformers import AutoConfig, AutoModelForObjectDetection

    try:
        return AutoModelForObjectDetection.from_pretrained(hf_id)
    except Exception:
        import json

        from huggingface_hub import hf_hub_download

        cfg_dict = json.load(open(hf_hub_download(hf_id, "config.json")))
        if cfg_dict.get("dilation") is None:
            cfg_dict["dilation"] = False
        model_type = cfg_dict.pop("model_type")
        cfg = AutoConfig.for_model(model_type, **cfg_dict)
        return AutoModelForObjectDetection.from_pretrained(hf_id, config=cfg)


def detr_label_mapping(id2label):
    """Model-output label id -> project 8-class id, resolved by NAME when possible.

    Different community VisDrone transformer finetunes disagree on label ids
    (VisDrone annotation categories 1..10, 0-indexed 10-class, etc.), so trusting
    ids silently mis-scores boxes. When id2label carries real class names, map by
    name (dropping tricycle/awning-tricycle/others/ignored). Generic names
    ("LABEL_3") carry no signal — fall back to the VisDrone-category assumption
    (VISDRONE_TO_YOLO_CLASS) that matches okhytrov/detr-finetuned-visdrone.
    """
    from src.constants import VISDRONE_CLASSES

    name_to_new = {n: i for i, n in enumerate(VISDRONE_CLASSES)}
    mapping = {}
    for raw_id, raw_name in id2label.items():
        name = str(raw_name).strip().lower().replace("_", "-")
        if name in name_to_new:
            mapping[int(raw_id)] = name_to_new[name]
    # Require most classes matched by name; partial matches smell like a
    # different taxonomy and the id fallback is more honest.
    if len(mapping) >= 6:
        return mapping
    return dict(VISDRONE_TO_YOLO_CLASS)


def cache_detr_predictions(hf_id, image_files, conf, device, img_size=None):
    """Run a HuggingFace transformers detector (e.g. DETR) over the images once and
    return cache-format preds + (w, h) list, so a cross-architecture member can join
    the same WBF fusion as the YOLO members. Boxes are xyxy abs-pixels, labels mapped
    into the project's 8 classes via detr_label_mapping (unmapped dropped).

    `img_size` is accepted for signature symmetry but ignored: the transformers
    processor applies its own preprocessing and post_process returns boxes in the
    original image resolution. Mirrors scripts/eval_finetuned.py's transformer path.
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    torch_device = torch.device(device)
    processor = AutoImageProcessor.from_pretrained(hf_id)
    model = _load_transformers_detector(hf_id).to(torch_device).eval()
    label_map = detr_label_mapping(getattr(model.config, "id2label", {}) or {})

    preds, img_whs = [], []
    for f in image_files:
        image = Image.open(f).convert("RGB")
        w, h = image.size
        inputs = {
            k: v.to(torch_device) for k, v in processor(images=image, return_tensors="pt").items()
        }
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[h, w]], device=torch_device)
        r = processor.post_process_object_detection(
            outputs, threshold=conf, target_sizes=target_sizes
        )[0]
        b = r["boxes"].cpu().numpy()
        s = r["scores"].cpu().numpy()
        raw = r["labels"].cpu().numpy()
        keep = np.array([int(c) in label_map for c in raw], dtype=bool)
        if keep.any():
            b, s = b[keep], s[keep]
            lbl = np.array([label_map[int(c)] for c in raw[keep]], dtype=int)
        else:
            b, s, lbl = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        preds.append({"boxes": b, "scores": s, "labels": lbl})
        img_whs.append((w, h))
    del model
    return preds, img_whs


def load_or_cache_detr_predictions(
    cache_dir, model_key, hf_id, image_files, scene, split, conf, device
):
    """Disk-backed cache_detr_predictions (transformers models pick their own input
    size, so the cache filename uses 'native' as the img-size token)."""
    path = cache_path(cache_dir, model_key, scene, split, conf, "native")
    if path.exists():
        preds, img_whs = load_predictions(path)
        if len(preds) == len(image_files):
            return preds, img_whs
    preds, img_whs = cache_detr_predictions(hf_id, image_files, conf, device)
    save_predictions(path, preds, img_whs)
    return preds, img_whs


def cache_path(cache_dir, model_key, scene, split, conf, img_size):
    """Canonical on-disk location for one model's cached predictions on one
    scene/split. The protocol (conf, img size) is part of the filename so a
    protocol change can never silently reuse stale predictions. Models registered in
    ``NATIVE_CACHE_IMG_TOKENS`` (non-Ultralytics, own input size) get their own token."""
    token = NATIVE_CACHE_IMG_TOKENS.get(model_key, img_size)
    return Path(cache_dir) / f"{model_key}_{scene}_{split}_conf{conf}_img{token}.npz"


def save_predictions(path, preds, img_whs):
    """Persist cache_model_predictions output. Boxes/scores/labels are ragged
    per image, so they go in as object arrays (allow_pickle on load)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        boxes=np.array([p["boxes"] for p in preds], dtype=object),
        scores=np.array([p["scores"] for p in preds], dtype=object),
        labels=np.array([p["labels"] for p in preds], dtype=object),
        img_whs=np.array(img_whs, dtype=int),
    )


def load_predictions(path):
    """Inverse of save_predictions: -> (preds, img_whs)."""
    with np.load(path, allow_pickle=True) as z:
        preds = [
            {"boxes": b, "scores": s, "labels": lbl}
            for b, s, lbl in zip(z["boxes"], z["scores"], z["labels"])
        ]
        img_whs = [tuple(wh) for wh in z["img_whs"]]
    return preds, img_whs


def load_or_cache_predictions(
    cache_dir, model_key, weights_path, image_files, scene, split, conf, img_size, device
):
    """Disk-backed cache_model_predictions: run inference once per (model, scene,
    split, protocol) ever, so downstream experiments (overlap, routing, bootstrap)
    re-fuse cached boxes instead of re-running models."""
    path = cache_path(cache_dir, model_key, scene, split, conf, img_size)
    if path.exists():
        preds, img_whs = load_predictions(path)
        if len(preds) == len(image_files):
            return preds, img_whs
        # image set changed under the same protocol -> recompute
    if model_key in NATIVE_CACHE_IMG_TOKENS:
        raise FileNotFoundError(
            f"{path} missing: {model_key} is not an Ultralytics model; produce its cache with "
            "its own adapter (e.g. scripts/rtdetrv4_predict.py + scripts/17_rtdetrv4_eval.py)"
        )
    preds, img_whs = cache_model_predictions(weights_path, image_files, conf, img_size, device)
    save_predictions(path, preds, img_whs)
    return preds, img_whs


def prediction_overlap(a, b, iou_thr=0.5):
    """Symmetric prediction overlap between two models' detections on one image:
    fraction of boxes with a same-class IoU>=thr match in the other set.
    1 = identical predictions, 0 = disjoint, None = both empty (no signal)."""
    n_a, n_b = len(a["boxes"]), len(b["boxes"])
    if n_a == 0 and n_b == 0:
        return None
    matched = _matched_count(a, b, iou_thr) + _matched_count(b, a, iou_thr)
    return matched / (n_a + n_b)


def _matched_count(a, b, iou_thr):
    from src.evaluation.metrics import compute_iou_matrix

    if len(a["boxes"]) == 0 or len(b["boxes"]) == 0:
        return 0
    iou = compute_iou_matrix(a["boxes"], b["boxes"])  # (na, nb)
    same = a["labels"][:, None] == b["labels"][None, :]
    hit = ((iou >= iou_thr) & same).any(axis=1)
    return int(hit.sum())


def filter_by_conf(pred, thr):
    """Keep only detections with score >= thr (for interpretable overlap stats)."""
    m = pred["scores"] >= thr
    return {"boxes": pred["boxes"][m], "scores": pred["scores"][m], "labels": pred["labels"][m]}


def benchmark_fps(
    model_paths, model_names, method, weights, test_image, device, conf, img_size, num_runs
):
    """FPS via EnsembleDetector.benchmark — the SAME meter for single (1 model) and
    ensembles, so the Pareto comparison is honest. Measured at deployment size."""
    det = EnsembleDetector(
        model_paths=model_paths,
        model_names=model_names,
        method=method,
        weights=weights,
        conf_thr=conf,
        img_size=img_size,
        device=device,
    )
    fps = det.benchmark(test_image, num_runs=num_runs)["total_fps"]
    del det
    return fps
