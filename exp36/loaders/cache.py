"""Single adaptation point between the on-disk prediction caches and exp36.

Actual cache format (produced by scripts/07 and scripts/08 through
src/ensemble/scene_eval.py): one .npz per (model, scene, split) holding object
arrays `boxes[i]` (n_i, 4) xyxy abs-pixels, `scores[i]`, `labels[i]` and
`img_whs` (n_images, 2). VisDrone caches are already remapped to the 8-class
layout; UAVDT caches keep the 8-class layout and are projected to the 3-class
UAVDT space here (van merged into car, ignore regions applied), exactly as in
scripts/08_uavdt_transfer_eval.py.

Everything downstream sees one contract:

    SceneData(gt_list[i] = {"boxes", "labels"}, whs[i] = (w, h), seq[i] or None,
              image_files[i], preds[model][i] = {"boxes", "scores", "labels"})

Frames are addressed by (scene, index) — there is no global image id on disk.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.ensemble.scene_eval import cache_path, load_ground_truth, load_predictions
from src.utils.general import load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "exp36" / "config"


@dataclass
class SceneData:
    name: str
    gt_list: list[dict]
    whs: list[tuple[int, int]]
    image_files: list[str]
    preds: dict[str, list[dict]]
    seq: list[str] | None = None  # cluster id per frame (UAVDT sequences)
    ignore_list: list[np.ndarray] | None = None

    @property
    def n_images(self) -> int:
        return len(self.gt_list)


@dataclass
class Dataset:
    name: str
    num_classes: int
    scenes: dict[str, SceneData] = field(default_factory=dict)
    class_map: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def scene_names(self) -> list[str]:
        return list(self.scenes)

    def n_images(self, scene: str) -> int:
        return self.scenes[scene].n_images


def load_params() -> dict:
    with open(CONFIG_DIR / "params.yaml") as f:
        return yaml.safe_load(f)


def load_pools() -> dict:
    with open(CONFIG_DIR / "pools.yaml") as f:
        return yaml.safe_load(f)


def load_class_map() -> dict:
    with open(CONFIG_DIR / "class_map.json") as f:
        return json.load(f)


def _load_script08():
    """scripts/08 is a numbered phase script (not importable by name); load it
    the same way scripts/analysis_qualitative_4scenes.py does."""
    spec = importlib.util.spec_from_file_location(
        "u08", ROOT / "scripts" / "08_uavdt_transfer_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _all_models(pools: dict) -> list[str]:
    return sorted({m for members in pools["pools"].values() for m in members})


def _truncate(scene: SceneData, k: int) -> SceneData:
    scene.gt_list = scene.gt_list[:k]
    scene.whs = scene.whs[:k]
    scene.image_files = scene.image_files[:k]
    scene.preds = {m: p[:k] for m, p in scene.preds.items()}
    if scene.seq is not None:
        scene.seq = scene.seq[:k]
    if scene.ignore_list is not None:
        scene.ignore_list = scene.ignore_list[:k]
    return scene


def _as_pred(p: dict) -> dict:
    return {
        "boxes": np.asarray(p["boxes"], dtype=float).reshape(-1, 4),
        "scores": np.asarray(p["scores"], dtype=float).reshape(-1),
        "labels": np.asarray(p["labels"], dtype=int).reshape(-1),
    }


def load_visdrone(pools: dict, cfg, max_images: int | None = None) -> Dataset:
    ds_cfg = pools["datasets"]["visdrone"]
    eval_conf = cfg.evaluation.eval_confidence_threshold
    eval_img = cfg.evaluation.eval_img_size or cfg.dataset.img_size
    ds = Dataset("visdrone", ds_cfg["num_classes"], class_map=load_class_map()["visdrone"])
    for scene, split in ds_cfg["scenes"].items():
        sdir = ROOT / ds_cfg["scenes_dir"] / scene
        gt_list, image_files = load_ground_truth(
            str(sdir / "labels" / split), str(sdir / "images" / split)
        )
        preds, whs = {}, None
        for m in _all_models(pools):
            path = cache_path(ROOT / ds_cfg["cache_dir"], m, scene, split, eval_conf, eval_img)
            if not path.exists():
                raise SystemExit(f"missing cache {path} — exp36 is fusion-only and never infers")
            p, whs = load_predictions(path)
            if len(p) != len(gt_list):
                raise SystemExit(f"{m}/{scene}: cache has {len(p)} images, GT has {len(gt_list)}")
            preds[m] = [_as_pred(x) for x in p]
        sd = SceneData(
            scene, gt_list, [tuple(int(v) for v in wh) for wh in whs], image_files, preds
        )
        ds.scenes[scene] = _truncate(sd, max_images) if max_images else sd
    return ds


def load_uavdt(pools: dict, cfg, max_images: int | None = None) -> Dataset:
    ds_cfg = pools["datasets"]["uavdt"]
    eval_conf = cfg.evaluation.eval_confidence_threshold
    eval_img = cfg.evaluation.eval_img_size or cfg.dataset.img_size
    u08 = _load_script08()
    scenes = u08.build_scene_index(ROOT / ds_cfg["data_root"], ds_cfg["stride"])
    ds = Dataset("uavdt", ds_cfg["num_classes"], class_map=load_class_map()["uavdt"])
    for scene, split in ds_cfg["scenes"].items():
        d = scenes[scene]
        preds, whs = {}, None
        for m in _all_models(pools):
            path = cache_path(ROOT / ds_cfg["cache_dir"], m, scene, split, eval_conf, eval_img)
            if not path.exists():
                raise SystemExit(f"missing cache {path} — exp36 is fusion-only and never infers")
            p, whs = load_predictions(path)
            if len(p) != len(d["image_files"]):
                raise SystemExit(
                    f"{m}/{scene}: cache has {len(p)} frames, index has {len(d['image_files'])}"
                )
            preds[m] = [
                _as_pred(u08.to_uavdt_classes(_as_pred(x), ign))
                for x, ign in zip(p, d["ignore_list"])
            ]
        sd = SceneData(
            scene,
            d["gt_list"],
            [tuple(int(v) for v in wh) for wh in whs],
            d["image_files"],
            preds,
            seq=list(d["seq"]),
            ignore_list=list(d["ignore_list"]),
        )
        ds.scenes[scene] = _truncate(sd, max_images) if max_images else sd
    return ds


def load_dataset(
    name: str, max_images: int | None = None, config: str = "configs/default.yaml"
) -> Dataset:
    cfg = load_config(str(ROOT / config))
    pools = load_pools()
    if name == "visdrone":
        return load_visdrone(pools, cfg, max_images)
    if name == "uavdt":
        return load_uavdt(pools, cfg, max_images)
    raise ValueError(name)
