"""Fusion cache: one ClassRun per (dataset, scene, method, class, variant).

A variant describes what the members contributed for that class:
    ("subset", members, N)         boxes of exactly these members; N = pool size
                                   used by WBF's confirmation multiplier. A1 with
                                   rho=1: N = full pool (the absent member is
                                   penalised); honest removal: N = len(members)
                                   (a genuinely smaller pool); clean: full pool.
    ("inject", members, n)         full pool
    `members` carry n phantoms/frame
    ("inflate", members, factor)   full pool
    `members` scores x factor

Because every fusion method is class-separable, every grid configuration is an
assembly of such runs — A1 with rho < 1 mixes the "subset" and the clean run
frame by frame. The fused boxes themselves are not kept
only their matches
to GT (ClassRun), which is all any metric needs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from exp36.attacks.primitives import attack_inflate, attack_inject, sample_phantoms
from exp36.eval.matching import ClassRun, match_frame_class
from exp36.fusion.methods import fuse_one_class, is_wbf_family
from exp36.loaders.cache import Dataset, SceneData


def variant_key(variant: tuple, method: str = "wbf") -> str:
    kind = variant[0]
    if kind == "subset":
        # N only matters for WBF; other methods share one run per member subset
        n = variant[2] if is_wbf_family(method) else 0
        return "S:" + ",".join(variant[1]) + f":N{n}"
    if kind == "inject":
        return f"I:{','.join(variant[1])}:{variant[2]}"
    if kind == "inflate":
        return f"F:{','.join(variant[1])}:{variant[2]}"
    raise ValueError(variant)


def run_path(root: Path, dataset: str, scene: str, method: str, cls: int, variant: tuple) -> Path:
    key = variant_key(variant, method)
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    d = root / dataset / scene / method / f"cls{cls}"
    return d / f"{h}.npz"


def compute_run(
    sd: SceneData,
    dataset: str,
    method: str,
    cls: int,
    variant: tuple,
    pool: list[str],
    fusion_params: dict,
    attack_params: dict,
    box_prior: np.ndarray | None,
) -> ClassRun:
    """Fuse class `cls` on every frame of `sd` under `variant`, match to GT."""
    kind = variant[0]
    n_models = int(variant[2]) if kind == "subset" else len(pool)
    frames = []
    for i in range(sd.n_images):
        wh = sd.whs[i]
        if kind == "subset":
            members = list(variant[1])
            member_preds = [sd.preds[m][i] for m in members]
        elif kind == "inject":
            poisoned, n_ph = set(variant[1]), int(variant[2])
            member_preds = []
            for m in pool:
                p = sd.preds[m][i]
                if m in poisoned:
                    ph = sample_phantoms(
                        n_ph,
                        box_prior,
                        wh,
                        tuple(attack_params["inject_conf"]),
                        attack_params["seed_attack"],
                        dataset,
                        sd.name,
                        m,
                        i,
                    )
                    p = attack_inject(p, cls, ph)
                member_preds.append(p)
        elif kind == "inflate":
            poisoned, factor = set(variant[1]), float(variant[2])
            member_preds = [
                attack_inflate(sd.preds[m][i], factor) if m in poisoned else sd.preds[m][i]
                for m in pool
            ]
        else:
            raise ValueError(variant)
        bl = [p["boxes"][p["labels"] == cls] for p in member_preds]
        sl = [p["scores"][p["labels"] == cls] for p in member_preds]
        fb, fs = fuse_one_class(
            method,
            bl,
            sl,
            cls,
            wh,
            n_models,
            fusion_params["iou_thr"],
            fusion_params["skip_box_thr"],
            fusion_params["soft_nms_sigma"],
        )
        gt = sd.gt_list[i]
        frames.append(match_frame_class(fb, fs, gt["boxes"][gt["labels"] == cls]))
    return ClassRun(frames)


class FusionCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, dataset, scene, method, cls, variant) -> Path:
        return run_path(self.root, dataset, scene, method, cls, variant)

    def has(self, dataset, scene, method, cls, variant) -> bool:
        return self.path(dataset, scene, method, cls, variant).exists()

    def get(self, dataset, scene, method, cls, variant) -> ClassRun:
        return ClassRun.load(self.path(dataset, scene, method, cls, variant))

    def put(self, dataset, scene, method, cls, variant, run: ClassRun) -> None:
        p = self.path(dataset, scene, method, cls, variant)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        run.save(tmp)
        tmp.replace(p)
        with open(p.with_suffix(".json"), "w") as f:
            json.dump(
                {"variant": variant_key(variant), "method": method, "cls": cls, "scene": scene}, f
            )


# ---------------------------------------------------------------- multiprocessing
_G: dict = {}


def _init_worker(
    ds: Dataset,
    pool_members: dict,
    fusion_params: dict,
    attack_params: dict,
    priors: dict,
    cache_root: str,
):
    _G.update(
        ds=ds,
        pools=pool_members,
        fusion=fusion_params,
        attack=attack_params,
        priors=priors,
        cache=FusionCache(Path(cache_root)),
    )


def _work(job: tuple) -> tuple:
    scene, method, cls, variant, pool_name = job
    cache: FusionCache = _G["cache"]
    ds: Dataset = _G["ds"]
    if cache.has(ds.name, scene, method, cls, variant):
        return job, "cached"
    run = compute_run(
        ds.scenes[scene],
        ds.name,
        method,
        cls,
        variant,
        _G["pools"][pool_name],
        _G["fusion"],
        _G["attack"],
        _G["priors"].get(cls),
    )
    cache.put(ds.name, scene, method, cls, variant, run)
    return job, "done"


def compute_missing(
    jobs: list[tuple],
    ds,
    pools,
    fusion_params,
    attack_params,
    priors,
    cache_root,
    workers: int,
    log=print,
):
    """jobs: (scene, method, cls, variant, pool_name). Computes every run not in
    the cache, `workers` processes (fork: the dataset is shared copy-on-write)."""
    import multiprocessing as mp

    cache = FusionCache(Path(cache_root))
    todo = [j for j in jobs if not cache.has(ds.name, j[0], j[1], j[2], j[3])]
    log(f"[fusion-cache] {len(jobs)} runs requested, {len(todo)} to compute, workers={workers}")
    if not todo:
        return
    if workers <= 1:
        _init_worker(ds, pools, fusion_params, attack_params, priors, str(cache_root))
        for n, j in enumerate(todo, 1):
            _work(j)
            if n % 50 == 0 or n == len(todo):
                log(f"[fusion-cache] {n}/{len(todo)}")
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(
        workers,
        initializer=_init_worker,
        initargs=(ds, pools, fusion_params, attack_params, priors, str(cache_root)),
    ) as p:
        for n, _ in enumerate(p.imap_unordered(_work, todo, chunksize=1), 1):
            if n % 50 == 0 or n == len(todo):
                log(f"[fusion-cache] {n}/{len(todo)}")
