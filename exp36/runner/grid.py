"""Grid of configurations (spec stage 6): enumeration, assembly from the
fusion cache, metrics, jsonl output.

Configuration = (dataset, pool, k, poisoned set S, attack, attack param,
fusion method). Attacks: suppress (A1, rho), inject (A2, n/frame), inflate
(A3, factor), plus controls: honest (S removed, no attack), decoy (A1 rho=1 on
the decoy class), clean (k=0).

Per-class plan of a configuration on one scene: which ClassRun provides each
class, possibly a frame-wise mixture of two runs (A1 with rho < 1).
"""

from __future__ import annotations

import itertools
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exp36.attacks.primitives import triggered_frames
from exp36.eval.matching import ConfigStats
from exp36.eval.target_metrics import attack_success_rate, fppi_delta
from exp36.loaders.cache import Dataset
from exp36.runner.cache_fusion import FusionCache, variant_key

ATTACKS_WITH_PARAM = {"suppress": "rho", "inject": "n_per_frame", "inflate": "factor"}


@dataclass(frozen=True)
class Config:
    pool: str
    k: int
    poisoned: tuple[str, ...]
    attack: str  # none | suppress | inject | inflate | honest | decoy
    param: float | None  # rho | n_per_frame | factor | None
    fusion: str

    @property
    def cid(self) -> str:
        p = "" if self.param is None else f"{self.param:g}"
        return f"{self.pool}|k{self.k}|{','.join(self.poisoned)}|{self.attack}|{p}|{self.fusion}"

    @property
    def is_attack(self) -> bool:
        return self.attack in ("suppress", "inject", "inflate", "decoy")


def poisoned_sets(pool: list[str], k: int, cap: int, seed_combos: int) -> list[tuple[str, ...]]:
    combos = list(itertools.combinations(pool, k))
    if len(combos) <= cap:
        return combos
    rng = np.random.default_rng(seed_combos + k)
    idx = sorted(rng.choice(len(combos), size=cap, replace=False).tolist())
    return [combos[i] for i in idx]


def enumerate_configs(
    pool_name: str,
    pool: list[str],
    params: dict,
    methods: list[str],
    seed_combos: int,
    ks: list[int] | None = None,
) -> list[Config]:
    N = len(pool)
    at = params["attacks"]
    ks = ks if ks is not None else list(range(1, N)) + [N]
    out = [Config(pool_name, 0, (), "none", None, m) for m in methods]
    for k in ks:
        for S in poisoned_sets(pool, k, params["combo_cap"], seed_combos):
            for m in methods:
                out.append(Config(pool_name, k, S, "honest", None, m))
                for rho in at["suppress_rho"]:
                    out.append(Config(pool_name, k, S, "suppress", float(rho), m))
                out.append(Config(pool_name, k, S, "decoy", 1.0, m))
                for n in at["inject_n_per_frame"]:
                    out.append(Config(pool_name, k, S, "inject", float(n), m))
                out.append(Config(pool_name, k, S, "inflate", float(at["inflate_factor"]), m))
    return out


# ---------------------------------------------------------------- per-class plans
def class_plan(
    cfg: Config, pool: list[str], num_classes: int, target: int, decoy: int
) -> dict[int, tuple]:
    """class -> ("run", variant) | ("mix", variant_on, variant_off) (mix uses the
    scene's triggered-frame mask at assembly time)."""
    rest_members = tuple(m for m in pool if m not in cfg.poisoned)
    full = ("subset", tuple(pool), len(pool))
    rest = ("subset", rest_members, len(pool))  # poisoned member present but silent
    rest_honest = ("subset", rest_members, len(rest_members))  # member genuinely removed
    plan = {c: ("run", full) for c in range(num_classes)}
    if cfg.attack == "none":
        return plan
    if cfg.attack == "honest":
        return {c: ("run", rest_honest) for c in range(num_classes)}
    if cfg.attack in ("suppress", "decoy"):
        c = target if cfg.attack == "suppress" else decoy
        plan[c] = ("run", rest) if cfg.param >= 1.0 else ("mix", rest, full)
        return plan
    if cfg.attack == "inject":
        plan[target] = ("run", ("inject", tuple(cfg.poisoned), int(cfg.param)))
        return plan
    if cfg.attack == "inflate":
        return {
            c: ("run", ("inflate", tuple(cfg.poisoned), float(cfg.param)))
            for c in range(num_classes)
        }
    raise ValueError(cfg.attack)


def required_runs(
    configs: list[Config],
    pool: list[str],
    num_classes: int,
    target: int,
    decoy: int,
    scenes: list[str],
    methods: list[str],
) -> list[tuple]:
    """Unique (scene, method, cls, variant, pool_name) jobs for the fusion cache."""
    seen = OrderedDict()
    for cfg in configs:
        for c, item in class_plan(cfg, pool, num_classes, target, decoy).items():
            variants = item[1:] if item[0] == "mix" else (item[1],)
            for v in variants:
                for s in scenes:
                    # dedup on the on-disk key: for non-WBF methods the pool size N in a
                    # "subset" variant is irrelevant and must not spawn a second writer
                    seen.setdefault(
                        (s, cfg.fusion, c, variant_key(v, cfg.fusion)),
                        (s, cfg.fusion, c, v, cfg.pool),
                    )
    return list(seen.values())


class RunStore:
    """Small LRU over ClassRun npz loads (clean runs are reused by every config)."""

    def __init__(self, cache: FusionCache, dataset: str, capacity: int = 256):
        self.cache, self.dataset, self.capacity = cache, dataset, capacity
        self._lru: OrderedDict = OrderedDict()

    def get(self, scene, method, cls, variant):
        key = (scene, method, cls, variant)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        run = self.cache.get(self.dataset, scene, method, cls, variant)
        self._lru[key] = run
        if len(self._lru) > self.capacity:
            self._lru.popitem(last=False)
        return run


def assemble(
    cfg: Config,
    ds: Dataset,
    scene: str,
    store: RunStore,
    pool: list[str],
    target: int,
    decoy: int,
    seed_attack: int,
) -> ConfigStats:
    sd = ds.scenes[scene]
    plan = class_plan(cfg, pool, ds.num_classes, target, decoy)
    per_class = []
    for c in range(ds.num_classes):
        item = plan[c]
        if item[0] == "run":
            run = store.get(scene, cfg.fusion, c, item[1])
            per_class.append([run.frame(i) for i in range(sd.n_images)])
        else:
            on, off = (
                store.get(scene, cfg.fusion, c, item[1]),
                store.get(scene, cfg.fusion, c, item[2]),
            )
            mask = triggered_frames(sd.n_images, cfg.param, seed_attack, ds.name, scene)
            per_class.append([(on if mask[i] else off).frame(i) for i in range(sd.n_images)])
    return ConfigStats(per_class)


# ---------------------------------------------------------------- rows
def _balanced(vals: dict[str, float]) -> float:
    return float(np.mean(list(vals.values())))


def _weighted(vals: dict[str, float], n: dict[str, int]) -> float:
    w = np.array([n[s] for s in vals], dtype=float)
    return float((np.array(list(vals.values())) * w).sum() / w.sum())


def config_row(
    cfg: Config,
    ds: Dataset,
    store: RunStore,
    pool: list[str],
    target: int,
    decoy: int,
    seed_attack: int,
    deploy_thr: float,
    clean_stats: dict[str, ConfigStats] | None,
    honest_map: float | None,
) -> tuple[dict, dict[str, ConfigStats]]:
    scenes = ds.scene_names
    n_img = {s: ds.n_images(s) for s in scenes}
    stats = {s: assemble(cfg, ds, s, store, pool, target, decoy, seed_attack) for s in scenes}
    cls = decoy if cfg.attack == "decoy" else target
    m50 = {s: round(stats[s].map50(), 4) for s in scenes}
    m5095 = {s: round(stats[s].map5095(), 4) for s in scenes}
    ap_t = {s: round(float(stats[s].per_class_ap50()[cls]), 4) for s in scenes}
    rec = {s: stats[s].recall(cls, deploy_thr) for s in scenes}
    row = {
        "id": cfg.cid,
        "dataset": ds.name,
        "pool": cfg.pool,
        "k": cfg.k,
        "poisoned": list(cfg.poisoned),
        "attack": cfg.attack,
        "param": cfg.param,
        "fusion": cfg.fusion,
        "target_class": cls,
        "map50": {
            **m50,
            "balanced": round(_balanced(m50), 4),
            "weighted": round(_weighted(m50, n_img), 4),
        },
        "map5095": {**m5095, "balanced": round(_balanced(m5095), 4)},
        "ap50_target": {**ap_t, "balanced": round(_balanced(ap_t), 4)},
        "target_recall": {s: (None if np.isnan(v) else round(v, 4)) for s, v in rec.items()},
        "n_boxes_out": round(
            float(
                sum(stats[s].n_boxes_per_image() * n_img[s] for s in scenes) / sum(n_img.values())
            ),
            1,
        ),
        "map50_honest_removal": honest_map,
    }
    # pooled (all frames) target recall
    det = sum(len(stats[s].detected(cls, deploy_thr)) for s in scenes)
    tot = sum(stats[s].n_gt_total(cls) for s in scenes)
    row["target_recall"]["all"] = round(det / tot, 4) if tot else None
    if clean_stats is not None and cfg.is_attack:
        asr = {s: attack_success_rate(clean_stats[s], stats[s], cls, deploy_thr) for s in scenes}
        num = sum(
            len(clean_stats[s].detected(cls, deploy_thr) - stats[s].detected(cls, deploy_thr))
            for s in scenes
        )
        den = sum(len(clean_stats[s].detected(cls, deploy_thr)) for s in scenes)
        row["asr"] = {
            **{s: (None if np.isnan(v) else round(v, 4)) for s, v in asr.items()},
            "all": round(num / den, 4) if den else None,
        }
        if cfg.attack == "inject":
            fd = {s: fppi_delta(clean_stats[s], stats[s], cls, deploy_thr) for s in scenes}
            row["fppi_delta"] = {
                **{s: round(v, 3) for s, v in fd.items()},
                "balanced": round(_balanced(fd), 3),
            }
        else:
            row["fppi_delta"] = None
    else:
        row["asr"], row["fppi_delta"] = None, None
    return row, stats


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
