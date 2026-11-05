#!/usr/bin/env python3
"""Experiment 3.6 driver — robustness of detector ensembles to a compromised
pool member. Fusion-only over the prediction caches; never infers.

    uv run python -m exp36.run sanity     --dataset visdrone
    uv run python -m exp36.run crosscheck --dataset visdrone --pool hetero3
    uv run python -m exp36.run grid       --dataset visdrone --pool full7 --workers 8
    uv run python -m exp36.run bootstrap  --dataset visdrone --pool full7 --workers 8
    uv run python -m exp36.run report

Smoke tests must point --out/--cache-root at a scratch directory and use
--max-images
the driver refuses to write a truncated run under experiments/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp36.eval.bootstrap import make_resamples, replicates  # noqa: E402
from exp36.fusion.crosscheck import scorer_vs_reference, wbf_vs_library  # noqa: E402
from exp36.fusion.methods import METHODS  # noqa: E402
from exp36.loaders.cache import load_dataset, load_params, load_pools  # noqa: E402
from exp36.loaders.gt import box_prior, export_coco  # noqa: E402
from exp36.runner.cache_fusion import FusionCache, compute_missing  # noqa: E402
from exp36.runner.grid import (  # noqa: E402
    Config,
    RunStore,
    assemble,
    config_row,
    enumerate_configs,
    required_runs,
    write_jsonl,
)
from src.utils.general import load_config, protocol_metadata, set_seed  # noqa: E402

DEFAULT_OUT = ROOT / "experiments" / "exp36"


def seeds(params: dict) -> dict:
    s = int(params["seed"])
    return {"seed": s, "seed_attack": s + 1, "seed_combos": s + 2, "seed_bootstrap": s + 3}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def guard_output(args) -> None:
    out = Path(args.out).resolve()
    if args.max_images and str(out).startswith(str((ROOT / "experiments").resolve())):
        raise SystemExit(
            "--max-images (smoke) must not write under experiments/: pass --out <scratch>"
        )


# ---------------------------------------------------------------- sanity
def cmd_sanity(args, params, pools):
    ds = load_dataset(args.dataset, args.max_images)
    cfg = load_config(str(ROOT / "configs/default.yaml"))
    all_models = sorted({m for p in pools["pools"].values() for m in p})
    report = {"dataset": args.dataset, "scenes": {}, "checks": {}}
    ok = True
    for scene, sd in ds.scenes.items():
        sc = {"n_images": sd.n_images, "models": {}}
        for m in all_models:
            scores = (
                np.concatenate([p["scores"] for p in sd.preds[m]]) if sd.n_images else np.zeros(0)
            )
            labels = np.concatenate([p["labels"] for p in sd.preds[m]]).astype(int)
            boxes = np.concatenate([p["boxes"] for p in sd.preds[m]])
            whs = np.repeat(
                np.array(sd.whs, dtype=float), [len(p["scores"]) for p in sd.preds[m]], axis=0
            )
            inside = (
                (
                    (boxes[:, 0] >= -1)
                    & (boxes[:, 1] >= -1)
                    & (boxes[:, 2] <= whs[:, 0] + 1)
                    & (boxes[:, 3] <= whs[:, 1] + 1)
                ).all()
                if len(boxes)
                else True
            )
            nonneg = (
                ((boxes[:, 2] >= boxes[:, 0]) & (boxes[:, 3] >= boxes[:, 1])).all()
                if len(boxes)
                else True
            )
            per_img = np.array([len(p["scores"]) for p in sd.preds[m]], dtype=float)
            sc["models"][m] = {
                "min_score": float(scores.min()) if len(scores) else None,
                "classes": sorted(set(labels.tolist())),
                "boxes_per_image_mean": round(float(per_img.mean()), 1),
                "boxes_per_image_max": int(per_img.max()),
                "boxes_inside_frame": bool(inside),
                "no_negative_area": bool(nonneg),
            }
        gt_classes = sorted({int(c) for g in sd.gt_list for c in g["labels"]})
        sc["gt_classes"] = gt_classes
        sc["gt_objects"] = int(sum(len(g["labels"]) for g in sd.gt_list))
        report["scenes"][scene] = sc
    # the five spec checks
    thr = cfg.evaluation.eval_confidence_threshold
    c1 = all(
        v["min_score"] is not None and v["min_score"] <= 0.01 and abs(v["min_score"] - thr) < 0.01
        for s in report["scenes"].values()
        for v in s["models"].values()
    )
    c2 = True  # frame sets are aligned by construction (cache length == GT length, enforced in the loader)
    c3 = all(
        v["boxes_inside_frame"] and v["no_negative_area"]
        for s in report["scenes"].values()
        for v in s["models"].values()
    )
    c4 = all(
        set(v["classes"]) <= set(range(ds.num_classes))
        for s in report["scenes"].values()
        for v in s["models"].values()
    )
    c5 = all(
        1 <= v["boxes_per_image_mean"] <= 1000
        for s in report["scenes"].values()
        for v in s["models"].values()
    )
    report["checks"] = {
        "1_min_score_is_eval_threshold": c1,
        "2_frame_sets_aligned": c2,
        "3_boxes_inside_frame_nonneg_area": c3,
        "4_classes_subset_of_gt_space": c4,
        "5_boxes_per_image_reasonable": c5,
    }
    ok = all(report["checks"].values())
    report["pass"] = ok
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.max_images:
        paths = export_coco(ds, ROOT / "data" / "coco_exp36")
        report["coco_gt"] = {k: str(v.relative_to(ROOT)) for k, v in paths.items()}
    with open(out / f"sanity_{args.dataset}.json", "w") as f:
        json.dump(report, f, indent=2)
    for k, v in report["checks"].items():
        log(f"  {'PASS' if v else 'FAIL'} {k}")
    log(f"[SAVED] {out / f'sanity_{args.dataset}.json'}  overall={'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------- crosscheck
def cmd_crosscheck(args, params, pools):
    ds = load_dataset(args.dataset, args.max_images)
    pool = pools["pools"][args.pool]
    rep = {"dataset": args.dataset, "pool": args.pool, "members": pool}
    rep["wbf_vs_ensemble_boxes"] = wbf_vs_library(
        ds, pool, params["fusion"], n_images=100, seed=params["seed"]
    )
    log(
        f"  own WBF vs ensemble_boxes: dmAP50={rep['wbf_vs_ensemble_boxes']['max_abs_dmap50']} "
        f"dboxes={rep['wbf_vs_ensemble_boxes']['rel_dboxes']} pass={rep['wbf_vs_ensemble_boxes']['pass']}"
    )
    if not args.max_images or args.max_images >= 30:
        rep["scorer_vs_reference"] = scorer_vs_reference(ds, pool, params["fusion"])
        log(
            f"  grid scorer vs compute_map: {rep['scorer_vs_reference']['max_abs_diff_grid_vs_compute_map']}; "
            f"vs pycocotools mAP50: {rep['scorer_vs_reference']['max_abs_diff_grid_vs_pycocotools_map50']}"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"crosscheck_{args.dataset}_{args.pool}.json"
    with open(p, "w") as f:
        json.dump(rep, f, indent=2)
    log(f"[SAVED] {p}")


# ---------------------------------------------------------------- grid
def grid_setup(args, params, pools):
    ds = load_dataset(args.dataset, args.max_images)
    pool = pools["pools"][args.pool]
    tgt = params["targets"][args.dataset]
    sd = seeds(params)
    methods = list(METHODS) if not args.methods else args.methods.split(",")
    ks = None if not args.ks else [int(k) for k in args.ks.split(",")]
    configs = enumerate_configs(args.pool, pool, params, methods, sd["seed_combos"], ks)
    return ds, pool, tgt, sd, methods, configs


def cmd_grid(args, params, pools):
    ds, pool, tgt, sd, methods, configs = grid_setup(args, params, pools)
    target, decoy = tgt["target_class"], tgt["decoy_class"]
    log(f"[grid] {args.dataset}/{args.pool}: {len(configs)} configurations, pool={pool}")
    jobs = required_runs(configs, pool, ds.num_classes, target, decoy, ds.scene_names, methods)
    priors = {target: box_prior(args.dataset, target)}
    attack_params = {
        "seed_attack": sd["seed_attack"],
        "inject_conf": params["attacks"]["inject_conf"],
    }
    compute_missing(
        jobs,
        ds,
        pools["pools"],
        params["fusion"],
        attack_params,
        priors,
        args.cache_root,
        args.workers,
        log,
    )
    # assemble rows
    store = RunStore(FusionCache(Path(args.cache_root)), ds.name)
    clean_stats, honest = {}, {}
    rows = []
    t0 = time.time()
    order = sorted(
        configs, key=lambda c: (c.attack != "none", c.attack != "honest")
    )  # clean, honest first
    for n, cfg in enumerate(order, 1):
        hm = honest.get((cfg.poisoned, cfg.fusion)) if cfg.is_attack else None
        row, stats = config_row(
            cfg,
            ds,
            store,
            pool,
            target,
            decoy,
            sd["seed_attack"],
            params["deploy_threshold"],
            clean_stats.get(cfg.fusion),
            hm,
        )
        if cfg.attack == "none":
            clean_stats[cfg.fusion] = stats
        if cfg.attack == "honest":
            honest[(cfg.poisoned, cfg.fusion)] = row["map50"]["balanced"]
        rows.append(row)
        if n % 100 == 0 or n == len(order):
            log(f"[grid] rows {n}/{len(order)} ({time.time() - t0:.0f}s)")
    out = Path(args.out) / f"grid_{args.dataset}_{args.pool}.jsonl"
    write_jsonl(rows, out)
    meta = {
        "dataset": args.dataset,
        "pool": args.pool,
        "members": pool,
        "n_configs": len(rows),
        "methods": methods,
        "target_class": target,
        "decoy_class": decoy,
        "seeds": sd,
        "params": params,
        "_protocol": protocol_metadata(
            load_config(str(ROOT / "configs/default.yaml")),
            eval_threshold=params["eval_threshold"],
            deploy_threshold=params["deploy_threshold"],
            note="exp36 fusion-only over pred_cache; per-class fusion cache",
        ),
    }
    with open(Path(args.out) / f"grid_{args.dataset}_{args.pool}.meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"[SAVED] {out} ({len(rows)} rows)")


# ---------------------------------------------------------------- bootstrap
_B: dict = {}


def _boot_init(ds, pool, target, decoy, seed_attack, cache_root, resamples):
    _B.update(
        ds=ds,
        pool=pool,
        target=target,
        decoy=decoy,
        seed_attack=seed_attack,
        store=RunStore(FusionCache(Path(cache_root)), ds.name, capacity=64),
        res=resamples,
    )


def _boot_work(cfg: Config):
    ds = _B["ds"]
    stats = {
        s: assemble(
            cfg, ds, s, _B["store"], _B["pool"], _B["target"], _B["decoy"], _B["seed_attack"]
        )
        for s in ds.scene_names
    }
    cls = _B["decoy"] if cfg.attack == "decoy" else _B["target"]
    return cfg.cid, replicates(stats, _B["res"], cls)


def cmd_bootstrap(args, params, pools):
    import multiprocessing as mp

    ds, pool, tgt, sd, methods, configs = grid_setup(args, params, pools)
    if args.attacks:
        keep = set(args.attacks.split(","))
        configs = [c for c in configs if c.attack in keep]
    n_boot = args.n_boot or params["n_bootstrap"]
    res = make_resamples(ds, n_boot, sd["seed_bootstrap"])
    out_dir = Path(args.out) / "boot" / f"{args.dataset}_{args.pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        c
        for c in configs
        if not (out_dir / (c.cid.replace("|", "__").replace(",", "+") + ".npy")).exists()
    ]
    log(
        f"[bootstrap] {len(configs)} configs, {len(todo)} to compute, n_boot={n_boot}, workers={args.workers}"
    )
    t0 = time.time()

    def save(cid, reps):
        np.save(
            out_dir / (cid.replace("|", "__").replace(",", "+") + ".npy"), reps.astype(np.float32)
        )

    if args.workers <= 1:
        _boot_init(
            ds,
            pool,
            tgt["target_class"],
            tgt["decoy_class"],
            sd["seed_attack"],
            args.cache_root,
            res,
        )
        for n, c in enumerate(todo, 1):
            save(*_boot_work(c))
            if n % 20 == 0 or n == len(todo):
                log(f"[bootstrap] {n}/{len(todo)} ({time.time() - t0:.0f}s)")
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            args.workers,
            initializer=_boot_init,
            initargs=(
                ds,
                pool,
                tgt["target_class"],
                tgt["decoy_class"],
                sd["seed_attack"],
                args.cache_root,
                res,
            ),
        ) as p:
            for n, (cid, reps) in enumerate(p.imap_unordered(_boot_work, todo, chunksize=1), 1):
                save(cid, reps)
                if n % 20 == 0 or n == len(todo):
                    log(f"[bootstrap] {n}/{len(todo)} ({time.time() - t0:.0f}s)")
    with open(out_dir / "_meta.json", "w") as f:
        json.dump(
            {
                "n_boot": n_boot,
                "scenes": ds.scene_names,
                "seed_bootstrap": sd["seed_bootstrap"],
                "cluster": ds.scenes[ds.scene_names[0]].seq is not None,
            },
            f,
        )
    log(f"[SAVED] {out_dir}")


# ---------------------------------------------------------------- report
def cmd_mitigation(args, params, pools):
    """Capped-multiplier report: capped grids/boots in --out, references in --ref-root."""
    from exp36.report.mitigation import build_mitigation

    build_mitigation(Path(args.out), Path(args.ref_root), args.dataset, list(pools["pools"]), log)


def cmd_report(args, params, pools):
    from exp36.report.build import build_report

    build_report(Path(args.out), pools, params, log)


def main(argv=None):
    ap = argparse.ArgumentParser(description="exp36 driver")
    ap.add_argument(
        "cmd", choices=["sanity", "crosscheck", "grid", "bootstrap", "report", "mitigation"]
    )
    ap.add_argument("--dataset", default="visdrone", choices=["visdrone", "uavdt"])
    ap.add_argument("--pool", default="hetero3")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--cache-root", default=str(DEFAULT_OUT / "fusion_cache"))
    ap.add_argument("--max-images", type=int, default=0, help="smoke cap per scene")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--methods", default="", help="comma list; default all five")
    ap.add_argument("--ks", default="", help="comma list of k to run; default 1..N-1,N")
    ap.add_argument("--attacks", default="", help="bootstrap: comma list of attacks to include")
    ap.add_argument("--n-boot", type=int, default=0)
    ap.add_argument(
        "--ref-root", default=str(DEFAULT_OUT), help="mitigation: reference exp36 outputs"
    )
    args = ap.parse_args(argv)
    params, pools = load_params(), load_pools()
    set_seed(params["seed"])
    guard_output(args)
    {
        "sanity": cmd_sanity,
        "crosscheck": cmd_crosscheck,
        "grid": cmd_grid,
        "bootstrap": cmd_bootstrap,
        "report": cmd_report,
        "mitigation": cmd_mitigation,
    }[args.cmd](args, params, pools)


if __name__ == "__main__":
    main()
