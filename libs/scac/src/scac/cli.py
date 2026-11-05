"""Command-line entry points over a frame table (no inference).

Input is a JSON list of per-frame records::

    {"id": "M0101/000001", "group": "M0101", "scene": "day",
     "signal": 23.0, "ap_t1": 0.61, "ap_t2": 0.64}

``scac gate frames.json``     — density of the per-frame gain and the oracle /
                                signal / random curves under a per-frame-AP
                                surrogate of quality (mean of the chosen
                                per-frame AP, scene-balanced).
``scac calibrate frames.json``— two-fold cross-fitted threshold under the same
                                surrogate; prints p*, tau per fold and the
                                realised escalation per scene.

The surrogate is additive by construction; for the pool-level mAP used in the
dissertation, call the library from Python with a custom score function (see
README). The CLI is meant as a first, cheap look at a new domain.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .calibrate import crossfit, realized_by_scene, stratified_folds
from .gate import gate_stats, oracle_curve, random_curve, signal_curve
from .policy import average_budget


def _load(path):
    rows = json.load(open(path))
    sig = np.array([r["signal"] for r in rows], float)
    t1 = np.array([r.get("ap_t1", np.nan) for r in rows], float)
    t2 = np.array([r.get("ap_t2", np.nan) for r in rows], float)
    scene = [r.get("scene", "all") for r in rows]
    group = [r.get("group", r.get("id", str(i))) for i, r in enumerate(rows)]
    return sig, t1, t2, scene, group


def _balanced_surrogate(t1, t2, scene):
    scene = np.asarray(scene)
    scenes = list(dict.fromkeys(scene.tolist()))

    def score(mask, eval_idx=None):
        idx = np.arange(len(t1)) if eval_idx is None else np.asarray(eval_idx)
        vals = []
        for s in scenes:
            sel = idx[scene[idx] == s]
            ap = np.where(mask[sel], t2[sel], t1[sel])
            ap = ap[~np.isnan(ap)]
            if ap.size:
                vals.append(ap.mean())
        return float(np.mean(vals)) if vals else float("nan")

    return score


def cmd_gate(args):
    sig, t1, t2, scene, _ = _load(args.frames)
    delta = t2 - t1
    score = _balanced_surrogate(t1, t2, scene)
    st = gate_stats(delta)
    print(
        f"frames with GT: {st['n']}  gain>0: {st['frac_positive']:.3f}  gain<0: {st['frac_negative']:.3f}  mean gain: {st['mean_gain']:+.4f}"
    )
    verdict = (
        "sparse gain -> cascade worth building"
        if st["frac_positive"] < 0.5
        else "dense gain -> uniform configuration likely absorbs any selectivity"
    )
    print("verdict:", verdict)
    rng = np.random.default_rng(args.seed)
    curves = {
        "oracle": oracle_curve(delta, lambda m: score(m), b1=args.b1, b2=args.b2),
        "signal": signal_curve(sig, lambda m: score(m), b1=args.b1, b2=args.b2),
        "random": random_curve(len(sig), lambda m: score(m), rng, b1=args.b1, b2=args.b2),
    }
    print(
        f"{'p':>5} {'oracle':>8} {'signal':>8} {'random':>8}"
        + ("  budget" if args.b1 is not None else "")
    )
    for o, s, r in zip(curves["oracle"], curves["signal"], curves["random"]):
        line = f"{o['p']:5.2f} {o['score']:8.4f} {s['score']:8.4f} {r['score']:8.4f}"
        if args.b1 is not None:
            line += f"  {average_budget(o['p'], args.b1, args.b2):8.1f}"
        print(line)
    if args.out:
        json.dump({"stats": st, "curves": curves}, open(args.out, "w"), indent=1)


def cmd_calibrate(args):
    sig, t1, t2, scene, group = _load(args.frames)
    score = _balanced_surrogate(t1, t2, scene)
    rng = np.random.default_rng(args.seed)
    fold_of = stratified_folds(group, scene, rng)
    res = crossfit(sig, fold_of, score)
    for f, c in res["folds"].items():
        print(f"fold {f}: p*={c['p']:.2f}  tau={c['tau']:.4g}  cal score={c['cal_score']:.4f}")
    print(
        f"realized escalation: {res['realized']:.3f}  per scene: "
        + ", ".join(f"{k} {v:.3f}" for k, v in realized_by_scene(res["escalate"], scene).items())
    )
    if args.b1 is not None:
        print(f"average budget: {average_budget(res['realized'], args.b1, args.b2):.1f}")
    if args.out:
        json.dump(
            {
                "folds": res["folds"],
                "realized": res["realized"],
                "escalate": res["escalate"].tolist(),
            },
            open(args.out, "w"),
            indent=1,
        )


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="scac", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("gate", cmd_gate), ("calibrate", cmd_calibrate)):
        p = sub.add_parser(name)
        p.add_argument("frames")
        p.add_argument("--b1", type=float, default=None, help="tier-1 cost per frame")
        p.add_argument(
            "--b2", type=float, default=None, help="cost of the additional tier-2 members"
        )
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--out", default=None, help="write results as JSON")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    if (args.b1 is None) != (args.b2 is None):
        ap.error("--b1 and --b2 must be given together")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
