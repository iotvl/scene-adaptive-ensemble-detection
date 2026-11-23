"""Cross-paradigm control for §3.6: the attack grid on the trio extended with a
transformer member (pool `hetero4` = trio + RT-DETRv4-L), set against the same
grid on the trio (`hetero3`).

The question is whether the two-sided action of the confirmation multiplier is a
property of the fusion rule or of the all-YOLO pool. Reads

  ctl_root  (experiments/exp36_rt)  — hetero4 grid + bootstrap, all methods incl. wbf_cap1
  ref_root  (experiments/exp36)     — hetero3 grid + bootstrap
  cap_root  (experiments/exp36_cap) — hetero3 wbf_cap1 grid (VisDrone only)

and writes control_{dataset}.json into ctl_root. All runs share the resample seed,
so replicate-wise differences inside a pool are paired.
"""

from __future__ import annotations

import json
from pathlib import Path

from exp36.eval.bootstrap import summarize
from exp36.report.build import Boot, _boot_mean_sets, _index, _mean_over_sets
from exp36.runner.grid import read_jsonl

METHODS = ["nms", "nmw", "soft_nms", "wbf", "wbf_no_conf", "wbf_cap1"]
ATTACKS = {
    "A1_suppress": ("suppress", 1.0),
    "A2_inject15": ("inject", 15.0),
    "A3_inflate": ("inflate", 1.5),
}
KEYS = {
    "map50": lambda r: r["map50"]["balanced"],
    "ap50_target": lambda r: r["ap50_target"]["balanced"],
    "target_recall": lambda r: r["target_recall"]["all"],
    "asr": lambda r: (r["asr"] or {}).get("all"),
    "fppi_delta": lambda r: (r["fppi_delta"] or {}).get("balanced"),
}


def _cell(rows):
    if not rows:
        return None
    out = {}
    for name, key in KEYS.items():
        v = _mean_over_sets(rows, key)
        out[name] = None if v != v else round(v, 4)
    return out


def _loss_diff(boot, idx, pool, k, attack, param, m_a, m_b, which):
    """summary of loss(m_a) − loss(m_b) under the attack; positive = m_b is more robust."""
    reps = [
        _boot_mean_sets(boot, idx.get(key, []), which)
        for key in (
            (pool, 0, "none", None, m_a),
            (pool, k, attack, param, m_a),
            (pool, 0, "none", None, m_b),
            (pool, k, attack, param, m_b),
        )
    ]
    if any(r is None for r in reps):
        return None
    ca, aa, cb, ab = reps
    return summarize((ca - aa) - (cb - ab))


def _pool_block(pool, n, idx, boot):
    block = {"n_members": n, "clean": {}, "attacks": {}, "bootstrap": {}}
    for m in METHODS:
        c = _cell(idx.get((pool, 0, "none", None, m), []))
        if c:
            block["clean"][m] = c
    for name, (attack, param) in ATTACKS.items():
        per_k = {}
        for k in range(1, n):
            per_k[str(k)] = {
                m: c for m in METHODS if (c := _cell(idx.get((pool, k, attack, param, m), [])))
            }
            hon = _cell(idx.get((pool, k, "honest", None, "wbf"), []))
            if hon:
                per_k[str(k)]["wbf_honest_removal"] = hon
        block["attacks"][name] = per_k
    if boot is None:
        return block
    for name, (attack, param) in ATTACKS.items():
        for k in range(1, n):
            for label, m_a, m_b in (
                ("nms_minus_wbf", "nms", "wbf"),
                ("noconf_minus_wbf", "wbf_no_conf", "wbf"),
                ("wbf_minus_cap1", "wbf", "wbf_cap1"),
            ):
                for metric, which in (("map50", 0), ("ap50_target", 1)):
                    s = _loss_diff(boot, idx, pool, k, attack, param, m_a, m_b, which)
                    if s:
                        block["bootstrap"][f"{name}|k{k}|loss_{label}|{metric}"] = s
    clean_wbf = _boot_mean_sets(boot, idx.get((pool, 0, "none", None, "wbf"), []), 0)
    for m in ("nms", "wbf_no_conf", "wbf_cap1"):
        other = _boot_mean_sets(boot, idx.get((pool, 0, "none", None, m), []), 0)
        if clean_wbf is not None and other is not None:
            block["bootstrap"][f"clean|{m}_minus_wbf|map50"] = summarize(other - clean_wbf)
    return block


def _per_member(pool, idx):
    """k = 1: one row per compromised member (is the transformer a special case?)."""
    out = {}
    for name, (attack, param) in ATTACKS.items():
        for m in ("nms", "wbf", "wbf_cap1"):
            clean = _cell(idx.get((pool, 0, "none", None, m), []))
            for r in idx.get((pool, 1, attack, param, m), []):
                c = _cell([r])
                c["map50_loss"] = round(clean["map50"] - c["map50"], 4)
                c["ap50_target_loss"] = round(clean["ap50_target"] - c["ap50_target"], 4)
                out.setdefault(r["poisoned"][0], {}).setdefault(name, {})[m] = c
    return out


def build_control(
    ctl_root: Path, ref_root: Path, cap_root: Path, dataset: str, pools: dict, log=print
):
    result = {"dataset": dataset, "pools": {}}
    sources = {
        "hetero3": [ref_root, cap_root],
        "hetero4": [ctl_root],
    }
    for pool, roots in sources.items():
        rows = []
        for root in roots:
            p = root / f"grid_{dataset}_{pool}.jsonl"
            if p.exists():
                rows += read_jsonl(p)
        if not rows:
            raise SystemExit(f"no grid rows for {dataset}/{pool}")
        idx = _index(rows)
        boot_dir = roots[0] / "boot" / f"{dataset}_{pool}"
        boot = Boot(boot_dir) if boot_dir.exists() else None
        block = _pool_block(pool, len(pools["pools"][pool]), idx, boot)
        block["members"] = pools["pools"][pool]
        if pool == "hetero4":
            block["per_member_k1"] = _per_member(pool, idx)
        meta = roots[0] / f"grid_{dataset}_{pool}.meta.json"
        if meta.exists():
            result.setdefault("_protocol", json.load(open(meta)).get("_protocol"))
        result["pools"][pool] = block
    out = ctl_root / f"control_{dataset}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    log(f"[SAVED] {out}")
