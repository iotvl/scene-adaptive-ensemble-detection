"""Report for the A1 mitigation: capped confirmation multiplier wbf_capK
(min(T, N - k_max) / (N - k_max), thesis §3.6 / §3.8 recommendation).

Reads the capped-method grids and bootstraps from `cap_root` (experiments/exp36_cap)
and the reference wbf / nms / wbf_no_conf rows and bootstraps from `ref_root`
(experiments/exp36). Both share the resample seed, so replicate-wise differences
are paired. Produces per dataset:

  mitigation_{ds}.csv / .tex  — per (pool, k_max, k, attack): clean cost, target
                                recall & ASR (A1), ΔFPPI & target AP (A2), mAP
                                loss (A3), paired bootstrap of the loss difference
                                "loss(WBF) − loss(WBF_capK)" for mAP@50 and target AP
  mitigation_{ds}.json        — the same numbers, machine-readable
  fig6_mitigation_{ds}.png/pdf — recall & ASR vs k (A1) and ΔFPPI vs k (A2) for
                                WBF, WBF_capK, NMS
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp36.eval.bootstrap import summarize  # noqa: E402
from exp36.fusion.methods import cap_kmax  # noqa: E402
from exp36.report.build import (  # noqa: E402
    Boot,
    _boot_mean_sets,
    _csv,
    _index,
    _mean_over_sets,
    _tex_table,
)
from exp36.runner.grid import read_jsonl  # noqa: E402

ATTACKS = [("suppress", 1.0), ("inject", 15.0), ("inflate", 1.5)]
KEY_MAP = lambda r: r["map50"]["balanced"]  # noqa: E731
KEY_AP = lambda r: r["ap50_target"]["balanced"]  # noqa: E731
KEY_REC = lambda r: r["target_recall"]["all"]  # noqa: E731
KEY_ASR = lambda r: (r["asr"] or {}).get("all")  # noqa: E731
KEY_FPPI = lambda r: (r["fppi_delta"] or {}).get("balanced")  # noqa: E731


class TwoBoots:
    """Replicates for a config id, looked up in the capped run first, then the reference."""

    def __init__(self, cap: Boot | None, ref: Boot | None):
        self.cap, self.ref = cap, ref

    def balanced(self, cid, which=0):
        for b in (self.cap, self.ref):
            if b is not None:
                v = b.balanced(cid, which)
                if v is not None:
                    return v
        return None


def _paired_loss_diff(boot, clean_a, att_a, clean_b, att_b, which):
    """summary of (loss_b − loss_a) replicates; None if any replicate array is missing."""
    reps = [_boot_mean_sets(boot, r, which) for r in (clean_a, att_a, clean_b, att_b)]
    if any(v is None for v in reps):
        return None
    ca, aa, cb, ab = reps
    return summarize((cb - ab) - (ca - aa))


def _paired_diff(boot, rows_a, rows_b, which):
    a, b = _boot_mean_sets(boot, rows_a, which), _boot_mean_sets(boot, rows_b, which)
    return None if a is None or b is None else summarize(a - b)


def _fmt(s):
    if s is None:
        return "—"
    return f"{s['mean']:+.4f} [{s['ci_lo']:+.4f}; {s['ci_hi']:+.4f}]"


def build_mitigation(cap_root: Path, ref_root: Path, dataset: str, pools: list[str], log=print):
    out = cap_root
    rows_all, json_out, curves = [], {}, {}
    for pool in pools:
        cap_grid = cap_root / f"grid_{dataset}_{pool}.jsonl"
        ref_grid = ref_root / f"grid_{dataset}_{pool}.jsonl"
        if not cap_grid.exists():
            log(f"[mitigation] no capped grid for {dataset}/{pool}, skipping")
            continue
        idx = _index(read_jsonl(ref_grid) + read_jsonl(cap_grid))
        cap_b = cap_root / "boot" / f"{dataset}_{pool}"
        ref_b = ref_root / "boot" / f"{dataset}_{pool}"
        boot = TwoBoots(
            Boot(cap_b) if cap_b.exists() else None, Boot(ref_b) if ref_b.exists() else None
        )
        methods = sorted(
            {m for (p, _, _, _, m) in idx if p == pool and cap_kmax(m) is not None}, key=cap_kmax
        )
        N = max(k for (p, k, _, _, _) in idx if p == pool)
        ks = sorted({k for (p, k, _, _, _) in idx if p == pool and 1 <= k < N})
        clean = {
            m: idx.get((pool, 0, "none", None, m), [])
            for m in ["wbf", "nms", "wbf_no_conf", *methods]
        }
        json_out[pool] = {"N": N, "methods": methods, "clean": {}, "rows": []}
        for m in ["wbf", "nms", "wbf_no_conf", *methods]:
            json_out[pool]["clean"][m] = {
                "map50": _mean_over_sets(clean[m], KEY_MAP),
                "ap50_target": _mean_over_sets(clean[m], KEY_AP),
                "target_recall": _mean_over_sets(clean[m], KEY_REC),
            }
            if m != "wbf":
                json_out[pool]["clean"][m]["map50_minus_wbf"] = _paired_diff(
                    boot, clean[m], clean["wbf"], 0
                )
        # curves for the figure
        curves[pool] = {"ks": ks, "N": N}
        for m in ["wbf", "nms", *methods]:
            curves[pool][m] = {
                "rec_a1": [
                    _mean_over_sets(idx.get((pool, k, "suppress", 1.0, m), []), KEY_REC) for k in ks
                ],
                "asr_a1": [
                    _mean_over_sets(idx.get((pool, k, "suppress", 1.0, m), []), KEY_ASR) for k in ks
                ],
                "fppi_a2": [
                    _mean_over_sets(idx.get((pool, k, "inject", 15.0, m), []), KEY_FPPI) for k in ks
                ],
                "ap_a2": [
                    _mean_over_sets(idx.get((pool, k, "inject", 15.0, m), []), KEY_AP) for k in ks
                ],
                "map_a3": [
                    _mean_over_sets(idx.get((pool, k, "inflate", 1.5, m), []), KEY_MAP) for k in ks
                ],
            }
        for m in methods:
            for k in ks:
                for att, par in ATTACKS:
                    g_w = idx.get((pool, k, att, par, "wbf"), [])
                    g_c = idx.get((pool, k, att, par, m), [])
                    g_n = idx.get((pool, k, att, par, "nms"), [])
                    if not g_c:
                        continue
                    rec = {
                        "pool": pool,
                        "k_max": cap_kmax(m),
                        "k": k,
                        "attack": f"{att} {par:g}",
                        "clean_map_cap": _mean_over_sets(clean[m], KEY_MAP),
                        "clean_map_wbf": _mean_over_sets(clean["wbf"], KEY_MAP),
                        "map_wbf": _mean_over_sets(g_w, KEY_MAP),
                        "map_cap": _mean_over_sets(g_c, KEY_MAP),
                        "map_nms": _mean_over_sets(g_n, KEY_MAP),
                        "ap_wbf": _mean_over_sets(g_w, KEY_AP),
                        "ap_cap": _mean_over_sets(g_c, KEY_AP),
                        "ap_nms": _mean_over_sets(g_n, KEY_AP),
                        "rec_wbf": _mean_over_sets(g_w, KEY_REC),
                        "rec_cap": _mean_over_sets(g_c, KEY_REC),
                        "rec_nms": _mean_over_sets(g_n, KEY_REC),
                        "asr_wbf": _mean_over_sets(g_w, KEY_ASR),
                        "asr_cap": _mean_over_sets(g_c, KEY_ASR),
                        "asr_nms": _mean_over_sets(g_n, KEY_ASR),
                        "fppi_wbf": _mean_over_sets(g_w, KEY_FPPI),
                        "fppi_cap": _mean_over_sets(g_c, KEY_FPPI),
                        "fppi_nms": _mean_over_sets(g_n, KEY_FPPI),
                        # paired bootstrap: loss(WBF) − loss(cap); positive => cap more robust
                        "loss_diff_map": _paired_loss_diff(
                            boot, clean[m], g_c, clean["wbf"], g_w, 0
                        ),
                        "loss_diff_ap": _paired_loss_diff(
                            boot, clean[m], g_c, clean["wbf"], g_w, 1
                        ),
                        # absolute: cap − wbf after the attack
                        "abs_diff_map": _paired_diff(boot, g_c, g_w, 0),
                        "abs_diff_ap": _paired_diff(boot, g_c, g_w, 1),
                    }
                    json_out[pool]["rows"].append(rec)
                    rows_all.append(
                        [
                            pool,
                            rec["k_max"],
                            k,
                            rec["attack"],
                            f"{rec['clean_map_cap'] - rec['clean_map_wbf']:+.4f}",
                            f"{rec['map_wbf']:.4f} / {rec['map_cap']:.4f} / {rec['map_nms']:.4f}",
                            f"{rec['rec_wbf']:.3f} / {rec['rec_cap']:.3f} / {rec['rec_nms']:.3f}",
                            (
                                f"{rec['asr_wbf']:.3f} / {rec['asr_cap']:.3f} / {rec['asr_nms']:.3f}"
                                if att != "inject"
                                else "—"
                            ),
                            (
                                f"{rec['fppi_wbf']:.1f} / {rec['fppi_cap']:.1f} / {rec['fppi_nms']:.1f}"
                                if att == "inject"
                                else "—"
                            ),
                            f"{rec['ap_wbf']:.3f} / {rec['ap_cap']:.3f} / {rec['ap_nms']:.3f}",
                            _fmt(rec["loss_diff_map"]),
                            _fmt(rec["loss_diff_ap"]),
                        ]
                    )
    header = [
        "пул",
        "k_max",
        "k",
        "атака",
        "чиста ціна mAP (cap − WBF)",
        "mAP@50: WBF / cap / NMS",
        "повнота цілі: WBF / cap / NMS",
        "ASR: WBF / cap / NMS",
        "ΔFPPI: WBF / cap / NMS",
        "AP@50 цілі: WBF / cap / NMS",
        "Δ втрат mAP (WBF − cap) [95 % ДІ]",
        "Δ втрат AP цілі (WBF − cap) [95 % ДІ]",
    ]
    _csv(out / f"mitigation_{dataset}.csv", header, rows_all)
    _tex_table(
        out / f"mitigation_{dataset}.tex",
        header,
        rows_all,
        f"Обмежений множник підтвердження: {dataset}",
    )
    with open(out / f"mitigation_{dataset}.json", "w") as f:
        json.dump(json_out, f, indent=1, ensure_ascii=False, default=float)
    _figure(curves, dataset, out)
    log(f"[mitigation] {dataset}: {len(rows_all)} rows -> {out}")
    return json_out


def _figure(curves, dataset, out):
    pools = [p for p in curves if curves[p]["ks"]]
    if not pools:
        return
    fig, axes = plt.subplots(len(pools), 3, figsize=(13, 3.8 * len(pools)), squeeze=False)
    style = {"wbf": ("WBF", "#2c5f9e", "-"), "nms": ("NMS", "#b06f1f", "-")}
    cap_cols = ["#3a7d3a", "#7b2d8e", "#c0392b", "#555"]
    for i, pool in enumerate(pools):
        c = curves[pool]
        ks = c["ks"]
        meths = [m for m in c if m not in ("ks", "N")]
        for m in meths:
            if m in style:
                lab, col, ls = style[m]
            else:
                kmax = cap_kmax(m)
                lab, col, ls = f"WBF, k_max = {kmax}", cap_cols[(kmax - 1) % len(cap_cols)], "--"
            axes[i, 0].plot(ks, c[m]["rec_a1"], "o", ls=ls, color=col, label=lab)
            axes[i, 1].plot(ks, c[m]["asr_a1"], "o", ls=ls, color=col, label=lab)
            axes[i, 2].plot(ks, c[m]["fppi_a2"], "o", ls=ls, color=col, label=lab)
        axes[i, 0].set_title(f"{pool} (N = {c['N']}): A1, повнота цілі на порозі 0,25", fontsize=10)
        axes[i, 1].set_title(f"{pool}: A1, частка успіху атаки ASR", fontsize=10)
        axes[i, 2].set_title(f"{pool}: A2 (15 фантомів/кадр), ΔFPPI", fontsize=10)
        for ax in axes[i]:
            ax.set_xlabel("k — кількість скомпрометованих членів")
            ax.set_xticks(ks)
            ax.grid(alpha=0.3)
        axes[i, 0].set_ylim(0, 1)
        axes[i, 1].set_ylim(0, 1.02)
        axes[i, 0].legend(fontsize=8)
    fig.suptitle(
        f"{dataset}: обмежений множник підтвердження min(T, N − k_max)/(N − k_max)", fontsize=11
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig6_mitigation_{dataset}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
