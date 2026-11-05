"""Stage 7: figures (PDF+PNG) and tables (CSV+LaTeX) from the grid jsonl files
and the bootstrap replicate arrays.

Figures
  fig1_degradation_{dataset}.pdf   mAP@50 vs k, one panel per pool, 4 solid
                                   methods + dashed honest removal, 95 % CI bands
  fig2_asr_{dataset}.pdf           ASR vs k (A1, rho=1)
  fig3_ablation_{dataset}.pdf      wbf - wbf_no_conf vs k with CI of the delta
  fig4_member_strength_{dataset}.pdf  k=1: degradation per member ordered by
                                   single-model mAP@50
Tables
  table1_bootstrap_{dataset}.csv/.tex   the four hypothesis deltas x k x pool
  table2_scenes_{dataset}.csv/.tex      per-scene degradation (A1, rho=1, WBF/NMS)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp36.eval.bootstrap import summarize  # noqa: E402
from exp36.runner.grid import read_jsonl  # noqa: E402

METHOD_STYLE = {
    "nms": ("NMS", "#b06f1f"),
    "soft_nms": ("Soft-NMS", "#888888"),
    "nmw": ("NMW", "#3a7d3a"),
    "wbf": ("WBF", "#2c5f9e"),
    "wbf_no_conf": ("WBF без підтвердження", "#b03030"),
}
MAIN_METHODS = ["nms", "soft_nms", "nmw", "wbf"]

# Print geometry: figures are placed at ~6.5 in width in the thesis / article, so
# every canvas is 6.8 in wide (1 x 2 panels at 3.4 in each) and fonts are sized for
# >= 8 pt effective after that scaling (ticks 8.5, labels 9, titles 10).
FIG_W = 6.8
PANEL_H = 3.6
DPI = 300
LW, MS = 2.0, 5
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.linewidth": 0.9,
    }
)


def _bottom_legend(fig, ax, ncol: int, rows: float = 1) -> None:
    """Shared legend under the panels (keeps the narrow panels free of boxes)."""
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045 * rows, 1, 1))


def _grid(n_rows: int, n_pools: int):
    """Subplot grid sized for print: fixed 6.8 in width, one column per pool."""
    return plt.subplots(n_rows, n_pools, figsize=(FIG_W, PANEL_H * n_rows), squeeze=False, dpi=DPI)


SCENE_NAME = {"day": "день", "night": "ніч", "rain": "дощ", "fog": "туман"}


def _cid_file(cid: str) -> str:
    return cid.replace("|", "__").replace(",", "+") + ".npy"


class Boot:
    """Lazy access to replicate arrays (n_boot, n_scenes) per config id."""

    def __init__(self, root: Path):
        self.root = root
        self.meta = json.load(open(root / "_meta.json")) if (root / "_meta.json").exists() else None

    def get(self, cid: str) -> np.ndarray | None:
        p = self.root / _cid_file(cid)
        return np.load(p) if p.exists() else None

    def balanced(self, cid: str, which: int = 0) -> np.ndarray | None:
        """which: 0 = mAP@50, 1 = AP@50 of the attacked class (scene-balanced)."""
        r = self.get(cid)
        if r is None:
            return None
        return r[:, :, which].mean(axis=1) if r.ndim == 3 else r.mean(axis=1)


def _index(rows: list[dict]) -> dict:
    idx = defaultdict(list)
    for r in rows:
        idx[(r["pool"], r["k"], r["attack"], r["param"], r["fusion"])].append(r)
    return idx


def _mean_over_sets(rows: list[dict], key=lambda r: r["map50"]["balanced"]) -> float:
    vals = [key(r) for r in rows if key(r) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _sets_curve(idx, pool, ks, attack, param, method, key):
    return [_mean_over_sets(idx.get((pool, k, attack, param, method), []), key) for k in ks]


def _boot_mean_sets(boot: Boot, rows: list[dict], which: int = 0) -> np.ndarray | None:
    """Replicate array of the mean over poisoned sets (paired across sets)."""
    arrs = [boot.balanced(r["id"], which) for r in rows]
    arrs = [a for a in arrs if a is not None]
    return None if not arrs else np.mean(np.stack(arrs), axis=0)


# ---------------------------------------------------------------- figures
def fig_degradation(
    idx, pools, ks_of, boot_of, dataset, out, attack="suppress", param=1.0, ylab="mAP@50"
):
    fig, axes = _grid(1, len(pools))
    for ax, pool in zip(axes[0], pools):
        ks = [0] + ks_of[pool]
        for m in MAIN_METHODS:
            name, col = METHOD_STYLE[m]
            ys, lo, hi = [], [], []
            for k in ks:
                rows = idx.get(
                    (pool, k, "none" if k == 0 else attack, None if k == 0 else param, m), []
                )
                ys.append(_mean_over_sets(rows))
                reps = _boot_mean_sets(boot_of[pool], rows) if boot_of[pool] is not None else None
                if reps is not None:
                    a, b = np.percentile(reps, [2.5, 97.5])
                    lo.append(a)
                    hi.append(b)
                else:
                    lo.append(np.nan)
                    hi.append(np.nan)
            ax.plot(ks, ys, "-o", color=col, label=name, ms=MS, lw=LW)
            if not all(np.isnan(lo)):
                ax.fill_between(ks, lo, hi, color=col, alpha=0.15, lw=0)
        # honest removal (dashed, WBF)
        ys = [
            _mean_over_sets(idx.get((pool, k, "none" if k == 0 else "honest", None, "wbf"), []))
            for k in ks
        ]
        ax.plot(ks, ys, "--", color="#000000", lw=1.8, label="чесне вилучення (WBF)")
        n = len(pools[pool])
        ax.set_title(f"{pool} (N = {n})")
        ax.set_xlabel("k — кількість скомпрометованих членів")
        ax.set_ylabel(ylab)
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{dataset}: A1 (придушення, ρ = {param:g}), збалансоване за сценами", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_degradation_{dataset}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_inject(idx, pools, ks_of, boot_of, dataset, out, n_ph=15.0):
    """A2: AP@50 of the target class (left axis per pool) and ΔFPPI vs k."""
    fig, axes = _grid(2, len(pools))
    for j, pool in enumerate(pools):
        ks = [0] + ks_of[pool]
        ax, ax2 = axes[0][j], axes[1][j]
        for m in MAIN_METHODS + ["wbf_no_conf"]:
            name, col = METHOD_STYLE[m]
            ls = ":" if m == "wbf_no_conf" else "-"
            ys, lo, hi, fp = [], [], [], []
            for k in ks:
                a, p = ("none", None) if k == 0 else ("inject", n_ph)
                rows = idx.get((pool, k, a, p, m), [])
                ys.append(_mean_over_sets(rows, lambda r: r["ap50_target"]["balanced"]))
                fp.append(
                    0.0
                    if k == 0
                    else _mean_over_sets(
                        rows, lambda r: r["fppi_delta"]["balanced"] if r["fppi_delta"] else None
                    )
                )
                reps = (
                    _boot_mean_sets(boot_of[pool], rows, 1) if boot_of[pool] is not None else None
                )
                if reps is not None:
                    q = np.percentile(reps, [2.5, 97.5])
                    lo.append(q[0])
                    hi.append(q[1])
                else:
                    lo.append(np.nan)
                    hi.append(np.nan)
            ax.plot(ks, ys, "o", color=col, label=name, ms=MS, ls=ls, lw=LW)
            if not all(np.isnan(lo)):
                ax.fill_between(ks, lo, hi, color=col, alpha=0.15, lw=0)
            ax2.plot(ks, fp, "o", color=col, label=name, ms=MS, ls=ls, lw=LW)
        ax.set_title(f"{pool} (N = {len(pools[pool])})")
        ax.set_ylabel("AP@50 цільового класу")
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
        ax2.set_ylabel("ΔFPPI на порозі 0,25")
        ax2.set_xlabel("k — кількість скомпрометованих членів")
        ax2.set_xticks(ks)
        ax2.grid(alpha=0.3)
    fig.suptitle(
        f"{dataset}: A2 (ін'єкція {n_ph:g} фантомів/кадр на член), збалансоване за сценами",
        fontsize=10,
    )
    _bottom_legend(fig, axes[0][0], ncol=5, rows=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig5_inject_{dataset}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_asr(idx, pools, ks_of, dataset, out):
    fig, axes = _grid(1, len(pools))
    for ax, pool in zip(axes[0], pools):
        ks = ks_of[pool]
        for m in MAIN_METHODS + ["wbf_no_conf"]:
            name, col = METHOD_STYLE[m]
            ys = _sets_curve(
                idx, pool, ks, "suppress", 1.0, m, lambda r: r["asr"]["all"] if r["asr"] else None
            )
            ax.plot(
                ks,
                ys,
                "-o",
                color=col,
                label=name,
                ms=MS,
                lw=LW,
                ls="-" if m != "wbf_no_conf" else ":",
            )
        ax.set_title(f"{pool} (N = {len(pools[pool])})")
        ax.set_xlabel("k")
        ax.set_ylabel("ASR (частка втрачених\nвиявлень цільового класу)")
        ax.set_xticks(ks)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{dataset}: A1, ρ = 1, робочий поріг 0,25", fontsize=10)
    _bottom_legend(fig, axes[0][0], ncol=5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_asr_{dataset}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(idx, pools, ks_of, boot_of, dataset, out):
    attacks = [
        ("suppress", 1.0, "A1 придушення ρ=1"),
        ("inject", 15.0, "A2 ін'єкція 15/кадр"),
        ("inflate", 1.5, "A3 інфляція ×1,5"),
    ]
    fig, axes = _grid(1, len(pools))
    for ax, pool in zip(axes[0], pools):
        ks = [0] + ks_of[pool]
        for (att, par, label), col in zip(attacks, ("#2c5f9e", "#b03030", "#3a7d3a")):
            ys, lo, hi = [], [], []
            for k in ks:
                a, p = ("none", None) if k == 0 else (att, par)
                r_w = idx.get((pool, k, a, p, "wbf"), [])
                r_n = idx.get((pool, k, a, p, "wbf_no_conf"), [])
                ys.append(_mean_over_sets(r_w) - _mean_over_sets(r_n))
                bw, bn = (
                    (_boot_mean_sets(boot_of[pool], r_w), _boot_mean_sets(boot_of[pool], r_n))
                    if boot_of[pool] is not None
                    else (None, None)
                )
                if bw is not None and bn is not None:
                    q = np.percentile(bw - bn, [2.5, 97.5])
                    lo.append(q[0])
                    hi.append(q[1])
                else:
                    lo.append(np.nan)
                    hi.append(np.nan)
            ax.plot(ks, ys, "-o", color=col, ms=MS, lw=LW, label=label)
            if not all(np.isnan(lo)):
                ax.fill_between(ks, lo, hi, color=col, alpha=0.15, lw=0)
        ax.axhline(0, color="#000", lw=1.0)
        ax.set_title(f"{pool} (N = {len(pools[pool])})")
        ax.set_xlabel("k")
        ax.set_ylabel("mAP@50:\nWBF − WBF без підтвердження")
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{dataset}: внесок множника підтвердження min(T, N)/N", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig3_ablation_{dataset}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_member_strength(idx, pools, single_map, dataset, out):
    fig, axes = _grid(1, len(pools))
    for ax, pool in zip(axes[0], pools):
        members = sorted(pools[pool], key=lambda m: single_map.get(m, 0))
        base = {m: _mean_over_sets(idx.get((pool, 0, "none", None, m), [])) for m in MAIN_METHODS}
        x = np.arange(len(members))
        w = 0.2
        for j, m in enumerate(MAIN_METHODS):
            name, col = METHOD_STYLE[m]
            ys = []
            for mem in members:
                rows = [
                    r for r in idx.get((pool, 1, "suppress", 1.0, m), []) if r["poisoned"] == [mem]
                ]
                ys.append(base[m] - _mean_over_sets(rows))
            ax.bar(x + (j - 1.5) * w, ys, w - 0.02, color=col, label=name, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{mm.replace('_vd', '')}\n({single_map.get(mm, float('nan')):.3f})"
                for mm in members
            ],
            fontsize=8,
        )
        ax.set_ylabel("втрата mAP@50 (чистий − отруєний), k = 1")
        ax.set_title(f"{pool}: члени за зростанням одиночного mAP@50")
        ax.grid(axis="y", alpha=0.3, zorder=0)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig4_member_strength_{dataset}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- tables
def _tex_table(path: Path, header: list[str], rows: list[list], caption: str) -> None:
    with open(path, "w") as f:
        f.write("\\begin{table}[ht]\\centering\\small\n\\caption{" + caption + "}\n")
        f.write("\\begin{tabular}{" + "l" * len(header) + "}\\hline\n")
        f.write(" & ".join(header) + " \\\\\\hline\n")
        for r in rows:
            f.write(" & ".join(str(x) for x in r) + " \\\\\n")
        f.write("\\hline\\end{tabular}\\end{table}\n")


def _csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _fmt(s: dict) -> list:
    verdict = "значуще" if (s["ci_lo"] > 0 or s["ci_hi"] < 0) else "н/з"
    return [f"{s['mean']:+.4f}", f"[{s['ci_lo']:+.4f}; {s['ci_hi']:+.4f}]", s["p_le_zero"], verdict]


def _loss_row(
    boot, clean_a, att_a, clean_b, att_b, label, pool, k, attack, which=0, metric="mAP@50"
):
    """(loss of B) − (loss of A), loss = clean − attacked; positive => A more robust."""
    key = (
        (lambda r: r["map50"]["balanced"])
        if which == 0
        else (lambda r: r["ap50_target"]["balanced"])
    )
    point = (_mean_over_sets(clean_b, key) - _mean_over_sets(att_b, key)) - (
        _mean_over_sets(clean_a, key) - _mean_over_sets(att_a, key)
    )
    reps = [_boot_mean_sets(boot, r, which) for r in (clean_a, att_a, clean_b, att_b)]
    if any(v is None for v in reps):
        return [pool, k, attack, metric, label, f"{point:+.4f}", "", "", "", ""]
    ca, aa, cb, ab = reps
    return [
        pool,
        k,
        attack,
        metric,
        label,
        f"{point:+.4f}",
        *_fmt(summarize((cb - ab) - (ca - aa))),
    ]


def _abs_row(boot, rows_a, rows_b, label, pool, k, attack, which=0, metric="mAP@50"):
    key = (
        (lambda r: r["map50"]["balanced"])
        if which == 0
        else (lambda r: r["ap50_target"]["balanced"])
    )
    point = _mean_over_sets(rows_a, key) - _mean_over_sets(rows_b, key)
    a, b = _boot_mean_sets(boot, rows_a, which), _boot_mean_sets(boot, rows_b, which)
    if a is None or b is None:
        return [pool, k, attack, metric, label, f"{point:+.4f}", "", "", "", ""]
    return [pool, k, attack, metric, label, f"{point:+.4f}", *_fmt(summarize(a - b))]


def _delta_row(boot: Boot, rows_a, rows_b, label, pool, k, attack):
    a = _boot_mean_sets(boot, rows_a)
    b = _boot_mean_sets(boot, rows_b)
    point = _mean_over_sets(rows_a) - _mean_over_sets(rows_b)
    if a is None or b is None:
        return [pool, k, attack, label, f"{point:+.4f}", "", "", "", ""]
    s = summarize(a - b)
    verdict = "значуще" if (s["ci_lo"] > 0 or s["ci_hi"] < 0) else "н/з"
    return [
        pool,
        k,
        attack,
        label,
        f"{point:+.4f}",
        f"{s['mean']:+.4f}",
        f"[{s['ci_lo']:+.4f}; {s['ci_hi']:+.4f}]",
        s["p_le_zero"],
        verdict,
    ]


def table_bootstrap(idx, pools, ks_of, boot_of, dataset, out):
    header = [
        "пул",
        "k",
        "атака",
        "метрика",
        "різниця",
        "Δ (повна вибірка)",
        "бутстреп-середнє",
        "95 % ДІ",
        "P(Δ≤0)",
        "висновок",
    ]
    rows = []
    attacks = [
        ("suppress", 1.0),
        ("suppress", 0.5),
        ("suppress", 0.1),
        ("inject", 15.0),
        ("inflate", 1.5),
    ]
    for pool in pools:
        boot = boot_of[pool]
        if boot is None:
            continue
        clean = {m: idx.get((pool, 0, "none", None, m), []) for m in METHOD_STYLE}
        for k in ks_of[pool]:
            hon = {m: idx.get((pool, k, "honest", None, m), []) for m in ("wbf", "nms")}
            for att, par in attacks:
                g = lambda m, a=att, p=par: idx.get((pool, k, a, p, m), [])  # noqa: E731
                lab = f"{att} {par:g}"
                for which, metric in ((0, "mAP@50"), (1, "AP@50 цілі")):
                    rows.append(
                        _loss_row(
                            boot,
                            clean["wbf"],
                            g("wbf"),
                            clean["nms"],
                            g("nms"),
                            "H1: втрата(NMS) − втрата(WBF)",
                            pool,
                            k,
                            lab,
                            which,
                            metric,
                        )
                    )
                    rows.append(
                        _loss_row(
                            boot,
                            clean["wbf"],
                            g("wbf"),
                            clean["nmw"],
                            g("nmw"),
                            "H1': втрата(NMW) − втрата(WBF)",
                            pool,
                            k,
                            lab,
                            which,
                            metric,
                        )
                    )
                    rows.append(
                        _loss_row(
                            boot,
                            clean["wbf"],
                            g("wbf"),
                            clean["wbf_no_conf"],
                            g("wbf_no_conf"),
                            "H2: втрата(WBF без підтв.) − втрата(WBF)",
                            pool,
                            k,
                            lab,
                            which,
                            metric,
                        )
                    )
                    rows.append(
                        _abs_row(
                            boot,
                            g("wbf"),
                            g("nms"),
                            "WBF − NMS (абс.)",
                            pool,
                            k,
                            lab,
                            which,
                            metric,
                        )
                    )
                    rows.append(
                        _abs_row(
                            boot,
                            g("wbf"),
                            hon["wbf"],
                            "H3: отруєний − чесне вилучення (WBF)",
                            pool,
                            k,
                            lab,
                            which,
                            metric,
                        )
                    )
                    rows.append(
                        _abs_row(
                            boot,
                            g("nms"),
                            hon["nms"],
                            "H3: отруєний − чесне вилучення (NMS)",
                            pool,
                            k,
                            lab,
                            which,
                            metric,
                        )
                    )
    # H4: larger pool at matched fraction k/N (nearest fractions)
    if all(p in pools for p in ("hetero3", "full7")) and all(
        boot_of[p] is not None for p in ("hetero3", "full7")
    ):
        for k3, k7 in ((1, 2), (2, 5)):
            for att, par in (("suppress", 1.0), ("inject", 15.0)):
                for m in ("wbf", "nms"):
                    a = idx.get(("full7", k7, att, par, m), [])
                    b = idx.get(("hetero3", k3, att, par, m), [])
                    # different pools have different clean baselines: compare relative losses
                    ra = _boot_mean_sets(boot_of["full7"], a)
                    rb = _boot_mean_sets(boot_of["hetero3"], b)
                    ca = _boot_mean_sets(
                        boot_of["full7"], idx.get(("full7", 0, "none", None, m), [])
                    )
                    cb = _boot_mean_sets(
                        boot_of["hetero3"], idx.get(("hetero3", 0, "none", None, m), [])
                    )
                    if any(v is None for v in (ra, rb, ca, cb)):
                        continue
                    loss_a, loss_b = (
                        ca - ra,
                        cb - rb,
                    )  # loss per pool (replicates paired within pool only)
                    s = summarize(loss_b - loss_a)
                    point = (
                        _mean_over_sets(idx.get(("hetero3", 0, "none", None, m), []))
                        - _mean_over_sets(b)
                    ) - (
                        _mean_over_sets(idx.get(("full7", 0, "none", None, m), []))
                        - _mean_over_sets(a)
                    )
                    verdict = "значуще" if (s["ci_lo"] > 0 or s["ci_hi"] < 0) else "н/з"
                    rows.append(
                        [
                            "hetero3 vs full7",
                            f"{k3}/3 vs {k7}/7",
                            f"{att} {par:g}",
                            "mAP@50",
                            f"H4: втрата(hetero3) − втрата(full7), {m}",
                            f"{point:+.4f}",
                            f"{s['mean']:+.4f}",
                            f"[{s['ci_lo']:+.4f}; {s['ci_hi']:+.4f}]",
                            s["p_le_zero"],
                            verdict,
                        ]
                    )
    _csv(out / f"table1_bootstrap_{dataset}.csv", header, rows)
    _tex_table(
        out / f"table1_bootstrap_{dataset}.tex",
        header,
        rows,
        f"Парний бутстреп ключових різниць mAP@50 ({dataset}); замість пропущених ДІ бутстреп не запускався",
    )
    return rows


def table_scenes(idx, pools, ks_of, dataset, out, scenes):
    header = ["пул", "метод", "k", "сцена", "чистий", "A1 ρ=1", "втрата", "чесне вилучення", "ASR"]
    rows = []
    for pool in pools:
        for m in ("wbf", "nms"):
            base = idx.get((pool, 0, "none", None, m), [])
            for k in ks_of[pool]:
                att = idx.get((pool, k, "suppress", 1.0, m), [])
                hon = idx.get((pool, k, "honest", None, m), [])
                for s in scenes:
                    c = _mean_over_sets(base, lambda r, s=s: r["map50"][s])
                    a = _mean_over_sets(att, lambda r, s=s: r["map50"][s])
                    h = _mean_over_sets(hon, lambda r, s=s: r["map50"][s])
                    asr = _mean_over_sets(att, lambda r, s=s: r["asr"][s] if r["asr"] else None)
                    rows.append(
                        [
                            pool,
                            m,
                            k,
                            SCENE_NAME.get(s, s),
                            f"{c:.4f}",
                            f"{a:.4f}",
                            f"{c - a:+.4f}",
                            f"{h:.4f}",
                            f"{asr:.3f}",
                        ]
                    )
    _csv(out / f"table2_scenes_{dataset}.csv", header, rows)
    _tex_table(
        out / f"table2_scenes_{dataset}.tex",
        header,
        rows,
        f"Стійкість за сценами, A1 ρ=1 ({dataset})",
    )
    return rows


def summary_json(idx, pools, ks_of, dataset, out, scenes):
    """Compact machine-readable summary for the thesis text / results-auditor."""
    summ = {"dataset": dataset, "pools": {}}
    for pool in pools:
        d = {
            "clean": {
                m: _mean_over_sets(idx.get((pool, 0, "none", None, m), [])) for m in METHOD_STYLE
            },
            "attacks": {},
        }
        for att, par in (
            ("suppress", 1.0),
            ("suppress", 0.5),
            ("suppress", 0.1),
            ("decoy", 1.0),
            ("inject", 5.0),
            ("inject", 15.0),
            ("inject", 30.0),
            ("inflate", 1.5),
            ("honest", None),
        ):
            key = f"{att}" + ("" if par is None else f"_{par:g}")
            d["attacks"][key] = {}
            for k in ks_of[pool]:
                per_m = {}
                for m in METHOD_STYLE:
                    rows = idx.get((pool, k, att, par, m), [])
                    if not rows:
                        continue
                    per_m[m] = {
                        "map50_balanced": round(_mean_over_sets(rows), 4),
                        "map50_weighted": round(
                            _mean_over_sets(rows, lambda r: r["map50"]["weighted"]), 4
                        ),
                        "target_recall_all": round(
                            _mean_over_sets(rows, lambda r: r["target_recall"]["all"]), 4
                        ),
                        "asr_all": round(
                            _mean_over_sets(rows, lambda r: r["asr"]["all"] if r["asr"] else None),
                            4,
                        )
                        if att != "honest"
                        else None,
                        "fppi_delta": round(
                            _mean_over_sets(
                                rows,
                                lambda r: r["fppi_delta"]["balanced"] if r["fppi_delta"] else None,
                            ),
                            3,
                        )
                        if att == "inject"
                        else None,
                        "n_sets": len(rows),
                    }
                d["attacks"][key][str(k)] = per_m
        summ["pools"][pool] = d
    with open(out / f"summary_{dataset}.json", "w") as f:
        json.dump(summ, f, indent=1, ensure_ascii=False)
    return summ


def build_report(out_root: Path, pools_cfg: dict, params: dict, log=print):
    pools = pools_cfg["pools"]
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    single = {}
    try:  # single-model mAP@50 for fig 4 ordering (VisDrone: hetero_pool_results; UAVDT: transfer results)
        h = json.load(open(out_root.parent / "ensembles" / "hetero_pool_results.json"))
        for m in pools["full7"]:
            single["visdrone", m] = float(
                np.mean([h["scenes"][s]["models"][m]["mAP50"] for s in h["scenes"]])
            )
        u = json.load(open(out_root.parent / "uavdt" / "uavdt_transfer_results.json"))
        for m in pools["full7"]:
            single["uavdt", m] = float(np.mean([u["scenes"][s]["models"][m] for s in u["scenes"]]))
    except Exception as e:  # noqa: BLE001
        log(f"[report] single-model mAP unavailable for fig4 ordering: {e}")
    for dataset in ("visdrone", "uavdt"):
        rows = []
        present = []
        for pool in pools:
            p = out_root / f"grid_{dataset}_{pool}.jsonl"
            if p.exists():
                rows += read_jsonl(p)
                present.append(pool)
        if not rows:
            log(f"[report] no grid for {dataset}, skipping")
            continue
        idx = _index(rows)
        scenes = [s for s in rows[0]["map50"] if s not in ("balanced", "weighted")]
        ks_of = {
            pool: sorted({r["k"] for r in rows if r["pool"] == pool and r["k"] > 0})
            for pool in present
        }
        boot_of = {}
        for pool in present:
            b = out_root / "boot" / f"{dataset}_{pool}"
            boot_of[pool] = Boot(b) if b.exists() else None
        sub_pools = {p: pools[p] for p in present}
        fig_degradation(idx, sub_pools, ks_of, boot_of, dataset, fig_dir)
        fig_asr(idx, sub_pools, ks_of, dataset, fig_dir)
        fig_inject(idx, sub_pools, ks_of, boot_of, dataset, fig_dir)
        fig_ablation(idx, sub_pools, ks_of, boot_of, dataset, fig_dir)
        fig_member_strength(
            idx, sub_pools, {m: v for (d, m), v in single.items() if d == dataset}, dataset, fig_dir
        )
        table_bootstrap(idx, sub_pools, ks_of, boot_of, dataset, out_root)
        table_scenes(idx, sub_pools, ks_of, dataset, out_root, scenes)
        summary_json(idx, sub_pools, ks_of, dataset, out_root, scenes)
        log(f"[report] {dataset}: pools={present}, figures -> {fig_dir}, tables -> {out_root}")
