# scac — Scene-Complexity Adaptive Cascade

A small, NumPy-only library for the two-tier detector cascade studied in the
dissertation "Optimisation of ensemble object recognition methods based on
neural networks for cybersecurity systems" (chapter 4): a cheap tier-1
detector runs on every frame, a complexity signal computed from its output
decides whether the frame is escalated to an expensive tier-2 ensemble, and
the threshold is calibrated by cross-fitting — never by hand.

The library is model-agnostic: detectors and fusion are plain callables, so it
works with Ultralytics, torchvision, ONNX Runtime or anything else.

```
pip install -e libs/scac          # from the project root
```

## What it provides

| Module | Purpose | Thesis |
|---|---|---|
| `scac.signals` | `c_count`, `c_conf`, `c_entropy`, `signal_values` — annotation-free complexity signals from tier-1 scores at the deployment threshold | §4.2 |
| `scac.policy` | `CascadePolicy` (escalate iff signal ≥ τ), `Cascade` (runs tier-1 / tier-2 callables and the fusion), `average_budget` (B(p) = B₁ + p·B₂) | §4.2, eq. 2.6 |
| `scac.calibrate` | `choose_threshold` (p* on a calibration set, τ = quantile), `stratified_folds` (2-fold by sequence, stratified by scene), `crossfit` (fold-wise τ, cross-applied) | §4.2, §2.4 |
| `scac.gate` | `gate_stats` (density of the per-frame ensemble gain), `oracle_curve`, `signal_curve`, `random_curve` | §4.1 |
| `scac.guard` | `EscalationGuard` — sliding-window cap on the realised escalation fraction (countermeasure to resource exhaustion) | §4.5 |
| `scac.cli` | `scac gate` / `scac calibrate` over a JSON frame table with a per-frame-AP surrogate | — |

## Usage

### 1. Gate first: is a cascade worth building?

```python
import numpy as np
from scac import gate_stats, oracle_curve, signal_curve

# per-frame AP of tier 1 and tier 2 on cached predictions (NaN = no GT)
delta = ap_tier2 - ap_tier1
print(gate_stats(delta))        # frac_positive 0.41 on UAVDT -> sparse gain -> build it
                                # frac_positive 0.75-0.90 on VisDrone -> dense -> don't

# quality of a mixture; for mAP this must re-score the whole set (non-additive)
def score(mask):
    return balanced_map50([t2[i] if m else t1[i] for i, m in enumerate(mask)])

oracle = oracle_curve(delta, score, b1=684.4, b2=1605.5)
count  = signal_curve([c_count(p["scores"]) for p in t1], score, b1=684.4, b2=1605.5)
```

### 2. Calibrate the threshold without leakage

```python
from scac import stratified_folds, crossfit, c_count

signals = [c_count(p["scores"]) for p in t1]           # one per frame
fold_of = stratified_folds(sequence_of, scene_of, np.random.default_rng(42))

def score(mask, eval_idx):                              # scored on the calibration fold only
    mixed = [t2[i] if m else t1[i] for i, m in enumerate(mask)]
    return balanced_map50(mixed, subset=eval_idx)

res = crossfit(signals, fold_of, score)
res["folds"]      # {0: {'p': 0.25, 'tau': 30.0, ...}, 1: {'p': 0.15, 'tau': 43.8, ...}}
res["realized"]   # 0.211 on UAVDT
```

### 3. Deploy

```python
from scac import Cascade, CascadePolicy, EscalationGuard

policy = CascadePolicy(tau=30.0, signal="C-count", deploy_conf=0.25)
guard  = EscalationGuard(p_max=0.4, window=200)        # cap the realised fraction

cascade = Cascade(
    tier1=lambda img: yolov10x(img),                    # -> {"boxes","scores","labels"}
    tier2=[lambda img: yolov9e(img), lambda img: yolo26x(img)],
    fuse=wbf,                                           # list[pred] -> pred
    policy=policy, guard=guard,
)
pred, escalated = cascade(frame)
cascade.realized_fraction, guard.budget_overrun(0.211, 684.4, 1605.5)
```

### CLI (cheap first look at a new domain)

```
scac gate frames.json --b1 684.4 --b2 1605.5
scac calibrate frames.json --b1 684.4 --b2 1605.5 --out calib.json
```

`frames.json` is a list of `{"id", "group", "scene", "signal", "ap_t1", "ap_t2"}`
records. The CLI scores mixtures with a scene-balanced mean of per-frame AP —
an additive surrogate of mAP; for the pool-level metric use the Python API.

## Reproducibility

The chapter-4 cross-fitted thresholds (p* = 0.25 / 0.15, τ = 30.0 / 43.8,
realised escalation 21.1 % on UAVDT) were reproduced through this library from
the same prediction caches as the research bench. The reference run is
`results/uavdt/scac_crossfit_bootstrap.json` and the recorded parity check is
`results/uavdt/scac_lib_parity.json` (`all_match: true`); the parity script
itself needs the prediction caches and is not part of this release.

Tests: `uv run pytest libs/scac/tests -q`.
