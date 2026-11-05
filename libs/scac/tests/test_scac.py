import numpy as np
import pytest
from scac import (
    Cascade,
    CascadePolicy,
    EscalationGuard,
    average_budget,
    c_conf,
    c_count,
    c_entropy,
    choose_threshold,
    crossfit,
    gate_stats,
    oracle_curve,
    signal_curve,
    signal_values,
    stratified_folds,
)
from scac.calibrate import p_grid, tau_for_fraction
from scac.gate import random_curve


# ---------------------------------------------------------------- signals
def test_signals_empty_frame():
    assert c_count([]) == 0.0
    assert c_conf([]) == 1.0
    assert c_entropy([]) == 1.0
    assert signal_values(np.array([0.1, 0.2])) == {"C-count": 0.0, "C-conf": 1.0, "C-entropy": 1.0}


def test_signals_values():
    s = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    assert c_count(s) == 3.0
    assert c_conf(s) == pytest.approx(1 - (0.9 + 0.8 + 0.3) / 3)
    p = np.array([0.9, 0.8, 0.3])
    h = np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    assert c_entropy(s) == pytest.approx(h)
    assert c_conf(np.full(20, 0.5), topk=10) == pytest.approx(0.5)


# ---------------------------------------------------------------- policy
def test_policy_and_budget():
    pol = CascadePolicy(tau=3.0)
    assert pol.escalate(np.array([0.9, 0.9, 0.9]))
    assert not pol.escalate(np.array([0.9, 0.9]))
    assert CascadePolicy(tau=float("inf")).escalate(np.ones(100)) is False
    assert list(pol.escalate_many([1, 3, 5])) == [False, True, True]
    assert average_budget(0.211, 684.4, 1605.5) == pytest.approx(1023.16)
    with pytest.raises(ValueError):
        CascadePolicy(tau=1.0, signal="nope")


def test_cascade_runner_reuses_tier1_and_counts():
    calls = {"t1": 0, "t2": 0, "fuse": 0}

    def t1(img):
        calls["t1"] += 1
        return {"scores": np.full(img, 0.9)}

    def t2(img):
        calls["t2"] += 1
        return {"scores": np.full(2, 0.5)}

    def fuse(preds):
        calls["fuse"] += 1
        assert preds[0]["scores"].size >= 3  # tier-1 prediction is passed through
        return {"scores": np.concatenate([p["scores"] for p in preds])}

    cas = Cascade(t1, [t2, t2], fuse, CascadePolicy(tau=3))
    out, esc = cas(1)
    assert not esc and out["scores"].size == 1
    out, esc = cas(5)
    assert esc and out["scores"].size == 9
    assert calls == {"t1": 2, "t2": 2, "fuse": 1}
    assert cas.realized_fraction == 0.5


# ---------------------------------------------------------------- calibrate
def test_tau_for_fraction_and_grid():
    sig = np.arange(100, dtype=float)
    assert tau_for_fraction(sig, 0.0) == float("inf")
    assert (sig >= tau_for_fraction(sig, 0.25)).mean() == pytest.approx(0.25, abs=0.02)
    assert p_grid(0.25) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_choose_threshold_prefers_first_best_on_ties():
    sig = np.linspace(0.0, 1.0, 1001)
    # quality is flat -> p* = 0 (cheapest)
    res = choose_threshold(sig, lambda mask, idx: 1.0)
    assert res["p"] == 0.0 and res["tau"] == float("inf")
    # quality = fraction escalated among the calibration frames, but only up to 0.5
    res = choose_threshold(sig, lambda mask, idx: min(mask[idx].mean(), 0.5))
    assert res["p"] == pytest.approx(0.5)


def test_stratified_folds_keep_groups_whole_and_balance_scenes():
    rng = np.random.default_rng(0)
    group = [f"g{i // 10}" for i in range(200)]  # 20 groups of 10 frames
    scene = ["day" if i < 120 else "night" for i in range(200)]
    fold = stratified_folds(group, scene, rng)
    for g in set(group):
        assert len({fold[i] for i in range(200) if group[i] == g}) == 1
    for s in ("day", "night"):
        f = fold[[i for i in range(200) if scene[i] == s]]
        assert abs((f == 0).mean() - 0.5) <= 0.1


def test_crossfit_applies_other_folds_threshold():
    rng = np.random.default_rng(1)
    n = 400
    sig = rng.random(n)
    fold = np.array([0, 1] * (n // 2))
    # each frame's "gain" is known only to the oracle: gain iff signal > 0.7
    gain = (sig > 0.7).astype(float)

    def score(mask, idx):
        # reward escalating gaining frames, penalise escalating the rest
        return float((mask[idx] * (2 * gain[idx] - 1)).mean())

    res = crossfit(sig, fold, score)
    assert set(res["folds"]) == {0, 1}
    for f in (0, 1):
        assert res["folds"][f]["p"] == pytest.approx(0.3, abs=0.05)
    # frames in fold 0 must be decided with fold 1's tau and vice versa
    for i in range(n):
        tau = res["folds"][1 - fold[i]]["tau"]
        assert res["escalate"][i] == (sig[i] >= tau)
    with pytest.raises(ValueError):
        crossfit(sig, np.zeros(n, int), score)


# ---------------------------------------------------------------- gate
def test_gate_stats_and_curves_monotone_endpoints():
    delta = np.array([0.1, -0.05, 0.0, np.nan, 0.2])
    st = gate_stats(delta)
    assert st["n"] == 4 and st["frac_positive"] == 0.5 and st["frac_negative"] == 0.25

    rng = np.random.default_rng(2)
    n = 300
    d = rng.normal(0, 1, n)

    def score(mask):  # additive surrogate: mean realised gain
        return float((mask * d).mean())

    oc = oracle_curve(d, score, grid=[0, 0.5, 1.0])
    assert oc[0]["score"] == 0.0 and oc[-1]["score"] == pytest.approx(d.mean())
    assert oc[1]["score"] >= max(r["score"] for r in signal_curve(rng.random(n), score, grid=[0.5]))
    rc = random_curve(n, score, rng, grid=[0.5], draws=5)
    assert oc[1]["score"] >= rc[0]["score"]
    oc_b = oracle_curve(d, score, grid=[0.0, 1.0], b1=684.4, b2=1605.5)
    assert oc_b[0]["budget"] == pytest.approx(684.4) and oc_b[1]["budget"] == pytest.approx(2289.9)


# ---------------------------------------------------------------- guard
def test_guard_caps_realized_fraction():
    g = EscalationGuard(p_max=0.2, window=10)
    admitted = [g.admit(True) for _ in range(50)]  # adversary wants every frame escalated
    assert sum(admitted[-10:]) <= 2
    assert g.realized_fraction <= 0.2 + 1e-9
    assert g.vetoed > 0
    assert g.budget_overrun(0.2, 684.4, 1605.5) <= 1.0 + 1e-9
    g.reset()
    assert g.realized_fraction == 0.0 and g.vetoed == 0
    # never interferes below the cap
    g2 = EscalationGuard(p_max=0.5, window=10)
    assert all(g2.admit(i % 4 == 0) == (i % 4 == 0) for i in range(40))
    with pytest.raises(ValueError):
        EscalationGuard(p_max=0.0)


def test_cascade_with_guard_degrades_to_tier1():
    def t1(img):  # always "complex"
        return {"scores": np.full(50, 0.9)}

    def t2(img):
        return {"scores": np.ones(1)}

    cas = Cascade(
        t1, [t2], lambda ps: ps[0], CascadePolicy(tau=10), guard=EscalationGuard(0.25, 20)
    )
    for _ in range(100):
        cas(None)
    assert cas.realized_fraction <= 0.3
