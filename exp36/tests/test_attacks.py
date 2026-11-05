"""Stage-3 invariants: A1 never adds boxes, A2 never removes, A3 keeps the box
set and coordinates; trigger/phantom sampling is deterministic."""

import numpy as np

from exp36.attacks.primitives import (
    attack_inflate,
    attack_inject,
    attack_suppress,
    sample_phantoms,
    triggered_frames,
)


def pred():
    rng = np.random.default_rng(0)
    return {
        "boxes": rng.uniform(0, 500, size=(30, 4)),
        "scores": rng.uniform(0, 1, 30),
        "labels": rng.integers(0, 8, 30),
    }


def test_suppress_removes_only_target_class():
    p = pred()
    q = attack_suppress(p, 3, True)
    assert (q["labels"] != 3).all()
    assert len(q["scores"]) == int((p["labels"] != 3).sum())
    assert attack_suppress(p, 3, False) is p


def test_inject_only_adds():
    p = pred()
    prior = np.array([[20.0, 40.0], [30.0, 60.0]])
    ph = sample_phantoms(5, prior, (640, 480), (0.6, 0.9), 43, "visdrone", "day", "m", 7)
    q = attack_inject(p, 0, ph)
    assert len(q["scores"]) == len(p["scores"]) + 5
    np.testing.assert_array_equal(q["boxes"][:30], p["boxes"])
    assert (q["labels"][30:] == 0).all()
    assert ((q["scores"][30:] >= 0.6) & (q["scores"][30:] <= 0.9)).all()
    b = q["boxes"][30:]
    assert (b[:, 0] >= 0).all() and (b[:, 2] <= 640).all() and (b[:, 3] <= 480).all()


def test_inflate_keeps_boxes():
    p = pred()
    q = attack_inflate(p, 1.5)
    np.testing.assert_array_equal(q["boxes"], p["boxes"])
    assert len(q["scores"]) == len(p["scores"]) and q["scores"].max() <= 1.0
    assert (q["scores"] >= p["scores"]).all()


def test_determinism():
    prior = np.array([[20.0, 40.0]])
    a = sample_phantoms(4, prior, (640, 480), (0.6, 0.9), 43, "visdrone", "day", "m", 7)
    b = sample_phantoms(4, prior, (640, 480), (0.6, 0.9), 43, "visdrone", "day", "m", 7)
    c = sample_phantoms(4, prior, (640, 480), (0.6, 0.9), 43, "visdrone", "day", "m", 8)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])
    m1 = triggered_frames(200, 0.1, 43, "visdrone", "day")
    m2 = triggered_frames(200, 0.1, 43, "visdrone", "day")
    assert m1.sum() == 20 and (m1 == m2).all()
    assert triggered_frames(200, 1.0, 43, "visdrone", "day").all()
