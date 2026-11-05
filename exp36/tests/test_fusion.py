"""Stage-2 unit tests: own WBF equals ensemble_boxes' WBF on synthetic data,
the confirmation multiplier does what the spec says, and every method is
class-separable (the identity the whole fusion cache relies on)."""

import numpy as np
import pytest
from ensemble_boxes import weighted_boxes_fusion

from exp36.fusion.methods import METHODS, fuse_all_classes, fuse_one_class, wbf_single_class


def synth(rng, n_models=3, n_obj=12, n_cls=3, w=1000, h=800, noise=6.0, dropout=0.3):
    """Members see the same objects with jittered boxes; some boxes dropped."""
    objs = []
    for _ in range(n_obj):
        bw, bh = rng.uniform(30, 120, size=2)
        x1, y1 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        objs.append((x1, y1, x1 + bw, y1 + bh, rng.integers(0, n_cls)))
    preds = []
    for _ in range(n_models):
        boxes, scores, labels = [], [], []
        for x1, y1, x2, y2, c in objs:
            if rng.random() < dropout:
                continue
            j = rng.normal(0, noise, size=4)
            boxes.append(
                [max(0, x1 + j[0]), max(0, y1 + j[1]), min(w, x2 + j[2]), min(h, y2 + j[3])]
            )
            scores.append(rng.uniform(0.05, 1.0))
            labels.append(c)
        preds.append(
            {
                "boxes": np.array(boxes).reshape(-1, 4),
                "scores": np.array(scores),
                "labels": np.array(labels, dtype=int),
            }
        )
    return preds, (w, h)


@pytest.mark.parametrize("seed", range(5))
def test_own_wbf_matches_ensemble_boxes(seed):
    rng = np.random.default_rng(seed)
    preds, wh = synth(rng)
    own = fuse_all_classes("wbf", preds, wh, 3, len(preds), 0.55, 0.01, 0.5)
    norm = np.array([wh[0], wh[1], wh[0], wh[1]], dtype=float)
    lb, ls, ll = weighted_boxes_fusion(
        [p["boxes"] / norm for p in preds],
        [p["scores"] for p in preds],
        [p["labels"] for p in preds],
        weights=None,
        iou_thr=0.55,
        skip_box_thr=0.01,
        conf_type="avg",
        allows_overflow=False,
    )
    assert len(own["scores"]) == len(ls)
    ko = np.lexsort((own["labels"], -own["scores"]))
    kl = np.lexsort((ll, -ls))
    np.testing.assert_allclose(own["scores"][ko], ls[kl], atol=1e-6)
    np.testing.assert_allclose(own["boxes"][ko], lb[kl] * norm, atol=1e-3)
    assert (own["labels"][ko] == ll[kl].astype(int)).all()


def test_confirmation_multiplier():
    # one box seen by 1 of 3 models: conf * 1/3 with confirmation, unchanged without
    b = [np.array([[0.1, 0.1, 0.3, 0.3]])]
    s = [np.array([0.9])]
    fb, fs = wbf_single_class(b, s, n_models=3, confirmation=True)
    assert np.isclose(fs[0], 0.9 / 3)
    fb2, fs2 = wbf_single_class(b, s, n_models=3, confirmation=False)
    assert np.isclose(fs2[0], 0.9)
    # seen by all 3: multiplier == 1, both agree
    b3 = [np.array([[0.1, 0.1, 0.3, 0.3]])] * 3
    s3 = [np.array([0.9]), np.array([0.6]), np.array([0.3])]
    _, f3 = wbf_single_class(b3, s3, n_models=3, confirmation=True)
    _, g3 = wbf_single_class(b3, s3, n_models=3, confirmation=False)
    assert np.isclose(f3[0], g3[0]) and np.isclose(f3[0], 0.6)


def test_absent_member_penalised_not_ignored():
    """N is the pool size: an absent (suppressed) member still counts in the
    denominator — that is the mechanism under test."""
    b = [np.array([[0.1, 0.1, 0.3, 0.3]]), np.array([[0.1, 0.1, 0.3, 0.3]])]
    s = [np.array([0.8]), np.array([0.8])]
    _, f_pool3 = wbf_single_class(b, s, n_models=3)
    _, f_pool2 = wbf_single_class(b, s, n_models=2)
    assert np.isclose(f_pool3[0], 0.8 * 2 / 3) and np.isclose(f_pool2[0], 0.8)


@pytest.mark.parametrize("method", METHODS)
def test_per_class_decomposition(method):
    """Fusing all classes at once == fusing class by class and concatenating."""
    rng = np.random.default_rng(7)
    preds, wh = synth(rng, n_models=4, n_obj=20, n_cls=4)
    whole = fuse_all_classes(method, preds, wh, 4, 4, 0.55, 0.01, 0.5)
    parts_b, parts_s, parts_l = [], [], []
    for c in range(4):
        fb, fs = fuse_one_class(
            method,
            [p["boxes"][p["labels"] == c] for p in preds],
            [p["scores"][p["labels"] == c] for p in preds],
            c,
            wh,
            4,
            0.55,
            0.01,
            0.5,
        )
        parts_b.append(fb)
        parts_s.append(fs)
        parts_l.append(np.full(len(fs), c))
    pb, ps, pl = np.concatenate(parts_b), np.concatenate(parts_s), np.concatenate(parts_l)
    assert len(ps) == len(whole["scores"])
    ka = np.lexsort((whole["boxes"][:, 0], whole["labels"], -whole["scores"]))
    kb = np.lexsort((pb[:, 0], pl, -ps))
    np.testing.assert_allclose(whole["scores"][ka], ps[kb], atol=1e-6)
    np.testing.assert_allclose(whole["boxes"][ka], pb[kb], atol=1e-4)


@pytest.mark.parametrize("method", METHODS)
def test_empty_members(method):
    fb, fs = fuse_one_class(
        method,
        [np.zeros((0, 4)), np.zeros((0, 4))],
        [np.zeros(0), np.zeros(0)],
        0,
        (100, 100),
        3,
        0.55,
        0.01,
        0.5,
    )
    assert len(fs) == 0 and fb.shape == (0, 4)
    fb, fs = fuse_one_class(
        method,
        [np.zeros((0, 4)), np.array([[10, 10, 50, 50.0]])],
        [np.zeros(0), np.array([0.7])],
        0,
        (100, 100),
        3,
        0.55,
        0.01,
        0.5,
    )
    assert len(fs) == 1


def test_capped_multiplier_equals_wbf_with_reduced_pool():
    """wbf_capK is WBF with N replaced by N - K (and never below 1)."""
    import numpy as np

    from exp36.fusion.methods import cap_kmax, fuse_one_class, is_wbf_family

    assert cap_kmax("wbf_cap2") == 2 and cap_kmax("wbf") is None and cap_kmax("nms") is None
    assert is_wbf_family("wbf_cap3") and not is_wbf_family("nmw")
    rng = np.random.default_rng(0)
    wh = (640, 480)
    boxes = [np.array([[100, 100, 200, 200.0]]) + rng.normal(0, 2, (1, 4)) for _ in range(3)]
    scores = [np.array([0.8]), np.array([0.7]), np.array([0.9])]
    # three members of a pool of N = 7: plain WBF confidence = mean * 3/7
    _, s_wbf = fuse_one_class("wbf", boxes, scores, 0, wh, 7, 0.55, 0.01, 0.5)
    assert s_wbf[0] == pytest.approx(0.8 * 3 / 7)
    # capped with k_max = 4: N_eff = 3 -> full confidence
    _, s_cap = fuse_one_class("wbf_cap4", boxes, scores, 0, wh, 7, 0.55, 0.01, 0.5)
    assert s_cap[0] == pytest.approx(0.8)
    # k_max = 2: N_eff = 5 -> 3/5
    _, s_cap2 = fuse_one_class("wbf_cap2", boxes, scores, 0, wh, 7, 0.55, 0.01, 0.5)
    assert s_cap2[0] == pytest.approx(0.8 * 3 / 5)
    # identical to plain WBF on a pool of size N - k_max
    _, s_ref = fuse_one_class("wbf", boxes, scores, 0, wh, 5, 0.55, 0.01, 0.5)
    assert s_cap2[0] == pytest.approx(s_ref[0])
    # a lone phantom seen by one member: 1/(N - k_max) instead of 1/N
    _, s_ph = fuse_one_class("wbf_cap2", [boxes[0]], [np.array([0.9])], 0, wh, 7, 0.55, 0.01, 0.5)
    assert s_ph[0] == pytest.approx(0.9 / 5)
